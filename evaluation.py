import os
import sys
import subprocess
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal
import random
import time
import wandb
from pathlib import Path
from typing import Optional, Tuple
from dataclasses import dataclass, asdict
from tqdm import tqdm
import tyro
from IPython.display import HTML, clear_output
from base64 import b64encode
import matplotlib.pyplot as plt

base_dir = os.getcwd()  
robopianist_dir = os.path.join(base_dir, "robopianist")
os.chdir(robopianist_dir)
if robopianist_dir not in sys.path:
    sys.path.insert(0, robopianist_dir)

# RoboPianist imports
from robopianist import suite, music
import dm_env_wrappers as wrappers
import robopianist.wrappers as robopianist_wrappers
import dm_env

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA device: {torch.cuda.get_device_name()}")

@dataclass(frozen=True)
class TrainingArgs:
    hpc_mode: bool = True  # Set to False for local runs with video recording
    
    # Environment settings
    environment_name: str = "RoboPianist-debug-TwinkleTwinkleRousseau-v0"
    seed: int = 42
    control_timestep: float = 0.05
    n_steps_lookahead: int = 10
    trim_silence: bool = True
    gravity_compensation: bool = True
    reduced_action_space: bool = True
    primitive_fingertip_collisions: bool = True
    action_reward_observation: bool = True
    
    # Training hyperparameters
    max_steps: int = 5000000
    warmstart_steps: int = 5000
    batch_size: int = 256
    replay_capacity: int = 5000000
    
    # SAC hyperparameters
    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    temp_lr: float = 3e-4
    hidden_dims: Tuple[int, ...] = (256, 256, 256)
    discount: float = 0.99
    tau: float = 0.005
    init_temperature: float = 1.0
    
    # Logging and evaluation
    log_interval: int = 1000
    eval_interval: int = 10000
    eval_episodes: int = 1  # Back to 1 to avoid identical episodes (can set to 3+ if needed)
    tqdm_bar: bool = True
    
    # Paths and wandb
    base_dir = os.getcwd()
    parent_dir = os.path.dirname(base_dir)
    root_dir: str = str(os.path.join(parent_dir, "runs"))
    project: str = "robopianist-pytorch"
    mode: str = "disabled"  # Set to "online" to enable wandb logging
    
    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

# Create training configuration
args = TrainingArgs()

def play_video(filename: str):
    if not os.path.exists(filename):
        print(f"Video file not found: {filename}")
        return None
    
    mp4 = open(filename, "rb").read()
    data_url = "data:video/mp4;base64," + b64encode(mp4).decode()
    
    return HTML(f"""
    <video controls width="640" height="480">
        <source src="{data_url}" type="video/mp4">
    </video>
    """)

def prefix_dict(prefix: str, d: dict) -> dict:
    return {f"{prefix}/{k}": v for k, v in d.items()}

class MLP(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden_dims: Tuple[int, ...]):
        super().__init__()
        dims = (input_dim,) + hidden_dims + (output_dim,)
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            layers.append(nn.GELU())
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)

class Policy(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int, hidden_dims: Tuple[int, ...]):
        super().__init__()
        self.backbone = MLP(obs_dim, 2 * action_dim, hidden_dims)
        self.action_dim = action_dim
        
    def forward(self, obs):
        outputs = self.backbone(obs)
        means, log_stds = torch.chunk(outputs, 2, dim=-1)
        log_stds = torch.clamp(log_stds, -20, 2)
        return means, log_stds
    
    def sample(self, obs, deterministic=False):
        means, log_stds = self.forward(obs)
        stds = torch.exp(log_stds)
        
        if deterministic:
            actions = torch.tanh(means)
            log_probs = None
        else:
            dist = Normal(means, stds)
            samples = dist.rsample()
            actions = torch.tanh(samples)
            
            log_probs = dist.log_prob(samples)
            log_probs -= torch.log(1 - actions.pow(2) + 1e-6)
            log_probs = log_probs.sum(dim=-1, keepdim=True)
            
        return actions, log_probs

class QNetwork(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int, hidden_dims: Tuple[int, ...]):
        super().__init__()
        self.network = MLP(obs_dim + action_dim, 1, hidden_dims)
    
    def forward(self, obs, action):
        x = torch.cat([obs, action], dim=-1)
        return self.network(x)

class TwinQNetwork(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int, hidden_dims: Tuple[int, ...]):
        super().__init__()
        self.q1 = QNetwork(obs_dim, action_dim, hidden_dims)
        self.q2 = QNetwork(obs_dim, action_dim, hidden_dims)
    
    def forward(self, obs, action):
        return self.q1(obs, action), self.q2(obs, action)
    
class ReplayBuffer:
    def __init__(self, obs_dim: int, action_dim: int, max_size: int, device: str):
        self.max_size = max_size
        self.device = device
        self.ptr = 0
        self.size = 0
        
        self.obs = torch.zeros((max_size, obs_dim), dtype=torch.float32)
        self.actions = torch.zeros((max_size, action_dim), dtype=torch.float32)
        self.rewards = torch.zeros((max_size, 1), dtype=torch.float32)
        self.next_obs = torch.zeros((max_size, obs_dim), dtype=torch.float32)
        self.dones = torch.zeros((max_size, 1), dtype=torch.float32)
        
        self._prev_timestep = None
        
    def add(self, timestep, action=None):
        if action is not None and self._prev_timestep is not None:
            obs = torch.from_numpy(self._prev_timestep.observation.astype(np.float32))
            next_obs = torch.from_numpy(timestep.observation.astype(np.float32))
            action_tensor = torch.from_numpy(action.astype(np.float32))
            reward = torch.tensor(float(timestep.reward), dtype=torch.float32).unsqueeze(0)
            done = torch.tensor(float(1.0 - timestep.discount), dtype=torch.float32).unsqueeze(0)
            
            self.obs[self.ptr] = obs
            self.actions[self.ptr] = action_tensor
            self.rewards[self.ptr] = reward
            self.next_obs[self.ptr] = next_obs
            self.dones[self.ptr] = done
            
            self.ptr = (self.ptr + 1) % self.max_size
            self.size = min(self.size + 1, self.max_size)
            
        self._prev_timestep = timestep
    
    def sample(self, batch_size: int):
        indices = torch.randint(0, self.size, (batch_size,))
        return (
            self.obs[indices].to(self.device),
            self.actions[indices].to(self.device),
            self.rewards[indices].to(self.device),
            self.next_obs[indices].to(self.device),
            self.dones[indices].to(self.device)
        )
    
    def __len__(self):
        return self.size
    
class SAC:
    def __init__(self, obs_dim: int, action_dim: int, args: TrainingArgs):
        self.device = torch.device(args.device)
        self.discount = args.discount
        self.tau = args.tau
        self.target_entropy = -action_dim
        
        # Networks
        self.actor = Policy(obs_dim, action_dim, args.hidden_dims).to(self.device)
        self.critic = TwinQNetwork(obs_dim, action_dim, args.hidden_dims).to(self.device)
        self.target_critic = TwinQNetwork(obs_dim, action_dim, args.hidden_dims).to(self.device)
        self.actor = self.actor.float()
        self.critic = self.critic.float()
        self.target_critic = self.target_critic.float()
        
        self.target_critic.load_state_dict(self.critic.state_dict())
        
        self.log_alpha = torch.tensor(np.log(args.init_temperature), dtype=torch.float32, requires_grad=True, device=self.device)
        
        # Optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=args.actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=args.critic_lr)
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=args.temp_lr)
        
    @property
    def alpha(self):
        return self.log_alpha.exp()
    
    def select_action(self, obs, deterministic=False):
        with torch.no_grad():
            obs_tensor = torch.from_numpy(obs.astype(np.float32)).unsqueeze(0).to(self.device)
            action, _ = self.actor.sample(obs_tensor, deterministic=deterministic)
            return action.cpu().numpy()[0].astype(np.float32)
    
    def update(self, replay_buffer: ReplayBuffer, batch_size: int):
        obs, actions, rewards, next_obs, dones = replay_buffer.sample(batch_size)
        
        obs = obs.float()
        actions = actions.float()
        rewards = rewards.float()
        next_obs = next_obs.float()
        dones = dones.float()
        
        # Update critic
        with torch.no_grad():
            next_actions, next_log_probs = self.actor.sample(next_obs)
            target_q1, target_q2 = self.target_critic(next_obs, next_actions)
            target_q = torch.min(target_q1, target_q2) - self.alpha * next_log_probs
            target_q = rewards + (1 - dones) * torch.tensor(self.discount, dtype=torch.float32, device=self.device) * target_q
        
        current_q1, current_q2 = self.critic(obs, actions)
        critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        # Update actor
        new_actions, log_probs = self.actor.sample(obs)
        q1, q2 = self.critic(obs, new_actions)
        q = torch.min(q1, q2)
        actor_loss = (self.alpha * log_probs - q).mean()
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        # Update temperature
        target_entropy_tensor = torch.tensor(self.target_entropy, dtype=torch.float32, device=self.device)
        alpha_loss = -(self.log_alpha * (log_probs + target_entropy_tensor).detach()).mean()
        
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()
        
        # Update target network
        self.soft_update(self.critic, self.target_critic)
        
        return {
            "critic_loss": critic_loss.item(),
            "actor_loss": actor_loss.item(),
            "alpha_loss": alpha_loss.item(),
            "alpha": self.alpha.item(),
        }
    
    def soft_update(self, source, target):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - self.tau) + param.data * self.tau)
    
    def save(self, filepath):
        torch.save({
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict(),
            'target_critic': self.target_critic.state_dict(),
            'log_alpha': self.log_alpha,
        }, filepath)
    
    def load(self, filepath):
        checkpoint = torch.load(filepath, map_location=self.device)
        self.actor.load_state_dict(checkpoint['actor'])
        self.critic.load_state_dict(checkpoint['critic'])
        self.target_critic.load_state_dict(checkpoint['target_critic'])
        self.log_alpha = checkpoint['log_alpha']

def get_env(args: TrainingArgs, record_dir: Optional[Path] = None):
    env = suite.load(
        environment_name=args.environment_name,
        seed=args.seed,
        task_kwargs=dict(
            n_steps_lookahead=args.n_steps_lookahead,
            trim_silence=args.trim_silence,
            gravity_compensation=args.gravity_compensation,
            reduced_action_space=args.reduced_action_space,
            control_timestep=args.control_timestep,
            primitive_fingertip_collisions=args.primitive_fingertip_collisions,
            change_color_on_activation=True,
        ),
    )
    
    if record_dir is not None and not args.hpc_mode:
        env = robopianist_wrappers.PianoSoundVideoWrapper(
            environment=env,
            record_dir=record_dir,
            record_every=1,
            camera_id="piano/back",
            height=480,
            width=640,
        )
    
    env = wrappers.EpisodeStatisticsWrapper(environment=env, deque_size=1)
    env = robopianist_wrappers.MidiEvaluationWrapper(environment=env, deque_size=1)
    
    if args.action_reward_observation:
        env = wrappers.ObservationActionRewardWrapper(env)
    
    env = wrappers.ConcatObservationWrapper(env)
    env = wrappers.CanonicalSpecWrapper(env, clip=True)
    env = wrappers.SinglePrecisionWrapper(env)
    env = wrappers.DmControlWrapper(env)
    
    return env

# Test environment creation
print("Creating test environment...")
test_env = get_env(args)
test_timestep = test_env.reset()

obs_dim = test_timestep.observation.shape[0]
action_dim = test_env.action_spec().shape[0]

print(f"Observation dimension: {obs_dim}")
print(f"Action dimension: {action_dim}")

def train(args: TrainingArgs):
    """Train the SAC agent."""
    
    # Set random seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    
    print(f"Using device: {args.device}")
    print(f"HPC Mode: {'ENABLED' if args.hpc_mode else 'DISABLED'}")
    if args.hpc_mode:
        print("  - Video recording disabled")
    
    # Create experiment directory
    run_name = f"PyTorch-SAC-{args.environment_name}-{args.seed}-{int(time.time())}"
    experiment_dir = Path(args.root_dir) / run_name
    experiment_dir.mkdir(parents=True, exist_ok=True)
    print(f"Experiment directory: {experiment_dir}")
    
    # Initialize wandb
    if args.mode != "disabled":
        wandb.init(
            project=args.project,
            config=asdict(args),
            mode=args.mode,
            name=run_name,
        )
    
    # Create environments
    env = get_env(args)
    eval_env = get_env(args, record_dir=experiment_dir / "eval")
    
    # Initialize SAC agent
    agent = SAC(obs_dim, action_dim, args)
    
    # Initialize replay buffer
    replay_buffer = ReplayBuffer(obs_dim, action_dim, args.replay_capacity, args.device)
    
    # Training metrics
    training_metrics = {
        'steps': [],
        'critic_losses': [],
        'actor_losses': [],
        'alpha_losses': [],
        'alphas': [],
        'eval_returns': [],
        'eval_f1_scores': [], 
        'eval_precision_scores': [], 
        'eval_recall_scores': [], 
        'eval_steps': []
    }
    
    # Add initial timestep
    timestep = env.reset()
    replay_buffer.add(timestep)
    
    start_time = time.time()
    eval_count = 0 
    
    # Training loop
    print(f"Starting training loop for {args.max_steps:,} steps...")
    progress_bar = tqdm(range(1, args.max_steps + 1), desc="Training", unit="steps")
    
    for step in progress_bar:
        # Select action
        if step < args.warmstart_steps:
            action = env.action_spec().generate_value()
        else:
            action = agent.select_action(timestep.observation)
        
        # Step environment
        timestep = env.step(action)
        replay_buffer.add(timestep, action)
        
        # Reset if episode ended
        if timestep.last():
            if args.mode != "disabled":
                wandb.log(prefix_dict("train", env.get_statistics()), step=step)
            timestep = env.reset()
            replay_buffer.add(timestep)
        
        # Update agent
        if step >= args.warmstart_steps and len(replay_buffer) >= args.batch_size:
            metrics = agent.update(replay_buffer, args.batch_size)
            
            # Store and update progress bar
            if step % args.log_interval == 0:
                training_metrics['steps'].append(step)
                training_metrics['critic_losses'].append(metrics['critic_loss'])
                training_metrics['actor_losses'].append(metrics['actor_loss'])
                training_metrics['alpha_losses'].append(metrics['alpha_loss'])
                training_metrics['alphas'].append(metrics['alpha'])
                
                # Update progress bar description with latest metrics
                progress_bar.set_postfix({
                    'Critic': f"{metrics['critic_loss']:.3f}",
                    'Actor': f"{metrics['actor_loss']:.3f}",
                    'Alpha': f"{metrics['alpha']:.3f}",
                    'Buffer': f"{len(replay_buffer)}"
                })
                
                if args.mode != "disabled":
                    wandb.log(prefix_dict("train", metrics), step=step)
        
        if step % args.eval_interval == 0:
            eval_count += 1 
            
            eval_returns = []
            eval_f1s = []
            eval_precisions = []
            eval_recalls = []
            
            actual_eval_episodes = args.eval_episodes
            
            for episode in range(actual_eval_episodes):
                episode_eval_env = get_env(args, record_dir=experiment_dir / "eval")
                eval_timestep = episode_eval_env.reset()
                eval_return = 0.0
                
                use_deterministic = (episode == 0) if actual_eval_episodes > 1 else True
                
                while not eval_timestep.last():
                    eval_action = agent.select_action(eval_timestep.observation, deterministic=use_deterministic)
                    eval_timestep = episode_eval_env.step(eval_action)
                    eval_return += eval_timestep.reward
                
                eval_returns.append(eval_return)
                
                try:
                    musical_metrics = episode_eval_env.get_musical_metrics()
                    eval_f1s.append(musical_metrics['f1'])
                    eval_precisions.append(musical_metrics['precision'])
                    eval_recalls.append(musical_metrics['recall'])
                except (AttributeError, ValueError):
                    eval_f1s.append(0.0)
                    eval_precisions.append(0.0)
                    eval_recalls.append(0.0)
            
            mean_return = np.mean(eval_returns)
            mean_f1 = np.mean(eval_f1s)
            mean_precision = np.mean(eval_precisions)
            mean_recall = np.mean(eval_recalls)
            
            # Calculate standard deviations
            std_return = np.std(eval_returns) if len(eval_returns) > 1 else 0.0
            std_f1 = np.std(eval_f1s) if len(eval_f1s) > 1 else 0.0
            std_precision = np.std(eval_precisions) if len(eval_precisions) > 1 else 0.0
            std_recall = np.std(eval_recalls) if len(eval_recalls) > 1 else 0.0
            
            # Store metrics
            training_metrics['eval_returns'].append(mean_return)
            training_metrics['eval_f1_scores'].append(mean_f1)
            training_metrics['eval_precision_scores'].append(mean_precision)
            training_metrics['eval_recall_scores'].append(mean_recall)
            training_metrics['eval_steps'].append(step)
            
            if args.mode != "disabled":
                eval_metrics = {
                    "eval/return_mean": mean_return,
                    "eval/return_std": std_return,
                    "eval/f1_mean": mean_f1,
                    "eval/f1_std": std_f1,
                    "eval/precision_mean": mean_precision,
                    "eval/precision_std": std_precision,
                    "eval/recall_mean": mean_recall,
                    "eval/recall_std": std_recall,
                }
                eval_metrics.update(prefix_dict("eval", eval_env.get_statistics()))
                wandb.log(eval_metrics, step=step)
            
            eval_info = f"Eval #{eval_count}: R={mean_return:.1f}, F1={mean_f1:.4f}"
            if std_return > 0:
                eval_info += f"±{std_return:.1f}"
            if std_f1 > 0:
                eval_info += f"±{std_f1:.4f}"
                
            progress_bar.set_postfix({
                'Last_Eval': eval_info,
                'Buffer': f"{len(replay_buffer)}"
            })
            
            # Save model at every evaluation step in HPC mode
            if args.hpc_mode:
                model_path = experiment_dir / f"model_step_{step}_eval_{eval_count}.pt"
                agent.save(model_path)
                print(f"Model saved at step {step}: {model_path}")
    
    progress_bar.close()
    
    elapsed_time = time.time() - start_time
    print(f"\nTraining completed in {elapsed_time:.2f} seconds")
    print(f"Total evaluations performed: {eval_count}")
    
    # Save final model
    model_path = experiment_dir / "final_model.pt"
    agent.save(model_path)
    print(f"Final model saved to: {model_path}")
    
    if args.hpc_mode:
        print(f"HPC Mode: {eval_count} evaluation models saved in {experiment_dir}")
        print("To generate videos locally later, set hpc_mode=False and reload saved models")
    
    if args.mode != "disabled":
        wandb.finish()
    
    return agent, training_metrics, experiment_dir

print(f"Training for {args.max_steps:,} steps with {args.warmstart_steps:,} warmstart steps")
print(f"Batch size: {args.batch_size}, Replay capacity: {args.replay_capacity:,}")
print(f"Evaluation interval: {args.eval_interval:,} steps")
print(f"Evaluation episodes per interval: {args.eval_episodes}")

# Run enhanced training with clean output
agent, metrics, exp_dir = train(args)

print("Training completed!")
print(f"Experiment directory: {exp_dir}")
if args.hpc_mode:
    print("HPC Mode was enabled - videos were not recorded to avoid rendering issues")
    print("Models were saved at every evaluation step for later video generation")
print(f"\nSummary:")
print(f"  Final Training F1: {metrics['eval_f1_scores'][-1]:.4f}")
print(f"  Final Training Return: {metrics['eval_returns'][-1]:.2f}")
print(f"  Best Training F1: {max(metrics['eval_f1_scores']):.4f}")
print(f"  Best Training Return: {max(metrics['eval_returns']):.2f}")
print(f"  F1 evaluations: {len(metrics['eval_f1_scores'])} data points")

# Create results directory
results_dir = Path("results")
results_dir.mkdir(exist_ok=True)

# Create enhanced visualization with F1 scores
fig, axes = plt.subplots(2, 3, figsize=(18, 10))

# Critic Loss
if 'steps' in metrics and 'critic_losses' in metrics:
    axes[0, 0].plot(metrics['steps'], metrics['critic_losses'])
    axes[0, 0].set_title('Critic Loss')
    axes[0, 0].set_xlabel('Steps')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].grid(True)
else:
    axes[0, 0].text(0.5, 0.5, 'No critic loss data', ha='center', va='center')
    axes[0, 0].set_title('Critic Loss')

# Actor Loss
if 'steps' in metrics and 'actor_losses' in metrics:
    axes[0, 1].plot(metrics['steps'], metrics['actor_losses'])
    axes[0, 1].set_title('Actor Loss')
    axes[0, 1].set_xlabel('Steps')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].grid(True)
else:
    axes[0, 1].text(0.5, 0.5, 'No actor loss data', ha='center', va='center')
    axes[0, 1].set_title('Actor Loss')

# Alpha (Temperature)
if 'steps' in metrics and 'alphas' in metrics:
    axes[0, 2].plot(metrics['steps'], metrics['alphas'])
    axes[0, 2].set_title('Temperature (Alpha)')
    axes[0, 2].set_xlabel('Steps')
    axes[0, 2].set_ylabel('Alpha')
    axes[0, 2].grid(True)
else:
    axes[0, 2].text(0.5, 0.5, 'No alpha data', ha='center', va='center')
    axes[0, 2].set_title('Temperature (Alpha)')

# Evaluation Returns
if 'eval_returns' in metrics and metrics['eval_returns']:
    axes[1, 0].plot(metrics['eval_steps'], metrics['eval_returns'], 'o-', color='blue', label='Returns')
    axes[1, 0].set_title('Evaluation Returns')
    axes[1, 0].set_xlabel('Steps')
    axes[1, 0].set_ylabel('Return')
    axes[1, 0].grid(True)
    axes[1, 0].legend()
else:
    axes[1, 0].text(0.5, 0.5, 'No evaluation data', ha='center', va='center')
    axes[1, 0].set_title('Evaluation Returns')

# F1 Scores During Training
if 'eval_f1_scores' in metrics and metrics['eval_f1_scores']:
    axes[1, 1].plot(metrics['eval_steps'], metrics['eval_f1_scores'], 'o-', color='green', label='F1 Score')
    axes[1, 1].set_title('Training F1 Scores')
    axes[1, 1].set_xlabel('Steps')
    axes[1, 1].set_ylabel('F1 Score')
    axes[1, 1].grid(True)
    axes[1, 1].legend()
    
    # Print F1 progression
    print("F1 Score Progression During Training:")
    for step, f1 in zip(metrics['eval_steps'], metrics['eval_f1_scores']):
        print(f"  Step {step}: F1 = {f1:.4f}")
else:
    axes[1, 1].text(0.5, 0.5, 'No F1 score data\n(Run training first)', ha='center', va='center')
    axes[1, 1].set_title('Training F1 Scores')

# Musical Metrics Combined
if ('eval_precision_scores' in metrics and metrics['eval_precision_scores'] and 
    'eval_recall_scores' in metrics and metrics['eval_recall_scores']):
    
    axes[1, 2].plot(metrics['eval_steps'], metrics['eval_precision_scores'], 'o-', color='purple', label='Precision')
    axes[1, 2].plot(metrics['eval_steps'], metrics['eval_recall_scores'], 'o-', color='orange', label='Recall')
    axes[1, 2].set_title('Precision & Recall')
    axes[1, 2].set_xlabel('Steps')
    axes[1, 2].set_ylabel('Score')
    axes[1, 2].grid(True)
    axes[1, 2].legend()
    
    # Print musical metrics progression
    print("\nMusical Metrics Progression During Training:")
    for step, p, r in zip(metrics['eval_steps'], metrics['eval_precision_scores'], metrics['eval_recall_scores']):
        print(f"  Step {step}: Precision = {p:.4f}, Recall = {r:.4f}")
else:
    axes[1, 2].text(0.5, 0.5, 'No musical metrics\n(Run training first)', ha='center', va='center')
    axes[1, 2].set_title('Precision & Recall')

plt.tight_layout()

# Save the plot with timestamp
timestamp = int(time.time())
plot_filename = results_dir / f"training_metrics_{timestamp}.png"
plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
print(f"\nPlot saved to: {plot_filename}")

# Also save as a generic filename (latest)
latest_plot_filename = results_dir / "training_metrics_latest.png"
plt.savefig(latest_plot_filename, dpi=300, bbox_inches='tight')
print(f"Plot also saved as: {latest_plot_filename}")

plt.show()

# Summary of training results
if 'eval_f1_scores' in metrics and metrics['eval_f1_scores']:
    print(f"\nTraining Summary:")
    print(f"Final F1 Score: {metrics['eval_f1_scores'][-1]:.4f}")
    print(f"Best F1 Score: {max(metrics['eval_f1_scores']):.4f}")
    print(f"F1 Score Improvement: {metrics['eval_f1_scores'][-1] - metrics['eval_f1_scores'][0]:+.4f}")
    
    if 'eval_returns' in metrics and metrics['eval_returns']:
        print(f"Final Return: {metrics['eval_returns'][-1]:.2f}")
        print(f"Best Return: {max(metrics['eval_returns']):.2f}")
        print(f"Return Improvement: {metrics['eval_returns'][-1] - metrics['eval_returns'][0]:+.2f}")