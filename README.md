# RoboPianist Evaluation Repository

This repository provides an end-to-end evaluation pipeline for **RoboPianist**—the dexterous piano-playing agent introduced in the paper [RoboPianist: Dexterous Piano Playing with Deep Reinforcement Learning](https://arxiv.org/abs/2304.04150) by Kevin Zakka *et al.*. It was developed in the context of the **IN2107 – Seminar on Robotics Science and Systems Intelligence** at the Technical University of Munich (TUM).

---

## Overview

RoboPianist is a reinforcement-learning–based system for generating dexterous piano-playing motions. This repo adapts the original implementation (in JAX) to PyTorch and provides an easy-to-use evaluation framework covering:

1. **End-to-end quantitative evaluation** on the full **ROBOPIANIST-REPERTOIRE-150** dataset. [Evaluation.ipynb](Evaluation.ipynb)

2. **Generalization test** on a pop music snippet (“Golden Hour” by JVKE, bars 7 & 9) outside the classical training domain. [Evaluation_golden_hour.ipynb](Evaluation_golden_hour.ipynb)

---

## Features

- **PyTorch Implementation**: Complete policy model rewritten in PyTorch for compatibility and ease of integration with modern research.  
- **Single-Notebook Execution**: Each evaluation pipeline is encapsulated in a Jupyter notebook:
  - `Evaluation.ipynb` for the full repertoire.  
  - `Evaluation_golden_hour.ipynb` for the pop-music snippet.  
- **Automated Fingering Integration**: Fingering annotations for the “Golden Hour” snippet provided as a single text file (`golden_hour_fingering.txt`).  
- **No External MIDI Dependencies**: In the Golden Hour pipeline, all file handling is contained within the notebook, requiring only the provided fingering file.

--- 

## Installation
1. **Clone the Repository**:
   ```bash
   git clone https://github.com/leongreiner/robopianist-evaluation.git
   cd robopianist-evaluation
   ```
2. **Initialize Submodules**:
   ```bash
   git submodule init && git submodule update
   ```
3. **Install Dependencies**:
   Follow the instructions in the Evaluation notebooks to install all dependencies.