what i did:
- changed installation so no sudo is required



# installation (adapted from original repo robopianist)
bash install_deps.sh
cd robopianist
git submodule init && git submodule update
conda create -n pianist python=3.10
conda activate pianist
conda install -y -c conda-forge pyaudio
pip install -e ".[dev]"

for test: make test
