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
pip install gdown

for test: make test

Optional: Download additional soundfonts
robopianist soundfont --download #(on active pianist conda env and choose SalamanderGrandPiano (option 1))

# dataset download
download here: https://drive.google.com/file/d/1Y8gSUWDXT0fuyKuwkzlXwtdSp_ZcOEaw/view?usp=sharing (do not distribut, since of licensincs restrictions you useally have to create an account, just for the purpose of this seminar)

cd .. # (in main repo)
mkdir -p datasets
gdown --id 1Y8gSUWDXT0fuyKuwkzlXwtdSp_ZcOEaw -O datasets/PianoFingeringDataset_v1.2.zip
unzip datasets/PianoFingeringDataset_v1.2.zip -d datasets/ && \
rm datasets/PianoFingeringDataset_v1.2.zip

robopianist preprocess --dataset-dir datasets/PianoFingeringDataset_v1.2
This will create a directory called pig_single_finger in robopianist_evaluation/robopianist/music/data.

to test: robopianist --check-pig-exists
