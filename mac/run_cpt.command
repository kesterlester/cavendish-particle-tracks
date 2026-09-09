#!/bin/zsh

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate napari-env

export CPT_SHUFFLING_SEED=1

python "/Users/lycanerf/Downloads/Particle Tracks/cavendish-particle-tracks/launch_debug.py"
