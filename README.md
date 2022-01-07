# genetic_expert_guided_learning
This is an official implementation of our paper Guiding Deep Molecular Optimization with Genetic Exploration (https://arxiv.org/pdf/2007.04897.pdf). Our code is largely inspired by GuacaMol baselines (https://github.com/BenevolentAI/guacamol_baselines).

## 1. Setting up the environment
You can set up the environment by following commands. dmo is shortcut for deep-molecular-optimization
```
conda create -n dmo python=3.6
conda activate dmo
conda install pytorch torchvision cudatoolkit=10.1 -c pytorch
conda install -c dglteam dgl-cuda10.1
conda install -c rdkit rdkit
pip install neptune-client
pip install tqdm
pip install psutil
pip install guacamol
pip install wandb
```

## 2. Run command: 
CUDA_VISIBLE_DEVICES=$GPU_DEVICE python run_gegl.py --benchmark_id $BENCHMARK_ID --dataset guacamol --apprentice_load_dir ./resource/checkpoint/guacamol

### GOAL: run above command with $BENCHMARK_ID=8-19 inclusive 
#### (ids 8-19 correspond to guacamol benchmarks we want to compare to)
#### Each run will take ~3 hours

## 3. Enter wandb API Key Below to track w/ wandb
dfa956c5bfb9fa492ebf9adede99093772919518

### Results will be tracked and recorded here:
https://wandb.ai/nmaus/gegl?workspace=user-nmaus


