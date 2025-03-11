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
```Bash
python run_gegl.py --benchmark_id $BENCHMARK_ID --dataset guacamol --apprentice_load_dir ./resource/checkpoint/guacamol --wandb_entity nmaus-upenn
```

$BENCHMARK_ID=1,2,..., 20 correspond to the 20 guacamol tasks in order that they show results for in table 2. 


Running takes ~3 hours

Results for Zaleplon already included in results_for_Zaleplon MPO.txt 

