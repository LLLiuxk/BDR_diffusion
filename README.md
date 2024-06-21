
## Installation
Following is the suggested way to install the dependencies of our code:
```
conda create -n bdr_diffusion
conda activate bdr_diffusion

conda install pytorch==1.9.0 torchvision==0.10.0 cudatoolkit=11.3 -c pytorch -c conda-forge


pip install tqdm fire einops trimesh ocnn scikit-image==0.18.2 scikit-learn==0.24.2 pytorch-lightning==1.6.1
```

## Usage
Please refer to the scripts in `scripts/` for the usage of our code.
### Train from Scratch
```
bash scripts/train.sh
```

### Conditioned generation
```
bash scripts/generate.sh
```



