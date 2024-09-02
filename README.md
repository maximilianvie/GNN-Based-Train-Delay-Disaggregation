# Simulation-Based Disaggregation of Train Delay Data Using Graph Neural Networks

## Preprocessing CSV Files and Graph Creation


The preprocessing_and_creation_of_graph folder contains scripts preparing the output datafiles of the agent-based model:

	•	CSV Data Preprocessing: This script handles the initial preprocessing of raw CSV data files, ensuring they are preprocessed for subsequent processing.
	•	Data to Graph Creation: This script transforms the preprocessed data into graph structures suitable for the GNN training. 

After generating the graph data, the resulting .pkl file should be copied into the datasets/raw directory to ensure it is accessible for training and inference.




## Training the GNN
### Python environment setup with Conda

```bash
conda create -n delaydis python=3.10
conda activate delaydis

conda install pytorch=1.13 torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia
conda install pyg=2.2 -c pyg -c conda-forge
pip install pyg-lib -f https://data.pyg.org/whl/torch-1.13.0+cu117.html


pip install pytorch-lightning yacs torchmetrics
pip install performer-pytorch
pip install tensorboardX
pip install ogb
pip install wandb

conda clean --all
```


### Training the model
```bash
conda activate delaydis

# Run classification training

python main.py --cfg configs/max/custom_simple_gatedgcn_classification.yaml

# Run regression training

python main.py --cfg configs/max/custom_simple_gatedgcn_regression.yaml

```


### Inference 
You need a saved pretrained model from the previous step, then run it with an "inference" script.

```bash

# Run inference regression
python main.py --cfg configs/custom_simple_gatedgcn_regression_inference.yaml

# Run inference classification
python main.py --cfg configs/custom_simple_gatedgcn_classification_inference.yaml

```


## Acknowledgments

This project builds upon the codebases of [GraphGPS](https://github.com/rampasek/GraphGPS) and [GraphGym](https://github.com/snap-stanford/GraphGym). We are grateful to the authors of these projects for their essential contributions to the field of graph neural networks.

If you use this code, please consider citing the original papers for GraphGPS and GraphGym:

```bibtex
@article{rampasek2022GPS,
  title={{Recipe for a General, Powerful, Scalable Graph Transformer}}, 
  author={Ladislav Ramp\'{a}\v{s}ek et al.},
  journal={Advances in Neural Information Processing Systems},
  volume={35},
  year={2022}
}

@article{you2020graphgym,
  title={{Design Space for Graph Neural Networks}},
  author={Jiaxuan You et al.},
  journal={Advances in Neural Information Processing Systems},
  volume={33},
  year={2020}
}


