One Pass for All: A Discrete Diffusion Model for Knowledge Graph Triple Set Prediction
================================================================


This is the code for DiffTSP. The denoising model architecture in `graph_dit/models`


## Requirements
All dependencies are specified in the `requirements.txt` file.

This code was developed and tested with Python 3.9, PyTorch 2.0.0, and PyG 2.3.0, Pytorch-lightning 2.0.1.


## Usage

We could train the model on an 4090 GPU card. Here is an example to running the code:



```
CUDA_VISIBLE_DEVICES=0 python main.py --config-name=config.yaml  general.gpus=1 model.diffusion_steps=20
```
All default configurations can be found in `configs/config.yaml`. 

## Citation

If you find this repository useful, please cite our paper