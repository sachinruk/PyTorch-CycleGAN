# Pytorch-CycleGAN
A clean and readable Pytorch implementation of CycleGAN (https://arxiv.org/abs/1703.10593). This is an adaptation from https://github.com/aitorzip/PyTorch-CycleGAN such that it is dockerised and is **more compatible with PyTorch 0.4, and can run with batch size of 10** (as opposed to one).

All of the training was run in an AWS p2.xlarge instance. One epoch took ~40 minutes. Do not attempt to train without a GPU. 

To run this do:
```
nvidia-docker run -d -p 8888:8888 -p 6006:6006 -e PASSWORD=BadPasswordChange -v ${PWD}:/notebook -v ${PWD}/data/:/notebook/data sachinruk/pytorch_gpu
```
**please please please change the password above**.

and then in a terminal run:
```
python train.py --dataroot ./datasets/monet2photo/ --batchSize 10 --n_cpu 4 --cuda
```

## Disclaimer
Software comes as is. I/we are not liable for costs you may incur by running this.