# Trio-ViT

This repo contains the official implementation of **["Trio-ViT: Post-Training Quantization and Acceleration for Softmax-Free Efficient Vision Transformer"](https://arxiv.org/abs/2405.03882).**

## Abstract

ViTs' huge model sizes and intensive computations hinder their deployment on embedded devices, calling for effective model compression methods, such as quantization. 
Unfortunately, due to the existence of hardware-unfriendly and quantization-sensitive non-linear operations, particularly Softmax, it is non-trivial to completely quantize all operations in ViTs, yielding either significant accuracy drops or non-negligible hardware costs. 
In response to challenges associated with standard ViTs, we focus our attention towards the quantization and acceleration for **efficient ViTs**, which not only eliminate the troublesome Softmax but also integrate linear attention with low computational complexity, and propose Trio-ViT accordingly. 

Specifically, at the algorithm level, we develop a **tailored post-training quantization engine** taking the unique activation distributions of Softmax-free efficient ViTs into full consideration, aiming to boost quantization accuracy. 
Furthermore, at the hardware level, we build **an accelerator** dedicated to the specific Convolution-Transformer hybrid architecture of efficient ViTs, thereby enhancing hardware efficiency.

## Quantization

### Run

Example: Quantize EfficientViT-b1-r224 with 8bit.
```bash
python main_imagenet.py --data_path PATH_TO_IMAGENET  --n_bits_w 8 --channel_wise --weight 0.5 --model b1-r224 --disable_8bit_head_stem  --n_bits_a 8  --act_quant --input_size 224 --test_before_calibration
```
- `--n_bits_w `: the quantization bit-width of weights
- `-channel_wise`: whether to use channel-wise quantization for quantizing weights
- `--model`: the chosed model to quantize
- `--n_bits_a`: the quantization bit-width of activation
- `--act_quant`: whether to quantize activation
- `--input_size`: input resolution for dataset

