<div align="center">
<h1>Q3R: Quadratic Reweighted Rank Regularizer for Effective Low-Rank Training </h1>

<a href="https://thate10.github.io/">Ethan Nguyen</a><sup>1*</sup>,
<a href="mailto:ipsita.ghosh@ucf.edu">Ipsita Ghosh</a><sup>2*</sup>,
<a href="mailto:kuemmerle@ucf.edu">Christian Kümmerle</a><sup>3</sup>,

<sup>1</sup>Department of Computer Science University of North Carolina at Charlotte,<br>
<sup>2</sup>Department of Computer Science University of Central Florida,<br>
<sup>3</sup>School of Data, Mathematical and Statistical Sciences Department of Computer Science University of Central Florida<br>
<sup>*</sup>Equal Contribution<br>

<p align="center" style="margin:20px;">
<a href="https://arxiv.org/pdf/2511.04485", target="_blank">
<img src="https://img.shields.io/badge/arXiv-2511.04485-b31b0b.svg?logo=arxiv&logoColor=white"></a>
</p>
</div>

## Overview

Q3R is a novel regularization technique for training low-rank neural networks. It promotes low-rank structures in weight matrices during training and can be applied to standard linear layers as well as fused layers (like QKV projections in Transformers).

**Key Features:**
- Rank regularization for weight matrices
- Fused layer support (e.g., Q, K, V slices)
- Distributed training (DDP) support
- Integrated `AdamQ3R` optimizer

## Quick Start

Two ways to use Q3R:

**Option 1: AdamQ3R (Recommended)**
```python
from Functions.AdamQ3R import AdamQ3R
from main_helper import extract_linear

trainable_modules = extract_linear(model, config)
optimizer = AdamQ3R(model.parameters(), 
                    trainable_modules=trainable_modules, 
                    target_rank=0.2, 
                    lmbda=0.1, 
                    steps=5)
```

**Option 2: Q3R Regularizer**
```python
from Functions.Q3R import Q3R

trainable_modules = extract_linear(model, config)
q3r = Q3R(trainable_modules=trainable_modules, 
          target_rank=0.2, 
          lmbda=0.1, 
          steps=5)

# In training loop
q3r.update()
total_loss = loss + q3r.val
total_loss.backward()
```

## Setup and Installation

Ensure you have CUDA installed. This project was tested with CUDA 12.6.

1. **Install PyTorch**:
   ```bash
   pip install torch==2.6.0+cu126 torchvision==0.21.0+cu126 --extra-index-url https://download.pytorch.org/whl/cu126
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Verify Installation**:
   ```bash
   python -c "import torch; print(torch.cuda.is_available())"
   ```

## Experiment Execution

**Basic AdamQ3R Training:**
```bash
python main.py --dataset CIFAR10 --model VIT_Tiny --learning_rate 0.0004 --epoch 100 --technique AdamQ3R --lmbda 0.1 --target_rank 0.05 --target_modules qkv
```

**LoRITa + Q3R:**
```bash
python main.py --dataset CIFAR10 --model VIT_Tiny --learning_rate 0.00004 --epoch 100 --technique LoRITaQuaRS --depth_lorita=1 --weight_decay_alpha=0.1 --target_modules qkv --target_rank 16 --epsilon_schedule linear --N 46875
```

## Hyperparameters

| Parameter           | Type              | Default / Example               | Description                                                              |
| ------------------- | ----------------- | ------------------------------- | ------------------------------------------------------------------------ |
| `lr`                | float             | 0.00004                         | Base learning rate for the optimizer.                                    |
| `trainable_modules` | dict              | `extract_linear(model, config)` | Linear modules that will receive Q3R updates.                            |
| `target_rank`       | float (0–1)       | 0.2                             | Fraction of singular values to retain for low-rank projection.           |
| `lmbda`             | float             | 0.1                             | Scaling factor for the Q3R regularization term.                          |
| `steps`             | int               | 5                               | Update period for SVD calculations (higher = faster, less frequent).     |

## Advanced Usage

### Fused Modules (QKV Layers)

Q3R supports fused modules where multiple linear projections are concatenated into a single weight matrix. Provide slice indices to regularize each component independently:

```python
# Fused QKV layer with output dimension 768 (256 for Q, 256 for K, 256 for V)
qkv_slices = [(0, 256), (256, 512), (512, 768)]

trainable_modules = {
    model.attention.qkv: qkv_slices,
    model.fc1: None  # None means use the full weight matrix
}

optimizer = AdamQ3R(
    model.parameters(),
    trainable_modules=trainable_modules,
    target_rank=0.1,
    lmbda=0.1
)
```

The gradients for each slice are computed independently and "stuffed" back into the full gradient tensor using `pad_tensor_with_slice_bounds`, ensuring correct regularization without physically splitting weights.

### Distributed Training

Q3R automatically supports PyTorch DDP. Regularizers are distributed across ranks for efficient computation:

```python
# Standard DDP setup
model = torch.nn.parallel.DistributedDataParallel(model, ...)

# Q3R will automatically distribute work across GPUs
optimizer = AdamQ3R(model.parameters(), trainable_modules=trainable_modules, ...)
```

## Citation

If you use Q3R in your research, please cite:

```bibtex
@article{nguyen2025q3r,
  title={Q3R: Quadratic Reweighted Rank Regularizer for Effective Low-Rank Training},
  author={Nguyen, Ethan and Ghosh, Ipsita and K{\"u}mmerle, Christian},
  journal={arXiv preprint arXiv:2511.04485},
  year={2025}
}
```
