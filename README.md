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

##Quick Start Guide
There are two methodologies to employ Q3R during training via direct gradient application in AdamQ3R() or via autograd regularization.

Example code snippet, hyperparameter recommendations and usage are below
```
model = Net()
trainable_modules = extract_linear(model, config)
optimizer = AdamQ3R(model.parameters(),lr=0.00004,
                    trainable_modules=trainable_modules, target_rank=0.2,
                    lmbda=0.1, period=5
                    )
```

## Research Replication Procedures:

### Dataset Preparation

### Experiment Execution
```bash
python main.py --dataset CIFAR10 --model VIT_Tiny --learning_rate 0.0004 --epoch 100 --technique AdamQ3R --lmbda 0.1 --target_rank 0.05 --target_modules qkv
```

Lorita Q3R script
```bash
python main.py --dataset CIFAR10 --model VIT_Tiny --learning_rate 0.00004 --epoch 100 --technique LoRITaQuaRS --depth_lorita=1 --weight_decay_alpha=0.1 --target_modules qkv --target_rank 16  --target_modules qkv --epsilon_schedule linear --N 46875
```

## Q3R Implementation Details

### Hyperparameter Q3R explanation
| Hyperparameter      | Type              | Default / Example               | Description                                                              |
| ------------------- | ----------------- | ------------------------------- | ------------------------------------------------------------------------ |
| `lr`                | float             | 0.00004                         | Base learning rate for the optimizer.                                    |
| `trainable_modules` | list of nn.Module | `extract_linear(model, config)` | Linear modules that will receive Q3R updates.                            |
| `target_rank`       | float (0–1)       | 0.2                             | Fraction of singular values to retain for low-rank projection.           |
| `lmbda`             | float             | 0.1                             | Scaling factor for the Q3R regularization term relative to Adam updates. |
| `period`             | int               | 5                              | The update period, increasing the period provides faster preformance by minimizing the amount of SVD updates calculated                   |


### Fused modules
- Gradient Stacking into modules
- 

### GPU Parallelization and Q3R module independence
- Faster increasing calculation of it
- Memory overhead

### Unittesting
- 
### Next Steps
- Torch.optim integration with Accerlate
