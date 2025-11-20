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

## 
Launch Q3R regularisation

```bash
python main.py --dataset CIFAR10 --model VIT_Tiny --learning_rate 0.0004 --epoch 100 --technique AdamQ3R --lmbda 0.1 --target_rank 0.05 --target_modules qkv
```

Lorita Q3R script
```bash
python main.py --dataset CIFAR10 --model VIT_Tiny --learning_rate 0.00004 --epoch 100 --technique LoRITaQuaRS --depth_lorita=1 --weight_decay_alpha=0.1 --target_modules qkv --target_rank 16  --target_modules qkv --epsilon_schedule linear --N 46875

```
