## Q3R

Launch Q3R regularisation

```bash
python main.py --dataset CIFAR10 --model VIT_Tiny --learning_rate 0.0004 --epoch 100 --technique AdamQ3R --lmbda 0.1 --target_rank 0.05 --target_modules qkv
```

Lorita Q3R script
```bash
python main.py --dataset CIFAR10 --model VIT_Tiny --learning_rate 0.00004 --epoch 100 --technique LoRITaQuaRS --depth_lorita=1 --weight_decay_alpha=0.1 --target_modules qkv --target_rank 16  --target_modules qkv --epsilon_schedule linear --N 46875

```