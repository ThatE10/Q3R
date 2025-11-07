#!/bin/bash
#SBATCH --job-name=Imagenet_pretrain
#SBATCH --output=outputs/iclr_experiments_%j.out
#SBATCH --error=outputs/iclr_experiments_%j.err
#SBATCH --nodes=1
#SBATCH --gres=gpu:L40S:1
#SBATCH --mem=100G
#SBATCH --cpus-per-task=16
#SBATCH --time=48:00:00
#SBATCH --partition=Nebula_GPU
#SBATCH --array=0-4%1 # launches 4 job indices from 0 to 3, max 1 concurrent jobs


module load cuda/12.1
module load anaconda3/2023.09
conda activate quars_env3

commands=(
#" python main.py --dataset CIFAR10 --model VIT_Tiny --learning_rate 0.00004 --epoch 100 --technique LoRA --target_rank 4 --target_modules qkv"
#" python main.py --dataset CIFAR10 --model VIT_Tiny --learning_rate 0.00004 --epoch 100 --technique LoRA --target_rank 9 --target_modules qkv"
#" python main.py --dataset CIFAR10 --model VIT_Tiny --learning_rate 0.00004 --epoch 100 --technique LoRA --target_rank 14 --target_modules qkv"
#" python main.py --dataset CIFAR10 --model VIT_Tiny --learning_rate 0.00004 --epoch 100 --technique LoRA --target_rank 19 --target_modules qkv"
#" python main.py --dataset CIFAR10 --model VIT_Tiny --learning_rate 0.00004 --epoch 100 --technique LoRA --target_rank .2 --target_modules qkv"
#" python main.py --dataset CIFAR10 --model VIT_Tiny --learning_rate 0.00004 --epoch 100 --technique LoRA --target_rank 38 --target_modules qkv"
#" python main.py --dataset CIFAR10 --model VIT_Tiny --learning_rate 0.00004 --epoch 100 --technique LoRA --target_rank 48 --target_modules qkv"
#" python main.py --dataset CIFAR10 --model VIT_Tiny --learning_rate 0.00004 --epoch 100 --technique LoRA --target_rank 57 --target_modules qkv"
#" python main.py --dataset CIFAR10 --model VIT_Tiny --learning_rate 0.00004 --epoch 100 --technique LoRA --target_rank 67 --target_modules qkv"
#" python main.py --dataset CIFAR10 --model VIT_Tiny --learning_rate 0.00004 --epoch 100 --technique LoRA --target_rank 76 --target_modules qkv"
#" python main.py --dataset CIFAR10 --model VIT_Tiny --learning_rate 0.00004 --epoch 100 --technique LoRA --target_rank 86 --target_modules qkv"
#" python main.py --dataset CIFAR10 --model VIT_Tiny --learning_rate 0.00004 --epoch 100 --technique LoRA --target_rank 96 --target_modules qkv"

#"python main2.py --dataset CIFAR10 --model VIT_Tiny --learning_rate 0.00004 --epoch 100 --technique LoRITa --depth_lorita=3 --weight_decay_alpha=0.01 --target_modules qkv"
#"python main2.py --dataset CIFAR10 --model VIT_Tiny --learning_rate 0.00004 --epoch 100 --technique LoRITa --depth_lorita=3 --weight_decay_alpha=0.1 --target_modules qkv"
#"python main2.py --dataset CIFAR10 --model VIT_Tiny --learning_rate 0.00004 --epoch 100 --technique LoRITa --depth_lorita=2 --weight_decay_alpha=0.001 --target_modules qkv"
#"python main2.py --dataset CIFAR10 --model VIT_Tiny --learning_rate 0.00004 --epoch 100 --technique LoRITa --depth_lorita=2 --weight_decay_alpha=0.01 --target_modules qkv"
#"python main2.py --dataset CIFAR10 --model VIT_Tiny --learning_rate 0.00004 --epoch 100 --technique LoRITa --depth_lorita=2 --weight_decay_alpha=0.1 --target_modules qkv"
#"python main2.py --dataset CIFAR10 --model VIT_Tiny --learning_rate 0.00004 --epoch 100 --technique LoRITa --depth_lorita=1 --weight_decay_alpha=0.001 --target_modules qkv"
#"python main2.py --dataset CIFAR10 --model VIT_Tiny --learning_rate 0.00004 --epoch 100 --technique LoRITa --depth_lorita=1 --weight_decay_alpha=0.01 --target_modules qkv"
#"python main2.py --dataset CIFAR10 --model VIT_Tiny --learning_rate 0.00004 --epoch 100 --technique LoRITa --depth_lorita=1 --weight_decay_alpha=0.1 --target_modules qkv"
#"python main2.py --dataset CIFAR10 --model VIT_Base --learning_rate 0.00004 --epoch 100 --technique LoRITa --depth_lorita=3 --weight_decay_alpha=0.001 --target_modules qkv"
#"python main2.py --dataset CIFAR10 --model VIT_Base --learning_rate 0.00004 --epoch 100 --technique LoRITa --depth_lorita=3 --weight_decay_alpha=0.01 --target_modules qkv"
#"python main2.py --dataset CIFAR10 --model VIT_Base --learning_rate 0.00004 --epoch 100 --technique LoRITa --depth_lorita=3 --weight_decay_alpha=0.1 --target_modules qkv"
#"python main2.py --dataset CIFAR10 --model VIT_Base --learning_rate 0.00004 --epoch 100 --technique LoRITa --depth_lorita=2 --weight_decay_alpha=0.001 --target_modules qkv"
#"python main2.py --dataset CIFAR10 --model VIT_Base --learning_rate 0.00004 --epoch 100 --technique LoRITa --depth_lorita=2 --weight_decay_alpha=0.01 --target_modules qkv"
#"python main2.py --dataset CIFAR10 --model VIT_Base --learning_rate 0.00004 --epoch 100 --technique LoRITa --depth_lorita=2 --weight_decay_alpha=0.1 --target_modules qkv"
#"python main2.py --dataset CIFAR10 --model VIT_Base --learning_rate 0.00004 --epoch 100 --technique LoRITa --depth_lorita=1 --weight_decay_alpha=0.001 --target_modules qkv"
#"python main2.py --dataset CIFAR10 --model VIT_Base --learning_rate 0.00004 --epoch 100 --technique LoRITa --depth_lorita=1 --weight_decay_alpha=0.01 --target_modules qkv"
#"python main2.py --dataset CIFAR10 --model VIT_Base --learning_rate 0.00004 --epoch 100 --technique LoRITa --depth_lorita=1 --weight_decay_alpha=0.1 --target_modules qkv"

#"python main.py --dataset CIFAR10 --model VIT_Tiny --learning_rate 0.0004 --epoch 100 --technique AdamQ3R --lmbda 0.1 --target_rank 0.05 --target_modules qkv --epsilon_schedule DEFAULT --N 46875 --save_location /scratch/enguye17/saved_model"
#"python main.py --dataset CIFAR10 --model VIT_Tiny --learning_rate 0.0004 --epoch 100 --technique AdamQ3R --lmbda 0.1 --target_rank 0.1 --target_modules qkv --epsilon_schedule DEFAULT --N 46875"
#"python main.py --dataset CIFAR10 --model VIT_Tiny --learning_rate 0.0004 --epoch 100 --technique AdamQ3R --lmbda 0.1 --target_rank 0.15 --target_modules qkv --epsilon_schedule DEFAULT --N 46875"

#"python main.py --dataset CIFAR10 --model VIT_Tiny --learning_rate 0.0004 --epoch 100 --mixup_active True --augmentation best --save_location /scratch/enguye17/saved_model --target_modules qkv fc1 fc2"
#"python main.py --dataset CIFAR10 --model VIT_Tiny --learning_rate 0.0004 --epoch 100 --mixup_active True --augmentation best --save_location /scratch/enguye17/saved_model --target_modules qkv fc1 fc2 --technique AdamQ3R --lmbda 0.001 --grad_clip False --target_rank 0.2 --epsilon_schedule DEFAULT"
#"python main.py --dataset CIFAR10 --model VIT_Tiny --learning_rate 0.0004 --epoch 100 --mixup_active True --augmentation best --save_location /scratch/enguye17/saved_model --target_modules qkv fc1 fc2 --technique AdamQ3R --lmbda 0.001 --grad_clip False --target_rank 0.15 --epsilon_schedule DEFAULT"
#"python main.py --dataset CIFAR10 --model VIT_Tiny --learning_rate 0.0004 --epoch 100 --mixup_active True --augmentation best --save_location /scratch/enguye17/saved_model --target_modules qkv fc1 fc2 --technique AdamQ3R --lmbda 0.001 --grad_clip False --target_rank 0.1 --epsilon_schedule DEFAULT"
#"python main.py --dataset CIFAR10 --model VIT_Tiny --learning_rate 0.0004 --epoch 100 --mixup_active True --augmentation best --save_location /scratch/enguye17/saved_model --target_modules qkv fc1 fc2 --technique AdamQ3R --lmbda 0.001 --grad_clip False --target_rank 0.05 --epsilon_schedule DEFAULT"
#"python main.py --dataset CIFAR10 --model VIT_Tiny --learning_rate 0.0004 --epoch 100 --mixup_active True --augmentation best --save_location /scratch/enguye17/saved_model --target_modules qkv fc1 fc2 --technique AdamQ3R --lmbda 0.001 --grad_clip False --target_rank 0.025 --epsilon_schedule DEFAULT"
#"python main.py --dataset CIFAR10 --model VIT_Tiny --learning_rate 0.0004 --epoch 100 --mixup_active True --augmentation best --save_location /scratch/enguye17/saved_model --target_modules qkv fc1 fc2 --technique AdamQ3R --lmbda 0.001 --grad_clip False --target_rank 0.01 --epsilon_schedule DEFAULT"

#"python main.py --dataset CIFAR10 --model VIT_Tiny --learning_rate 0.0004 --epoch 100 --mixup_active True --augmentation best --save_location /scratch/enguye17/saved_model --target_modules qkv fc1 fc2 --technique AdamQ3R --lmbda 0.01 --grad_clip False --target_rank 0.15 --epsilon_schedule DEFAULT"
#"python main.py --dataset CIFAR10 --model VIT_Tiny --learning_rate 0.0004 --epoch 100 --mixup_active True --augmentation best --save_location /scratch/enguye17/saved_model --target_modules qkv fc1 fc2 --technique AdamQ3R --lmbda 0.1 --grad_clip False --target_rank 0.15 --epsilon_schedule DEFAULT"
#"python main.py --dataset CIFAR10 --model VIT_Tiny --learning_rate 0.0004 --epoch 100 --mixup_active True --augmentation best --save_location /scratch/enguye17/saved_model --target_modules qkv fc1 fc2 --technique LoRA --target_rank 0.2"
#"python main.py --dataset CIFAR10 --model VIT_Tiny --learning_rate 0.0004 --epoch 100 --mixup_active True --augmentation best --save_location /scratch/enguye17/saved_model --target_modules qkv fc1 fc2 --technique LoRITa --depth_lorita=2 --weight_decay_alpha=0.1"
#"python main.py --dataset CIFAR10 --model VIT_Tiny --learning_rate 0.0004 --epoch 100 --mixup_active True --augmentation best --save_location /scratch/enguye17/saved_model --target_modules qkv fc1 fc2 --technique LoRITa --depth_lorita=3 --weight_decay_alpha=0.1"
#"python main.py --dataset CIFAR10 --model VIT_Tiny --learning_rate 0.0004 --epoch 100 --mixup_active True --augmentation best --save_location /scratch/enguye17/saved_model --target_modules qkv fc1 fc2 --technique LoRITa --depth_lorita=1 --weight_decay_alpha=0.01"

#"torchrun --nproc-per-node=4 main.py --dataset IMAGENET --model VIT_Base --batch_size 384 --learning_rate 0.0004 --epoch -1 --mixup_active True --technique AdamQ3R --lmbda 0.01 --grad_clip True --target_rank 0.19 --target_modules qkv --DATA_PARALLEL True --amp True --save_location /scratch/enguye17/saved_model"
#"torchrun --nproc-per-node=4 main.py --dataset IMAGENET --model VIT_Base --batch_size 384 --learning_rate 0.0004 --epoch -1 --mixup_active True --grad_clip True --target_rank 0.19 --target_modules qkv --DATA_PARALLEL True --amp True --save_location /scratch/enguye17/saved_model"

"torchrun --nproc-per-node=1 main.py --dataset IMAGENET --model VIT_Base --batch_size 384 --learning_rate 0.0004 --epoch -1 --mixup_active True --technique AdamQ3R --lmbda 0.01 --grad_clip True --target_rank 0.19 --target_modules qkv --DATA_PARALLEL True --amp True --save_location /scratch/enguye17/saved_model  --load_model_location /scratch/enguye17/saved_model/VIT_Base-2025-10-19_16_03_27-2895/model_epoch_105.pt"

)

# Select the command to run based on the SLURM_ARRAY_TASK_ID

command=${commands[$SLURM_ARRAY_TASK_ID]}

# Run the selected command using srun
echo "Running command: $command"

python -c "import torch; print(torch.__file__)"

srun $command

echo "Task ID mod 5: $(expr $SLURM_ARRAY_TASK_ID % 5)"
echo "Job ID: $SLURM_JOB_ID"
echo "Task ID: $SLURM_ARRAY_TASK_ID"
