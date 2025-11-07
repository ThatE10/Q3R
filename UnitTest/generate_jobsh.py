
LoRA_experiments = True
QuaRS_experiments = True
Hoyer_experiments = True
LoRiTA_experiments = True

target_ranks = [4,8,16,32,100]

Recectangular_experiments = True

Quars_lambda = [0.00003]
LoRiTA_alphas = [0.01,0.001,0.001]

Dataset = "CIFAR10"
Model = "ViT_Small"
BS = 64
LR = 0.0004
EPOCH = 2

base_command = f"python main.py --dataset {Dataset} --model {Model} --learning_rate {LR} --epoch {EPOCH} "

commands = []
if LoRA_experiments:
    for rank in target_ranks:
        command = base_command + f"--technique LoRA --target_rank {rank} "
        commands.append(command)

        if Recectangular_experiments:
            command += f"--rectangular_mode True"
            commands.append(command)

if QuaRS_experiments:
    for rank in target_ranks:
        for lmbda in Quars_lambda:
            command = base_command + f"--technique QuaRS --lmbda {lmbda} --target_rank {rank} "
            commands.append(command)

            if Recectangular_experiments:
                command += f"--rectangular_mode True"
                commands.append(command)

if Hoyer_experiments:
    command = base_command + f"--technique Hoyer "
    commands.append(command)

    if Recectangular_experiments:
        command += f"--rectangular_mode True"
        commands.append(command)

if LoRiTA_experiments:
    for alpha in LoRiTA_alphas:
        command = base_command + f"--technique LoRiTA --weight_decay_alpha {alpha}"
        commands.append(command)
print(f"Length: {len(commands)}")
for command in commands:
    print("\""+command+"\"")