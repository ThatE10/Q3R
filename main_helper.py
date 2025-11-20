import argparse
import os
import copy
import threading

from tqdm import tqdm

import wandb
import torch
import re
import math

from concurrent.futures import ThreadPoolExecutor
import copy

from torch import nn, optim
from torchvision import transforms
from timm.data.mixup import Mixup
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from timm.data import create_transform
from torch.cuda.amp import autocast, GradScaler


from typing import List, Tuple, Dict
from timm import create_model
from timm.models import ResNet

from Functions.AdamQ3R import AdamQ3R
from Functions.Hoyer import Hoyer
from Functions.LoRITa import LoRITaQKV, LoRITaLinear
from Functions.ModuleModificationHandler import ModuleReplacementHandler
from Functions.Q3R import Q3R
from models.DNN_model import SimpleDNN
from Functions.timer import Timer
from functools import lru_cache
from torchvision.datasets import ImageNet
from Functions.EffectiveRank import effective_rank, tail_ratio

from models.EnsembleNet import EnsNet

TIMER = Timer()

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
                                         
def parse_args(args):
    # Open and load the dictionary
    parser = argparse.ArgumentParser(description='Train a model with specified dataset and configuration.')

    # Add arguments
    parser.add_argument('--dataset', type=str, choices=['CIFAR10', 'CIFAR100', 'MNIST', 'IMAGENET'], required=True,
                        help='Dataset to use: CIFAR10, CIFAR100, or MNIST.')
    parser.add_argument('--DATA_PARALLEL', type=str2bool, default=False)
    parser.add_argument('--mixup_active', type=str2bool, default=False,help='Use Mixup for training, mainly used for ImageNet dataset formerly reffered to --imagenet_mixup_active')
    parser.add_argument('--grad_clip', type=str2bool, default=False)
    parser.add_argument('--epsilon_schedule', type=str, choices=['DEFAULT', 'linear', 'constant', 'exploration'],
                        default="DEFAULT",
                        help='Q3R hyperparameter choose epsilon update schema, if chosen linear or constant ensure to update N to be epsilon constant or total number of training iterations')

    parser.add_argument('--N', type=int, default=46875,
                        help='Q3R hyperparameter total number of iterations or constant epsilon choice')

    parser.add_argument('--model', type=str,
                        choices=['VIT_Large', 'VIT_Base', 'VIT_Tiny', 'EnsNet', 'DNN6', 'DNN8'],
                        required=True,
                        help='Model to use: ViT_Small, DNN, or RESNET.')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='Batch size for training and validation.')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='Learning rate for the optimizer.')
    parser.add_argument('--epoch', type=int, required=True,
                        help='Number of epochs for the function training')
    parser.add_argument('--technique', type=str,
                        choices=['Hoyer', 'QuaRS', 'LoRA', 'LoRITa', 'AdamQ3R', 'LoRITaQuaRS', 'LoRITaAdamQ3R'],
                        default=None,
                        help='Regularizer to use: Hoyer or Q3R.')

    parser.add_argument('--step', type=int, default=5,
                        help='Step size for the Q3R update.')

    parser.add_argument('--save_location', type=str,
                        default=None,
                        help='Chooses where to save model.pt and imspectral images')
    parser.add_argument('--load_adamq3r_state', type=str,
                        default=None,
                        help='Loads AdamQ3R state from a given file')
    parser.add_argument('--load_regulariser_location', type=str,
                        default=None,
                        help='Loads Q3R regulariser from a given file')
    
    parser.add_argument('--start_epoch', type=int,
                        default=0,
                        help='starting epch')

    parser.add_argument('--load_model_location', type=str,
                        default=None,
                        help='loads model from state given config')

    parser.add_argument('--rectangular_mode', type=str2bool, default=False,
                        help='Impacts LoRA/Hoyer/Q3R intialization and application')
    parser.add_argument('--target_rank', type=float, default=None,
                        help='Target rank for Q3R regularizer/LoRA')
    parser.add_argument('--lmbda', type=float, default=10 ** -2,
                        help='lambda value for Q3R regularize.')
    parser.add_argument('--depth_lorita', type=int, default=1,
                        help='Used in LoRITa as N')
    parser.add_argument('--weight_decay_alpha', type=float, default=10 ** -2,
                        help='alpha value for weight decay in AdamW regularizer.')
    parser.add_argument('--dropout', type=float, default=0,
                        help='Use dropout on models with dropout, values between [0,1]')

    parser.add_argument('--threads', type=int, default=4,
                        help='number of workers used to load the data')
    parser.add_argument('--amp', type=str2bool, default=False,
                        help='Use automatic mixed precision for training, requires CUDA')
    parser.add_argument('--augmentation', type=str, default=None,choices=['basic', 'best'],
                        help='Used to apply data augmentation onto datasets, mainly used on CIFAR10/100')
    parser.add_argument('--pretrained', type=str2bool, default=False,
                        help='Use pretrained weights for the model, only applies to ViT models')
    
    parser.add_argument('--target_modules', nargs='+', type=str, default=[],
                        help='List of module names, e.g., Q K V')

    args = parser.parse_args(args)

    return args


def create_experiment_name(config):
    experiment_name = f"{config.model}"

    if config.technique:
        experiment_name += f"_{config.technique}"
        if config.technique == "LoRITa":
            experiment_name += f"-depth{config.depth_lorita}-{config.weight_decay_alpha}"

        if config.technique == "LoRA" or config.technique == "Q3R":
            if config.target_rank:
                experiment_name += f"_rank{config.target_rank}"
            if config.rectangular_mode:
                experiment_name += "_rectangular"

    return experiment_name


def get_model_memory(model):
    total_params = sum(p.numel() for p in model.parameters())

    # Convert to millions for readability
    total_params_m = total_params / 1e6

    return total_params, total_params_m


def dataset_parser(config):
    """
    Parses the specified dataset and returns its properties and dataset objects.

    Parameters:
        config: Configuration object with dataset, augmentation, thread details.

    Returns:
        Tuple[Tuple[int, int, int, int], int, DataLoader, DataLoader]:
            - size (Tuple[int, int, int, int]): (N, C, H, W) shape of the dataset.
            - labels (int): The number of classes.
            - train_loader (DataLoader): The DataLoader for the training dataset.
            - validation_loader (DataLoader): The DataLoader for the validation dataset.
    """
    
    # Define transforms
    if config.augmentation == "basic":
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        transform_test = transforms.Compose([
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
    elif config.augmentation == "best":
        transform_train = create_transform(
            input_size=224,
            is_training=True,
            color_jitter=0.4,
            auto_augment='rand-m20-n2-mstd0.5',
            interpolation='bicubic',
            re_prob=0.25,  # Random erasing probability
            re_mode='pixel',
            re_count=1,
            mean = [0.4914, 0.4822, 0.4465],
            std  = [0.2470, 0.2435, 0.2616]

        )

        transform_test = create_transform(
            input_size=224,
            is_training=False,
            interpolation='bicubic',
            mean = [0.4914, 0.4822, 0.4465],
            std  = [0.2470, 0.2435, 0.2616]
        )
    else:
        transform_train = transforms.Compose([transforms.Resize(224), transforms.ToTensor()])
        transform_test = transforms.Compose([transforms.Resize(224), transforms.ToTensor()])

    # Load dataset
    if config.dataset == "CIFAR10":
        from torchvision.datasets import CIFAR10
        C, H, W, labels = 3, 224, 224, 10
        training_set = CIFAR10('../data/CIFAR10_DATA', train=True, transform=transform_train, download=True)
        validation_set = CIFAR10('../data/CIFAR10_DATA', train=False, transform=transform_test, download=True)

    elif config.dataset == "CIFAR100":
        from torchvision.datasets import CIFAR100
        C, H, W, labels = 3, 224, 224, 100
        training_set = CIFAR100('../data/CIFAR100_DATA', train=True, transform=transform_train, download=True)
        validation_set = CIFAR100('../data/CIFAR100_DATA', train=False, transform=transform_test, download=True)

    elif config.dataset == "MNIST":
        from torchvision.datasets import MNIST
        C, H, W, labels = 1, 28, 28, 10
        training_set = MNIST('../data/MNIST', train=True, download=True, transform=transforms.ToTensor())
        validation_set = MNIST('../data/MNIST', train=False, download=True, transform=transforms.ToTensor())



    elif config.dataset == "IMAGENET":
        C, H, W, labels = 3, 224, 224, 1000  # 1000 classes in standard ImageNet-1K

        # Best performing configuration for ViT-Base on ImageNet
        transform_train = create_transform(
            input_size=224,
            is_training=True,
            color_jitter=0.4,
            auto_augment='rand-m20-n2-mstd0.5',
            interpolation='bicubic',
            re_prob=0.25,  # Random erasing probability
            re_mode='pixel',
            re_count=1,
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        )

        # Validation transform
        transform_test = create_transform(
            input_size=224,
            is_training=False,
            interpolation='bicubic',
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        )

    

        training_set = ImageFolder('/users/enguye17/data/ImageNet/train/', transform=transform_train)
        # validation_set = ImageNet('/users/enguye17/data/ImageNet/test/', transform=transform_test)
        validation_set = ImageFolder('/users/enguye17/data/ImageNet/val/', transform=transform_test)
    else:
        raise ValueError(f"Dataset '{config.dataset}' is not recognized.")

    if config.DATA_PARALLEL:
        train_sampler = torch.utils.data.distributed.DistributedSampler(training_set,num_replicas=config.world_size, rank=config.local_rank, shuffle=True)
        val_sampler = torch.utils.data.distributed.DistributedSampler(validation_set,num_replicas=config.world_size, rank=config.local_rank, shuffle=True)            

        train_loader = DataLoader(
            training_set,
            batch_size=config.batch_size,
            sampler=train_sampler,  # Use sampler instead of shuffle
            pin_memory=True,
            prefetch_factor=2,
            persistent_workers=True if config.threads > 0 else False,
            num_workers=config.threads,
        )

        validation_loader = DataLoader(
            validation_set,
            batch_size=config.batch_size,
            sampler=val_sampler,  # Use sampler instead of shuffle
            pin_memory=True,
            prefetch_factor=2,
            persistent_workers=True if config.threads > 0 else False,
            num_workers=config.threads
        )

    else:
        train_loader = DataLoader(
            training_set,
            batch_size=config.batch_size,
            shuffle=True,
            pin_memory=True,
            persistent_workers=True if config.threads > 0 else False,
            num_workers=config.threads
        )

        validation_loader = DataLoader(
            validation_set,
            batch_size=config.batch_size,
            shuffle=False,
            pin_memory=True,
            persistent_workers=True if config.threads > 0 else False,
            num_workers=config.threads
        )
    size = (len(training_set), C, H, W)
    if config.augmentation:
        size = (len(training_set), size[1], 224, 224)

    return size, labels, train_loader, validation_loader


def model_parser(size: Tuple[int, int, int, int], labels: int, config) -> Tuple[nn.Module, List[nn.Module]]:
    """
    Parses the specified model configuration and prepares it based on the given technique.

    Parameters:
        model (str): The name of the model to configure. Supported options are "ViT_Small" and "DNN".
        technique (str): The technique to apply to the model (e.g., 'LoRA').
        rectangular_mode (bool): Flag to indicate whether rectangular mode is enabled.

    Returns:
        Tuple[nn.Module, List[nn.Module]]:
            - model (nn.Module): The constructed and configured model.
            - layers (List[nn.Module]): A list of key layers or modules extracted from the model for further processing.
    """
    image_size = (size[2], size[3])
    channels = size[1]

    if "VIT" in config.model:
        if config.model == "VIT_Base":
            model_config = 'timm/vit_base_patch16_224'
        elif config.model == "VIT_Tiny":
            model_config = 'timm/vit_tiny_patch16_224'
        elif config.model == 'VIT_Large':
            model_config = 'timm/vit_large_patch16_224'
        else:
            raise ValueError("No VIT model assoicated... ")
        if image_size != (224, 224) or channels != 3:
            raise ValueError("VIT_base can only accept 224 sized images, and channels = 3")

        # if technique != 'LoRA':
        model = create_model(model_config, pretrained=config.pretrained, num_classes=labels, )
        model.to(config.DEVICE)

        for block in model.blocks:
            qkv_weight = block.attn.qkv.weight
            dim = qkv_weight.shape[0] // 3

            block.attn.qkv.q_weight = qkv_weight[:dim]
            block.attn.qkv.k_weight = qkv_weight[dim:2 * dim]
            block.attn.qkv.v_weight = qkv_weight[2 * dim:]


    elif config.model == "DNN6":
        model = SimpleDNN(
            layer_sizes=[512, 512, 512, 512, 512],
            NUM_CLASSES=labels,
            IMAGE_SIZE=image_size[1],
            CHANNELS=size[1]
        )
        model.to(config.DEVICE)

    elif config.model == "DNN8":
        model = SimpleDNN(
            layer_sizes=[512, 512, 512, 512, 512, 512, 512],
            NUM_CLASSES=labels,
            IMAGE_SIZE=image_size[1],
            CHANNELS=size[1]
        )
        model.to(config.DEVICE)


    elif config.model == "EnsNet":
        if config.dataset != "MNIST":
            raise ValueError("Must train on mnist")
        model = EnsNet(num_subnetworks=10).to(config.DEVICE)

    else:
        raise ValueError(f"{config.model} is not a valid option for args, please choose from options in parser.  ")

    total_params, total_memory = get_model_memory(model)
    print(f"Total parameters: {total_params}")
    print(f"Total memory (MB): {total_memory / (1024 ** 2):.2f}")

    return model


def extract_linear(model, config) -> Dict[nn.Linear, List[Tuple]]:
    extracted_layers = {}

    for name, module in model.named_modules():
        for named_module in config.target_modules:
            if isinstance(module, nn.Linear):
                extracted_layers.update({module: None})
            if hasattr(module, named_module):
                attr = getattr(module, named_module)
                # print(f"Module Name: {name} / {module}\nSearch Name: {named_module}\n attr:{attr}")

                if isinstance(attr, nn.Linear) or (isinstance(attr, torch.Tensor) and attr.dim() >= 2):
                    if hasattr(module, "qkv"):

                        qkv_weight = attr.weight
                        dim = qkv_weight.shape[0] // 3

                        attr.q_weight = qkv_weight[:dim]
                        attr.k_weight = qkv_weight[dim:2 * dim]
                        attr.v_weight = qkv_weight[2 * dim:]

                        extracted_layers.update({module.qkv: [(0, dim), (dim, 2 * dim), (2 * dim, -1), ]})
                    else:
                        extracted_layers.update({attr: None})
                else:
                    print(type(attr))
                    print(f"Module {named_module} is not a Linear Layer or Weight")
                    raise (f"Module {named_module} is not a Linear Layer or Weight")
    return extracted_layers


def extract_modules(model, config):
    extracted_modules = []

    for name, module in model.named_modules():
        for named_module in config.target_modules:
            if hasattr(module, named_module):
                extracted_modules.append((name + f".{named_module}", getattr(module, named_module)))

    return extracted_modules


def configure_model_experiment(model, config):
    """
    DoRA, LoRA, LoRITa all configure the running weights in some way
     -  Module injection necessary therefore custom implementation of certain modules (QKV, MLP)

    Hoyer, Q3R, LoRITa have custom regularizers as well that should be added to it.

    """

    handler = ModuleReplacementHandler(model)

    regulariser = None

    captured_lorita = []
    if config.technique == "Truncate":
        modules_to_modify = []
        for name, module in model.named_modules():
            if isinstance(module, LoRITaQKV):
                captured_lorita.append(name)
                modules_to_modify.append((name, module))
            elif isinstance(module, LoRITaLinear):
                captured_lorita.append(name)
                modules_to_modify.append((name, module))

        print(f"LoRITa Modules: {captured_lorita}")

        for name, module in modules_to_modify:
            if isinstance(module, LoRITaQKV):
                handler.simplify_lorita_qkv_module(name, module)
            elif isinstance(module, LoRITaLinear):
                if "attn" in name:
                    continue
                handler.simplify_lorita_module(name, module)

    modules_to_modify = extract_modules(model, config)
    extracted_layers = extract_linear(model, config)

    print(modules_to_modify)

    if config.technique == "LoRA":
        for name, module in modules_to_modify:
            if "qkv" in name:
                handler.add_qkv_lora(name, rank=config.target_rank)
            else:
                handler.add_lora_layer(name, rank=config.target_rank)

    elif config.technique == "LoRITa":

        for name, module in modules_to_modify:
            if "qkv" in name:
                handler.add_qkv_lorita(name, N=config.depth_lorita)
            else:
                handler.add_lorita_layer(name, N=config.depth_lorita)

    elif config.technique == "Truncate":
        if config.target_rank is None:
            raise ValueError("Target rank is none, should be initialized between 0<x<1 or a positive integer")
        modules_to_modify = extract_modules(model, config)
        

        for name, module in modules_to_modify:
            rank = config.target_rank
            if "qkv" in name:
                if hasattr(config, 'spectral_density') and config.spectral_density != 0.0:
                    """function that finds r such that it is density<SPD_r(X)
                rank = max(r,config.target_rank)
                """
                    density = config.spectral_density
                    qkv_module = handler._get_module_by_name(name)
                    dim = qkv_module.weight.shape[0] // 3 

                    if 0 <= rank < 1:
                        rank = int(math.floor((rank * dim * dim) / (2 * dim)))
                    
                    list_of_weights = [qkv_module.weight[:dim],qkv_module.weight[dim:2 * dim], qkv_module.weight[2 * dim:]]
                    for index, weight in enumerate(list_of_weights):
                        _, S, _ = torch.linalg.svd(weight, full_matrices=False)

                        prob_S = S / S.sum()
                        tail_ratio_val = 0.0
                        
                        for i,s  in enumerate(prob_S):
                            tail_ratio_val+=s
                            if density <= tail_ratio_val:
                                rank = i
                                break
                            else:
                                rank = max(rank,i)
                    print(f"target_rank: {config.target_rank}, rank: {rank}, density: {density}, tr: {tail_ratio(weight, rank, S)}")
                                            
                
                handler.truncate_qkv_module(name, rank=rank)
            else:                
                
                if hasattr(config, 'spectral_density') and config.spectral_density != 0.0:
                    density = config.spectral_density
                    mlp_module = handler._get_module_by_name(name)
                    weight = mlp_module.weight
                    if 0 <= rank < 1:
                        true_rank = int(math.floor((rank * min(weight.shape) * max(weight.shape)) / (min(weight.shape) + max(weight.shape))))
                        rank = true_rank
                    _, S, _ = torch.linalg.svd(weight, full_matrices=False)

                    tail_ratio_val = 0.0
                    prob_S = S / S.sum()

                    for i,s  in enumerate(prob_S):
                        tail_ratio_val+=s
                        rank = max(rank,i)
                        if density <= tail_ratio_val:
                            break
                    
                    print(f"target_rank: {config.target_rank}, rank: {rank}, density: {density}, tr: {tail_ratio(weight, rank,S)}")
                                            
                handler.truncate_module(name, rank=rank)

        """for name, module in modules_to_modify:
            if "VIT" in config.model:
                if regex_qkv.match(name):
                    handler.truncate_qkv_module(name, rank=config.target_rank)
                if "FC1" in config.target_modules or "FC2" in config.target_modules or "MLP" in config.target_modules:
                    if regex_mlp1.match(name):
                        handler.truncate_module(name, rank=config.target_rank)
                    if regex_mlp2.match(name):
                        handler.truncate_module(name, rank=config.target_rank)

        modules_to_modify = extract_modules(model, config)
    extracted_layers = extract_weight_or_linear(model, config)

    print(modules_to_modify)

    if config.technique == "LoRA":

                """
    elif config.technique == "Q3R":
        if config.rectangular_mode:
            rectangular_mode = 1

        regulariser = Q3R(trainable_modules=extracted_layers, target_rank=config.target_rank, lmbda=1, verbose=True,
                            N=config.N,
                            epsilon_schedule=config.epsilon_schedule, steps=config.step)

    elif config.technique == "Hoyer":
        regulariser = Hoyer(trainable_weights=extracted_layers)

    elif config.technique == "LoRITaQuaRS":

        for name, module in modules_to_modify:  # First convert the model architecture into a LoRITa type
            if "qkv" in name:
                handler.add_qkv_lorita(name, N=config.depth_lorita)
            else:
                handler.add_lorita_layer(name, N=config.depth_lorita)

        if config.rectangular_mode:
            rectangular_mode = 1
        else:
            rectangular_mode = 0

        extracted_modules = extract_modules(model, config)  # extract modified LoRITa modules

        extracted_weights = []
        for name, module in extracted_modules:
            print(module)
            extracted_weights.extend(module.get_layer_reference())  # get the linear layer references to pass into QuaRS

        print(f"LoRITa Q3R weights: {extracted_weights}")

        regulariser = Q3R(trainable_modules=extracted_weights, target_rank=config.target_rank, lmbda=1, verbose=True,
                            N=config.N,
                            epsilon_schedule=config.epsilon_schedule, steps=config.step)

    elif config.technique == "LoRITaAdamQ3R":

        for name, module in modules_to_modify:  # First convert the model architecture into a LoRITa type
            if "qkv" in name:
                handler.add_qkv_lorita(name, N=config.depth_lorita)
            else:
                handler.add_lorita_layer(name, N=config.depth_lorita)

        """
            todo: add code to extract LoRiTA layers if using them in in the experiment
            """

    else:
        print(f"NO TECHNIQUE APPLIED TO {config.technique}")
    total_params, total_memory = get_model_memory(model)
    print(f"Configured total parameters: {total_params}")
    print(f"Configured total memory (MB): {total_memory / (1024 ** 2):.2f}")

    return regulariser


def get_grad_memory(model):
    total_grad_size = 0
    try:
        for param in model.parameters():
            if param.grad is not None:  # Ensure the gradient exists
                total_grad_size += param.grad.numel() * param.grad.element_size()
        return total_grad_size
    except:
        return 0


def optim_parser(model, config):
    if config.technique == "LoRITa":
        if not config.weight_decay_alpha:
            raise ValueError("Argument_weight decay not given")
        optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay_alpha)
    elif config.technique == "AdamQ3R":

        trainable_modules = extract_linear(model, config)
        print(f"AdamQ3R: {trainable_modules}")
        if len(trainable_modules) == 0:
            raise ValueError("Config not passed valid trainable weights")
        optimizer = AdamQ3R(model.parameters(), trainable_modules=trainable_modules, target_rank=config.target_rank,
                            lr=config.learning_rate, lmbda=config.lmbda,
                            N=config.N,
                            epsilon_schedule=config.epsilon_schedule, steps=config.step
                            )
    elif config.technique == "LoRITaAdamQ3r":
        if config.rectangular_mode:
            rectangular_mode = 1
        else:
            rectangular_mode = 0

        extracted_modules = extract_modules(model, config)  # extract modified LoRITa modules

        extracted_modules = []
        for name, module in extracted_modules:
            print(module)
            extracted_modules.extend(module.get_layer_reference())  # get the linear layer references to pass into QuaRS

        print(f"LoRITa Q3R weights: {extracted_modules}")

        optimizer = AdamQ3R(model.parameters(), trainable_weights=extracted_modules, target_rank=config.target_rank,
                            lr=config.learning_rate, lmbda=config.lmbda,
                            N=config.N,
                            epsilon_schedule=config.epsilon_schedule
                            )
    else:
        optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)

    return optimizer


def train_epoch(optimizer, criterion, model, train_loader, rank_regularizer, logger_object, config):
    model.train()
    DEVICE = next(model.parameters()).device
    scaler = torch.cuda.amp.GradScaler(enabled=config.amp)
    
    train_regularizer = torch.tensor(0.0, device=DEVICE)
    train_loss = torch.tensor(0.0, device=DEVICE)
    train_total_accuracy = torch.tensor(0.0, device=DEVICE)
    
    if config.local_rank == 0 and logger_object:
        logger_object.start_epoch()
    
    # Only show progress bar on rank 0
    if config.local_rank == 0:
        progress_bar = tqdm(train_loader, desc="Training", leave=False)
    else:
        progress_bar = train_loader

    if config.mixup_active:
        # Use Mixup for ImageNet if specified in the config
        print("Using Mixup")
        cutmix_or_mixup = Mixup(
                    mixup_alpha=0.8, cutmix_alpha=1.0, cutmix_minmax=None,
                    prob=1.0, switch_prob=0.5, mode='batch',
                    label_smoothing=0.1, num_classes=config.labels)
    
    for i, data in enumerate(progress_bar):
        
        inputs, labels = data

        if config.mixup_active:
            inputs, labels = cutmix_or_mixup(inputs, labels)

        inputs, labels = inputs.to(DEVICE, non_blocking=True), labels.to(DEVICE, non_blocking=True)
        
        optimizer.zero_grad()
        
        with torch.cuda.amp.autocast(enabled=bool(config.amp)):
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            if rank_regularizer is not None:
                rank_regularizer.update()
                total_loss = loss + rank_regularizer.val
                print(f"Rank Regularizer Value: {rank_regularizer.val.item()}")
            else:
                total_loss = loss
        
        # Backward pass
        if config.amp:
            scaler.scale(total_loss).backward()

            if config.grad_clip:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            total_loss.backward()
            if config.grad_clip:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
        
        # REMOVE most synchronize calls - DDP handles synchronization automatically
        # torch.cuda.synchronize()  # <- REMOVE THIS
        
        # EnsNet-specific logic
        if isinstance(model, EnsNet):
            with torch.cuda.amp.autocast(enabled=bool(config.amp)):
                subnet_loss = model.train_subnetworks(inputs, labels, optimizer, criterion)
        
        # Accumulate metrics
        if not config.mixup_active:
            with torch.no_grad():
                correct = (torch.argmax(outputs, dim=1) == labels).float().sum()
                train_total_accuracy += correct
                train_loss += loss.detach()
                if rank_regularizer is not None:
                    train_regularizer += rank_regularizer.val.detach()
        
        # Reduce logging frequency and make it less expensive
        """if config.local_rank == 0 and logger_object and i % 10 == 0:  # Every 10 batches instead of every batch
            logger_object.regularise_log()"""
        
        # Update progress bar only on rank 0
        if config.local_rank == 0:
            progress_bar.set_postfix(loss=f"{loss.item():.4f}")
    
    if config.local_rank == 0 and logger_object:
        logger_object.end_epoch()
    
    # Aggregate metrics across all ranks for DDP
    if config.DATA_PARALLEL:
        torch.distributed.all_reduce(train_loss, op=torch.distributed.ReduceOp.SUM)
        torch.distributed.all_reduce(train_regularizer, op=torch.distributed.ReduceOp.SUM)
        torch.distributed.all_reduce(train_total_accuracy, op=torch.distributed.ReduceOp.SUM)
        
        # Average across ranks
        train_loss /= config.world_size
        train_regularizer /= config.world_size
        train_total_accuracy /= config.world_size
    
    return {
        "train_loss": train_loss.detach().cpu(),
        "train_reg": train_regularizer.detach().cpu(),
        "train_obj": train_loss.detach().cpu() + train_regularizer.detach().cpu(),
        "train_acc": train_total_accuracy.detach().cpu()
    }
def train_epoch_new(optimizer, criterion, model, train_loader, rank_regularizer, logger_object, config):
    model.train()
    accelerator = config.accelerator
    DEVICE = accelerator.device
    
    # Accelerator handles mixed precision automatically - no need for manual GradScaler
    
    train_regularizer = torch.tensor(0.0, device=DEVICE)
    train_loss = torch.tensor(0.0, device=DEVICE)
    train_total_accuracy = torch.tensor(0.0, device=DEVICE)
    
    if accelerator.is_main_process and logger_object:
        logger_object.start_epoch()
    
    # Only show progress bar on main process
    if accelerator.is_main_process:
        progress_bar = tqdm(train_loader, desc="Training", leave=False)
    else:
        progress_bar = train_loader

    if config.mixup_active:
        # Use Mixup for ImageNet if specified in the config
        if accelerator.is_main_process:
            print("Using Mixup")
        cutmix_or_mixup = Mixup(
            mixup_alpha=0.8, cutmix_alpha=1.0, cutmix_minmax=None,
            prob=1.0, switch_prob=0.5, mode='batch',
            label_smoothing=0.1, num_classes=config.labels)
    
    for i, data in enumerate(progress_bar):
        
        inputs, labels = data

        if config.mixup_active:
            inputs, labels = cutmix_or_mixup(inputs, labels)

        # No need to move to device - accelerator.prepare() already handled this
        # inputs, labels = inputs.to(DEVICE, non_blocking=True), labels.to(DEVICE, non_blocking=True)
        
        # Gradient accumulation context (if configured)
        with accelerator.accumulate(model):
            optimizer.zero_grad()
            
            # Accelerator handles mixed precision automatically
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            if rank_regularizer is not None:
                rank_regularizer.update()
                total_loss = loss + rank_regularizer.val
                if accelerator.is_main_process and i % 100 == 0:  # Less frequent printing
                    print(f"Rank Regularizer Value: {rank_regularizer.val.item()}")
            else:
                total_loss = loss
            
            # Use accelerator.backward() - handles mixed precision and DDP automatically
            accelerator.backward(total_loss)
            
            # Gradient clipping through accelerator
            if config.grad_clip:
                accelerator.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
        
        # EnsNet-specific logic
        if isinstance(model, EnsNet):
            # Note: For EnsNet, you might need to unwrap the model
            # unwrapped_model = accelerator.unwrap_model(model)
            subnet_loss = model.train_subnetworks(inputs, labels, optimizer, criterion)
        
        # Accumulate metrics
        if not config.mixup_active:
            with torch.no_grad():
                correct = (torch.argmax(outputs, dim=1) == labels).float().sum()
                train_total_accuracy += correct
                train_loss += loss.detach()
                if rank_regularizer is not None:
                    train_regularizer += rank_regularizer.val.detach()
        
        # Update progress bar only on main process
        if accelerator.is_main_process:
            progress_bar.set_postfix(loss=f"{loss.item():.4f}")
    
    if accelerator.is_main_process and logger_object:
        logger_object.end_epoch()
    
    # Aggregate metrics across all processes using Accelerate
    # gather_for_metrics preserves the total across all processes
    train_loss = accelerator.reduce(train_loss, reduction="mean")
    train_regularizer = accelerator.reduce(train_regularizer, reduction="mean")
    train_total_accuracy = accelerator.reduce(train_total_accuracy, reduction="sum")
    
    return {
        "train_loss": train_loss.detach().cpu(),
        "train_reg": train_regularizer.detach().cpu(),
        "train_obj": train_loss.detach().cpu() + train_regularizer.detach().cpu(),
        "train_acc": train_total_accuracy.detach().cpu()
    }


def test_model(criterion, model, test_loader, size, labels, config, override_module_truncation=None):
    results = {}
    criterion = nn.CrossEntropyLoss()
    if config.technique == "LoRA" or config.technique == "Truncate":
        print("Truncated/LoRA model")
    elif config.technique is None or config.technique == "LoRITa" or config.technique == "QuaRS" or config.technique == "Hoyer" or config.technique == "LoRITaQuaRS" or config.technique == "AdamQ3R" or config.technique == "LoRITaAdamQ3R":
        test_results = test_truncated_model(criterion, model, test_loader, size, labels, config,
                                            override_module_truncation)
        results.update(test_results)
    
    return results

def test_truncated_model(criterion, model, test_loader, size, labels, config, override_module_truncation=None):
    if config.technique == "LoRA":
        raise ValueError("LoRA is not supported for truncation")

    results = {}

    if config.local_rank == 0:
        print("Testing truncated models")

    def test_single_model_ddp(cloned_model, cloned_config, target_rank, test_loader):
        """Test a single model using DDP with distributed validation"""
        test_results = {}
        test_loss = 0
        test_total_accuracy = 0
        total_samples = 0
        
        cloned_model.eval()
        
        with torch.no_grad():
            if config.local_rank == 0:
                progress_bar = tqdm(test_loader, desc=f"Testing {target_rank * 100:.1f}%", leave=False)
            else:
                progress_bar = test_loader
                
            for i, data in enumerate(progress_bar):
                inputs, labels_batch = data
                inputs = inputs.to(config.DEVICE)
                labels_batch = labels_batch.to(config.DEVICE)

                outputs = cloned_model(inputs)
                loss = criterion(outputs, labels_batch)

                batch_accuracy = torch.sum(torch.argmax(outputs, 1) == labels_batch).detach()
                test_total_accuracy += batch_accuracy
                test_loss += loss.detach()
                total_samples += labels_batch.size(0)
                
                if config.local_rank == 0:
                    progress_bar.set_postfix(loss=f"{loss.item():.4f}")

        # Gather results from all GPUs
        if config.DATA_PARALLEL:
            # Convert to tensors for all_reduce
            test_loss_tensor = test_loss.clone().detach()
            test_accuracy_tensor = test_total_accuracy.clone().detach()
            total_samples_tensor = torch.tensor(total_samples, dtype=torch.long, device=config.DEVICE)
            
            # Sum across all processes
            torch.distributed.all_reduce(test_loss_tensor, op=torch.distributed.ReduceOp.SUM)
            torch.distributed.all_reduce(test_accuracy_tensor, op=torch.distributed.ReduceOp.SUM)
            torch.distributed.all_reduce(total_samples_tensor, op=torch.distributed.ReduceOp.SUM)
            
            # Average the loss and accuracy
            test_results["test_loss"] = test_loss_tensor / config.world_size
            test_results["test_acc"] = test_accuracy_tensor.float()
            test_results["total_samples"] = total_samples_tensor
        else:
            test_results["test_loss"] = test_loss
            test_results["test_acc"] = test_total_accuracy.float() 
            test_results["total_samples"] = total_samples

        return test_results

    def create_distributed_test_loader(original_loader, rank, world_size):
        """Create a distributed test loader for DDP"""
        if config.DATA_PARALLEL:
            # Create distributed sampler for test set
            test_sampler = torch.utils.data.distributed.DistributedSampler(
                original_loader.dataset,
                num_replicas=world_size,
                rank=rank,
                shuffle=False
            )
            
            distributed_loader = torch.utils.data.DataLoader(
                original_loader.dataset,
                batch_size=original_loader.batch_size,
                sampler=test_sampler,
                pin_memory=original_loader.pin_memory,
                num_workers=original_loader.num_workers,
                persistent_workers=getattr(original_loader, 'persistent_workers', False)
            )
            return distributed_loader
        else:
            return original_loader

    # Define target ranks to test
    target_ranks = [0.1,0.15,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]
    spectral_energy = [0.0]#[0.5,0.6,0.7,0.8,0.9,1.0]

    if config.DATA_PARALLEL:
        # DDP Implementation: Test one model at a time across all GPUs
        for density in spectral_energy:
            for target_rank in target_ranks:
                if config.local_rank == 0:
                    print(f"Running truncated model {target_rank * 100}% original parameters on {config.target_modules}")

                # Clone model and config
                if target_rank == 1.0 or density == 1.0:
                    cloned_config = copy.deepcopy(config)
                    cloned_config.technique = "Baseline"
                    # Use the original DDP wrapped model
                    cloned_model = model
                    if "test_acc_Baseline" in results.keys():
                        print("Baseline model already tested, skipping...")
                        continue
                else:
                    model.cpu()
                    cloned_model = model_cloner(model, size, labels, config)
                    cloned_model = cloned_model.to(config.DEVICE)
                    cloned_config = copy.deepcopy(config)

                    
                    cloned_config.technique = "Truncate"
                    if override_module_truncation is not None:
                        cloned_config.target_modules = override_module_truncation
                    cloned_config.target_rank = target_rank
                    cloned_config.spectral_density = density

                # Configure model experiment
                if target_rank != 1.0:
                    configure_model_experiment(cloned_model, cloned_config)

                # Create distributed test loader
                distributed_test_loader = create_distributed_test_loader(test_loader, config.local_rank, config.world_size)

                # Test model
                test_results = test_single_model_ddp(cloned_model, cloned_config, target_rank, distributed_test_loader)

                # Only log results on main process
                if config.local_rank == 0:
                    loss, acc = test_results["test_loss"], test_results["test_acc"]
                    if target_rank == 1.0:
                        print(f"Baseline | Loss {loss:.4f} | Accuracy {acc:.4f}")
                        results.update({f"{k}_Baseline": v for k, v in test_results.items()})
                    else:
                        print(f"{target_rank * 100}% SPD: {density}% original parameters | Loss {loss:.4f} | Accuracy {acc:.4f}")
                        results.update({f"{k}_{target_rank}_{density}": v for k, v in test_results.items()})

                # Synchronize all processes before moving to next model
                if target_rank != 1.0:
                    del cloned_model
                    torch.cuda.empty_cache()
                    import gc
                    gc.collect()
                    model.to(config.DEVICE)
                torch.distributed.barrier()

    else:
        # Single GPU Implementation: Test multiple models concurrently
        thread_local = threading.local()

        def process_target_rank(target_rank):
            if config.local_rank == 0:
                print(f"Running truncated model {target_rank * 100}% original parameters on {config.target_modules}")

            # Clone model and config
            if target_rank == 1.0:
                cloned_config = copy.deepcopy(config)
                cloned_config.technique = "Baseline"
                cloned_model = model
            else:
                cloned_model = model_cloner(model, size, labels, config)
                cloned_config = copy.deepcopy(config)
                cloned_config.technique = "Truncate"
                if override_module_truncation is not None:
                    cloned_config.target_modules = override_module_truncation
                cloned_config.target_rank = target_rank

            # Create thread-local test loader
            if not hasattr(thread_local, 'test_loader'):
                thread_local.test_loader = torch.utils.data.DataLoader(
                    test_loader.dataset,
                    batch_size=test_loader.batch_size,
                    shuffle=False,
                    num_workers=0,
                    pin_memory=test_loader.pin_memory,
                    persistent_workers=False
                )

            # Configure and test model
            if target_rank != 1.0:
                configure_model_experiment(cloned_model, cloned_config)

            cloned_model.eval()

            # Test the model
            test_results = {}
            test_loss = 0
            test_total_accuracy = 0
            total_samples = 0

            with torch.no_grad():
                progress_bar = tqdm(thread_local.test_loader, desc=f"Testing {target_rank * 100:.1f}%", leave=False)
                for i, data in enumerate(progress_bar):
                    inputs, labels_batch = data
                    inputs = inputs.to(config.DEVICE)
                    labels_batch = labels_batch.to(config.DEVICE)

                    outputs = cloned_model(inputs)
                    loss = criterion(outputs, labels_batch)

                    batch_accuracy = torch.sum(torch.argmax(outputs, 1) == labels_batch).detach()
                    test_total_accuracy += batch_accuracy
                    test_loss += loss.detach()
                    total_samples += labels_batch.size(0)
                    progress_bar.set_postfix(loss=f"{loss.item():.4f}")

            test_results["test_loss"] = test_loss
            test_results["test_acc"] = test_total_accuracy.float()
            test_results["total_samples"] = total_samples

            loss, acc = test_results["test_loss"], test_results["test_acc"]
            if target_rank == 1.0:
                print(f"Baseline | Loss {loss:.4f} | Accuracy {acc:.4f}")
                return {f"{k}_Baseline": v for k, v in test_results.items()}
            else:
                print(f"{target_rank * 100}% original parameters | Loss {loss:.4f} | Accuracy {acc:.4f}")
                return {f"{k}_{target_rank}": v for k, v in test_results.items()}

        # Process target ranks in batches of 4 for single GPU
        for i in range(0, len(target_ranks), 4):
            batch = target_ranks[i:i + 4]

            with ThreadPoolExecutor(max_workers=4) as executor:
                futures = [executor.submit(process_target_rank, rank) for rank in batch]

                for future in futures:
                    try:
                        batch_result = future.result()
                        results.update(batch_result)
                    except Exception as e:
                        print(f"Error processing batch: {e}")
                        import traceback
                        traceback.print_exc()

    return results



def model_cloner(base_model, size, labels, config):
    cloned_model = model_parser(size, labels, config)  # creates model architecture
    regularizer = configure_model_experiment(cloned_model, config)  # restructures model into correct shape
    cloned_model.to(config.DEVICE)
    if config.DATA_PARALLEL:
        print("Converting ")
        local_rank = torch.distributed.get_rank()

        cloned_model = torch.nn.parallel.DistributedDataParallel(
            cloned_model,
            device_ids=[local_rank],  # Use local_rank instead of cuda.current_device()
            output_device=local_rank,
            broadcast_buffers=False
        )
       

    cloned_model.load_state_dict(base_model.state_dict())

    return cloned_model


def extract_trainable_weights(trainable_modules: Dict[nn.Linear, List[Tuple]]):
    """Args:
        trainable_modules: Dict where keys are modules (e.g., nn.Linear) and values are:
                   - None: use full module.weight
                   - List of (start, end) tuples: use slices of module.weight

    Returns:
        List of weight tensors (or slices thereof) to be used for training or regularization.
    """
    weights = []
    for module, slice_list in trainable_modules.items():
        if not hasattr(module, 'weight'):
            continue  # skip modules without weights
        full_weight = module.weight

        if slice_list is None:
            weights.append(full_weight)
        else:
            for start, end in slice_list:
                if end is None or end == -1:
                    weights.append(full_weight[start:])
                else:
                    weights.append(full_weight[start:end])
    return weights


def pad_tensor_with_slice_bounds(m_k: torch.Tensor, full_shape: tuple, dim: int, bounds: tuple) -> torch.Tensor:
    """
    Pads `m_k` into a tensor of shape `full_shape` along the specified dimension using flexible bounds.

    Args:
        m_k (torch.Tensor): The smaller tensor to embed.
        full_shape (tuple): Target shape of the padded tensor.
        dim (int): The dimension along which `m_k` is inserted (e.g., 0 for rows, 1 for columns).
        bounds (tuple): A (start, end) tuple, where values can be None or -1 to represent flexible slicing.

    Returns:
        torch.Tensor: A new tensor of shape `full_shape` with `m_k` inserted into the specified slice.
    """
    if bounds is None:
        return m_k
    start, end = bounds
    full_dim = full_shape[dim]

    # Normalize slice bounds
    start = 0 if start is None else start
    end = full_dim if end in (None, -1) else end

    # Validate shape compatibility
    expected_size = end - start
    if m_k.shape[dim] != expected_size:
        raise ValueError(
            f"Shape mismatch along dimension {dim}: m_k has size {m_k.shape[dim]}, expected {expected_size}")

    # Create empty padded tensor
    padded = torch.zeros(full_shape, dtype=m_k.dtype, device=m_k.device)

    # Build index slices
    index = [slice(None)] * len(full_shape)
    index[dim] = slice(start, end)

    # Assign m_k into padded tensor
    padded[tuple(index)] = m_k
    return padded
