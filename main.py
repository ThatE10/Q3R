import safetensors
import torch
from torch import nn
import os
import gc
import psutil

from Functions.data_manager import DataCollector
from main_helper import dataset_parser, model_parser, parse_args, test_model, configure_model_experiment, train_epoch, \
    optim_parser
from Functions.timer import Timer
from timm.loss import SoftTargetCrossEntropy

torch.manual_seed(0)

torch.backends.cudnn.benchmark = True

os.environ["WANDB_START_METHOD"] = "thread"
WANDB_MODE = "online"
MAX_PARALLEL_VALIDATE = 4
os.environ["WANDB_MODE"] = WANDB_MODE
TIMER = Timer()

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

def setup_ddp():
    """Initialize DDP environment with proper device validation"""

    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    torch.cuda.set_device(local_rank)
    
    torch.cuda.init()

    if not torch.distributed.is_initialized():
        torch.distributed.init_process_group(backend='nccl')

    world_size = torch.distributed.get_world_size()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")

    num_gpus = torch.cuda.device_count()
    print(f"Available GPUs: {num_gpus}, Local rank: {local_rank}")

    if local_rank >= num_gpus:
        raise RuntimeError(f"Local rank {local_rank} is >= number of available GPUs {num_gpus}")

    device = torch.device(f"cuda:{local_rank}")

    try:
        torch.cuda.empty_cache()
        _ = torch.tensor([1.0]).to(device)
        print(f"Rank {local_rank}: Successfully initialized device {device}")
    except Exception as e:
        raise RuntimeError(f"Failed to initialize device {device}: {e}")

    return local_rank, world_size, device


def cleanup_ddp():
    """Clean up DDP"""
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


def print_memory(prefix="", rank=0):
    """Print GPU memory usage"""
    if rank == 0:
        try:
            allocated = torch.cuda.memory_allocated() / 1024 ** 2
            reserved = torch.cuda.memory_reserved() / 1024 ** 2
            max_allocated = torch.cuda.max_memory_allocated() / 1024 ** 2
            max_reserved = torch.cuda.max_memory_reserved() / 1024 ** 2
            print(f"[{prefix}] Memory allocated: {allocated:.2f}MB | Reserved: {reserved:.2f}MB | "
                  f"Max Allocated: {max_allocated:.2f}MB | Max Reserved: {max_reserved:.2f}MB")
        except Exception as e:
            print(f"Error printing memory: {e}")
            print(f"[{prefix}] Failed to get memory usage: {e}")


def validate_gpu_setup():
    if not torch.cuda.is_available():
        print("CUDA is not available!")
        return False

    num_gpus = torch.cuda.device_count()
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"Number of GPUs: {num_gpus}")

    for i in range(num_gpus):
        try:
            device = torch.device(f"cuda:{i}")
            props = torch.cuda.get_device_properties(i)
            print(f"GPU {i}: {props.name} ({props.total_memory // 1024 ** 2} MB)")
            test_tensor = torch.tensor([1.0]).to(device)
        except Exception as e:
            print(f"GPU {i}: Test failed - {e}")
            return False

    return True


def main(args):
    config = args

    if config.DATA_PARALLEL:
        local_rank, world_size, device = setup_ddp()
        config.DEVICE = device
        config.local_rank = local_rank
        config.world_size = world_size

        if local_rank == 0:
            print(f"Initialized DDP with {world_size} processes")
    else:
        config.DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        config.local_rank = 0
        config.world_size = 1

    size, labels, train_loader, validation_loader = dataset_parser(config)
    model = model_parser(size, labels, config)
    config.labels = labels

    model = model.to(config.DEVICE)

    if config.local_rank == 0:
        print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
        print_memory("After model to device", config.local_rank)

    regulariser = configure_model_experiment(model, config)

    optimizer = optim_parser(model, config)
    criterion = SoftTargetCrossEntropy() if config.mixup_active else nn.CrossEntropyLoss()

    if config.DATA_PARALLEL:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            broadcast_buffers=False,
            find_unused_parameters=False,
            gradient_as_bucket_view=True,
            static_graph=True
        )

    if config.load_model_location:
        checkpoint = safetensors.torch.load_file(config.load_model_location)
        model.load_state_dict(checkpoint)
        if config.local_rank == 0:
            print(f"Loaded model from {config.load_model_location}")
    if config.load_regulariser_location:
        try:
            regulariser.load_state_dict(torch.load(config.load_regulariser_location, weights_only=False))
            if config.local_rank == 0:
                print(f"Loaded regulariser from {config.load_regulariser_location}")
        except ValueError as e:
            raise ValueError("Regulariser is not an instance Q3R, cannot load state.")
        
    if config.load_adamq3r_state:
        try:
            optimizer.load_state_dict(torch.load(config.load_adamq3r_state,weights_only=False))
            if config.local_rank == 0:
                print(f"Loaded AdamQ3R state from {config.load_adamq3r_state}")
        except ValueError as e:
            raise ValueError("Optimizer is not an instance of AdamQ3R, cannot load state.")

    logger_object = None
    if config.local_rank == 0:
        logger_object = DataCollector(
            model, optimizer, regulariser, train_loader,
            validation_loader, size, labels, config, step=1
        )

    for epoch in range(config.start_epoch, config.epoch):
        if config.local_rank == 0:
            print(f"\n--- Epoch {epoch} ---")

        if config.DATA_PARALLEL:
            train_loader.sampler.set_epoch(epoch)
            if hasattr(validation_loader.sampler, 'set_epoch'):
                validation_loader.sampler.set_epoch(epoch)

        torch.cuda.reset_peak_memory_stats()
        if config.epoch > 0:
            train_results = train_epoch(
                optimizer, criterion, model, train_loader,
                regulariser, logger_object, config
            )

        print_memory("After training", config.local_rank)

        """TIMER.start("test_time")
        with torch.no_grad():
            test_results = test_model(
                criterion, model, validation_loader,
                size, labels, config
            )
        TIMER.lap("test_time")"""

        print_memory("After validation", config.local_rank)

        if config.local_rank == 0 and logger_object:
            #train_results.update(test_results)
            logger_object.epoch_summary(train_results, model, config, verbose=False)
        
    if config.epoch <= 0:
        TIMER.start("test_time")
        with torch.no_grad():
            test_results = test_model(
                criterion, model, validation_loader,
                size, labels, config
            )
        TIMER.lap("test_time")

        print_memory("After validation", config.local_rank)

        if config.local_rank == 0 and logger_object:
            logger_object.epoch_summary(test_results, model, config, verbose=False)

    if config.local_rank == 0 and logger_object:
        logger_object.print_summary()

    if config.DATA_PARALLEL:
        cleanup_ddp()


if __name__ == "__main__":
    print("Starting distributed training script")
    args = parse_args(None)
    args.WANDB_MODE = WANDB_MODE
    args.MAX_PARALLEL_VALIDATE = MAX_PARALLEL_VALIDATE

    try:
        main(args)
    except Exception as e:
        print(f"Training failed with error: {e}")
        if torch.distributed.is_initialized():
            cleanup_ddp()
        raise
