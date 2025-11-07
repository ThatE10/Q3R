import os
import re
from typing import List, Union

import pandas as pd
import torch
import safetensors
import wandb
import matplotlib
import random


matplotlib.use("Agg")
from datetime import datetime

from matplotlib import pyplot as plt

from Functions.AdamQ3R import AdamQ3R
from Functions.EffectiveRank import effective_rank, tail_ratio
from Functions.LoRITa import LoRITaLinear, LoRITaQKV
from Functions.Q3R import QuaRS
from main_helper import create_experiment_name, extract_modules, extract_linear, \
    extract_trainable_weights
from Functions.timer import Timer

torch.autograd.set_detect_anomaly(True)

TIMER = Timer()
# Path to the file
DEVICE = torch.device("cuda")

WANDB_MODE = "online"
DATA_PARALLEL = False
PRELOAD_GPU_DATA = False

if WANDB_MODE == "offline":
    os.environ["WANDB_MODE"] = WANDB_MODE

PROJECT_NAME = "NewConfig"


class DataCollector():
    def __init__(self, model, optimizer, regulariser, train_loader, validation_loader, size, labels, config, step=1,
                 verbose=False, wandb_enabled=True):

        self.iterations = 0
        self.epoch = config.start_epoch
        self.optimizer = optimizer
        self.config = config
        self.train_loader = train_loader
        self.size = size
        self.labels = labels
        self.validation_loader = validation_loader
        self.save_all = True
        self.regulariser = regulariser
        if isinstance(optimizer, AdamQ3R):
            self.regulariser = optimizer.q3r

        date_time_str = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
        experiment_name = create_experiment_name(config)

        if config.technique in ["Baseline", "LoRITa", "Hoyer"]:
            self.truncated_accuracy = []
        else:
            if isinstance(regulariser, QuaRS):
                rect = "_rect" if regulariser.rectangular_mode else ""
                note = f"t_acc{rect}{regulariser.target_rank}"
                self.max_truncated_accuracy = None

        if config.technique in ["AdamQ3R","Hoyer","QuaRS"]:
            self.trainable_modules = extract_linear(model, config)
            trainable_weights = extract_trainable_weights(self.trainable_modules)
        else:
            try:
                extracted_modules = extract_modules(model, config)  # extract modified LoRITa modules

                extracted_weights: List[Union[LoRITaLinear, LoRITaQKV]] = []
                for name, module in extracted_modules:
                    print(module)
                    if isinstance(module, LoRITaLinear) or isinstance(module, LoRITaQKV):
                        extracted_weights.extend(module)
                self.trainable_modules = extracted_weights
            except Exception as e:
                print(f"Error Processing weights: {e}")
                self.trainable_modules = []

        self.experiment_name = f'{experiment_name}-{date_time_str}-{random.randint(0, 10000)}'
        save_folder = re.sub(r'[<>:"/\\|?*\[\]]', '_', self.experiment_name)
        if config.save_location is None:
            self.save_dir = f"./saved_models/{save_folder}/"
        else:
            self.save_dir = f"{config.save_location}/{save_folder}/"

        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir, exist_ok=True)

        model_path = os.path.join(self.save_dir, f"model_epoch_{-1}.pt")
        print(f"Saving model to: {model_path}")
        safetensors.torch.save_file(model.state_dict(), model_path)

        self.run = None
        self.best_metrics = {}
        if wandb_enabled:
            # start run
            config = {
                "LEARNING_RATE": config.learning_rate,
                "BATCH_SIZE": config.batch_size,
                "MODEL": config.model,
                "RANK": config.target_rank,
                "DATASET": config.dataset,
                "EPOCHS": config.epoch,
                "PARAMETER_COUNT": sum(p.numel() for p in model.parameters()),
            }
            import torch.distributed as dist

            if self.config.DATA_PARALLEL and dist.is_initialized():
                if dist.get_rank() == 0:
                    run = wandb.init(
                        project=PROJECT_NAME,
                        config=config,
                        name=f"{self.experiment_name}",
                        group="DDP",
                        settings=wandb.Settings(x_label="rank_0", mode="shared", x_primary=True)
                    )
                    self.run = run
                else:
                    run = wandb.init(
                        project=PROJECT_NAME,
                        config=config,
                        name=f"{self.experiment_name}",
                        group="DDP",
                        settings=wandb.Settings(x_label=f"rank_{dist.get_rank()}", mode="disabled")
                    )
                    self.run = run
            else:
                self.run = wandb.init(
                    project=PROJECT_NAME,
                    config=config,
                    name=f"{self.experiment_name}",
                )

        device = next(model.parameters()).device
        print(f"Model is on device: {device}")
        num_params = sum(p.numel() for p in model.parameters())
        print(f"Number of parameters: {num_params}")
        print(model)

        if regulariser is not None:
            if hasattr(regulariser, "trainable_weights"):
                for layer in regulariser.trainable_weights:
                    try:
                        print(f"Linear Layer, ({layer.weight.shape}): {layer.weight.device}")
                    except:
                        print(f"Weight Tensor, ({layer.shape}): {layer.device}")

    def start_epoch(self):
        TIMER.start(name=str(self.epoch))
        print(f"Starting Epoch {self.epoch} / {self.config.epoch}@ {TIMER.get(name=str(self.epoch))}")

    def end_epoch(self):
        TIMER.lap(name=str(self.epoch))
        print(f"Ending Epoch {self.epoch} / {self.config.epoch}")
        self.epoch += 1

    def regularise_log(self):
        self.iterations += 1

    def epoch_summary(self, results: dict, model, config, verbose=False):
        

        if self.run and isinstance(self.regulariser, QuaRS):
            self.run.log({f"QuaRS/val": self.regulariser.val}, step=self.epoch)

        """if self.run and isinstance(self.optimizer, AdamQ3R):
            self.run.log({f"QuaRS/val": self.optimizer.q3r.val}, step=self.epoch)"""

        if self.save_all:
            model_path = os.path.join(self.save_dir, f"model_epoch_{self.epoch}.pt")
            print(f"Saving model to: {model_path}")
            safetensors.torch.save_file(model.state_dict(), model_path)

        is_best = False

        if self.run:
            log_obj = {}

            print(f"Epoch {self.epoch}: Starting to log metrics")

            for key, value in results.items():
                if 'loss' in key or 'acc' in key:
                    if 'test' in key:
                        loader_len = len(self.validation_loader.dataset)
                        print(f"Test metric: {key}, value: {value}, loader length: {loader_len}")
                    else:
                        loader_len = len(self.train_loader.dataset)
                        print(f"Train metric: {key}, value: {value}, loader length: {loader_len}")

                    avg_value = value / loader_len
                    log_obj[f"total_{key}"] = value
                    log_obj[f"avg_{key}"] = avg_value

                    # Check if this metric is better than the previous best
                    metric_key = f"avg_{key}"

                    # For accuracy metrics, higher is better
                    if 'acc' in key:
                        if metric_key not in self.best_metrics or avg_value > self.best_metrics[metric_key]:
                            self.best_metrics[metric_key] = avg_value
                            is_best = True
                            print(f"New best {metric_key}: {avg_value}")

            print(f"Logging metrics: {log_obj}")
            self.run.log(log_obj, step=self.epoch)

        if isinstance(self.trainable_modules, dict):
            weight_ptrs = extract_trainable_weights(self.trainable_modules)
        elif isinstance(self.trainable_modules, list):
            weight_ptrs = []
            for module in self.trainable_modules:
                weight_ptrs.extend(module.get_combined_weight_factors())
        else:
            weight_ptrs = self.trainable_modules

        for index, weight_ptr in enumerate(weight_ptrs):
            print(f"Processing weight matrix: {index}, shape: {weight_ptr.data.shape}")
            ########################################################
            # What the sigma
            weight_matrix = weight_ptr.data  # or any tensor
            # print(f"Running SVD for {index}")
            U, S, V = torch.svd(weight_matrix)
            # print(f"SVD complete. Singular values shape: {S.shape}, max: {S.max().item()}, min: {S.min().item()}")

            # Convert to DataFrame for wandb
            df = pd.DataFrame([(i, s.item()) for i, s in enumerate(S)], columns=["index", "singular_value"])

            # Create wandb Table and Line Plot
            table = wandb.Table(dataframe=df)
            plot = wandb.plot.line(table, "index", "singular_value", title=f"Singular Values at Epoch {self.epoch}")

            # Save plot locally
            plt.figure(figsize=(10, 6))
            plt.plot(df["index"], df["singular_value"])
            plt.title(f"Singular Values at Epoch {self.epoch} - Matrix {index}")
            plt.xlabel("Index")
            plt.ylabel("Singular Value")
            os.makedirs(f"{self.save_dir}/sigma_graph/{self.epoch}", exist_ok=True)
            plot_path = os.path.join(f"{self.save_dir}/sigma_graph/{self.epoch}", f"layer_{index}.png")
            plt.savefig(plot_path)
            plt.close()
            # print(f"Saved singular values plot to {plot_path}")

            #######################################################
            # Effective Rank
            if self.run:
                er = effective_rank(weight_ptr)
                self.run.log({f"effective_rank/{index}": er}, step=self.epoch)

            #######################################################
            # Tail Ratio
            if self.run:
                r = 10
                if isinstance(self.regulariser, QuaRS):
                    r = int(self.regulariser.regularizers[index].target_rank)
                    self.run.log({f"epsilon/{index}": self.regulariser.regularizers[index].epsilon}, step=self.epoch)

                tr = tail_ratio(weight_ptr, r)
                self.run.log({f"tail_ratio/{index}": tr}, step=self.epoch)

            # print(f"Completed processing for epoch {self.epoch}")
        if self.save_all or is_best:
            model_path = os.path.join(self.save_dir, f"model_epoch_{self.epoch}.pt")
            print(f"Saving model to: {model_path}")
            safetensors.torch.save_file(model.state_dict(), model_path)
            if isinstance(self.regulariser, QuaRS) and not isinstance(self.optimizer, AdamQ3R):
                self.regulariser.state_dict()
                torch.save(self.regulariser.state_dict(),
                           os.path.join(self.save_dir, f"quars_epoch_{self.epoch}.pt"))
                print(f"Saved QuaRS state to: {os.path.join(self.save_dir, f'quars_epoch_{self.epoch}.pt')}")
            if isinstance(self.optimizer, AdamQ3R):
                self.optimizer.q3r.state_dict()
                torch.save(self.optimizer.state_dict(),
                           os.path.join(self.save_dir, f"adamq3r_epoch_{self.epoch}.pt"))
                print(f"Saved AdamQ3R state to: {os.path.join(self.save_dir, f'adamq3r_epoch_{self.epoch}.pt')}")

                
            # Do a lot of truncactions and calculations :facepalm:
            """
                Todo
                    - Calculate LogDet
                    - print(f'Target Ranks:           {[wx.target_rank for wx in rank_regularizer.regularizers]}')
                    print(f'Epsilon Envelope Rank:  {[wx.epsilon_rank_envelope for wx in rank_regularizer.regularizers]}')
                    print(f'Smallest Sigma:         {[wx.smallest_computed_sigma for wx in rank_regularizer.regularizers]}')
                    print(f'Length of Values:       {[len(wx.S) for wx in rank_regularizer.regularizers]}')
                    print(f'Epsilon:                {[wx.epsilon for wx in rank_regularizer.regularizers]}')
                    print(f'Val:                    {[wx.val for wx in rank_regularizer.regularizers]}')
                - save_model(run=run, model=model, accuracy=avg_vacc, max_accuracy=max_valid_accuracy,
                            save_dir=save_dir, epoch=epoch,
                            )
            """

    def weight_rank_details(self,model):
        #calulcates the spectral mass at each index, using the tail ratio
        # weight_name: 
        # enumerated list: [fo
        extracted_layers = {}

        for name, module in model.named_modules():
            for named_module in self.config.target_modules:
                if isinstance(module, torch.nn.Linear):
                    extracted_layers.update({module: None})
                if hasattr(module, named_module):
                    attr = getattr(module, named_module)
                    # print(f"Module Name: {name} / {module}\nSearch Name: {named_module}\n attr:{attr}")

                    if isinstance(attr, torch.nn.Linear) or (isinstance(attr, torch.Tensor) and attr.dim() >= 2):
                        if hasattr(module, "qkv"):

                            qkv_weight = attr.weight
                            dim = qkv_weight.shape[0] // 3

                            attr.q_weight = qkv_weight[:dim]
                            attr.k_weight = qkv_weight[dim:2 * dim]
                            attr.v_weight = qkv_weight[2 * dim:]

                            list_of_weights = extract_trainable_weights({module.qkv: [(0, dim), (dim, 2 * dim), (2 * dim, -1), ]})
                            
                            for index, weight in enumerate(list_of_weights):
                                print(f"{['Q', 'K', 'V'][index]} Weight: {weight.shape}, Name:{name}") if index in [0,1,2] else None
                                s=""
                                for i in range(0,min(weight.shape),25):
                                    tail_ratio(weight, i)
                                    param_save_ratio = i *(min(weight.shape) + max(weight.shape)) / (min(weight.shape) * max(weight.shape))
                                    s += f": {param_save_ratio:.1f}\% {tail_ratio(weight, i):.4f} "
                                print(s)
                            
                        else:
                            weight_singleton = extract_trainable_weights({attr: None})
                            for index, weight in enumerate(weight_singleton):
                                print(f"MLP Weight: {weight.shape}, Name:{name}")
                                s=""
                                for i in range(0,min(weight.shape),25):
                                    tail_ratio(weight, i)
                                    param_save_ratio = i *(min(weight.shape) + max(weight.shape)) / (min(weight.shape) * max(weight.shape))
                                    s += f": {param_save_ratio:.1f}\% {tail_ratio(weight, i):.4f} "
                                print(s)
                    else:
                        try:
                            print(f"Module {named_module} is not a Linear Layer or Weight")
                            weight_singleton = extract_trainable_weights[{attr: None}]
                            for index, weight in enumerate(list_of_weights):
                                print(f"Weight: {weight.shape}, Name:{name}")
                                for i in range(0,min(weight.shape),25):
                                    tail_ratio(weight, i)
                                    param_save_ratio = i *(min(weight.shape) + max(weight.shape)) / (min(weight.shape) * max(weight.shape))
                                    s += f": {param_save_ratio:.1f}\% {tail_ratio(weight, i):.4f} "
                                print(s)
                        except:
                            print("meh")
                            print(f"Module {named_module} is not a Linear Layer or Weight")
                        print(type(attr))
                        print(f"Module {named_module} is not a Linear Layer or Weight")
                        raise (f"Module {named_module} is not a Linear Layer or Weight")
    
        
    def print_metric_extremes(self):
        history = self.run.history(keys=None)  # Load all keys

        metric_keys = [col for col in history.columns if not col.startswith('_')]

        print(f"\nMetric extremes for run {self.run.name}:")
        for key in metric_keys:
            series = history[key].dropna()
            if not series.empty:
                print(f"{key}: min = {series.min():.4f}, max = {series.max():.4f}")

    def print_summary(self):
        if self.run:  # inside your training class
            wandb.finish()
            #self.print_metric_extremes()
