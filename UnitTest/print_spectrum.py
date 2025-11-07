import torch
from timm import create_model
import torch.nn as nn
from collections import OrderedDict
import os
from datetime import datetime

def load_and_print_spectrum(model_path: str, model_name: str, num_classes: int = 10):
    """
    Loads a PyTorch model, extracts q, k, and v weights, computes their spectrum,
    and prints them.

    Args:
        model_path (str): Absolute path to the saved PyTorch model state_dict.
        model_name (str): The `timm` model name (e.g., 'vit_base_patch16_224').
        num_classes (int): Number of output classes (default: 10).
    """
    # Create the model
    # Create the model
    model = create_model(model_name, pretrained=False, num_classes=num_classes)
    creation_time = os.path.getctime(model_path)

    # Convert to a human-readable format
    creation_time_human = datetime.fromtimestamp(creation_time)

    # Print the creation date
    print(f"File creation date: {creation_time_human}")
    # Load the state dictionary
    try:
        checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
        #print("Checkpoint keys:", checkpoint.keys())

        # Remove 'module.' prefix if present
        new_state_dict = OrderedDict()
        for k, v in checkpoint.items():
            new_key = k.replace("module.", "")  # Remove the 'module.' prefix
            new_state_dict[new_key] = v
        
        # Load the modified state_dict
        model.load_state_dict(new_state_dict)
        #print(f"State dictionary loaded successfully from: {model_path}")
    except Exception as e:
        #print(f"Error loading state dictionary: {e}")
        return

    model.eval()  # Set model to evaluation mode

    # Iterate over layers and process qkv weights
    for name, module in model.named_modules():
        if hasattr(module, 'qkv') and module.qkv is not None:
            try:
                # Extract the qkv weights
                qkv_weight = module.qkv.weight.detach()
                dim = qkv_weight.shape[0] // 3
                
                # Split into q, k, and v weights
                q_weight = qkv_weight[:dim]
                k_weight = qkv_weight[dim:2 * dim]
                v_weight = qkv_weight[2 * dim:]
                
                # Compute singular values (spectrums)
                q_spectrum = torch.linalg.svdvals(q_weight)
                k_spectrum = torch.linalg.svdvals(k_weight)
                v_spectrum = torch.linalg.svdvals(v_weight)

                # Print spectra
                print(f"Layer: {name}")
                q,k,v= " ".join([f"{s:.4f}" for s in q_spectrum.tolist()[:20]]), " ".join([f"{s:.4f}" for s in k_spectrum.tolist()[:20]])," ".join([f"{s:.4f}" for s in v_spectrum.tolist()[:20]])
                print(f"Q weight shape: {q_weight.shape}, Spectrum: {q}")
                print(f"K weight shape: {k_weight.shape}, Spectrum: {k}")
                print(f"V weight shape: {v_weight.shape}, Spectrum: {v}")
                print("-" * 40)
            except Exception as e:
                print(f"Error processing layer {name}: {e}")

# Example usage
# Specify the absolute path to your saved model
model_path = "/users/enguye17/iclr/saved_models/QuaRS_Rank_8__Lambda_1.5__models/QuaRS[Rank:8][Lambda:1.5]/model_epoch_95.pt"
load_and_print_spectrum(model_path,"timm/vit_base_patch16_224",10)