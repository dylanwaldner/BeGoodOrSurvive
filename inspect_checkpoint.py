import torch
import os
from bnn.bayesnn import BayesianNN
import openai

openai.api_key = os.getenv("OPENAI_API_KEY")

# Load the checkpoint
checkpoint_path = "checkpoint_2025-02-04_01-59-22.pth"
checkpoint = torch.load(checkpoint_path, map_location="cpu")

# Print key metadata
print(f"Counter: {checkpoint.get('counter', 'Not found')}")
print(f"Global Counter: {checkpoint.get('global_counter', 'Not found')}")
print(f"Last Step: {checkpoint.get('last_step', 'Not found')}")
print(f"Danger Level: {checkpoint.get('danger', 'Not found')}")
print(f"Generation Loss History (last 5): {checkpoint.get('gen_loss_history', [])[-5:]}")
print(f"Generation Ethical History (last 5): {checkpoint.get('gen_ethical_history', [])[-5:]}")
print(f"len(Bnn_history): {checkpoint.get('bnn_history') and len(checkpoint['bnn_history']) or 'Not found'}")

# Check if NEAT trainer state exists
if checkpoint.get('neat_trainer_state') is not None:
    print("NEAT trainer state is present.")
else:
    print("No NEAT trainer state found.")

# Check if generational history exists
if checkpoint.get('generational_history'):
    print(f"Generational History Entries: {len(checkpoint['generational_history'])}")
else:
    print("No generational history found.")

# Check for overall summary
if 'overall_summary' in checkpoint:
    print("Overall summary is present.")
else:
    print("No overall summary found.")

print("MODEL LOADING---------------------------------")

import os
import torch

# Define the winner genome checkpoint path
winner_genome_checkpoint_path = "test3_prod_winner_genome_model_iteration_3.pth"

# Ensure the checkpoint is loaded correctly
if os.path.exists(winner_genome_checkpoint_path):
    print(f"Loading winner genome checkpoint from {winner_genome_checkpoint_path}...")
    winner_genome_checkpoint = torch.load(winner_genome_checkpoint_path, map_location=torch.device("cpu"))

    # Validate required keys exist in checkpoint
    required_keys = ['genome', 'attention_layers', 'config', 'model_state_dict']
    missing_keys = [key for key in required_keys if key not in winner_genome_checkpoint]

    if missing_keys:
        raise ValueError(f"Checkpoint is missing required keys: {missing_keys}")

    # Extract components
    winner_genome = winner_genome_checkpoint['genome']
    attention_layers = winner_genome_checkpoint['attention_layers']
    config = winner_genome_checkpoint['config']
    model_state_dict = winner_genome_checkpoint['model_state_dict']

    # Validate model state dictionary
    if not isinstance(model_state_dict, dict):
        raise ValueError("Invalid model_state_dict format in checkpoint.")

    # Reconstruct the BNN using the winner genome
    strong_bnn = BayesianNN(winner_genome, config, attention_layers=attention_layers)
    
    num_params = sum(p.numel() for p in strong_bnn.parameters())
    print(f"Total parameters in BNN: {num_params}")


    # Load state dict and check for missing or unexpected keys
    missing_keys, unexpected_keys = strong_bnn.load_state_dict(model_state_dict, strict=False)

    if missing_keys:
        print(f"Warning: The following keys are missing in the loaded state dict: {missing_keys}")
    if unexpected_keys:
        print(f"Warning: The following unexpected keys were found in the state dict: {unexpected_keys}")

    print("Winner genome and BNN successfully loaded.")

    # Verify model's device and tensor shapes
    for name, param in strong_bnn.named_parameters():
        print(f"Parameter: {name}, Shape: {param.shape}, Device: {param.device}")


else:
    print(f"Winner genome checkpoint not found at {winner_genome_checkpoint_path}. Using previous model state.")

    # Ensure the main checkpoint exists before restoring
    if 'model_state_dict' not in checkpoint:
        raise ValueError("No valid 'model_state_dict' found in the previous checkpoint!")

    # Load previous model state
    strong_bnn.load_state_dict(checkpoint['model_state_dict'])

