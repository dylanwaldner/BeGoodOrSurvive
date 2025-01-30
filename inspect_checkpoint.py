import torch
import os
import openai

openai.api_key = os.getenv("OPENAI_API_KEY")

# Load the checkpoint
checkpoint_path = "checkpoint.pth"
checkpoint = torch.load(checkpoint_path, map_location="cpu")

# Print key metadata
print(f"Counter: {checkpoint.get('counter', 'Not found')}")
print(f"Global Counter: {checkpoint.get('global_counter', 'Not found')}")
print(f"Last Step: {checkpoint.get('last_step', 'Not found')}")
print(f"Danger Level: {checkpoint.get('danger', 'Not found')}")
print(f"Generation Loss History (last 5): {checkpoint.get('gen_loss_history', [])[-5:]}")
print(f"Generation Ethical History (last 5): {checkpoint.get('gen_ethical_history', [])[-5:]}")
print(f"Rounds Survived History (last 5): {checkpoint.get('rounds_survived_history', [])[-5:]}")

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

