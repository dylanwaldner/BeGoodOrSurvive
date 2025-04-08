import os
import json
import torch
import openai

openai.api_key = os.getenv("OPENAI_API_KEY")

# ✅ Function to serialize data for JSON compatibility
def serialize_data(obj):
    if isinstance(obj, torch.Tensor):
        return obj.tolist()  # Convert tensors to lists
    elif isinstance(obj, range):
        return list(obj)  # Convert range to a list
    elif isinstance(obj, dict):
        return {key: serialize_data(value) for key, value in obj.items()}  # Recursively process dicts
    elif isinstance(obj, list):
        return [serialize_data(item) for item in obj]  # Recursively process lists
    else:
        return obj  # Return as-is if JSON-compatible

# ✅ Function to modify the checkpoint data
def modify_checkpoint(file_path, new_global_counter):
    # 🔹 Load the checkpoint file
    data = torch.load(file_path, map_location=torch.device('cpu'))

    # 🔹 Modify `global_counter`
    if "global_counter" in data:
        print(f"Original global_counter: {data['global_counter']}")
    data["global_counter"] = new_global_counter  # Update global_counter

    # 🔹 Modify `bnn_history` by **removing entries with id == 309**
    if "bnn_history" in data and isinstance(data["bnn_history"], list):
        original_size = len(data["bnn_history"])
        data["bnn_history"] = [entry for entry in data["bnn_history"] if entry.get("id") != 309]
        new_size = len(data["bnn_history"])
        print(f"Removed {original_size - new_size} entries from `bnn_history` (id == 309)")

    # 🔹 Save the modified checkpoint back under the **same name**
    torch.save(data, file_path)
    print(f"✅ Checkpoint updated and saved as {file_path}")

    # 🔹 Save cleaned `bnn_history` as JSON for reference
    output_json_path = file_path.replace(".pth", "_bnn_history.json")
    with open(output_json_path, 'w') as json_file:
        json.dump(serialize_data(data["bnn_history"]), json_file, indent=4)
    print(f"✅ Cleaned `bnn_history` has been saved separately to {output_json_path}")

# ✅ Define new global counter value
new_global_counter = 499  # Example: Set to a new global count

# ✅ Path to the checkpoint file
file_path = "/scratch/10384/dylantw15/Bayesian-Neat-Project/checkpoint_2025-02-03_12-05-28.pth"

# ✅ Modify and Save
modify_checkpoint(file_path, new_global_counter)

