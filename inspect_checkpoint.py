"""
inspect_checkpoint.py
-------------------------------------------------
Utility to sanity‑check the main training checkpoint
and the final winner‑genome checkpoint.
"""

import os, torch, types, sys
from pprint import pprint

# -------------------------------------------------------------------------
# 0) Patch utils.text_generation to avoid OpenAI import side‑effects
# -------------------------------------------------------------------------
sys.modules["utils.text_generation"] = types.ModuleType("utils.text_generation")
sys.modules["utils.text_generation"].generate_text = lambda *a, **k: "dummy"

# -------------------------------------------------------------------------
# 1) Load main checkpoint ---------------------------------------------------
# -------------------------------------------------------------------------
CKPT_PATH = "checkpoint.pth"
ckpt = torch.load(CKPT_PATH, map_location="cpu")

print(f"\n🗂  Loaded {CKPT_PATH}")
print(ckpt.keys())
print(ckpt.get("model_state_dict").keys())
print("Overall Summary Keys: ", ckpt.get("overall_summary").keys())
print("Overall summary ground truth length: ", len(ckpt.get("overall_summary")["ground_truth_labels"]))
print("Overall summary ground truth first five: ", ckpt.get("overall_summary")["ground_truth_labels"][:5])
print("Overall summary ethical ground truth first five: ", ckpt.get("overall_summary")["ethical_ground_truths"][:5])
print("ckpt groun truth labels length: ", len(ckpt.get('ground_truth_label_list')))

# helper to report size/len or 'missing'
def report(key, obj):
    if obj is None:
        print(f"  {key:30s}: ✖  (missing)")
    elif isinstance(obj, (list, tuple, dict)):
        print(f"  {key:30s}: ✔  len = {len(obj)}")
    else:
        print(f"  {key:30s}: ✔  type = {type(obj)}")

to_check = [
    ("counter",                          ckpt.get("counter")),
    ("global_counter",                   ckpt.get("global_counter")),
    ("danger",                           ckpt.get("danger")),
    ("bnn_history",                      ckpt.get("bnn_history")),
    ("ground_truth_label_list",          ckpt.get("ground_truth_label_list")),
    ("ethical_ground_truths",            ckpt.get("ethical_ground_truths")),
    ("attention_layers",                 ckpt.get("attention_layers")),
    ("genome (init)",                    ckpt.get("genome")),
    ("config (NEAT)",                    ckpt.get("config")),
    ("neat_trainer_state",               ckpt.get("neat_trainer_state")),
    ("generational_history",             ckpt.get("generational_history")),
    ("overall_summary",                  ckpt.get("overall_summary")),
]

print("\n📋  Checkpoint contents:")
for key, obj in to_check:
    report(key, obj)

gen_history = ckpt.get("generational_history")
#print(f"gen history: {gen_history}")
ground_truths = ckpt.get("ground_truth_label_list")
#print(f"ground truths: {ground_truths}")
overall_summary = ckpt.get("overall_summary")
bnn_history_print = ckpt["bnn_history"][:5]
for i in bnn_history_print:
    print(i['id'])
#print(f"overall summary keys: {overall_summary.keys()}")
# -------------------------------------------------------------------------
# 2) Optional: rebuild BNN from winner‑genome checkpoint --------------------
# -------------------------------------------------------------------------
WINNER_PTH = "53_prod_winner_genome_model_iteration_3.pth"
if os.path.exists(WINNER_PTH):
    print(f"\n🔧  Loading winner genome checkpoint: {WINNER_PTH}")
    wckpt = torch.load(WINNER_PTH, map_location="cpu")

    # Ensure required keys
    required = ["genome", "attention_layers", "config", "model_state_dict"]
    missing  = [k for k in required if k not in wckpt]
    if missing:
        raise ValueError(f"Winner checkpoint missing keys: {missing}")

    from bnn.bayesnn import BayesianNN
    genome           = wckpt["genome"]
    attention_layers = wckpt["attention_layers"]
    print(f"model attention layers: {attention_layers.keys()}")
    neat_config      = wckpt["config"]
    print(f"neat_config: {neat_config}")
    state_dict       = ckpt["model_state_dict"]
    print(f"state dict: {state_dict.keys()}")

    # Rebuild BNN
    bnn = BayesianNN(genome, neat_config)
    miss, extra = bnn.load_state_dict(state_dict, strict=False)
    if miss:
        print(f"  ❕ Missing keys:   {miss}")
    if extra:
        print(f"  ❕ Unexpected keys:{extra}")

    print("✅ Query weight loaded correctly:", torch.allclose(bnn.query_proj.weight, state_dict['query_proj.weight']))

    print(f"  ✔  BNN rebuilt: parameters = {sum(p.numel() for p in bnn.parameters())}")

else:
    print(f"\n⚠️  Winner genome file not found: {WINNER_PTH}")

