---
license: mit
library_name: pytorch
pipeline_tag: robotics
tags: [robotics, dexterous-manipulation, grasp-generation, flow-matching, se3-equivariance, allegro-hand]
---
# EquiDexFlow &mdash; Allegro Hand (ContactOnly Ablation Variant)
SE(3)-equivariant flow-matching model for dexterous grasp generation. The checkpoint and config file in this repo are for the `ContactOnly` ablation variant of EquiDexFlow.

Load the checkpoint with `hf_hub_download` and `torch.load` (see snippet below).

## Usage

```python
from huggingface_hub import hf_hub_download
import torch
from equidexflow.models import get_dex_model

model = get_dex_model(hand="allegro")
path = hf_hub_download("cenwerem/equidexflow-allegro-contact-only", "checkpoint_best.pt")
ckpt = torch.load(path, map_location="cpu", weights_only=False)
model.load_state_dict(ckpt["model"])
# ckpt also carries: "epoch", "optimizer", "best_val_loss"
```