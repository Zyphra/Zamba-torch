# Zamba Inference in PyTorch

Note: Under construction! Use as a reference, and load the model using our [HuggingFace Transformers fork](https://github.com/huggingface/transformers/pull/30950).

Pure-torch inference code for the Zamba-7B model (https://huggingface.co/Zyphra)

# Installation

`pip3 install torch packaging`

`pip3 install -e .`

# Forward pass
```
from mamba_model import MambaModel
from mamba_config import MambaConfig
import torch

config = MambaConfig(
    num_layers = 76,
    hidden_size = 3712,
    state_size = 16,
    conv_dimension = 4,
    expansion_factor = 2,
    rms_norm = True,
    bias = False,
    use_mem_mlp = True,
    num_attention_heads = 16,
    vocab_size = 50000,
    layer_mapping = str(["r", "r", "g", "r", "r", "r", "r", "g", "r", "r", "r", "r", "r", "g", "r", "r", "r", "r", "r", "g", "r", "r", "r", "r", "r", "g", "r", "r", "r", "r", "r", "g", "r", "r", "r", "r", "r", "g", "r", "r", "r", "r", "r", "g", "r", "r", "r", "r", "r", "g", "r", "r", "r", "r", "r", "g", "r", "r", "r", "r", "r", "g", "r", "r", "r", "r", "r", "g", "r", "r", "r", "r", "r", "g", "r", "r"])
)
model = MambaModel(config = config, max_sequence_length = 4096)
model = model.cuda().half()
inputs = torch.tensor([1, 2]).cuda().long().unsqueeze(0)
out = model(inputs)
```