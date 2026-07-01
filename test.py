from transformers import AutoModel
import torch

device = (
    "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

print(device)

model = AutoModel.from_pretrained(
    "ai4bharat/IndicF5",
    trust_remote_code=True,
)

model = model.to(device)

print("Loaded!")