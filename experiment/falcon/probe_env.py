import sys, os
print("python", sys.executable, sys.version)
try:
    import torch
    print("torch", torch.__version__)
    print("cuda_available", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("cuda_devices", torch.cuda.device_count())
        for i in range(torch.cuda.device_count()):
            try:
                print(i, torch.cuda.get_device_name(i))
            except Exception as e:
                print("err", e)
    else:
        print("no cuda")
except Exception as e:
    print("torch import failed", e)

try:
    import falcon_perception
    print("falcon_perception imported", falcon_perception.__version__)
except Exception as e:
    print("falcon_perception import failed", e)

# check hf cache
from pathlib import Path
import os
hf_home = os.environ.get("HF_HOME", "")
print("HF_HOME", hf_home)
cache = Path.home() / ".cache" / "huggingface" / "hub"
print("cache exists", cache.exists())
if cache.exists():
    for p in cache.iterdir():
        if "Falcon" in p.name:
            print(p.name, list(p.iterdir())[:3])
