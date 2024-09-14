uv venv --python 3.12.5
uv add ruff
uv add tensorboard
uv add tensorboardX
uv pip install -r requirements.txt
uv pip install -U torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
uv pip install -U segmentation_models_pytorch
uv pip install -U timm
