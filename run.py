import torch
import os

# Set the fraction of GPU memory to be used by each process
torch.cuda.set_per_process_memory_fraction(0.5, 0)  # GPU 0
torch.cuda.set_per_process_memory_fraction(0.5, 1)  # GPU 1
torch.cuda.set_per_process_memory_fraction(0.5, 2)  # GPU 2
torch.cuda.set_per_process_memory_fraction(0.5, 3)  # GPU 3

# Run your server command here
os.system("CUDA_VISIBLE_DEVICES=0,1,2,3 python -m third_party.vllm.vllm.entrypoints.openai.api_server --host 0.0.0.0 --port 8000 --max-model-len 1024 --model models/mistral --tensor-parallel 4 --disable-log-requests --swap-space 16 >benchmarks/server_output.log")
