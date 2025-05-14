import torch

checkpoint_path = "ctc_model_checkpoint.pth"
checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))

print("Checkpoint keys:", checkpoint.keys())
