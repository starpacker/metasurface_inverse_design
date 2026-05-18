import torch
import torch.nn as nn
import numpy as np
from MetasurfaceCNN import Meta_CNN_Net
import random
from ipdb import set_trace


class EncoderDecoderTrainer:
    def __init__(self, device, name):
        self.device = device
        self.model_name = name
        self.forward_model = None

        if self.model_name == "CRNNAG":
            from MetasurfaceCNN import Meta_CRNNAG_Net
            self.forward_model = Meta_CRNNAG_Net().to(self.device)
            checkpoint = torch.load('/data/group_003/yjh/logs/CRNNAG_v0_8k/checkpoint_max.pth', map_location="cpu")
        elif self.model_name == "CRNNAG_attention":
            from new_agnet import Meta_CRNNAG_Net
            self.forward_model = Meta_CRNNAG_Net().to(self.device)
            checkpoint = torch.load('/data/group_003/yjh/logs/CRNNAG_attention_v0_8k/checkpoint_max.pth', map_location="cpu")
        elif self.model_name == "CNN":
            self.forward_model = Meta_CNN_Net().to(self.device)
            self.forward_model.CNNLayer[0].layer3[0].relu3 = nn.ELU()
            checkpoint = torch.load('/data/group_003/yjh/logs/CNN_20k/checkpoint_max.pth', map_location="cpu")
        else:
            raise ValueError("Invalid model name")

        self.forward_model.load_state_dict(checkpoint['net'])
        # self.forward_model.load_state_dict(checkpoint)
        self.forward_model.eval()

    def forward_step(self, patterns):
        if self.model_name == "CRNNAG" or self.model_name == "CRNNAG_attention":
            trg = torch.rand(patterns.shape[0], 6, 201).to(self.device)  # [B, 6, 201]
            with torch.no_grad():
                spectrum = self.forward_model(patterns.float().unsqueeze(1), trg, 0)
        elif self.model_name == "CNN":
            with torch.no_grad():
                spectrum = self.forward_model(patterns.float().unsqueeze(1))
        else:
            raise ValueError("Invalid model name")
        return spectrum


def save_dataset_test(patterns,name,device):
    path_spectra = f"/data/group_003/yjh/data_set_s/spectra_{name}.pt"

    models = ["CNN"]
    # models = ["CRNNAG"]
    # models = ["CRNNAG_attention"]
    models = ["CNN","CRNNAG","CRNNAG_attention"]
    for model_name in models:
        trainer = EncoderDecoderTrainer(device, model_name)
        spectrum = trainer.forward_step(patterns)
        print(f"Processed spectra by {model_name}")
        
        torch.save(spectrum.cpu(), path_spectra.replace("spectra", f"spectra_{model_name}"))


if __name__ == "__main__":
    
    # Save the dataset
    device = "cuda:3"
    pat_list = ["patterns_synthesis_3000.pt","patterns_synthesis_6000.pt","patterns_synthesis_5000.pt","patterns_synthesis_5000_2.pt"]
    for pat in pat_list:
        patterns = torch.load(f"/data/group_003/yjh/data_set_s/{pat}")
        save_dataset_test(patterns.to(device), pat[9:23] ,device)
        print(f"Dataset {pat} saved successfully")
