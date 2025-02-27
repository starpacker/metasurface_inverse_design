import torch
import torch.nn as nn
import numpy as np
from MetasurfaceCNN import Meta_CNN_Net, Meta_CRNNAG_Net
import random
from ipdb import set_trace

class PatternRects:
    def __init__(self, pra):
        self.pra = pra
        self.pra1 = pra[0]
        self.pra2 = pra[1]
        self.pra3 = pra[2]
        self.pra4 = pra[3]
        self.pra5 = pra[4]
        self.pra6 = pra[5]
        self.pra7 = pra[6]
        self.pra8 = pra[7]
        self.pra9 = pra[8]
        self.pra10 = pra[9]

    def get_pattern(self):
        pattern = np.zeros((400, 400), dtype=int)
        for xi in range(pattern.shape[0]):
            for yj in range(pattern.shape[1]):
                if xi >= self.pra1 and xi <= self.pra1 + self.pra3:
                    if yj >= self.pra10 + self.pra9 + self.pra6 + self.pra5 and \
                            yj <= self.pra10 + self.pra9 + self.pra6 + self.pra5 + self.pra2:
                        pattern[xi, yj] = 1
                if xi >= self.pra1 and xi <= self.pra1 + self.pra4:
                    if yj >= self.pra10 + self.pra9 + self.pra6 and \
                            yj <= self.pra10 + self.pra9 + self.pra6 + self.pra5:
                        pattern[xi, yj] = 1
                if xi >= self.pra7 and xi <= self.pra7 + self.pra8:
                    if yj >= self.pra10 and yj <= self.pra10 + self.pra9:
                        pattern[xi, yj] = 1
        return torch.from_numpy(np.transpose(pattern))

def get_random_pra(unit=400):
    pra1 = random.randrange(30, 160, 1)
    pra3 = random.randrange(40, 80, 1)
    pra4 = random.randrange(80, unit - 30 - pra1, 1)

    pra7 = random.randrange(30, 150, 1)
    pra8 = random.randrange(40, unit - 30 - pra7, 1)

    while True:
        pra10 = random.randrange(30, 100, 1)
        pra9 = random.randrange(40, 100, 1)
        pra6 = random.randrange(40, 100, 1)
        pra5 = random.randrange(40, 100, 1)
        pra2 = random.randrange(40, 100, 1)
        if pra10 + pra9 + pra6 + pra5 + pra2 <= unit:
            return [pra1, pra2, pra3, pra4, pra5, pra6, pra7, pra8, pra9, pra10]

def get_random_pra_optimized(unit=400):
    # Generate pra1, pra3, pra4, pra7, pra8
    pra1 = random.randrange(30, 160, 1)  # Range: 30-159
    pra3 = random.randrange(40, 80, 1)   # Range: 40-79
    pra4 = random.randrange(80, unit - 30 - pra1, 1)  # Range: 80- (unit - 30 - pra1 - 1)
    pra7 = random.randrange(30, 150, 1)  # Range: 30-149
    pra8 = random.randrange(40, unit - 30 - pra7, 1)  # Range: 40- (unit - 30 - pra7 -1)
    
    # Generate pra2, pra5, pra6, pra9, pra10 using a deterministic approach to avoid retries
    # The sum of these values must be <= unit
    remaining = unit
    # Generate pra2 with a range [40, min(100, remaining - 4*40)]
    pra2 = random.randrange(40, min(100, remaining - 3*40 - 30 + 1), 1)
    remaining -= pra2
    
    # Generate pra5 with remaining >=40*3
    pra5 = random.randrange(40, min(100, remaining - 2*40 -30 + 1), 1)
    remaining -= pra5
    
    # Generate pra6 with remaining >=40*2
    pra6 = random.randrange(40, min(100, remaining - 40 -30 + 1), 1)
    remaining -= pra6
    
    # Generate pra9 with remaining >=40
    pra9 = random.randrange(40, min(100, remaining - 30 + 1), 1)
    remaining -= pra9
    
    # Remaining becomes pra10's value
    pra10 = remaining
    
    # Ensure all values are within their ranges
    # Since we generate them in a controlled way, this should be guaranteed
    if pra10 < 30:
        print("what the shit")
    return [pra1, pra2, pra3, pra4, pra5, pra6, pra7, pra8, pra9, pra10]

class new_pattern:
    def __init__(self, pra):
        # 参数初始化
        self.pra = pra
        self.pra1 = pra[0]
        self.pra2 = pra[1]
        self.pra3 = pra[2]
        self.pra4 = pra[3]
        self.pra5 = pra[4]
        self.pra6 = pra[5]
        self.pra7 = pra[6]
        self.pra8 = pra[7]

    # done
    def get_pattern(self):
        pattern = np.zeros((400, 400), dtype=int)
        for xi in range(pattern.shape[0]):
            for yj in range(pattern.shape[1]):

                if xi >= self.pra1 and xi <= self.pra1 + self.pra2:
                    if yj >= self.pra4 + self.pra7 + self.pra8 and \
                            yj <= self.pra4 + self.pra7 + self.pra8 + self.pra3:
                        pattern[xi, yj] = 1

                if xi >= self.pra5 and xi <= self.pra5 + self.pra6:
                    if yj >= self.pra7 and yj <= self.pra7 + self.pra8:
                        pattern[xi, yj] = 1

        return torch.from_numpy(np.transpose(pattern))
  
def get_para_random_pra_test(unit=400):
    
    max_unit = unit - 10
    min_unit = 20

    pra5 = random.randint(min_unit, max_unit - 40)
    pra7 = random.randint(min_unit, max_unit - 200)

    pra6 = random.randint(min_unit, max_unit - pra5)
    pra8 = random.randint(min_unit, max_unit - pra7 - 180)

    pra4 = random.randint(min_unit, max_unit - pra7 -pra8 - 80)

    pra1 = random.randint(min_unit, max_unit - 40)
    pra2 = random.randint(min_unit, max_unit - pra1)
    pra3 = random.randint(min_unit, max_unit - pra4 - pra7 -pra8)

    return [pra1, pra2, pra3, pra4, pra5, pra6, pra7, pra8]

class EncoderDecoderTrainer:
    def __init__(self, device):
        self.device = device

        # Initialize forward and backward models
        self.forward_model = Meta_CRNNAG_Net().to(self.device)
        self.forward_model.CNNLayer[0].layer3[0].relu3 = nn.ELU()
        checkpoint = torch.load('/data/group_003/yjh/logs/b512_adam_lr0.001_c1281WCRNNAG200epoch_v0.3/checkpoint_max.pth', map_location="cpu")
        self.forward_model.load_state_dict(checkpoint['net'])
        self.forward_model.eval()

    def forward_step(self, num):
        patterns = []
        for idx in range(num):
            # raw_data = get_random_pra_optimized()
            # pattern = PatternRects(raw_data).get_pattern()
            raw_data = get_para_random_pra_test()
            pattern = new_pattern(raw_data).get_pattern()
            patterns.append(pattern)
            if idx % 1000 ==0 and idx >0:
                print(f"generate {idx} samples.......")


        patterns = torch.stack(patterns).to(self.device)  # [B, 400, 400]
        trg = torch.rand(num, 6, 201).to(self.device)  # [B, 6, 201]
        with torch.no_grad():
            spectrum = self.forward_model(patterns.float().unsqueeze(1),trg,0)
        return patterns,spectrum

if __name__=="__main__":
    gpu_num = 0
    device = f"cuda:{gpu_num}"
    train = EncoderDecoderTrainer(device)
    pattern,specturm = train.forward_step(num=3000)
    print(pattern.shape)
    print(specturm.shape)
    torch.save(pattern,"sythesis_patterns_3000.pt")
    torch.save(specturm,"sythesis_spectra_3000.pt")

    
