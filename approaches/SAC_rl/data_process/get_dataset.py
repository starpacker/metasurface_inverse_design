from FDTDModel import FDTDModel
import torch
import time
import numpy as np
import os
import random
# from ipdb import set_trace

def get_random_pra_optimized(unit=400):  # 用于三矩形生成
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

        return np.transpose(pattern)

    # done
    def get_optical_response(self):
        stru = self.get_pattern()
        fdtd_model = FDTDModel(self.pra)
        optical_response = fdtd_model.get_results()
        return optical_response

    def get_details_yjh(self, op_show=False):
        # Eij 表示 i偏振入射情况下的 j偏振的反射光
        op = self.get_optical_response()

        stru_lambda = op[:, 0]

        r_lr_real = op[:, 1]
        r_lr_imag = op[:, 2]
        r_rl_real = op[:, 3]
        r_rl_imag = op[:, 4]
        r_rr_real = op[:, 5]
        r_rr_imag = op[:, 6]


        return [stru_lambda, r_rl_real, r_rl_imag, r_lr_real,  r_lr_imag, 
                r_rr_real,  r_rr_imag]
    
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

def get_now_time():
    """获取当前时间（以可读格式返回）"""
    return time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())

def generate(data_num=2000):
    now_time = get_now_time()
    print(f"Start generating data at {now_time}")
    # Lists to hold the parameters and corresponding spectra
    pra_list = []
    spectrum_list = []

    # Generate the dataset
    start_time = time.time()
    for idx in range(data_num):
        # Generate random parameters (a list of 10 values)
        pra = get_para_random_pra_test(unit=400)

        parameter = new_pattern(pra)

        pra_idx = parameter.get_pattern()

        pra_index = torch.from_numpy(pra_idx)

        pra_list.append(pra_index)


        # Retrieve the corresponding data dictionary using the generated parameters
        data_spectrum = parameter.get_details_yjh()

        spectrum_list.append(data_spectrum)

        end_time = time.time()  # 记录循环结束时间
        elapsed_time = end_time - start_time  # 计算循环耗时
        print("parameter:",pra)
        print(f"{idx + 1} data completed. Time taken: {elapsed_time:.4f} seconds")

        if idx % 10 ==0 and idx>0:
            # Convert lists to tensors using torch.stack
            pra_tensor = torch.stack(pra_list, dim=0)
            spectrum_tensor = torch.tensor(np.stack(spectrum_list))

            # Save the list of parameters to a file named 'tensor_data_matrix{data_num}.pt'
            torch.save(pra_tensor, f'E:/yjh/tensor_data_matrix_{idx+1}_{now_time}.pt')
            # Save the list of spectra to a file named 'tensor_spectrum{data_num}.pt'
            torch.save(spectrum_tensor, f'E:/yjh/tensor_spectrum_{idx+1}_{now_time}.pt')

            print("successfully saved!")
        
        

    # Convert lists to tensors using torch.stack
    pra_tensor = torch.stack(pra_list, dim=0)
    spectrum_tensor = torch.tensor(np.stack(spectrum_list))

    # Save the list of parameters to a file named 'tensor_data_matrix{data_num}.pt'
    torch.save(pra_tensor, f'E:/yjh/tensor_data_matrix_{data_num}_{now_time}.pt')
    # Save the list of spectra to a file named 'tensor_spectrum{data_num}.pt'
    torch.save(spectrum_tensor, f'E:/yjh/tensor_spectrum_{data_num}_{now_time}.pt')

    # # Print a sample of the data for verification
    # print("Sample parameter (pra):")
    # print(pra_list[0])
    # print("\nSample spectrum:")
    # print(spectrum_list[0])

if __name__ == '__main__':
    generate(data_num=2000)
