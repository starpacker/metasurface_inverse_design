from FDTDModel import get_para_random_pra_test,PatternRects
import torch
import time

def get_now_time():
    """获取当前时间（以可读格式返回）"""
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())


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

        parameter = PatternRects(pra)

        pra_idx = parameter.get_pattern()

        pra_list.append(pra_idx)


        # Retrieve the corresponding data dictionary using the generated parameters
        data_dict = parameter.get_details()

        # Construct the spectrum dictionary with selected fields
        data_spectrum = {
            'wavelength': data_dict['wavelength'],
            'r_lr_real': data_dict['r_lr_real'], 
            'r_lr_imag': data_dict['r_lr_imag'],
            'r_rl_real': data_dict['r_rl_real'], 
            'r_rl_imag': data_dict['r_rl_imag'],
            'r_rr_real': data_dict['r_rr_real'], 
            'r_rr_imag': data_dict['r_rr_imag']
        }
        spectrum_list.append(data_spectrum)

        print(f'{idx} data completed')
        end_time = time.time()  # 记录循环结束时间
        elapsed_time = end_time - start_time  # 计算循环耗时
        print(f"{idx + 1} data completed. Time taken: {elapsed_time:.4f} seconds")

        time.sleep(5)


    # Save the list of parameters to a file named 'tensor_data_matrix{data_num}.pt'
    torch.save(pra_list, f'D:/subject/physics/AI4S/project1/tensor_data_2000/tensor_data_matrix{data_num}_{now_time}.pt')
    # Save the list of spectra to a file named 'tensor_spectrum{data_num}.pt'
    torch.save(spectrum_list, f'D:/subject/physics/AI4S/project1/tensor_data_2000/tensor_spectrum{data_num}_{now_time}.pt')

    # # Print a sample of the data for verification
    # print("Sample parameter (pra):")
    # print(pra_list[0])
    # print("\nSample spectrum:")
    # print(spectrum_list[0])

if __name__ == '__main__':
    generate(data_num=2)
