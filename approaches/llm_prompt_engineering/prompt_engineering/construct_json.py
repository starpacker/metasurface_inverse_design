import json
import numpy as np
import torch

wave_length = 650
initial_prompt = f"""Analyze a metal-insulator-metal (MIM) metasurface structure and predict its optical response using a dataset of similar structures. The target structure consists of:
1. A 40 nm thick continuous gold (Au) substrate.
2. A 100 nm thick SiO2 spacer layer.
3. A top layer with 40 nm tall Au nanostructures shaped as two rectangular blocks.
4. Unit cell period: 400 nm * 400 nm.

**Dataset Description**:
- The dataset contains optical response data for a variety of MIM metasurfaces with two rectangular Au nanostructures, including:
  - Varying dimensions (length, width) of the rectangular blocks.
  - Different positions of the rectangular blocks within the unit cell.
  - Measurements of circular polarization reflection coefficients at a wavelength of {wave_length} nm.
- The dataset is structured as follows:
  - **Input Structure Representation**: [a, b, c, d, e, f, g, h]
    - a: x-coordinate of the top-left corner of the first rectangular block.
    - b: y-coordinate of the top-left corner of the first rectangular block.
    - c: Length of the first rectangular block.
    - d: Width of the first rectangular block.
    - e: x-coordinate of the top-left corner of the second rectangular block.
    - f: y-coordinate of the top-left corner of the second rectangular block.
    - g: Length of the second rectangular block.
    - h: Width of the second rectangular block.
- **Output Spectrum Specifications**:
  - The optical response is defined by circular polarization reflection coefficients, split into real and imaginary components. The coefficients correspond to the following polarization conversions:
    - r_rl: Right-handed circular polarization (RCP) → Left-handed circular polarization (LCP).
    - r_lr: Left-handed circular polarization (LCP) → Right-handed circular polarization (RCP).
    - r_rr: Right-handed circular polarization (RCP) → Right-handed circular polarization (RCP).
  - The output format for the spectrum response at {wave_length} nm follows the following structure:
    {{
        "spectrum_response": {{
        "r_rl_real": <value>, "r_rl_imag": <value>,
        "r_lr_real": <value>, "r_lr_imag": <value>,
        "r_rr_real": <value>, "r_rr_imag": <value>
        }}
    }}

**Task Requirements**:
1. Carefully examine the dataset to identify patterns and relationships between the structural parameters [a, b, c, d, e, f, g, h] and the optical responses at {wave_length} nm.
2. Analyze the following key aspects to guide your prediction:
   - **Symmetry Effects**: Investigate how the positions (a, b, e, f) and dimensions (c, d, g, h) of the two blocks influence the symmetry of the structure. Note that symmetry tends to make r_rl and r_lr similar, while asymmetry (due to block positions) introduces small differences.
   - **Coupling Effects**: Consider how the separation between the two blocks (e.g., distance between (a, b) and (e, f)) affects near-field coupling, which may alter the magnitude and phase of the reflection coefficients.
   - **Size Effects**: Explore how the dimensions of the blocks (c, d, g, h) influence the resonant behavior and optical response at {wave_length} nm.
   - **Sensitivity to Small Changes**: Be aware that small variations in structural parameters might lead to significant changes in the response, particularly near exceptional points (EPs).
3. Look for correlations between the structural parameters and the sign, magnitude, and phase of the reflection coefficients (r_rl, r_lr, r_rr).
4. Predict the polarization-resolved reflection spectra for the target structure at a wavelength of {wave_length} nm, providing the real and imaginary parts for r_rl, r_lr, and r_rr in the specified output format.
5. Ensure the prediction accounts for both near-field coupling between the blocks and far-field interference effects within the periodic unit cell.

**Notes**:
- The dataset ensures that the two rectangular blocks do not overlap.
- Due to the breaking of symmetry by the positions of the blocks, expect slight differences between r_rl and r_lr despite their tendency to be similar.
- Pay close attention to the sign (positive or negative) and scale (magnitude) of both the real and imaginary parts of the coefficients, as these are sensitive to structural variations.
"""

def transform_matrix(matrix):
    sum_array = matrix.sum(dim=1)
    flag = True
    for i in range(400):
        if sum_array[i] != 0 and flag:
            b = i
            flag = False
            continue
        if sum_array[i] == 0 and not flag:
            d = i - b
            break
    flag = True
    for i in range(b+d,400):
        if sum_array[i] != 0 and flag:
            f = i
            flag = False
            continue
        if sum_array[i] == 0 and not flag:
            h = i - f
            break
    flag = True
    for i in range(400):
        if matrix[b][i] != 0 and flag:
            a = i
            flag = False
            continue
        if matrix[b][i] == 0 and not flag:
            c = i - a
            flag = True
            break
    flag = True
    for i in range(400):
        if matrix[f][i] != 0 and flag:
            e = i
            flag = False
            continue
        if matrix[f][i] == 0 and not flag:
            g = i - e
            break
    return [a,b,c,d,e,f,g,h]

num_samples = 120
# pattern = torch.load('/data/group_003/yjh/tensor_data_matrix_100.pt')
# spectrum = torch.load('/data/group_003/yjh/tensor_spectrum_100.pt')
pattern = torch.load('/data/group_003/yjh/combined_pattern.pt')
spectrum = torch.load('/data/group_003/yjh/combined_spectrum.pt')
idx = (wave_length-600) * 2
spectrum = spectrum[:,:,idx]
spectrum_list = spectrum.tolist()
# pattern_list = pattern.tolist()

# 构造带物理注释的数据集
meta_surface_data = []
for i in range(num_samples):
    sample = {
        "index": i,
        "structure": transform_matrix(pattern[i]),
        "spectrum_response": {
        "r_rl_real": round(spectrum_list[i][1],5), "r_rl_imag": round(spectrum_list[i][2],5),
        "r_lr_real": round(spectrum_list[i][3],5), "r_lr_imag": round(spectrum_list[i][4],5),
        "r_rr_real": round(spectrum_list[i][5],5), "r_rr_imag": round(spectrum_list[i][6],5)
        }
        }
    
    meta_surface_data.append(sample)
    

# 构造完整JSON结构
json_data = {
    "physical_background": initial_prompt.strip(),
    "dataset_description": {
        "sample_count": num_samples,
        "data_types": {
            "structure": "parameter array",
            "spectrum": "complex reflectance values"
        },
        "coordinate_system": "Cartesian (x-right, y-down),top-left corner as origin"
    },
    "samples": meta_surface_data
}

# 保存JSON文件
with open(f"meta_surface_dataset_{wave_length}.json", "w") as json_file:
    json.dump(json_data, json_file, indent=None)

print("JSON successfully generated with enhanced physical descriptions.")



"""
请你基于你的知识库和我给出的数据集,分析给定的纳米结构图案,并预测其在620nm光照射下的光学响应:r_rl,r_lr,r_rr;
请着重注意两个矩形之间的位置关系和他们的距离，这会对光学响应产生影响。
[315, 117, 34, 22, 254, 309, 50, 70]
"""

"""Exceptional points (EPs) are critical in non-Hermitian optics, where the coalescence of eigenvalues and eigenvectors leads to unique optical responses.
Near EPs, the parameters r_rl, r_lr, and r_rr exhibit anomalous behavior. 
Specifically, r_rr (co-polarized reflection) often shows a sharp drop or abrupt phase transition, while r_rl and r_lr (cross-polarized reflections) may experience enhanced magnitudes and sign changes.
These phenomena arise due to the non-reciprocal coupling and topological dynamics near EPs. 
For instance, in MIM metasurfaces with Au nanostructures, the interaction between closely spaced or coupled resonators can lead to EPs, resulting in unconventional optical responses."""



# old version
"""
    Analyze a metal-insulator-metal (MIM) metasurface structure and predict its optical response using a dataset of similar structures. The target structure consists of:
    1. A 40 nm thick continuous gold (Au) substrate.
    2. A 100 nm thick SiO2 spacer layer.
    3. A top layer with 40 nm tall Au nanostructures shaped as two rectangular blocks.
    4. Unit cell period: 400 nm * 400 nm.

    **Dataset Description**:
    - The dataset contains optical response data for a variety of MIM metasurfaces with two rectangular Au nanostructures, including:
        - Varying dimensions (length, width) of the rectangular blocks.
        - Different positions of the rectangular blocks within the unit cell.
        - Measurements of circular polarization reflection coefficients at wavelength of {wave_length}nm.
    - The dataset is structured as follows:
        - **Input Structure Representation**: [a, b, c, d, e, f, g, h]
            - a: x-coordinate of the top-left corner of the first rectangular block.
            - b: y-coordinate of the top-left corner of the first rectangular block.
            - c: Length of the first rectangular block.
            - d: Width of the first rectangular block.
            - e: x-coordinate of the top-left corner of the second rectangular block.
            - f: y-coordinate of the top-left corner of the second rectangular block.
            - g: Length of the second rectangular block.
            - h: Width of the second rectangular block.
    **Output Spectrum Specifications**:
    - The optical response is defined by circular polarization reflection coefficients, split into real and imaginary components. The coefficients correspond to the following polarization conversions:
        - r_rl: Right-handed circular polarization (RCP) → Left-handed circular polarization (LCP).
        - r_lr: Left-handed circular polarization (LCP) → Right-handed circular polarization (RCP).
        - r_rr: Right-handed circular polarization (RCP) → Right-handed circular polarization (RCP).
    - The output format for the spectrum response at {wave_length}nm follows the following structure:
        {{
            "wavelength": "{wave_length}nm",
            "spectrum_response": {{
                "r_rl": {{"real": <value>, "imag": <value>}},
                "r_lr": {{"real": <value>, "imag": <value>}},
                "r_rr": {{"real": <value>, "imag": <value>}}
            }}
        }}
        
    **Task Requirements**:
    1. Carefully scan through the dataset and try to discover some universe rules and relationship between structures and responses.
    2. Think thoroughly and predict precisely the polarization-resolved reflection spectra for the target structure at a wavelength of {wave_length} nm.
    3. Consider near-field coupling and far-field interference effects
    4. Assume normal incidence illumination with plane waves.
    5. Output the results in the following format: 
        {{
            "wavelength": "{wave_length}nm",
            "r_rl": {{"real": 0.1, "imag": -0.1}},
            "r_lr": {{"real": 0.1, "imag": -0.09}},
            "r_rr": {{"real": 0.9, "imag": 0.1}}
        }}
    6.The data points are designed to ensure that the two rectangular structures do not overlap. 
    **Note**: 
    The size and position of the two blocks will affect(actually determine) the optical response.
    Small changes in structure may lead to significant changes in optical response, especially near exceptional points (EPs).
    When the structure is more symmetric, r_rr is expected to be larger, while r_rl and r_lr will decrease.
    When near-field coupling effect is significant, the amplitude of r_rl and r_lr may increase quickly, while r_rr often shows a sharp drop or abrupt phase transition.
    Due to symmetry, r_rl and r_lr are expected to be close to each other. But the position of the two blocks will break the symmetry, leading to small differences.
    """
# version 2
"""
    Analyze a metal-insulator-metal (MIM) metasurface structure and predict its optical response using a dataset of similar structures. The target structure consists of:
    1. A 40 nm thick continuous gold (Au) substrate.
    2. A 100 nm thick SiO2 spacer layer.
    3. A top layer with 40 nm tall Au nanostructures shaped as two rectangular blocks.
    4. Unit cell period: 400 nm * 400 nm.

    **Dataset Description**:
    - The dataset contains optical response data for a variety of MIM metasurfaces with two rectangular Au nanostructures, including:
        - Varying dimensions (length, width) of the rectangular blocks.
        - Different positions of the rectangular blocks within the unit cell.
        - Measurements of circular polarization reflection coefficients at wavelength of {wave_length}nm.
    - The dataset is structured as follows:
        - **Input Structure Representation**: [a, b, c, d, e, f, g, h]
            - a: x-coordinate of the top-left corner of the first rectangular block.
            - b: y-coordinate of the top-left corner of the first rectangular block.
            - c: Length of the first rectangular block.
            - d: Width of the first rectangular block.
            - e: x-coordinate of the top-left corner of the second rectangular block.
            - f: y-coordinate of the top-left corner of the second rectangular block.
            - g: Length of the second rectangular block.
            - h: Width of the second rectangular block.
    **Output Spectrum Specifications**:
    - The optical response is defined by circular polarization reflection coefficients, split into real and imaginary components. The coefficients correspond to the following polarization conversions:
        - r_rl: Right-handed circular polarization (RCP) → Left-handed circular polarization (LCP).
        - r_lr: Left-handed circular polarization (LCP) → Right-handed circular polarization (RCP).
        - r_rr: Right-handed circular polarization (RCP) → Right-handed circular polarization (RCP).
    - The output format for the spectrum response at {wave_length}nm follows the following structure:
        {{
            "spectrum_response": {{
                "r_rl": {{"real": <value>, "imag": <value>}},
                "r_lr": {{"real": <value>, "imag": <value>}},
                "r_rr": {{"real": <value>, "imag": <value>}}
            }}
        }}
        
    **Task Requirements**:
    1. Carefully scan through the dataset and try to discover some universe rules and relationship between structures and responses.
    2. Think thoroughly and predict precisely the polarization-resolved reflection spectra for the target structure at a wavelength of {wave_length} nm.
    3. Consider near-field coupling and far-field interference effects.
    4.The data points are designed to ensure that the two rectangular structures do not overlap. 
    **Note**: 
    Small changes in structure may lead to significant changes in optical response, especially near exceptional points (EPs).
    Due to symmetry, r_rl and r_lr are expected to be close to each other. But the position of the two blocks will break the symmetry, leading to small differences.
    Please be carefully when analyzing the sign and scale of both the real and imaginary parts.
    """
