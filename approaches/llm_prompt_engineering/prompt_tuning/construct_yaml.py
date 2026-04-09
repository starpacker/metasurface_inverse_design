import yaml
def generate_prompt_config(wave_length: int):
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
  config = {
        "system_prompt": initial_prompt,  # 使用前述的initial_prompt变量
        "response_format": 
        """
        please answer in the following format:
        <ans>
        {
        "spectrum_response": {
            "r_rl": {"real": <float>, "imag": <float>},
            "r_lr": {"real": <float>, "imag": <float>},
            "r_rr": {"real": <float>, "imag": <float>}
        }
        }
        </ans>
        """,
        "template_variables": {
            "wavelength": wave_length,
            "material": "Au/SiO2",
            "polarization_types": [
                "RCP→LCP (r_rl)",
                "LCP→RCP (r_lr)",
                "RCP→RCP (r_rr)"
            ]
        }
    }

  with open(f"/data/group_003/prompt_tuning/prompt_config_{wave_length}.yaml", 'w') as f:
        yaml.dump(config, f, 
                  allow_unicode=True, 
                  sort_keys=False,
                  default_flow_style=False,
                  width=120)
  print("finished!")


if __name__ == "__main__":
  wave_length = 650

  generate_prompt_config(wave_length=wave_length)
