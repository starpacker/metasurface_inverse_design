import matplotlib.pyplot as plt
import numpy as np

# Data for the first structure
sample1_structure = "[315, 117, 34, 22, 254, 309, 50, 70]"
sample1_data = {
    "600nm": {
        "expected": [0.0204, 0.0081, 0.0206, 0.0057, 0.9027, 0.0385],
        "actual": [0.02448, -0.00224, 0.04814, -0.01751, 0.84241, 0.0342]
    },
    "610nm": {
        "expected": [0.0161, 0.0351, 0.0170, 0.0324, 0.8689, 0.0225],
        "actual": [0.03, 0.05, 0.02, 0.04, 0.68, -0.03]
    },
    "620nm": {
        "expected": [0.0443, 0.0565, 0.0456, 0.0518, 0.8289, 0.0400],
        "actual": [-0.05, 0.05, -0.04, 0.04, 0.80, 0.02]
    },
    "640nm": {
        "expected": [0.1042, 0.0211, 0.0990, 0.0138, 0.7961, 0.1216],
        "actual": [0.15, 0.1, 0.1, -0.15, 0.7, -0.1]
    },
    "670nm": {
        "expected": [0.0883, -0.0368, 0.0677, -0.0417, 0.8554, 0.1906],
        "actual": [0.2, -0.3, 0.1, -0.2, 0.7, -0.1]
    },
    "700nm": {
        "expected": [0.0549, -0.0602, 0.0568, -0.0337, 0.8939, 0.1931],
        "actual": [0.05, 0, 0.04, -0.02, 0.9, 0.2]
    }
}

# Data for the second structure
sample2_structure = "[253, 21, 33, 127, 249, 100, 156, 25]"
sample2_data = {
    "600nm": {
        "expected": [-0.0198, -0.0479, -0.0202, -0.0465, 0.9483, -0.0097],
        "actual": [0.1, 0.05, 0.08, 0.06, 0.85, 0.03]
    },
    "610nm": {
        "expected": [-0.0122, -0.0602, -0.0135, -0.0601, 0.9548, -0.0184],
        "actual": [0.1, -0.06, 0.1, -0.07, 0.9, 0.1]
    },
    "620nm": {
        "expected": [-0.0109, -0.0761, -0.0128, -0.0744, 0.9519, -0.0302],
        "actual": [0.12, 0.08, 0.10, 0.09, 0.85, -0.02]
    },
    "640nm": {
        "expected": [-0.0217, -0.1191, -0.0218, -0.1168, 0.9405, -0.0524],
        "actual": [0.15, -0.2, 0.1, -0.18, 0.75, 0.05]
    },
    "670nm": {
        "expected": [-0.0808, -0.1967, -0.0770, -0.1966, 0.9025, -0.1209],
        "actual": [-0.15, 0.05, 0.05, -0.15, 0.70, 0.05]
    },
    "700nm": {
        "expected": [-0.2829, -0.2435, -0.2872, -0.2507, 0.7063, -0.1793],
        "actual": [-0.3, -0.2, -0.3, -0.15, 0.6, 0.1]
    }
}

def plot_spectrum_versus_wavelength(data, structure_str):
    wavelengths = list(data.keys())
    wavelengths.sort()  # Sort wavelengths for proper x-axis order

    # Extract real and imaginary parts for each reflection coefficient
    expected_rrl = []
    expected_rlr = []
    expected_rrr = []
    actual_rrl = []
    actual_rlr = []
    actual_rrr = []

    for wave in wavelengths:
        expected = data[wave]["expected"]
        actual = data[wave]["actual"]

        # Order of the values: [r_rl_real, r_rl_imag, r_lr_real, r_lr_imag, r_rr_real, r_rr_imag]
        expected_r_rl_real = expected[0]
        expected_r_rl_imag = expected[1]
        expected_r_lr_real = expected[2]
        expected_r_lr_imag = expected[3]
        expected_r_rr_real = expected[4]
        expected_r_rr_imag = expected[5]

        actual_r_rl_real = actual[0]
        actual_r_rl_imag = actual[1]
        actual_r_lr_real = actual[2]
        actual_r_lr_imag = actual[3]
        actual_r_rr_real = actual[4]
        actual_r_rr_imag = actual[5]

        expected_rrl.append(expected_r_rl_real)
        expected_rlr.append(expected_r_lr_real)
        expected_rrr.append(expected_r_rr_real)
        actual_rrl.append(actual_r_rl_real)
        actual_rlr.append(actual_r_lr_real)
        actual_rrr.append(actual_r_rr_real)

    # Create subplots
    fig, axs = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
    fig.suptitle(f"Spectral Response Comparison for {structure_str}", fontsize=14)

    # Plot r_rr
    axs[0].plot(wavelengths, expected_rrr, 'r-', label='Expected Real')
    axs[0].plot(wavelengths, actual_rrr, 'b-', label='Actual Real')
    axs[0].set_title("r_rr")
    axs[0].set_ylabel('Value')
    axs[0].grid(True, linestyle='--', alpha=0.7)
    axs[0].legend()

    # Plot r_rl
    axs[1].plot(wavelengths, expected_rrl, 'r-', label='Expected Real')
    axs[1].plot(wavelengths, actual_rrl, 'b-', label='Actual Real')
    axs[1].set_title("r_rl")
    axs[1].set_ylabel('Value')
    axs[1].grid(True, linestyle='--', alpha=0.7)
    axs[1].legend()

    # Plot r_lr
    axs[2].plot(wavelengths, expected_rlr, 'r-', label='Expected Real')
    axs[2].plot(wavelengths, actual_rlr, 'b-', label='Actual Real')
    axs[2].set_title("r_lr")
    axs[2].set_ylabel('Value')
    axs[2].set_xlabel('Wavelength (nm)')
    axs[2].grid(True, linestyle='--', alpha=0.7)
    axs[2].legend()

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.xticks(rotation=45)

# Plot for Sample 1
plot_spectrum_versus_wavelength(sample1_data, sample1_structure)
plt.savefig(f"{sample1_structure}_spectrum.png", dpi=300, bbox_inches='tight')

# Plot for Sample 2
plot_spectrum_versus_wavelength(sample2_data, sample2_structure)
plt.savefig(f"{sample2_structure}_spectrum.png", dpi=300, bbox_inches='tight')

# plt.show()
