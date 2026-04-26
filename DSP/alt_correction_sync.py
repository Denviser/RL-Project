import numpy as np
import matplotlib.pyplot as plt
import cma_utils
import cma_utils_pilot
from scipy.signal import find_peaks

N_symbols = 100000
freq_offset = 2e3
generated_offset = 200 #Number of samples where the pilot start is offset



def check_correlation(symbols,pilot_sequence):
    correlation_x_pol = np.correlate(symbols[:, 0], pilot_sequence[:, 0], mode='valid')
    correlation_y_pol = np.correlate(symbols[:, 1], pilot_sequence[:, 1], mode='valid')

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    frame_len = 3712 # Defined in cma_utils_pilot.py

    # Expected peak characteristics based on how the stream is generated
    # np.roll(stream, offset) shifts the start of the first frame (and thus its pilot) to 'offset'.
    expected_first_peak_index = generated_offset
    expected_peak_distance = frame_len

    # --- X-polarization ---
    abs_corr_x = np.abs(correlation_x_pol)
    # Find peaks: height is a percentage of max, distance ensures we find distinct frame peaks
    # The distance parameter helps to avoid detecting multiple points within a single broad peak
    # or noise. It should be roughly the expected distance between peaks.
    peaks_x, _ = find_peaks(abs_corr_x, height=np.max(abs_corr_x) * 0.5, distance=frame_len - 100) # Adjust height/distance as needed

    if len(peaks_x) > 1:
        peak_distances_x = np.diff(peaks_x)
        avg_peak_distance_x = np.mean(peak_distances_x)
        print(f"X-polarization: Detected {len(peaks_x)} peaks at indices: {peaks_x}")
        print(f"X-polarization: Average distance between peaks: {avg_peak_distance_x:.2f} (Expected: {expected_peak_distance})")
    elif len(peaks_x) == 1:
        print(f"X-polarization: Detected 1 peak at index: {peaks_x[0]}")
        avg_peak_distance_x = None
    else:
        print("X-polarization: No significant peaks detected.")
        avg_peak_distance_x = None

    axes[0].plot(abs_corr_x)
    axes[0].axvline(x=expected_first_peak_index, color='r', linestyle='--', label=f'Expected 1st Peak ({expected_first_peak_index})')
    if len(peaks_x) > 0:
        axes[0].plot(peaks_x, abs_corr_x[peaks_x], "x", color='m', markersize=8, label='Detected Peaks')
        axes[0].axvline(x=peaks_x[0], color='g', linestyle='-', label=f'Actual 1st Peak ({peaks_x[0]})')
    axes[0].set_title(f"Cross-correlation with full frame pilot mask (X pol) | Offset = {generated_offset}")
    axes[0].set_ylabel("Magnitude")
    axes[0].legend()
    axes[0].grid(True)

    # --- Y-polarization ---
    abs_corr_y = np.abs(correlation_y_pol)
    peaks_y, _ = find_peaks(abs_corr_y, height=np.max(abs_corr_y) * 0.5, distance=frame_len - 100) # Adjust height/distance as needed

    if len(peaks_y) > 1:
        peak_distances_y = np.diff(peaks_y)
        avg_peak_distance_y = np.mean(peak_distances_y)
        print(f"Y-polarization: Detected {len(peaks_y)} peaks at indices: {peaks_y}")
        print(f"Y-polarization: Average distance between peaks: {avg_peak_distance_y:.2f} (Expected: {expected_peak_distance})")
    elif len(peaks_y) == 1:
        print(f"Y-polarization: Detected 1 peak at index: {peaks_y[0]}")
        avg_peak_distance_y = None
    else:
        print("Y-polarization: No significant peaks detected.")
        avg_peak_distance_y = None

    axes[1].plot(abs_corr_y)
    axes[1].axvline(x=expected_first_peak_index, color='r', linestyle='--', label=f'Expected 1st Peak ({expected_first_peak_index})')
    if len(peaks_y) > 0:
        axes[1].plot(peaks_y, abs_corr_y[peaks_y], "x", color='m', markersize=8, label='Detected Peaks')
        axes[1].axvline(x=peaks_y[0], color='g', linestyle='-', label=f'Actual 1st Peak ({peaks_y[0]})')
    axes[1].set_title(f"Cross-correlation with full frame pilot mask (Y pol) | Offset = {generated_offset}")
    axes[1].set_xlabel("Lag (samples)")
    axes[1].set_ylabel("Magnitude")
    axes[1].legend()
    axes[1].grid(True)
    plt.tight_layout()
    plt.show()

def main():
    pilot_sequence = cma_utils_pilot.generate_pilot_mask()
    initial_symbols = cma_utils_pilot.generate_stream(N_symbols,offset=generated_offset)

    E_with_pmd = cma_utils.apply_pmd(initial_symbols,
                                     DGD_ps_per_sqrt_km=50,
                                     L_m=10000,
                                     N_sections=100,
                                     Rs=32e9,
                                     SpS=4)
    
    check_correlation(E_with_pmd,pilot_sequence)


if __name__ == "__main__":
    main()