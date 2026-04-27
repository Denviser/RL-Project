import numpy as np
import matplotlib.pyplot as plt
import cma_utils
import cma_utils_pilot
from scipy.signal import find_peaks

N_symbols = 1000000
freq_offset = 2e3
generated_offset = 200 #Number of samples where the pilot start is offset
fs = 2e9
num_taps = 51

def find_peaks_like_real_time(signal,window=10000,threshold_coeff=5,frame_len=3712):
    """This function takes in the signal skips the first window symbols in order to find a good estimate for the mean and variance of the noise floor
    It then scans the signal looking for peaks. We subtract the peak location from k*3172 to get an estimate of the offset"""
    start_sum = np.sum(signal[:window])
    start_sum_squares = np.sum(signal[:window]**2)
    peaks = []
    current_sum = start_sum
    current_sum_squares = start_sum_squares
    average_window_size = 20 # If I have peaks within this window I will take average of them to find the true peak index
    peak_num = 0
    for index in range(window,len(signal)):
        current_sum = current_sum + signal[index]-signal[index-window]
        current_sum_squares = current_sum_squares + signal[index]**2 - signal[index-window]**2
        current_mean = current_sum/window
        current_std = np.sqrt(current_sum_squares/window - current_mean**2)
        #If the peak is much larger than the current statistics then return that index
        if signal[index]>=current_mean+threshold_coeff*current_std: #I am making sure that mult

            if peaks and index - peaks[-1] < average_window_size:
                peaks_index_sum += index
                peaks_count_sum +=1
            elif peaks and index - peaks[-1] >= average_window_size:
                peaks[-1] = int(peaks_index_sum/peaks_count_sum)
                peaks.append(index)
                peaks_index_sum = index
                peaks_count_sum = 1
            else:
                peaks.append(index)
                peaks_index_sum = index
                peaks_count_sum = 1

    return peaks

def find_tau(peaks):
    tau = 200
    return tau

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
    #peaks_x, _ = find_peaks(abs_corr_x, height=np.max(abs_corr_x) * 0.5, distance=frame_len - 100) # Adjust height/distance as needed
    peaks_x = find_peaks_like_real_time(abs_corr_x,window=10000)

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
    #peaks_y, _ = find_peaks(abs_corr_y, height=np.max(abs_corr_y) * 0.5, distance=frame_len - 100) # Adjust height/distance as needed
    peaks_y = find_peaks_like_real_time(abs_corr_y,window=10000)

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

# def main():
#     initial_symbols = cma_utils_pilot.generate_stream(N_symbols,offset=generated_offset)

#     E_after_pmd = cma_utils.apply_pmd(initial_symbols,
#                                      DGD_ps_per_sqrt_km=30,
#                                      L_m=10000,
#                                      N_sections=100,
#                                      Rs=32e9,
#                                      SpS=4)
#     E_cfo = cma_utils.apply_cfo_both_polarisations(E_after_pmd, freq_offset, fs)

#     E_noisy = cma_utils.apply_awgn(E_cfo, 30)

#     # # uncomment this once code is complete
#     # peaks = find_peaks_like_real_time(E_noisy)
#     # tau_est = find_tau(peaks)

#     tau_est = 200

#     E_out, stats = cma_utils_pilot.lms_cfo_joint_with_pilots(E_noisy, num_taps, tau_est, mu=1e-4, mu_f=1e-6, fs = 2e9)

#     pxx, pxy, pyx, pyy, f_est, cma_error = stats['pxx'], stats['pyx'], stats['pyx'], stats['pyy'], stats['f_est'], stats['cma_error']
#     smoothed_log_errors = cma_utils.plot_conv(cma_error)
#     convergence_symbol= cma_utils.find_convergence_backward(smoothed_log_errors)
#     print("convergence symbol = ", convergence_symbol)
#     print("frequency offset estimated = ", f_est)

#     cma_utils.plot_constellation(E_out)

#     # check_correlation(E_with_pmd,pilot_sequence)


def main():
    initial_symbols = cma_utils_pilot.generate_stream(N_symbols, offset=generated_offset)

    E_after_pmd = cma_utils.apply_pmd(initial_symbols,
                                     DGD_ps_per_sqrt_km=30,
                                     L_m=10000,
                                     N_sections=100,
                                     Rs=32e9,
                                     SpS=4)
    E_cfo = cma_utils.apply_cfo_both_polarisations(E_after_pmd, freq_offset, fs)

    E_noisy = cma_utils.apply_awgn(E_cfo, 30)

    # initial tau estimation on noisy signal
    peaks = find_peaks_like_real_time(E_noisy)
    tau_est = find_tau(peaks)
    print(f"Initial tau estimate from noisy signal: {tau_est}")

    max_iters = 10
    converged = False
    iteration = 0
    
    E_out = None
    stats = None

    while not converged and iteration < max_iters:
        iteration += 1
        
        # run LMS on noisy signal using current tau_est
        E_out, stats = cma_utils_pilot.lms_cfo_joint_with_pilots(E_noisy, num_taps, tau_est, mu=1e-4, mu_f=1e-6, fs=2e9)
        
        # re-estimate tau
        new_peaks = find_peaks_like_real_time(E_out)
        new_tau = find_tau(new_peaks)
        
        print(f"Iteration {iteration}: Current tau = {tau_est}, Re-estimated tau = {new_tau}")
        
        # Check for convergence
        if new_tau == tau_est:
            converged = True
            print(f"Convergence reached, final tau: {tau_est}\n")
        else:
            tau_est = new_tau

    if not converged:
        print(f"did not converge after {max_iters} iterations, latest tau: {tau_est}\n")

    error_list = stats['cma_error']
    f_est = stats['f_est']
    
    smoothed_log_errors = cma_utils.plot_conv(error_list)
    convergence_symbol = cma_utils.find_convergence_backward(smoothed_log_errors)
    
    print("Final Tau =", tau_est)
    print("Convergence symbol =", convergence_symbol)
    print("Frequency offset estimated =", f_est)

    cma_utils.plot_constellation(E_out)


if __name__ == "__main__":
    main()