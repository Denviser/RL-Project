import numpy as np
import matplotlib.pyplot as plt
import cma_utils
import cma_utils_pilot
from scipy.signal import find_peaks


def remove_outliers_iqr(data_array, iqr_multiplier=1.5):
    """
    Removes outliers from a 1D NumPy array using the Interquartile Range (IQR) method.

    Outliers are defined as data points that fall below (Q1 - iqr_multiplier * IQR)
    or above (Q3 + iqr_multiplier * IQR).

    Args:
        data_array (np.ndarray): The input 1D NumPy array from which to remove outliers.
        iqr_multiplier (float, optional): The multiplier for the IQR to define the
                                          outlier bounds. Common values are 1.5 (for
                                          mild outliers) or 3.0 (for extreme outliers).
                                          Defaults to 1.5.

    Returns:
        np.ndarray: A new NumPy array with outliers removed.
    """
    if not isinstance(data_array, np.ndarray) or data_array.ndim != 1:
        raise ValueError("Input must be a 1D NumPy array.")
    
    if data_array.size == 0:
        return np.array([])
    
    # Calculate Q1 (25th percentile)
    Q1 = np.percentile(data_array, 25)
    
    # Calculate Q3 (75th percentile)
    Q3 = np.percentile(data_array, 75)
    
    # Calculate the Interquartile Range (IQR)
    IQR = Q3 - Q1
    
    # Define the outlier bounds
    lower_bound = Q1 - (iqr_multiplier * IQR)
    upper_bound = Q3 + (iqr_multiplier * IQR)
    
    # Filter the array to keep only the elements within the bounds
    filtered_array = data_array[(data_array >= lower_bound) & (data_array <= upper_bound)]
    
    return filtered_array

def find_offset_like_real_time(signal,window_factor=3,threshold_coeff=5,frame_len=3712):
    """This function takes in the signal skips the first window symbols in order to find a good estimate for the mean and variance of the noise floor
    It then scans the signal looking for peaks. We subtract the peak location from k*3172 to get an estimate of the offset"""
    
    window = window_factor*frame_len
    start_sum = np.sum(signal[:window])
    start_sum_squares = np.sum(signal[:window]**2)
    offsets = []
    current_sum = start_sum
    current_sum_squares = start_sum_squares
    average_window_size = 20 # If I have peaks within this window I will take average of them to find the true peak index
    peak_num = window_factor-1
    last_peak_index = 0
    for index in range(window,len(signal)):
        current_sum = current_sum + signal[index]-signal[index-window]
        current_sum_squares = current_sum_squares + signal[index]**2 - signal[index-window]**2
        current_mean = current_sum/window
        current_std = np.sqrt(current_sum_squares/window - current_mean**2)
        #If the peak is much larger than the current statistics then return that index
        if signal[index]>=current_mean+threshold_coeff*current_std: #I am making sure that mult
            if index - last_peak_index < average_window_size:
                #Taking max over current index and last_peak_index
                if signal[index]>signal[last_peak_index]:
                    # peak_true_index =  index
                    last_peak_index = index
            else:
                #print(last_peak_index)
                #print("the start is",peak_num*frame_len)
                found_offset = last_peak_index - peak_num*frame_len
                if found_offset < 0:
                    while found_offset < 0:
                        peak_num -= 1
                        found_offset = last_peak_index - peak_num*frame_len
                elif found_offset > frame_len:
                    while found_offset > frame_len:
                        peak_num += 1
                        found_offset = last_peak_index - peak_num*frame_len
                offsets.append(found_offset)
                last_peak_index = index
                peak_num += 1


    offsets.append(last_peak_index-peak_num*frame_len)
    offsets_cleaned = remove_outliers_iqr(np.array(offsets))
    print(offsets_cleaned)
    #print(offsets_cleaned)
    return int(offsets_cleaned.mean())

def find_peaks_like_real_time(signal,window_factor=3,threshold_coeff=6,frame_len=3712):
    """This function takes in the signal skips the first window symbols in order to find a good estimate for the mean and variance of the noise floor
    It then scans the signal looking for peaks. We subtract the peak location from k*3172 to get an estimate of the offset"""
    
    window = window_factor*frame_len
    start_sum = np.sum(signal[:window])
    start_sum_squares = np.sum(signal[:window]**2)
    peaks = []
    current_sum = start_sum
    current_sum_squares = start_sum_squares
    average_window_size = 20 # If I have peaks within this window I will take average of them to find the true peak index
    peak_num = window_factor-1
    last_peak_index = 0
    for index in range(window,len(signal)):
        current_sum = current_sum + signal[index]-signal[index-window]
        current_sum_squares = current_sum_squares + signal[index]**2 - signal[index-window]**2
        current_mean = current_sum/window
        current_std = np.sqrt(current_sum_squares/window - current_mean**2)
        #If the peak is much larger than the current statistics then return that index
        if signal[index]>=current_mean+threshold_coeff*current_std: #I am making sure that mult
            if index - last_peak_index < average_window_size:
                #Taking max over current index and last_peak_index
                if signal[index]>signal[last_peak_index]:
                    # peak_true_index =  index
                    last_peak_index = index
            else:
                #print(last_peak_index)
                #print("the start is",peak_num*frame_len)
                found_peak = last_peak_index
                peaks.append(found_peak)
                last_peak_index = index
                peak_num += 1


    peaks.append(last_peak_index)
    #print(offsets_cleaned)
    return peaks[1:]
def check_correlation(symbols,pilot_sequence,generated_offset):
    correlation_x_pol = np.correlate(symbols[:, 0], pilot_sequence[:, 0], mode='valid')
    correlation_y_pol = np.correlate(symbols[:, 1], pilot_sequence[:, 1], mode='valid')

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    frame_len = 3712 # Defined in cma_utils_pilot.py

    # Expected peak characteristics based on how the stream is generated
    # np.roll(stream, offset) shifts the start of the first frame (and thus its pilot) to 'offset'.
    expected_first_peak_index = generated_offset + 3*frame_len 
    expected_peak_distance = frame_len

    # --- X-polarization ---
    abs_corr_x = np.abs(correlation_x_pol)
    # Find peaks: height is a percentage of max, distance ensures we find distinct frame peaks
    # The distance parameter helps to avoid detecting multiple points within a single broad peak
    # or noise. It should be roughly the expected distance between peaks.
    #peaks_x, _ = find_peaks(abs_corr_x, height=np.max(abs_corr_x) * 0.5, distance=frame_len - 100) # Adjust height/distance as needed
    peaks_x = find_peaks_like_real_time(abs_corr_x)

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
    peaks_y = find_peaks_like_real_time(abs_corr_y)

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
    fs = 2e9
    num_taps =51
    N_symbols = 1000000
    freq_offset = 2e3
    generated_offset = 200 #Number of samples where the pilot start is offset
    frames_to_estimate_offset = 15
    frame_len = 3712
    initial_symbols = cma_utils_pilot.generate_stream(N_symbols,offset=generated_offset)

    E_after_pmd = cma_utils.apply_pmd(initial_symbols,
                                     DGD_ps_per_sqrt_km=30,
                                     L_m=10000,
                                     N_sections=100,
                                     Rs=32e9,
                                     SpS=4)
    E_cfo = cma_utils.apply_cfo_both_polarisations(E_after_pmd, freq_offset, fs)
    pilot_seq = cma_utils_pilot.generate_pilot_mask()
    E_noisy = cma_utils.apply_awgn(E_cfo, 30)
    #E_noisy = E_after_pmd
    # # uncomment this once code is complete
    # peaks = find_peaks_like_real_time(E_noisy)
    # tau_est = find_tau(peaks)
    correlated_x_pol = np.correlate(E_noisy[:15*frame_len, 0], pilot_seq[:, 0], mode='valid')
    tau_est = find_offset_like_real_time(abs(correlated_x_pol))
    print("tau est = ", tau_est)
    #tau_est = 200
    E_out, stats = cma_utils_pilot.lms_cfo_joint_with_pilots(E_noisy[15*frame_len:], num_taps, tau_est, mu=1e-4, mu_f=1e-6, fs = 2e9)

    correlated_x_pol = np.correlate(E_out[20*frame_len:, 0], pilot_seq[:, 0], mode='valid')
    correlated_x_with_other_pilot = np.correlate(E_out[20*frame_len:, 0], pilot_seq[:, 1], mode='valid')
    plt.subplot(2,1,1)
    plt.plot(abs(correlated_x_pol))
    plt.subplot(2,1,2)
    plt.plot(abs(correlated_x_with_other_pilot))
    plt.show()
    tau_est_after = find_offset_like_real_time(abs(correlated_x_pol))
    print("tau est after = ", tau_est_after)

    pxx, pxy, pyx, pyy, f_est, cma_error = stats['pxx'], stats['pyx'], stats['pyx'], stats['pyy'], stats['f_est'], stats['cma_error']
    smoothed_log_errors = cma_utils.plot_conv(cma_error)
    convergence_symbol= cma_utils.find_convergence_backward(smoothed_log_errors)
    print("convergence symbol = ", convergence_symbol)
    print("frequency offset estimated = ", f_est)

    cma_utils.plot_constellation(E_out)

    # check_correlation(E_with_pmd,pilot_sequence)

def compare_avg_performance_pilot_vs_no_pilot():
    fs = 2e9
    num_taps =51
    N_symbols = 1000000
    freq_offset = 2e3
    generated_offset = 200 #Number of samples where the pilot start is offset
    frames_to_estimate_offset = 15
    frame_len = 3712
    initial_symbols = cma_utils_pilot.generate_stream(N_symbols,offset=generated_offset)
    Num_avg = 15
    pilot_seq = cma_utils_pilot.generate_pilot_mask()
    convergence_symbol_pilot_sum = 0
    convergence_symbol_no_pilot_sum = 0
    output_snr_pilot_sum = 0
    output_snr_no_pilot_sum = 0
    f_est_pilots_sum = 0
    f_est_no_pilots_sum = 0
    for epoch in range(Num_avg):
        E_after_pmd = cma_utils.apply_pmd(initial_symbols,
                                        DGD_ps_per_sqrt_km=30,
                                        L_m=10000,
                                        N_sections=100,
                                        Rs=32e9,
                                        SpS=4)
        E_cfo = cma_utils.apply_cfo_both_polarisations(E_after_pmd, freq_offset, fs)
        E_noisy = cma_utils.apply_awgn(E_cfo, 30)
    
        tau_est = generated_offset
        E_out_pilots, stats_pilots = cma_utils_pilot.lms_cfo_joint_with_pilots(E_noisy, num_taps, tau_est, mu=1e-4, mu_f=1e-6, fs = 2e9)
        E_out_no_pilots, stats_no_pilots = cma_utils.lms_cfo_joint(E_noisy, num_taps, mu=1e-4, mu_f=1e-6, fs = 2e9)
        pxx, pxy, pyx, pyy, f_est_pilots, cma_error_pilots = stats_pilots['pxx'], stats_pilots['pyx'], stats_pilots['pyx'], stats_pilots['pyy'], stats_pilots['f_est'], stats_pilots['cma_error']
        pxx, pxy, pyx, pyy, f_est_no_pilots, cma_error_no_pilots = stats_no_pilots['pxx'], stats_no_pilots['pyx'], stats_no_pilots['pyx'], stats_no_pilots['pyy'], stats_no_pilots['f_est'], stats_no_pilots['cma_error']
        
        smoothed_log_errors_pilots = cma_utils.plot_conv(cma_error_pilots)
        convergence_symbol_pilots= cma_utils.find_convergence_backward(smoothed_log_errors_pilots)
        print("convergence symbol pilots = ", convergence_symbol_pilots)
        convergence_symbol_pilot_sum += convergence_symbol_pilots
        output_snr_pilots = cma_utils.cluster_and_get_avg_snr(E_out_pilots,n_clusters=4)[0] #Only taking X
        output_snr_no_pilot_sum += output_snr_pilots
        print("frequency offset estimated pilots = ", f_est_pilots)
        f_est_pilots_sum += f_est_pilots
        cma_utils.save_constellation(E_out_pilots,f"E_out_pilots_{epoch}")

        smoothed_log_errors_no_pilots = cma_utils.plot_conv(cma_error_no_pilots)
        convergence_symbol_no_pilots= cma_utils.find_convergence_backward(smoothed_log_errors_no_pilots)
        print("convergence symbol no pilots = ", convergence_symbol_no_pilots)
        convergence_symbol_no_pilot_sum += convergence_symbol_no_pilots
        output_snr_no_pilots = cma_utils.cluster_and_get_avg_snr(E_out_no_pilots,n_clusters=4)[0] #Only taking X
        output_snr_no_pilot_sum += output_snr_no_pilots
        f_est_no_pilots_sum += f_est_no_pilots   
        print("frequency offset estimated no pilots = ", f_est_no_pilots)
        cma_utils.save_constellation(E_out_no_pilots,f"E_out_no_pilots_{epoch}")



# def main():
#     N_symbols = 100000
#     pilot_sequence = cma_utils_pilot.generate_pilot_mask()
#     initial_symbols = cma_utils_pilot.generate_stream(N_symbols,offset=1000)

#     E_after_pmd = cma_utils.apply_pmd(initial_symbols,
#                                      DGD_ps_per_sqrt_km=30,
#                                      L_m=10000,
#                                      N_sections=100,
#                                      Rs=32e9,
#                                      SpS=4)
#     x_corr = np.correlate(initial_symbols[:, 0],pilot_sequence[:, 0], mode='valid')
#     y_corr = np.correlate(E_after_pmd[:, 1], pilot_sequence[:, 1], mode='valid')

#     plt.plot(abs(x_corr))
#     plt.plot(abs(y_corr))
#     plt.show()
#     offset_predicted = find_offset_like_real_time(abs(x_corr))
#     offset_predicted_y = find_offset_like_real_time(abs(y_corr))
#     print(offset_predicted)
#     print(offset_predicted_y)
#     #check_correlation(E_after_pmd,pilot_sequence)

def plotting_correlation():
    N_symbols = 100000
    pilot_sequence = cma_utils_pilot.generate_pilot_mask()
    initial_symbols = cma_utils_pilot.generate_stream(N_symbols,offset=1000)
    E_after_pmd = cma_utils.apply_pmd(initial_symbols,
                                     DGD_ps_per_sqrt_km=30,
                                     L_m=10000,
                                     N_sections=100,
                                     Rs=32e9,
                                     SpS=4)
    E_after_cfo = cma_utils.apply_cfo_both_polarisations(E_after_pmd, 2e3, 2e9)
    E_after_noise = cma_utils.apply_awgn(E_after_cfo, 5)
    check_correlation(E_after_noise,pilot_sequence,1000)

if __name__ == "__main__":
    compare_avg_performance_pilot_vs_no_pilot()