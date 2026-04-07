import numpy as np
import matplotlib.pyplot as plt
import cma_utils
from collections import deque
from scipy import stats
from scipy import stats

N_SYMBOLS = 100000
NUM_TAPS=51

def plot_conv(cma_error, plot_string):
def plot_conv(cma_error, plot_string):

    # errors = np.array(list(cma_error.values()))
    errors = np.array(list(cma_error))

    eps = 1e-12
    errors = np.maximum(errors, eps)
    log_errors = np.log10(errors)

    window = 200
    smooth_log_errors = np.convolve(
        log_errors,
        np.ones(window) / window,
        mode="valid"
    )
    # symbols = np.array(list(cma_error.keys()))
    symbols = np.array(list(cma_error))

    symbols_smooth = symbols[window - 1:]
    
    plt.figure(figsize=(12, 6))
    # plt.plot(symbols_smooth, smooth_log_errors, linewidth=2)
    plt.plot(smooth_log_errors, linewidth=2)
    plt.xlabel("Symbol number")
    plt.ylabel("log10(CMA error)")
    plt.title("log10(CMA Error) vs symbol number (window=15)")
    plt.grid(True)
    plt.savefig(f"cma_error_smoothed_{plot_string}.png")
    plt.savefig(f"cma_error_smoothed_{plot_string}.png")

    plt.close('all')
    return smooth_log_errors

def cma_python_with_cma_error_convergence(E_in, num_taps, mu_CMA=0.01, error_threshold=1e-3,Num_loops = 1,num_symbols_store = 2000 , patience_counter_threshold = 100):
    """
    CMA with moving-average convergence detection (last 10 symbols)
    In order to calculate the error threshold I store the last (lets say 2000) N symbols.
    I now calculate avg error over last N symbols and compare the means of N/2 start and N/2 end
    """

    # ---- Copy and normalize ----
    xpol = E_in[:, 0].astype(complex)
    ypol = E_in[:, 1].astype(complex)

    xpol = xpol / np.sqrt(np.mean(np.abs(xpol)**2))
    ypol = ypol / np.sqrt(np.mean(np.abs(ypol)**2))


    N = len(xpol)
    R = 1

    # ---- Tap initialization ----
    pxx = np.zeros(num_taps, dtype=complex)
    pxy = np.zeros(num_taps, dtype=complex)
    pyx = np.zeros(num_taps, dtype=complex)
    pyy = np.zeros(num_taps, dtype=complex)

    center = (num_taps - 1) // 2
    pxx[center] = 1
    pyy[center] = 1

    convergence_symbol = None

    #I am storing the last 
    old_error_window = deque(maxlen=num_symbols_store//2)   # last 10 CMA errors
    recent_error_window = deque(maxlen=num_symbols_store//2)   # last 10 CMA errors
    #this counts how long we have relative error to be less than threshold


    #print("initial_filters",initial_filters)
    #print("shape is",initial_filters.shape)
    patience_counter = 0
    half_win = num_symbols_store//2
    sum_old = 0.0
    sum_recent = 0.0
    log_cma_errors = []
    difference_cma_errors = []

    for k in range(Num_loops):
        for ii in range(num_taps - 1, N):

            x_vec = xpol[ii - (num_taps - 1): ii + 1][::-1]
            y_vec = ypol[ii - (num_taps - 1): ii + 1][::-1]

            x_cap = np.dot(pxx, x_vec) + np.dot(pxy, y_vec)
            y_cap = np.dot(pyx, x_vec) + np.dot(pyy, y_vec)

            e_x = R**2 - np.abs(x_cap)**2
            e_y = R**2 - np.abs(y_cap)**2

            log_cma_error = np.log10(0.5*(np.abs(e_x) + np.abs(e_y)))
            log_cma_errors.append(log_cma_error)
            #Add the magnitude of the filter vector which I define as sum of mod square of its elements
            
            if len(recent_error_window) == half_win:
                leaving_recent = recent_error_window[0]
                
                # Move from recent to old
                if len(old_error_window) == half_win:
                    sum_old -= old_error_window[0] # Drop oldest from old sum
                
                old_error_window.append(leaving_recent)
                sum_old += leaving_recent
                
                # Remove from recent sum
                sum_recent -= leaving_recent

            recent_error_window.append(log_cma_error)
            sum_recent += log_cma_error

            # --- CONVERGENCE CHECK (Using O(1) Sums) ---
            m_rec = sum_recent / half_win
            m_old = sum_old / half_win
            
            # Relative error check
            diff_error = m_rec - m_old
            difference_cma_errors.append(np.abs(diff_error))
            
            if convergence_symbol is None and len(old_error_window) == half_win:
                
                if np.abs(diff_error)< error_threshold:
                    patience_counter += 1
                    if patience_counter >= patience_counter_threshold:
                        convergence_symbol = ii - patience_counter_threshold
                else:
                    patience_counter = 0


            # ---- Tap updates ----
            pxx += 2 * mu_CMA * e_x * x_cap * np.conj(x_vec)
            pxy += 2 * mu_CMA * e_x * x_cap * np.conj(y_vec)
            pyx += 2 * mu_CMA * e_y * y_cap * np.conj(x_vec)
            pyy += 2 * mu_CMA * e_y * y_cap * np.conj(y_vec)
    
    def conv_same(sig, taps):
        full = np.convolve(sig, taps, mode='full')
        start = (len(taps) - 1) // 2
        return full[start : start + len(sig)]

    x_out = conv_same(xpol, pxx) + conv_same(ypol, pxy)
    y_out = conv_same(xpol, pyx) + conv_same(ypol, pyy)

    return (
        np.column_stack((x_out, y_out)),
        {
            'pxx': pxx, 'pxy': pxy, 'pyx': pyx, 'pyy': pyy,
            'cma_diff_arr': difference_cma_errors,
            'convergence_symbol': convergence_symbol,
            'log_cma_errors': log_cma_errors
        }
    )


def cma_python_with_filter_convergence_index(E_in, num_taps, mu_CMA=0.01, error_threshold=1e-3,Num_loops = 1,num_symbols_store = 2000):
    """
    CMA with moving-average convergence detection (last 10 symbols)
    In order to calculate the error threshold I store the last (lets say 2000) N symbols.
    I now calculate avg error over last N symbols and compare the means of N/2 start and N/2 end
    """

    # ---- Copy and normalize ----
    xpol = E_in[:, 0].astype(complex)
    ypol = E_in[:, 1].astype(complex)

    xpol = xpol / np.sqrt(np.mean(np.abs(xpol)**2))
    ypol = ypol / np.sqrt(np.mean(np.abs(ypol)**2))


    N = len(xpol)
    R = 1

    # ---- Tap initialization ----
    pxx = np.zeros(num_taps, dtype=complex)
    pxy = np.zeros(num_taps, dtype=complex)
    pyx = np.zeros(num_taps, dtype=complex)
    pyy = np.zeros(num_taps, dtype=complex)

    center = (num_taps - 1) // 2
    pxx[center] = 1
    pyy[center] = 1

    filter_vector_magnitude = []
    convergence_symbol = None

    #I am storing the last 
    old_filter_window = deque(maxlen=num_symbols_store//2)   # last 10 CMA errors
    recent_filter_window = deque(maxlen=num_symbols_store//2)   # last 10 CMA errors
    #this counts how long we have relative error to be less than threshold

    initial_filters = np.concat([pxx, pxy, pyx, pyy])

    #print("initial_filters",initial_filters)
    #print("shape is",initial_filters.shape)
    patience_counter = 0
    half_win = num_symbols_store//2
    sum_old = np.zeros(len(initial_filters),dtype=np.complex128)
    sum_recent = np.zeros(len(initial_filters),dtype=np.complex128)
    mse_bw_before_and_after_avg_filter_vectors = []
    cma_error = []


    for k in range(Num_loops):
        for ii in range(num_taps - 1, N):

            x_vec = xpol[ii - (num_taps - 1): ii + 1][::-1]
            y_vec = ypol[ii - (num_taps - 1): ii + 1][::-1]

            x_cap = np.dot(pxx, x_vec) + np.dot(pxy, y_vec)
            y_cap = np.dot(pyx, x_vec) + np.dot(pyy, y_vec)

            e_x = R**2 - np.abs(x_cap)**2
            e_y = R**2 - np.abs(y_cap)**2

            cma_error.append(0.5*(np.abs(e_x) + np.abs(e_y)))
            filter_vector = np.concat([pxx, pxy, pyx, pyy])
            #Add the magnitude of the filter vector which I define as sum of mod square of its elements
            filter_vector_magnitude.append(np.vdot(filter_vector,filter_vector).real)
            
            if len(recent_filter_window) == half_win:
                leaving_recent = recent_filter_window[0]
                
                # Move from recent to old
                if len(old_filter_window) == half_win:
                    sum_old -= old_filter_window[0] # Drop oldest from old sum
                
                old_filter_window.append(leaving_recent)
                sum_old += leaving_recent
                
                # Remove from recent sum
                sum_recent -= leaving_recent

            recent_filter_window.append(filter_vector)
            sum_recent += filter_vector

            # --- CONVERGENCE CHECK (Using O(1) Sums) ---
            m_rec = sum_recent / half_win
            m_old = sum_old / half_win
            
            # Relative error check
            diff_vector = m_rec - m_old
            mse_bw_before_and_after_avg_filter_vectors.append(np.vdot(diff_vector,diff_vector).real)
            
            if convergence_symbol is None and len(old_filter_window) == half_win:
                
                if np.vdot(diff_vector,diff_vector).real< error_threshold:
                    patience_counter += 1
                    if patience_counter >= 3:
                        convergence_symbol = ii - 3
                else:
                    patience_counter = 0


            # ---- Tap updates ----
            pxx += 2 * mu_CMA * e_x * x_cap * np.conj(x_vec)
            pxy += 2 * mu_CMA * e_x * x_cap * np.conj(y_vec)
            pyx += 2 * mu_CMA * e_y * y_cap * np.conj(x_vec)
            pyy += 2 * mu_CMA * e_y * y_cap * np.conj(y_vec)

    # ---- MATLAB conv(..., 'same') ----
    def conv_same(sig, taps):
        full = np.convolve(sig, taps, mode='full')
        start = (len(taps) - 1) // 2
        return full[start : start + len(sig)]

    x_out = conv_same(xpol, pxx) + conv_same(ypol, pxy)
    y_out = conv_same(xpol, pyx) + conv_same(ypol, pyy)

    return (
        np.column_stack((x_out, y_out)),
        {
            'pxx': pxx, 'pxy': pxy, 'pyx': pyx, 'pyy': pyy,
            'filter_vector_mag_arr': filter_vector_magnitude,
            'square_error_arr': mse_bw_before_and_after_avg_filter_vectors,
            'convergence_symbol': convergence_symbol,
            'cma_error': cma_error
        }
    )

def min_max_normalise(array):
    return (array - np.min(array)) / (np.max(array) - np.min(array))


def find_cma_convergence(data, short_w=500, long_w=5000, epsilon=0.005, patience=1000):
    """
    Optimized for high-noise convergence like CMA Error plots.
    """
    # 1. Calculate Moving Averages
    ma_short = np.convolve(data, np.ones(short_w)/short_w, mode='valid')
    ma_long = np.convolve(data, np.ones(long_w)/long_w, mode='valid')
    
    # 2. Align arrays
    offset = long_w - short_w
    diff = np.abs(ma_short[offset:] - ma_long)
    
    # 3. Patience Loop: Find where diff < epsilon for 'patience' consecutive steps
    low_diff_mask = diff < epsilon
    counter = 0
    
    for i, is_low in enumerate(low_diff_mask):
        if is_low:
            counter += 1
        else:
            counter = 0 # Reset if noise kicks it back up
            
        if counter >= patience:
            # Return the index where the 'stable' period began
            return i - patience + long_w 
            
    return None

def find_convergence_backward(error_signal, tail_size=50000, eval_window=1000, alpha=2.5):
    """
    Finds the convergence index by scanning backwards from the steady-state tail.
    
    Parameters:
    - error_signal: 1D numpy array of the CMA error (logged or raw).
    - tail_size: Number of samples at the end to define the 'constant' noise floor.
    - eval_window: Window size to smooth the signal during the backward scan.
    - alpha: Sensitivity multiplier (how many std deviations above the mean is a breakout).
    
    Returns:
    - index: The point where the signal converged.
    """
    
    # 1. Establish the steady-state baseline from the very end of the array
    tail_data = error_signal[-tail_size:]
    mu_ss = np.mean(tail_data)
    sigma_ss = np.std(tail_data)
    
    # Define the breakout threshold (mean + alpha * standard deviations)
    threshold = mu_ss + alpha * sigma_ss
    
    # 2. Smooth the signal to prevent single random spikes from triggering it early
    kernel = np.ones(eval_window) / eval_window
    smoothed_signal = np.convolve(error_signal, kernel, mode='valid')
    
    # 3. Search backwards from the end of the smoothed signal
    for i in range(len(smoothed_signal) - 1, -1, -1):
        if smoothed_signal[i] > threshold:
            # The signal has climbed out of the flat noise floor.
            # Add eval_window to account for the convolution offset.
            return i + eval_window
            
    # Returns 0 if no breakout is found
    return 0

# --- How to use it with your data ---
# Assuming 'cma_error' is your 1M length numpy array
# convergence_idx = find_convergence_backward(cma_error)
# print(f"Converged at symbol: {convergence_idx}")
def find_convergence_backward(error_signal, tail_size=50000, eval_window=1000, alpha=2.5):
    """
    Finds the convergence index by scanning backwards from the steady-state tail.
    
    Parameters:
    - error_signal: 1D numpy array of the CMA error (logged or raw).
    - tail_size: Number of samples at the end to define the 'constant' noise floor.
    - eval_window: Window size to smooth the signal during the backward scan.
    - alpha: Sensitivity multiplier (how many std deviations above the mean is a breakout).
    
    Returns:
    - index: The point where the signal converged.
    """
    
    # 1. Establish the steady-state baseline from the very end of the array
    tail_data = error_signal[-tail_size:]
    mu_ss = np.mean(tail_data)
    sigma_ss = np.std(tail_data)
    
    # Define the breakout threshold (mean + alpha * standard deviations)
    threshold = mu_ss + alpha * sigma_ss
    
    # 2. Smooth the signal to prevent single random spikes from triggering it early
    kernel = np.ones(eval_window) / eval_window
    smoothed_signal = np.convolve(error_signal, kernel, mode='valid')
    
    # 3. Search backwards from the end of the smoothed signal
    for i in range(len(smoothed_signal) - 1, -1, -1):
        if smoothed_signal[i] > threshold:
            # The signal has climbed out of the flat noise floor.
            # Add eval_window to account for the convolution offset.
            return i + eval_window
            
    # Returns 0 if no breakout is found
    return 0

# --- How to use it with your data ---
# Assuming 'cma_error' is your 1M length numpy array
# convergence_idx = find_convergence_backward(cma_error)
# print(f"Converged at symbol: {convergence_idx}")
def main():
    intial_symbols=cma_utils.gen_I_Q_16_qam(N_SYMBOLS)
    E_after_pmd= cma_utils.apply_pmd(intial_symbols, DGD_ps_per_sqrt_km=31.6, L_m=10000, N_sections=100, Rs=32e9, SpS=4)
    
    # cma_corrected_pmd, info_dict = cma_utils.cma_python(after_pmd, NUM_TAPS,mu_CMA=1e-5,Num_loops=1)

    # cma_utils.save_constellation(cma_corrected_pmd,"cma_corrected_pmd")
    # #print(info_dict["convergence_symbol"])
    
    

    #print("length of filter error array",len(info_dict['square_error_arr']))
    #plt.plot(min_max_normalise(info_dict['cma_error']),alpha =0.5)
    #plt.plot(range(len(info_dict["cma_error"])),info_dict["cma_error"])
    #plt.plot(np.log(info_dict["relative_error_arr"]))
    #smoothed_log_errors = plot_conv(info_dict['cma_error'])

    #print("convergence symbols is",find_cma_convergence(smoothed_log_errors))
    snr_list = [5, 15, 25, 35, 45]
    snr_vs_conv = {}
    SNR_arr = []
    converged_samples_arr = []
    for snr_added in snr_list:
        E_noisy = cma_utils.apply_awgn(E_after_pmd, snr_added)
        E_out, stats = cma_utils.mcma_python(E_noisy,NUM_TAPS,mu_CMA=1e-5)
        pxx, pxy, pyx, pyy, cma_error= stats['pxx'], stats['pyx'], stats['pyx'], stats['pyy'], stats['cma_error']
        cma_utils.save_constellation(E_out,"mcma_corrected_"+str(snr_added))
        plt.close('all')
        smoothed_log_errors = plot_conv(cma_error,str(snr_added))
        #converged_sample = find_cma_convergence(smoothed_log_errors)
        converged_sample_stats= find_convergence_backward(smoothed_log_errors)
        print("converged sample backward detection is",converged_sample_stats) 
        smoothed_log_errors = plot_conv(cma_error,str(snr_added))
        #converged_sample = find_cma_convergence(smoothed_log_errors)
        converged_sample_stats= find_convergence_backward(smoothed_log_errors)
        print("converged sample backward detection is",converged_sample_stats) 
        #snr_vs_conv[snr_added] = convergence_symbol
        snr_x,snr_y = cma_utils.cluster_and_get_avg_snr(E_out, 16)
        print("SNR is", snr_x)

        SNR_arr.append(snr_x)
        #print("SNR is", cma_utils.cluster_and_get_avg_snr(E_out, 16)
        converged_samples_arr.append(converged_sample_stats)
        converged_samples_arr.append(converged_sample_stats)
    

    plt.plot(snr_list,converged_samples_arr)
    plt.xlabel("SNR(dB)")
    plt.ylabel("Converged Samples")
    plt.savefig("converged_samples.png")
    #plt.show()

def plot_convergence_vs_num_taps():
    intial_symbols=cma_utils.gen_I_Q_16_qam(N_SYMBOLS)
    
    converged_samples_arr = []
    NUM_AVG = 10

    for num_taps in range(11,301,20):
        converged_sample_sum = 0
        for iter in range(NUM_AVG):
            E_after_pmd= cma_utils.apply_pmd(intial_symbols, DGD_ps_per_sqrt_km=31.6, L_m=10000, N_sections=100, Rs=32e9, SpS=4)
            E_out, stats = cma_utils.mma_python(E_after_pmd,num_taps,mu_CMA=1e-3)
            pxx, pxy, pyx, pyy, cma_error= stats['pxx'], stats['pyx'], stats['pyx'], stats['pyy'], stats['cma_error']
            #cma_utils.save_constellation(E_out,"mcma_corrected_taps_"+str(num_taps))
            #plt.close('all')
            smoothed_log_errors = plot_conv(cma_error,"taps_"+str(num_taps))
            converged_symbol = find_convergence_backward(smoothed_log_errors)
            converged_sample_sum +=converged_symbol
            print("Finished iteration",iter , "converged sample is",converged_symbol)
        print("Done num taps",num_taps)
        converged_samples_arr.append(converged_sample_sum/NUM_AVG)
        
    plt.plot(range(11,201,10),converged_samples_arr)
    plt.xlabel("Number of taps")
    plt.ylabel("Converged Samples")
    plt.savefig("converged_samples_vs_num_taps_mma.png")
    #plt.show()

def plot_convergence_vs_DGD():
    num_taps = 151
    intial_symbols=cma_utils.gen_I_Q_16_qam(N_SYMBOLS)
    
    converged_samples_arr = []
    NUM_AVG = 10
    DGD_arr = []
    for DGD in range(2,100,10):
        converged_sample_sum = 0
        for iter in range(NUM_AVG):
            E_after_pmd= cma_utils.apply_pmd(intial_symbols, DGD_ps_per_sqrt_km=DGD, L_m=10000, N_sections=100, Rs=32e9, SpS=4)
            E_out, stats = cma_utils.cma_python(E_after_pmd,num_taps,mu_CMA=1e-5)
            pxx, pxy, pyx, pyy, cma_error= stats['pxx'], stats['pyx'], stats['pyx'], stats['pyy'], stats['cma_error']
            #cma_utils.save_constellation(E_out,"mcma_corrected_taps_"+str(num_taps))
            #plt.close('all')
            smoothed_log_errors = plot_conv(cma_error,"taps_"+str(num_taps))
            converged_symbol = find_convergence_backward(smoothed_log_errors)
            converged_sample_sum += converged_symbol
            print("Finished iteration",iter,"converged sample is",converged_symbol)
        print("Done num taps",num_taps)
        converged_samples_arr.append(converged_sample_sum/NUM_AVG)

    plt.plot(range(2,100,10),converged_samples_arr)
    plt.xlabel("DGD")
    plt.ylabel("Converged Samples")
    plt.savefig("converged_samples_vs_DGD.png")


if __name__ == "__main__": plot_convergence_vs_num_taps()