import numpy as np
import matplotlib.pyplot as plt
import cma_utils
from collections import deque

N_SYMBOLS = 1000000
NUM_TAPS=51

def plot_conv(cma_error):

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
    plt.savefig("cma_error_smoothed.png")

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
        smoothed_log_errors = plot_conv(cma_error)
        converged_sample = find_cma_convergence(smoothed_log_errors)
        print("converged sample is",converged_sample) 
        #snr_vs_conv[snr_added] = convergence_symbol
        snr_x,snr_y = cma_utils.cluster_and_get_avg_snr(E_out, 16)
        print("SNR is", snr_x)

        SNR_arr.append(snr_x)
        #print("SNR is", cma_utils.cluster_and_get_avg_snr(E_out, 16)
        converged_samples_arr.append(converged_sample)
    

    plt.plot(snr_list,converged_samples_arr)
    plt.xlabel("SNR(dB)")
    plt.ylabel("Converged Samples")
    plt.savefig("converged_samples.png")
    #plt.show()
if __name__ == "__main__": main()