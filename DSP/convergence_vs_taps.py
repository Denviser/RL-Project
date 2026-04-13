import numpy as np
import matplotlib.pyplot as plt
import cma_utils
from collections import deque
from scipy import stats
from scipy import stats

N_SYMBOLS = 100000
NUM_TAPS=4

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
            E_out, stats = cma_utils.mma_python(E_after_pmd,num_taps,mu_CMA=2e-4)
            pxx, pxy, pyx, pyy, cma_error= stats['pxx'], stats['pyx'], stats['pyx'], stats['pyy'], stats['cma_error']
            cma_utils.save_constellation(E_out,"mma_corrected_taps_"+str(num_taps))
            plt.close('all')
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