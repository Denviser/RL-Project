import cma_utils
import numpy as np
from scipy.linalg import expm
import matplotlib.pyplot as plt

N_SYMBOLS = 1000000
N_TAPS = 151

def normalise(E):
    E = E.astype(complex)
    E[:,0] /= np.sqrt(np.mean(np.abs(E[:,0])**2))
    E[:,1] /= np.sqrt(np.mean(np.abs(E[:,1])**2))
    return E


def conv_same(sig, taps):
    full = np.convolve(sig, taps, mode='full')
    start = (len(taps) - 1) // 2
    return full[start : start + len(sig)]

def initialise_filters(NUM_TAPS):
    filters={}
    filters['pxx'] = np.zeros(NUM_TAPS, dtype = complex)
    filters['pxx'][NUM_TAPS//2] = 1
    filters['pxy'] = np.zeros(NUM_TAPS, dtype = complex)
    filters['pyx'] = np.zeros(NUM_TAPS, dtype = complex)
    filters['pyy'] = np.zeros(NUM_TAPS, dtype = complex)
    filters['pyy'][NUM_TAPS//2] = 1
    return filters

def mma_python(E_in, num_taps, mu_CMA=0.01, Radius_options = (1/np.sqrt(10))*np.array([np.sqrt(2),np.sqrt(10),np.sqrt(18)])):
    """
    CMA with moving-average convergence detection (last 10 symbols)
    """
    def get_nearest_radius(x):
        decision_radiuses = [Radius_options[0] + 1/3*(Radius_options[1]-Radius_options[0]), Radius_options[1] + 2/3*(Radius_options[2]-Radius_options[1])]
        if np.abs(x) < decision_radiuses[0]:
            return Radius_options[0]
        elif np.abs(x) < decision_radiuses[1]:
            return Radius_options[1]
        else:
            return Radius_options[2]
    E_norm = normalise(E_in.copy())

    xpol = E_norm[:, 0]
    ypol = E_norm[:, 1]
    N = len(xpol)

    #Setting this for unit energy 16 QAM for now

    filters = initialise_filters(num_taps)
    pxx, pyy, pxy, pyx = filters['pxx'], filters['pyy'], filters['pxy'], filters['pyx']

    cma_error = []

    for ii in range(num_taps - 1, N):

        x_vec = xpol[ii - (num_taps - 1): ii + 1][::-1]
        y_vec = ypol[ii - (num_taps - 1): ii + 1][::-1]

        x_cap = np.dot(pxx, x_vec) + np.dot(pxy, y_vec)
        y_cap = np.dot(pyx, x_vec) + np.dot(pyy, y_vec)

        nearest_radius_x = get_nearest_radius(x_cap)
        nearest_radius_y = get_nearest_radius(y_cap)

        #print(nearest_radius_x)
        
        e_x = nearest_radius_x**2 - np.abs(x_cap)**2
        e_y = nearest_radius_y**2 - np.abs(y_cap)**2

        e_cma = 0.5 * (np.abs(e_x) + np.abs(e_y))
        cma_error.append(e_cma)

        # ---- Tap updates ----
        pxx += 2 * mu_CMA * e_x * x_cap * np.conj(x_vec)
        pxy += 2 * mu_CMA * e_x * x_cap * np.conj(y_vec)
        pyx += 2 * mu_CMA * e_y * y_cap * np.conj(x_vec)
        pyy += 2 * mu_CMA * e_y * y_cap * np.conj(y_vec)

    x_out = conv_same(xpol, pxx) + conv_same(ypol, pxy)
    y_out = conv_same(xpol, pyx) + conv_same(ypol, pyy)

    return (
        np.column_stack((x_out, y_out)),
        {'pxx': pxx, 'pxy': pxy, 'pyx': pyx, 'pyy': pyy,'cma_error': cma_error}
    )


def generate_4_4_unitary_complex_matrix():

    # 1. Define a Hermitian Generator Matrix (K)
    # This represents the coupling rates.
    b1, b2 = 1.5, 1.5      # Propagation constants
    k_xt = 0.3 + 0.1j      # Inter-core cross-talk rate
    pmd  = 0.05 + 0.02j    # Intra-core polarization mixing rate

    K = np.array([
        [b1,            pmd,           k_xt,          0],
        [np.conj(pmd),  b1,            0,             k_xt],
        [np.conj(k_xt), 0,             b2,            pmd],
        [0,             np.conj(k_xt), np.conj(pmd),  b2]
    ], dtype=complex)

    # Verify generator is Hermitian
    assert np.allclose(K, K.conj().T), "K must be Hermitian!"

    # 2. Generate the Unitary Matrix (H) using the matrix exponential
    # The -1j factor ensures the result is Unitary
    H = expm(-1j * K)

    # 3. Extract the sub-blocks
    A = H[0:2, 0:2]  # Core 1 transmission
    C = H[0:2, 2:4]  # Cross-talk from Core 2 to 1
    D = H[2:4, 0:2]  # Cross-talk from Core 1 to 2
    B = H[2:4, 2:4]  # Core 2 transmission

    return H

def cross_talk(E_in_core1, E_in_core2, H):
    E_in_core1_x = E_in_core1[:, 0]
    E_in_core1_x_freq = np.fft.fft(E_in_core1_x) 
    E_in_core1_y = E_in_core1[:, 1]
    E_in_core1_y_freq = np.fft.fft(E_in_core1_y)
    E_in_core2_x = E_in_core2[:, 0]
    E_in_core2_x_freq = np.fft.fft(E_in_core2_x)
    E_in_core2_y = E_in_core2[:, 1]
    E_in_core2_y_freq = np.fft.fft(E_in_core2_y)

    E_out_core1_x_freq = H[0,0]*E_in_core1_x_freq + H[0,1]*E_in_core1_y_freq + H[0,2]*E_in_core2_x_freq + H[0,3]*E_in_core2_y_freq
    E_out_core1_y_freq = H[1,0]*E_in_core1_x_freq + H[1,1]*E_in_core1_y_freq + H[1,2]*E_in_core2_x_freq + H[1,3]*E_in_core2_y_freq
    E_out_core2_x_freq = H[2,0]*E_in_core1_x_freq + H[2,1]*E_in_core1_y_freq + H[2,2]*E_in_core2_x_freq + H[2,3]*E_in_core2_y_freq
    E_out_core2_y_freq = H[3,0]*E_in_core1_x_freq + H[3,1]*E_in_core1_y_freq + H[3,2]*E_in_core2_x_freq + H[3,3]*E_in_core2_y_freq

    E_out_core1_x = np.fft.ifft(E_out_core1_x_freq)
    E_out_core1_y = np.fft.ifft(E_out_core1_y_freq)
    E_out_core2_x = np.fft.ifft(E_out_core2_x_freq)
    E_out_core2_y = np.fft.ifft(E_out_core2_y_freq)

    E_out_core1 = np.column_stack((E_out_core1_x, E_out_core1_y))
    E_out_core2 = np.column_stack((E_out_core2_x, E_out_core2_y))

    return E_out_core1, E_out_core2

def main():    
    initial_symbols=cma_utils.gen_I_Q_16_qam(N_SYMBOLS)
    E_after_pmd = cma_utils.apply_pmd(
        E_in=initial_symbols, 
        # DGD_ps_per_sqrt_km=31.6, # typical mean DGD 
        DGD_ps_per_sqrt_km=31.6,
        L_m=10000, # fiber length
        N_sections=100, # number of fiber sections 
        Rs=32e9, # 32 gbaud
        SpS=4 # samples per symbol
    )

    symbols_with_pmd_corrected_mma,info_dict_mcma = cma_utils.mma_python(E_after_pmd,N_TAPS,mu_CMA=1e-4)
    cma_utils.save_constellation(symbols_with_pmd_corrected_mma,"mma_corrected")
    smoothed_log_errors_mma = cma_utils.plot_conv(info_dict_mcma['cma_error'])
    plt.plot(smoothed_log_errors_mma)
    plt.show()
    plt.close()
    converged_symbol_mma = cma_utils.find_convergence_backward(smoothed_log_errors_mma)

    symbols_with_pms_corrected_cma,info_dict_cma = cma_utils.cma_python(E_after_pmd,N_TAPS,mu_CMA=1e-4)
    cma_utils.save_constellation(symbols_with_pms_corrected_cma,"cma_corrected")
    smoothed_log_errors_cma = cma_utils.plot_conv(info_dict_cma['cma_error'])
    converged_symbol_cma = cma_utils.find_convergence_backward(smoothed_log_errors_cma)

    print("Converged symbol MMA:", converged_symbol_mma)
    print("EVM of MMA:",cma_utils.cluster_and_get_avg_snr(symbols_with_pmd_corrected_mma))

    print("Converged symbol CMA:", converged_symbol_cma)
    print("EVM of CMA:",cma_utils.cluster_and_get_avg_snr(symbols_with_pms_corrected_cma))

if __name__ == "__main__":
    main()