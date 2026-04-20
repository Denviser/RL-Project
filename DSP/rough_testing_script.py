import cma_utils
import numpy as np
from scipy.linalg import expm
import matplotlib.pyplot as plt

N_SYMBOLS = 100000
N_TAPS = 151

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

def apply_crosstalk(E_in_1, E_in_2, DGD_ps_per_sqrt_km=1.0, L_m=10000, N_sections=20, Rs=32e9, SpS=2 , seed = 0):
    """This function does crosstalk between the x and y polarisation and 
    returns back the x and y polarisation after crosstalk
    
    L_m - fiber length
    N_sections - number of fiber sections
    Rs - sampling rate
    Sps - samples per symbol
    """
    "TODO: for now we are doing 2 cores and also I have assumed that the DGD is same for both cores, we can change this in future. I have assumed no delays between x polarisation of both cores"

    N_samples = E_in_1.shape[0]
    SD_tau = np.sqrt(3 * np.pi / 8) * DGD_ps_per_sqrt_km

    tau = (SD_tau * np.sqrt(L_m * 1e-3) / np.sqrt(N_sections)) * 1e-12
    print("tau_Ts",tau*Rs)
    w = 2 * np.pi * np.fft.fftshift(np.linspace(-0.5, 0.5, N_samples)) * Rs

    E_V_1 = np.fft.fft(E_in_1[:, 0])
    E_H_1 = np.fft.fft(E_in_1[:, 1])
    E_V_2 = np.fft.fft(E_in_2[:, 0])
    E_H_2 = np.fft.fft(E_in_2[:, 1])

    if seed:
        np.random.seed(42)
    else:
        pass

    for _ in range(N_sections):
        # Random complex coupling matrices (unitary)
        X = np.random.randn(4, 4) + 1j * np.random.randn(4, 4)
        U, _, Vh = np.linalg.svd(X)
        V = Vh.conj().T
        # Rotate fields by U†
        E1 = U[0, 0].conj() * E_V_1 + U[0, 1].conj() * E_H_1 + U[0, 2].conj() * E_V_2 + U[0, 3].conj() * E_H_2
        E2 = U[1, 0].conj() * E_V_1 + U[1, 1].conj() * E_H_1 + U[1, 2].conj() * E_V_2 + U[1, 3].conj() * E_H_2
        E3 = U[2, 0].conj() * E_V_1 + U[2, 1].conj() * E_H_1 + U[2, 2].conj() * E_V_2 + U[2, 3].conj() * E_H_2
        E4 = U[3, 0].conj() * E_V_1 + U[3, 1].conj() * E_H_1 + U[3, 2].conj() * E_V_2 + U[3, 3].conj() * E_H_2

        # Apply differential delay
        E1 *= np.exp(1j * w * tau / 2)
        E2 *= np.exp(-1j * w * tau / 2)
        E3 *= np.exp(1j * w * tau / 2)
        E4 *= np.exp(-1j * w * tau / 2)

        # Rotate by V
        E_V_1 = V[0, 0] * E1 + V[0, 1] * E2 + V[0, 2] * E3 + V[0, 3] * E4
        E_H_1 = V[1, 0] * E1 + V[1, 1] * E2 + V[1, 2] * E3 + V[1, 3] * E4
        E_V_2 = V[2, 0] * E1 + V[2, 1] * E2 + V[2, 2] * E3 + V[2, 3] * E4
        E_H_2 = V[3, 0] * E1 + V[3, 1] * E2 + V[3, 2] * E3 + V[3, 3] * E4
       

    E_out_1_x = np.fft.ifft(E_V_1)
    E_out_1_y = np.fft.ifft(E_H_1)
    E_out_2_x = np.fft.ifft(E_V_2)
    E_out_2_y = np.fft.ifft(E_H_2)

    return np.column_stack((E_out_1_x, E_out_1_y)), np.column_stack((E_out_2_x, E_out_2_y))

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
        DGD_ps_per_sqrt_km=10,
        L_m=10000, # fiber length
        N_sections=100, # number of fiber sections 
        Rs=32e9, # 32 gbaud
        SpS=4 # samples per symbol
    )

    symbols_with_pmd_corrected_mma,info_dict_mcma = cma_utils.mma_python(E_after_pmd,N_TAPS,mu_CMA=1e-4)
    cma_utils.save_constellation(symbols_with_pmd_corrected_mma,"mma_corrected")
    smoothed_log_errors_mma = cma_utils.plot_conv(info_dict_mcma['cma_error'])
    converged_symbol_mma = cma_utils.find_convergence_backward(smoothed_log_errors_mma)

    symbols_with_pms_corrected_cma,info_dict_cma = cma_utils.cma_python(E_after_pmd,N_TAPS,mu_CMA=1e-4)
    cma_utils.save_constellation(symbols_with_pms_corrected_cma,"cma_corrected")
    smoothed_log_errors_cma = cma_utils.plot_conv(info_dict_cma['cma_error'])
    converged_symbol_cma = cma_utils.find_convergence_backward(smoothed_log_errors_cma)

    print("Converged symbol MMA:", converged_symbol_mma)
    print("EVM of MMA:",cma_utils.cluster_and_get_avg_snr(symbols_with_pmd_corrected_mma))

    print("Converged symbol CMA:", converged_symbol_cma)
    print("EVM of CMA:",cma_utils.cluster_and_get_avg_snr(symbols_with_pms_corrected_cma))

def normalise(E):
    E = E.astype(complex)
    E[:,0] /= np.sqrt(np.mean(np.abs(E[:,0])**2))
    E[:,1] /= np.sqrt(np.mean(np.abs(E[:,1])**2))
    return E

def initialise_filters_crosstalk(num_taps):
    Filter_matrix = np.zeros((4,4,num_taps),dtype=complex)
    for i in range(4):
        Filter_matrix[i,i,num_taps//2] = 1 + 0j
    return Filter_matrix

def cma_crosstalk(E_in_core1,E_in_core2,num_taps,mu_CMA):
    E_norm_core1 = normalise(E_in_core1)
    E_norm_core2 = normalise(E_in_core2)

    x_pol_core1 = E_norm_core1[:,0]
    y_pol_core1 = E_norm_core1[:,1]
    x_pol_core2 = E_norm_core2[:,0]
    y_pol_core2 = E_norm_core2[:,1]
    
    N=len(x_pol_core1)
    R=1
    filter_matrix = initialise_filters_crosstalk(num_taps)

    cma_error = []

    for ii in range(num_taps-1,N):
        x_vec_core1 = x_pol_core1[ii-num_taps+1:ii+1][::-1]
        y_vec_core1 = y_pol_core1[ii-num_taps+1:ii+1][::-1]
        x_vec_core2 = x_pol_core2[ii-num_taps+1:ii+1][::-1]
        y_vec_core2 = y_pol_core2[ii-num_taps+1:ii+1][::-1]

        x_cap_core1 = np.dot(filter_matrix[0,0],x_vec_core1) + np.dot(filter_matrix[0,1],y_vec_core1) + np.dot(filter_matrix[0,2],x_vec_core2) + np.dot(filter_matrix[0,3],y_vec_core2)
        y_cap_core1 = np.dot(filter_matrix[1,0],x_vec_core1) + np.dot(filter_matrix[1,1],y_vec_core1) + np.dot(filter_matrix[1,2],x_vec_core2) + np.dot(filter_matrix[1,3],y_vec_core2)
        x_cap_core2 = np.dot(filter_matrix[2,0],x_vec_core1) + np.dot(filter_matrix[2,1],y_vec_core1) + np.dot(filter_matrix[2,2],x_vec_core2) + np.dot(filter_matrix[2,3],y_vec_core2)
        y_cap_core2 = np.dot(filter_matrix[3,0],x_vec_core1) + np.dot(filter_matrix[3,1],y_vec_core1) + np.dot(filter_matrix[3,2],x_vec_core2) + np.dot(filter_matrix[3,3],y_vec_core2)

        e_x_core1 = -np.abs(x_cap_core1)**2 +R**2
        e_y_core1 = -np.abs(y_cap_core1)**2 +R**2
        e_x_core2 = -np.abs(x_cap_core2)**2 +R**2
        e_y_core2 = -np.abs(y_cap_core2)**2 +R**2

        e_cma = 0.25*(abs(e_x_core1) + abs(e_y_core1) + abs(e_x_core2) + abs(e_y_core2) )
        cma_error.append(e_cma)

        input_vec = np.array([x_vec_core1, y_vec_core1, x_vec_core2, y_vec_core2])
        cap_vec = np.array([x_cap_core1, y_cap_core1, x_cap_core2, y_cap_core2])
        error_vec = np.array([e_x_core1, e_y_core1, e_x_core2, e_y_core2])
        for row in range(4):
            filter_matrix[row,0] += mu_CMA *cap_vec[row]*error_vec[row] * np.conj(input_vec[0])
            filter_matrix[row,1] += mu_CMA *cap_vec[row]*error_vec[row] * np.conj(input_vec[1])
            filter_matrix[row,2] += mu_CMA *cap_vec[row]*error_vec[row] * np.conj(input_vec[2])
            filter_matrix[row,3] += mu_CMA *cap_vec[row]*error_vec[row] * np.conj(input_vec[3])

    
    x_out_core1 = np.convolve(x_pol_core1, filter_matrix[0,0], mode='same') + np.convolve(y_pol_core1, filter_matrix[0,1], mode='same') + np.convolve(x_pol_core2, filter_matrix[0,2], mode='same') + np.convolve(y_pol_core2, filter_matrix[0,3], mode='same')
    y_out_core1 = np.convolve(x_pol_core1, filter_matrix[1,0], mode='same') + np.convolve(y_pol_core1, filter_matrix[1,1], mode='same') + np.convolve(x_pol_core2, filter_matrix[1,2], mode='same') + np.convolve(y_pol_core2, filter_matrix[1,3], mode='same')
    x_out_core2 = np.convolve(x_pol_core1, filter_matrix[2,0], mode='same') + np.convolve(y_pol_core1, filter_matrix[2,1], mode='same') + np.convolve(x_pol_core2, filter_matrix[2,2], mode='same') + np.convolve(y_pol_core2, filter_matrix[2,3], mode='same')
    y_out_core2 = np.convolve(x_pol_core1, filter_matrix[3,0], mode='same') + np.convolve(y_pol_core1, filter_matrix[3,1], mode='same') + np.convolve(x_pol_core2, filter_matrix[3,2], mode='same') + np.convolve(y_pol_core2, filter_matrix[3,3], mode='same')

    E_out_core1 = np.column_stack((x_out_core1, y_out_core1))
    E_out_core2 = np.column_stack((x_out_core2, y_out_core2))

    return E_out_core1, E_out_core2, filter_matrix, cma_error

def test_crosstalk():
    initial_symbols_core1 = cma_utils.gen_I_Q_16_qam(N_SYMBOLS)
    initial_symbols_core2 = cma_utils.gen_I_Q_16_qam(N_SYMBOLS)

    output_symbols_core1, output_symbols_core2 = apply_crosstalk(initial_symbols_core1, initial_symbols_core2,DGD_ps_per_sqrt_km=0)
    cma_utils.save_constellation(output_symbols_core1,"core1_after_crosstalk")
    cma_utils.save_constellation(output_symbols_core2,"core2_after_crosstalk")
    cma_corrected_core1, cma_corrected_core2, filter_matrix, cma_error = cma_crosstalk(output_symbols_core1,output_symbols_core2,num_taps=151,mu_CMA=2e-4)
    cma_utils.save_constellation(cma_corrected_core1,"core1_after_crosstalk_cma_corrected")
    cma_utils.save_constellation(cma_corrected_core2,"core2_after_crosstalk_cma_corrected")
if __name__ == "__main__":
    test_crosstalk()