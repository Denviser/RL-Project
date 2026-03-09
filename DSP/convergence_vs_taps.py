import numpy as np
import matplotlib.pyplot as plt
import cma_utils
from collections import deque

N_SYMBOLS = 1000000
NUM_TAPS=51



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

            cma_error.append(0.5*(e_x + e_y))
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

def main():
    intial_symbols=cma_utils.gen_I_Q_16_qam(N_SYMBOLS)
    after_pmd= cma_utils.apply_pmd(intial_symbols, DGD_ps_per_sqrt_km=31.6, L_m=10000, N_sections=100, Rs=32e9, SpS=4)
    
    cma_corrected_pmd, info_dict = cma_python_with_filter_convergence_index(after_pmd, NUM_TAPS,mu_CMA=1e-4)

    cma_utils.save_constellation(cma_corrected_pmd,"cma_corrected_pmd")
    #print(info_dict["convergence_symbol"])
    
    
    print("convergence symbol is",info_dict["convergence_symbol"])

    print("length of filter error array",len(info_dict['square_error_arr']))
    plt.plot(np.log10(info_dict['square_error_arr']))
    #plt.plot(min_max_normalise(info_dict['cma_error']),alpha =0.5)
    plt.show()
    #plt.plot(range(len(info_dict["cma_error"])),info_dict["cma_error"])
    #plt.plot(np.log(info_dict["relative_error_arr"]))

    #plt.show()
if __name__ == "__main__": main()