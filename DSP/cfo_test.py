import numpy as np
import matplotlib.pyplot as plt
import cma_utils



N_SYMBOLS = 10000
fs = 2e9
freq_offset = 5e6

def main():

    
    initial_symbols=cma_utils.gen_I_Q_qpsk(N_SYMBOLS)
    #print(initial_symbols)
    
    #Uncomment bottom two lines for cfo then pmd
    # symbols_with_cfo = cma_utils.apply_cfo_both_polarisations(initial_symbols,freq_offset,fs)
    # symbols_with_cfo_and_pmd=cma_utils.apply_pmd(symbols_with_cfo)


    #Uncommment bottom two lines for pms then cfo
    symbols_with_pmd = cma_utils.apply_pmd(initial_symbols)
    symbols_with_cfo_and_pmd=cma_utils.apply_cfo_both_polarisations(symbols_with_pmd,freq_offset,fs)

    #Initial plot with both distortion
    cma_utils.plot_constellation(symbols_with_cfo_and_pmd)

    symbols_with_cfo_correction=cma_utils.cfo_correction_both_pol(symbols_with_cfo_and_pmd,fs)
    cma_utils.plot_constellation(symbols_with_cfo_correction)

    symbols_with_pmd_correction,converged_filters = cma_utils.mcma_python(symbols_with_cfo_correction,num_taps=51,mu_CMA=1e-4)

    cma_utils.plot_constellation(symbols_with_pmd_correction)


main()