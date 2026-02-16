import numpy as np
import matplotlib.pyplot as plt
import cma_utils



N_SYMBOLS = 100000
fs = 2e9
freq_offset = 5e6

def main():

    
    initial_symbols=cma_utils.gen_I_Q_16_qam(N_SYMBOLS)
    #print(initial_symbols)
    
    #Uncomment bottom two lines for cfo then pmd
    # symbols_with_cfo = cma_utils.apply_cfo_both_polarisations(initial_symbols,freq_offset,fs)
    # symbols_with_cfo_and_pmd=cma_utils.apply_pmd(symbols_with_cfo)

    cma_utils.plot_constellation(initial_symbols)

    #Uncommment bottom two lines for pms then cfo
    symbols_with_pmd = cma_utils.apply_pmd( 
    E_in=initial_symbols, 
    # DGD_ps_per_sqrt_km=31.6, # typical mean DGD 
    DGD_ps_per_sqrt_km=31.6,
    L_m=10000, # fiber length
    N_sections=20, # number of fiber sections 
    Rs=32e9, # 32 gbaud
    SpS=2 # samples per symbol 
    )
    
    
    
    symbols_with_cfo_and_pmd=cma_utils.apply_cfo_both_polarisations(symbols_with_pmd,freq_offset,fs)

    #symbols_with_only_cfo = cma_utils.apply_cfo_both_polarisations(initial_symbols,freq_offset,fs)
    #cma_utils.plot_constellation(symbols_with_only_cfo)
    #Initial plot with both distortion
    #cma_utils.plot_constellation(symbols_with_cfo_and_pmd)

    #Uncomment this for doing pmd first then cfo
    symbols_with_pmd_corrrected,converged_filters =  cma_utils.cma_python(symbols_with_cfo_and_pmd,num_taps=51,mu_CMA=1e-4)

    cma_utils.plot_constellation(symbols_with_pmd_corrrected)

    symbols_with_both_corrected = cma_utils.cfo_correction_both_pol(symbols_with_pmd_corrrected,fs)

    cma_utils.plot_constellation(symbols_with_both_corrected)
    
    
    # symbols_with_cfo_correction=cma_utils.cfo_correction_both_pol(symbols_with_cfo_and_pmd,fs)
    # cma_utils.plot_constellation(symbols_with_cfo_correction)

    # symbols_with_pmd_correction,converged_filters = cma_utils.mcma_python(symbols_with_cfo_correction,num_taps=51,mu_CMA=1e-4)

    # cma_utils.plot_constellation(symbols_with_pmd_correction)


main()