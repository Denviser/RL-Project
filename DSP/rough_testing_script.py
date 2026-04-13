import cma_utils
import numpy as np

N_SYMBOLS = 1000000
N_TAPS = 4
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

    symbols_with_pmd_corrected_mma,info_dict_mcma = cma_utils.cma_python(E_after_pmd,N_TAPS,mu_CMA=1e-4)
    cma_utils.save_constellation(symbols_with_pmd_corrected_mma,"mma_corrected")
if __name__ == "__main__":
    main()