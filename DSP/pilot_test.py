import cma_utils_pilot
import cma_utils
import numpy as np

def lms_test(E_distorted,pilot_sequence):
    #print("E_in shape:", E_in.shape)
    pass

def main():
    N_symbols = 10000
    E_in = cma_utils_pilot.generate_stream(N_symbols, offset=200)

    E_after_pmd = cma_utils.apply_pmd(E_in,DGD_ps_per_sqrt_km=20)



if __name__ == "__main__":    main()