import numpy as np
import cma_utils
import pandas as pd
import matplotlib.pyplot as plt

N_SYMBOLS = 100000
NUM_TAPS=51

def plot_cost_vs_symbols_log(E_after_pmd, num_taps,
                             R=1,
                             maxiter=200):

    # Log-spaced symbol counts from 1e2 to 1e6
    symbol_list = np.unique(
        np.logspace(2, 6, 25).astype(int)
    )

    symbol_list = [5000, 10000, 15000, 20000, 25000, 30000, 35000, 40000, 45000, 50000, 55000, 60000, 65000, 70000, 75000, 80000, 85000, 90000, 95000]

    final_costs = []

    for Nsym in symbol_list:

        print(f"LBFGS CMA: optimizing with N = {Nsym}")

        # Take block of Nsym symbols
        E_block = E_after_pmd[:Nsym, :]

        # Run LBFGS CMA
        E_out, avg_cost, last_symbol_error, pxx, pxy, pyx, pyy = cma_utils.cma_lbfgs(
            E_block,
            num_taps=num_taps,
            R=R,
            maxiter=maxiter
        )
        print(avg_cost)

        final_costs.append(avg_cost)

        

    # Plot
    plt.figure()
    plt.plot(symbol_list, final_costs, marker="o")
    plt.xscale("log")
    plt.xlabel("Number of Symbols (log scale)")
    plt.ylabel("Final CMA Cost After LBFGS")
    plt.title("Final CMA Cost vs Symbols Used")
    plt.grid(True)
    plt.savefig("final_cost_vs_symbols.png")

    return symbol_list, final_costs


def main():
    initial_symbols=cma_utils.gen_I_Q_16_qam(N_SYMBOLS)
    
    E_after_pmd = cma_utils.apply_pmd( 
    E_in=initial_symbols, 
    # DGD_ps_per_sqrt_km=31.6, # typical mean DGD 
    DGD_ps_per_sqrt_km=19.764,
    L_m=10000, # fiber length
    N_sections=20, # number of fiber sections 
    Rs=32e9, # 32 gbaud
    SpS=4 # samples per symbol 
    )

    # #Doing LBFGS
    # symbol_list, final_costs = plot_cost_vs_symbols_log(E_after_pmd, NUM_TAPS, R=1, maxiter=200)

    # Doing Mcma
    symbols_with_pmd_corrected_mcma,info_dict_mcma = cma_utils.mcma_python(E_after_pmd,NUM_TAPS,mu_CMA=1e-4)

    cma_utils.save_constellation(symbols_with_pmd_corrected_mcma,save_path_prefix="mcma_corrected")

    symbols_with_pmd_corrected_cma,info_dict_cma = cma_utils.cma_python(E_after_pmd,NUM_TAPS,mu_CMA=1e-4)

    cma_utils.save_constellation(symbols_with_pmd_corrected_cma,save_path_prefix="cma_corrected")
    #Close all plots
    plt.close("all")

    # print(np.array(info_dict_mcma["cma_error"]).shape)
    plt.title("Error vs Iterations")
    plt.plot(info_dict_mcma["cma_error"], label = "mcma_error" , alpha = 0.6)
    plt.plot(info_dict_cma["cma_error"],label = "cma_error" ,alpha = 0.6)
    plt.legend()
    plt.show()

if __name__=="__main__": main()