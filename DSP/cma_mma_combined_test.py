import cma_utils
import numpy as np
def test_cma_mma_combined(E_in, num_taps, mu_CMA=0.01, Radius_options = (1/np.sqrt(10))*np.array([np.sqrt(2),np.sqrt(10),np.sqrt(18)])):
    cma_convergence_symbols = 50000
    E_after_cma,info_dict_cma = cma_utils.cma_python(E_in[:cma_convergence_symbols], num_taps, mu_CMA)
    filters_after_cma = {'pxx': info_dict_cma['pxx'], 'pyy': info_dict_cma['pyy'], 'pxy': info_dict_cma['pxy'], 'pyx': info_dict_cma['pyx']}
    E_after_mma,info_dict_mma = cma_utils.mma_python(E_in[cma_convergence_symbols:], num_taps, mu_CMA, Radius_options,initial_filters=filters_after_cma)
    
    return E_after_mma