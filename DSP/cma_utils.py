import numpy as np
import matplotlib.pyplot as plt
#np.random.seed(42)
from scipy.optimize import minimize
from sklearn.cluster import KMeans


def gen_I_Q_qpsk(N_symbols):
    """This function generates I,Q symbols for two sets of polarisation with unit energy and
    return a tensor with shape (N_symbols,2)"""

    levels = np.array([-1, 1]) / np.sqrt(2)

    qam_symbols_x = (np.random.choice(levels, N_symbols) +
                1j * np.random.choice(levels, N_symbols))
    
    qam_symbols_y = (np.random.choice(levels, N_symbols) +
                1j * np.random.choice(levels, N_symbols))

    E_in = np.column_stack((qam_symbols_x, qam_symbols_y))

    return E_in

def gen_I_Q_16_qam(N_symbols):
    """This function generates I,Q symbols for two sets of polarisation with unit energy and 
    return a tensor with shape (N_symbols,2)"""

    levels = np.array([-3, -1, 1, 3]) / np.sqrt(10) #Dividing by root 10 to make average energy 1

    qam_symbols_x = (np.random.choice(levels, N_symbols) +
                1j * np.random.choice(levels, N_symbols))
    
    qam_symbols_y = (np.random.choice(levels, N_symbols) +
                1j * np.random.choice(levels, N_symbols))

    E_in = np.column_stack((qam_symbols_x, qam_symbols_y))

    return E_in

def apply_pmd(E_in, DGD_ps_per_sqrt_km=1.0, L_m=10000, N_sections=20, Rs=32e9, SpS=2 , seed = 0):
    """This function does pmd between the x and y polarisation and 
    returns back the x and y polarisation after pmd
    
    L_m - fiber length
    N_sections - number of fiber sections
    Rs - sampling rate
    Sps - samples per symbol
    """

    N_samples = E_in.shape[0]
    SD_tau = np.sqrt(3 * np.pi / 8) * DGD_ps_per_sqrt_km

    tau = (SD_tau * np.sqrt(L_m * 1e-3) / np.sqrt(N_sections)) * 1e-12
    #print(tau)
    #tau=2/(SpS*Rs)
    print("tau_Ts",tau*Rs)
    w = 2 * np.pi * np.fft.fftshift(np.linspace(-0.5, 0.5, N_samples)) * Rs

    E_V = np.fft.fft(E_in[:, 0])
    E_H = np.fft.fft(E_in[:, 1])

    if seed:
        np.random.seed(42)
    else:
        pass

    for _ in range(N_sections):
        # Random complex coupling matrices (unitary)
        X = np.random.randn(2, 2) + 1j * np.random.randn(2, 2)
        U, _, Vh = np.linalg.svd(X)
        V = Vh.conj().T
        # Rotate fields by U†
        E1 = U[0, 0].conj() * E_V + U[0, 1].conj() * E_H
        E2 = U[1, 0].conj() * E_V + U[1, 1].conj() * E_H

        # Apply differential delay
        E1 *= np.exp(1j * w * tau / 2)
        E2 *= np.exp(-1j * w * tau / 2)

        # Rotate by V
        E_V = V[0, 0] * E1 + V[0, 1] * E2
        E_H = V[1, 0] * E1 + V[1, 1] * E2
       

    E_out_x = np.fft.ifft(E_V)
    E_out_y = np.fft.ifft(E_H)

    return np.column_stack((E_out_x, E_out_y))

def plot_constellation(E):
    "Here E has a shape (N_symbols,2) and we plot the real and imag part of both the constellation"
    
    plt.scatter(E[:,0].real, E[:,0].imag, color='blue', label='Input X-pol', alpha=0.6)
    plt.show()
    
    
    plt.scatter(E[:,1].real, E[:,1].imag, color='blue', label='Input X-pol', alpha=0.6)
    plt.show()

    return


def save_constellation(E , save_path_prefix=""):
    "Here E has a shape (N_symbols,2) and we plot the real and imag part of both the constellation"
    
    plt.scatter(E[:,0].real, E[:,0].imag, color='blue', label='Input X-pol', alpha=0.6)
    
    if not save_path_prefix == "":
        plt.savefig(save_path_prefix + "_x_pol.png")
    else:
        plt.savefig("x_pol.png")
    
    plt.clf()
    
    plt.scatter(E[:,1].real, E[:,1].imag, color='blue', label='Input X-pol', alpha=0.6)
    
    if not save_path_prefix == "":
        plt.savefig(save_path_prefix + "_y_pol.png")
    else:
        plt.savefig("y_pol.png")

    plt.close("all")
    return
# def cma_matlab_engine(E_in, num_taps, learning_rate,matlab_function_path):
#     import matlab.engine
#     eng = matlab.engine.start_matlab()
#     eng.addpath(matlab_function_path, nargout=0)

#     ml_x_pol=matlab.double(E_in[:,0].tolist(),is_complex=True)
#     ml_x_pol=eng.transpose(ml_x_pol)
#     ml_y_pol=matlab.double(E_in[:,1].tolist(),is_complex=True)
#     ml_y_pol=eng.transpose(ml_y_pol)
#     result_x,result_y=eng.f_DSP_pol_demux_CMA(ml_x_pol,ml_y_pol,num_taps,learning_rate,nargout=2)

#     E_out=np.column_stack((result_x,result_y))
#     return E_out

def mma_python(E_in, num_taps, mu_CMA=0.01, error_threshold=0.05, Radius_options = (1/np.sqrt(10))*np.array([np.sqrt(2),np.sqrt(10),np.sqrt(18)])):
    """
    CMA with moving-average convergence detection (last 10 symbols)
    """

    # ---- Copy and normalize ----
    xpol = E_in[:, 0].astype(complex)
    ypol = E_in[:, 1].astype(complex)

    xpol = xpol / np.sqrt(np.mean(np.abs(xpol)**2))
    ypol = ypol / np.sqrt(np.mean(np.abs(ypol)**2))

    N = len(xpol)

    #Setting this for unit energy 16 QAM for now

    # ---- Tap initialization ----
    pxx = np.zeros(num_taps, dtype=complex)
    pxy = np.zeros(num_taps, dtype=complex)
    pyx = np.zeros(num_taps, dtype=complex)
    pyy = np.zeros(num_taps, dtype=complex)

    center = (num_taps - 1) // 2
    pxx[center] = 1
    pyy[center] = 1

    cma_error = {}
    convergence_symbol = None
    error_window = []   # last 10 CMA errors

    for ii in range(num_taps - 1, N):

        x_vec = xpol[ii - (num_taps - 1): ii + 1][::-1]
        y_vec = ypol[ii - (num_taps - 1): ii + 1][::-1]

        x_cap = np.dot(pxx, x_vec) + np.dot(pxy, y_vec)
        y_cap = np.dot(pyx, x_vec) + np.dot(pyy, y_vec)

        nearest_radius_x = Radius_options[np.argmin(np.abs(np.abs(x_cap) - Radius_options))]
        nearest_radius_y = Radius_options[np.argmin(np.abs(np.abs(y_cap) - Radius_options))]
        
        #print(nearest_radius_x)
        
        e_x = nearest_radius_x**2 - np.abs(x_cap)**2
        e_y = nearest_radius_y**2 - np.abs(y_cap)**2

        e_cma = 0.5 * (np.abs(e_x) + np.abs(e_y))
        cma_error[ii] = e_cma

        error_window.append(e_cma)
        if len(error_window) > 100:
            error_window.pop(0)

        if (
            convergence_symbol is None
            and len(error_window) == 100
            and np.mean(error_window) < error_threshold
        ):
            convergence_symbol = ii

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
            'cma_error': cma_error,
            'convergence_symbol': convergence_symbol
        }
    )

def cma_python(E_in, num_taps, mu_CMA=0.01, error_threshold=0.05,Num_loops = 1):
    """
    CMA with moving-average convergence detection (last 10 symbols)
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

    cma_error = []
    convergence_symbol = None
    error_window = []   # last 10 CMA errors

    for k in range(Num_loops):
        for ii in range(num_taps - 1, N):

            x_vec = xpol[ii - (num_taps - 1): ii + 1][::-1]
            y_vec = ypol[ii - (num_taps - 1): ii + 1][::-1]

            x_cap = np.dot(pxx, x_vec) + np.dot(pxy, y_vec)
            y_cap = np.dot(pyx, x_vec) + np.dot(pyy, y_vec)

            e_x = R**2 - np.abs(x_cap)**2
            e_y = R**2 - np.abs(y_cap)**2

            e_cma = 0.5 * (np.abs(e_x) + np.abs(e_y))
            cma_error.append(e_cma)

            error_window.append(e_cma)
            if len(error_window) > 10:
                error_window.pop(0)

            if (
                convergence_symbol is None
                and len(error_window) == 10
                and np.mean(error_window) < error_threshold
            ):
                convergence_symbol = ii

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
            'cma_error': cma_error,
            'convergence_symbol': convergence_symbol
        }
    )

def compute_evm(symbol, ideal_constellation):
    distances = np.abs(symbol - ideal_constellation)
    closest_symbol = ideal_constellation[np.argmin(distances)]
    return np.abs(symbol - closest_symbol)

def mcma_python(
    E_in,
    num_taps,
    mu_CMA,
    alpha=0.8,
    error_threshold=0.05,
    avg_len=100
):
    """
    CMA blind equalizer with true momentum (tap-difference based)

    g(k+1) = g(k) + mu * grad + alpha * (g(k) - g(k-1))

    Returns:
    - Equalized signal
    - Dictionary with taps, CMA error evolution, convergence symbol
    """

    # ---- Copy and normalize ----
    xpol = E_in[:, 0].astype(complex)
    ypol = E_in[:, 1].astype(complex)

    xpol /= np.sqrt(np.mean(np.abs(xpol)**2))
    ypol /= np.sqrt(np.mean(np.abs(ypol)**2))

    N = len(xpol)
    R = 1.0

    # ---- Tap initialization ----
    pxx = np.zeros(num_taps, dtype=complex)
    pxy = np.zeros(num_taps, dtype=complex)
    pyx = np.zeros(num_taps, dtype=complex)
    pyy = np.zeros(num_taps, dtype=complex)

    center = (num_taps - 1) // 2
    pxx[center] = 1.0
    pyy[center] = 1.0

    # ---- Momentum (Δg = g(k) − g(k−1)) ----
    dpxx = np.zeros_like(pxx)
    dpxy = np.zeros_like(pxy)
    dpyx = np.zeros_like(pyx)
    dpyy = np.zeros_like(pyy)

    levels = np.array([-3, -1, 1, 3]) / np.sqrt(10)
    I, Q = np.meshgrid(levels, levels)
    ideal_constellation = (I + 1j * Q).flatten()

    cma_error = []
    error_window = []
    convergence_symbol = None

    evm_list = []
    snr_list = []

    # ---- Adaptation loop ----
    for ii in range(num_taps - 1, N):

        x_vec = xpol[ii - (num_taps - 1): ii + 1][::-1]
        y_vec = ypol[ii - (num_taps - 1): ii + 1][::-1]

        # Equalizer output
        x_cap = np.dot(pxx, x_vec) + np.dot(pxy, y_vec)
        y_cap = np.dot(pyx, x_vec) + np.dot(pyy, y_vec)

        # CMA errors
        e_x = R**2 - np.abs(x_cap)**2
        e_y = R**2 - np.abs(y_cap)**2

        e_cma = 0.5 * (np.abs(e_x) + np.abs(e_y))
        cma_error.append(e_cma)

        evm_x = compute_evm(x_cap, ideal_constellation)
        evm_y = compute_evm(y_cap, ideal_constellation)

        evm_avg = 0.5 * (evm_x + evm_y)
        evm_list.append(evm_avg)

        snr_value = 1 / (evm_avg**2 + 1e-12)
        snr_list.append(snr_value)
        

        error_window.append(e_cma)
        if len(error_window) > avg_len:
            error_window.pop(0)

        if (
            convergence_symbol is None
            and len(error_window) == avg_len
            and np.mean(error_window) < error_threshold
        ):
            convergence_symbol = ii

        # ---- CMA gradients ----
        gxx = 2 * e_x * x_cap * np.conj(x_vec)
        gxy = 2 * e_x * x_cap * np.conj(y_vec)
        gyx = 2 * e_y * y_cap * np.conj(x_vec)
        gyy = 2 * e_y * y_cap * np.conj(y_vec)

        # ---- Momentum update ----
        dpxx = mu_CMA * gxx + alpha * dpxx
        dpxy = mu_CMA * gxy + alpha * dpxy
        dpyx = mu_CMA * gyx + alpha * dpyx
        dpyy = mu_CMA * gyy + alpha * dpyy

        # ---- Tap update ----
        pxx += dpxx
        pxy += dpxy
        pyx += dpyx
        pyy += dpyy

    # ---- Final filtering (MATLAB conv(...,'same')) ----
    def conv_same(sig, taps):
        full = np.convolve(sig, taps, mode='full')
        start = (len(taps) - 1) // 2
        return full[start:start + len(sig)]

    x_out = conv_same(xpol, pxx) + conv_same(ypol, pxy)
    y_out = conv_same(xpol, pyx) + conv_same(ypol, pyy)

    return (
        np.column_stack((x_out, y_out)),
        {
            "pxx": pxx,
            "pxy": pxy,
            "pyx": pyx,
            "pyy": pyy,
            "cma_error": cma_error,
            "evm": evm_list,
            "snr": snr_list,
            "convergence_symbol": convergence_symbol,
        }
    )

def vmcma_python(
    E_in,
    num_taps,
    mu_CMA,
    alpha_init=0.8,
    eta=0.95,
    gamma=1e-3,
    alpha_min=0.1,
    alpha_max=0.95,
    error_threshold=0.05,
    avg_len=10
):
    """
    Variable Momentum CMA
    alpha(k) = eta * alpha(k-1) + gamma * ||grad||^2
    """

    # ---- Copy and normalize ----
    xpol = E_in[:, 0].astype(complex)
    ypol = E_in[:, 1].astype(complex)

    xpol /= np.sqrt(np.mean(np.abs(xpol)**2))
    ypol /= np.sqrt(np.mean(np.abs(ypol)**2))

    N = len(xpol)
    R = 1.0

    # ---- Tap initialization ----
    pxx = np.zeros(num_taps, dtype=complex)
    pxy = np.zeros(num_taps, dtype=complex)
    pyx = np.zeros(num_taps, dtype=complex)
    pyy = np.zeros(num_taps, dtype=complex)

    center = (num_taps - 1) // 2
    pxx[center] = 1.0
    pyy[center] = 1.0

    # ---- Momentum state ----
    dpxx = np.zeros_like(pxx)
    dpxy = np.zeros_like(pxy)
    dpyx = np.zeros_like(pyx)
    dpyy = np.zeros_like(pyy)

    alpha_k = alpha_init

    cma_error = {}
    error_window = []
    convergence_symbol = None

    # ---- Adaptation loop ----
    for ii in range(num_taps - 1, N):

        x_vec = xpol[ii - (num_taps - 1): ii + 1][::-1]
        y_vec = ypol[ii - (num_taps - 1): ii + 1][::-1]

        # Equalizer output
        x_cap = np.dot(pxx, x_vec) + np.dot(pxy, y_vec)
        y_cap = np.dot(pyx, x_vec) + np.dot(pyy, y_vec)

        # CMA error
        e_x = R**2 - np.abs(x_cap)**2
        e_y = R**2 - np.abs(y_cap)**2

        e_cma = 0.5 * (np.abs(e_x) + np.abs(e_y))
        cma_error[ii] = e_cma

        error_window.append(e_cma)
        if len(error_window) > avg_len:
            error_window.pop(0)

        if (
            convergence_symbol is None
            and len(error_window) == avg_len
            and np.mean(error_window) < error_threshold
        ):
            convergence_symbol = ii

        # ---- CMA gradients ----
        gxx = 2 * e_x * x_cap * np.conj(x_vec)
        gxy = 2 * e_x * x_cap * np.conj(y_vec)
        gyx = 2 * e_y * y_cap * np.conj(x_vec)
        gyy = 2 * e_y * y_cap * np.conj(y_vec)

        # ---- Gradient energy (for VMCMA) ----
        grad_norm_sq = (
            np.vdot(gxx, gxx).real +
            np.vdot(gxy, gxy).real +
            np.vdot(gyx, gyx).real +
            np.vdot(gyy, gyy).real
        )

        # ---- Update momentum factor ----
        alpha_k = eta * alpha_k + gamma * grad_norm_sq
        alpha_k = np.clip(alpha_k, alpha_min, alpha_max)

        # ---- Momentum update ----
        dpxx = mu_CMA * gxx + alpha_k * dpxx
        dpxy = mu_CMA * gxy + alpha_k * dpxy
        dpyx = mu_CMA * gyx + alpha_k * dpyx
        dpyy = mu_CMA * gyy + alpha_k * dpyy

        # ---- Tap update ----
        pxx += dpxx
        pxy += dpxy
        pyx += dpyx
        pyy += dpyy

    # ---- Final filtering ----
    def conv_same(sig, taps):
        full = np.convolve(sig, taps, mode='full')
        start = (len(taps) - 1) // 2
        return full[start:start + len(sig)]

    x_out = conv_same(xpol, pxx) + conv_same(ypol, pxy)
    y_out = conv_same(xpol, pyx) + conv_same(ypol, pyy)

    return (
        np.column_stack((x_out, y_out)),
        {
            "pxx": pxx,
            "pxy": pxy,
            "pyx": pyx,
            "pyy": pyy,
            "cma_error": cma_error,
            "convergence_symbol": convergence_symbol,
        }
    )

def cma_lbfgs(E_in, num_taps,
                   R=1,
                   maxiter=200,
                   maxcor=10):

    xpol = E_in[:, 0].astype(complex)
    ypol = E_in[:, 1].astype(complex)

    xpol /= np.sqrt(np.mean(np.abs(xpol)**2))
    ypol /= np.sqrt(np.mean(np.abs(ypol)**2))

    N = len(xpol)

    pxx0 = np.zeros(num_taps, dtype=complex)
    pxy0 = np.zeros(num_taps, dtype=complex)
    pyx0 = np.zeros(num_taps, dtype=complex)
    pyy0 = np.zeros(num_taps, dtype=complex)

    center = (num_taps - 1) // 2
    pxx0[center] = 1
    pyy0[center] = 1

    def pack(pxx, pxy, pyx, pyy):
        w = np.concatenate([pxx, pxy, pyx, pyy])
        return np.concatenate([w.real, w.imag])

    def unpack(theta):
        half = len(theta) // 2
        wc = theta[:half] + 1j * theta[half:]

        pxx = wc[0:num_taps]
        pxy = wc[num_taps:2*num_taps]
        pyx = wc[2*num_taps:3*num_taps]
        pyy = wc[3*num_taps:4*num_taps]

        return pxx, pxy, pyx, pyy

    def cost_function(theta):

        pxx, pxy, pyx, pyy = unpack(theta)

        J = 0.0
        count = 0

        for ii in range(num_taps - 1, N):

            x_vec = xpol[ii-(num_taps-1):ii+1][::-1]
            y_vec = ypol[ii-(num_taps-1):ii+1][::-1]

            x_cap = np.dot(pxx, x_vec) + np.dot(pxy, y_vec)
            y_cap = np.dot(pyx, x_vec) + np.dot(pyy, y_vec)

            ex = (np.abs(x_cap)**2 - R**2)
            ey = (np.abs(y_cap)**2 - R**2)

            J += ex**2 + ey**2
            count += 1

        return J / count

    def gradient(theta):

        pxx, pxy, pyx, pyy = unpack(theta)

        g_pxx = np.zeros(num_taps, dtype=complex)
        g_pxy = np.zeros(num_taps, dtype=complex)
        g_pyx = np.zeros(num_taps, dtype=complex)
        g_pyy = np.zeros(num_taps, dtype=complex)

        count = 0

        for ii in range(num_taps - 1, N):

            x_vec = xpol[ii-(num_taps-1):ii+1][::-1]
            y_vec = ypol[ii-(num_taps-1):ii+1][::-1]

            x_cap = np.dot(pxx, x_vec) + np.dot(pxy, y_vec)
            y_cap = np.dot(pyx, x_vec) + np.dot(pyy, y_vec)

            ex = (np.abs(x_cap)**2 - R**2)
            ey = (np.abs(y_cap)**2 - R**2)

            g_pxx += 4 * ex * x_cap * np.conj(x_vec)
            g_pxy += 4 * ex * x_cap * np.conj(y_vec)

            g_pyx += 4 * ey * y_cap * np.conj(x_vec)
            g_pyy += 4 * ey * y_cap * np.conj(y_vec)

            count += 1

        # Normalize gradient
        g_pxx /= count
        g_pxy /= count
        g_pyx /= count
        g_pyy /= count

        grad_complex = np.concatenate([g_pxx, g_pxy, g_pyx, g_pyy])

        return np.concatenate([grad_complex.real,
                               grad_complex.imag])


    theta0 = pack(pxx0, pxy0, pyx0, pyy0)

    result = minimize(
        fun=cost_function,
        x0=theta0,
        jac=gradient,
        method="L-BFGS-B",
        options={
            "maxiter": maxiter,
            "maxcor": maxcor,
            "disp": True
        }
    )

    pxx_opt, pxy_opt, pyx_opt, pyy_opt = unpack(result.x)

    ii = N - 1  # last symbol index

    x_vec = xpol[ii-(num_taps-1):ii+1][::-1]
    y_vec = ypol[ii-(num_taps-1):ii+1][::-1]

    x_cap = np.dot(pxx_opt, x_vec) + np.dot(pxy_opt, y_vec)
    y_cap = np.dot(pyx_opt, x_vec) + np.dot(pyy_opt, y_vec)

    ex_last = (np.abs(x_cap)**2 - R**2)
    ey_last = (np.abs(y_cap)**2 - R**2)

    # Per-symbol CMA error at final symbol
    J_last = ex_last**2 + ey_last**2
    
    # ---- Final filtering ----
    def conv_same(sig, taps):
        full = np.convolve(sig, taps, mode='full')
        start = (len(taps) - 1) // 2
        return full[start:start + len(sig)]

    x_out = conv_same(xpol, pxx_opt) + conv_same(ypol, pxy_opt)
    y_out = conv_same(xpol, pyx_opt) + conv_same(ypol, pyy_opt)

    return (np.column_stack((x_out, y_out)), result.fun, J_last, pxx_opt, pxy_opt, pyx_opt, pyy_opt)


def mcma_python_adam(
    E_in,
    num_taps,
    mu_CMA,
    beta1=0.9,      # Adam first moment decay
    beta2=0.999,    # Adam second moment decay
    eps=1e-8,
):

    xpol = E_in[:, 0].astype(complex)
    ypol = E_in[:, 1].astype(complex)

    xpol /= np.sqrt(np.mean(np.abs(xpol)**2))
    ypol /= np.sqrt(np.mean(np.abs(ypol)**2))

    N = len(xpol)
    R = 1.0

    pxx = np.zeros(num_taps, dtype=complex)
    pxy = np.zeros(num_taps, dtype=complex)
    pyx = np.zeros(num_taps, dtype=complex)
    pyy = np.zeros(num_taps, dtype=complex)

    center = (num_taps - 1) // 2
    pxx[center] = 1.0
    pyy[center] = 1.0

    mpxx = np.zeros_like(pxx)
    mpxy = np.zeros_like(pxy)
    mpyx = np.zeros_like(pyx)
    mpyy = np.zeros_like(pyy)

    vpxx = np.zeros_like(pxx)
    vpxy = np.zeros_like(pxy)
    vpyx = np.zeros_like(pyx)
    vpyy = np.zeros_like(pyy)

    t = 0

    cma_error = {}

    for ii in range(num_taps - 1, N):

        t += 1

        x_vec = xpol[ii - (num_taps - 1): ii + 1][::-1]
        y_vec = ypol[ii - (num_taps - 1): ii + 1][::-1]

        x_cap = np.dot(pxx, x_vec) + np.dot(pxy, y_vec)
        y_cap = np.dot(pyx, x_vec) + np.dot(pyy, y_vec)

        e_x = R**2 - np.abs(x_cap)**2
        e_y = R**2 - np.abs(y_cap)**2

        e_cma = 0.5 * (np.abs(e_x) + np.abs(e_y))
        cma_error[ii] = e_cma

        gxx = 2 * e_x * x_cap * np.conj(x_vec)
        gxy = 2 * e_x * x_cap * np.conj(y_vec)
        gyx = 2 * e_y * y_cap * np.conj(x_vec)
        gyy = 2 * e_y * y_cap * np.conj(y_vec)

        # ---- First Moment (Adam) ----
        mpxx = beta1 * mpxx + (1 - beta1) * gxx
        mpxy = beta1 * mpxy + (1 - beta1) * gxy
        mpyx = beta1 * mpyx + (1 - beta1) * gyx
        mpyy = beta1 * mpyy + (1 - beta1) * gyy

        # ---- Second Moment (Adam) ----
        vpxx = beta2 * vpxx + (1 - beta2) * (np.abs(gxx)**2)
        vpxy = beta2 * vpxy + (1 - beta2) * (np.abs(gxy)**2)
        vpyx = beta2 * vpyx + (1 - beta2) * (np.abs(gyx)**2)
        vpyy = beta2 * vpyy + (1 - beta2) * (np.abs(gyy)**2)

        # ---- Bias correction ----
        mxx_hat = mpxx / (1 - beta1**t)
        mxy_hat = mpxy / (1 - beta1**t)
        myx_hat = mpyx / (1 - beta1**t)
        myy_hat = mpyy / (1 - beta1**t)

        vxx_hat = vpxx / (1 - beta2**t)
        vxy_hat = vpxy / (1 - beta2**t)
        vyx_hat = vpyx / (1 - beta2**t)
        vyy_hat = vpyy / (1 - beta2**t)

        # ---- Update taps (Adam rule) ----
        pxx += mu_CMA * mxx_hat / (np.sqrt(vxx_hat) + eps)
        pxy += mu_CMA * mxy_hat / (np.sqrt(vxy_hat) + eps)
        pyx += mu_CMA * myx_hat / (np.sqrt(vyx_hat) + eps)
        pyy += mu_CMA * myy_hat / (np.sqrt(vyy_hat) + eps)

    # ---- Final filtering ----
    def conv_same(sig, taps):
        full = np.convolve(sig, taps, mode='full')
        start = (len(taps) - 1) // 2
        return full[start:start + len(sig)]

    x_out = conv_same(xpol, pxx) + conv_same(ypol, pxy)
    y_out = conv_same(xpol, pyx) + conv_same(ypol, pyy)

    return (
        np.column_stack((x_out, y_out)),
        {
            "pxx": pxx,
            "pxy": pxy,
            "pyx": pyx,
            "pyy": pyy,
            "cma_error": cma_error,
        }
    )



def normalise(E):
    """This function takes in the polarisations and normalises them"""
    E[:,0] = E[:,0] / np.sqrt(np.mean(np.abs(E[:,0])**2))
    E[:,1] = E[:,1] / np.sqrt(np.mean(np.abs(E[:,1])**2))
    return E

def apply_filters(E_in,cur_ind,num_taps,filters):
    """This function basically just does convolution with filters and gives output at current index
    NOTE: Make sure that E_in to this function is normalised"""
    

    x=E_in[:,0][cur_ind-num_taps+1:cur_ind+1][::-1]
    y=E_in[:,1][cur_ind-num_taps+1:cur_ind+1][::-1]

    x_out=np.dot(filters['pxx'],x)+np.dot(filters['pxy'],y)
    y_out=np.dot(filters['pyx'],x)+np.dot(filters['pyy'],y)
    
    return x_out,y_out

def apply_entire_filters(E_in,filters):
    """This function basically just does convolution with filters and gives output E"""

    def conv_same(sig, taps):
        full = np.convolve(sig, taps, mode='full')
        start = (len(taps) - 1) // 2
        return full[start : start + len(sig)]

    x_out = conv_same(E_in[:,0], filters['pxx']) + conv_same(E_in[:,1], filters['pxy'])
    y_out = conv_same(E_in[:,0], filters['pyx']) + conv_same(E_in[:,1], filters['pyy'])

    return np.column_stack((x_out, y_out))

def cma_error_dualpol(x_out,y_out,Radius=1):
    """This function gives CMA error for dual polarisation and sums them"""
    e_x=((np.abs(x_out)**2-Radius**2))**2
    e_y=((np.abs(y_out)**2-Radius**2))**2
    return e_x+e_y

def compute_reward(x_out,y_out,REWARD_CLIP=-10):
    """For reward we need it to be neg of cma as cma error is minimum for good filters and we want high reward"""
    reward=-cma_error_dualpol(x_out,y_out)
    reward=np.clip(reward,REWARD_CLIP,0)
    return reward+10

def initialise_filters(NUM_TAPS):
    filters={}
    filters['pxx'] = np.zeros(NUM_TAPS, dtype=complex)
    filters['pxx'][NUM_TAPS//2] = 1

    filters['pxy'] = np.zeros(NUM_TAPS,dtype=complex)
    
    filters['pyx'] = np.zeros(NUM_TAPS,dtype=complex)
    
    filters['pyy'] = np.zeros(NUM_TAPS)
    filters['pyy'][NUM_TAPS//2] = 1
    
    
    return filters


def _interleave_real_imag(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x)
    # shape (N,) -> (N,2) as [real, imag] per element, then flatten -> (2N,)
    return np.stack([x.real, x.imag], axis=1).ravel().astype(np.float32)

def convert_filter_to_state(filters: dict) -> np.ndarray:
    keys = ["pxx", "pxy", "pyx", "pyy"]
    parts = [_interleave_real_imag(filters[k]) for k in keys]
    return np.concatenate(parts, axis=0)

def _deinterleave_to_complex(v: np.ndarray, num_taps: int, complex_dtype=np.complex64) -> np.ndarray:
    v = np.asarray(v)
    assert v.size == 2 * num_taps, f"Expected {2*num_taps} values, got {v.size}"
    ri = v.reshape(num_taps, 2)                # [[r0,i0],[r1,i1],...]
    return (ri[:, 0] + 1j * ri[:, 1]).astype(complex_dtype) 

def state_to_filter(state: np.ndarray, num_taps: int) -> dict:
    """
    Inverse of convert_filter_to_state (the interleaved version):
    state layout:
      pxx: r0,i0,r1,i1,...,r_{N-1},i_{N-1},
      pxy: ...,
      pyx: ...,
      pyy: ...
    Returns dict with complex tap arrays.
    """
    state = np.asarray(state, dtype=np.float32)
    n_per_filter = 2 * num_taps
    expected = 4 * n_per_filter
    assert state.size == expected, f"Expected state length {expected}, got {state.size}"

    keys = ["pxx", "pxy", "pyx", "pyy"]
    filters = {}

    offset = 0
    for k in keys:
        chunk = state[offset: offset + n_per_filter]
        filters[k] = _deinterleave_to_complex(chunk, num_taps)
        offset += n_per_filter

    return filters

def calculate_state_distance(cur_state,intial_state):
    return np.linalg.norm(cur_state-intial_state)

def apply_cfo_both_polarisations(E_in,freq_offset,fs):
    x_out,y_out=apply_cfo(E_in[:,0],freq_offset,fs),apply_cfo(E_in[:,1],freq_offset,fs)
    return np.column_stack((x_out,y_out))

def apply_cfo(symbols,freq_offset,fs):
    phase_shift = np.exp(1j * 2 * np.pi * freq_offset * np.arange(len(symbols)) / fs)
    #print(phase_shift)
    return symbols * phase_shift

def apply_awgn(E_in, SNR_dB):
    signal_power = np.mean(np.abs(E_in)**2)
    print("signal power is", signal_power)
    SNR_linear = 10**(SNR_dB / 10)
    noise_power = signal_power / SNR_linear
    
    noise = (np.sqrt(noise_power/2) *
            (np.random.randn(*E_in.shape) + 
             1j*np.random.randn(*E_in.shape)))
    
    return E_in + noise

def cfo_correction_both_pol(E_in,fs):
    x_out,y_out = fourth_power_cfo_correction(E_in[:,0],fs),fourth_power_cfo_correction(E_in[:,1],fs)
    return np.column_stack((x_out,y_out))

def fourth_power_cfo_correction(symbols,fs):
    input_fourth_power = np.power(symbols,4,dtype=np.complex64)
    freq_domain_fourth_power = np.fft.fftshift(np.fft.fft(input_fourth_power))
    
    max_value_index = np.argmax(np.abs(freq_domain_fourth_power))
    #print(max_value_index)
    freq_axis = np.fft.fftshift(np.fft.fftfreq(len(freq_domain_fourth_power),1/fs))
    #print(freq_axis)
    freq_offset = freq_axis[max_value_index] / 4
    #print(freq_offset)
    corrected_symbols = apply_cfo(symbols,-freq_offset,fs)
    return corrected_symbols


def cluster_constellations(complex_points: np.ndarray, n_clusters:int =16)-> tuple[dict, np.ndarray]:
    """
    Clusters complex points and returns grouped points and their centers 
    as complex numbers.
    """
    # 1. Prepare data (Real = x, Imag = y)
    data = np.array([[z.real, z.imag] for z in complex_points])
    
    # 2. Run KMeans
    kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
    labels = kmeans.fit_predict(data)
    
    # 3. Convert centers back to complex numbers (x + iy)
    # The order of kmeans.cluster_centers_ matches the labels 0 to (n-1)
    centers_xy = kmeans.cluster_centers_
    complex_centers = [complex(c[0], c[1]) for c in centers_xy]

    # 4. Group points by cluster label
    clusters = {i: [] for i in range(n_clusters)}
    for label, original_z in zip(labels, complex_points):
        clusters[label].append(original_z)
        
    # Return both the dictionary and the list of complex centers
    return clusters, np.array(complex_centers)

def calculate_snr(center, points):
    """This function takes in the center of a cluster and the cluster points and returns the average SNR"""
    evm_total = 0
    for point in points:
        evm_total+=np.square(np.abs(point-center))
    avg_evm_sq = evm_total/len(points)
    avg_snr = 10*np.log10(1/avg_evm_sq)
    return avg_snr


def caculate_avg_cluster_snr(clusters:dict,centers:np.ndarray)->float:
    """This function takes in the dictionary of clusters and their centers and returns the average SNR"""
    total_snr=0
    for cluster_ind,cluster in clusters.items():
        total_snr+=calculate_snr(centers[cluster_ind],cluster)
    return total_snr/len(clusters)

def cluster_and_get_avg_snr(input_polarisations: np.ndarray, n_clusters:int =16)-> tuple[float,float]:
    """This function takes in input polarisation forms clusters and returns average SNR for x and y polarisation"""
    clusters_x,centers_x = cluster_constellations(input_polarisations[:,0],n_clusters)
    avg_snr_x = caculate_avg_cluster_snr(clusters_x,centers_x)

    clusters_y,centers_y = cluster_constellations(input_polarisations[:,1],n_clusters)
    avg_snr_y = caculate_avg_cluster_snr(clusters_y,centers_y)

    return avg_snr_x,avg_snr_y