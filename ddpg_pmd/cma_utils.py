import numpy as np
import matplotlib.pyplot as plt
np.random.seed(42)


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

def apply_pmd(E_in, DGD_ps_per_sqrt_km=1, L_m=10000, N_sections=20, Rs=32e9, SpS=2):
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
    w = 2 * np.pi * np.fft.fftshift(np.linspace(-0.5, 0.5, N_samples)) * SpS * Rs

    E_V = np.fft.fft(E_in[:, 0])
    E_H = np.fft.fft(E_in[:, 1])

    np.random.seed(42)
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

def cma_python(E_in, num_taps,mu_CMA=0.01):
    """
    This is a python only function to implement the CMA algorithm
    Here mu_CMA is the learning rate
    This function returns the output x,y polarisation along with the converged filter taps
    """

    # Copy and normalize (MATLAB RMS normalization)
    xpol = E_in[:, 0].astype(complex)
    ypol = E_in[:, 1].astype(complex)

    xpol = xpol / np.sqrt(np.mean(np.abs(xpol)**2))
    ypol = ypol / np.sqrt(np.mean(np.abs(ypol)**2))

    N = len(xpol)
    R = 1  # MATLAB fixed modulus

    # ---- Tap initialization (MATLAB: center tap = ceil(N/2)) ----
    pxx = np.zeros(num_taps, dtype=complex)
    pxy = np.zeros(num_taps, dtype=complex)
    pyx = np.zeros(num_taps, dtype=complex)
    pyy = np.zeros(num_taps, dtype=complex)

    center = (num_taps - 1) // 2
    pxx[center] = 1
    pyy[center] = 1

    # ---- Adaptation loop ----
    for ii in range(num_taps - 1, N):

        # MATLAB slice: xpol(ii:-1:ii-num_taps+1)
        x_vec = xpol[ii - (num_taps - 1): ii + 1][::-1]
        y_vec = ypol[ii - (num_taps - 1): ii + 1][::-1]

        # ---- Compute estimated symbols ----
        x_cap = np.dot(pxx, x_vec) + np.dot(pxy, y_vec)
        y_cap = np.dot(pyx, x_vec) + np.dot(pyy, y_vec)

        # ---- CMA errors ----
        e_x = R**2 - np.abs(x_cap)**2
        e_y = R**2 - np.abs(y_cap)**2

        # ---- Tap updates (identical to MATLAB) ----
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

    return np.column_stack((x_out, y_out)), {
        'pxx': pxx, 'pxy': pxy, 'pyx': pyx, 'pyy': pyy
    }

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
