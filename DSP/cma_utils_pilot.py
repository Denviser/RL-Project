import numpy as np
import matplotlib.pyplot as plt
import cma_utils

def gen_I_Q_qpsk(N_symbols):
    levels = np.array([-1, 1]) / np.sqrt(2)

    qam_symbols_x = (np.random.choice(levels, N_symbols) +
                     1j * np.random.choice(levels, N_symbols))
    qam_symbols_y = (np.random.choice(levels, N_symbols) +
                     1j * np.random.choice(levels, N_symbols))

    E_in = np.column_stack((qam_symbols_x, qam_symbols_y))
    return E_in

def first_eleven():
    X = np.array([
        -3+3j,  3+3j, -3+3j,  3+3j, -3-3j, 3+3j, -3-3j, -3-3j,  3+3j,  3-3j,  3-3j
        ])
    Y = np.array([
        -3-3j, -3-3j,  3-3j, -3+3j, -3+3j, 3+3j, -3-3j, -3+3j,  3-3j,  3+3j,  3-3j
        ])
    norm = np.sqrt(18)
    return X / norm, Y / norm

def lfsr_sequence(seed, length):
    # 10-bit shift register based on x^10 + x^8 + x^4 + x^3 + 1
    
    state = seed & 0x3FF  # keep 10 bits
    seq = []

    for _ in range(length):
        b9 = (state >> 9) & 1 #x^10
        b7 = (state >> 7) & 1 #x^8
        b3 = (state >> 3) & 1 #x^4
        b2 = (state >> 2) & 1 #x^3
        b0 = (state >> 0) & 1 #1

        new_bit = b9 ^ b7 ^ b3 ^ b2 ^ b0

        # output is last bit
        seq.append(b0)

        # state update with feedback loop
        state = ((state << 1) & 0x3FF) | new_bit

    return np.array(seq)


def generate_lfsr_pilots(seed, num_pilots):
    bits = lfsr_sequence(seed, num_pilots * 2)
    symbols = []
    # generating symbols from the bits
    # using pairwise bits to get the real and imag part of symbol
    for i in range(num_pilots):
        b1 = bits[2*i]
        b2 = bits[2*i + 1]
        # generate real part as 3 if odd bit is 1, else -3
        # generate imag part as 3 if even bit is 1, else -3
        real = 1 if b1 else -1
        imag = 1 if b2 else -1

        symbols.append(real + 1j * imag)

    return np.array(symbols) / np.sqrt(2)

def generate_pilot_mask():
    frame_len = 3712

    frame_x = np.zeros(frame_len, dtype=complex)
    frame_y = np.zeros(frame_len, dtype=complex)

    # first 32 bit section:

    x_pilots_11, y_pilots_11 = first_eleven()

    idx = 0
    frame_x[idx:idx+11] = x_pilots_11
    frame_y[idx:idx+11] = y_pilots_11
    idx += 11
    
    #Just make the 21 indices 0 (They are already 0 so just skip 21 indices)
    idx += 21

    # remaining sections:

    # seed X = 0x19E, seed Y = 0x0D0 for the remaining 115 pilots
    lfsr_x = generate_lfsr_pilots(0x19E, 115)
    lfsr_y = generate_lfsr_pilots(0x0D0, 115)

    #print(lfsr_x[:10])

    for i in range(115):
        # pilot
        frame_x[idx] = lfsr_x[i]
        frame_y[idx] = lfsr_y[i]
        idx += 1

        #Just make the 31 indices 0 (They are already 0 so just skip 21 indices)
        idx += 31

    return np.column_stack((frame_x, frame_y))

def generate_frame():
    frame_len = 3712

    frame_x = np.zeros(frame_len, dtype=complex)
    frame_y = np.zeros(frame_len, dtype=complex)

    # first 32 bit section:

    x_pilots_11, y_pilots_11 = first_eleven()

    idx = 0
    frame_x[idx:idx+11] = x_pilots_11
    frame_y[idx:idx+11] = y_pilots_11
    idx += 11
    
    data = gen_I_Q_qpsk(21)
    frame_x[idx : idx+21] = data[ : , 0]
    frame_y[idx : idx+21] = data[ : , 1]
    idx += 21

    # remaining sections:

    # seed X = 0x19E, seed Y = 0x0D0 for the remaining 115 pilots
    lfsr_x = generate_lfsr_pilots(0x19E, 115)
    lfsr_y = generate_lfsr_pilots(0x0D0, 115)

    #print(lfsr_x[:10])

    for i in range(115):
        # pilot
        frame_x[idx] = lfsr_x[i]
        frame_y[idx] = lfsr_y[i]
        idx += 1

        # data
        data = gen_I_Q_qpsk(31)
        frame_x[idx:idx+31] = data[:, 0]
        frame_y[idx:idx+31] = data[:, 1]
        idx += 31

    return np.column_stack((frame_x, frame_y))

def plot_loss_vs_tau(pilot_stream):
    shift_arr = []
    loss_arr = []
    for shift in range(-20,20):
        shifted_pilots = np.roll(pilot_stream, shift, axis=0)
        shifted_pilots_fft = np.fft.fft(shifted_pilots, axis=0)
        pilot_stream_fft = np.fft.fft(pilot_stream, axis=0)
        print("shifted_pilots_fft ", shifted_pilots_fft)
        loss = np.mean(np.abs(pilot_stream_fft - shifted_pilots_fft)**2)
        shift_arr.append(shift)
        loss_arr.append(loss)
    plt.plot(shift_arr, loss_arr)
    plt.xlabel("Shift")
    plt.ylabel("Loss")
    plt.title("Loss vs Shift for Pilot Stream")
    plt.show()

def plot_loss_vs_tau_time(signal, pilots):
    losses = []
    shifts = range(0, len(signal) - len(pilots))  # FIX

    for shift in shifts:
        segment = signal[shift:shift+len(pilots)]

        loss = np.mean(np.abs(segment - pilots)**2)
        losses.append(loss)

    plt.plot(shifts, losses)
    plt.xlabel("Shift")
    plt.ylabel("Loss")
    plt.title("Pilot Alignment Loss vs Shift")
    plt.grid()
    plt.show()

def generate_stream(N_total,offset):
    frame_len = 3712
    num_frames = int(np.ceil(N_total / frame_len)) + 1

    #print("num_frames:", num_frames)
    stream = np.vstack([generate_frame() for _ in range(num_frames)])

    #print("stream shape:", stream.shape)
    stream[:,0] = np.roll(stream[:,0],offset) # Shifts right by offset
    stream[:,1] = np.roll(stream[:,1],offset)
    #print("stream shape after offset:", stream.shape)
    #print("offset is",offset)
    return stream[:N_total]

def lms_cfo_joint_with_pilots(E_in, num_taps, tau, mu=1e-4, mu_f=1e-8, fs=2e9):
    """
    Joint LMS equalizer + CFO estimation
    Error computed between either pilots or slicer outputs based on symbol index
    tau  : symbol index where the first complete frame begins
    """
    E_norm = cma_utils.normalise(E_in.copy())
    xpol = E_norm[:, 0]
    ypol = E_norm[:, 1]

    N = len(xpol)

    filters = cma_utils.initialise_filters(num_taps)
    pxx, pyy, pxy, pyx = filters['pxx'], filters['pyy'], filters['pxy'], filters['pyx']

    pilot_frame = generate_pilot_mask()
    frame_len = 3712

    f_est = 0.0
    error_list = []

    for ii in range(num_taps - 1, N):

        # Input vectors
        x_vec = xpol[ii - (num_taps - 1): ii + 1][::-1]
        y_vec = ypol[ii - (num_taps - 1): ii + 1][::-1]

        # Equalizer output
        x_cap = np.dot(pxx, x_vec) + np.dot(pxy, y_vec)
        y_cap = np.dot(pyx, x_vec) + np.dot(pyy, y_vec)

        # CFO correction
        n = ii
        phase = np.exp(-1j * 2 * np.pi * f_est * n/fs)

        x_corr = x_cap * phase
        y_corr = y_cap * phase

        # deciding error function
        # if the pilot mask value is non zero, we compute error with the pilots, else with the slicer output
        if ii >= tau:
            # find where we are within the current frame
            frame_idx = (ii - tau) % frame_len
            p_x, p_y = pilot_frame[frame_idx, 0], pilot_frame[frame_idx, 1]
        else:
            # masked symbol value are zero before the first frame
            p_x = 0.0 + 0j
            p_y = 0.0 + 0j

        # if mask has a non-zero value, it's a pilot
        if np.abs(p_x) > 0:
            d_x = p_x
            d_y = p_y
        else:
            # d_x = cma_utils.slicer_16qam(x_corr)
            # d_y = cma_utils.slicer_16qam(y_corr)
            d_x = cma_utils.slicer_qpsk(x_corr)
            d_y = cma_utils.slicer_qpsk(y_corr)

        e_x = x_corr - d_x
        e_y = y_corr - d_y

        error_list.append(0.5 * ((np.abs(e_x))**2 + (np.abs(e_y))**2))

        pxx -= mu * e_x * np.conj(x_vec) * np.conj(phase)
        pxy -= mu * e_x * np.conj(y_vec) * np.conj(phase)
        pyx -= mu * e_y * np.conj(x_vec) * np.conj(phase)
        pyy -= mu * e_y * np.conj(y_vec) * np.conj(phase)

        # dy/df = -j (2π n / fs) * y_corr
        coeff = -1j * 2 * np.pi * n/fs

        grad_fx = np.real(e_x * np.conj(coeff * x_corr))
        grad_fy = np.real(e_y * np.conj(coeff * y_corr))

        grad_f = grad_fx + grad_fy

        # Update CFO
        f_est -= mu_f * grad_f * fs

    x_out = cma_utils.conv_same(xpol, pxx) + cma_utils.conv_same(ypol, pxy)
    y_out = cma_utils.conv_same(xpol, pyx) + cma_utils.conv_same(ypol, pyy)

    n_arr = np.arange(len(x_out))
    phase_arr = np.exp(-1j * 2 * np.pi * f_est * n_arr/fs)

    x_out *= phase_arr
    y_out *= phase_arr

    return (
        np.column_stack((x_out, y_out)),
        {
            'pxx': pxx, 'pxy': pxy,
            'pyx': pyx, 'pyy': pyy,
            'f_est': f_est,
            'cma_error': error_list
        }
    )


def remove_outliers_iqr(data_array, iqr_multiplier=1.5):
    """
    Removes outliers from a 1D NumPy array using the Interquartile Range (IQR) method.

    Outliers are defined as data points that fall below (Q1 - iqr_multiplier * IQR)
    or above (Q3 + iqr_multiplier * IQR).

    Args:
        data_array (np.ndarray): The input 1D NumPy array from which to remove outliers.
        iqr_multiplier (float, optional): The multiplier for the IQR to define the
                                          outlier bounds. Common values are 1.5 (for
                                          mild outliers) or 3.0 (for extreme outliers).
                                          Defaults to 1.5.

    Returns:
        np.ndarray: A new NumPy array with outliers removed.
    """
    if not isinstance(data_array, np.ndarray) or data_array.ndim != 1:
        raise ValueError("Input must be a 1D NumPy array.")
    
    if data_array.size == 0:
        return np.array([])
    
    # Calculate Q1 (25th percentile)
    Q1 = np.percentile(data_array, 25)
    
    # Calculate Q3 (75th percentile)
    Q3 = np.percentile(data_array, 75)
    
    # Calculate the Interquartile Range (IQR)
    IQR = Q3 - Q1
    
    # Define the outlier bounds
    lower_bound = Q1 - (iqr_multiplier * IQR)
    upper_bound = Q3 + (iqr_multiplier * IQR)
    
    # Filter the array to keep only the elements within the bounds
    filtered_array = data_array[(data_array >= lower_bound) & (data_array <= upper_bound)]
    
    return filtered_array

def find_offset_like_real_time(signal,window_factor=3,threshold_coeff=5,frame_len=3712):
    """This function takes in the signal skips the first window symbols in order to find a good estimate for the mean and variance of the noise floor
    It then scans the signal looking for peaks. We subtract the peak location from k*3172 to get an estimate of the offset"""
    
    window = window_factor*frame_len
    start_sum = np.sum(signal[:window])
    start_sum_squares = np.sum(signal[:window]**2)
    offsets = []
    current_sum = start_sum
    current_sum_squares = start_sum_squares
    average_window_size = 20 # If I have peaks within this window I will take average of them to find the true peak index
    peak_num = window_factor-1
    last_peak_index = 0
    for index in range(window,len(signal)):
        current_sum = current_sum + signal[index]-signal[index-window]
        current_sum_squares = current_sum_squares + signal[index]**2 - signal[index-window]**2
        current_mean = current_sum/window
        current_std = np.sqrt(current_sum_squares/window - current_mean**2)
        #If the peak is much larger than the current statistics then return that index
        if signal[index]>=current_mean+threshold_coeff*current_std: #I am making sure that mult
            if index - last_peak_index < average_window_size:
                #Taking max over current index and last_peak_index
                if signal[index]>signal[last_peak_index]:
                    # peak_true_index =  index
                    last_peak_index = index
            else:
                #print(last_peak_index)
                #print("the start is",peak_num*frame_len)
                found_offset = last_peak_index - peak_num*frame_len
                if found_offset < 0:
                    while found_offset < 0:
                        peak_num -= 1
                        found_offset = last_peak_index - peak_num*frame_len
                elif found_offset > frame_len:
                    while found_offset > frame_len:
                        peak_num += 1
                        found_offset = last_peak_index - peak_num*frame_len
                offsets.append(found_offset)
                last_peak_index = index
                peak_num += 1


    offsets.append(last_peak_index-peak_num*frame_len)
    offsets_cleaned = remove_outliers_iqr(np.array(offsets))
    #print(offsets_cleaned)
    return int(offsets_cleaned.mean())

# N_symbols = 10000
# E_in = generate_stream(N_symbols,offset=200)
# x_p, y_p = first_eleven()
# pilots = np.column_stack((x_p, y_p))
# print("shape:")
# print(pilots.shape)
# print(E_in.shape)
# print("hello")
# plot_loss_vs_tau_time(E_in, pilots)
