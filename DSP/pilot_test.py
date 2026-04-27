import cma_utils_pilot
import cma_utils
import numpy as np
import matplotlib.pyplot as plt

def test_pilot_correlation():
    N_symbols = 20000
    offset = 200
    
    # 1. Generate stream with pilots
    E_in = cma_utils_pilot.generate_stream(N_symbols, offset=offset)
    
    # 2. Get the known pilot sequence (the first 11 symbols of the frame)
    pilot_x, pilot_y = cma_utils_pilot.first_eleven()
    
    # 3. Correlate the generated stream with the 11-symbol pilot
    # np.correlate computes the cross-correlation (sliding dot product)
    correlation_x = np.correlate(E_in[:, 0], pilot_x, mode='valid')
    correlation_y = np.correlate(E_in[:, 1], pilot_y, mode='valid')
    
    # 4. Plot the magnitude of the correlation
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    
    frame_len = 3712
    expected_peak = frame_len - offset
    
    axes[0].plot(np.abs(correlation_x))
    axes[0].axvline(x=expected_peak, color='r', linestyle='--', label=f'Expected 1st Peak ({expected_peak})')
    axes[0].set_title(f"Cross-correlation with 11-symbol pilot (X pol) | Offset = {offset}")
    axes[0].set_ylabel("Magnitude")
    axes[0].legend()
    axes[0].grid(True)
    
    axes[1].plot(np.abs(correlation_y))
    axes[1].axvline(x=expected_peak, color='r', linestyle='--', label=f'Expected 1st Peak ({expected_peak})')
    axes[1].set_title(f"Cross-correlation with 11-symbol pilot (Y pol) | Offset = {offset}")
    axes[1].set_xlabel("Lag (samples)")
    axes[1].set_ylabel("Magnitude")
    axes[1].legend()
    axes[1].grid(True)
    
    plt.tight_layout()
    plt.show()

def lms_test(E_distorted,pilot_sequence):
    #print("E_in shape:", E_in.shape)
    pass

def main():
    test_pilot_correlation()

if __name__ == "__main__":
    main()