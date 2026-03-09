import numpy as np
import matplotlib.pyplot as plt
import cma_utils

def optimise_with_reg_gradient_descent(points,y_true,num_epochs=10000,lr=1e-6):
    a_intial=1
    b_intial=0
    c_intial=0

    a=a_intial
    b=b_intial
    c=c_intial

    loss_history=[]
    for epoch in range(num_epochs):
        loss = np.sum((y_true - a*np.square(points) - b*points - c)**2)
        loss_history.append(loss)
        print("a,b,c are",a,b,c)
       # print(lr * np.sum((a*np.square(points) + b*points + c - y_true)*np.square(points)))
        a -= lr * np.sum((a*np.square(points) + b*points + c - y_true)*np.square(points))
        b -= lr * np.sum((a*np.square(points) + b*points + c - y_true)*points)
        c -= lr * np.sum((a*np.square(points) + b*points + c - y_true))
    
    print("converged a,b,c are",a,b,c)
    plt.plot(loss_history[500:])
    plt.show()
    return a,b,c


def apply_pmd_and_get_optimal_filter_coeffs(E_in, DGD_ps_per_sqrt_km=1.0, L_m=10000, N_sections=20, Rs=32e9, SpS=2 , seed = 0):
    """This function does pmd between the x and y polarisation and 
    returns back the x and y polarisation after pmd
    
    L_m - fiber length
    N_sections - number of fiber sections
    Rs - sampling rate
    Sps - samples per symbol
    """
    def extend_to_2N_size(matrix,N):
        "Assume the matrix is a 2*2 matrix which I want to extend to a 2N*2N matrix"
        return np.kron(matrix, np.eye(N))
    
    def get_product_of_V_A_U(V,A,U,N):
        #print("V,A,U",V.shape,A.shape,U.shape)
        V_extended = extend_to_2N_size(V,N)
        A_extended = extend_to_2N_size(A,N)
        #U_extended = extend_to_2N_size(U,N)
        return np.matmul(np.matmul(V_extended,A_extended),U)

    N_samples = E_in.shape[0]
    SD_tau = np.sqrt(3 * np.pi / 8) * DGD_ps_per_sqrt_km

    tau = (SD_tau * np.sqrt(L_m * 1e-3) / np.sqrt(N_sections)) * 1e-12
    
    print("tau_Ts",tau*Rs)
    w = 2 * np.pi * np.fft.fftshift(np.linspace(-0.5, 0.5, N_samples)) * Rs

    E_V = np.fft.fft(E_in[:, 0])
    E_H = np.fft.fft(E_in[:, 1])

    intial_E_V=E_V
    intiial_E_H=E_H
    if seed:
        np.random.seed(42)
    else:
        pass
    Total = np.eye(2*N_samples,dtype=np.complex64)
    for _ in range(N_sections):
        # Random complex coupling matrices (unitary)
        X = np.random.randn(2, 2) + 1j * np.random.randn(2, 2)
        U, _, Vh = np.linalg.svd(X)
        V = Vh.conj().T
        # print("U is",U)
        # print("THe matmul is",np.matmul(U,U.conj().T))
        # # Rotate fields by U†
        E1 = U[0, 0].conj() * E_V + U[0, 1].conj() * E_H
        E2 = U[1, 0].conj() * E_V + U[1, 1].conj() * E_H

        # Apply differential delay
        E1 *= np.exp(1j * w * tau / 2)
        E2 *= np.exp(-1j * w * tau / 2)

        #print(np.array([[np.exp(1j * w * tau / 2), 0], [0, np.exp(-1j * w * tau / 2)]],dtype=np.complex64))
        combined_delay = np.concatenate([np.exp(1j * w * tau / 2), np.exp(-1j * w * tau / 2)])
        delay_matrix = np.diag(combined_delay)
        #print("delay matrix is",delay_matrix)
        #delay_matrix = np.array([[np.exp(1j * w * tau / 2), 0], [0, np.exp(-1j * w * tau / 2)]],dtype=np.complex64)
        # Rotate by V
        E_V = V[0, 0] * E1 + V[0, 1] * E2
        E_H = V[1, 0] * E1 + V[1, 1] * E2
       
        Total = np.matmul(get_product_of_V_A_U(V,U.conj().T,delay_matrix,N_samples),Total)
    #print("Total unitary check is",np.matmul(Total,Total.conj().T))
    #Getting the inverse of Total which is same as conjugate transpsoe
    Total=Total.conj().T
    
    print("Got E_V is",E_V)
    print()
    
    # Extract vectors directly from the full matrix
    pxx_optimal = np.diagonal(Total, offset=0)[:N_samples]  # First half of main diagonal
    pxy_optimal = np.diagonal(Total, offset=0)[N_samples:]  # Second half of main diagonal

    # offset=N is the diagonal of the top-right block (B)
    pyx_optimal = np.diagonal(Total, offset=N_samples)

    # offset=-N is the diagonal of the bottom-left block (C)
    pyy_optimal = np.diagonal(Total, offset=-N_samples)
    #p#rint("E_v and E_h are",E_V,E_H)
    #print("E_v, E_h from total is",np.matmul(Total,intial_E_V),np.matmul(Total,intiial_E_H))
    E_out_x = np.fft.ifft(E_V)
    E_out_y = np.fft.ifft(E_H)

    return np.column_stack((E_out_x, E_out_y)) , {"pxx": pxx_optimal, "pxy": pxy_optimal, "pyx": pyx_optimal, "pyy": pyy_optimal}


def main():
    N_symbols = 5
    intial_symbols=cma_utils.gen_I_Q_16_qam(N_symbols)
    after_pmd,optimal_filters = apply_pmd_and_get_optimal_filter_coeffs(intial_symbols, DGD_ps_per_sqrt_km=31.6, L_m=10000, N_sections=20, Rs=32e9, SpS=4)
    #print(after_pmd) 
    #print(len(optimal_filters['pxx']))
    corrected_pmd = cma_utils.apply_entire_filters(after_pmd,optimal_filters)
    cma_utils.save_constellation(corrected_pmd,"corrected_pmd")

if __name__=="__main__": main()