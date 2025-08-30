import numpy as np
import matplotlib.pyplot as plt

class Constellation:
    def __init__(self,number_timesteps,omega,angle_noise_variance) -> None:
        self.number_timesteps = number_timesteps
        self.omega=omega
        self.angle_noise_variance=angle_noise_variance
        self.basis=[np.pi/4,3*np.pi/4,5*np.pi/4,7*np.pi/4]
        #self.radius=np.sqrt(2)

    def generate_constellations(self):
        constellation_array=np.zeros(self.number_timesteps)
        noise=np.random.normal(0,np.sqrt(self.angle_noise_variance),self.number_timesteps)
        for i in range(self.number_timesteps):
            phase=self.omega*i+noise[i]
            basis_num=np.random.randint(0,4)
            constellation_array[i]=(self.basis[basis_num]+phase)%(2*np.pi)

        return constellation_array
    
    def plot_constellation_animation(self,constellation_array):
        plt.scatter(np.cos(constellation_array)[:5],np.sin(constellation_array)[:5])
        print(np.cos(constellation_array))
        plt.show()


const=Constellation(1000,0.1,angle_noise_variance=0.0001)
constellation_array=const.generate_constellations()
const.plot_constellation_animation(constellation_array)
