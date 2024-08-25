"""
Example of use of the nanWelch module.

David Raus
21/08/24
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
rng = np.random.default_rng()

import nanWelch


def example_nanWelch():
    """
    Example of use of the nanWelch module.
    
    This example is based on the one provided in the scipy.signal.welch documentation (https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.welch.html), with NaN values added to the input signal.
    
    """
    
    fs = 10e3
    N = 1e5
    amp = 2*np.sqrt(2)
    freq = 1234.0
    noise_power = 0.001 * fs / 2
    
    time = np.arange(N) / fs
    x = amp*np.sin(2*np.pi*freq*time)
    x += rng.normal(scale=np.sqrt(noise_power), size=time.shape)
    
    ### Add NaNs in the signal
    x_NaN = np.copy(x)
    x_NaN[int(len(x_NaN)/4):int(len(x_NaN)/3)] = np.nan
    
    
    # =============================================================================
    # PSD    
    # =============================================================================
    ### Parameters of the PSD
    params_PSD = dict()
    params_PSD['N_window'] = 1024
    params_PSD['N_overlap'] = int(params_PSD['N_segment']/2)
    params_PSD['window'] = 'hann'
    
    ### PSD of the signal WITHOUT NaNs
    f, Pxx_den = signal.welch(x, fs, nperseg =params_PSD['N_segment'],noverlap = params_PSD['N_overlap'],window  = params_PSD['window'])                        # PSD using scipy
    PSD, f_PSD,_,FFT_matrice_abs2_norm,_ = nanWelch.nanWelch(x, fs,params_PSD['N_segment'], params_PSD['N_overlap'], type_fenetre = params_PSD['window'])                      # PSD using nanDSP

    ### PSD of the signal WITH NaNs
    f_NaN, Pxx_den_NaN = signal.welch(x_NaN, fs, nperseg = params_PSD['N_segment'],noverlap = params_PSD['N_overlap'],window  = params_PSD['window'])          # PSD using scipy
    PSD_NaN, f_PSD_NaN,_,FFT_matrice_abs2_norm_NaN,_ = nanWelch.nanWelch(x_NaN, fs, params_PSD['N_segment'], params_PSD['N_overlap'], type_fenetre = params_PSD['window'])         # PSD using nanDSP



    # =============================================================================
    #  Plot
    # =============================================================================
    
    plt.figure()
    plt.plot(time,x,label = 'Without NaN')
    plt.plot(time,x_NaN,label = 'With NaN')
    plt.xlabel('t (s)')
    plt.ylabel('amplitude')
    plt.legend()
    
    fig, axs = plt.subplots(1,2,figsize=(14, 5))
    axs[0].pcolormesh(np.arange(0,FFT_matrice_abs2_norm.shape[0]),f_PSD, FFT_matrice_abs2_norm.T,shading='auto', cmap='jet') 
    axs[0].set_title('Without NaN')
    
    axs[1].pcolormesh(np.arange(0,FFT_matrice_abs2_norm.shape[0]),f_PSD, FFT_matrice_abs2_norm_NaN.T,shading='auto', cmap='jet') 
    axs[1].set_title('With NaN')
    for jj in range(0,2):
        axs[jj].set_yscale('log')
        axs[jj].set_xlabel('$t$')
        axs[jj].set_ylabel('frequency [Hz]')
        # axs[kk,jj].set_ylim(5,6000)
        
    fig,axs = plt.subplots(1,2,figsize = (14,5))
    axs[0].semilogy(f, Pxx_den,label = 'scipy')
    axs[0].semilogy(f_PSD, PSD,'--',label = 'nanWelch')
    axs[0].set_ylim([0.5e-3, 1])
    axs[0].set_xlabel('frequency [Hz]')
    axs[0].set_ylabel('PSD [V**2/Hz]')
    axs[0].set_title('Without NaN')
    axs[0].legend()
    
    axs[1].semilogy(f_NaN, Pxx_den_NaN,label = 'scipy')
    axs[1].semilogy(f_PSD_NaN, PSD_NaN,'--',label = 'nanWelch')
    axs[1].set_ylim([0.5e-3, 1])
    axs[1].set_xlabel('frequency [Hz]')
    axs[1].set_ylabel('PSD [V**2/Hz]')
    axs[1].set_title('With NaN')
    axs[1].legend()
    plt.show()
    
    

if __name__ == '__main__':                  
    
    example_nanWelch()

















