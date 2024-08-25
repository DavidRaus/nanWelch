"""
David Raus
21/08/24
"""
import numpy as np


def nanWelch(signal,f_s,N_window ,N_overlap,type_fenetre):
    """
    Power Spectral Density (PSD) of a signal by using the Welch algorithm
    Unlike signal.welch function, this function accepts the presence of NaN in the initial signal:
          If NaN is present in a window, the fft of that segment is not used in the final average. 
    
    Parameters
    ----------
    signal : 1D array of floats
        signal.
    f_s : float
        sampling frequency of the signal
    N_window : int
        Number of samples in the window
    N_overlap : int
        Number of samples of overlap. Should respect:
                N_overlap = int(N_window - N_window/K) with K an integer
                so the result corresponds to the result of the function signal.welch

    type_window : str
        Name of the window to use
        
    Returns
    -------
    PSD : 1D array of floats
        Power Spectral Density 
    freq_fft : 1D array of floats
        frequencies corresponding to the PSD

    """
    
    ### Creation of the window and extraction of its parameters
    window,amplitude_correction,energy_correction,ENBW = create_window(N_window ,type_fenetre)

    ### Reshape the signal into an array where each line is a segment of length N_window with overlap
    signal_dim = reshape_overlap(signal,N_window,N_overlap)
    
    ### Subtraction of local average
    signal_mean_segment = np.mean(signal_dim,axis=1)             
    signal_mean_segment = np.expand_dims(signal_mean_segment,axis = 1)
    signal_dim = signal_dim - signal_mean_segment
    
    ### Windowing each segment
    signal_dim_window = signal_dim * window
    
    ### Fast fourier transform of each row of the array
    freq_fft = f_s*np.arange(0,int(N_window /2)+1)/N_window                      
    FFT_matrice =  np.fft.rfft(signal_dim_window,axis=1)
    # FFT_matrice =  FFT_matrice[0:len(freq_fft)]
    
    FFT_array_scale = amplitude_correction * (2/N_window) * np.abs(FFT_matrice)    # *2 for the energy in the negative frequencies and (1/N) for rescaling
    FFT_array_abs2 = np.real(FFT_matrice) ** 2 + np.imag(FFT_matrice) ** 2   # faster than np.abs(x)**2
    FFT_array_abs2_norm = FFT_array_abs2 / (np.sum(window * window))     # compensate the energy loss by windowing the signal

    FFT_array_abs2_norm[:,1:-1] = FFT_array_abs2_norm[:,1:-1] * 2          # recover the energy lost by deleting the spectra in the negative frequencies 
                            
    # Average of the segments
    PSD = np.nanmean(FFT_array_abs2_norm,axis = 0)                           # average of spectra (thanks to 'nanmean', spectra from segments with NaN are not included in the average)

    # Scaled to match signal.welch's result (and to respect Parseval's theorem) 
    # DSP = ENBW * DSP/ f_s                     # If you multiply by ENBW, the peaks have the right amplitude, but Parseval is no longer respected
    PSD = PSD/ f_s       
    
    
    ### Checking Parseval theorem
    energie_time = np.mean((signal-np.mean(signal))**2)
    energie_freq = PSD.sum()*(freq_fft[1] - freq_fft[0])
    if abs(energie_time - energie_freq) > 1:
        print("Beware, Parseval\'s theorem is not respected after the PSD calculation!")
    
    
    
    return PSD, freq_fft,FFT_array_scale,FFT_array_abs2_norm,ENBW


# =============================================================================
# Utils
# =============================================================================

def create_window(N,type_window):
    """
    Create a window function for windowing a signal
    
    Parameters
    ----------
    N : int
        Number of sample in the signal.
    type_window : str
        Name of the window (we use here the )

    Returns
    -------
    window : list of float
        window function
    amplitude_correction : float
        Correction coefficient of the amplitude of the spectra
    energy_correction : float
        Correction coefficient of the energy of the spectra
    ENBW: list of float
        Equivalent noise bandwith 
    """
    if (type_window == "rectangle") or (type_window == "boxcar"):       
        window = np.ones(N)
        amplitude_correction = 1
        energy_correction = 1
        
    elif (type_window == "hann") or (type_window == "hanning"):
        window = np.hanning(N)
        amplitude_correction = 2
        energy_correction = 1.63
        
    elif type_window == "blackman":
        window = np.blackman(N)    
        amplitude_correction = 2.38
        energy_correction = 1.81
        
    elif type_window == "hamming":
        window = np.hamming(N)
        amplitude_correction = 1.85
        energy_correction = 1.59
        
    ENBW = len(window) * np.sum(window**2)/(np.sum(window)**2)

    return window, amplitude_correction, energy_correction,ENBW    



def reshape_overlap(x,N_segment,N_overlap):
    """
    Reshape a 1D array into a 2D array, with overlap between the different rows of the array

    Parameters
    ----------
    x : 1D array of floats
        signal en entrée
    N_segment : int 
        number of points of the segment.
    N_overlap : int
        number of points of overlap

    Returns
    -------
    x_redim : 2D array of floats
        Reshaped array.

    """
    ### Step between two segments
    step = N_segment - N_overlap                                               
        
    ### Reshape the array with overlap between each segment
    x_redim = np.copy([x[ii : ii + N_segment] for ii in range(0, len(x) - N_segment, step)]) 
    
    ### Transform the list of vectors in an array 
    x_redim = np.stack(x_redim)    
    
    return x_redim

