The nanWelch module allows to compute the Power Spectral Density (PSD) of a signal containing NaN values, by using the Welch algorithm.
The windows containing NaN values are simply not used when computing the averaged spectra.

## Example 

The example nanWelch_example is based on the one provided in the scipy.signal.welch documentation (https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.welch.html).
Few NaN values are added to the input signal, the scipy.signal.welch module thus produces a PSD filled with NaN while the nanWelch module still allows to compute the PSD.

![example_signals](https://github.com/user-attachments/assets/c2409efe-03c9-4121-95df-9a6a1024d510)

![result_nanWelch](https://github.com/user-attachments/assets/e0c4bbfd-8da2-4f66-ac10-b7f1c5767731)
