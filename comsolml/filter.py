from scipy.signal import butter, filtfilt


def butter_lowpass_filter(data, cutoff, order, fs=1e6):

    #fs = 1/1e-6#sample rate in Hz
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    #print(normal_cutoff)
    # Get the filter coefficients
    b, a = butter(order, normal_cutoff, btype='lowpass', analog=False)
    y = filtfilt(b, a, data.T, axis=-1)
    return y.T