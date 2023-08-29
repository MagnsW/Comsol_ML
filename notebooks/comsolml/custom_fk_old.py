import numpy as np
from numpy import fft
import matplotlib.pylab as plt
import seaborn as sns


def fk_plot(traces, dt, dx, title, log=True, maxfreq=150000, dbdown=-12):
    nt, nx = traces.shape
    f_array = fft.fftshift(fft.fftfreq(nt, dt))
    x_array = fft.fftshift(fft.fftfreq(nx, dx))
    fk = fft.fftshift(fft.fft2(traces))

    sns.set_style("white")
    plt.figure(figsize=(12, 8))
    if log:
        plt.pcolormesh(x_array, f_array, np.log(np.abs(fk) / np.max(np.abs(fk))), cmap='jet')
    else:
        plt.pcolormesh(x_array, f_array, np.abs(fk), cmap='jet')
    plt.colorbar()
    plt.ylim(0, maxfreq)
    plt.clim(dbdown, 0)
    plt.title(title)
    plt.show()


def tx_fk_plot(traces, dt, dx, title, log=True, minfreq=0, maxfreq=150, dbdown=-12):
    nt, nx = traces.shape
    t_array = np.arange(nt)*dt
    f_array = fft.fftshift(fft.fftfreq(nt, dt))
    x_array = fft.fftshift(fft.fftfreq(nx, dx))
    fk = fft.fftshift(fft.fft2(traces))

    sns.set_style("white")
    plt.figure(figsize=(12, 8))
    plt.subplot(1, 2, 1)
    plt.pcolormesh(np.arange(1, nx+1), t_array*1e6, traces, cmap='gray')
    plt.gca().invert_yaxis()
    plt.title('TX')
    plt.ylabel('microseconds')
    plt.subplot(1, 2, 2)
    if log:
        plt.pcolormesh(x_array, f_array/1000, np.log(np.abs(fk) / np.max(np.abs(fk))), cmap='jet')
    else:
        plt.pcolormesh(x_array, f_array/1000, np.abs(fk), cmap='jet')
    plt.colorbar()
    plt.ylabel('kHz')
    plt.ylim(minfreq, maxfreq)
    plt.clim(dbdown, 0)
    plt.title('FK')
    plt.suptitle(title)
    plt.show()

def make_fk(traces, dt, dx):
    nt, nx = traces.shape
    #t_array = np.arange(nt) * dt
    f_array = fft.fftshift(fft.fftfreq(nt, dt))
    x_array = fft.fftshift(fft.fftfreq(nx, dx))
    fk = fft.fftshift(fft.fft2(traces))
    return fk, x_array, f_array
