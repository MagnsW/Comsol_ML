import numpy as np
from numpy import fft
import matplotlib.pylab as plt
import seaborn as sns


def fk_plot(traces, dt, dx, title, log=True, maxfreq=150000, dbdown=-60):
    nt, nx = traces.shape
    f_array = fft.fftshift(fft.fftfreq(nt, dt))
    k_array = fft.fftshift(fft.fftfreq(nx, dx))*2*np.pi
    fk = fft.fftshift(fft.fft2(traces))

    sns.set_style("white")
    plt.figure(figsize=(6, 4))
    if log:
        plt.pcolormesh(k_array, f_array/1000, 20*np.log10(np.abs(fk) / np.max(np.abs(fk))), cmap='jet')
    else:
        plt.pcolormesh(k_array, f_array/1000, np.abs(fk), cmap='jet')
    plt.colorbar()
    plt.ylim(0, maxfreq)
    plt.clim(dbdown, 0)
    plt.title(title)
    plt.tight_layout()
    plt.show()


def tx_fk_plot(traces, dt, dx, title, log=True, minfreq=0, maxfreq=150, dbdown=-60, multx=1, filename=None):
    nt, nx_native = traces.shape
    nx = nx_native*multx
    t_array = np.arange(nt)*dt
    f_array = fft.fftshift(fft.fftfreq(nt, dt))
    k_array = fft.fftshift(fft.fftfreq(nx, dx))*2*np.pi
    fk = fft.fftshift(fft.fft2(traces, s=[nt, nx]))

    sns.set_style("white")
    plt.figure(figsize=(6, 4))
    plt.subplot(1, 2, 1)
    plt.pcolormesh(np.arange(1, nx_native+1), t_array*1e6, traces, cmap='gray')
    plt.gca().invert_yaxis()
    plt.title('TX')
    plt.ylabel('Time ($\mu s$)')
    plt.xlabel('Sensor number')
    plt.subplot(1, 2, 2)
    if log:
        plt.pcolormesh(k_array, f_array/1000, 20*np.log10(np.abs(fk) / np.max(np.abs(fk))), cmap='jet')
    else:
        plt.pcolormesh(k_array, f_array/1000, np.abs(fk), cmap='jet')
    plt.colorbar()
    plt.ylabel('Frequency (kHz)')
    plt.xlabel('Wavenumber')
    plt.ylim(minfreq, maxfreq)
    plt.clim(dbdown, 0)
    plt.title('FK')
    plt.suptitle(title)
    plt.tight_layout()
    if filename:
        plt.savefig(filename, bbox_inches='tight', dpi=600)
    plt.show()

def make_fk(traces, dt, dx, multx=1):
    nt, nx = traces.shape
    nx = nx*multx
    #t_array = np.arange(nt) * dt
    f_array = fft.fftshift(fft.fftfreq(nt, dt))
    k_array = fft.fftshift(fft.fftfreq(nx, dx))*2*np.pi #To get k in wavenumber
    fk = fft.fftshift(fft.fft2(traces, s=[nt, nx]))
    return fk, k_array, f_array

def make_multiple_fk(xt_data, dt, dx, fmax=120e3, db=True):
    fk = np.zeros(shape=xt_data.shape)
    for i in range(len(xt_data)):
        fk_temp, k_array, f_array = make_fk(xt_data[i,:,:].T, dt, dx)
        if db:
            fk[i,:,:] = 20*np.log10(np.abs(fk_temp.T) / np.max(np.abs(fk_temp.T)))
        else:
            fk[i,:,:] = np.abs(fk_temp.T) / np.max(np.abs(fk_temp.T))
    selection = np.where((f_array<=fmax) & (f_array >= 0))
    return fk[:,:,selection[0]], k_array, f_array[selection[0]]



