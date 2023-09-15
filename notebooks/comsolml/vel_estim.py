# -*- coding: utf-8 -*-
"""
Created on Wed Sep 15 11:35:58 2021

@author: Magnus
"""

import numpy as np
import scipy.interpolate
#import scipy
from scipy.fftpack import fft
import matplotlib.pyplot as plt
from .lamb import Lamb
#import pandas as pd

def define_windows(df, window1_start, window2_start, window_length=130):
    df_s1 = df.iloc[window1_start:window1_start+window_length]
    df_s2 = df.iloc[window2_start:window2_start+window_length]
    return df_s1, df_s2

def compute_delt(window1_start, window2_start, fs):
    delt = (window2_start - window1_start)/fs
    return delt

def return_center_freqs(df):
    fc = np.array([])
    for item in list(df.columns):
        fc = np.append(fc, [int(item[:3])*1000])
        #fc = np.append(fc, [int(item[:3])])
    return fc

def make_lamb_curves(E=205e9, p=7850, v=0.28, d=6.8):
    c_L = np.sqrt(E*(1-v) / (p*(1+v)*(1-2*v)))
    c_S = np.sqrt(E / (2*p*(1+v)))
    c_R = c_S * ((0.862+1.14*v) / (1+v))

    steel = Lamb(thickness=d, 
            nmodes_sym=5, 
            nmodes_antisym=5, 
            fd_max=10000, 
            vp_max=15000, 
            c_L=c_L, 
            c_S=c_S, 
            c_R=c_R, 
            material='Steel')
    return steel

def phase_velocity_estimation(df_s1, df_s2, delt, delz, fs, fc, wm_interpolator, d):
    #Frequency scan case
    L = 1000
    Sw1 = fft(df_s1, n=L, axis=0)
    Sw2 = fft(df_s2, n=L, axis=0)
    f = fs*np.arange(0, L, 1)/L
    
    infc = np.array([], dtype='int64')
    for fci in fc:
        infc = np.append(infc, np.where(f==fci)[0][0])
        
    w = 2*np.pi*f
    
    ang = np.unwrap(np.angle(Sw2/Sw1), axis=0)
    vphi = delz/(delt-(ang.T/w).T)
    
    plt.figure(figsize=(12, 8))
    plt.plot(f/1000, vphi)
    plt.title('Uncorrected phase velocity spectrums')
    plt.xlabel('kHz')
    plt.ylabel('Velocity')
    plt.grid(which='both')
    plt.show()
    
    vphi_fc = np.array([])
    for i, infc_item in enumerate(infc):
        vphi_fc = np.append(vphi_fc, [vphi[infc_item, i]])
        
    
    vfap_fc = wm_interpolator(fc/1000*d)
    
    plt.figure(figsize=(12, 8))
    plt.plot(fc/1000, vphi_fc)
    plt.plot(fc/1000, vfap_fc)
    plt.grid(which='both')
    plt.legend(['Uncorrected phase velocity', 'Theoretical dispersion curve'])
    plt.xlabel('kHz')
    plt.ylabel('Velocity')
    plt.show()
    
    #Pi skip correction
    ang_apri = np.array([])
    #ang_corrected = np.array([])
    for k in range(len(fc)):
        ang_apri = np.append(ang_apri, 2*np.pi*fc[k]*(delt - delz/vfap_fc[k]))
        if np.abs(ang[infc[k], k] - ang_apri[k]) > np.pi:
            #ang_correction = np.round((ang_apri[k] - ang[infc[k], k])/2/np.pi)*2*np.pi
            ang[:,k] = np.round((ang_apri[k] - ang[infc[k], k])/2/np.pi)*2*np.pi + ang[:,k]
            #ang_corrected = np.append(ang_corrected, np.round((ang_apri[k] - ang[infc[k], k])/2/np.pi) + ang[k])
            #print(k, infc[k], ang[60,k], ang_uncorrected[60,k], ang_correction)
    vphi = delz/(delt-(ang.T/w).T)
    
    vphi_fc = np.array([])
    for i, infc_item in enumerate(infc):
        vphi_fc = np.append(vphi_fc, [vphi[infc_item, i]])
        
    plt.figure(figsize=(12, 8))
    plt.plot(f/1000, vphi)
    plt.title('Corrected phase velocity spectrums')
    plt.xlabel('kHz')
    plt.ylabel('Velocity')
    plt.grid(which='both')
    plt.show()
        
    plt.figure(figsize=(12, 8))
    plt.plot(fc/1000, vphi_fc)
    plt.plot(fc/1000, vfap_fc)
    plt.grid(which='both')
    plt.legend(['Corrected phase velocity', 'Theoretical dispersion curve'])
    plt.xlabel('kHz')
    plt.ylabel('Velocity')
    plt.show()
    return vphi_fc, fc/1000
            
def do_aliasing(curve, kNyq):
    aliased = True
    while aliased:
        curve = np.where(curve < kNyq, curve, curve - 2*kNyq)
        curve = np.where(curve > -kNyq, curve, curve + 2*kNyq)
        if (np.max(curve) < kNyq) & (np.min(curve) > -kNyq):
            aliased = False
    return curve

def do_aliasing_with_nan(curve, kNyq, f_array_dispersion):
    '''
    
    :param curve: Unaliased curve
    :param kNyq: Nyquist wavenumber
    :param f_array_dispersion: Frequency array of curve
    :return: Aliased curve and new frequency array with NaN inserted at aliasing locations
    '''
    disp_curve_temp = do_aliasing(curve, kNyq)
    nan_positions = np.where(np.abs(np.diff(disp_curve_temp)) > np.mean(np.abs(np.diff(disp_curve_temp)))*2)[0] + 1
    disp_curve_temp = np.insert(disp_curve_temp, nan_positions, np.nan)
    f_array_dispersion_temp = np.insert(f_array_dispersion, nan_positions, np.nan)
    return disp_curve_temp, f_array_dispersion_temp

def conv_m_to_rad(meters, circumference):
    theta = 2*np.pi*meters/circumference
    return theta


def make_circ_wall_middle(c_outer, d):
    r_outer = c_outer/(2*np.pi)
    r_middle = r_outer - d/2
    c_middle = r_middle*2*np.pi
    return c_middle

def FK_thickness_estimation(fk, k_array, f_array, dx, disp_curve_interpolator, ds, minfreq, maxfreq, angular=False,
                            circumference=None, circ_correction=False, plot=True, d_true=None, filename=None,
                            return_power_curve=False, db=True, db_down=-60, aux_lines=None, slim_fig=False, tx=None):
    '''

    :param fk: FK domain data in absolute amplitude, 2D array. Linear or dB scale.
    :param k_array: Wavenumber array associated with FK
    :param f_array: Frequency array associated with FK
    :param dx: Receiver sampling
    :param disp_curve_interpolator: Dispersion curve interpolator
    :param ds: Array of investigated thicknesses
    :param minfreq: Minimum frequency for analysis
    :param maxfreq: Maximum frequency for analysis
    :param circumference: Outer circumference
    :return:
    '''
    kNyq = 2*np.pi/dx/2 #Nyquist wavenumber calculated from dx or dtheta, whichever is input into the dx argument

    f_array_dispersion = f_array[f_array >= minfreq]
    f_array_dispersion = f_array_dispersion[f_array_dispersion <= maxfreq]

    disp_curves = {}
    for d in ds:
        if angular:
            if circ_correction:
                disp_curves[round(d, 3)] = 2*np.pi/conv_m_to_rad(np.divide(disp_curve_interpolator(f_array_dispersion / 1000 * d),
                                                     f_array_dispersion), circumference=make_circ_wall_middle(circumference, d*1e-3))
            else:
                disp_curves[round(d, 3)] = 2 * np.pi / conv_m_to_rad(
                    np.divide(disp_curve_interpolator(f_array_dispersion / 1000 * d),
                              f_array_dispersion), circumference=circumference)
        else:
            disp_curves[round(d, 3)] = 2*np.pi/np.divide(disp_curve_interpolator(f_array_dispersion/1000*d), f_array_dispersion)


    interpolator2d = scipy.interpolate.RectBivariateSpline(f_array, k_array, fk)

    curves_sum = {}
    for curve in disp_curves:
        curves_sum[curve] = 0
        for i in range(len(f_array_dispersion)):
            curves_sum[curve] += interpolator2d(f_array_dispersion[i], do_aliasing(disp_curves[curve], kNyq)[i])[0][0]
            curves_sum[curve] += interpolator2d(f_array_dispersion[i], do_aliasing(-disp_curves[curve], kNyq)[i])[0][0]

    d_estimate = max(curves_sum, key=curves_sum.get)

    if plot:
        if slim_fig:
            plt.figure(figsize=(8,3))
            plt.subplot(1,4,1)
            if tx is not None:
                plt.pcolormesh(tx, cmap='gray', rasterized=True)
                plt.gca().invert_yaxis()
                plt.xlabel('Trace number')
                plt.tick_params(left=False, labelleft=False)
                plt.title('TX')

            else:
                plt.pcolormesh(k_array, f_array / 1000, fk, cmap='jet', rasterized=True)
                plt.ylim(0, maxfreq / 1000)
                plt.xlabel('Wavenumber')
                plt.title('FK')

        else:
            plt.figure(figsize=(6, 6))
            plt.subplot(2,2,1)
            plt.pcolormesh(k_array, f_array/1000, fk, cmap='jet')
            plt.ylim(0, maxfreq/1000)
            plt.title('FK plot')
            plt.ylabel('Frequency (kHz)')
            plt.xlabel('Wavenumber')
            if db:
                plt.clim(db_down, 0)
                plt.colorbar()



        if slim_fig and tx is not None:
            plt.subplot(1,4,2)
            plt.pcolormesh(k_array, f_array / 1000, fk, cmap='jet', rasterized=True)
            disp_curve_select, f_array_dispersion_select = do_aliasing_with_nan(disp_curves[round(d_estimate, 3)], kNyq,
                                                                                f_array_dispersion)
            plt.plot(disp_curve_select, f_array_dispersion_select / 1000, 'r-', linewidth=1, alpha=0.5)
            disp_curve_select, f_array_dispersion_select = do_aliasing_with_nan(-disp_curves[round(d_estimate, 3)],
                                                                                kNyq,
                                                                                f_array_dispersion)
            plt.plot(disp_curve_select, f_array_dispersion_select / 1000, 'r-', linewidth=1, alpha=0.5)
            plt.title('FK')
        else:
            plt.subplot(2,2,2)
            plt.pcolormesh(k_array, f_array/1000, fk, cmap='jet', rasterized=True)
            for curve in disp_curves:
                disp_curve_temp, f_array_dispersion_temp = do_aliasing_with_nan(disp_curves[curve], kNyq,
                                                                                f_array_dispersion)
                plt.plot(disp_curve_temp, f_array_dispersion_temp / 1000, "-", alpha=0.25)
                disp_curve_temp, f_array_dispersion_temp = do_aliasing_with_nan(-disp_curves[curve], kNyq,
                                                                                f_array_dispersion)
                plt.plot(disp_curve_temp, f_array_dispersion_temp / 1000, "-", alpha=0.25)
            disp_curve_select, f_array_dispersion_select = do_aliasing_with_nan(disp_curves[round(d_estimate, 3)], kNyq, f_array_dispersion)
            plt.plot(disp_curve_select, f_array_dispersion_select / 1000, 'r-', linewidth=3)
            disp_curve_select, f_array_dispersion_select = do_aliasing_with_nan(-disp_curves[round(d_estimate, 3)], kNyq,
                                                                                f_array_dispersion)
            plt.plot(disp_curve_select, f_array_dispersion_select / 1000, 'r-', linewidth=3)
        plt.ylim(0, maxfreq/1000)
        plt.xlabel('Wavenumber')
        plt.xlim(k_array.min(), k_array.max())
        plt.tick_params(left=False, labelleft=False)
        #plt.ylabel('Frequency (kHz)')
        if not slim_fig:
            plt.title('FK plot with evaluated dispersion curves')

        if db:
            plt.clim(db_down, 0)
            plt.colorbar()
        #plt.tight_layout()

        if slim_fig:
            plt.subplot(1,4,(3,4))
        else:
            plt.subplot(2,2,(3,4))
        plt.plot(curves_sum.keys(), curves_sum.values(), 'o-', label='Stacking Power')
        plt.axvline(x=d_estimate, color='r', label='Estimated Thickness: ' + str(d_estimate) + ' mm')
        if aux_lines:
            plt.vlines(x=aux_lines, ymin=plt.axis()[2], ymax=plt.axis()[3], colors='k', linestyles='dashed')

        if d_true:
            plt.axvline(x=d_true, color='g', linestyle='--', label='True Thickness: ' + str(d_true) + ' mm')
        plt.grid(visible=True, which='major')
        plt.grid(visible=True, which='minor', alpha=0.5)
        plt.minorticks_on()
        if slim_fig:
            plt.xticks(ticks=ds[::len(ds)//20], rotation=45)
        else:
            plt.xticks(ticks=ds[::len(ds)//25], rotation=45)
        plt.xlabel('Corresponding Thickness (mm)')
        if not slim_fig:
            if db:
                plt.ylabel('Amplitude (dB)')
            else:
                plt.ylabel('Amplitude')
        if slim_fig:
            plt.tick_params(right=True, labelright=True, labelleft=False)
            plt.title('Stacking Power Curve')
        else:
            plt.title('Selection of Best Fitting Dispersion Curve')

        plt.legend()
        plt.tight_layout()
        if filename:
            plt.savefig(filename, dpi=300)
        plt.show()
    if return_power_curve:
        return d_estimate, curves_sum
    else:
        return d_estimate


