import numpy as np
import pandas as pd
import os
import scipy


def read_waveforms(pathname, longname=True):
    df_waveforms = pd.DataFrame()
    df_time = pd.DataFrame()
    for (dirpath, dirnames, filenames) in os.walk(pathname):
        for filename in filenames:
            print(filename)
            df_temp = pd.read_csv(pathname + filename, header=[0, 1])
            df_temp.columns = df_temp.columns.droplevel(1)
            df_time = df_temp[['Time']]
            df_temp.drop(columns=['Time', 'Channel A'], inplace=True)
            fname, ext = filename.split('.')
            if longname:
                _, _, colname = fname.split('-')
            else:
                _, colname = fname.split('-')
            df_temp.rename(columns={'average(A)': colname}, inplace=True)
            df_waveforms = pd.concat([df_waveforms, df_temp], axis=1)

    return df_waveforms, df_time

def read_synth(filename, header, firsttrace=0, decim=6, geom_spread=True):
    '''

    :param filename:
    :param header: usually 'concat_traces' or 'trace_pos_clock'
    :param geom_spread:
    :return:
    '''
    mat = scipy.io.loadmat(filename)
    X = mat[header].astype('float32')
    X = np.swapaxes(X, 2, 1)
    X = X[:,:-1,:] #Removes repeated trace
    X = X[:,firsttrace::decim,:]
    time_scaling = np.sqrt(np.arange(1, X.shape[-1] + 1, dtype='float32'))
    time_scaling = np.expand_dims(time_scaling, -1).T
    if geom_spread:
        X = np.divide(X, time_scaling)

    return X