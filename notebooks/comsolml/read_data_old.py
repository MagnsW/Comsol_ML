import pandas as pd
import os


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