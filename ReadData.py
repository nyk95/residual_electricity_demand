# Libraries
import pandas as pd
import numpy as np

# Custom libraries
import Setup as setup


def demand_measured(m):
    file = setup.measured_demand_folder + '/demand_' + m.region + '.csv'
    df = pd.read_csv(file)
    df = df.set_index('Time') 
    df.index = pd.to_datetime(df.index, format='%Y-%m-%d %H:%M:%S') 
    df_slice = df.loc[m.t_start:m.t_end]

    return df_slice


def temperature(m):
    file = setup.temperature_file
    df = pd.read_csv(file)
    df = df.set_index('Time')  
    df.index = pd.to_datetime(df.index, format='%Y-%m-%d %H:%M:%S')

    col = m.geo_points.index.astype(str)
    start = pd.Timestamp(f'{m.t_start}-01-01 00:00') - pd.DateOffset(hours=setup.t_offset)
    end = pd.Timestamp(f'{m.t_end}-12-31 23:00')
    df_slice = df[col].loc[start:end]
    df_slice = df_slice.add_prefix('Temp_')

    return df_slice
