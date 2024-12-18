import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os


file_overview = 'calibration_processing/calibration_data_residual_demand_MOPO.xlsx'
df_overview = pd.read_excel(file_overview,index_col=0)
reg_list = df_overview.index.values

df_overview['entire_elec_TWh'] = np.nan
df_overview['elec_to_spaceHeat_est_TWh'] = np.nan
df_overview['classic_est_TWh'] = np.nan
df_overview['elec_to_spaceHeat_est_perc'] = np.nan
df_overview['elec_to_spaceHeat_source_perc'] = np.nan
df_overview['elec_to_hotWater_source_perc'] = np.nan
df_overview['elec_to_industry_source_perc'] = np.nan



for reg in reg_list:
    print(reg)    
    file_entire_demand = 'Data/Hindcast 2016 2018/'+reg+' Hindcast_entire_electricity_demand.csv'
    file_classic_demand = 'Data/Hindcast 2016 2018/'+reg+' Hindcast_classic_electricity_demand_est.csv'
    #file_entire_demand = 'Data/Hindcast output/'+reg+'/'+reg+' Hindcast_entire_electricity_demand.csv'
    #file_classic_demand = 'Data/Hindcast output/'+reg+'/'+reg+' Hindcast_classic_electricity_demand_est.csv'
    
    if os.path.exists(file_entire_demand):  

        df_entire = pd.read_csv(file_entire_demand,index_col=0,parse_dates=True)
        df_classic = pd.read_csv(file_classic_demand,index_col=0,parse_dates=True)
        df_elec_spaceHeat = df_entire - df_classic

        annual_elec = df_entire['2016':'2018'].sum().item()/(1e6*3)
        annual_elec_spaceHeat = df_elec_spaceHeat['2016':'2018'].sum().item()/(1e6*3)
        annual_elec_classic = df_classic['2016':'2018'].sum().item()/(1e6*3)

        df_overview.loc[reg,'elec_to_spaceHeat_est_TWh'] = annual_elec_spaceHeat
        df_overview.loc[reg,'entire_elec_TWh'] = annual_elec
        
        df_overview.loc[reg,'elec_to_spaceHeat_est_perc'] = 100*(annual_elec_spaceHeat/annual_elec)
        if  not pd.isna(df_overview.loc[reg,'elec_to_spaceHeat_source_TWh']):
            df_overview.loc[reg,'elec_to_spaceHeat_source_perc'] = 100*(df_overview.loc[reg,'elec_to_spaceHeat_source_TWh']/annual_elec)
            df_overview.loc[reg,'elec_to_hotWater_source_perc'] = 100*(df_overview.loc[reg,'elec_to_hotWater_source_TWh']/annual_elec)
        if  not pd.isna(df_overview.loc[reg,'elec_to_industry_source_TWh']):    
            df_overview.loc[reg,'elec_to_industry_source_perc'] = 100*(df_overview.loc[reg,'elec_to_industry_source_TWh']/annual_elec)

    else:
        print(reg+' demand file doesnt exsit')


cols = ['EU status','Country_name','elec_to_spaceHeat_est_perc','elec_to_spaceHeat_source_perc',
        'elec_to_hotWater_source_perc','elec_to_industry_source_perc','elec_to_spaceHeat_est_TWh','elec_to_spaceHeat_source_TWh',
        'elec_to_hotWater_source_TWh','elec_to_industry_source_TWh','entire_elec_TWh']

df_overview = df_overview[cols]


# assumption on regions not available in source
# reg_assumption_based = ['UK','CH','NO','RS','ME','MT']:
    
df_overview.loc['UK','elec_to_hotWater_source_TWh'] = df_overview.loc['IE','elec_to_hotWater_source_perc']*df_overview.loc['UK','entire_elec_TWh']/100   
df_overview.loc['CH','elec_to_hotWater_source_TWh'] = df_overview.loc['AT','elec_to_hotWater_source_perc']*df_overview.loc['CH','entire_elec_TWh']/100   
df_overview.loc['NO','elec_to_hotWater_source_TWh'] = df_overview.loc['SE','elec_to_hotWater_source_perc']*df_overview.loc['NO','entire_elec_TWh']/100 
hot_water_perc_avg =   np.mean([df_overview.loc['HU','elec_to_hotWater_source_perc'],df_overview.loc['HR','elec_to_hotWater_source_perc'],df_overview.loc['BG','elec_to_hotWater_source_perc'],df_overview.loc['RO','elec_to_hotWater_source_perc']])
df_overview.loc['RS','elec_to_hotWater_source_TWh'] = hot_water_perc_avg*df_overview.loc['RS','entire_elec_TWh']/100   
df_overview.loc['ME','elec_to_hotWater_source_TWh'] = hot_water_perc_avg*df_overview.loc['ME','entire_elec_TWh']/100
df_overview.loc['MT','entire_elec_TWh'] = 2.276 # from Eurostats 
df_overview.loc['MT','elec_to_spaceHeat_est_TWh'] = df_overview.loc['IT','elec_to_spaceHeat_est_perc']*df_overview.loc['MT','entire_elec_TWh']/100 # from Eurostats 
   
df_overview.loc['RS','elec_to_industry_source_TWh'] = 0   
df_overview.loc['ME','elec_to_industry_source_TWh'] = 0   
  
            
          







df_overview.to_excel(file_overview)


