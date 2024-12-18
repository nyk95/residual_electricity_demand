import pandas as pd

df = pd.read_excel('country_list.xlsx',index_col=2)
df_spaceHeat = pd.read_csv('elec_to_spaceHeat_2016_17_18_mean_TWh_IDEES_2021.csv',index_col=0)
df_hotWater = pd.read_csv('elec_to_hotWater_2016_17_18_mean_TWh_IDEES_2021.csv',index_col=0)
df_industry = pd.read_csv('elec_to_industry_2018_TWh.csv',index_col=0)

df_spaceHeat = df_spaceHeat.rename(columns={'space_heat':'elec_to_spaceHeat_source_TWh'})
df_spaceHeat = df_spaceHeat.rename(index={'EL':'GR'})
df_hotWater = df_hotWater.rename(columns={'hot_water':'elec_to_hotWater_source_TWh'})
df_hotWater = df_hotWater.rename(index={'EL':'GR'})
df_industry = df_industry[['total']]
df_industry = df_industry.rename(columns={'total':'elec_to_industry_source_TWh'})
df_industry = df_industry.rename(index={'EL':'GR'})

df = df.join(df_spaceHeat,how='left')
df = df.join(df_hotWater,how='left')
df = df.join(df_industry,how='left')


df.to_excel('calibration_data_residual_demand_MOPO.xlsx')











