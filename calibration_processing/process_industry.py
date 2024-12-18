import pandas as pd
import matplotlib.pyplot as plt


pj_to_TWh_factor = 0.27778

df_ind_chem = pd.read_excel('FW__mopo_industry_data_update/diff_chem_total_aidres_idees_eurostat.xlsx')
df_fec_cement = pd.read_excel('FW__mopo_industry_data_update/diff_fec_cement.xlsx')
df_fec_glass = pd.read_excel('FW__mopo_industry_data_update/diff_fec_glass.xlsx')
df_steel_finishing_rolling = pd.read_excel('FW__mopo_industry_data_update/diff_with_finishing_rolling_fuel.xlsx')

df_no_uk_ch = pd.read_excel('NO_UK_CH_electricity_use.xlsx')

df_ind_chem = df_ind_chem.loc[df_ind_chem['fuel']=='electricity'].set_index('country_code')['aidres']
df_fec_cement = df_fec_cement.loc[df_fec_cement['fuel']=='electricity'].set_index('country_code')['aidres']
df_fec_glass = df_fec_glass.loc[df_fec_glass['fuel']=='electricity'].set_index('country_code')['aidres']
df_steel_finishing_rolling = df_steel_finishing_rolling.loc[df_steel_finishing_rolling['fuel']=='electricity'].set_index('country_code')['aidres']


df_ind = df_ind_chem.rename('chem').to_frame().join(df_fec_cement.rename('cement'),how='left')
df_ind = df_ind.join(df_fec_glass.rename('glass'),how='left')
df_ind = df_ind.join(df_steel_finishing_rolling.rename('steel'),how='left')

df_ind = df_ind*pj_to_TWh_factor
df_ind['total'] = df_ind.sum(axis=1)
df_no_uk_ch = df_no_uk_ch.groupby('country_code').sum()['total aidres electricity use in mwh'].rename('total').to_frame()/1e6

df_ind = pd.concat([df_ind,df_no_uk_ch],axis=0).sort_index()


df_ind.to_csv('elec_to_industry_2018_TWh.csv')


