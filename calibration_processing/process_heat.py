import pandas as pd
import matplotlib.pyplot as plt



reg_list = ['AT','BE','BG','CY','CZ','DE','DK','EE','EL',
            'ES','FI','FR','HR','HU','IE','IT','LT','LU',
            'LV','MT','NL','PL','PT','RO','SE','SI','SK']

ktoe_to_TWh_factor = 0.01163

df_res_hotWater = pd.DataFrame(index=[2016,2017,2018])
df_res_spaceHeat = pd.DataFrame(index=[2016,2017,2018])
df_ter_hotWater = pd.DataFrame(index=[2016,2017,2018])
df_ter_spaceHeat = pd.DataFrame(index=[2016,2017,2018])


# Residential
print('Residential')
for reg in reg_list:
    print(reg)

    df_res = pd.read_excel('JRC-IDEES-2021 (1)/'+reg+'/JRC-IDEES-2021_Residential_'+reg+'.xlsx',sheet_name='RES_hh_fec')
    df_res_spaceHeat_temp = df_res.iloc[[10,11,12],:]
    df_res_spaceHeat_temp = df_res_spaceHeat_temp[[2016,2017,2018]]
    df_res_spaceHeat_temp = df_res_spaceHeat_temp.sum(axis=0)
    
    df_res_hotWater_temp = df_res.iloc[23,:]
    df_res_hotWater_temp = df_res_hotWater_temp[[2016,2017,2018]]

    df_res_spaceHeat = df_res_spaceHeat.join(df_res_spaceHeat_temp.rename(reg),how='left')
    df_res_hotWater = df_res_hotWater.join(df_res_hotWater_temp.rename(reg),how='left')
    
df_res_spaceHeat = df_res_spaceHeat.mean()   
df_res_spaceHeat = df_res_spaceHeat*ktoe_to_TWh_factor

df_res_hotWater = df_res_hotWater.mean()  
df_res_hotWater = df_res_hotWater*ktoe_to_TWh_factor
      


# Tertiary
print('Tertiary')
for reg in reg_list:
    print(reg)

    df_ter = pd.read_excel('JRC-IDEES-2021 (1)/'+reg+'/JRC-IDEES-2021_Tertiary_'+reg+'.xlsx',sheet_name='SER_hh_fec')
    df_ter_spaceHeat_temp = df_ter.iloc[[11,12,13],:]
    df_ter_spaceHeat_temp = df_ter_spaceHeat_temp[[2016,2017,2018]]
    df_ter_spaceHeat_temp = df_ter_spaceHeat_temp.sum(axis=0)
    
    df_ter_hotWater_temp = df_ter.iloc[24,:]
    df_ter_hotWater_temp = df_ter_hotWater_temp[[2016,2017,2018]]

    df_ter_spaceHeat = df_ter_spaceHeat.join(df_res_spaceHeat_temp.rename(reg),how='left')
    df_ter_hotWater = df_ter_hotWater.join(df_ter_hotWater_temp.rename(reg),how='left')
    

df_ter_spaceHeat = df_ter_spaceHeat.mean()    
df_ter_spaceHeat = df_ter_spaceHeat*ktoe_to_TWh_factor

df_ter_hotWater = df_ter_hotWater.mean()  
df_ter_hotWater = df_ter_hotWater*ktoe_to_TWh_factor
      
df_spaceHeat = df_res_spaceHeat + df_ter_spaceHeat
df_hotWater = df_res_hotWater + df_ter_hotWater

df_hotWater.index.name = 'country_code'
df_spaceHeat.index.name = 'country_code'

df_spaceHeat.rename('space_heat').to_csv('elec_to_spaceHeat_2016_17_18_mean_TWh_IDEES_2021.csv')
df_hotWater.rename('hot_water').to_csv('elec_to_hotWater_2016_17_18_mean_TWh_IDEES_2021.csv')




