import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

def check_if_dir_exists(destination_path):
    Path(destination_path).mkdir(parents=True, exist_ok=True)
    


file_overview = 'calibration_processing/calibration_data.xlsx'
df_overview = pd.read_excel(file_overview,index_col=0)
reg_list = df_overview.index.values
df_final = pd.DataFrame(index=pd.date_range('1982-01-01 00:00:00','2021-12-31 23:00:00',freq='h'))


for reg in reg_list:
    print(reg)    
    
    if  pd.isna(df_overview.loc[reg,'elec_to_spaceHeat_est_TWh']):
        continue

    file_classic_est = 'Data/Simulation output/'+reg+' classic_electricity_demand_est.csv'
    file_entire = 'Data/Simulation output/'+reg+' entire_electricity_demand.csv'
    if reg == 'MT':
        file_classic_est = 'Data/Simulation output/IT classic_electricity_demand_est.csv'
        file_entire = 'Data/Simulation output/IT entire_electricity_demand.csv'
            
    
    df_entire = pd.read_csv(file_entire,index_col=0,parse_dates=True)
    df_classic_est = pd.read_csv(file_classic_est,index_col=0,parse_dates=True)
    #elec_spaceHeat_source_TWh = df_overview.loc[reg,'elec_to_spaceHeat_source_TWh']
    elec_hotWater_per_hour = df_overview.loc[reg,'elec_to_hotWater_source_TWh']*(1e6/8760)
    elec_industry_per_hour = df_overview.loc[reg,'elec_to_industry_source_TWh']*(1e6/8760)
    
    if reg == 'MT':
        elec_tot = df_overview.loc[reg,'entire_elec_TWh']
        df_classic_est = df_classic_est*(elec_tot/(df_entire['2016':'2018'].sum()/(3*1e6)))
        df_entire = df_entire*(elec_tot/(df_entire['2016':'2018'].sum()/(3*1e6)))

    #adj_factor = (elec_tot-elec_spaceHeat_source_TWh)df_classic_est['2016':'2018'].mean()
    #df_classic_adj = df_classic_est*adj_factor
    df_classic_adj_rem = df_classic_est - elec_hotWater_per_hour - elec_industry_per_hour

    df = df_entire.join(df_classic_est,how='left',rsuffix='_classic_est')
    df = df.join(df_classic_adj_rem,how='left',rsuffix='_classic_adj_rem')

    df_orig = df.copy()
    df = df['2019':'2019']
    fig, ax = plt.subplots(2,1,figsize=(10,7))
    
    (df['Demand'].resample('d').mean()/1e3).plot(ax=ax[0],color='black',alpha=0.7)
    (df['Demand_classic_est'].resample('d').mean()/1e3).plot(ax=ax[0],color='blue',alpha=0.7)
    #(df['Demand_classic_adj'].resample('d').mean()/1e3).plot(ax=ax,color='red',alpha=0.7)
    (df['Demand_classic_adj_rem'].resample('d').mean()/1e3).plot(ax=ax[0],color='red',alpha=0.7)
    
    (df['Demand']/1e3).plot(ax=ax[0],color='black',alpha=0.2,linewidth=0.5)
    (df['Demand_classic_est']/1e3).plot(ax=ax[0],color='blue',alpha=0.2,linewidth=0.5)
    #(df['Demand_classic_adj']/1e3).plot(ax=ax,color='red',alpha=0.1,linewidth=0.5)
    (df['Demand_classic_adj_rem']/1e3).plot(ax=ax[0],color='red',alpha=0.2,linewidth=0.5)
    
    leg_entire = df['Demand'].sum()/(df.index.year.unique().values.size*1e6)
    leg_classic = df['Demand_classic_est'].sum()/(df.index.year.unique().values.size*1e6)
    #leg_classic_adj = df['Demand_classic_adj'].sum()/(df.index.year.unique().values.size*1e6)
    leg_classic_rem = df['Demand_classic_adj_rem'].sum()/(df.index.year.unique().values.size*1e6)
    ax[0].set_ylabel('Demand (GWh)')
    #ax.legend(['Entire '+str(round(leg_entire,2))+' TWh','Classic '+str(round(leg_classic,2))+' TWh','Classic_adj '+str(round(leg_classic_adj,2))+' TWh','Classic_adj_removed '+str(round(leg_classic_rem,2))+' TWh'])
    ax[0].legend(['Elec Demand : '+str(round(leg_entire,2))+' TWh','Elec Demand w/o spaceHeat : '+str(round(leg_classic,2))+' TWh ('+str(round(100*leg_classic/leg_entire,2))+'%)','Other Elec Demand : '+str(round(leg_classic_rem,2))+' TWh ('+str(round(100*leg_classic_rem/leg_entire,2))+'%)'])

    df = df_orig.copy()
    (df['Demand'].resample('m').mean()/1e3).plot(ax=ax[1],color='black')
    (df['Demand_classic_est'].resample('m').mean()/1e3).plot(ax=ax[1],color='blue')
    (df['Demand_classic_adj_rem'].resample('m').mean()/1e3).plot(ax=ax[1],color='red')
    ax[1].set_ylabel('Monthly avg demand (GWh)')
    ax[1].legend(['Elec Demand ','Elec Demand w/o spaceHeat ','Other Elec Demand '])
    ax[1].set_ylim([0,1.2*(np.max(df['Demand'].resample('m').mean()/1e3))])
    plt.suptitle(reg)
    
    check_if_dir_exists('Final output/overview_plots')
    plt.savefig('Final output/overview_plots/'+reg+'.png')
    
    df_final = df_final.join(df['Demand_classic_adj_rem'].rename(reg),how='left')
    

df_final.to_csv('Final outputn/residual_electricity_demand.csv')

    