# Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.stats import t
import datetime
from matplotlib import gridspec
import numpy.ma as ma
from pathlib import Path


import Setup as setup
def check_if_dir_exists(destination_path):
    Path(destination_path).mkdir(parents=True, exist_ok=True)
    




def plot_model_fit_diag(reg,df_meas,df_sim,df_resid,df_temp):
    
    
    df = pd.DataFrame(index=pd.date_range(start=df_meas.index[0],end=df_meas.index[-1],freq='h'))
    df_meas = df.join(df_meas,how='left')
    df_sim = df.join(df_sim,how='left')
    df_resid = df.join(df_resid,how='left')
    df_temp = df.join(df_temp,how='left')

    df_sim[pd.isna(df_meas).squeeze()] = np.nan
    df_resid[pd.isna(df_meas).squeeze()] = np.nan
    df_temp[pd.isna(df_meas).squeeze()] = np.nan
    
    
    #----------------------------------------------------------------------
    # create figure object
    fig = plt.figure(figsize=(18, 15))
    gs = gridspec.GridSpec(30,30)


    #----------------------------------------------------------------------
    # full time series plot
    ax0 = plt.subplot(gs[0:8,:])
    (df_meas.resample('d').mean()/1e3).plot(ax=ax0,color='black')
    (df_sim.resample('d').mean()/1e3).plot(ax=ax0,color='blue')
    (df_meas/1e3).plot(ax=ax0,color='black',alpha=0.2,linewidth=0.5)
    (df_sim/1e3).plot(ax=ax0,color='blue',alpha=0.2,linewidth=0.5)
    
    ax0.set_ylabel('GWh')
    ax0.legend(['Meas','Fit'])
    
    # PDF
    ax1 = plt.subplot(gs[10:18,:13])
    (df_resid.squeeze()/1e3).hist(ax=ax1,histtype='step',bins=50,color='blue',density=True)
    resid_std = -df_resid.std().item()/1e3
    ax1.set_xlabel('Resid (GWh)')
    ax1.set_ylabel('PDF')
    ax1.legend(['Resid std : '+str(round(resid_std,2))+' GWh'])
    
    
    
    # scatter w temp
    ax2 = plt.subplot(gs[10:18,17:])
    ax2.plot(df_temp.squeeze().dropna(), df_meas.squeeze().dropna()/1e3,linestyle='none',color='black',marker='o',mfc='none')
    ax2.plot(df_temp.squeeze().dropna(), df_sim.squeeze().dropna()/1e3,linestyle='none',color='blue',marker='x',mfc='none',alpha=0.3)
    ax2.legend(['Meas','Fit'])
    ax2.set_xlabel('Temperature degC')
    ax2.set_ylabel('GWh')
    
    # heteroscadaticity check
    ax3 = plt.subplot(gs[22:,:13])
    ax3.plot(df_sim/1e3,df_resid/1e3,mfc='none',linestyle='none',color='blue',marker='o',alpha=0.5)
    ax3.set_xlabel('Fit (GWh)')
    ax3.set_ylabel('Resid (GWh)')
    
    # resid ACF
    lags=24
    x = df_resid.copy()
    corr_pos_lag = [] 
    for i in range(lags+1):  
        x1 = x.shift(i)
        a=ma.masked_invalid(x1.astype(float))
        b=ma.masked_invalid(x.astype(float))
        msk = (~a.mask & ~b.mask)
        corr_pos_lag.append(ma.corrcoef(a[msk],b[msk])[0,1])
    acf = corr_pos_lag
    acf = np.array(acf)        
    lags = np.arange(lags+1)  
    
    ax4 = plt.subplot(gs[22:,17:])    
    ax4.stem(lags, acf, basefmt='-k',linefmt='-b')
    ax4.set_xlabel('hours')
    ax4.set_ylabel('ACF (Resid)')    
    
    plt.suptitle(reg)
    #plt.show()
    
    check_if_dir_exists(setup.model_fit_diagnostics_plot_folder)
    plt.savefig(setup.model_fit_diagnostics_plot_folder+'/'+reg+'_val_plot.png')
 
    
    return





def plot_model_sim_val(reg,df_meas,df_sim,df_temp):
    
    df = pd.DataFrame(index=pd.date_range(start=df_meas.index.values[0],end=df_meas.index.values[-1],freq='h'))
    df_meas = df.join(df_meas,how='left')
    df_sim = df.join(df_sim,how='left')
    df_temp = df.join(df_temp,how='left')

    df_sim[df_meas.isna().squeeze()] = np.nan
    df_resid = df_meas - df_sim
    df_temp[df_meas.isna().squeeze()] = np.nan

    
    #----------------------------------------------------------------------
    # create figure object
    fig = plt.figure(figsize=(18, 15))
    gs = gridspec.GridSpec(30,30)


    #----------------------------------------------------------------------
    # full time series plot
    ax0 = plt.subplot(gs[0:8,:])
    (df_meas.resample('d').mean()/1e3).plot(ax=ax0,color='black')
    (df_sim.resample('d').mean()/1e3).plot(ax=ax0,color='blue')
    (df_meas/1e3).plot(ax=ax0,color='black',alpha=0.2,linewidth=0.5)
    (df_sim/1e3).plot(ax=ax0,color='blue',alpha=0.2,linewidth=0.5)
    
    ax0.set_ylabel('GWh')
    ax0.legend(['Meas','Sim'])
    
    
    #----------------------------------------------------------------------
    # PDF of error : bias
    ax1 = plt.subplot(gs[10:18,:13])
    (df_resid.squeeze()/1e3).hist(ax=ax1,histtype='step',bins=50,color='blue',density=True)
    bias = -df_resid.mean().item()/1e3
    ax1.set_xlabel('Error (GWh)')
    ax1.set_ylabel('PDF')
    ax1.legend(['Resid bias : '+str(round(bias,2))+' GWh'])
    
    
    
    #----------------------------------------------------------------------
    # scatter : corr, mape
    ax2 = plt.subplot(gs[10:18,17:])
    ax2.plot(df_meas.squeeze().dropna()/1e3, df_sim.squeeze().dropna()/1e3,linestyle='none',color='blue',marker='o',mfc='none',alpha=0.5)
    mape = 100*(np.abs(df_meas-df_sim)/df_meas).mean().item()
    a=ma.masked_invalid(df_meas.astype(float))
    b=ma.masked_invalid(df_sim.astype(float))
    msk = (~a.mask & ~b.mask)
    corr = ma.corrcoef(a[msk],b[msk])[0,1] 
    ax2.legend(['corr : '+str(round(corr,2))+' mape : '+str(round(mape,2))])
    ax2.set_ylabel('Sim (GWh)')
    ax2.set_xlabel('Meas (GWh)')
    
    
    #----------------------------------------------------------------------
    # scatter w temp
    ax3 = plt.subplot(gs[22:,:13])
    ax3.plot(df_temp.squeeze().dropna(), df_meas.squeeze().dropna()/1e3,linestyle='none',color='black',marker='o',mfc='none',alpha=0.5)
    ax3.plot(df_temp.squeeze().dropna(), df_sim.squeeze().dropna()/1e3,linestyle='none',color='blue',marker='o',mfc='none',alpha=0.5)
    ax3.legend(['Meas','Sim'])
    ax3.set_xlabel('Temperature degC')
    ax3.set_ylabel('GWh')   

    
    #----------------------------------------------------------------------
    # ACF comparison
    lags=45
    x = df_meas.copy()
    corr_pos_lag = [] 
    for i in range(lags+1):  
        x1 = x.shift(i)
        a=ma.masked_invalid(x1.astype(float))
        b=ma.masked_invalid(x.astype(float))
        msk = (~a.mask & ~b.mask)
        corr_pos_lag.append(ma.corrcoef(a[msk],b[msk])[0,1])
    acf = corr_pos_lag
    acf_meas = np.array(acf)        
    lags_meas = np.arange(lags+1)  
    
    x = df_sim.copy()
    corr_pos_lag = [] 
    for i in range(lags+1):  
        x1 = x.shift(i)
        a=ma.masked_invalid(x1.astype(float))
        b=ma.masked_invalid(x.astype(float))
        msk = (~a.mask & ~b.mask)
        corr_pos_lag.append(ma.corrcoef(a[msk],b[msk])[0,1])
    acf = corr_pos_lag
    acf_sim = np.array(acf)        
    lags_sim = np.arange(lags+1)  
    
    ax4 = plt.subplot(gs[22:,17:])    
    ax4.plot(lags_meas,acf_meas, color='black')
    ax4.plot(lags_sim,acf_sim, color='blue')
    
    #plot_pacf(df_resid.copy(),lags=72,ax=ax4)
    ax4.set_xlabel('hours')
    ax4.set_ylabel('ACF')     
    plt.suptitle(reg)

    
    check_if_dir_exists(setup.model_validation_plots_folder)
    plt.savefig(setup.model_validation_plots_folder+'/'+reg+'_val_plot.png')

    return









































def MC_time_series(m, year, df_meas, df_sim):
    months = np.arange(1,13)
    month_name = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']

    for t in range(0, len(months)):
        df_sim_slice = df_sim.loc[(df_sim.index.month == months[t]) & (df_sim.index.year == year)]
        df_meas_slice = df_meas.loc[(df_meas.index.month == months[t]) & (df_meas.index.year == year)]
        plt.figure(figsize=(12,8))
        plt.plot(df_sim_slice['Demand_mean'])
        plt.plot(df_meas_slice['Demand'])
        plt.fill_between(df_sim_slice.index, df_sim_slice['Demand_lb'], df_sim_slice['Demand_ub'], alpha=0.2)
        plt.xlim([df_meas_slice.index[0], df_meas_slice.index[-1]])
        plt.ylim([min(df_sim['Demand_lb']), max(df_sim['Demand_ub'])])
        plt.legend(['Simulated', 'Measured', '95% PI'], loc='upper right')
        plt.grid(alpha=0.7)
        plt.ylabel('Power [MW]')
        plt.title(str(month_name[t]) + ' ' + m.region)
        filename = setup.model_validation_folder + '/' + m.region + '/' + m.type[0] + ' ' + m.region + ' ' + str(month_name[t]) + '.png'
        plt.savefig(filename, dpi=300, transparent=False)
        #plt.show()
        plt.close()
