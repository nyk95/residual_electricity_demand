# Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import copy
import os

# Custom libraries
import Setup as setup
import ReadData as read
import DataProcessing as dp
import Statistics as stats
import GeoProcessing as gp
import LeastSquaresModel as LSQ
import Plotter as plotter
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

# Working directory
os.chdir(setup.datafolder)

def check_if_dir_exists(destination_path):
    Path(destination_path).mkdir(parents=True, exist_ok=True)
    

# Model object
class M:
    def __init__(self, type, t_start, t_end, region, country):
        self.type = type
        self.t_start = t_start
        self.t_end = t_end
        self.region = region
        self.country = country
        self.geo_points = gp.points_in_geo(region)
        self.holiday_list = dp.create_holiday_list(country, int(t_start))
        #self.n_clusters = 1
        self.n_clusters = gp.n_clusters(region)


# Main Script
region = 'DK'
country = 'DK'
## ----------------------------- Initialize model ----------------------------- ##
m_cluster = M(setup.model_type, setup.t_start_cluster, setup.t_end_cluster, region, country)

# Cluster data points and update geo_points in the model object
stats.cluster(m_cluster, read.temperature(m_cluster), 'Temp_cluster')

# Instantiate models
m_fit = copy.deepcopy(m_cluster)
m_fit.t_start, m_fit.t_end = setup.t_start_fit, setup.t_end_fit

# Create data for fiting the model
df_data_fit = dp.get_data_table(m_fit,include_meas=True)
data_fit_col_names = list(df_data_fit.drop('Demand', axis=1).columns)

# Create data for simulating the model
init_cond = float(df_data_fit['Demand'].mean())
m_sim = copy.deepcopy(m_cluster)
m_sim.t_start, m_sim.t_end = setup.t_start_sim, setup.t_end_sim
df_data_sim = dp.get_data_table(m_sim,True,data_fit_col_names)


## ----------------------------- Fit model and simulate fit/sim periods ----------------------------- ##
coef, resid_fit = LSQ.fit(df_data_fit, data_fit_col_names)


## ----------------------------- simulate model instances ----------------------------- ##
df_fit_1step = LSQ.simulate_1step(df_data_fit, coef)
df_sim = LSQ.simulate(m_sim, df_data_sim.drop('Demand',axis=1), coef, init_cond)

df_data_sim = dp.get_data_table(m_sim,True,data_fit_col_names)
df_sim_classic = LSQ.simulate_classic(m_sim, df_data_sim.drop('Demand',axis=1), data_fit_col_names, coef, init_cond)


## ----------------------------- Model diagnostics ----------------------------- ##
df_resid_fit = pd.DataFrame({'Resid': resid_fit}, index = df_fit_1step.index)
df_pop_weighted_temp = stats.pop_weighted_mean_total(m_fit, df_data_fit, 'Temp_cluster')
df_pop_weighted_temp = df_pop_weighted_temp[['Pop weight mean']]
plotter.plot_model_fit_diag(m_fit.region,df_data_fit['Demand'],df_fit_1step,df_resid_fit,df_pop_weighted_temp)


## ----------------------------- Model validation ----------------------------- ##
df_pop_weighted_temp = stats.pop_weighted_mean_total(m_sim, df_data_sim, 'Temp_cluster')
df_pop_weighted_temp = df_pop_weighted_temp[['Pop weight mean']]
plotter.plot_model_sim_val(m_sim.region,df_data_sim['Demand'],df_sim,df_pop_weighted_temp)


## ----------------------------- export simulation results ----------------------------- ##
check_if_dir_exists(setup.simulation_output_folder)
df_sim.to_csv(setup.simulation_output_folder + '/'  + m_sim.region + ' entire_electricity_demand.csv')
df_sim_classic.to_csv(setup.simulation_output_folder + '/' + m_sim.region + ' classic_electricity_demand_est.csv')


