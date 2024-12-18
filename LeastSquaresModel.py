# Libraries
import numpy as np
import pandas as pd
from scipy.optimize import lsq_linear
import copy

# Custom libraries
import Setup as setup
import Statistics as stats

# Fit least squares model and return coefficients and residuals
def fit(df, col_names):

    A = np.array(df.drop('Demand', axis=1), dtype=np.float64) 
    b = np.array(df['Demand'], dtype=np.float64) 

    temp_idx = np.where(np.char.startswith(col_names, 'Temp') == True)[0]   
    daylength_idx = np.where(np.char.startswith(col_names, 'Daylength') == True)[0]

    # Constrain temperatures and daylength
    lb = np.full(len(col_names), -np.inf)
    ub = np.full(len(col_names), np.inf)
    # coefficients can not be positive
    ub[temp_idx] = 0 
    ub[daylength_idx] = 0

    # Fit model to data
    model_fit = lsq_linear(A, b, bounds=(lb, ub), max_iter=3000, tol=1e-10, lsq_solver='lsmr', lsmr_tol='auto', verbose=2)

    # Model results
    coef = model_fit.x 
    resid = A.dot(coef) - b
    
    return coef, resid



# Simulate electricity demand  
def simulate(m, df_features, coef, init_cond):
  # Simulate year
  scenario = np.zeros(len(df_features))
  exog_features = np.array(df_features)

  # if dynamic model : simulate autoregressive effects
  if m.type[0] == 'Dynamic':
    n_lags = m.type[1]
    # Initial time step
    for i in range(n_lags):
      exog_features[0,-(i+1)] = init_cond  
    scenario[0] = np.dot(coef, exog_features[0,:])

    # Intermediate time steps
    for t in range(1,len(scenario)-1):
      scenario[t] = np.dot(coef, exog_features[t,:])
      for i in range(1,n_lags+1):
        if i <= n_lags-t:
          exog_features[t+1,-i] = init_cond
        else:
          exog_features[t+1,-i] = scenario[t-(n_lags-i)]

    # Final time step
    scenario[-1] = np.dot(coef, exog_features[-1,:])

  # if static model : one shot
  else:
    scenario = exog_features.dot(coef)
    
  df = pd.DataFrame({'Demand': scenario}, index=df_features.index)
  # Cut away run-in period (due to setup.t_offset)
  df_slice = df.loc[m.t_start:m.t_end]

  return df_slice





# Simulate electricity demand where the actual measured demand in the previous time step is used for the autoregressive effects in the next time step
def simulate_1step(df_features, coef):
  exog_features = np.array(df_features.drop('Demand', axis=1))
  scenario = np.zeros(len(df_features))  
  scenario = exog_features.dot(coef)
  df = pd.DataFrame({'Demand': scenario}, index=df_features.index)

  return df




# Simulate the classic demand given a lower temperature threshold (setup.T_set)
def simulate_classic(m, df_features, feature_list, coef, init_cond):

  # Simulate with actual temperature profile
  df_sim = simulate(m, df_features, coef, init_cond)
  
  # Select all temperature clusters
  df_exog_features = copy.deepcopy(df_features)
  temp_idx = np.where(np.char.startswith(feature_list, 'Temp') == True)[0]
  if len(setup.T_breakpoint) == 2: # for cooling breakpoint
    temp_idx_pw1 = np.where(np.char.startswith(feature_list, 'PW_'+str(setup.T_breakpoint[0])+'_Temp') == True)[0]
    temp_idx_pw2 = np.where(np.char.startswith(feature_list, 'PW_'+str(setup.T_breakpoint[1])+'_Temp') == True)[0]

  # Adjust all temperature columns to never go below/above than the temperature threshold
  for i in temp_idx:
      df_exog_features.iloc[:,i] = np.maximum(setup.T_breakpoint[0], df_exog_features.iloc[:,i])
  if len(setup.T_breakpoint) == 2: # for cooling breakpoint   
    for i in temp_idx:
      df_exog_features.iloc[:,i] = np.minimum(setup.T_breakpoint[1], df_exog_features.iloc[:,i])
    for i in temp_idx_pw1:
      df_exog_features.iloc[:,i] = np.minimum(setup.T_breakpoint[1]-setup.T_breakpoint[0], df_exog_features.iloc[:,i])
    for i in temp_idx_pw2:
      df_exog_features.iloc[:,i] = np.minimum(setup.T_breakpoint[1]-setup.T_breakpoint[1], df_exog_features.iloc[:,i])  


  # Simulate with temperature constraints imposed
  df_sim_const_temp = simulate(m, df_exog_features, coef, init_cond)

  demand_tot = df_sim['Demand'].sum()
  demand_const_temp = df_sim_const_temp['Demand'].sum()
  print('\n Total demand : '+str(round(demand_tot))+'  Classic demand : '+str(round(demand_const_temp))+' estimated fraction : '+str(round(demand_const_temp/demand_tot,3))+' \n\n')
  return df_sim_const_temp



# Simulate electricity demand based on a Monte Carlo approach that accounts for the stochastic residual term
def simulate_monte_carlo(m, df_features, coef, init_cond, N, fit_resid):
  # Simulate year
  scenario = np.zeros((len(df_features), N))
  exog_features = np.array(df_features)

  # Fit t-distribution
  t_params = stats.fit_t(fit_resid)

  for n in range(0,N):
    
    # Sample error terms
    e = stats.sample_t(t_params, len(df_features))
    
    # if dynamic model : simulate autoregressive effects
    if m.type[0] == 'Dynamic':
      n_lags = m.type[1]
      # Initial time step
      for i in range(n_lags):
        exog_features[0,-(i+1)] = init_cond  
      scenario[0] = np.dot(coef, exog_features[0,:]) + e[0]

      # Intermediate time steps
      for t in range(1,len(scenario)-1):
        scenario[t] = np.dot(coef, exog_features[t,:]) + e[t]
        for i in range(1,n_lags+1):
          if i <= n_lags-t:
            exog_features[t+1,-i] = init_cond
          else:
            exog_features[t+1,-i] = scenario[t-(n_lags-i)]

      # Final time step
      scenario[-1] = np.dot(coef, exog_features[-1,:]) + e[-1]

    # if static model : one shot
    else:
      scenario = exog_features.dot(coef) + e

  scenario_mean = np.mean(scenario, axis=1)
  scenario_lb = np.percentile(scenario, 2.5, axis=1)
  scenario_ub = np.percentile(scenario, 97.5, axis=1)

  # Create dataframe
  df = pd.DataFrame({'Demand_mean': scenario_mean, 'Demand_lb': scenario_lb, 'Demand_ub': scenario_ub}, index=df_features.index)

  # Cut away run-in period
  df_slice = df.loc[m.t_start:m.t_end]

  return df_slice