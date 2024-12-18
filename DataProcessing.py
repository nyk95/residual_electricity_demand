# Libraries
import numpy as np
import pandas as pd
import holidays
from suntimes import SunTimes

# Custom libraries
import Setup as setup
import ReadData as read
import Statistics as stats

# Create a list of national holidays
def create_holiday_list(country, year):
    # Instantiate holiday object for country
    holidays_obj = holidays.country_holidays(country, years=year)
    holidays_obj.observed = False
    holiday_list = []

    for date, holiday_name in sorted(holidays_obj.items()): 
        holiday_list.append(holiday_name)
    
    # Exclude from holiday list
    df_custom_holidays = pd.read_excel(setup.custom_holidays, sheet_name='Exclude')
    excluded_holidays = df_custom_holidays.loc[(df_custom_holidays['Country'] == 'All') | (df_custom_holidays['Country'] == country), 'Holiday'].to_list()
    holiday_list = list(filter(lambda a: a not in excluded_holidays, holiday_list))

    return holiday_list

# Create a time series of national holidays
def holiday(m, time_range):
    # Instantiate holiday object for country
    if m.t_start != m.t_end:
        year_range = range(int(m.t_start), int(m.t_end))
    else:
        year_range = int(m.t_start)

    holidays_obj = holidays.country_holidays(m.country, years=year_range)
    holidays_obj.observed = False

    # Create dataframe to store holidays
    df_holidays = pd.DataFrame(index=time_range)
    df_holidays['Holiday'] = np.full(len(time_range), np.nan)

    # Get date of holiday based on names in list and insert into holidays dataframe
    for i in m.holiday_list:
        dates = holidays_obj.get_named(i)
        for j in dates:
            # Old version
            # df_holidays.loc[(df_holidays.index.year == j.year) & (df_holidays.index.month == j.month) & (df_holidays.index.day == j.day)] = i
            # New version 
            df_holidays.loc[(df_holidays.index.year == j.year) & (df_holidays.index.month == j.month) & (df_holidays.index.day == j.day)] = str(i)
    # Select universal and region specific custom holidays
    df_custom_holidays = pd.read_excel(setup.custom_holidays, sheet_name='Include')
    df_custom_holidays_region = df_custom_holidays.loc[(df_custom_holidays['Country'] == 'All') | (df_custom_holidays['Country'] == m.country)]

    # Put custom holidays into holidays dataframe
    for i in range(0, len(df_custom_holidays_region)):
        df_holidays.loc[(df_holidays.index.month == df_custom_holidays_region['Month'].iloc[i]) & (df_holidays.index.day == df_custom_holidays_region['Day'].iloc[i]), 'Holiday'] = df_custom_holidays_region['Holiday'].iloc[i]

    return df_holidays

# Create a time series of daylengths
def daylength(m, time_range):
    daylength = np.zeros(len(time_range))

    longitude, latitude = m.geo_points['Longitude'].mean(), m.geo_points['Latitude'].mean()
    sun_obj = SunTimes(longitude, latitude, altitude=0)

    for t in range(0,len(time_range)):
        try:
            delta_sec = sun_obj.durationdelta(time_range[t])
            delta_hours = round(delta_sec.total_seconds()/3600, 2)
            daylength[t] = delta_hours
        except Exception: # Special case if daylength cannot be calculated due to polar position
            if delta_sec == 'Not calculable : PD':
                delta_hours = 24
                daylength[t] = delta_hours
            elif delta_sec == 'Not calculable : PN':
                delta_hours = 0
                daylength[t] = delta_hours
            pass
    
    df_daylength = pd.DataFrame({'Daylength': daylength}, index = time_range)


    return df_daylength

# Collect all exogenous features for demand model and output as a dataframe
def request_exog_inputs(m):
    # Create dataframe to store exogenous variables
    df = pd.DataFrame()
    df['Time'] = pd.Series(pd.date_range(start=pd.Timestamp(f'{m.t_start}-01-01 00:00') - pd.DateOffset(hours=setup.t_offset), end=f'{m.t_end}-12-31 23:00', freq='h'))
    df = df.set_index('Time')

    # Constant term in regression model
    df['Const'] = 1
    
    # Create time variables
    df['Day'] = df.index.weekday
    df.loc[df['Day'] <= 4, 'Day'] = 1
    df.loc[df['Day'] == 5, 'Day'] = 2
    df.loc[df['Day'] == 6, 'Day'] = 3
    df['Hour'] = df.index.hour
    
    # Holidays
    df_holidays = holiday(m, df.index)
    df = df.join(df_holidays) 
    # Unassign the weekday of all holidays
    df.loc[df['Holiday'] == df['Holiday'].astype(str), 'Day'] = 0

    # Daylength
    df_daylength = daylength(m, df.index)
    df = df.join(df_daylength)

    # Temperature
    df_temp = read.temperature(m)
    df_temp_cluster = stats.pop_weighted_mean(m, df_temp, 'Temp_cluster')
    df = df.join(df_temp_cluster)
    # Backfill missing temperatures in hindcast if requested period exceeds dataset (due to setup.t_offset)
    idx = df.columns[df.columns.str.startswith('Temp_cluster')]
    df[idx] = df[idx].fillna(method='bfill')
    
    # Non-linearities for least squares models

    # Piece wise function
    df_piecewise_temp1 = piecewise(df, 'Temp', setup.T_breakpoint)
    df = df.join(df_piecewise_temp1)

    # One-hot encoding of categorical variables
    df = pd.get_dummies(df, columns=['Day', 'Hour', 'Holiday'], prefix=['Day', 'Hour', 'Holiday'])
    
    # Interaction effects between regressors
    dummy_days = ['Day_', 'Daylength', 'Holiday']
    for i in range(0, len(dummy_days)):
        df = df.join(dummy_interaction(df, dummy_days[i], 'Hour_'))

    # Allocate space for autoregressive terms in dynamic models
    if m.type[0] == 'Dynamic':
        for n in range(m.type[1]):
            df['Demand_Lag'+str(n+1)] = np.zeros(len(df.index))

    return df



def get_data_table(m,include_meas=True,data_fit_col_names=None):
    
    
    df_data = request_exog_inputs(m)
    # if measured data is used in model
    if include_meas == True:
        df_meas = read.demand_measured(m)
        df_data = df_data.join(df_meas,how='left')
        # Add Autoregressive terms if dynamic models
        if m.type[0] == 'Dynamic':
            for i in range(m.type[1]):
                df_data['Demand_Lag'+str(i+1)] = df_data['Demand'].shift(i+1)
                
    df_data = df_data.dropna()
    
    # to make sure columns in data matrix while fitting the model match 
    # columns in simulation data matrix. .
    # if simumation data has more regressors than fit, get rid of those regressors
    # if simulation has less regressors than fit
    # make uncommon regressors from fit side to zero to have no effect 
    if data_fit_col_names!=None:
        if include_meas==True:
            df_data_new = pd.DataFrame(index=df_data.index,columns=data_fit_col_names+['Demand'])
            df_data_new = df_data_new.fillna(0)
            common_columns = df_data.columns.intersection(data_fit_col_names+['Demand'])
        else:
            df_data_new = pd.DataFrame(index=df_data.index,columns=data_fit_col_names)
            df_data_new = df_data_new.fillna(0)
            common_columns = df_data.columns.intersection(data_fit_col_names)
        df_data_new[common_columns] = df_data[common_columns]
        df_data = df_data_new

    return df_data



# gets Dataframe with interactions between dummy variable a and dummy variable b
def dummy_interaction(df, dummy_a, dummy_b):
    a = df.columns[df.columns.str.startswith(dummy_a)]
    b = df.columns[df.columns.str.startswith(dummy_b)]

    names = [('Interact_' + col1 + '_' + col2) for col1 in a for col2 in b]
    interactions = [(df[col1].mul(df[col2]).values) for col1 in a for col2 in b]

    df = pd.DataFrame(np.transpose(interactions), columns=names, index=df.index)

    return df

# Make piecewise function from dataframe based on variable name and threshold value
def piecewise_old(df, var, thr):
    var_col = df.columns[df.columns.str.startswith(var)]
    names = [('PW_' + str(thr) + '_' + i) for i in var_col]
    pw = [np.multiply((df[i] - thr).values, ((df[i] - thr) > 0).astype('int').values) for i in var_col]
    df = pd.DataFrame(np.transpose(pw), columns=names, index=df.index)
    
    return df


# Make piecewise function from dataframe based on variable name and threshold value
def piecewise(df, var, thr):
    var_col = df.columns[df.columns.str.startswith(var)]
    df_final = pd.DataFrame(index=df.index)
    for j in range(len(thr)):
        names = [('PW_' + str(thr[j]) + '_' + i) for i in var_col]
        pw = [np.multiply((df[i] - thr[j]).values, ((df[i] - thr[j]) > 0).astype('int').values) for i in var_col]
        df_temp = pd.DataFrame(np.transpose(pw), columns=names, index=df.index)
        df_final = df_final.join(df_temp)
        
    return df_final


