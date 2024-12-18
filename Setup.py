# Model settings
model_type = ['Dynamic',2] # Set model type : [Dynamic/Static, n_lags]

## Variables
T_breakpoint = [15] # Breakpoint for piece wise temperature function
t_offset = 24 # Number of extra hours to include before simulation period to stabilize autoregressive effects
clusters_per_degree_lat = 0.4 # Number of climate clusters per degree latitude

## -------------------------------- Demand model time period definition -------------------------------- ##
# Year selection for model clustering
t_start_cluster = '2016'
t_end_cluster = '2019'

# Year selection for model fitting
t_start_fit = '2016'
t_end_fit = '2018'

# Year selection for model simulation
t_start_sim = '2019'
t_end_sim = '2019'

## -------------------------------- Filepaths -------------------------------- ##

# Main data folder
datafolder = 'Data'

# Paths relative to main data folder
simulation_output_folder = 'Simulation output' # Folder for storing demand model output
measured_demand_folder = 'Input data/ENTSOE/' # Historical electricity demand data as measured by ENTSOE
model_validation_plots_folder = 'Model validation'
model_fit_diagnostics_plot_folder = 'Model fit diagnostics'

model_geography_file = 'Model setup data/Model geography.xlsx' # Speadsheet used for setting up the model geography (e.g. defining regions, countries, etc.)
custom_holidays = 'Model setup data/custom_holidays.xlsx' # List of custom holidays

temperature_file = 'Input data/ERA5/temperature_100km_2016_2019_dk.csv' # Ambient temperature time series at each weather sample point
geo_polygon_file = 'Input data/mopo_country_shapefile/mopo_country_shapefile.shp' # Polygons representing model regions
grid_file = 'Input data/Grid/100km_grid_dk.xlsx' # Locations of weather sample points
population_file = 'Input data/Eurostat/JRC_GRID_2018/JRC_POPULATION_2018_25km.shp' # Population density of geography