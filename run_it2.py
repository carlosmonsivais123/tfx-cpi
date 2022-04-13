from Model.time_series_model import Run_Model
from Variables.create_vars import *

import pandas as pd


train_data = pd.read_csv('gs://cpi_bucket/Airline fares/Airline fares_training.csv', 
                   storage_options={"token": "{}".format(gcp_key)})

run_model = Run_Model()
run_model.run_time_series_model(train_data = train_data)