from BLS_Data_Retrievel.retrieve_data import *

# Downloading data from BLS site regarding CPI
download_data_bls = Download_Data_BLS()

# Create Mapping Variables
download_data_bls.create_mapping()

# Pushing Data into BigQuery
download_data_bls.download_and_push_to_bq()