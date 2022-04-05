from BLS_Data_Retrievel.retrieve_data import *
from BLS_Data_Retrievel.sending_items_bq_to_gcs_bucket import *

# Downloading data from BLS site regarding CPI
download_data_bls = Download_Data_BLS()
print('Finished Downloading Data')

# Create Mapping Variables
download_data_bls.create_mapping()
print('Finished Mapping Variables')

# Pushing Data into BigQuery
download_data_bls.download_and_push_to_bq()
print('Data Has Been Pushed to BigQuery')

# Pushing Data into GCP Buckets and Splitting it int Training and Testing
# 80% Training and 20% Testing
send_data_gcs_bucket = Send_Data_To_Bucket()
send_data_gcs_bucket.bg_to_bucket()
print('Data Has Been Split and Pushed to GCS Bucket')