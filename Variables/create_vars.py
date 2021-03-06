from Variables.read_in_vars import Read_YAML

read_yaml_class = Read_YAML()
yaml_vars = read_yaml_class.read_vars()

# GCP Key
gcp_key = yaml_vars['GCP_Key']['Key_Location']

# BigQuery Variables
gcp_project_id = yaml_vars['BigQuery']['GCP_Project_ID']
bq_dataset_id = yaml_vars['BigQuery']['BigQuery_Dataset_ID']

# GCP Bucket
gcp_project_id_bucket = yaml_vars['GCP_Bucket']['GCP_Project_ID_Bucket']
gcp_bucket_name = yaml_vars['GCP_Bucket']['GCP_Bucket_Name']