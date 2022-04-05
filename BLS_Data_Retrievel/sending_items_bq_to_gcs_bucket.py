import pandas as pd
from google.cloud import bigquery
import pandas_gbq

from BigQuery_Classes.bq_client import BQ_Client_Connect
from Variables.create_vars import *
from GCS_Classes.gcs_client import GCS_Client_Connect
from BLS_Data_Retrievel.data_train_test_split import Data_Pre_Process

class Send_Data_To_Bucket:
    def bg_to_bucket(self):
        gcs_client_connection = GCS_Client_Connect()
        bucket = gcs_client_connection.create_client(project = gcp_project_id_bucket, bucket_name = gcp_bucket_name)


        connect_client = BQ_Client_Connect()
        client = connect_client.create_client()

        dataset_id = gcp_project_id + '.' + bq_dataset_id

        tables = client.list_tables(dataset_id)

        all_tables = []
        for table in tables:
            all_tables.append("{}.{}.{}".format(table.project, table.dataset_id, table.table_id))
        
        all_tables = [i for i in all_tables if not ('current_0' in i)]

        for table in all_tables:
            query = '''SELECT *
                       FROM `{}`'''.format(table)
            read_in_data_from_bq = pd.read_gbq(query = query, project_id = '{}'.format(gcp_project_id))
            
            unique_items = list(read_in_data_from_bq['item'].unique())

            for item in unique_items:
                item_df = read_in_data_from_bq[read_in_data_from_bq['item'] == '{}'.format(item)]
                item_df.reset_index(inplace = True, drop = True)

                split_data = Data_Pre_Process()
                data_splits = split_data.sub_split_train_test(item_df)

                training = data_splits[0]
                testing = data_splits[1]
                
                bucket.blob('{}/{}_training.csv'.format(item, item)).upload_from_string(training.to_csv(index = False), 'text/csv')
                bucket.blob('{}/{}_testing.csv'.format(item, item)).upload_from_string(testing.to_csv(index = False), 'text/csv')