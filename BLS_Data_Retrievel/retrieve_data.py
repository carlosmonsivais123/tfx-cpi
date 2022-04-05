import pandas as pd
from google.cloud import bigquery

from Variables.create_vars import *
from BigQuery_Classes.bq_client import BQ_Client_Connect

class Download_Data_BLS:
    def __init__(self):
        self.mapping_link = ['https://download.bls.gov/pub/time.series/cu/cu.area', 
                             'https://download.bls.gov/pub/time.series/cu/cu.base',
                             'https://download.bls.gov/pub/time.series/cu/cu.item',
                             'https://download.bls.gov/pub/time.series/cu/cu.period',
                             'https://download.bls.gov/pub/time.series/cu/cu.periodicity',
                             'https://download.bls.gov/pub/time.series/cu/cu.seasonal']

        self.mapping_link_names = ['area_url', 'base_url', 'cu_item', 'cu_period', 'cu_periodicity', 'cu_seasonal']

        self.bls_data_links = ['https://download.bls.gov/pub/time.series/cu/cu.data.0.Current',
                               'https://download.bls.gov/pub/time.series/cu/cu.data.1.AllItems',
                               'https://download.bls.gov/pub/time.series/cu/cu.data.11.USFoodBeverage',
                               'https://download.bls.gov/pub/time.series/cu/cu.data.12.USHousing',
                               'https://download.bls.gov/pub/time.series/cu/cu.data.13.USApparel',
                               'https://download.bls.gov/pub/time.series/cu/cu.data.14.USTransportation',
                               'https://download.bls.gov/pub/time.series/cu/cu.data.15.USMedical',
                               'https://download.bls.gov/pub/time.series/cu/cu.data.16.USRecreation',
                               'https://download.bls.gov/pub/time.series/cu/cu.data.17.USEducationAndCommunication',
                               'https://download.bls.gov/pub/time.series/cu/cu.data.18.USOtherGoodsAndServices',
                               'https://download.bls.gov/pub/time.series/cu/cu.data.2.Summaries',
                               'https://download.bls.gov/pub/time.series/cu/cu.data.20.USCommoditiesServicesSpecial']
        
        self.bls_data_names = ['current_0', 'all_items_1', 'food_beverage_11', 'housing_12', 'apparel_13', 
                               'transportation_14', 'medical_15', 'recreation_16', 'education_and_communication_17', 
                               'other_goods_and_services_18', 'summaries_2', 'commodities_services_special_20']


    def create_mapping(self):
        self.mapping_dict = {}
        mapping_name_counter = 0

        for mapping_dataset in self.mapping_link:

            map_dataframe = pd.read_csv(mapping_dataset, 
                                        delimiter = '\t',
                                        usecols=[0,1])

            map_dataframe.columns = map_dataframe.columns.str.replace(' ','')
            map_dataframe.drop_duplicates(inplace = True)

            mapping_dict_vals = dict(map_dataframe.values)

            self.mapping_dict['{}'.format(self.mapping_link_names[mapping_name_counter])] = mapping_dict_vals

            mapping_name_counter = mapping_name_counter + 1

        return self.mapping_dict 
                             

    def download_and_push_to_bq(self):
        connect_client = BQ_Client_Connect()
        client = connect_client.create_client()

        link_name_counter = 0

        for dataset in self.bls_data_links:
            print(dataset)
            bls_dataframe = pd.read_csv(dataset, 
                                        delimiter = '\t')

            bls_dataframe.columns = bls_dataframe.columns.str.replace(' ','')

            bls_dataframe = bls_dataframe[bls_dataframe['period'].str.contains('M13|S03|S02|S01')==False]
            bls_dataframe['period_abr'] = bls_dataframe['period'].replace(self.mapping_dict['cu_period'])
            bls_dataframe['date'] = pd.to_datetime(bls_dataframe['year'].astype(str) + bls_dataframe['period_abr'], format='%Y%b')

            bls_dataframe['item'] = bls_dataframe['series_id'].str[8:]
            bls_dataframe['item'] = bls_dataframe['item'].str.replace(' ', '')
            bls_dataframe['item'] = bls_dataframe['item'].replace(self.mapping_dict['cu_item'])

            bls_dataframe['area_code_desc'] = bls_dataframe['series_id'].str.slice(4, 8)
            bls_dataframe['area_code_desc'] = bls_dataframe['area_code_desc'].replace(self.mapping_dict['area_url'])

            bls_dataframe['periodicity'] = bls_dataframe['series_id'].str.slice(3, 4)
            bls_dataframe['periodicity'] = bls_dataframe['periodicity'].replace(self.mapping_dict['cu_periodicity'])

            bls_dataframe['seasonality'] = bls_dataframe['series_id'].str.slice(2, 3)
            bls_dataframe['seasonality'] = bls_dataframe['seasonality'].replace(self.mapping_dict['cu_seasonal'])

            bls_dataframe = bls_dataframe[(bls_dataframe['area_code_desc'] == 'U.S. city average')\
                                           & (bls_dataframe['seasonality'] == 'Not Seasonally Adjusted')]

            bls_dataframe.sort_values(by = ['date'], inplace = True)
            bls_dataframe.reset_index(drop = True, inplace = True)
            bls_dataframe = bls_dataframe[['series_id', 'value', 'date', 'item']]
           
            job_config = bigquery.LoadJobConfig(write_disposition = 'WRITE_TRUNCATE')


            table_id_name = '{}.{}'.format(bq_dataset_id, self.bls_data_names[link_name_counter])
            job = client.load_table_from_dataframe(bls_dataframe, 
                                                   destination = table_id_name,
                                                   job_config = job_config)
            job.result()

            link_name_counter = link_name_counter + 1