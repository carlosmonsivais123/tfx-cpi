import pandas as pd

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
                               'https://download.bls.gov/pub/time.series/cu/cu.data.10.OtherWest',
                               'https://download.bls.gov/pub/time.series/cu/cu.data.11.USFoodBeverage',
                               'https://download.bls.gov/pub/time.series/cu/cu.data.12.USHousing',
                               'https://download.bls.gov/pub/time.series/cu/cu.data.13.USApparel',
                               'https://download.bls.gov/pub/time.series/cu/cu.data.14.USTransportation',
                               'https://download.bls.gov/pub/time.series/cu/cu.data.15.USMedical',
                               'https://download.bls.gov/pub/time.series/cu/cu.data.16.USRecreation',
                               'https://download.bls.gov/pub/time.series/cu/cu.data.17.USEducationAndCommunication',
                               'https://download.bls.gov/pub/time.series/cu/cu.data.18.USOtherGoodsAndServices',
                               'https://download.bls.gov/pub/time.series/cu/cu.data.19.PopulationSize',
                               'https://download.bls.gov/pub/time.series/cu/cu.data.2.Summaries',
                               'https://download.bls.gov/pub/time.series/cu/cu.data.20.USCommoditiesServicesSpecial',
                               'https://download.bls.gov/pub/time.series/cu/cu.data.3.AsizeNorthEast',
                               'https://download.bls.gov/pub/time.series/cu/cu.data.4.AsizeNorthCentral',
                               'https://download.bls.gov/pub/time.series/cu/cu.data.5.AsizeSouth',
                               'https://download.bls.gov/pub/time.series/cu/cu.data.6.AsizeWest',
                               'https://download.bls.gov/pub/time.series/cu/cu.data.7.OtherNorthEast',
                               'https://download.bls.gov/pub/time.series/cu/cu.data.8.OtherNorthCentral',
                               'https://download.bls.gov/pub/time.series/cu/cu.data.9.OtherSouth',
                               'https://download.bls.gov/pub/time.series/cu/cu.series']
        
        self.bls_data_names = ['current_0', 'all_items_1', 'other_west_10', 'food_beverage_11',
                               'housing_12', 'apparel_13', 'transportation_14', 'medical_15', 'recreation_16', 
                               'education_and_communication_17', 'other_goods_and_services_18', 'population_size_19', 'summaries_2',
                               'commodities_services_special_20', 'size_north_east_3', 'size_north_central_4', 'size_south_5',
                               'size_west_6', 'other_northeast_7', 'other_north_central_8', 'other_south_9', 'series']


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

            try:
                bls_dataframe['period_abr'] = bls_dataframe['period'].replace(self.mapping_dict['cu_period'])
            except:
                bls_dataframe['begin_period_abr'] = bls_dataframe['begin_period'].replace(self.mapping_dict['cu_period'])
                bls_dataframe['end_period_abr'] = bls_dataframe['end_period'].replace(self.mapping_dict['cu_period'])

            bls_dataframe['item'] = bls_dataframe['series_id'].str[8:]
            bls_dataframe['item'] = bls_dataframe['item'].str.replace(' ', '')
            bls_dataframe['item'] = bls_dataframe['item'].replace(self.mapping_dict['cu_item'])

            bls_dataframe['area_code_desc'] = bls_dataframe['series_id'].str.slice(4, 8)
            bls_dataframe['area_code_desc'] = bls_dataframe['area_code_desc'].replace(self.mapping_dict['area_url'])

            bls_dataframe['periodicity'] = bls_dataframe['series_id'].str.slice(3, 4)
            bls_dataframe['periodicity'] = bls_dataframe['periodicity'].replace(self.mapping_dict['cu_periodicity'])

            bls_dataframe['seasonality'] = bls_dataframe['series_id'].str.slice(2, 3)
            bls_dataframe['seasonality'] = bls_dataframe['seasonality'].replace(self.mapping_dict['cu_seasonal'])


            table_id_name = '{}.{}'.format(bq_dataset_id, self.bls_data_names[link_name_counter])
            job = client.load_table_from_dataframe(bls_dataframe, 
                                                   destination = table_id_name)
            job.result()

            link_name_counter = link_name_counter + 1