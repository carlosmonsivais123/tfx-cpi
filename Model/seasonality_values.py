import pandas as pd
import numpy as np
from datetime import datetime

class Model_Inputs:
    def seasonality_matrix(self, train_data):
        train_data = train_data.astype({'series_id': str, 
                                        'value': float, 
                                        'date': 'datetime64[ns]',
                                        'item': str})
        train_data['month'] = train_data['date'].dt.month
        train_data['year'] = train_data['date'].dt.year
        train_data.sort_values(by = ['date'], inplace = True, ascending = True)

        train_data.set_index('date', inplace = True, drop = False)
        train_data.index = train_data.index.to_pydatetime()
        train_data.asfreq('MS')

        minimum_year = '01/01/{}'.format(train_data['year'].iloc[0])
        minimum_year_date = datetime.strptime(minimum_year, '''%m/%d/%Y''')

        maximum_year = '12/01/{}'.format(train_data['year'].iloc[-1])
        maximum_year_date = datetime.strptime(maximum_year, '''%m/%d/%Y''')

        idx = pd.period_range(minimum_year_date, maximum_year_date, freq = 'M')
        idx = idx.to_timestamp(freq = None)

        train_data.reset_index(inplace = True, drop = False)

        data = {'index': idx}
        all_timestamps_df = pd.DataFrame(data = data)

        final_train_data = train_data.merge(all_timestamps_df, on=['index'], how='outer')
        final_train_data.sort_values(by = 'index', inplace = True)
        final_train_data.reset_index(inplace = True, drop = True)

        shape_1 = int(len(final_train_data)/12)
        seasonal_array = final_train_data['month'].to_numpy().reshape((shape_1, -1))

        return seasonal_array