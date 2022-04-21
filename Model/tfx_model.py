import pandas as pd
import numpy as np
from datetime import datetime
import tensorflow_probability as tfp
from tensorflow_probability import sts

import functools
import absl
import os
from typing import List, Text

from tfx.components.trainer.executor import TrainerFnArgs
from tfx.components.trainer.fn_args_utils import FnArgs
from tfx.components.trainer.fn_args_utils import DataAccessor
from tfx_bsl.tfxio import dataset_options
from tensorflow_metadata.proto.v0 import schema_pb2
from tfx.utils import io_utils
import tensorflow as tf

import dill as pickle

# from Variables.create_vars import *
# from Model.seasonality_values import Model_Inputs
# from Plots.plotly_plots import Plot_Model_Ouptuts

def seasonality_matrix(train_data):
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


def _input_fn(file_pattern, 
              data_accessor: DataAccessor, 
              schema, 
              batch_size: int = 200) -> tf.data.Dataset:
    """Generates features and label for training.

    Args:
    file_pattern: List of paths or patterns of input tfrecord files.
    data_accessor: DataAccessor for converting input to RecordBatch.
    schema: schema of the input data.

    Returns:
    A dataset that contains (features, indices) tuple where features is a
        dictionary of Tensors, and indices is a single Tensor of label indices.
    """

    dataset = data_accessor.tf_dataset_factory(file_pattern, 
                                               dataset_options.TensorFlowDatasetOptions(batch_size = batch_size, 
                                                                                        shuffle = False, 
                                                                                        num_epochs = 1), schema).repeat(1)

    data_dict = {'date': [], 'value': [], 'item': [], 'series_id': []}

    for element in dataset:
        data_dict['date'].append(tf.sparse.to_dense(element['date']).numpy().flatten()[0].decode('utf-8'))
        data_dict['value'].append(tf.sparse.to_dense(element['value']).numpy().flatten()[0])
        data_dict['item'].append(tf.sparse.to_dense(element['item']).numpy().flatten()[0].decode('utf-8'))
        data_dict['series_id'].append(tf.sparse.to_dense(element['series_id']).numpy().flatten()[0].decode('utf-8'))
        
    df_test = pd.DataFrame.from_dict(data_dict)

    return df_test


def _build_model(observed_time_series, seasonal_array):
    trend = sts.LocalLinearTrend(observed_time_series = observed_time_series,
                                name = 'trend')
    seasonal = sts.Seasonal(num_seasons = 12,
                            num_steps_per_season = seasonal_array,
                            observed_time_series = observed_time_series,
                            name = 'yearly')  
    autoregressive = sts.Autoregressive(order = 1, 
                                        observed_time_series=observed_time_series,
                                        name='autoregressive') 
    model = sts.Sum([trend,
                    seasonal,
                    autoregressive], 
                    observed_time_series = observed_time_series)       

    return model

def save_model(object_to_save, send_model_to, model_name):
    temp_path = os.path.join(send_model_to, model_name)
    os.makedirs(send_model_to)
    with open('{}'.format(temp_path), 'wb') as f:
        pickle.dump(object_to_save, f)



def run_fn(fn_args: TrainerFnArgs):
    schema = io_utils.parse_pbtxt_file(fn_args.schema_file, schema_pb2.Schema())
    train_dataset = _input_fn(file_pattern = fn_args.train_files,
                              data_accessor = fn_args.data_accessor,
                              schema = schema,
                              batch_size=1)  

    

    seasonal_array = seasonality_matrix(train_data = train_dataset)

    model = _build_model(observed_time_series = train_dataset['value'], seasonal_array = seasonal_array)

    # # ########## Variational Loss Function ##########
    # variational_posteriors = tfp.sts.build_factored_surrogate_posterior(model = model)
    
    # # Allow external control of optimization to reduce test runtimes.
    # num_variational_steps = 100 # @param { isTemplate: true}
    # num_variational_steps = int(num_variational_steps)

    # # Build and optimize the variational loss function.
    # elbo_loss_curve_var = tfp.vi.fit_surrogate_posterior(target_log_prob_fn = model.joint_log_prob(observed_time_series = train_dataset['value']),
    #                                                         surrogate_posterior = variational_posteriors,
    #                                                         optimizer = tf.optimizers.Adam(learning_rate = 0.1),
    #                                                         num_steps = num_variational_steps)

    # # Draw samples from the variational posterior.
    # q_samples_var = variational_posteriors.sample(50)


    # ########## Component Breakdown ##########
    # component_dists = sts.decompose_by_component(model = model, 
    #                                              observed_time_series = train_dataset['value'], 
    #                                              parameter_samples = q_samples_var)
    
    # component_means_vals, component_stddevs_vals = ({k.name: c.mean() for k, c in component_dists.items()},
    #                                                 {k.name: c.stddev() for k, c in component_dists.items()})



    # ########## One Step Predictions ##########     
    # one_step_dist = sts.one_step_predictive(model = model,
    #                                         observed_time_series = train_dataset['value'],
    #                                         parameter_samples = q_samples_var)
    
    # one_step_mean_var, one_step_scale_var = (one_step_dist.mean().numpy(), one_step_dist.stddev().numpy())


    serving_model_directory = fn_args.serving_model_dir
    model_filename = 'test_model.pkl'
    save_model(object_to_save = model, send_model_to = serving_model_directory, model_name = model_filename)