import pandas as pd
import numpy as np
import tensorflow_probability as tfp
from tensorflow_probability import sts

import functools
import absl
import os
from typing import List, Text

from tfx.components.trainer.fn_args_utils import FnArgs
from tfx.components.trainer.fn_args_utils import DataAccessor
from tfx_bsl.tfxio import dataset_options
from tensorflow_metadata.proto.v0 import schema_pb2
from tfx.utils import io_utils
import tensorflow as tf

# from Variables.create_vars import *
# from Model.seasonality_values import Model_Inputs
# from Plots.plotly_plots import Plot_Model_Ouptuts

def _input_fn(file_pattern, data_accessor: DataAccessor, schema, batch_size: int = 200) -> tf.data.Dataset:
    """Generates features and label for training.

    Args:
    file_pattern: List of paths or patterns of input tfrecord files.
    data_accessor: DataAccessor for converting input to RecordBatch.
    schema: schema of the input data.

    Returns:
    A dataset that contains (features, indices) tuple where features is a
        dictionary of Tensors, and indices is a single Tensor of label indices.
    """
    return data_accessor.tf_dataset_factory(file_pattern, dataset_options.TensorFlowDatasetOptions(batch_size = batch_size, shuffle = False, num_epochs = 1), schema).repeat(1)



@tf.function(experimental_compile = True)
def train(model, observed_time_series, variational_posteriors, num_variational_steps, optimizer = tf.optimizers.Adam(learning_rate = 0.1)):
    elbo_loss_curve_var = tfp.vi.fit_surrogate_posterior(target_log_prob_fn = model.joint_log_prob(observed_time_series = observed_time_series),
                                                            surrogate_posterior = variational_posteriors,
                                                            optimizer = optimizer,
                                                            num_steps = num_variational_steps,
                                                            jit_compile = True)
    return elbo_loss_curve_var


def _build_model(observed_time_series, seasonal_array) -> tf.keras.Model:
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


def run_fn(fn_args: FnArgs):
    environment = fn_args.custom_config['environment']
    fn_args.custom_config.pop('environment')

    trainer_settings = fn_args.custom_config['trainer_settings']
    fn_args.custom_config.pop('trainer_settings')


    schema = io_utils.parse_pbtxt_file(fn_args.schema_file, schema_pb2.Schema())
    train_dataset = _input_fn(fn_args.train_files,
                              fn_args.data_accessor,
                              schema,
                              batch_size = 1)    


    

    seasonal_array = model_inputs.seasonality_matrix(train_data = fn_args.train_files)

    model = _build_model(observed_time_series = fn_args.train_files, seasonal_array = seasonal_array)
# def variational_loss_function(self, model, observed_time_series):

    # ########## Variational Loss Function ##########
    # variational_posteriors = tfp.sts.build_factored_surrogate_posterior(model = model)
    # # Allow external control of optimization to reduce test runtimes.
    # num_variational_steps = 200 # @param { isTemplate: true}
    # num_variational_steps = int(num_variational_steps)
    # # Build and optimize the variational loss function.
    # elbo_loss_curve_var = tfp.vi.fit_surrogate_posterior(target_log_prob_fn = model.joint_log_prob(observed_time_series = observed_time_series),
    #                                                         surrogate_posterior = variational_posteriors,
    #                                                         optimizer = tf.optimizers.Adam(learning_rate = 0.1),
    #                                                         num_steps = num_variational_steps,
    #                                                         jit_compile = True)
    # Draw samples from the variational posterior.

    elbo_loss_curve = train(model, observed_time_series, variational_posteriors, num_variational_steps, optimizer = tf.optimizers.Adam(learning_rate = 0.1))

    q_samples_var = variational_posteriors.sample(50)
    




    ########## Component Breakdown ##########
    component_dists = sts.decompose_by_component(model = model, 
                                                observed_time_series = observed_time_series, 
                                                parameter_samples = q_samples_var)
    
    component_means_vals, component_stddevs_vals = ({k.name: c.mean() for k, c in component_dists.items()},
                                            {k.name: c.stddev() for k, c in component_dists.items()})



    ########## One Step Predictions ##########     
    one_step_dist = sts.one_step_predictive(model = model,
                                            observed_time_series = observed_time_series,
                                            parameter_samples = q_samples_var)
    
    one_step_mean_var, one_step_scale_var = (one_step_dist.mean().numpy(), one_step_dist.stddev().numpy())
