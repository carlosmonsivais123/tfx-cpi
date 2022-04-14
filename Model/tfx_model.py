import pandas as pd
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow_probability import sts

from tfx_bsl.tfxio import dataset_options
from tfx.components.trainer.fn_args_utils import DataAccessor


from Variables.create_vars import *
from Model.seasonality_values import Model_Inputs
from Plots.plotly_plots import Plot_Model_Ouptuts


def _input_fn(file_pattern: List[Text],
              data_accessor: DataAccessor,
              schema,
              batch_size: int = 200) -> tf.data.Dataset:
              
              dataset = data_accessor.tf_dataset_factory(file_pattern,
                                             dataset_options.TensorFlowDatasetOptions(batch_size=batch_size, shuffle = False, num_epochs = 1),
                                             schema)
    
              return dataset

def _build_model(self, observed_time_series, seasonal_array):   
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

def variational_loss_function(self, model, observed_time_series):

    ########## Variational Loss Function ##########
    variational_posteriors = tfp.sts.build_factored_surrogate_posterior(model = model)
    # Allow external control of optimization to reduce test runtimes.
    num_variational_steps = 200 # @param { isTemplate: true}
    num_variational_steps = int(num_variational_steps)
    # Build and optimize the variational loss function.
    elbo_loss_curve_var = tfp.vi.fit_surrogate_posterior(target_log_prob_fn = model.joint_log_prob(observed_time_series = observed_time_series),
                                                            surrogate_posterior = variational_posteriors,
                                                            optimizer = tf.optimizers.Adam(learning_rate = 0.1),
                                                            num_steps = num_variational_steps,
                                                            jit_compile = True)
    # Draw samples from the variational posterior.
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
