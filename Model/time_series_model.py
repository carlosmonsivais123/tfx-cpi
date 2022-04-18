import pandas as pd
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow_probability import sts

from Variables.create_vars import *
from Model.seasonality_values import Model_Inputs
from Plots.plotly_plots import Plot_Model_Ouptuts

class TensorFlow_Time_Series_Model:
    def model_1(self, observed_time_series, seasonal_array):   
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


    def variational_loss_function(self, model, observed_time_series):
        variational_posteriors = tfp.sts.build_factored_surrogate_posterior(model = model)
        
        # Allow external control of optimization to reduce test runtimes.
        num_variational_steps = 200 # @param { isTemplate: true}
        num_variational_steps = int(num_variational_steps)

        # Build and optimize the variational loss function.
        elbo_loss_curve_var = tfp.vi.fit_surrogate_posterior(target_log_prob_fn = model.joint_log_prob(observed_time_series = observed_time_series),
                                                                surrogate_posterior = variational_posteriors,
                                                                optimizer = tf.optimizers.Adam(learning_rate = 0.1),
                                                                num_steps = num_variational_steps)

        # Draw samples from the variational posterior.
        q_samples_var = variational_posteriors.sample(50)
        
        return elbo_loss_curve_var, q_samples_var


    def component_breakdown(self, model, observed_time_series, parameter_samples):
        component_dists = sts.decompose_by_component(model = model, 
                                                    observed_time_series = observed_time_series, 
                                                    parameter_samples = parameter_samples)
        
        component_means_vals, component_stddevs_vals = ({k.name: c.mean() for k, c in component_dists.items()},
                                                {k.name: c.stddev() for k, c in component_dists.items()})
        
        return component_means_vals, component_stddevs_vals


    def one_step_predictions(self, model, observed_time_series, parameter_samples):      
        one_step_dist = sts.one_step_predictive(model = model,
                                                observed_time_series = observed_time_series,
                                                parameter_samples = parameter_samples)
        
        one_step_mean_var, one_step_scale_var = (one_step_dist.mean().numpy(), one_step_dist.stddev().numpy())
        
        return one_step_mean_var, one_step_scale_var

class Run_Model:
    def run_time_series_model(self, train_data):
        model_inputs = Model_Inputs()
        seasonal_array_matrix = model_inputs.seasonality_matrix(train_data = train_data)

        tf_time_series_model = TensorFlow_Time_Series_Model()
        plot_model_outputs = Plot_Model_Ouptuts()

        ts_model = tf_time_series_model.model_1(observed_time_series = train_data['value'], 
                                                seasonal_array = seasonal_array_matrix)

        variational_loss_values = tf_time_series_model.variational_loss_function(model = ts_model, 
                                                                                 observed_time_series = train_data['value'])
        elbo_loss_curve = variational_loss_values[0]
        q_samples = variational_loss_values[1]
        plot_model_outputs.elbow_plot(elbo_loss_curve = elbo_loss_curve)

        component_breakdowns = tf_time_series_model.component_breakdown(model = ts_model, 
                                                                        observed_time_series = train_data['value'], 
                                                                        parameter_samples = q_samples)

        component_means_ = component_breakdowns[0]
        component_stddevs_ = component_breakdowns[1]
        plot_model_outputs.component_breakdown_plot(component_means_ = component_means_, 
                                                    date = train_data['date'])


        prediction_values = tf_time_series_model.one_step_predictions(model = ts_model, 
                                                                      observed_time_series = train_data['value'], 
                                                                      parameter_samples = q_samples)
        one_step_mean = prediction_values[0]
        one_step_scale = prediction_values[1]

        plot_model_outputs.prediction_plot(one_step_mean = one_step_mean, 
                                           actual_data = train_data['value'], 
                                           date = train_data['date'])