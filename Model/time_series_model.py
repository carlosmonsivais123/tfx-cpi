from turtle import onclick
import pandas as pd
import numpy as np

import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow_probability import sts

# from Variables.create_vars import *

# data = pd.read_csv('gs://cpi_bucket/Airline fares/Airline fares_training.csv', 
#                    storage_options={"token": "{}".format(gcp_key)})

# train_data = pd.read_csv('gs://cpi_bucket/Airline fares/Airline fares_training.csv',
#                          storage_options={"token": '/Users/CarlosMonsivais/Desktop/CPI Project/cpi-tfx-203342181a05.json'})

train_data = pd.read_csv('plane_data_test.csv')

train_data = train_data.astype({'series_id': str, 
                                'value': float, 
                                'date': 'datetime64[ns]',
                                'item': str})
train_data['month'] = train_data['date'].dt.month
train_data['year'] = train_data['date'].dt.year
train_data.sort_values(by = ['date'], inplace = True, ascending = True)

# Try this when I get home bacfill the dataframe with nan based on the date being 12 on the monthly basis
# train_data.bfill()
# print(train_data)

seasonal_months = train_data.groupby('year')['month'].apply(np.array).reset_index(name='months')['months'].to_numpy()
seasonal_months_array = np.reshape(seasonal_months, (seasonal_months.shape[0], -1))

# print(seasonal_months_array)

fix_this_array = np.array([[np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, 12],
[np.nan, np.nan, np.nan, np.nan, np.nan, 6, np.nan, np.nan, np.nan, np.nan, np.nan, 12],
[np.nan, np.nan, np.nan, np.nan, np.nan, 6, np.nan, np.nan, np.nan, np.nan, np.nan, 12],
[np.nan, np.nan, 3, np.nan, np.nan, 6, np.nan, np.nan, 9, np.nan, np.nan, 12],
[np.nan, np.nan, 3, np.nan, np.nan, 6, np.nan, np.nan, 9, np.nan, np.nan, 12],
[np.nan, np.nan, 3, np.nan, np.nan, 6, np.nan, np.nan, 9, np.nan, np.nan, 12],
[1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12],
[1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12],
[1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12],
[1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12],
[1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12],
[1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12],
[1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12],
[1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12],
[1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12],
[1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12],
[1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12],
[1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12],
[1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12],
[1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12],
[1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12],
[1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12],
[1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12],
[1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12],
[1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12],
[1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12],
[1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12],
[1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12],
[1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12],
[1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12],
[1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12],
[1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12],
[1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12],
[1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12],
[1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12],
[1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12],
[1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12],
[1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12],
[1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12],
[1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12],
[1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12],
[1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12],
[1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12],
[1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12],
[1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12],
[1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12],
[1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12],
[1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12],
[1, 2, 3, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan]])

# print(fix_this_array)




def model_1(observed_time_series):   
    trend = sts.LocalLinearTrend(observed_time_series = observed_time_series,
                                 name = 'trend')

    seasonal = sts.Seasonal(num_seasons = 12,
                            num_steps_per_season = fix_this_array,
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


def variational_loss_function(model, observed_time_series):
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
    
    return elbo_loss_curve_var, q_samples_var

time_series_model = model_1(observed_time_series = train_data['value'])

variational_loss_values = variational_loss_function(model = time_series_model, observed_time_series = train_data['value'])

elbo_loss_curve = variational_loss_values[0]
q_samples = variational_loss_values[1]

# print('Elbo Losss Values: {}\n'.format(elbo_loss_curve))
# print('Q Samples: {}'.format(q_samples))
# print('\n\n\n\n\n\n\n\n')
# print("Inferred parameters:")
# for param in time_series_model.parameters:
#     print("{}: {} +- {}".format(param.name,
#                                 np.mean(q_samples[param.name], axis=0),
#                                 np.std(q_samples[param.name], axis=0)))

def component_breakdown(model, observed_time_series, parameter_samples):
    component_dists = sts.decompose_by_component(model = model, 
                                                 observed_time_series = observed_time_series, 
                                                 parameter_samples = parameter_samples)
    
    component_means_vals, component_stddevs_vals = ({k.name: c.mean() for k, c in component_dists.items()},
                                            {k.name: c.stddev() for k, c in component_dists.items()})
    
    return component_means_vals, component_stddevs_vals

component_breakdowns = component_breakdown(model = time_series_model, 
                                           observed_time_series = train_data['value'], 
                                           parameter_samples = q_samples)

component_means_ = component_breakdowns[0]
component_stddevs_ = component_breakdowns[1]

print(component_means_)
print(component_stddevs_)