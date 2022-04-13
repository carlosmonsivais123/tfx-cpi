import plotly.graph_objects as go
from plotly.subplots import make_subplots

class Plot_Model_Ouptuts:
    def elbow_plot(self, elbo_loss_curve):
        fig = go.Figure()
        fig.add_trace(go.Scatter(x = list(range(0, len(elbo_loss_curve))), 
                                y = elbo_loss_curve,
                                mode = 'lines+markers',
                                name = 'Elbow Loss Curve'))

        fig.update_layout(
            title={
                'text': "Elbow Loss Curve",
                'x':0.5,
                'xanchor': 'center',
                'yanchor': 'top'})

        fig.show()


    def component_breakdown_plot(self, component_means_, date):
        fig = make_subplots(rows = len(component_means_), 
                            cols = 1,
                            subplot_titles = (tuple(component_means_)))

        i = 0
        for key in component_means_.keys():
            i = i + 1
            fig.add_trace(go.Scatter(x = date, 
                                     y = component_means_['{}'.format(key)]),
                          row = i, 
                          col = 1)

        fig.show()


    def prediction_plot(self, one_step_mean, actual_data, date):
        fig = go.Figure()
        fig.add_trace(go.Scatter(x = date, 
                                 y = actual_data,
                                 mode = 'lines+markers',
                                 name = 'Train'))

        fig.add_trace(go.Scatter(x = date, 
                                 y = one_step_mean,
                                 mode = 'lines+markers',
                                 name = 'Test'))

        fig.update_layout(
            title={
                'text': "Predictions",
                'x':0.5,
                'xanchor': 'center',
                'yanchor': 'top'})

        fig.show()