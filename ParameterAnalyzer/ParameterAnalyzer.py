import os
import shap
import matplotlib.pyplot as plt
from datetime import datetime

class ParameterAnalyzer:

    def __init__(self):
        self.explainer = None
    def get_explainer(self):
        return self.explainer

    def plot_explainer(self, data):

        shap_values = self.explainer.shap_values(data)

        # Create folder to store plots
        folder_name = 'shap_plots'
        if not os.path.exists(folder_name):
            os.makedirs(folder_name)

        # Add timestamp to filenames
        timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

        # Visualize the SHAP values using a summary plot
        shap.summary_plot(shap_values[1], data)
        plt.savefig('{}/summary_plot_{}.png'.format(folder_name, timestamp))

        # Explore the relationship between a feature and the model output using a dependence plot
        shap.dependence_plot('worst perimeter', shap_values[1], data)
        plt.savefig('{}/dependency_plot_{}.png'.format(folder_name, timestamp))