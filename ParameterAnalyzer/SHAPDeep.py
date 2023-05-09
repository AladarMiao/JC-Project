import shap
import numpy as np
from ParameterAnalyzer.ParameterAnalyzer import ParameterAnalyzer
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Flatten

class SHAPDeep(ParameterAnalyzer):
    def add_flatten_layer(model):
        x = model.output
        x = Flatten()(x)
        model_with_flatten = Model(inputs=model.input, outputs=x)
        return model_with_flatten
    def __init__(self, model, data):
        flattened_model = SHAPDeep.add_flatten_layer(model)
        self.explainer = shap.DeepExplainer(flattened_model, data)
