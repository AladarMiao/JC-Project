from sklearn.cluster import AgglomerativeClustering
import shap

from ParameterAnalyzer.ParameterAnalyzer import ParameterAnalyzer


class SHAPTree(ParameterAnalyzer):
    def __init__(self, model):
        self.explainer = shap.TreeExplainer(model)
