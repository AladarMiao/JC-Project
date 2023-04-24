from xgboost import XGBClassifier
from Classifier.Classifier import Classifier

class XGBoost(Classifier):

    def __init__(self, **kwargs):
        super().__init__()
        self.clf = XGBClassifier(**kwargs)