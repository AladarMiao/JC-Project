from sklearn.ensemble import RandomForestClassifier

from Classifier.Classifier import Classifier

class RandomForest(Classifier):

    def __init__(self, **kwargs):
        super().__init__()
        self.clf = RandomForestClassifier(**kwargs)