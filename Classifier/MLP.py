from sklearn.neural_network import MLPClassifier

from Classifier.Classifier import Classifier


class MLP(Classifier):

    def __init__(self, **kwargs):
        super().__init__()
        self.clf = MLPClassifier(**kwargs)
