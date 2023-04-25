from Classifier.Classifier import Classifier
from sklearn.tree import DecisionTreeClassifier


class DecisionTree(Classifier):

    def __init__(self, **kwargs):
        super().__init__()
        self.clf = DecisionTreeClassifier(**kwargs)
