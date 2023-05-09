from sklearn.metrics import accuracy_score

class Classifier:

    def __init__(self):
        # clf stands for classifier
        self.clf = None

    def fit(self, X, y):
        self.clf.fit(X, y)

    def predict(self, X):
        return self.clf.predict(X)

    def get_classifier(self):
        return self.clf

    def score(self, X, y):
        y_pred = self.predict(X)
        print(accuracy_score(y, y_pred))
        return accuracy_score(y, y_pred)
