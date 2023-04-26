class DimensionReduction:
    def __init__(self, n_components=2):
        self.model = None
        self.n_components = n_components

    def fit_transform(self, data):
        return self.model.fit_transform(data)