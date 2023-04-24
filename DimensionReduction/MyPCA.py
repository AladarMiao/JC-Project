from sklearn.decomposition import PCA
from DimensionReduction.DimensionReduction import DimensionReduction


class MyPCA(DimensionReduction):
    def __init__(self, n_components):
        super().__init__(n_components)

    def fit_transform(self, data):
        self.model = PCA(n_components=self.n_components)
        X_pca = self.model.fit_transform(data)
        return X_pca