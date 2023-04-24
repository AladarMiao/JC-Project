from sklearn.manifold import TSNE

from DimensionReduction.DimensionReduction import DimensionReduction


class MyTSNE(DimensionReduction):
    def __init__(self, n_components):
        super().__init__(n_components)

    def fit_transform(self, data):
        self.model = TSNE(n_components=self.n_components, perplexity=30, n_iter=500)
        X_tsne = self.model.fit_transform(data)
        return X_tsne