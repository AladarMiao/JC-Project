from sklearn.manifold import TSNE

from DimensionReduction.DimensionReduction import DimensionReduction


class MyTSNE(DimensionReduction):
    def __init__(self, n_components=None):
        super().__init__()
        if not n_components:
            n_components = self.n_components
        self.model = TSNE(n_components=n_components)