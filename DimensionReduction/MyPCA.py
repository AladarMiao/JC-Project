from sklearn.decomposition import PCA
from DimensionReduction.DimensionReduction import DimensionReduction


class MyPCA(DimensionReduction):
    def __init__(self, n_components=None):
        super().__init__()
        if not n_components:
            n_components = self.n_components
        self.model = PCA(n_components=n_components)
