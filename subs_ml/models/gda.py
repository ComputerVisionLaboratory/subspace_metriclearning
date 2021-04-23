import numpy as np
import torch
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics.pairwise import euclidean_distances as l2dist
from subs_ml.utils import mutual_subsim, pca_for_sets, subspacea_subs

from .base_model import MSMBase


class GrassmannDiscriminantAnalysis(MSMBase):

    param_names = {"p", "n_sdim", "dic"}

    def __init__(self, n_sdim, normalize=False, verbose=0, seed=0):
        super(GrassmannDiscriminantAnalysis, self).__init__(
            n_sdim, normalize=normalize, verbose=verbose, seed=seed
        )
        self.lda = LinearDiscriminantAnalysis()

    def _fit(self, x_train, y_train):

        self.metric_mat = torch.eye(x_train[0].shape[0])

        k_mat, _ = mutual_subsim(self.sub_basis, torch.eye(400), False)
        self.lda = self.lda.fit(k_mat.cpu().numpy(), y_train)
        self.dic = self.lda.transform(k_mat.cpu().numpy())

    def predict_proba(self, x_pred):
        sub_basis = pca_for_sets(x_pred, self.n_sdim, self.p_norm)
        sim, _, _ = subspacea_subs(
            self.sub_basis, sub_basis, self.metric_mat, False
        )

        sim = sim.cpu().numpy()
        sim = self.lda.transform(sim.T)
        sim = l2dist(self.dic, sim)

        pred = [
            1 / (1e-10 + np.min(sim[self.labels == i, :], axis=0))
            for i in range(self.n_classes)
        ]
        return np.asarray(pred)
