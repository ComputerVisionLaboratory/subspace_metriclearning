import numpy as np
import torch
from subs_ml.utils import ortha_subs, pca_for_sets, subspacea_subs

from .base_model import MSMBase


class MutualSubspaceMethod(MSMBase):

    param_names = {"p", "n_sdim", "dic"}

    def __init__(self, n_sdim, normalize=False, verbose=0, seed=0):
        super(MutualSubspaceMethod, self).__init__(
            n_sdim, normalize=normalize, verbose=verbose, seed=seed
        )

    def _fit(self, x_train, y_train):
        self.metric_mat = torch.eye(x_train[0].shape[0]).to(x_train[0].device)
        self.dic = self.sub_basis

    def predict_proba(self, x_pred, ret_kmat=False):
        if self.reddim_mat is not None:
            if self.reddim_mat.shape[0] != self.reddim_mat.shape[1]:
                x_pred = [self.reddim_mat @ i for i in x_pred]

        sub_basis = pca_for_sets(x_pred, self.n_sdim, self.p_norm)
        sim, _, _ = subspacea_subs(self.dic, sub_basis, self.metric_mat, False)

        sim = sim.cpu().numpy()
        pred = [
            np.max(sim[self.labels == i, :], axis=0)
            for i in range(self.n_classes)
        ]

        if ret_kmat:
            return np.asarray(pred), sim

        return np.asarray(pred)


class ConstrainedMutualSubspaceMethod(MSMBase):

    param_names = {"p", "n_sdim", "dic", "n_reducedim"}

    def __init__(self, n_sdim, n_reducedim, normalize=False, verbose=0, seed=0):
        super(ConstrainedMutualSubspaceMethod, self).__init__(
            n_sdim, normalize=normalize, verbose=verbose, seed=seed
        )
        self.n_reducedim = n_reducedim

    def _fit(self, x_train, y_train):
        cls_data = [
            torch.cat(
                [x_train[j] for j in range(len(y_train)) if y_train[j] == i], 1
            )
            for i in set(y_train)
        ]
        sub_basis = (
            pca_for_sets(cls_data, self.n_sdim, self.p_norm)
            .contiguous()
            .permute((2, 0, 1))
        )

        gram_mat = (sub_basis @ sub_basis.permute((0, 2, 1))).sum(0)
        full_dim = torch.matrix_rank(gram_mat)
        _, eig_vec = torch.symeig(gram_mat, eigenvectors=True)
        eig_vec = eig_vec.flip(1)[:, self.n_reducedim : full_dim]
        self.metric_mat = eig_vec @ eig_vec.T
        self.dic = ortha_subs(self.sub_basis, self.metric_mat)


class OrthogonalMutualSubspaceMethod(MSMBase):

    param_names = {"p", "n_sdim", "dic"}

    def __init__(self, n_sdim, normalize=False, verbose=0, seed=0):
        super(OrthogonalMutualSubspaceMethod, self).__init__(
            n_sdim, normalize=normalize, verbose=verbose, seed=seed
        )

    def _fit(self, x_train, y_train):
        cls_data = [
            torch.cat(
                [x_train[j] for j in range(len(y_train)) if y_train[j] == i], 1
            )
            for i in set(y_train)
        ]
        sub_basis = (
            pca_for_sets(cls_data, self.n_sdim, self.p_norm)
            .contiguous()
            .permute((2, 0, 1))
        )

        gram_mat = (sub_basis @ sub_basis.permute((0, 2, 1))).sum(0)
        full_dim = torch.matrix_rank(gram_mat)
        eig_val, eig_vec = torch.symeig(gram_mat, eigenvectors=True)
        eig_vec = eig_vec.flip(1)[:, 0:full_dim]
        eig_val = eig_val.flip(0)[0:full_dim]

        self.metric_mat = (
            eig_vec @ torch.diag(eig_val.abs().pow(-1)) @ eig_vec.T
        )
        self.dic = ortha_subs(self.sub_basis, self.metric_mat)
