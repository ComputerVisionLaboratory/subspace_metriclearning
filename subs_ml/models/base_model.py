import abc

import numpy as np
import scipy as sp
import torch
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import LabelEncoder
from subs_ml.utils import gpu_free, ortha_subs, pca_for_sets, subspacea_subs


class MSMBase(BaseEstimator, ClassifierMixin):

    param_names = {"normalize", "n_sdim", "device", "max_iter"}

    def __init__(self, n_sdim, normalize=False, verbose=0, seed=0):
        self.n_sdim = n_sdim
        self.verbose = verbose
        self.label_encoder = LabelEncoder()
        self.p_norm = 2 if normalize else -1

        self.n_classes = None
        self.labels = None
        self.sub_basis = None
        self.n_samples = None
        self.dic = None
        self._do_orth = False
        self.metric_mat = None
        self.reddim_mat = None

        self.params = ()

        torch.manual_seed(seed)
        np.random.seed(seed)
        sp.random.seed(seed)

    def get_params(self, deep=True):
        return {name: getattr(self, name) for name in self.param_names}

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self

    def _train_le(self, label):
        self.labels = self.label_encoder.fit_transform(label)
        self.n_classes = self.label_encoder.classes_.size

    def fit(self, x_train, y_train):
        self._train_le(y_train)
        self.n_samples = len(x_train)
        self.sub_basis = pca_for_sets(
            x_train, self.n_sdim, self.p_norm
        ).contiguous()

        self._fit(x_train, y_train)

        gpu_free()

    @abc.abstractmethod
    def _fit(self, x_train, y_train):
        raise NotImplementedError()

    def predict(self, x_pred):
        proba = self.predict_proba(x_pred)
        return self.proba2class(proba)

    def proba2class(self, proba):
        pred = np.argmax(proba, axis=0)
        return self.label_encoder.inverse_transform(pred)

    def predict_proba(self, x_pred, ret_kmat=False):
        if self.reddim_mat is not None:
            if self.reddim_mat.shape[0] != self.reddim_mat.shape[1]:
                x_pred = [self.reddim_mat @ i for i in x_pred]
        sub_basis = pca_for_sets(x_pred, self.n_sdim, self.p_norm)
        sub_basis = ortha_subs(sub_basis, self.metric_mat)
        sim, _, _ = subspacea_subs(self.dic, sub_basis, self.metric_mat, False)

        sim = sim.cpu().numpy()
        pred = [
            np.max(sim[self.labels == i, :], axis=0)
            for i in range(self.n_classes)
        ]

        if ret_kmat:
            return np.asarray(pred), sim

        return np.asarray(pred)
