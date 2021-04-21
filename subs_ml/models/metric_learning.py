import numpy as np
import pymanopt
import torch
from pymanopt import Problem
from subs_ml.utils import (
    lmm_cost,
    lmm_egrad,
    logdet_cost,
    logdet_egrad,
    lp_normalize,
    ortha_subs,
    pca_for_sets,
)
from subs_ml.utils.spd_manifold import SPDManopt as SPD

from .base_model import MSMBase
from .riemannian_conjugate_gradient import ConjugateGradient, LineSearchAdaptive


class AbasedMetricLearningSubspace(MSMBase):

    param_names = {"n_sdim", "max_iter"}

    def __init__(
        self,
        n_sdim,
        n_neighbors=10,
        alpha=0.5,
        normalize=False,
        max_iter=5,
        min_iter=0,
        tol=1e-3,
        weight_trnorm=0,
        verbose=0,
        seed=0,
    ):
        """"""
        super(AbasedMetricLearningSubspace, self).__init__(
            n_sdim, normalize=normalize, verbose=verbose, seed=seed
        )

        self.n_neighbors = n_neighbors
        self.alpha = torch.FloatTensor([alpha])
        self.max_iter = max_iter
        self.min_iter = min_iter
        self.cost_trans = torch.zeros(max_iter + 1)
        self.margin_trans = torch.zeros(max_iter + 1)
        self.tol = tol
        self.weight_trnorm = weight_trnorm

        self.metric_mat = None
        self.k_mat = None
        self._sub_basis = None
        self.n_sw = None
        self.n_sb = None
        self.metric_dim = []

    def _prox(self, metric_mat, _x_train, eta):
        # TODO: move this function to utilities
        eig_val, eig_vec = torch.symeig(metric_mat, True)
        med = eig_val.median() * 0.8
        if med < eta:
            eig_val = torch.relu(eig_val - med)
            eta *= med
        else:
            eig_val = torch.relu(eig_val - eta)

        space_dim = (eig_val > 1e-8).sum()
        s_dim = eig_val.shape[0] - space_dim
        new_metric = (
            eig_vec[:, s_dim:] @ eig_val[s_dim:].diag() @ eig_vec[:, s_dim:].T
        )
        space_dim = torch.matrix_rank(new_metric)

        if space_dim != new_metric.shape[0]:
            _x_data = lp_normalize(
                self.reddim_mat @ torch.cat(_x_train, 1), self.p_norm
            )
            autocorr_mat = new_metric @ _x_data @ _x_data.T
            d, sing_vec = autocorr_mat.cpu().eig(True)
            d = d.to(autocorr_mat)
            sing_vec = sing_vec.to(autocorr_mat)
            sing_vec = sing_vec * d[:, 0]
            sing_vec, _, _ = sing_vec.svd()
            proj_basis = sing_vec[:, :space_dim].T

            for _ in range(10):
                _tmp_new_metric = proj_basis @ new_metric @ proj_basis.T
                _rank = torch.matrix_rank(_tmp_new_metric)
                if _rank < space_dim:
                    space_dim = _rank
                    proj_basis = sing_vec[:, :space_dim].T
                    _tmp_new_metric = proj_basis @ new_metric @ proj_basis.T
                else:
                    break
            new_metric = _tmp_new_metric
            _reddim_mat = proj_basis @ self.reddim_mat

            _red_subs = proj_basis @ self.sub_basis.permute((2, 0, 1))
            sub_basis, _ = torch.qr(_red_subs)
            sub_basis = sub_basis.to(_red_subs.device).permute((1, 2, 0))

        else:
            _reddim_mat = self.reddim_mat
            sub_basis = self.sub_basis
        return new_metric, _reddim_mat, sub_basis, eta

    def _cost(self, metric_mat, sw_idx, sb_idx, _sub_basis=None):
        if _sub_basis is None:
            _sub_basis = self.sub_basis

        cost, self.k_mat = lmm_cost(
            _sub_basis,
            metric_mat,
            sb_idx,
            sw_idx,
            self.n_sb,
            self.n_sw,
            self.alpha,
            self.n_sdim,
        )
        return cost

    def _egrad(self, metric_mat, sw_idx, sb_idx):
        grad_sb, grad_sw, _, _, penalty = lmm_egrad(
            self.k_mat,
            self._sub_basis,
            metric_mat,
            sb_idx,
            sw_idx,
            self.n_sb,
            self.n_sw,
        )
        euc_grad = (1 - self.alpha) * grad_sb - self.alpha * grad_sw - penalty
        return euc_grad

    def _initialize(self, x_train):
        if self.weight_trnorm > 0:
            self.reddim_mat = torch.eye(x_train[0].shape[0]).to(
                x_train[0].device
            )
            _x_data = lp_normalize(torch.cat(x_train, 1), self.p_norm)
            d, _ = torch.symeig(_x_data @ _x_data.T)
            d = d.flip(0)
            self.n_mindim = int((d.cumsum(0) / d.sum() < 0.99).sum())
        else:
            self.n_mindim = 0

        self.alpha = self.alpha.to(x_train[0].device)
        self._sub_basis = self.sub_basis.transpose(2, 0).contiguous()
        lmat = np.tile(self.labels, (self.labels.shape[0], 1))
        self.sw_idx = lmat == lmat.T
        self.sb_idx = lmat != lmat.T

        self.n_sw = 1 + np.min(
            (self.n_neighbors, np.sum(self.sw_idx, axis=1)[0] - 1)
        )
        self.n_sb = np.min((self.n_neighbors, np.sum(self.sb_idx, axis=1)[0]))

        @pymanopt.function.Callable
        def cost(_metric_mat):
            return self._cost(_metric_mat, self.sw_idx, self.sb_idx)

        @pymanopt.function.Callable
        def egrad(_metric_mat):
            return self._egrad(_metric_mat, self.sw_idx, self.sb_idx)

        man = SPD(self.sub_basis.shape[0])
        line_search = LineSearchAdaptive()
        problem = Problem(
            manifold=man, cost=cost, egrad=egrad, verbosity=self.verbose
        )
        solver = ConjugateGradient(
            problem, minstepsize=1e-1, linesearch=line_search
        )

        return man, problem, solver, line_search

    def _dimension_reduction(self, metric_mat, x_train, i, solver, line_search):
        def cost(_metric_mat):
            return self._cost(_metric_mat, self.sw_idx, self.sb_idx)

        def egrad(_metric_mat):
            return self._egrad(_metric_mat, self.sw_idx, self.sb_idx)

        eta = self.weight_trnorm
        for _ in range(3):
            # back tracking
            _new_metric, _reduce_mat, _sub_basis, eta = self._prox(
                metric_mat, x_train, eta
            )
            _cost_val = self._cost(
                _new_metric, self.sw_idx, self.sb_idx, _sub_basis
            )
            if self.cost_trans[i] - _cost_val > -1e-2:
                metric_mat = _new_metric
                if metric_mat.shape[0] != self.sub_basis.shape[0]:
                    _dim_update = i
                    self.reddim_mat = _reduce_mat
                    self.sub_basis = _sub_basis
                    self._sub_basis = self.sub_basis.transpose(
                        2, 0
                    ).contiguous()
                    man = SPD(metric_mat.shape[0])
                    problem = Problem(
                        manifold=man,
                        cost=cost,
                        egrad=egrad,
                        verbosity=self.verbose,
                    )
                    _oldalpha = solver.linesearch._oldalpha
                    _initial_stepsize = solver.linesearch._initial_stepsize
                    solver = ConjugateGradient(
                        problem,
                        minstepsize=1e-1,
                        linesearch=line_search,
                    )
                    solver.linesearch._oldalpha = _oldalpha
                    solver.linesearch._initial_stepsize = _initial_stepsize
                break
            else:
                eta *= 0.9

        self.weight_trnorm = eta

        return _dim_update

    def _fit(self, x_train, y_train):
        man, problem, solver, line_search = self._initialize(x_train)

        metric_mat = man.rand().to(x_train[0])
        metric_mat += torch.eye(metric_mat.shape[0]).to(metric_mat)
        metric_mat /= metric_mat.norm()
        min_cost = np.Inf
        self.cost_trans[0] = self._cost(metric_mat, self.sw_idx, self.sb_idx)

        # TODO: move this function to utilities
        def calc_min_margin():
            margin = np.inf
            for i in range(self.k_mat.shape[0]):
                sw_sim = self.k_mat[i, self.labels == self.labels[i]]
                sw_sim, _ = sw_sim.sort(descending=True)

                sb_sim = self.k_mat[i, self.labels != self.labels[i]]
                sb_sim, _ = sb_sim.sort(descending=True)

                _m = sw_sim[1] - sb_sim[0]

                if margin > _m:
                    margin = _m
            return margin

        self.margin_trans[0] = calc_min_margin()
        _dim_update = 0

        for i in range(1, self.max_iter + 1):
            try:
                # RCG Step
                metric_mat, self.cost_trans[i] = solver.step(metric_mat)

                # Save the current minimum margin
                self.margin_trans[i] = calc_min_margin()

                if (
                    self.weight_trnorm > 0
                    and self.n_mindim < metric_mat.shape[0]
                    and i < self.max_iter - self.min_iter
                ):
                    # Apply dimension reduction
                    _dim_update = self._dimension_reduction(
                        metric_mat, x_train, i, solver, line_search
                    )

                # Save the current dimension of the metric space
                if self.weight_trnorm > 0:
                    self.metric_dim.append(metric_mat.shape[0])

                # Update the metric matrix
                if self.cost_trans[i] < min_cost or self.weight_trnorm > 0:
                    min_cost = self.cost_trans[i]
                    self.metric_mat = metric_mat.clone()

                # Finish the loop if the cost cannot be decreased
                diff_cost = (self.cost_trans[i] - self.cost_trans[i - 1]).abs()
                if diff_cost < self.tol and i - _dim_update > self.min_iter:
                    break

            except RuntimeError as err:
                # Restart due to numerical error
                metric_mat += 1e-2 * man.rand().to(x_train[0].device)
                self.cost_trans[i] = np.Inf
                if self.verbose > 0:
                    print(err)

        # Set dictionary subspaces with respect to the learnd metric
        self.dic = ortha_subs(self.sub_basis, self.metric_mat)
