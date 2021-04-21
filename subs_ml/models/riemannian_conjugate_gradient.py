""""""
from __future__ import division, print_function

import time
from copy import deepcopy

import numpy as np
import torch
from pymanopt.solvers.solver import Solver


class LineSearchAdaptive:
    """
    Adaptive line-search
    """

    def __init__(
        self,
        contraction_factor=0.7,
        suff_decr=0.5,
        maxiter=3,
        initial_stepsize=100,
    ):
        self._contraction_factor = contraction_factor
        self._suff_decr = suff_decr
        self._maxiter = maxiter
        self._initial_stepsize = initial_stepsize
        self._oldalpha = None

    def search(self, objective, man, x, d, f0, df0):
        """
        :param objective:
        :param man:
        :param x:
        :param d:
        :param f0:
        :param df0:
        :return:
        """
        norm_d = man.norm(x, d)

        if self._oldalpha is None:
            alpha = self._initial_stepsize / norm_d
        else:
            alpha = self._oldalpha

        alpha = np.min((float(alpha), 300))

        newx = man.retr(x, alpha * d)

        for _ in range(200):
            if torch.isnan(newx).any():
                alpha *= 0.9
                newx = man.retr(x, alpha * d)
            else:
                break

        newf = objective(newx)
        cost_evaluations = 1

        for _ in range(self._maxiter):
            if newf <= f0 + self._suff_decr * alpha * df0:
                break
            alpha *= self._contraction_factor
            newx = man.retr(x, alpha * d)
            newf = objective(newx)
            cost_evaluations += 1

        if cost_evaluations <= 2:
            self._oldalpha = alpha
        # If things went very well or we backtracked a lot (meaning the step
        # size is probably quite small), speed up.
        else:
            self._oldalpha = 2 * alpha

        if self._oldalpha < 1e-1 and newf < f0:
            self._oldalpha = None
            self._initial_stepsize /= 10

        return newx, newf


class ConjugateGradient(Solver):
    """
    pytorch impl of pymanopt's CG method
    """

    def __init__(self, problem, linesearch=None, *args, **kwargs):
        super(ConjugateGradient, self).__init__(*args, **kwargs)

        if linesearch is None:
            self._linesearch = LineSearchAdaptive()
        else:
            self._linesearch = linesearch
        self.linesearch = deepcopy(self._linesearch)

        self._cost = None
        self._grad = None
        self._gradnorm = None
        self._Pgrad = None
        self._gradPgrad = None
        self._desc_dir = None
        self.man = problem.manifold
        self.verbosity = problem.verbosity
        self.objective = problem.cost
        self.gradient = problem.grad
        self.precon = problem.precon

    def step(self, x):
        """
        :param x:
        :return:
        """
        # Calculate initial cost-related quantities
        if self._cost is None:
            self._cost = self.objective(x)
        if self._grad is None:
            self._grad = self.gradient(x)
        if self._Pgrad is None:
            self._Pgrad = self.precon(x, self._grad)
        if self._gradPgrad is None:
            self._gradPgrad = self.man.inner(x, self._grad, self._Pgrad)
        if self._desc_dir is None:
            self._desc_dir = -1 * self._Pgrad

        cost = self._cost
        grad = self._grad
        Pgrad = self._Pgrad
        gradPgrad = self._gradPgrad
        desc_dir = self._desc_dir
        inner = self.man.inner

        if self.verbosity >= 2:
            gradnorm = self.man.norm(x, grad)
            print("%+.16e\t %.8e\t %.8e" % (cost, gradnorm, x.norm()))

        df0 = inner(x, grad, desc_dir)

        if df0 >= 0:
            # Reset to negative gradient: this discards the CG memory.
            desc_dir = -1 * Pgrad
            df0 = -1 * gradPgrad

        # Execute line search
        newx, newcost = self.linesearch.search(
            self.objective, self.man, x, desc_dir, cost, df0
        )
        # Compute the new cost-related quantities for newx
        newgrad = self.gradient(newx)
        Pnewgrad = self.precon(newx, newgrad)

        newgradPnewgrad = inner(newx, newgrad, Pnewgrad)

        # Apply the CG scheme to compute the next search direction
        oldgrad = self.man.transp(x, newx, grad)
        desc_dir = self.man.transp(x, newx, desc_dir)

        # Hester's Stiefel
        diff = newgrad - oldgrad
        ip_diff = inner(newx, Pnewgrad, diff)

        try:
            beta = max(0, ip_diff / inner(newx, diff, desc_dir))
        except ZeroDivisionError:
            beta = 1

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        # Update the necessary variables for the next iteration.
        self._cost = newcost
        self._grad = newgrad
        self._Pgrad = Pnewgrad
        self._gradPgrad = newgradPnewgrad
        self._desc_dir = -Pnewgrad + beta * desc_dir

        return newx, newcost

    def solve(self, x, max_iter=10):
        """
        :param x:
        :param max_iter:
        :return:
        """
        for _ in range(max_iter):
            x, _ = self.step(x)
        return x
