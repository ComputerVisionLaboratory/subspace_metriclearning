""""""
from copy import deepcopy

import geomstats.backend as gs
import numpy as np
import torch
from pymanopt.solvers.solver import Solver


class LineSearch:
    """
    Adaptive line-search
    """

    def __init__(
        self,
        maxiter=30,
        c1=1e-2,
        c2=0.2,
        alpha_max=10,
        init_alpha=1e-1,
    ):
        self.max_iter = maxiter
        self.c1 = c1
        self.c2 = c2
        self.alpha_max = alpha_max
        self._init_alpha = torch.FloatTensor([init_alpha])

    def interpolate(self, x, y, dfx, dfy, fx, fy):
        """
        cubic interpolation of x and y with the safe guard.
        equation 35, 35 in [1]
        [1] H. Sato, "A Dai–Yuan-type Riemannian conjugate gradient method with the weak Wolfe conditions.
        " Computational Optimization and Applications, 2016.

        """
        sgn = torch.sign(y - x)
        d1 = dfx + dfy - 3 * (fx - fy) / (x - y)
        d2 = sgn * torch.sqrt(d1.pow(2) - dfx * dfy)
        alpha = y - (y - x) * (dfy + d2 - d1) / (dfy - dfx + 2 * d2)
        return max(alpha, 2 * y - x)
        # return min(max(alpha, 2 * y - x), y + 9 * (y - x))

    def zoom(self, objective, man, x, d, f0, df0, lo, hi, gradient):
        """
        Algorithm 3.6 in [2]
        [2] J. Nocedal and S. Wright, "Numerical Optimization",
        Series in Operations Research and Financial Engineering, Springer, 2006.
        """
        if lo > hi:
            lo, hi = hi, lo
        alpha = (lo + hi) / 2

        def get_new_obs(_x, _alpha, _d):
            _newx = man.retr(_x, _alpha * _d)
            _newf = objective(_newx)
            _newgrad = gradient(_newx)
            _transd = man.transp(_x, _newx, _d)
            _newdf = man.inner(_x, _newgrad, _transd)

            return _newf, _newdf

        lof, lodf = get_new_obs(x, lo, d)
        hif, hidf = get_new_obs(x, hi, d)

        for _ in range(self.max_iter):
            newf, df = get_new_obs(x, alpha, d)

            if newf > f0 + self.c1 * alpha * df0 or newf >= lof:
                hi = alpha
                hif = newf
                hidf = df

            else:
                if torch.abs(df) <= -self.c2 * df0:
                    return alpha

                if df * (hi - lo) >= 0:
                    hi = lo
                    hif = lof
                    hidf = lof

                lo = alpha
                lof = newf
                lodf = df
            if lo > hi:
                lo, hi = hi, lo
            alpha = self.interpolate(lo, hi, lodf, hidf, lof, hif)

            if torch.abs(hi - lo) < 1e-1:
                return hi

            if alpha > self.alpha_max:
                return alpha

        return alpha

    def check_strong_wolf(self, alpha, f0, df0, newf, newdf):
        cond1 = newf <= f0 + self.c1 * alpha * df0
        cond2 = gs.abs(newdf) <= -self.c2 * df0

        return cond1, cond2

    def search(self, objective, man, x, d, f0, df0, gradient):
        """
        Algorithm 3.5 in [2]
        :param objective: objective function
        :param man: manifold class
        :param x: current variable
        :param d: descent direction
        :param f0: current objective value
        :param df0: value of inner product of grad and descdir at x
        :return:
        """

        p_alpha = torch.zeros(1).to(x)
        prevf = f0
        alpha = self._init_alpha.to(x)
        alpha_star = deepcopy(self._init_alpha).to(x)
        # alpha = 2 * f0 / df0

        r = min(
            [torch.exp(torch.log(self.alpha_max / alpha) / self.max_iter), 2]
        )

        for i in range(self.max_iter):
            newx = man.retr(x, alpha * d)
            newf = objective(newx)

            grad = gradient(newx)
            _d = man.transp(x, newx, d)
            newdf = man.inner(newx, _d, grad)

            cond1, cond2 = self.check_strong_wolf(alpha, f0, df0, newf, newdf)

            if (not cond1) or (newf >= prevf and i > 1):
                alpha_star = self.zoom(
                    objective, man, x, d, f0, df0, p_alpha, alpha, gradient
                )
                break

            if cond2:
                alpha_star = alpha
                break

            if newdf > 0:
                alpha_star = self.zoom(
                    objective, man, x, d, f0, df0, alpha, p_alpha, gradient
                )
                break

            alpha *= r
            prevf = newf

        if alpha_star > self.alpha_max:
            alpha_star = deepcopy(self._init_alpha * 10).to(x)

        alpha_star = max(alpha_star, self._init_alpha.to(x))
        newx = man.retr(x, alpha_star * d)
        newf = objective(newx)
        self._old_alpha = alpha_star

        return newx, newf


class ConjugateGradient(Solver):
    """
    pytorch impl of pymanopt's CG method
    """

    def __init__(
        self, problem, linesearch=None, orth_value=np.inf, *args, **kwargs
    ):
        super(ConjugateGradient, self).__init__(*args, **kwargs)

        if linesearch is None:
            self._linesearch = LineSearch()
        else:
            self._linesearch = linesearch
        self.linesearch = deepcopy(self._linesearch)

        self._cost = None
        self._newcost = None
        self._grad = None
        self._newgrad = None
        self._Pgrad = None
        self._gradPgrad = None
        self._desc_dir = None
        self._gradnorm = None
        self.man = problem.manifold
        self.verbosity = problem.verbosity
        self.objective = problem.cost
        self.gradient = problem.grad
        self.precon = problem.precon
        self._orth_value = orth_value

    def step(self, x):
        """
        :param x:
        :return:
        """
        # Calculate initial cost-related quantities
        if self._desc_dir is None:
            self._desc_dir = -1 * self.gradient(x)
        if self._newcost is not None:
            cost = self._newcost
        else:
            cost = self.objective(x)
        if self._newgrad is not None:
            grad = self._newgrad
        else:
            grad = self.gradient(x)

        desc_dir = self._desc_dir
        inner = self.man.inner

        df0 = inner(x, grad, desc_dir)
        if df0 > -1e-6:
            # Powell's restart strategy
            # page 12 of Hager and Zhang's
            # survey on conjugate gradient methods, for example)
            desc_dir = -1 * grad
            df0 = -inner(x, grad, grad)

        # Line search
        newx, newcost = self.linesearch.search(
            self.objective, self.man, x, desc_dir, cost, df0, self.gradient
        )

        # Compute Day-Yuan-type's beta [1].
        newgrad = self.gradient(newx)
        newgradnorm = self.man.norm(newx, newgrad)
        numo = newgradnorm ** 2
        transposed = self.man.transp(x, newx, desc_dir)

        transnorm = self.man.norm(newx, transposed)
        descnorm = self.man.norm(x, desc_dir)
        if transnorm > descnorm:
            transposed *= descnorm / transnorm
        left = inner(newx, newgrad, transposed)
        beta = min(numo / (left - df0), 20)
        if beta < 0:
            # beta definitely positive [1].
            # However, it could be negative due to computational stability
            beta = 0

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        gradnorm = self.man.norm(x, grad)
        self._desc_dir = -newgrad + beta * transposed
        self._newgrad = newgrad
        self._newcost = newcost
        self._gradnorm = gradnorm

        return newx, newcost, gradnorm

    def solve(self, x, max_iter=10):
        """
        :param x:
        :param max_iter:
        :return:
        """
        for _ in range(max_iter):
            x, _, _ = self.step(x)
        return x
