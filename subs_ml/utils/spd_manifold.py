import abc

import geomstats.backend as gs
from geomstats.geometry import spd_matrices as spd
from geomstats.geometry.general_linear import GeneralLinear
from pymanopt.manifolds.manifold import Manifold


class SPDManifold(spd.SPDMatrices):
    def __init__(self, dim, metric_type="bw"):
        super(SPDManifold, self).__init__(dim)

        if metric_type == "bw":
            self.metric = spd.SPDMetricBuresWasserstein(dim)
        elif metric_type == "euclid":
            self.metric = spd.SPDMetricEuclidean(dim)
        elif metric_type == "logeuclid":
            self.metric = spd.SPDMetricLogEuclidean(dim)
        else:
            self.metric = spd.SPDMetricAffine(dim)


class SPDManopt(Manifold):
    def __init__(self, dim, metric_type="bw", exact_transport=True):
        super(SPDManopt, self).__init__(
            "{}-dimensional SPD manifold".format(dim), dim
        )
        self._manifold = SPDManifold(dim, metric_type)
        self.exact_transport = exact_transport

    def norm(self, base_point, tangent_vector):
        return self._manifold.metric.norm(tangent_vector, base_point=base_point)

    def inner(self, base_point, tangent_vector_a, tangent_vector_b):
        return self._manifold.metric.inner_product(
            tangent_vector_a, tangent_vector_b, base_point=base_point
        )

    def proj(self, base_point, ambient_vector):
        return self._manifold.to_tangent(ambient_vector, base_point=base_point)

    def retr(self, base_point, tangent_vector):
        # FIXME: there may be a bug in the geomstats
        device = tangent_vector.device
        ret = self._manifold.metric.exp(
            tangent_vector.cpu(), base_point=base_point.cpu()
        ).to(device)
        return ret
        # return self._manifold.metric.exp(tangent_vector, base_point=base_point)

    def rand(self):
        return self._manifold.random_point()

    def randvec(self, base_point):
        random_point = gs.random.normal(size=self.dim + 1)
        random_tangent_vector = self.proj(base_point, random_point)
        return random_tangent_vector / gs.linalg.norm(random_tangent_vector)

    def zerovec(self, base_point):
        return gs.zeros_like(self.rand())

    def transp(self, base_point, end_point, tangent):
        """
        transports a tangent vector at a base_point to the tangent space at end_point
        """

        if self.exact_transport:
            # https://github.com/geomstats/geomstats/blob/master/geomstats/geometry/spd_matrices.py#L613
            inverse_base_point = GeneralLinear.inverse(base_point)
            congruence_mat = GeneralLinear.mul(end_point, inverse_base_point)
            congruence_mat = gs.linalg.sqrtm(congruence_mat.cpu()).to(tangent)
            return GeneralLinear.congruent(tangent, congruence_mat)

        # https://github.com/NicolasBoumal/manopt/blob/master/manopt/manifolds/symfixedrank/sympositivedefinitefactory.m#L181
        return tangent

    def egrad2rgrad(self, x, eta):
        # https://github.com/NicolasBoumal/manopt/blob/master/manopt/manifolds/symfixedrank/sympositivedefinitefactory.m#L101

        sym_eta = 0.5 * eta @ eta.T
        return x @ sym_eta @ x
