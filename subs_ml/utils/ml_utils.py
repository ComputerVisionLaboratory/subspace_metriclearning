import torch

from .subspace_funcs import mutual_subsim, subspacea_subs


# @torch.jit.script
def _projection_mat(sub_basis, metric):
    _p_mat = sub_basis.matmul(metric)
    _p_mat = _p_mat.matmul(sub_basis.transpose(1, 2))
    _p_mat = torch.inverse(_p_mat)
    _p_mat = sub_basis.transpose(1, 2).matmul(_p_mat)
    _p_mat = _p_mat.matmul(sub_basis)
    return _p_mat


def _calc_grad(_p_mat, met_p, p_met, index, identity):
    diff1 = identity - met_p
    grad = (p_met @ _p_mat[index, :, :].sum(1) @ diff1).sum(0)

    for i in range(index.shape[1]):
        _subs = _p_mat[index[:, i], :, :]
        diff2 = diff1[index[:, i], :, :]
        grad += (_subs @ met_p @ diff2).sum(0)

    return grad / (met_p.shape[0] * index.shape[1])


def lmm_egrad(k_mat, sub_basis, metric, sb_idx, sw_idx, n_sb, n_sw):

    sw_kmat = k_mat.clone()
    sb_kmat = sw_kmat.clone()
    n_samples = k_mat.shape[0]

    sw_kmat[~sw_idx] = 0
    _, sw_kmat = torch.sort(sw_kmat, axis=1)
    idw = sw_kmat[:, n_samples - n_sw : n_samples - 1]

    sb_kmat[~sb_idx] = 0
    _, sb_kmat = torch.sort(sb_kmat, axis=1)
    idb = sb_kmat[:, n_samples - n_sb : n_samples]

    identity_mat = torch.eye(sub_basis.shape[2]).to(sub_basis.device)
    _p_mat = _projection_mat(sub_basis, metric)
    met_p = metric.matmul(_p_mat)
    p_met = met_p.permute((0, 2, 1))

    grad_sb = _calc_grad(_p_mat, met_p, p_met, idb, identity_mat)
    grad_sw = _calc_grad(_p_mat, met_p, p_met, idw, identity_mat)

    met_norm = torch.norm(metric)
    penalty = (
        (1 - met_norm) / (torch.relu(met_norm - 1e-2) + 1e-2) * 2e-2 * metric
    )

    return grad_sb, grad_sw, idb, idw, penalty


def lmm_cost(
    sub_basis,
    metric_mat,
    sb_idx,
    sw_idx,
    n_sb,
    n_sw,
    alpha,
    n_sdim,
    compute_each_loss=False,
):
    k_mat, _sub_basis = mutual_subsim(sub_basis, metric_mat, True)
    n_samples = k_mat.shape[0]
    sw_kmat = k_mat.clone()
    sb_kmat = k_mat.clone()

    sw_kmat[~sw_idx] = 0
    sw_kmat, idw = torch.sort(sw_kmat, axis=1)
    sw_kmat = sw_kmat[:, n_samples - n_sw : n_samples - 1]
    sw_cost = alpha * (n_sdim - sw_kmat).mean()

    sb_kmat[~sb_idx] = 0
    sb_kmat, idb = torch.sort(sb_kmat, axis=1)
    sb_kmat = sb_kmat[:, n_samples - n_sb : n_samples]
    sb_cost = (1 - alpha) * torch.mean(sb_kmat)

    if not compute_each_loss:
        return sb_cost + sw_cost, k_mat

    idw = idw[:, n_samples - n_sw : n_samples - 1]
    idb = idb[:, n_samples - n_sb : n_samples]
    return sb_cost, sw_cost, idb, idw, _sub_basis, k_mat


def _partial_det_grad(index, grad, basis, metric_mat):
    if index is not None:
        gram_mat = (
            basis.unsqueeze(1) @ metric_mat @ basis[index, :, :].transpose(2, 3)
        )
        inv_gram_mat = torch.inverse(gram_mat.cpu()).to(gram_mat.device)
        grad_sw = (
            basis[index, :, :].transpose(2, 3)
            @ inv_gram_mat
            @ basis.unsqueeze(1)
        )
        grad_sw = grad_sw + grad_sw.transpose(2, 3)
        grad_sw = grad_sw.sum(0).sum(0)
        grad_sw -= (grad.sum(0) * index.shape[1]) + grad[index, :, :].sum(
            0
        ).sum(0)
        grad_sw = grad_sw / (grad_sw.shape[0] * grad_sw.shape[1])
    else:
        grad_sw = None

    return grad_sw


def logdet_egrad(basis, metric_mat, idb, idw):
    inv_gram_mat = torch.inverse(
        (basis @ metric_mat @ basis.transpose(1, 2)).cpu()
    ).to(basis.device)
    grad = basis.transpose(1, 2) @ inv_gram_mat @ basis

    grad_sw = _partial_det_grad(idw, grad, basis, metric_mat)
    grad_sb = _partial_det_grad(idb, grad, basis, metric_mat)
    return grad_sb, grad_sw


def logdet_cost(_sub_basis, metric_mat, idb, idw, alpha):
    dim, sub_dim, n_subs = _sub_basis.shape

    basis_vecs = (
        _sub_basis.permute(0, 2, 1).reshape(dim, sub_dim * n_subs).contiguous()
    )
    gram_mat = basis_vecs.permute(1, 0).matmul(metric_mat).matmul(basis_vecs)
    gram_mat = gram_mat.reshape(n_subs, sub_dim, n_subs, sub_dim)
    gram_mat = gram_mat.permute(0, 2, 1, 3).contiguous()
    gram_mat = gram_mat.det().abs().log()
    gram_mat[torch.isinf(gram_mat)] = 0

    idx = [[i] for i in range(gram_mat.shape[0])]
    logdet_err_sw = alpha * gram_mat[idx, idw].mean() / 20
    logdet_err_sb = (1 - alpha) * gram_mat[idx, idb].mean() / 20
    return logdet_err_sb, logdet_err_sw


def lmm_multi_metric_cost(
    sub_basis,
    metric_mats,
    anchors,
    sb_idx,
    sw_idx,
    n_sb,
    n_sw,
    alpha,
    n_sdim,
    compute_each_loss=False,
):
    k_mat = torch.zeros((sub_basis.shape[2], sub_basis.shape[2]))
    _sub_basis = []
    for _, metric_mat in zip(metric_mats, anchors):
        _k, _basis = mutual_subsim(sub_basis, metric_mat, True)
        _w, _ = subspacea_subs(sub_basis, anchors, None, False)

        k_mat += _k
        _sub_basis.append(_basis)

    n_samples = k_mat.shape[0]
    sw_kmat = k_mat.clone()
    sb_kmat = k_mat.clone()

    sw_kmat[~sw_idx] = 0
    sw_kmat, idw = torch.sort(sw_kmat, axis=1)
    sw_kmat = sw_kmat[:, n_samples - n_sw : n_samples - 1]
    sw_cost = alpha * (n_sdim - sw_kmat).mean()

    sb_kmat[~sb_idx] = 0
    sb_kmat, idb = torch.sort(sb_kmat, axis=1)
    sb_kmat = sb_kmat[:, n_samples - n_sb : n_samples]
    sb_cost = (1 - alpha) * torch.mean(sb_kmat)

    if not compute_each_loss:
        return sb_cost + sw_cost, k_mat

    idw = idw[:, n_samples - n_sw : n_samples - 1]
    idb = idb[:, n_samples - n_sb : n_samples]
    return sb_cost, sw_cost, idb, idw, _sub_basis, k_mat
