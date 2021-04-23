import torch


def ortha(basis, metric):
    gram_mat = basis.permute((1, 0)).matmul(metric).matmul(basis)

    if gram_mat.shape[0] < 500:
        eig_val, eig_vec = torch.symeig(gram_mat.cpu(), True)
        eig_val = eig_val.to(basis.device)
        eig_vec = eig_vec.to(basis.device)
    else:
        eig_val, eig_vec = torch.symeig(gram_mat, True)

    eig_val = eig_val.abs().pow(-0.5).diag()

    return basis.matmul(eig_vec).matmul(eig_val)


def ortha_subs(basis_set, metric):
    gram_mat = (
        basis_set.permute((2, 1, 0))
        .matmul(metric)
        .matmul(basis_set.permute((2, 0, 1)))
    )
    if gram_mat.shape[2] < 500:
        eig_val, eig_vec = torch.symeig(gram_mat.cpu(), True)
        eig_vec = eig_vec.to(basis_set.device)
        eig_val = eig_val.to(basis_set.device)

    else:
        eig_val, eig_vec = torch.symeig(gram_mat, True)

    eig_val = eig_val.abs().pow(-0.5)
    eig_val = (
        torch.repeat_interleave(
            torch.eye(eig_val.shape[1], device=eig_val.device).unsqueeze(0),
            eig_val.shape[0],
            0,
        )
        * torch.repeat_interleave(eig_val.unsqueeze(2), eig_val.shape[1], 2)
    )

    orth_basis = basis_set.permute((2, 0, 1)).matmul(eig_vec).matmul(eig_val)
    orth_basis = orth_basis.permute((1, 2, 0)).contiguous()

    return orth_basis


def subspacea(_basis1, _basis2, metric, is_ortha):
    if is_ortha:
        basis1 = ortha(_basis1, metric)
        basis2 = ortha(_basis2, metric)
    else:
        basis1 = _basis1
        basis2 = _basis2

    sim = basis1.permute((1, 0)).matmul(metric).matmul(basis2).pow(2).sum()
    return sim, basis1, basis2


def subspacea_subs(_basis_set1, _basis_set2, metric, is_ortha):
    if is_ortha:
        basis_set1 = ortha_subs(_basis_set1, metric)
        basis_set2 = ortha_subs(_basis_set2, metric)
    else:
        basis_set1 = _basis_set1
        basis_set2 = _basis_set2

    x_size0, x_size1, x_size2 = basis_set1.shape
    y_size0, y_size1, y_size2 = basis_set2.shape

    _basis_set1 = (
        basis_set1.permute((0, 2, 1))
        .reshape((x_size0, x_size1 * x_size2))
        .permute((1, 0))
    )
    _basis_set2 = basis_set2.permute((0, 2, 1)).reshape(
        (y_size0, y_size1 * y_size2)
    )

    if metric is not None:
        gram_mat = _basis_set1.matmul(metric).matmul(_basis_set2)
    else:
        gram_mat = _basis_set1.matmul(_basis_set2)

    gram_mat = gram_mat.pow(2).reshape((x_size2, x_size1, y_size2, y_size1))
    gram_mat = torch.einsum("ijkl->ik", gram_mat)

    return gram_mat, basis_set1, basis_set2


def mutual_subsim(basis_set, metric, is_ortha):
    if is_ortha:
        basis_set = ortha_subs(basis_set, metric)

    x_size0, x_size1, x_size2 = basis_set.shape

    _basis_set = (
        basis_set.permute((0, 2, 1))
        .reshape((x_size0, x_size1 * x_size2))
        .contiguous()
    )
    sim_mat = _basis_set.permute((1, 0)).matmul(metric).matmul(_basis_set)

    sim_mat = sim_mat.pow(2).reshape((x_size2, x_size1, x_size2, x_size1))
    sim_mat = torch.einsum("ijkl->ik", sim_mat)

    return sim_mat, basis_set
