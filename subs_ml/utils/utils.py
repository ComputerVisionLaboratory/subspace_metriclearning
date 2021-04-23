import gc
import os
import pickle

import cv2
import numpy as np
import torch


def gpu_free():
    if torch.cuda.is_available():
        torch.cuda.reset_max_memory_allocated(0)
        torch.cuda.reset_max_memory_cached(0)
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

    gc.collect()


def read_img(img_path, is_color):
    img = cv2.imread(img_path)
    if not is_color:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img.flatten()


def load_cvdata(path, split="_", pos=0, is_color=False, verbose=False):

    cls_dirs = os.listdir(path)
    set_name = ""
    img_sets = []
    label = []

    for cls_ in cls_dirs:
        cls_dir = os.path.join(path, cls_)
        imgs = os.listdir(cls_dir)
        img_set = None
        cnt = 0

        for img_name in imgs:
            img = read_img(os.path.join(cls_dir, img_name), is_color)
            if img_set is None:
                img_set = np.zeros((img.shape[0], len(imgs)))
            _set_name = img_name.split(split)[pos]
            if set_name != _set_name:
                if cnt > 0:
                    img_sets.append(img_set[:, 0:cnt])
                    label.append(cls_)
                    if verbose:
                        print(cls_ + ": " + str(cnt))
                    cnt = 0
                    img_set = np.zeros((img.shape[0], len(imgs)))
            set_name = _set_name
            img_set[:, cnt] = img
            cnt += 1

        if cnt > 0:
            img_sets.append(img_set[:, 0:cnt])
            label.append(cls_)
            if verbose:
                print(cls_ + ": " + str(cnt))

    return img_sets, label


def lp_normalize(features, p_norm):
    return features / features.norm(p_norm, 0)


def pca(_feat_set, dim, p_norm):
    feat_set = _feat_set.clone()

    if p_norm > 0:
        feat_set = lp_normalize(feat_set, p_norm)

    if feat_set.shape[0] < feat_set.shape[1]:
        _feat_set = feat_set.cpu()
        principal_vec, sigma, _ = torch.svd(_feat_set)
        principal_vec = principal_vec[:, 0:dim].to(feat_set.device)
        sigma = sigma[0:dim].to(feat_set.device)

    else:
        gram_mat = feat_set.permute((1, 0)).matmul(feat_set)
        n_samples = feat_set.shape[1]
        if gram_mat.shape[0] < 500:
            gram_mat = gram_mat.cpu()

        eiv_val, sigma = torch.symeig(gram_mat, eigenvectors=True)
        eiv_val = (
            eiv_val.to(feat_set.device)
            .contiguous()[n_samples - dim : n_samples]
            .abs()
        )
        principal_vec = feat_set.matmul(
            sigma.to(feat_set.device).contiguous()[
                :, n_samples - dim : n_samples
            ]
        ).matmul(eiv_val.pow(-0.5).diag())

    return principal_vec, sigma


def pca_for_sets(feat_sets, dim, p_norm, _reddim_mat=None):
    if _reddim_mat is None:
        basis = (
            torch.zeros((feat_sets[0].shape[0], dim, len(feat_sets)))
            .to(feat_sets[0].device)
            .contiguous()
        )
    else:
        basis = (
            torch.zeros((_reddim_mat.shape[0], dim, len(feat_sets)))
            .to(feat_sets[0].device)
            .contiguous()
        )

    for i, _feat_set in enumerate(feat_sets):
        if _reddim_mat is None:
            basis[:, :, i], _ = pca(_feat_set, dim, p_norm)
        else:
            basis[:, :, i], _ = pca(_reddim_mat @ _feat_set, dim, p_norm)

    return basis


def dataset_loader(dataset_name, root_folder=None, save_folder=None):

    if dataset_name == "YTC":
        if root_folder is None:
            root_folder = "B:/YTC/detected_ytc_cv_v4.pkl"

        if save_folder is None:
            save_dir = "B:/YTC/a-based_new_exp/"
        else:
            save_dir = save_folder
        data = pickle.load(open(root_folder, "rb"))
        n_sdims = range(5, 32, 5)
        det_weight = [0.1, 0.3, 0.5, 0.7, 0.9]
        n_cv = 10

    elif dataset_name == "ETH":
        if root_folder is None:
            root_folder = "B:/eth/resized_eth/eth"

        if save_folder is None:
            save_dir = "B:/eth/a-based_new_exp/"
        else:
            save_dir = save_folder

        n_sdims = range(10, 31, 10)
        det_weight = [0.1, 0.3, 0.5, 0.7, 0.9]
        n_cv = 10

        data = {"X_train": [], "X_test": [], "y_train": [], "y_test": []}

        for i in range(10):
            with open(root_folder + str(i + 1) + ".pkl", "rb") as f:
                _data = pickle.load(f)
                data["X_train"].append(_data["X_train"])
                data["X_test"].append(_data["X_test"])
                data["y_train"].append(_data["y_train"])
                data["y_test"].append(_data["y_test"])

    elif dataset_name == "RGBD":
        if root_folder is None:
            root_folder = "Z:/Dataset/rgbd"

        if save_folder is None:
            save_dir = "Z:/Dataset/rgbd/"
        else:
            save_dir = save_folder

        n_sdims = range(5, 21, 5)
        det_weight = [0.1, 0.3, 0.5, 0.7, 0.9]
        n_cv = 5

        with open(os.path.join(root_folder, "rgbd_cv_cnn_feat.pkl"), "rb") as f:
            data = pickle.load(f)

    elif dataset_name == "YTF":
        if root_folder is None:
            root_folder = "Z:/Dataset/YTF/"

        if save_folder is None:
            save_dir = "Z:/Dataset/YTF/"
        else:
            save_dir = save_folder

        # n_sdims = range(1, 41, 3)
        n_sdims = [20, 25, 30, 35]
        det_weight = [0.1, 0.3, 0.5, 0.7, 0.9]
        n_cv = 5

        with open(
            os.path.join(root_folder, "ytf_resnet50_vggface_512dim_223.pkl"),
            "rb",
        ) as f:
            data = pickle.load(f)

    return data, n_sdims, det_weight, n_cv, save_dir, root_folder


def get_min_margin(_model, x):
    _, sim = _model.predict_proba(x, True)
    sim = torch.from_numpy(sim)

    margin = np.inf
    for i in range(sim.shape[0]):
        sw_sim = sim[i, _model.labels == _model.labels[i]]
        sw_sim, _ = sw_sim.sort(descending=True)

        sb_sim = sim[i, _model.labels != _model.labels[i]]
        sb_sim, _ = sb_sim.sort(descending=True)

        _m = sw_sim[1] - sb_sim[0]

        if margin > _m:
            margin = _m
    return margin
