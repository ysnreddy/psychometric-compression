"""Assemble jnp model-argument tuples from FieldData/ClipData — one place so the
ladder runner and LOMO/selection harnesses never hand-order args (bug-prone)."""
import numpy as np
import jax.numpy as jnp

# Confirmatory family -> ability-dimension maps for M3 (simple structure).
FAM_DIM = {
    2: {"action": 0, "scene_type": 0, "ppe": 1, "count": 1, "safety": 1},              # recognition | detail+safety
    3: {"action": 0, "scene_type": 0, "ppe": 1, "count": 1, "safety": 2},              # recog | perception-detail | safety
    4: {"action": 0, "scene_type": 0, "ppe": 1, "count": 2, "safety": 3},              # recog | ppe | counting | safety
}


def _j(x):
    return jnp.asarray(x)


def field_arrays(fd):
    """Return a dict of jnp arrays + scalars shared by M1/M2/M3."""
    return dict(
        R=_j(fd.resp), I=_j(fd.item), C=_j(fd.clip), Y=_j(fd.y), KE=_j(fd.keff),
        IF=_j(fd.item_family), RG=_j(fd.resp_group),
        idx_bin=_j(np.where(fd.binary_mask)[0]),
        idx_tern=_j(np.where(~fd.binary_mask)[0]),
        n_resp=fd.n_resp, n_item=fd.n_item, n_clip=fd.n_clip,
        n_family=fd.n_family, n_group=fd.n_group,
    )


def item_dim_array(fd, D):
    m = FAM_DIM[D]
    return _j(np.array([m[fd.family_names[f]] for f in fd.item_family], dtype=np.int64))


def m1_args(A):
    return (A["R"], A["I"], A["Y"], A["KE"], A["n_resp"], A["n_item"], A["n_family"],
            A["n_group"], A["IF"], A["RG"])


def m2_args(A):
    return (A["R"], A["I"], A["C"], A["Y"], A["idx_bin"], A["idx_tern"], A["n_resp"],
            A["n_item"], A["n_clip"], A["n_family"], A["n_group"], A["IF"], A["RG"])


def m3_args(A, item_dim, D):
    return (A["R"], A["I"], A["C"], A["Y"], A["idx_bin"], A["idx_tern"], item_dim,
            A["n_resp"], A["n_item"], A["n_clip"], A["n_family"], A["n_group"], D,
            A["IF"], A["RG"])


def clip_args(cd):
    return (_j(cd.resp), _j(cd.clip), _j(cd.y), cd.n_resp, cd.n_clip, cd.n_group,
            _j(cd.resp_group))


def rotation_field_arrays(fd, exclude_resp_idx):
    """Like field_arrays but with observations from `exclude_resp_idx` respondents dropped,
    keeping FULL n_item/n_clip/n_resp so fitted item params stay aligned across rotations
    (leakage-safe LOMO — exclude a base model AND its pseudo-respondents)."""
    exclude = np.asarray(sorted(exclude_resp_idx))
    keep = ~np.isin(fd.resp, exclude)
    bm = fd.binary_mask[keep]
    return dict(
        R=_j(fd.resp[keep]), I=_j(fd.item[keep]), C=_j(fd.clip[keep]),
        Y=_j(fd.y[keep]), KE=_j(fd.keff[keep]),
        IF=_j(fd.item_family), RG=_j(fd.resp_group),
        idx_bin=_j(np.where(bm)[0]), idx_tern=_j(np.where(~bm)[0]),
        n_resp=fd.n_resp, n_item=fd.n_item, n_clip=fd.n_clip,
        n_family=fd.n_family, n_group=fd.n_group,
    )


# Secondary (cross) loading dimension for M3x: safety / PPE / count items may also load on
# recognition; action / scene items have no secondary dimension (-1).
XDIM_3 = {"action": -1, "scene_type": -1, "ppe": 0, "count": 0, "safety": 0}


def item_xdim_array(fd):
    return _j(np.array([XDIM_3[fd.family_names[f]] for f in fd.item_family], dtype=np.int64))


def m3x_args(A, item_dim, item_xdim, D):
    return (A["R"], A["I"], A["C"], A["Y"], A["idx_bin"], A["idx_tern"], item_dim, item_xdim,
            A["n_resp"], A["n_item"], A["n_clip"], A["n_family"], A["n_group"], D,
            A["IF"], A["RG"])
