"""Fitted item-parameter container + the graded-response score/likelihood/information
primitives shared by selection and reconstruction. Pure numpy (no jax) so the released
tool has a light runtime. Loads the .npz artifacts written by run_ea1_ladder.save_fit.
"""
from dataclasses import dataclass
import numpy as np


def _sig(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


@dataclass
class ItemParams:
    a: np.ndarray          # (J,) discrimination
    b: np.ndarray          # (J,) binary-item threshold (used where keff==2)
    k0: np.ndarray         # (J,) ternary lower cutpoint (used where keff==3)
    k1: np.ndarray         # (J,) ternary upper cutpoint
    u: np.ndarray          # (C,) clip random effect (posterior mean)
    keff: np.ndarray       # (J,) 2 or 3
    family: np.ndarray     # (J,) family index
    clip: np.ndarray       # (J,) clip index per item
    family_names: list
    item_dim: np.ndarray = None   # (J,) ability-dimension per item (M3 only)
    n_dim: int = 1
    link: str = "grm"             # "grm" (Samejima) or "pcm" (partial credit) for ternary items
    prior_sd: float = 1.0         # ability prior sd used by the two-stage MAP/EAP

    @classmethod
    def from_fit(cls, derived, fd, item_dim=None):
        """Build from an in-memory SVI fit's `derived` dict (posterior means over draws)."""
        m = lambda k: np.asarray(derived[k]).mean(0)
        idim = None if item_dim is None else np.asarray(item_dim)
        ndim = int(idim.max()) + 1 if idim is not None else 1
        return cls(a=m("a"), b=m("b"), k0=m("k0"), k1=m("k1"), u=m("u"),
                   keff=np.asarray(fd.item_keff), family=np.asarray(fd.item_family),
                   clip=np.asarray(fd.item_clip), family_names=list(fd.family_names),
                   item_dim=idim, n_dim=ndim)

    @classmethod
    def from_npz(cls, path):
        d = np.load(path, allow_pickle=True)
        post = {k: (d[k].mean(0) if d[k].ndim and k in ("a", "b", "k0", "k1", "u") else d[k])
                for k in d.files}
        a = d["a"].mean(0) if d["a"].ndim == 2 else d["a"]
        b = d["b"].mean(0) if d["b"].ndim == 2 else d["b"]
        k0 = d["k0"].mean(0) if d["k0"].ndim == 2 else d["k0"]
        k1 = d["k1"].mean(0) if d["k1"].ndim == 2 else d["k1"]
        u = d["u"].mean(0) if d["u"].ndim == 2 else d["u"]
        idim = d["item_dim"] if "item_dim" in d.files else None
        ndim = int(idim.max()) + 1 if idim is not None else 1
        return cls(a=np.asarray(a), b=np.asarray(b), k0=np.asarray(k0), k1=np.asarray(k1),
                   u=np.asarray(u), keff=np.asarray(d["item_keff"]),
                   family=np.asarray(d["item_family"]), clip=np.asarray(d["item_clip"]),
                   family_names=list(d["family_names"]),
                   item_dim=(np.asarray(idim) if idim is not None else None), n_dim=ndim)


# --------------------------------------------------------------------------- category probabilities
def category_probs(p, theta_vec):
    """P(Y=k | theta) for every item, given a per-item ability `theta_vec` (J,).

    theta_vec already includes dimension selection for M3 (theta of the item's dim).
    Returns (P_wrong, P_partial, P_correct) each (J,). Binary items put mass on
    {wrong=0, correct=2}; partial=0.
    """
    eta = p.a * theta_vec + p.u[p.clip]
    tern = p.keff == 3
    # binary: correct prob = sigmoid(eta - b)
    pc = _sig(eta - p.b)
    P0 = 1.0 - pc                        # wrong
    P1 = np.zeros_like(pc)               # partial (binary: none)
    P2 = pc                              # correct
    # ternary override
    if getattr(p, "link", "grm") == "pcm":
        T0, T1, T2 = _pcm_probs(eta, p.k0, p.k1)
    else:
        s0 = _sig(eta - p.k0)
        s1 = _sig(eta - p.k1)
        T0, T1, T2 = 1.0 - s0, s0 - s1, s1
    P0 = np.where(tern, T0, P0)
    P1 = np.where(tern, T1, P1)
    P2 = np.where(tern, T2, P2)
    eps = 1e-9
    return np.clip(P0, eps, 1), np.clip(P1, eps, 1), np.clip(P2, eps, 1)


def _pcm_probs(eta, k0, k1):
    """Partial-credit (adjacent-category logit) probabilities with steps k0, k1."""
    l1 = np.clip(eta - k0, -30, 30); l2 = np.clip(2 * eta - k0 - k1, -60, 60)
    m = np.maximum(0.0, np.maximum(l1, l2))
    e0, e1, e2 = np.exp(-m), np.exp(l1 - m), np.exp(l2 - m)
    z = e0 + e1 + e2
    return e0 / z, e1 / z, e2 / z


def expected_score(p, theta_vec):
    """Per-item expected benchmark score in [0,1] (wrong=0, partial=0.5, correct=1)."""
    P0, P1, P2 = category_probs(p, theta_vec)
    return 0.0 * P0 + 0.5 * P1 + 1.0 * P2


def item_information(p, theta_vec):
    """Per-item Fisher information about the item's ability dimension at `theta_vec`."""
    eta = p.a * theta_vec + p.u[p.clip]
    tern = p.keff == 3
    # binary: I = a^2 p(1-p)
    pc = _sig(eta - p.b)
    info_bin = p.a ** 2 * pc * (1 - pc)
    # ternary GRM: I = sum_k (dP_k/dθ)^2 / P_k
    if getattr(p, "link", "grm") == "pcm":
        P0, P1, P2 = _pcm_probs(eta, p.k0, p.k1)
        Ek = P1 + 2 * P2
        # dP_k/dtheta = a P_k (k - E[k])  =>  I = a^2 Var(k)
        info_tern = p.a ** 2 * (P1 * (1 - Ek) ** 2 + P2 * (2 - Ek) ** 2 + P0 * Ek ** 2)
        return np.where(tern, info_tern, info_bin)
    s0 = _sig(eta - p.k0); s1 = _sig(eta - p.k1)
    d0 = -p.a * s0 * (1 - s0)
    d2 = p.a * s1 * (1 - s1)
    d1 = p.a * (s0 * (1 - s0) - s1 * (1 - s1))
    P0 = np.clip(1 - s0, 1e-9, 1); P1 = np.clip(s0 - s1, 1e-9, 1); P2 = np.clip(s1, 1e-9, 1)
    info_tern = d0 ** 2 / P0 + d1 ** 2 / P1 + d2 ** 2 / P2
    return np.where(tern, info_tern, info_bin)


def theta_for_items(p, theta_by_dim):
    """Map a (respondent's) per-dimension ability vector to per-item ability (J,).

    theta_by_dim: scalar (1D) or (n_dim,) array. Returns per-item theta using item_dim.
    """
    if p.item_dim is None or np.ndim(theta_by_dim) == 0:
        return np.full(len(p.a), float(theta_by_dim))
    return np.asarray(theta_by_dim)[p.item_dim]
