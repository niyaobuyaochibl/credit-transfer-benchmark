"""Step 1: feedback formats as observation matrices over latent outcome contributions."""
import numpy as np


def _rowspace(m, tol):
    m = np.atleast_2d(np.asarray(m, dtype=float))
    _, s, vt = np.linalg.svd(m, full_matrices=False)
    return vt[: int((s > tol * max(s.max(initial=0.0), 1.0)).sum())]


def common_observation(formats, tol=1e-9):
    """Basis of the outcome information that every format reveals.

    Each format is a matrix F (observed x latent) with observation y = F z. The
    returned rows span the intersection of the row spaces of all F, i.e. the linear
    functions of z that can be computed from the feedback under every format. A
    model that reads only these features receives identical input under all formats.
    Each row is scaled so that its largest-magnitude entry is +1.
    """
    formats = [np.atleast_2d(np.asarray(f, dtype=float)) for f in formats]
    width = formats[0].shape[1]
    assert all(f.shape[1] == width for f in formats), 'formats must share the latent dimension'
    basis = _rowspace(formats[0], tol)
    for f in formats[1:]:
        other = _rowspace(f, tol)
        if not len(basis) or not len(other):
            return np.zeros((0, width))
        stacked = np.vstack([basis, -other]).T
        _, s, vt = np.linalg.svd(stacked)
        rank = int((s > tol * max(s.max(initial=0.0), 1.0)).sum())
        null = vt[rank:]
        if not len(null):
            return np.zeros((0, width))
        basis = _rowspace(null[:, : len(basis)] @ basis, tol)
    out = basis.copy()
    for row in out:
        row /= row[np.argmax(np.abs(row))]
    out[np.abs(out) < tol] = 0.0
    return np.round(out, 12)


def project(observations, fmt, common):
    """Map observations y = F z (last axis) to the shared features C z."""
    fmt = np.atleast_2d(np.asarray(fmt, dtype=float))
    weights = np.asarray(common, dtype=float) @ np.linalg.pinv(fmt)
    assert np.allclose(weights @ fmt, common), 'common features are not computable from this format'
    return np.asarray(observations, dtype=float) @ weights.T
