"""KAN (Kolmogorov-Arnold Network) layer parity reference.

Computes the forward pass of a single KAN layer in float64 numpy with FIXED
ramp parameters and FIXED inputs, and prints the flattened output as JSON so the
Mojo KAN test can transcribe the same constants and assert equality to 1e-5.

KAN layer (Liu et al. 2024, arXiv:2404.19756, §2.2 Eq. 2.10), residual
activation on each edge (i -> j):

    phi_{j,i}(x) = w_base_{j,i} * silu(x) + w_spline_{j,i} * spline_{j,i}(x)
    spline_{j,i}(x) = sum_m c_{j,i,m} * B_{m,k}(x)
    y_j = sum_i phi_{j,i}(x_i)

with silu(x) = x * sigmoid(x). The B-spline basis B_{m,k}(x) is computed by the
Cox-de Boor recursion below, TRANSCRIBED IDENTICALLY into the Mojo layer
(`_bspline_basis`) so parity is exact-by-construction rather than reproduced from
an external spline library.

Config: in_features=4, out_features=4, grid_size=5, spline_order=3,
grid range [-1, 1]. Two input rows: one in-range point and one BELOW the grid
minimum (x < -1) so the out-of-range compact-support behavior is exercised
(spline branch -> 0, base branch only). Parameters are deterministic ramps.
"""

import json

import numpy as np

IN_F, OUT_F = 4, 4
GRID_SIZE, SPLINE_ORDER = 5, 3
GRID_MIN, GRID_MAX = -1.0, 1.0
N_COEFF = GRID_SIZE + SPLINE_ORDER  # 8


def knot(idx: int) -> float:
    """Open-uniform knot t_idx (matches Mojo KAN._knot)."""
    h = (GRID_MAX - GRID_MIN) / GRID_SIZE
    return GRID_MIN + (idx - SPLINE_ORDER) * h


def bspline_basis(x: float) -> np.ndarray:
    """Cox-de Boor order-k B-spline basis; returns array of length N_COEFF.

    Identical recursion to the Mojo layer: degree-0 indicator on knot spans,
    then raise degree 1..k. Compact support => all zeros for x outside the grid.
    """
    n_knots = GRID_SIZE + 2 * SPLINE_ORDER + 1
    b = np.zeros(n_knots - 1, dtype=np.float64)
    for m in range(n_knots - 1):
        lo, hi = knot(m), knot(m + 1)
        b[m] = 1.0 if (x >= lo and x < hi) else 0.0
    for p in range(1, SPLINE_ORDER + 1):
        nb = np.zeros(n_knots - 1 - p, dtype=np.float64)
        for m in range(n_knots - 1 - p):
            tm, tmp = knot(m), knot(m + p)
            tm1, tmp1 = knot(m + 1), knot(m + p + 1)
            left = 0.0
            den_l = tmp - tm
            if den_l != 0.0:
                left = (x - tm) / den_l * b[m]
            right = 0.0
            den_r = tmp1 - tm1
            if den_r != 0.0:
                right = (tmp1 - x) / den_r * b[m + 1]
            nb[m] = left + right
        b = nb
    return b[:N_COEFF]


def silu(x: float) -> float:
    return x / (1.0 + np.exp(-x))


# Deterministic ramp parameters (float64), flat layouts matching Mojo:
#   base_weight, spline_weight: [in, out] row-major (i*out + j)
#   spline_coeff: [in, out, n_coeff] row-major ((i*out + j)*n_coeff + m)
base_w = (np.arange(IN_F * OUT_F, dtype=np.float64) * 0.01 - 0.05).reshape(IN_F, OUT_F)
spline_w = (np.arange(IN_F * OUT_F, dtype=np.float64) * 0.02 - 0.10).reshape(IN_F, OUT_F)
coeff = (np.arange(IN_F * OUT_F * N_COEFF, dtype=np.float64) * 0.003 - 0.05).reshape(IN_F, OUT_F, N_COEFF)

# Two input rows: row 0 in-range, row 1 has a component below grid_min (-1.3).
X = np.array(
    [
        [-0.7, -0.2, 0.35, 0.9],
        [-1.3, 0.0, 0.5, 1.0],
    ],
    dtype=np.float64,
)

B = X.shape[0]
out = np.zeros((B, OUT_F), dtype=np.float64)
for bi in range(B):
    for j in range(OUT_F):
        acc = 0.0
        for i in range(IN_F):
            x = X[bi, i]
            basis = bspline_basis(x)
            spline = float(np.dot(coeff[i, j], basis))
            acc += base_w[i, j] * silu(x) + spline_w[i, j] * spline
        out[bi, j] = acc

print(
    json.dumps(
        {
            "config": {
                "in_features": IN_F,
                "out_features": OUT_F,
                "grid_size": GRID_SIZE,
                "spline_order": SPLINE_ORDER,
                "grid_min": GRID_MIN,
                "grid_max": GRID_MAX,
                "batch": B,
            },
            "base_weight": base_w.flatten().tolist(),
            "spline_weight": spline_w.flatten().tolist(),
            "spline_coeff": coeff.flatten().tolist(),
            "X": X.flatten().tolist(),
            "out": out.flatten().tolist(),
        },
        indent=2,
    )
)
