"""Prodigy parity reference (arXiv:2306.06101, Algorithm 4 — Adam variant).

Runs SEVERAL Prodigy steps for a fixed (params, gradient) sequence with zero
initial moment/accumulator state and prints, after each step, both the updated
params AND the scalar distance estimate d. The gradient sequence is chosen so
that the online distance estimate d_t GROWS past its tiny initial value d_0
(a constant positive gradient on a positive starting point — a quadratic bowl
`f(x) = 0.5*||x||^2`-style descent — makes <g, x_0 - x_k> positive and growing,
which drives the numerator r_t up and hence d_t up). The Mojo `prodigy_step`
is compared against these values on identical inputs.

REFERENCE PROVENANCE: the official `prodigyopt` package is NOT installed in the
project interpreter, so this file hand-rolls a NumPy transcription of the
paper's Algorithm 4 (Adam variant) exactly as stated below. NumPy is used only
as a calculator for that algorithm — there is no third-party Prodigy dependency
to diff against. The transcription follows the tracking issue's update rule
(mvillmow/Random#80): the D-Adaptation d-estimate sequence
`d_{t+1} = max(d_t, r_{t+1} / ||s_{t+1}||_1)` with d·lr-weighted accumulators.

Prodigy Adam-variant step (per step k, using CURRENT d_k in the accumulators
and the parameter update; k here is 1-indexed for printing but the math is the
paper's k=0.. recurrence):

    m_{k+1} = beta1 * m_k + (1 - beta1) * d_k * g_k
    v_{k+1} = beta2 * v_k + (1 - beta2) * d_k^2 * g_k^2
    r_{k+1} = sqrt(beta2) * r_k
              + (1 - sqrt(beta2)) * gamma * d_k^2 * <g_k, x_0 - x_k>
    s_{k+1} = sqrt(beta2) * s_k
              + (1 - sqrt(beta2)) * gamma * d_k^2 * g_k
    d_hat   = r_{k+1} / ||s_{k+1}||_1        (0 if ||s||_1 == 0)
    d_{k+1} = max(d_k, d_hat)
    x_{k+1} = x_k - gamma * d_k * m_{k+1} / (sqrt(v_{k+1}) + d_k * eps)

where gamma is the base step-size schedule (default 1.0 — Prodigy has "no
learning rate to tune"; gamma stays fixed here), x_0 is the initial parameter
vector, r/d are SCALARS, and s/m/v/params are per-coordinate.
"""

import json

import numpy as np

BETA1, BETA2, EPS, GAMMA = 0.9, 0.999, 1e-8, 1.0
D0 = 1e-6
N_STEPS = 5

# Positive start, constant positive gradient: <g, x0 - x_k> grows as x_k
# descends, so r_t (hence d_t) increases monotonically past D0.
x0 = np.array([1.0, 2.0, 3.0, 4.0])
grad = np.array([0.5, 0.5, 0.5, 0.5])

sqrt_b2 = np.sqrt(BETA2)


def run():
    x = x0.copy()
    m = np.zeros_like(x0)
    v = np.zeros_like(x0)
    s = np.zeros_like(x0)
    r = 0.0
    d = D0

    params_per_step = []
    d_per_step = []
    for _ in range(N_STEPS):
        g = grad  # constant gradient sequence
        d_cur = d
        m = BETA1 * m + (1.0 - BETA1) * d_cur * g
        v = BETA2 * v + (1.0 - BETA2) * (d_cur**2) * (g * g)
        inner = float(np.dot(g, x0 - x))
        r = sqrt_b2 * r + (1.0 - sqrt_b2) * GAMMA * (d_cur**2) * inner
        s = sqrt_b2 * s + (1.0 - sqrt_b2) * GAMMA * (d_cur**2) * g
        s_l1 = float(np.sum(np.abs(s)))
        d_hat = r / s_l1 if s_l1 != 0.0 else 0.0
        d = max(d_cur, d_hat)
        denom = np.sqrt(v) + d_cur * EPS
        x = x - GAMMA * d_cur * m / denom

        params_per_step.append(x.tolist())
        d_per_step.append(d)

    return params_per_step, d_per_step


def main():
    params_per_step, d_per_step = run()
    print(
        json.dumps(
            {
                "inputs": {
                    "x0": x0.tolist(),
                    "grad": grad.tolist(),
                    "beta1": BETA1,
                    "beta2": BETA2,
                    "eps": EPS,
                    "gamma": GAMMA,
                    "d0": D0,
                    "n_steps": N_STEPS,
                },
                "params_per_step": params_per_step,
                "d_per_step": d_per_step,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
