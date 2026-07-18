"""ScheduleFree+ parity reference (large-batch-stable schedule-free).

Runs SEVERAL ScheduleFree+ steps for a fixed (params, grad, objective)
sequence with the documented initial state and prints the resulting state after
each step, so the Mojo `schedule_free_plus_step` can be compared on identical
inputs. The step counter drives BOTH the increasing outer-momentum anneal AND
the L1-EMA bias behaviour, so multiple steps with DIFFERENT gradients are needed
to exercise every sequence.

This is a hand-rolled NumPy transcription of the ScheduleFree+ update rule
EXACTLY as written in the tracking issue body (mvillmow/Random#78), which wins
over any base-paper ambiguity. NumPy is used only as a calculator for that
transcribed algorithm; there is no external optimizer to defer to for this
variant. The base Schedule-Free method is Defazio et al. 2024
("The Road Less Scheduled", arXiv:2405.15682); the ScheduleFree+ stability
variant is Defazio 2026 ("ScheduleFree+: Scaling Learning-Rate-Free &
Schedule-Free Learning to Large Language Models", arXiv:2605.19095). The three
mechanisms transcribed below are the issue's rule:

    (1) Inner momentum: a momentum buffer m accumulates the gradient and the
        fast sequence z steps on the momentum buffer (momentum INSIDE the
        base-seq z-update), NOT on the raw gradient.

        m_{t+1} = mu * m_t + (1 - mu) * g_t
        z_{t+1} = z_t - lr_t * m_{t+1}

    (2) Polyak step-size: the per-step lr_t is derived from the (caller-supplied)
        objective value f(y_t), the correlation between the gradient and
        (z_t - x_t), and an L1 EMA of the gradient magnitude:

        s_t     = beta_sf * gamma_ema_L1_{t}                 # rho * EMA
        gnorm_t = rho * gnorm_{t-1} + (1 - rho) * mean(|g_t|)  # L1 EMA of grad
        polyak  = max(0, f(y_t) + beta_sf * <g_t, z_t - x_t>) / (gnorm_t + eps)
        lr_t    = learning_rate * polyak                     # scaled Polyak lr

    (3) Increasing outer momentum: the interpolation/averaging weight c anneals
        the outer momentum beta_out from beta_sf up to beta_max across the run:

        frac      = min(1, (t - 1) / max(1, horizon - 1))
        beta_out  = beta_sf + (beta_max - beta_sf) * frac
        c_{t+1}   = (1 - beta_out)                            # averaging weight
        x_{t+1}   = beta_out * x_t + (1 - beta_out) * z_{t+1}

    Query point (where the caller evaluates the next gradient/objective):
        y_{t+1}   = (1 - beta_out) * z_{t+1} + beta_out * x_{t+1}

On the first step the state is initialized z_1 = x_1 = params, m_1 = 0,
gnorm_0 = 0. The caller supplies f(y_t) (the globally-reduced objective at the
current query point), per the issue's "Polyak step needs the globally-reduced
objective f(y_t)".
"""

import json

import numpy as np

LR = 0.1
MU = 0.9  # inner momentum coefficient
BETA_SF = 0.9  # base outer momentum / Polyak correlation weight
BETA_MAX = 0.98  # annealed-to outer momentum
RHO = 0.9  # L1-EMA decay for the gradient-magnitude normaliser
EPS = 1e-8
HORIZON = 4  # anneal horizon (steps) for the increasing outer momentum

params = np.array([0.1, -0.2, 0.3, -0.4, 0.5])
# One gradient + objective per step; different each step so every sequence moves.
grads = [
    np.array([0.05, 0.15, -0.25, 0.35, -0.45]),
    np.array([0.08, 0.05, -0.20, 0.40, -0.30]),
    np.array([-0.02, 0.20, -0.10, 0.25, -0.35]),
]
objectives = [1.2, 0.9, 0.7]  # f(y_t) supplied by the caller each step


def sfplus_step(params, g, f_y, z, x, m, gnorm, t):
    # (1) Inner momentum buffer, then fast z-sequence steps on it.
    m = MU * m + (1.0 - MU) * g

    # (2) Polyak step-size from the objective, correlation, and L1-EMA norm.
    gnorm = RHO * gnorm + (1.0 - RHO) * float(np.mean(np.abs(g)))
    corr = float(np.dot(g, z - x))
    polyak = max(0.0, f_y + BETA_SF * corr) / (gnorm + EPS)
    lr_t = LR * polyak
    z = z - lr_t * m

    # (3) Increasing outer momentum anneal beta_sf -> beta_max.
    frac = min(1.0, (t - 1) / max(1, HORIZON - 1))
    beta_out = BETA_SF + (BETA_MAX - BETA_SF) * frac
    x = beta_out * x + (1.0 - beta_out) * z

    # Next query point.
    y = (1.0 - beta_out) * z + beta_out * x
    return params, z, x, m, gnorm, y, lr_t, beta_out


# Initial state.
z = params.copy()
x = params.copy()
m = np.zeros_like(params)
gnorm = 0.0

step_outs = []
for i, (g, f_y) in enumerate(zip(grads, objectives), start=1):
    params, z, x, m, gnorm, y, lr_t, beta_out = sfplus_step(params, g, f_y, z, x, m, gnorm, i)
    step_outs.append(
        {
            "z": z.tolist(),
            "x": x.tolist(),
            "m": m.tolist(),
            "gnorm": gnorm,
            "y": y.tolist(),
            "lr_t": lr_t,
            "beta_out": beta_out,
        }
    )

print(
    json.dumps(
        {
            "inputs": {
                "params": params.tolist(),
                "grads": [g.tolist() for g in grads],
                "objectives": objectives,
                "lr": LR,
                "mu": MU,
                "beta_sf": BETA_SF,
                "beta_max": BETA_MAX,
                "rho": RHO,
                "eps": EPS,
                "horizon": HORIZON,
            },
            "steps": step_outs,
        },
        indent=2,
    )
)
