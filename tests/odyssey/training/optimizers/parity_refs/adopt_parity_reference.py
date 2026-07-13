"""ADOPT parity reference: run a fixed (params, grad, state) through the public
pytorch_optimizer.ADOPT for ONE step, print the resulting params + state so the
Mojo adopt_step can be compared against identical inputs.

We disable the clip (clip_lambda -> huge) and weight_decay to isolate the core
update, and set betas/eps to the Mojo defaults. Second moment is initialized to
grad^2 (v_0 = g_0^2) to match the well-scaled-first-step recommendation, which
is exactly what pytorch_optimizer.ADOPT does internally on the first step (it
sets exp_avg_sq = grad^2 and returns without updating params on step 1). To
compare a *general* step we run TWO steps: step 1 seeds the state, step 2 is the
one we compare (both optimizers start step 2 from the same seeded state).
"""
import torch, json, sys
from pytorch_optimizer import ADOPT

torch.manual_seed(0)
N = 6
# fixed inputs (deterministic, no randomness) so Mojo can replay them exactly
params0 = torch.tensor([0.10, -0.20, 0.30, -0.40, 0.50, -0.60], dtype=torch.float64)
grad_a  = torch.tensor([0.05, 0.15, -0.25, 0.35, -0.45, 0.55], dtype=torch.float64)  # step-1 grad
grad_b  = torch.tensor([-0.02, 0.08, 0.12, -0.18, 0.22, -0.28], dtype=torch.float64)  # step-2 grad

LR, B1, B2, EPS = 0.01, 0.9, 0.9999, 1e-6
NO_CLIP = lambda step: 1.0e30  # effectively disable clipping

p = params0.clone().requires_grad_(True)
opt = ADOPT([p], lr=LR, betas=(B1, B2), eps=EPS, weight_decay=0.0,
            clip_lambda=NO_CLIP)

# --- step 1: seeds exp_avg_sq = grad_a^2, exp_avg stays 0, params unchanged ---
p.grad = grad_a.clone()
opt.step()
st = opt.state[p]
after1 = dict(params=p.detach().tolist(),
             m=st['exp_avg'].tolist(),
             v=st['exp_avg_sq'].tolist())

# --- step 2: the step we compare. State entering step 2 = after1's m,v ---
m_in = st['exp_avg'].clone().tolist()
v_in = st['exp_avg_sq'].clone().tolist()
p.grad = grad_b.clone()
opt.step()
st2 = opt.state[p]
after2 = dict(params=p.detach().tolist(),
             m=st2['exp_avg'].tolist(),
             v=st2['exp_avg_sq'].tolist())

out = dict(
    inputs_step2=dict(
        params=after1['params'],   # params entering step 2 (= after step 1)
        grad=grad_b.tolist(),
        m=m_in, v=v_in,
        lr=LR, beta1=B1, beta2=B2, eps=EPS, clip=1.0e30, wd=0.0,
    ),
    reference_step2_out=after2,
)
print(json.dumps(out, indent=2))
