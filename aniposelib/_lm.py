"""The Levenberg-Marquardt engine shared by the solvers (1.1, plan 04 §3).

One damped Gauss-Newton loop with a Schur complement serves bundle adjustment
(:func:`aniposelib.bundle.bundle_adjust_lm`), spatiotemporal BA (:mod:`aniposelib.stba`), the
SfM refinement (:func:`aniposelib.init.sfm.refine_extrinsics`) and the board LM of the lens
ladder (:func:`aniposelib.intrinsics.lm_board_adjust`). A solver only describes its problem:
residuals and Jacobians, how a step is applied, and its priors.

API
---
``robust(e2, loss, f) -> (rho, weight)``
    Robust cost ``rho(e^2)`` (squared-residual units) and the IRLS weight ``rho'(e^2)`` for
    ``loss in LOSSES = ('linear', 'huber', 'cauchy', 'soft_l1')`` with scale ``f``.
``Blocks(r, w, Js, sidx, n, D, Jc=None, cols=None, scols=None, Hp=None, gp=None)``
    The linearisation of one set of point-like blocks (3D points: ``D = 3``, rigid boards:
    ``D = 6``, spline segments: ``D = 3 K``). One row per observation: residual ``r [m,k]``,
    IRLS weight ``w [m]`` (0 = unused), block Jacobian ``Js [m,k,d]`` on block ``sidx [m]``
    (columns ``scols [m,d]`` inside the block, default all ``D``), camera-like Jacobian
    ``Jc [m,k,q]`` on the reduced parameters ``cols [m,q]`` (``NC`` = unused slot). ``Hp``
    (``[D,D]`` or ``[n,D,D]``) and ``gp [n,D]`` add a quadratic block prior.
``System(blocks, NC, H=None, g=None)``
    Normal equations of a linearisation (``H, g``: camera-like priors, ``[NC,NC]`` and ``[NC]``).
    ``.step(lam, ...)`` -> ``(d_theta, [d_values per block set])`` or None (singular);
    ``.reduced(...)`` -> the Schur-reduced ``S, b`` (and per-block gradients, for sandwich
    covariances).
``levenberg_marquardt(problem, state, n_iter, ...) -> LMResult``
    The loop. ``problem`` provides ``cost(state)``, ``linearize(state) -> list[Blocks]``,
    ``NC``, optionally ``prior(state) -> (H, g)``, and ``trial(state, d_theta, d_values) ->
    (new_state, new_cost)`` (None = infeasible step) and ``after_step(state, cost)``.
    Marquardt damping, steps accepted on the true (robust) cost.
``block_lm(problem, values, mu, iters, ...)``
    Block-only LM (no camera-like parameters) with per-block damping ``mu`` and acceptance:
    ``problem.block_linearize(values) -> list[Blocks]``, ``problem.block_costs(values) ->
    list[[n]]``.

Damping
-------
Default (``damp_on='camera'``): blocks ``V + lam diag(V) + 1e-9 (1 + max|diag V|) I``, camera
system ``H + lam diag(H)`` before the reduction. ``damp_on='reduced'`` damps the reduced
system instead (``S + lam diag(S)``); ``damp_blocks(V, lam)`` overrides the block rule. Large
reduced systems are solved matrix free (PCG with a Jacobi preconditioner).
"""

from dataclasses import dataclass, field
from typing import Any, List, Optional

import numpy as np
import torch

__all__ = ['LOSSES', 'robust', 'Blocks', 'System', 'LMResult', 'levenberg_marquardt', 'block_lm',
           'default_damp_blocks']

LOSSES = ('linear', 'huber', 'cauchy', 'soft_l1')


def robust(e2, loss, f):
    """Robust cost rho(e^2) in squared-residual units, and IRLS weight rho'(e^2)."""
    if loss == 'linear':
        return e2, torch.ones_like(e2)
    if loss == 'cauchy':
        s = f * f
        return s * torch.log1p(e2 / s), 1.0 / (1.0 + e2 / s)
    if loss == 'huber':
        e = torch.sqrt(e2)
        rho = torch.where(e <= f, e2, 2 * f * e - f * f)
        w = torch.where(e <= f, torch.ones_like(e), f / e.clamp(min=1e-12))
        return rho, w
    if loss == 'soft_l1':
        s = f * f
        z = torch.sqrt(1 + e2 / s)
        return 2 * s * (z - 1), 1.0 / z
    raise ValueError('unknown loss {!r}; use one of {}'.format(loss, LOSSES))


@dataclass
class Blocks:
    """Linearisation of one set of point-like blocks (see the module docstring)."""
    r: Any
    w: Any
    Js: Any
    sidx: Any
    n: int
    D: int
    Jc: Any = None
    cols: Any = None
    scols: Any = None
    Hp: Any = None
    gp: Any = None


def default_damp_blocks(V, lam):
    D = V.shape[-1]
    eye = torch.eye(D, dtype=V.dtype, device=V.device)
    dg = V.diagonal(dim1=-2, dim2=-1)
    return V + lam * dg[..., None] * eye + 1e-9 * eye * (1 + dg.abs().amax(-1)[:, None, None])


def _block_solver(Vd):
    """A solve ``x -> Vd^-1 x`` for batched blocks (explicit inverse for small blocks)."""
    D = Vd.shape[-1]
    if D <= 6:
        Vinv = torch.linalg.inv(Vd)
        return lambda x: Vinv @ x
    L, info = torch.linalg.cholesky_ex(Vd)
    if bool((info > 0).any()):
        bad = info > 0
        V2 = Vd.clone()
        eye = torch.eye(D, dtype=Vd.dtype, device=Vd.device)
        V2[bad] = V2[bad] + 1e-3 * eye * Vd[bad].diagonal(dim1=-2, dim2=-1).mean(-1)[:, None, None]
        L, info = torch.linalg.cholesky_ex(V2)
    return lambda x: torch.cholesky_solve(x, L)


class _Lin:
    """Accumulated normal-equation pieces of one :class:`Blocks`."""

    def __init__(self, b, NC):
        self.b = b
        dt, dev = b.Js.dtype, b.Js.device
        n, D = b.n, b.D
        d = b.Js.shape[-1]
        wJs = b.w[:, None, None] * b.Js
        Vo = torch.einsum('mki,mkj->mij', wJs, b.Js)
        go = torch.einsum('mki,mk->mi', wJs, b.r)
        if b.scols is None:
            self.V = torch.zeros(n, D, D, dtype=dt, device=dev).index_add_(0, b.sidx, Vo)
            self.gs = torch.zeros(n, D, dtype=dt, device=dev).index_add_(0, b.sidx, go)
        else:
            self.V = torch.zeros(n, D, D, dtype=dt, device=dev)
            flat = (b.sidx * D * D)[:, None, None] + b.scols[:, :, None] * D + b.scols[:, None, :]
            self.V.view(-1).index_add_(0, flat.reshape(-1), Vo.reshape(-1))
            self.gs = torch.zeros(n, D, dtype=dt, device=dev)
            self.gs.view(-1).index_add_(0, ((b.sidx * D)[:, None] + b.scols).reshape(-1), go.reshape(-1))
        if b.Hp is not None:
            self.V = self.V + b.Hp
        if b.gp is not None:
            self.gs = self.gs + b.gp
        self.Hcs = None
        if NC and b.Jc is not None and b.Jc.shape[-1]:
            wJc = b.w[:, None, None] * b.Jc
            self.Uo = torch.einsum('mki,mkj->mij', wJc, b.Jc)
            self.gco = torch.einsum('mki,mk->mi', wJc, b.r)
            self.Hcs = torch.einsum('mki,mkj->mij', wJc, b.Js)               # [m,q,d]
        self.flat_s = None if b.scols is None else (b.sidx * D)[:, None] + b.scols

    # y [n,D] = W v  (W: [n,D,NC], from the per-observation Hcs)
    def W_apply(self, vpad):
        b = self.b
        y = torch.einsum('mqd,mq->md', self.Hcs, vpad[b.cols])
        out = torch.zeros(b.n, b.D, dtype=y.dtype, device=y.device)
        if self.flat_s is None:
            return out.index_add_(0, b.sidx, y)
        out.view(-1).index_add_(0, self.flat_s.reshape(-1), y.reshape(-1))
        return out

    # W^T z -> [NC+1]
    def WT_apply(self, z, NC):
        b = self.b
        zo = z[b.sidx] if self.flat_s is None else z.reshape(-1)[self.flat_s]
        v = torch.einsum('mqd,md->mq', self.Hcs, zo)
        return torch.zeros(NC + 1, dtype=z.dtype, device=z.device).index_add_(0, b.cols.reshape(-1), v.reshape(-1))

    def W_dense(self, s0, s1, NC):
        """W of blocks [s0, s1) as a dense [s1-s0, D, NC+1] tensor."""
        b = self.b
        m = (b.sidx >= s0) & (b.sidx < s1)
        W = torch.zeros(s1 - s0, b.D, NC + 1, dtype=self.Hcs.dtype, device=self.Hcs.device)
        if not bool(m.any()):
            return W
        Hm = self.Hcs[m]                                                         # [m',q,d]
        q, d = Hm.shape[1], Hm.shape[2]
        si = (b.sidx[m] - s0)[:, None, None].expand(-1, q, d)
        sc = (torch.arange(d, device=Hm.device)[None, None, :].expand(si.shape[0], q, d) if b.scols is None
              else b.scols[m][:, None, :].expand(-1, q, d))
        cc = b.cols[m][:, :, None].expand(-1, q, d)
        W.index_put_((si, sc, cc), Hm, accumulate=True)
        return W


class System:
    """Normal equations of a linearisation (see the module docstring)."""

    def __init__(self, blocks, NC, H=None, g=None, damp_blocks=None):
        self.NC = NC
        self.lins = [_Lin(b, NC) for b in blocks]
        dt = blocks[0].Js.dtype if blocks else torch.float64
        dev = blocks[0].Js.device if blocks else None
        self.dtype, self.device = dt, dev
        U = torch.zeros(NC + 1, NC + 1, dtype=dt, device=dev)
        gc = torch.zeros(NC + 1, dtype=dt, device=dev)
        for L in self.lins:
            if L.Hcs is None:
                continue
            c = L.b.cols
            U.view(-1).index_add_(0, (c[:, :, None] * (NC + 1) + c[:, None, :]).reshape(-1), L.Uo.reshape(-1))
            gc.index_add_(0, c.reshape(-1), L.gco.reshape(-1))
        U, gc = U[:NC, :NC], gc[:NC]
        if H is not None:
            U = U + H
        if g is not None:
            gc = gc + g
        self.U, self.g = U, gc
        self.damp_blocks = damp_blocks or default_damp_blocks

    def _factor(self, lam):
        return [_block_solver(self.damp_blocks(L.V, lam)) for L in self.lins]

    def _rhs(self, solves):
        NC = self.NC
        b = self.g.clone()
        ys = []
        for L, sv in zip(self.lins, solves):
            y = sv(L.gs[..., None])[..., 0]
            ys.append(y)
            if L.Hcs is not None and NC:
                b = b - L.WT_apply(y, NC)[:NC]
        return -b, ys

    def _schur_dense(self, solves, max_elems):
        NC = self.NC
        Hs = torch.zeros_like(self.U)
        for L, sv in zip(self.lins, solves):
            if L.Hcs is None:
                continue
            step = max(1, int(max_elems // max(1, L.b.D * (NC + 1))))
            for s0 in range(0, L.b.n, step):
                s1 = min(L.b.n, s0 + step)
                W = L.W_dense(s0, s1, NC)[..., :NC]
                Y = sv_slice(sv, L, s0, s1, W)
                Hs = Hs - torch.einsum('nda,ndb->ab', W, Y)
        return Hs

    def step(self, lam, solver='auto', damp_on='camera', max_elems=1 << 24):
        """Damped step ``(d_theta, [d_values])``, or None if the system could not be solved."""
        NC = self.NC
        solves = self._factor(lam)
        if NC == 0:
            dth = torch.zeros(0, dtype=self.dtype, device=self.device)
        else:
            if solver == 'auto':
                solver = 'dense' if NC <= 1500 else 'pcg'
            b, _ = self._rhs(solves)
            if damp_on == 'camera':
                dU = self.U.diagonal().clone()
                dU = torch.where(dU > 0, dU, torch.full_like(dU, float(dU.max()) * 1e-12 + 1e-12))
            if solver == 'dense':
                Hs = self._schur_dense(solves, max_elems)
                if damp_on == 'camera':
                    Hd = self.U + Hs + torch.diag(lam * dU)
                else:
                    Sm = self.U + Hs
                    Hd = Sm + lam * torch.diag(torch.diagonal(Sm).clamp(min=1e-12))
                try:
                    Lc = torch.linalg.cholesky(Hd)
                    dth = torch.cholesky_solve(b[:, None], Lc)[:, 0]
                except Exception:
                    try:
                        dth = torch.linalg.solve(Hd, b)
                    except Exception:
                        return None
            else:
                if damp_on != 'camera':
                    raise ValueError("solver='pcg' needs damp_on='camera'")
                dth = _pcg(self, solves, self.U + torch.diag(lam * dU), b)
            if not bool(torch.isfinite(dth).all()):
                return None
        dpad = torch.cat([dth, dth.new_zeros(1)])
        dvals = []
        for L, sv in zip(self.lins, solves):
            rhs = L.gs
            if L.Hcs is not None and NC:
                rhs = rhs + L.W_apply(dpad)
            dvals.append(-sv(rhs[..., None])[..., 0])
        return dth, dvals

    def reduced(self, lam=0.0, per_block=False, max_elems=1 << 24):
        """Schur-reduced system ``(S, b)`` (no camera damping; blocks damped with ``lam``).
        With ``per_block``, also the reduced gradient of each block ``[n, NC]`` per block set
        (observation terms only; their sum is ``b`` minus the prior gradient)."""
        solves = self._factor(lam)
        Sm = self.U + self._schur_dense(solves, max_elems)
        b, ys = self._rhs(solves)
        if not per_block:
            return Sm, -b
        NC = self.NC
        out = []
        for L, sv, y in zip(self.lins, solves, ys):
            gq = torch.zeros(L.b.n, NC + 1, dtype=self.dtype, device=self.device)
            if L.Hcs is not None:
                bb = L.b
                gq.view(-1).index_add_(0, ((bb.sidx * (NC + 1))[:, None] + bb.cols).reshape(-1), L.gco.reshape(-1))
                yo = y[bb.sidx] if L.flat_s is None else y.reshape(-1)[L.flat_s]
                v = torch.einsum('mqd,md->mq', L.Hcs, yo)
                gq.view(-1).index_add_(0, ((bb.sidx * (NC + 1))[:, None] + bb.cols).reshape(-1), -v.reshape(-1))
            out.append(gq[:, :NC])
        return Sm, -b, out


def sv_slice(sv, L, s0, s1, W):
    """Apply the block solve of blocks [s0, s1) to ``W [s1-s0, D, NC]``."""
    if s0 == 0 and s1 == L.b.n:
        return sv(W)
    full = torch.zeros(L.b.n, *W.shape[1:], dtype=W.dtype, device=W.device)
    full[s0:s1] = W
    return sv(full)[s0:s1]


def _pcg(sysm, solves, Hc, b, max_iter=500, rtol=1e-10):
    """Matrix-free preconditioned CG on the reduced camera system."""
    NC = sysm.NC

    def matvec(v):
        vpad = torch.cat([v, v.new_zeros(1)])
        acc = torch.zeros(NC + 1, dtype=v.dtype, device=v.device)
        for L, sv in zip(sysm.lins, solves):
            if L.Hcs is None:
                continue
            z = sv(L.W_apply(vpad)[..., None])[..., 0]
            acc = acc + L.WT_apply(z, NC)
        return Hc @ v - acc[:NC]

    Minv = 1.0 / Hc.diagonal().clamp(min=1e-300)
    x = torch.zeros_like(b)
    r = b.clone()
    z = Minv * r
    p = z.clone()
    rz = r @ z
    bn = float(torch.linalg.norm(b)) + 1e-300
    for _ in range(max_iter):
        Ap = matvec(p)
        alpha = rz / (p @ Ap)
        x = x + alpha * p
        r = r - alpha * Ap
        if float(torch.linalg.norm(r)) < rtol * bn:
            break
        z = Minv * r
        rz_new = r @ z
        p = z + (rz_new / rz) * p
        rz = rz_new
    return x


@dataclass
class LMResult:
    """Output of :func:`levenberg_marquardt`: final ``state`` and ``cost``, per-iteration
    ``history`` (``iter, cost, lam, accepted, step_max``), ``converged`` and the stop ``reason``."""
    state: Any
    cost: float
    history: List[dict] = field(default_factory=list)
    converged: bool = False
    reason: str = ''


def levenberg_marquardt(problem, state, n_iter=25, lam=1e-3, tol=1e-7, solver='auto', damp_on='camera',
                        damp_blocks=None, lam_up=5.0, lam_down=3.0, lam_fail=10.0, lam_min=1e-9,
                        lam_max=1e10, max_tries=None, rel_floor=1e-300, step_tol=None, cost=None,
                        max_elems=1 << 24, verbose=False, label='LM'):
    """Marquardt-damped Gauss-Newton with a Schur complement (see the module docstring).

    A step is accepted if the true cost decreases (then ``lam /= lam_down``, floored at
    ``lam_min``), else ``lam *= lam_up`` and the step is recomputed, up to ``lam_max`` or
    ``max_tries`` attempts; a singular system multiplies ``lam`` by ``lam_fail``. Stops when no
    step is accepted, when the relative decrease ``(c - c') / max(|c|, rel_floor)`` is below
    ``tol``, or (``step_tol``) after 3 accepted steps with ``max |d_theta| < step_tol``."""
    NC = problem.NC
    cost = float(problem.cost(state)) if cost is None else cost
    history = [dict(iter=0, cost=cost, lam=lam, accepted=True)]
    prior = getattr(problem, 'prior', None)
    trial = getattr(problem, 'trial', None)
    after = getattr(problem, 'after_step', None)
    converged, reason, n_small = False, 'max_iter', 0
    for it in range(n_iter):
        H, g = prior(state) if prior is not None else (None, None)
        sysm = System(problem.linearize(state), NC, H, g, damp_blocks=damp_blocks)
        improved, tries, rel, smax = False, 0, 0.0, 0.0
        while lam <= lam_max and (max_tries is None or tries < max_tries):
            tries += 1
            st = sysm.step(lam, solver=solver, damp_on=damp_on, max_elems=max_elems)
            if st is None:
                lam *= lam_fail
                continue
            out = trial(state, st[0], st[1]) if trial is not None else problem.apply(state, st[0], st[1])
            if out is None:
                lam *= lam_up
                continue
            if trial is None:
                out = (out, float(problem.cost(out)))
            new_state, new_cost = out
            if np.isfinite(new_cost) and new_cost < cost:
                rel = (cost - new_cost) / max(abs(cost), rel_floor)
                smax = float(st[0].abs().max()) if st[0].numel() else 0.0
                state, cost = new_state, new_cost
                lam = max(lam / lam_down, lam_min)
                improved = True
                break
            lam *= lam_up
        history.append(dict(iter=it + 1, cost=cost, lam=lam, accepted=improved, step_max=smax))
        if verbose:
            print('{} iter {:3d}: cost {:.6e}  lambda {:.1e}{}'.format(
                label, it + 1, cost, lam, '' if improved else ' (no step)'), flush=True)
        if not improved:
            converged, reason = True, 'no_decrease'
            break
        if rel < tol:
            converged, reason = True, 'tol'
            break
        if step_tol is not None:
            n_small = n_small + 1 if smax < step_tol else 0
            if n_small >= 3:
                converged, reason = True, 'step_tol'
                break
        if after is not None:
            state, cost = after(state, cost)
    return LMResult(state=state, cost=cost, history=history, converged=converged, reason=reason)


def block_lm(problem, values, mu, iters=8, mu_c=0.0, mu_down=3.0, mu_up=5.0, mu_min=1e-7, mu_max=1e6,
             tol=1e-9, const=0.0):
    """Block-only LM with per-block damping and acceptance (see the module docstring).

    ``values``: list of ``[n_k, ...]`` tensors (one per block set; steps are reshaped to them);
    ``mu``: list of per-block damping ``[n_k]`` (updated in place). The damped block is
    ``V (1 + mu) + mu_c`` on the diagonal. ``const`` is added to the total (fixed terms). Stops when
    the total changes by less than ``tol * max(1, |total|)``. Returns ``(values, total)``."""
    def damp_for(mk):
        def f(V, lam):
            out = V.clone()
            dg = out.diagonal(dim1=-2, dim2=-1)
            dg.mul_(1 + mk[:, None]).add_(mu_c)
            return out
        return f

    c_old = [c.clone() for c in problem.block_costs(values)]
    prev = float(sum(c.sum() for c in c_old)) + const
    for _ in range(iters):
        blocks = problem.block_linearize(values)
        new = []
        for b, v, mk in zip(blocks, values, mu):
            L = _Lin(b, 0)
            dx = -_block_solver(damp_for(mk)(L.V, 0.0))(L.gs[..., None])[..., 0]
            new.append(v + dx.reshape(v.shape))
        c_new = problem.block_costs(new)
        for k in range(len(values)):
            acc = (c_new[k] <= c_old[k]) & torch.isfinite(c_new[k])
            shape = (-1,) + (1,) * (values[k].dim() - 1)
            values[k] = torch.where(acc.reshape(shape), new[k], values[k])
            c_old[k] = torch.where(acc, c_new[k], c_old[k])
            mu[k].copy_(torch.where(acc, (mu[k] / mu_down).clamp(min=mu_min), (mu[k] * mu_up).clamp(max=mu_max)))
        cur = float(sum(c.sum() for c in c_old)) + const
        if abs(prev - cur) < tol * max(1.0, abs(prev)):
            prev = cur
            break
        prev = cur
    return values, prev
