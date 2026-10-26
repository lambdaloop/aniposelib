# 1.0-sync: `aniposelib.sync` (plan 01 F1, F2, F7)

New optional module; nothing existing changes. Import with `from aniposelib import sync` (not imported by
`aniposelib/__init__.py`). numpy/scipy only, except soft-DTW (torch). No offset is ever applied silently: every
estimator returns a result with `identifiable` + `reasons`.

- **Sign convention** (module docstring, `sync.CONVENTION`): `source_time = global_time + offset`; a pair estimate
  `(a, b)` is `offset[b] - offset[a]` (an event at `t` in a appears at `t + offset` in b). Same as disco
  `trim_seconds`; opposite of mouse-outdoor `load_data.align(offsets=)` ("seconds to add"); TC mouse-sync
  `frame_A = frame_B + d` is `offset(a=B, b=A) = d`.
- `align_streams({name: (t, values)}, rate, tol, offsets, reuse=False)` -> `(values [S,T,...], t_grid, src_idx)`:
  nearest sample within `tol`, **no frame reuse by default** (`reuse=True` gives the old `merge_asof` behaviour),
  float seconds or datetimes (datetime64 / pandas / mixed ISO strings; unparseable entries dropped), offsets as
  floats or `TimeMap`s.
- `TimeMap(c, beta, knots)`: `global = c + beta*pts` + piecewise-linear deviation knots (disco `ClipMap`), exact
  inverse, `from_offset`, `to_dict`/`from_dict`.
- `OffsetEstimate` dataclass (offset, sigma, confidence, identifiable, reasons, method, units, pair, diagnostics,
  `to_dict()`).
- `audio_offset(a, b, sr, method='gcc'|'spectrogram'|'onset')` (disco overlap-normalised GCC, legacy multiscale
  spectrogram, TC onset flux), with second-peak / boundary / confidence checks and window-spread sigma;
  `onset_envelope`; `load_audio(path, sr)` via soundfile / PyAV (`[sync]` extra) or an `ffmpeg` executable.
- `motion_signal(points)` + `motion_offset(sig_a, sig_b, rate)` (disco `refine_sync` diagnostics; NaN-aware).
- `solve_offset_graph(edges, anchor)` -> `OffsetGraph` (weighted LS, sigma per stream, cycle RMS, iterative
  rejection of inconsistent edges, unidentifiable edges rejected and listed, unreachable streams -> None).
- `fit_drift(t, offsets)` -> `DriftFit` (robust affine offset + clock rate, `.timemap()`).
- `epipolar_offset(p2d_a, p2d_b, cgroup|F|None)`: fractional lag scan (median Sampson), parabola vertex, block
  bootstrap sigma/CI; cv2-free (`fit_fundamental` robust 8-point when F is not given).
- `frame_offset_sweep(cgroup, sample_fn, offsets)`: integer per-camera sweep against 3D from reference cameras.
- `soft_dtw(D, gamma, band)` (banded anti-diagonal soft-DTW in torch, differentiable; returns value + expected
  alignment; accepts a callable cost for band-only memory) and `soft_dtw_align(sig_a, sig_b, rate, gamma, band)` ->
  `WarpEstimate` (per-row lag + dispersion, robust offset/drift summary, identifiability: dispersion, band-edge hits,
  path-vs-band cost ratio).
- `pyproject.toml`: new optional extra `[sync] = soundfile, av`.
