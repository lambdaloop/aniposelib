- `bundle_adjust(method='adam')` / `bundle_adjust_iter(method='adam')` are now scale-invariant. Translation-like
  parameters (camera tvecs, 3D points, board translations) get learning rates scaled by
  `rig_scale / ADAM_REF_SCALE` (rig_scale = median camera-to-point distance, `ADAM_REF_SCALE = 1.0`), and their
  gradients are clipped in normalised units. Before, the absolute step sizes stalled on mm-scale or tiny rigs and
  drifted or diverged on others: camera-centre error after BA ranged 0.09–3.5% of rig scale depending only on
  units; it is now identical at 1e-3, 1 and 1e3. Old behaviour: `trans_scale=None`. A fixed scale:
  `trans_scale=<float>`.
