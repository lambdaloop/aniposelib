"""One entry point per workflow (plan 05). See ``docs/workflows.md``.

=========================  =====================================  ========================
workflow                   function                               result
=========================  =====================================  ========================
W1 board / object calib.   :func:`calibrate_from_board`           ``BoardCalibration``
W2 calib. from points      :func:`calibrate_from_points`          ``PointCalibration``
W3 metric scale / world    :func:`estimate_scale`                 ``ScaleEstimate``
W4 QA / reuse / moves      :func:`evaluate_calibration`           ``CalibrationEvaluation``
W5 syncing cameras         :func:`estimate_offsets`               ``OffsetEstimation``
W6 2D -> 3D                :func:`reconstruct_3d`                 ``Reconstruction``
W7 multi-animal assoc.     :func:`associate_views`                ``Association``
=========================  =====================================  ========================

Every result has ``.report``, ``.verdict`` ('ok' | 'warn' | 'fail'), ``.reasons``,
``.settings``, ``.provenance``, ``.to_dict()``, ``.to_json(path)`` and ``.summary()``.
Workflows raise only on unusable inputs; a bad result is a ``'fail'`` verdict with reasons.
"""

from ._common import WorkflowResult, Rule, noise_floor  # noqa: F401

_EXPORTS = {
    'calibrate_from_board': 'calibration', 'BoardCalibration': 'calibration',
    'calibrate_from_points': 'calibration', 'PointCalibration': 'calibration',
    'estimate_scale': 'calibration', 'ScaleEstimate': 'calibration',
    'evaluate_calibration': 'calibration', 'CalibrationEvaluation': 'calibration',
    'estimate_offsets': 'reconstruction', 'OffsetEstimation': 'reconstruction',
    'reconstruct_3d': 'reconstruction', 'Reconstruction': 'reconstruction',
    'associate_views': 'reconstruction', 'Association': 'reconstruction',
}
WORKFLOWS = ['calibrate_from_board', 'calibrate_from_points', 'estimate_scale',
             'evaluate_calibration', 'estimate_offsets', 'reconstruct_3d', 'associate_views']

__all__ = list(_EXPORTS) + ['WorkflowResult', 'Rule', 'noise_floor', 'WORKFLOWS']


def __getattr__(name):
    if name in _EXPORTS:
        import importlib
        mod = importlib.import_module('.' + _EXPORTS[name], __name__)
        val = getattr(mod, name)
        globals()[name] = val
        return val
    raise AttributeError('module {!r} has no attribute {!r}'.format(__name__, name))
