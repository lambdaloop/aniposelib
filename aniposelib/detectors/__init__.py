"""Board corner detectors usable as ``CharucoBoard(detector=...)``.

A detector is any object with ``detect(image) -> (corners (N, 1, 2) float32, ids (N, 1) int32)``
(image: BGR or gray array); ``detect_batch(images)`` (a list of those) and ``batch_size`` are
optional and enable batched detection in ``CharucoBoard.detect_frames`` /
``detect_video(frame_indices=...)``.

- ``'opencv'`` (default): built into :class:`aniposelib.boards.CharucoBoard`
  (``cv2.aruco.ArucoDetector`` + ``CharucoDetector``).
- :class:`DeepCharucoDetector` (``'deepcharuco'``): learned detector, imported lazily.
"""

__all__ = ['BoardDetector', 'DeepCharucoDetector']


class BoardDetector:
    """Optional base class for custom detectors."""
    batch_size = 1

    def detect(self, image):
        raise NotImplementedError

    def detect_batch(self, images):
        return [self.detect(im) for im in images]


def __getattr__(name):
    if name == 'DeepCharucoDetector':
        from .deepcharuco import DeepCharucoDetector
        return DeepCharucoDetector
    raise AttributeError(name)
