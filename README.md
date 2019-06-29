# Camibrate

An easy-to-use library for calibrating cameras in python

The calibration library should follow these requirements:
- [x] be able to detect points from checkerboard or charuco board easily
- [ ] save intermediate points somewhere, so that optimization can restart quickly
- [ ] optimize extrinsics and intrinsics jointly using LM method
- [ ] start with opencv intrinsics calibration if possible
- [ ] support calibration with arbitrary correspondence points using bundle adjustment
- [ ] output common diagnostics for calibration
  - reprojection error
  - length of checkerboard grids
  - picture of calibrated camera positions
  - undistorted images to check intrinsics
- [x] triangulation functions
- [ ] end to end function to calibrate full set of cameras from list of videos

Nice to haves
- [ ] reject outlier points


Functions to implement

Camera
.distort\_points
.undistort\_points
.project
.undistort\_image
.calibrate\_images
.calibrate\_points
.calibrate\_videos
.extract\_points\_images
.extract\_points\_videos

CameraGroup
.distort\_points
.undistort\_points
.project
.triangulate
.reprojection\_error
.calibrate\_images
.calibrate\_points
.calibrate\_videos

CalibrationObject
.draw
.detect\_image
.detect\_video
.get\_object\_points
.estimate\_pose
