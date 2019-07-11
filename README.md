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
- [ ] add objective in optimization to match the object points of calibration board

Functions to implement

Camera
- [X] distort\_points
- [X] undistort\_points
- [X] project
- [X] undistort\_image
- [ ] calibrate\_images
- [ ] calibrate\_points
- [ ] calibrate\_videos

CameraGroup
- [X] distort\_points
- [X] undistort\_points
- [X] project
- [X] triangulate
- [X] reprojection\_error
- [ ] calibrate\_images
- [ ] calibrate\_points
- [ ] calibrate\_videos

CalibrationObject
- [X] draw
- [X] detect\_image
- [X] detect\_images
- [X] detect\_video
- [X] get\_object\_points
- [X] estimate\_pose
