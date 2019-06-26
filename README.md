# Camibrate

An easy-to-use library for calibrating cameras in python

The calibration library should follow these requirements:
- [ ] be able to detect points from checkerboard or charuco board easily
- [ ] save intermediate points somewhere, so that optimization can restart quickly
- [ ] optimize extrinsics and intrinsics jointly using LM method
- [ ] start with opencv intrinsics calibration if possible
- [ ] support calibration with arbitrary correspondence points using bundle adjustment
- [ ] output common diagnostics for calibration
  - reprojection error
  - length of checkerboard grids
  - picture of calibrated camera positions
  - undistorted images to check intrinsics

Nice to haves
- [ ] reject outlier points

