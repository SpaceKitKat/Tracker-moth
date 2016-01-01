#!/usr/bin/env python
# encoding: utf-8

"""
* This script was borrowed from Dave C Williams *
undistort_points.py - Dedistort and perspective transform points
according to camera and distortion model. Points must be passed
via txt or csv file where the last two columns are the x and y
coordinates. A csv file is generated containing the dedistorted
set of points.
"""

import sys
import cv2
import numpy as np

#Default distortion and perspective matrices
cam_distortion = np.array([-0.20724,0.02757,0.00007,-0.00041], dtype='float32')
cam_matrix = np.array([ [232.90480,0,        321.92218],
                        [0,        234.09177,225.46547],
                        [0,        0,          1 ] ], dtype='float32')
new_cam_matrix = np.array([np.zeros((3,3),dtype='float32')])
cam_perspective = np.identity(3, dtype='float32') # Was very close to identity so didn't bother with correction

# Args: Nx2 array of distorted points, 3x3 camera matrix, distortion coefficents
# Returns: Nx2 array of dedistorted points
# performs transformation to distorted points t
def undistort_points(points2):
  """Apply a distortion profile to a set of points"""
  # passing points2 generates error: "OpenCV Error: Assertion failed (CV_IS_MAT(_src) && CV_IS_MAT(_dst)..."
  points3=np.array([points2])
  # use new camera matrix to obtain uncropped data
  undistPoints3=cv2.undistortPoints(points3, cam_matrix,cam_distortion,None,new_cam_matrix)
  undistPoints2=undistPoints3[0]
  return undistPoints2

def main():
  global new_cam_matrix

  if len(sys.argv) < 3:
    raise ValueError("Usage: undistort_points.py distorted_data_file output_data_file")
    return

  # check file type and return if not txt or csv

  # initialize new camera matrix
  retval=cv2.getOptimalNewCameraMatrix(cam_matrix,cam_distortion,(640,480),1)
  new_cam_matrix=retval[0]

  print("-- Dedistortion Parameters --")
  print("camMatrix: \n"+str(cam_matrix)) #DEBUG#
  print("distCoeffs:\n"+ str(cam_distortion)) #DEBUG#
  print("newCamMatrix: \n"+str(new_cam_matrix)) #DEBUG#
  print("-----------------------------")
  # load xy coordinates
  raw_data = np.loadtxt(sys.argv[1],delimiter = ',') # ncol-2 = x, ncol-1 = y
  ncols = len(raw_data[0])
  dist_xy = raw_data[:,ncols-2:]
  # copy raw data and replace xy cols with dedistorted xy cols
  undist_data = raw_data
  undist_data[:,ncols-2:] = undistort_points(dist_xy)

  print("Saving dedistorted points to file, \""+sys.argv[2]+"\"")
  # save to csv
  np.savetxt(sys.argv[2],undist_data,delimiter=",")
  print("Done.")

  return

main()
