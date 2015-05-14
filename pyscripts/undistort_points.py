#!/usr/bin/env python
# encoding: utf-8

"""
* This script was borrowed from Dave C Williams *
distortion.py - Dedistort and perspective transform points and images
according to which camera or distortion model is passed.
"""

import cv, cv2
import numpy as np
from matplotlib import pyplot as plt
import sys


#Default distortion and perspective matrices
cam_distortion = np.array([-0.20724,0.02757,0.00007,-0.00041], dtype='float32')
cam_matrix = np.array([ [232.90480,0,        321.92218],
                        [0,        234.09177,225.46547],
                        [0,        0,          1 ] ], dtype='float32')
new_cam_matrix = np.array([np.zeros((3,3),dtype='float32')])
cam_perspective = np.identity(3, dtype='float32') # Was very close to identity so didn't bother with correction

def undistort_points(points2,cam_matrix,distortion_profile):
  global new_cam_matrix
  """Apply a distortion profile to a set of points"""
  # passing points2 generates error: "OpenCV Error: Assertion failed (CV_IS_MAT(_src) && CV_IS_MAT(_dst)..."
  points3=np.array([points2])
  # use new camera matrix to obtain uncropped data
  undistPoints3=cv2.undistortPoints(points3, cam_matrix,distortion_profile,None,new_cam_matrix)
  undistPoints2=undistPoints3[0]
  return undistPoints2

def loadFile(file_name):
  return np.loadtxt(file_name,delimiter = ',')

ax = plt.figure().add_subplot(111)
def plot_data(hdata,mdata):
  ax.scatter(hdata[:,0],hdata[:,1],s=5,c='c',marker="x",label="undist_human")
  ax.scatter(mdata[:,1],mdata[:,2],s=5,c='r',marker="x",label="undist_machine")
  plt.xlabel("x in pels")
  plt.ylabel("y in pels")
  plt.legend(loc='upper right')
  plt.title("human_tracked vs machine_tracked")
  plt.savefig("human_vs_comp.png") #**call before show**#
  plt.show()
  return


def main():
  global new_cam_matrix
  # initialize new camera matrix
  retval=cv2.getOptimalNewCameraMatrix(cam_matrix,cam_distortion,(640,480),1)
  new_cam_matrix=retval[0]
  print("camMatrix: \n"+str(cam_matrix)) #DEBUG#
  print("distCoeffs:\n"+ str(cam_distortion)) #DEBUG#
  print("newCamMatrix: \n"+str(new_cam_matrix)) #DEBUG#

  if sys.argv < 3:
    raise ValueError("Incorrect number of arguments")
  else:
    dist_h_data = loadFile(sys.argv[1]) # col1=x, col2=y
    m_data = loadFile(sys.argv[2]) # col1=frame#, col2=x, col3=y

  undist_h_data=undistort_points(dist_h_data,cam_matrix,cam_distortion)

  print("\ndistData:\n"+str(dist_h_data)) #DEBUG#
  print("undistData:\n"+str(undist_h_data)) #DEBUG#
  #ax.scatter(dist_h_data[:,0],dist_h_data[:,1],s=5,c='y',marker="x",label="dist_human")
  plot_data(undist_h_data,m_data)
  #plot_data(dist_h_data,m_data)

  # save to csv
  np.savetxt("undistorted_data.csv",undist_h_data,delimiter=",")



  return



main()
