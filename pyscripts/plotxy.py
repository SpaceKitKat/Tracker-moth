#!/usr/bin/python

import sys
import numpy as np
import matplotlib.pyplot as plt

def main():
  if(len(sys.argv) < 4):
    raise ValueError("Usage: ./plot.py human_data_file comp_data_file output_plot.png")
    return
  # verify .txt or .csv and .png

  ax = plt.figure().add_subplot(111)

  hum_data = np.loadtxt(sys.argv[1],delimiter = ',')
  comp_data = np.loadtxt(sys.argv[2],delimiter = ',')
  hcols = len(hum_data[0])
  ccols = len(comp_data[0])

  ax.scatter(hum_data[:,hcols-2],hum_data[:,hcols-1],s=5,c='c',marker="x",label="human")
  ax.scatter(comp_data[:,ccols-2],comp_data[:,ccols-1],s=5,c='b',marker=".",label="auto")
  plt.title('Human vs. Computer Generated Moth Trajectory')
  plt.xlabel('x in pels')
  plt.ylabel('y in pels')
  plt.legend(loc='upper right')
  plt.savefig(sys.argv[3]) #**call before show**#
  return

main()
