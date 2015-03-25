#!/usr/bin/python

import numpy as np
import matplotlib.pyplot as plt

DIST_THRESH = 20;
TIME_THRESH = 0;
BORDER      = 50;

traj    = np.loadtxt('x_y_data_human_tracked_july_16th_2014_moth2_v2.csv',delimiter = ',')
fData   = np.loadtxt('flower_xy_coordinates_july_16th_2014_moth2.csv',delimiter = ',')
gndTruth= np.loadtxt('FlowerNumbers_StartFrames_EndFrames_july_16th_2014_moth2.csv',delimiter = ',')

# get moth pos
mx = traj[:,0]
my = traj[:,1]
################################################
## remove last flower for this data set ##
fData = np.delete(fData,36,0)
################################################
    #### INFO ####
print "done reading data"

# plot moth and flower pos
ax = plt.figure().add_subplot(111)
plt.gca().set_xlim((0,max(mx)+BORDER))
plt.gca().set_ylim((0,max(my)+BORDER))
ax.scatter(mx,my,s=5,c='c',marker="x",label="trajectory")
ax.scatter(fData[:,0],fData[:,1],s=5,c='g',marker="o",label="flowers")

# find points in traj which are D within each flower
p_per_fData = [0]*len(fData)
for i in range(0,len(fData)):
  # get ith flower
  # find m coords within D of flower center
  fx = fData[i,0]
  fy = fData[i,1]
  p_per_fData[i] = [p for p in traj if( (p[0]-fx)**2+(p[1]-fy)**2 <= DIST_THRESH**2 )]

  # plot points and circles around each flower
  ax.scatter([p[0] for p in p_per_fData[i]],[p[1] for p in p_per_fData[i]],s=5,c='r',marker="x")
  c = plt.Circle((fx,fy),DIST_THRESH,color='r',fill=False)
  plt.gcf().gca().add_artist(c)




    #### INFO ####
k=0
print "flower\tcount"
for f in p_per_fData:
  print "%d\t%d" %(k,len(f))
  k=k+1
    #### INFO ####

    #### INFO ####
print "finished!"

plt.title('Moth Trajectory Within D of Flowers')
plt.xlabel('x in pels')
plt.ylabel('y in pels')
plt.legend(loc='upper right',prop={'size':10})
#plt.savefig('visits.png') #**call before show**#
plt.show()



