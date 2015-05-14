#!/usr/bin/python

import numpy as np
import matplotlib.pyplot as plt


ax = plt.figure().add_subplot(111)
hum_data = np.loadtxt('h_test.csv',delimiter = ',')
comp_data = np.loadtxt('undist_test16.txt',delimiter = ',')

ax.scatter(hum_data[:,0],hum_data[:,1],s=5,c='c',marker="x",label="human")
ax.scatter(comp_data[:,1],comp_data[:,2],s=5,c='b',marker=".",label="auto")
plt.title('Human vs. Computer Generated Moth Trajectory')
plt.xlabel('x in pels')
plt.ylabel('y in pels')
plt.legend(loc='upper right')
plt.savefig('human_vs_comp.png') #**call before show**#
plt.show()
# new figure is created

# old plotxy.py
#-ax = plt.figure().add_subplot(111)
#-hum_data = np.loadtxt('test.csv',delimiter = ',')
#-comp_data = np.loadtxt('test.txt',delimiter = ',')
#-
#-ax.scatter(hum_data[:,0],hum_data[:,1],s=5,c='c',marker="x",label="human")
#-ax.scatter(comp_data[:,1],comp_data[:,2],s=5,c='b',marker=".",label="auto")
#-plt.title('Human vs. Computer Generated Moth Trajectory')
#-plt.xlabel('x in pels')
#-plt.ylabel('y in pels')
#-plt.legend(loc='upper right')
#-plt.savefig('human_vs_comp.png') #**call before show**#
#-plt.show()


