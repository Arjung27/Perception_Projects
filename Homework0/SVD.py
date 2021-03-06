# -*- coding: utf-8 -*-
"""
Created on Wed Feb  5 15:33:34 2020

@author: A BANERJEE
"""
import numpy as np

x1 = 5
y1 = 5
xp1 = 100
yp1 = 100

x2 = 150
y2 = 5
xp2 = 200
yp2 = 80

x3 = 150
y3 = 150
xp3 = 220
yp3 = 80

x4 = 5
y4 = 150
xp4 = 100
yp4 = 200

A = np.array([[-x1,-y1,-1,0,0,0,x1*xp1,y1*xp1,xp1],[0,0,0,-x1,-y1,-1,x1*yp1,y1*yp1,yp1],[-x2,-y2,-1,0,0,0,x2*xp2,y2*xp2,xp2],\
[0,0,0,-x2,-y2,-1,x2*yp2,y2*yp2,yp2],[-x3,-y3,-1,0,0,0,x3*xp3,y3*xp3,xp3],[0,0,0,-x3,-y3,-1,x3*yp3,y3*yp3,yp3],\
[-x4,-y4,-1,0,0,0,x4*xp4,y4*xp4,xp4],[0,0,0,-x4,-y4,-1,x4*yp4,y4*yp4,yp4]])

A_trans = A.transpose()

A_prod = np.dot(A_trans,A)
w,v = np.linalg.eig(A_prod)
#v = np.flip(v, axis=0

H = v[:,8]
H = np.reshape(H,(3,3))
print(H)


#uA_prod = np.matmul(A,A_trans)
#uw,uv = np.linalg.eig(uA_prod)


