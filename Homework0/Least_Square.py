# -*- coding: utf-8 -*-
"""
Created on Wed Feb  5 10:59:07 2020

@author: A BANERJEE
"""

import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt

'''
def ransacer(x_data,y_data,num_point,num_iter,thresh,thresh_point):
    for i in range(num_iter):
        data = random.sample(range(len(x_data)))
'''
        

dataset = pd.read_csv('data_2.csv')
x = dataset.iloc[:, 0].values
y = dataset.iloc[:, 1].values

A = []

for i in range(len(x)):
    a = x[i]**2
    b = x[i]
    c = 1
    A.append([a,b,c])
    
A = np.asarray(A)
A_trans = A.transpose()
A_prod = np.matmul(A_trans,A)
# For regaularization uncomment the below line
#A_prod = A_prod + 0.001*np.identity(A_prod.shape[0])
A_inv = np.linalg.inv(A_prod)

sol = np.matmul(A_inv,A_trans)
solutin = np.matmul(sol,y)

parabola = []
threshold = 11
inlier = 0

for i in range(len(x)):
    para = solutin[0]*(x[i]**2) + solutin[1]*(x[i]) + solutin[2]
    parabola.append([para,x[i]])

parabola=np.asarray(parabola)
for i in range(len(x)):
    if abs(parabola[i,0] - y[i]) <= threshold :
        inlier = inlier + 1
err = (np.linalg.norm(parabola[:,0] - y))**2

inlier =  float(inlier)/len(x) 
print (inlier)

print(err/len(x))
        


plt.scatter(x, y)
plt.plot(parabola[:,1],parabola[:,0],'b-',label="Fit")
plt.title('Error=205.131, B vector: a=-0.0024, b=1.23, c=-51.92')
plt.xlabel('x')
plt.ylabel('y')
plt.savefig('Data_2_LSE_Regu')
plt.show()
