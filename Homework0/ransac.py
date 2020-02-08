import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
import random
import os

def ransacer(x_data,y_data,num_point,num_iter,thresh,thresh_point):
    for i in range(num_iter):
        data = random.sample(range(len(x_data)))
        
	
	
def main(num_params, num_iters, ratio_of_inliners):
	dataset = pd.read_csv('data_2.csv')
	x = dataset.iloc[:, 0].values
	x = np.reshape(np.array(x), [-1,1])
	# y = dataset.iloc[:, 1].values
	# y = np.reshape(np.array(y), [-1, 1])
	indices = np.arange(x.shape[0])
	yticks = np.linspace(-10, 10, x.shape[0])
	y = 4*yticks**2 + yticks + 1
	for i in range(x.shape[0]):
		if np.random.uniform(0,1) > 0.1:
			y[i] = y[i] + np.random.randint(0,100) + 1
	y = np.reshape(np.array(y), [-1, 1])
	threshold = 1
	max_inliners = 0
	num_inliners = 0
	total_error = 0
	error = np.zeros(x.shape[0])

	if not os.path.exists('./'+str(num_iters)+'_'+str(threshold)):
		os.mkdir('./'+str(num_iters)+'_'+str(threshold))
	
	for k in range(num_iters):
		print('------------------',k)
		A = np.random.choice(indices, 3, replace=False)
		x_data = x[A]
		y_data = y[A]
		x2_data = x_data**2
		values = np.concatenate((x2_data, x_data, np.ones((3,1))), axis=1)
		model = np.dot(np.linalg.inv(values), y_data)
		for j in range(x.shape[0]):
			point = np.reshape(np.array([[x[j]**2], [x[j]], [1]]), [1,3])
			predicted = np.dot(point, model)     # 
			# print(point, model, predicted)
			# exit(-1)
			error[j] = np.abs(predicted - y[j])  # ordinary least square
			total_error += error[j]
			if error[j] < threshold: 			 # 
				num_inliners += 1 				 # if within threshold, it is a inlier
		error_mean = total_error/(x.shape[0]); # calc mean error for each fit		
		if num_inliners > max_inliners:
			max_inliners = num_inliners
			best_parameters = model
			best_model_pts = A
			best_error_mean = error_mean 
			# print(model, max_inliners*100/x.shape[0])
		total_error = 0	
		num_inliners = 0
	# # for i in range(len(x)):
	#     a = x[i]**2
	#     b = x[i]
	#     c = 1
	#     A.append([a,b,c])
	    
	# A = np.asarray(A)
	# A_trans = A.transpose()
	# A_prod = np.matmul(A_trans,A)
	# A_prod = A_prod + 1000*np.identity(A_prod.shape[0])
	# A_inv = np.linalg.inv(A_prod)

	# sol = np.matmul(A_inv,A_trans)
	# solutin = np.matmul(sol,y)

		if k%1000 == 0:
			parabola = []

			for i in range(x.shape[0]):
			    para = best_parameters[0]*(x[i]**2) + best_parameters[1]*(x[i]) + best_parameters[2]
			    parabola.append([para,x[i]])

			parabola=np.asarray(parabola)
			fig = plt.figure()
			plt.scatter(x, y)
			plt.plot(parabola[:,1],parabola[:,0],'r-',label="Fit")
			plt.title('Inlier Ratio: ' + str(float(max_inliners)/(x.shape[0]))+ ', Model Params: ' + '%.3f'%model[0] + ' ' + '%.3f'%model[1] + ' ' + '%.3f'%model[2])
			plt.xlabel('x')
			plt.ylabel('y')
			plt.savefig('./'+str(num_iters)+"_"+str(threshold)+"/"+'plot_'+str(k)+'_'+str(threshold)+'.png')
			plt.close()
	print best_model_pts

if __name__ == '__main__':
    num_iters = 10000
    num_params = 3 # 3 - Parabola , 2 - Line. Minimum number of points required to fit your model.
    ratio_of_inliners = 0.95 # Number of inliners.
    main(num_params, num_iters, ratio_of_inliners)
