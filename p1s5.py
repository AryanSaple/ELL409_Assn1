import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score		#REMOVE

x = pd.read_csv('C:/Users/saple/Desktop/python/ELL409/Assn1/train.csv').values
t = x[:, 6]
x = x[:, :6]

#normalizing t,x using mean and standard deviation
tm = t.mean()
ts = t.std()
xm = x.mean(axis=0)[np.newaxis, :]
xs = x.std(axis=0)[np.newaxis, :]

t = (t-tm)/ts
x = (x-xm)/xs

N = len(x)
x0 = [1 for _ in range(N)]
x = np.column_stack((x0, x))
# print(t[:10])
# print(x[:10])

#Loading test values
x_test = pd.read_csv('C:/Users/saple/Desktop/python/ELL409/Assn1/test.csv').values
t_test = x_test[:, 6]
x_test = x_test[:, :6]

t_test = (t_test - tm)/ts
x_test = (x_test - xm)/xs

N_test = len(x_test)
x0_test = [1 for _ in range(N_test)]
x_test = np.column_stack((x0_test, x_test))


#Varying sample size
#Stochastic Gradient Descent

for i in range(5):
	plt.figure()

	w = np.random.rand(7)
	sample_size = (i+1)*(N//10)
	perm = np.random.permutation(N)[:sample_size]
	x_modified=x[perm, :]
	t_modified=t[perm]
	
	epochs=1000
	for i in range(sample_size*epochs):
		sample = (int)(sample_size*np.random.random())
		x_s = x_modified[sample, :]
		t_s = t_modified[sample]
		y = np.matmul(w.T, x_s)
		diff = y-t_s
		alpha = 0.001
		grad = diff*x_s
		w = w - alpha*grad
		if i%N == 0:
			loss = np.sum(np.square(np.matmul(x_modified,w) - t_modified))/N
			plt.plot (i, loss, 'r.')
		#if (loss < 0.00001): break
	loss = np.sum(np.square(np.matmul(x_modified,w) - t_modified))/N
	print("Weights using Stochastic Gradient Descent: ", w)
	print("Final Loss: ", loss)

	plt.figure()
	plt.scatter(np.arange(N_test), t_test, color = 'blue', marker = '.')
	y_test = np.matmul(x_test,w)
	plt.scatter(np.arange(N_test), y_test, color = 'orange', marker = '.')
	rmse = np.sqrt(np.sum(np.square(y_test - t_test))/N_test)
	r2 = r2_score(t_test, y_test)
	print(rmse, r2)
	
plt.show()