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

#Batch Gradient Descent
w = np.random.rand(7)
plt.figure()
for i in range(1000):
	y = np.matmul(x,w)					#predictions
	diff = y-t							#differences
	alpha = 0.1						
	#learning rate
	grad = np.matmul(diff.T, x)/N		#gradient
	loss = np.sum(np.square(diff))/N	#loss (error)
	w = w - alpha*grad					#update
	# print(loss, w)
	plt.plot(i, loss , 'r.')			#plotting loss against number of iterations, making sure it is decreasing
	#if (loss < 0.01): break				#threshold for early stopping
print("Weights using Batch Gradient Descent: ", w)
print("Final Loss: ", loss)

plt.figure()
plt.scatter(np.arange(N_test), t_test)
y_test = np.matmul(x_test,w)
plt.scatter(np.arange(N_test), y_test)
rmse = np.sqrt(np.sum(np.square(y_test - t_test))/N_test)
r2 = r2_score(t_test, y_test)
print(rmse, r2)




#Stochastic Gradient Descent
w = np.random.rand(7)
plt.figure()
epochs = 1000
for i in range(N*epochs):
	sample = (int)(N*np.random.random())
	x_s = x[sample, :]
	t_s = t[sample]
	y = np.matmul(x_s.T, w)
	diff = y-t_s
	alpha = 0.001
	grad = diff*x_s
	w = w - alpha*grad
	if (i%N == 0):
		loss = np.sum(np.square(np.matmul(x,w) - t))/N
		plt.plot (i, loss, 'r.')
	#if (loss < 0.01): break

loss = np.sum(np.square(np.matmul(x,w) - t))/N
print("Weights using Stochastic Gradient Descent: ", w)
print("Final Loss: ", loss)

plt.figure()
plt.scatter(np.arange(N_test), t_test)
y_test = np.matmul(x_test,w)
plt.scatter(np.arange(N_test), y_test)
rmse = np.sqrt(np.sum(np.square(y_test - t_test))/N_test)
r2 = r2_score(t_test, y_test)
print(rmse, r2)




#Mini Batch Gradient Descent
batch_sizes = [10, 60, 120, 360, 600]
epochs=1000
for size in batch_sizes:
	w = np.random.rand(7)
	plt.figure()
	for i in range((N//size)*epochs):
		sample = np.random.permutation(N)[:size]
		x_s = x[sample, :]
		t_s = t[sample]
		y = np.matmul(x_s, w)
		diff = y-t_s
		alpha = 0.001 + ((size/10)*0.001)
		grad = np.matmul(diff.T, x_s)/size
		w = w - alpha*grad
		if i%(N//size) == 0:
			loss = np.sum(np.square(np.matmul(x,w) - t))/N
			plt.plot (i, loss, 'r.')
		#if (loss < 0.00001): break
	loss = np.sum(np.square(np.matmul(x,w) - t))/N
	print("Weights using Mini-Batch Gradient Descent, with batch size of {}: ".format(size), w)
	print("Final Loss: ", loss)

	plt.figure()
	plt.scatter(np.arange(N_test), t_test)
	y_test = np.matmul(x_test,w)
	plt.scatter(np.arange(N_test), y_test)
	rmse = np.sqrt(np.sum(np.square(y_test - t_test))/N_test)
	r2 = r2_score(t_test, y_test)
	print(rmse, r2)


plt.show()