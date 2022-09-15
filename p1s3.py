import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score

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

x_test = pd.read_csv('C:/Users/saple/Desktop/python/ELL409/Assn1/test.csv').values
t_test = x_test[:, 6]
x_test = x_test[:, :6]

t_test = (t_test - tm)/ts
x_test = (x_test - xm)/xs

x0_test = [1 for _ in range(400)]
x_test = np.column_stack((x0_test, x_test))


#Stochastic Gradient Descent Ridge Regression
w = np.random.rand(7)
plt.figure()
for i in range(10000):					#Note that we have 10'000 iterations instead of 1'000 here
	sample = (int)(1200*np.random.random())
	x_s = x[sample, :]
	t_s = t[sample]
	y = np.matmul(w.T, x_s)
	diff = y-t_s
	alpha = 0.001
	lamda = 0.1
	regpam = lamda*w
	regpam[0] = 0
	grad = diff*x_s + regpam			#Remove w_0 from this equation
	w = w - alpha*grad
	loss = np.sum(np.square(np.matmul(x,w) - t))/N + lamda*np.sum(np.square(w))	#confirm this loss function
	plt.plot (i, loss, 'ro')
	if (loss < 0.01): break
print("Weights using Ridge Regression: ", w)
print("Final Loss: ", loss)

plt.figure()
plt.scatter(np.arange(400), t_test)
y_test = np.matmul(x_test,w)
plt.scatter(np.arange(400), y_test)
rmse = np.sqrt(mean_squared_error(t_test, y_test))
r2 = r2_score(t_test, y_test)
print(rmse, r2)

#Stochastic Gradient Descent Lasso Regression
w = np.random.rand(7)
plt.figure()
for i in range(10000):					#Note that we have 10'000 iterations instead of 1'000 here
	sample = (int)(1200*np.random.random())
	x_s = x[sample, :]
	t_s = t[sample]
	y = np.matmul(w.T, x_s)
	diff = y-t_s
	alpha = 0.001
	lamda = 0.01
	regpam = lamda*np.sign(w)
	regpam[0] = 0
	grad = diff*x_s + regpam			#Remove w_0 from this equation
	w = w - alpha*grad
	loss = np.sum(np.square(np.matmul(x,w) - t))/N		#confirm this loss function
	plt.plot (i, loss, 'ro')
	if (loss < 0.01): break
print("Weights using Lasso Regression: ", w)
print("Final Loss: ", loss)

plt.figure()
plt.scatter(np.arange(400), t_test)
y_test = np.matmul(x_test,w)
plt.scatter(np.arange(400), y_test)
rmse = np.sqrt(mean_squared_error(t_test, y_test))
r2 = r2_score(t_test, y_test)
print(rmse, r2)

plt.show()