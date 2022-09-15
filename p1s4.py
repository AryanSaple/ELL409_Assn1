import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score

data = pd.read_csv('C:/Users/saple/Desktop/python/ELL409/Assn1/train.csv')
data = data.drop(labels=910, axis=0)
x = data.values
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


#through subpart 1, we see that x2, x3 have high positive correlation with x1, and x6 has high negative correlation with x1.
#though x4, x5 have lower correlation coeff, they are also clearly linearly related to x1, as seen in the graphs.
#therefore, we are modelling as: y ~ w0 + w1x1

w = np.random.rand(2)
x_train_1 = x[:, :2]
x_test_1 = x_test[:, :2]
plt.figure()
for i in range(10000):					#Note that we have 10'000 iterations instead of 1'000 here
	sample = (int)(1199*np.random.random())
	x_s = x_train_1[sample, :]
	t_s = t[sample]
	y = np.matmul(w.T, x_s)
	diff = y-t_s
	alpha = 0.001
	lamda = 0
	regpam = lamda*w
	regpam[0] = 0
	grad = diff*x_s + regpam			#Remove w_0 from this equation
	w = w - alpha*grad
	loss = np.sum(np.square(np.matmul(x_train_1,w) - t))/N		#confirm this loss function
	plt.plot (i, loss, 'r.')
	if (loss < 0.01): break

print("Weights using Ridge Regression: ", w)
print("Final Loss: ", loss)

plt.figure()
plt.scatter(np.arange(400), t_test, color = 'blue', marker = '.')
y_test = np.matmul(x_test_1,w)
plt.scatter(np.arange(400), y_test, color = 'orange', marker = '.')
rmse = np.sqrt(mean_squared_error(t_test, y_test))
r2 = r2_score(t_test, y_test)
print(rmse, r2)

plt.show()