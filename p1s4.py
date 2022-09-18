import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

data = pd.read_csv('C:/Users/saple/Desktop/python/ELL409/Assn1/train.csv')
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

N_test = len(x_test)
x0_test = [1 for _ in range(N_test)]
x_test = np.column_stack((x0_test, x_test))


#Using a modulo loss function to reduce dependence on outlier

w = np.random.rand(7)
plt.figure()
epochs = 100
for i in range(N*epochs):
	sample = (int)(N*np.random.random())
	x_s = x[sample, :]
	t_s = t[sample]
	y = np.matmul(w.T, x_s)
	diff = y-t_s
	alpha = 0.001
	lamda = 0.01
	regpam = lamda*np.sign(w)
	regpam[0] = 0
	grad = np.sign(diff)*x_s + regpam			#Remove w_0 from this equation
	w = w - alpha*grad
	if (i%N==0):
		loss = np.sum(np.square(np.matmul(x,w) - t))/N		#confirm this loss function
		plt.plot (i, loss, 'r.')
	#if (loss < 0.01): break

print("Weights: ", w)
print("Final Loss: ", loss)

plt.figure()
plt.scatter(np.arange(N_test), t_test, color = 'blue', marker = '.')
y_test = np.matmul(x_test,w)
plt.scatter(np.arange(N_test), y_test, color = 'orange', marker = '.')
rmse = np.sqrt(np.sum(np.square(y_test - t_test))/N_test)
print("Root Mean Square error on given test set: ", rmse, end='\n\n')

plt.show()