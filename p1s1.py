import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

data = pd.read_csv('C:/Users/saple/Desktop/python/ELL409/Assn1/train.csv')
data = data.drop(labels=910, axis=0) ###to drop that one outlier in the dataset
x = data.values

# for i in range(6):
#     for j in range (i+1, 6):
#         plt.figure()
#         plt.scatter(x[:,i], x[:,j])
#         plt.xlabel("x{}".format(i+1))
#         plt.ylabel("x{}".format(j+1))

# for i in range(6):
#     plt.figure()
#     plt.scatter(x[:,i], x[:,6])
#     plt.xlabel("x{}".format(i+1))
#     plt.ylabel("t")

# plt.show()

print(data.corr())