import matplotlib.pyplot as plt
import matplotlib as mat
import numpy as np
from sklearn.datasets import load_boston

def Normalize(factor,result):
    for i in range(13):
        factor[:,i] = (factor[:,i]-factor[:,i].min())/(factor[:,i].max()-factor[:,i].min())
    result = result.reshape(-1, 1)
    factor = np.insert(factor, 0, 1, 1)  # 增加偏置项
    result[:,0] = (result[:,0]-result[:,0].min())/(result[:,0].max()-result[:,0].min())
    return factor,result.reshape(1,-1)

factor,result = load_boston(return_X_y = True)
factor = np.array(factor)
result = np.array(result)
factor,result = Normalize(factor,result)

labels = ['CRIM','ZN','INDUS','CHAS','NOX','RM','AGE','DIS','RAD','TAX','PTRATIO','B','LSTAT']
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.ylabel('房价',fontsize=15)
for i in range(1,14):
    plt.xlabel(labels[i-1], fontsize=10)
    plt.scatter(factor[:,i],result,s=1)
plt.show()