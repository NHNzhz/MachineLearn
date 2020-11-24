from sklearn.datasets import load_boston
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mat

input_num = 13
hidden_num = 30
output_num = 1

factor,result = load_boston(return_X_y = True)
factor = np.array(factor)
result = np.array(result)

def Normalize(factor,result):
    for i in range(input_num):
        factor[:,i] = (factor[:,i]-factor[:,i].min())/(factor[:,i].max()-factor[:,i].min())
    result = result.reshape(-1, 1)
    factor = np.insert(factor,0,1,1)#增加偏置项
    result[:,0] = (result[:,0]-result[:,0].min())/(result[:,0].max()-result[:,0].min())
    return factor,result.reshape(1,-1)

def sigmoid(z):
    return 1/(1+np.exp(-z))

def Loss(Y,result):
    size = Y.shape[1]
    temp = result-Y
    res = np.dot(temp,temp.T)
    return np.sum(res)/(2*size)

def BPloss(X,Hid,Y,V,result):
    sum = 0
    siz = result.shape[1]
    temp0 = Y-result
    temp1 = 1-Y
    temp2 = 1-Hid
    temp3 = np.zeros(shape=(input_num+1,hidden_num))
    temp4 = np.zeros(shape=(hidden_num+1,output_num))
    for i in range(input_num+1):
        for j in range(hidden_num):
            for k in range(siz):
                sum = sum+temp0[0][k]*temp1[0][k]*Y[0][k]*V[j][0]*Hid[k][j]*temp2[k][j]*X[k][i]
            temp3[i][j] = sum/siz
            sum = 0
    sum = 0
    for i in range(hidden_num+1):
        for j in range(output_num):
            for k in range(siz):
                sum = sum+temp0[0][k]*temp1[0][k]*Y[0][k]*Hid[k][i]
            temp4[i][j] = sum/siz
            sum = 0
    delta_W = temp3
    delta_V = temp4
    return delta_W,delta_V

def neuralnetwork(W,V,X,R,alpha,itera):
    for i in range(itera):
        Z = sigmoid(np.dot(X,W))
        Z = np.insert(Z,13,1,1)
        Y = sigmoid(np.dot(Z,V)).reshape(1,-1)
        dW,dV = BPloss(X,Z,Y,V,R)
        W = W-alpha*dW
        V = V-alpha*dV
        if i%1000==0 :
            print("train loss:%d" % Loss(Y,R))
    return W,V

def saveweight(W,V):
    dfW = pd.DataFrame()
    dfV = pd.DataFrame()
    dfV['V'] = V.flatten()
    dfW['W_0'] = W[0]
    dfW['W_1'] = W[1]
    dfW['W_2'] = W[2]
    dfW['W_3'] = W[3]
    dfW['W_4'] = W[4]
    dfW['W_5'] = W[5]
    dfW['W_6'] = W[6]
    dfW['W_7'] = W[7]
    dfW['W_8'] = W[8]
    dfW['W_9'] = W[9]
    dfW['W_10'] = W[10]
    dfW['W_11'] = W[11]
    dfW['W_12'] = W[12]
    dfW['W_13'] = W[13]
    dfV.to_csv('veight.csv')
    dfW.to_csv('weight.csv')

factor,result = Normalize(factor,result)
train_X = factor[0:506:2]
train_Y = result[:,0:506:2]
test_X = factor[1:506:2]
test_Y = result[:,1:506:2]
#W1 = np.random.normal(size=(input_num+1,hidden_num))
#W2 = np.random.normal(size=(hidden_num+1,output_num))
#调用现有权重
df1=pd.read_csv('weight.csv')
df2=pd.read_csv('veight.csv')
W1=np.array(df1.iloc[:,1:]).T
W2=np.array(df2.iloc[:,1:])
#W1,W2 = neuralnetwork(W1,W2,train_X,train_Y,0.5,90000)
predict_Y = sigmoid(np.dot(np.insert(sigmoid(np.dot(test_X,W1)),13,1,1),W2)).reshape(1,-1)
print('test loss: %f' % Loss(predict_Y,test_Y))
#saveweight(W1,W2)
x_list = []
for i in range(253):
    x_list.append(i+1)
x_list = np.array(x_list).flatten()
predict_Y = predict_Y.flatten()
test_Y = test_Y.flatten()
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.xlabel('样本序号',fontsize=10)
plt.ylabel('房价',fontsize=10)
plt.plot(x_list,predict_Y,label='预测值')
plt.plot(x_list,test_Y,color='red',label='实际值')
plt.legend()
plt.show()