import numpy as np
from sklearn import svm
from sklearn.preprocessing import OneHotEncoder
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

input_shape=500
train = np.loadtxt(open(str(input_shape)+"-dim-z-one-hot-labelmnist-train.csv", "rb"), delimiter=",", skiprows=0)
trainX = train[:,:-10]
trainYone_hot = train[:,-10:].astype(np.int32)
test = np.loadtxt(open(str(input_shape)+"-dim-z-one-hot-labelmnist-test.csv", "rb"), delimiter=",", skiprows=0)
testX = test[:,:-10]
testYone_hot = test[:,-10:].astype(np.int32)
print(trainX.shape)
print(trainYone_hot.shape)
print(testX.shape)
print(testYone_hot.shape)
trainY = np.argmax(trainYone_hot,axis=1)
testY = np.argmax(testYone_hot,axis=1)
# print(trainY)
# print(trainY.shape)
# print(testY.shape)


def SVM():
    model = svm.SVC(kernel="rbf",decision_function_shape='ovo',random_state=0,max_iter=10000)
    model.fit(trainX,trainY)
    print(model.score(trainX,trainY))
    print(model.score(testX,testY))

def neural_net():
    epochs = 100
    batch_size = 128
    model = Sequential()
    model.add(Dense(1000, input_dim=input_shape, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(2000, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1000, activation='relu'))
    model.add(Dense(1000,activation='relu'))
    model.add(Dense(10, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(trainX,trainYone_hot,shuffle=True,epochs=epochs,batch_size=batch_size,validation_data=(testX, testYone_hot))
    
def kNN():
    model = KNeighborsClassifier(n_neighbors=30)
    model.fit(trainX,trainY)
    print(model.score(trainX,trainY))
    print(model.score(testX,testY))
    
def random_forest():
    model = RandomForestClassifier(n_estimators = 25)
    model.fit(trainX,trainY)
    print(model.score(trainX,trainY))
    print(model.score(testX,testY))
    
# print("SVM result")
# SVM()
# print("nn result")    
neural_net()
# print("knn result")
# kNN()
# print("random forest result")
# random_forest()