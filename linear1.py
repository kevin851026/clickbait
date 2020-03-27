import json
import time
import pickle
import gensim
import numpy as np
from sklearn.svm import LinearSVR
from sklearn.svm import SVR
from sklearn import linear_model
model=gensim.models.Word2Vec.load('model.txt')
X_train=[]
Y_train=[]
with open("dictt0.json", encoding='utf8') as f:
	data=json.load(f)
	for i in data:
		try:
			X_train.append(model[i])
			Y_train.append(0)
		except:
			continue
with open("dictt1.json", encoding='utf8') as f:
	data=json.load(f)
	for i in data:
		try:
			X_train.append(model[i])
			Y_train.append(0.33333)
		except:
			continue
with open("dictt2.json", encoding='utf8') as f:
	data=json.load(f)
	for i in data:
		try:
			X_train.append(model[i])
			Y_train.append(0.66666)
		except:
			continue
with open("dictt3.json", encoding='utf8') as f:
	data=json.load(f)
	for i in data:
		try:
			X_train.append(model[i])
			Y_train.append(1)
		except:
			continue
X_train=np.array(X_train)
Y_train=np.array(Y_train)

print(time.asctime( time.localtime(time.time()) ))
#model = LinearSVR(loss='squared_epsilon_insensitive',C=0.01,epsilon=0,verbose=1,max_iter=4000)
#model = linear_model.Ridge(max_iter=8000,solver='svd')
#model = linear_model.LinearRegression(normalize=True)
model =  SVR(max_iter=1000,verbose =True)
model.fit(X_train,Y_train)
print(time.asctime( time.localtime(time.time()) ))
print(model.score(X_train,Y_train))

pickle.dump(model, open('linear_model', 'wb'))