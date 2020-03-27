from sklearn import linear_model
from sklearn.svm import LinearSVR
from sklearn.datasets import make_regression
import time
import sen2Vec
import json_lines
import pickle
import numpy as np
input_data=sen2Vec.trainVector('instances.jsonl')
test_data=sen2Vec.trainVector('test.jsonl')
labels={}
means={}
with open('truth.jsonl', 'rb') as f:
    for label in json_lines.reader(f):
        labels[label['id']]=label['truthMode']
        means[label['id']]=label['truthMean']
X_train=[]
Y_train=[]
for i in range(0,len(input_data)):
	for j in range(0,len(input_data[i])):
		for k in range(0,len(input_data[i][j])-1):
			X_train.append(input_data[i][j][k])
			Y_train.append(labels[input_data[i][j][-1]])
X_train=np.array(X_train)
Y_train=np.array(Y_train)

labels={}
means={}
with open('test_truth.jsonl', 'rb') as f:
    for label in json_lines.reader(f):
        labels[label['id']]=label['truthMode']
        means[label['id']]=label['truthMean']
X_test=[]
Y_test=[]
for i in range(0,len(test_data)):
	for j in range(0,len(test_data[i])):
		for k in range(0,len(test_data[i][j])-1):
			X_test.append(test_data[i][j][k])
			Y_test.append(labels[test_data[i][j][-1]])
X_test=np.array(X_test)
Y_test=np.array(Y_test)

print(time.asctime( time.localtime(time.time()) ))
model = LinearSVR(loss='squared_epsilon_insensitive' ,verbose=1,max_iter=1000)
#model = linear_model.Ridge(max_iter=5000,fit_intercept=True)
model.fit(X_train,Y_train)
print(time.asctime( time.localtime(time.time()) ))
print(model.score(X_train,Y_train))

save_classifier = open("linear_model","wb")
pickle.dump(model, save_classifier)
save_classifier.close()

f1=open('truthMean.txt','w')
f2=open('linear_predict.txt','w')
for i in Y_test:
	f1.write(str(i))
	f1.write('\n')
for i in list(model.predict(X_test)):
	tmp=[abs(1-i),abs(0.6666667-i),abs(0.33333334-i),abs(0-i)]
	x=tmp.index(min(tmp))
	if x==0:
		x=1
	elif x==1:
		x=0.6666667
	elif x==2:
		x=0.33333334
	elif x==3:
		x=0
	f2.write(str(x))
	f2.write('\n')
f1.close()
f2.close()
print('done')