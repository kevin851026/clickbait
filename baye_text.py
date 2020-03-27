import json
import time
import pickle
import gensim
from textblob.classifiers import NaiveBayesClassifier
training_data=[]
model=gensim.models.Word2Vec.load('model.txt')
with open("para0.json", encoding='utf8') as f:
	data=json.load(f)
	for i in data:
		try:
			x=model[i]
			tmp=(i,str(0))
			training_data.append(tmp)
		except:
			continue
with open("para3.json", encoding='utf8') as f:
	data=json.load(f)
	for i in data:
		try:
			x=model[i]
			tmp=(i,str(0.3))
			training_data.append(tmp)
		except:
			continue
with open("para6.json", encoding='utf8') as f:
	data=json.load(f)
	for i in data:
		try:
			x=model[i]
			tmp=(i,str(0.6))
			training_data.append(tmp)
		except:
			continue
with open("para10.json", encoding='utf8') as f:
	data=json.load(f)
	for i in data:
		try:
			x=model[i]
			tmp=(i,str(1))
			training_data.append(tmp)
		except:
			continue
x=[]
for i in range(0,500):
    x.append(training_data[i])
    x.append(training_data[i+16000])
    x.append(training_data[i+20000])
    x.append(training_data[i+22000])

print(time.asctime( time.localtime(time.time()) ))
model = NaiveBayesClassifier(x)
print(time.asctime( time.localtime(time.time()) ))
'''
save_classifier = open("naivebayes.pickle","wb")
pickle.dump(model, save_classifier)
save_classifier.close()
'''