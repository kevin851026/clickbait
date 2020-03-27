import json_lines
import time
import pickle
import getword
from textblob.classifiers import NaiveBayesClassifier

keyword=getword.trainVector('instances.jsonl')
labels={}
with open('truth.jsonl', 'rb') as f:
    for label in json_lines.reader(f):
        labels[label['id']]=label['truthMode']
training_data=[]
for i in keyword:
    for j in range(0,len(i)-1):
        tmp=(i[j],str(labels[i[-1]]))
        training_data.append(tmp)
x=[]
for i in range(0,100):
    x.append(training_data.pop())

print(time.asctime( time.localtime(time.time()) ))
model = NaiveBayesClassifier(training_data)
print(time.asctime( time.localtime(time.time()) ))
"""
save_classifier = open("naivebayes.pickle","wb")
pickle.dump(model, save_classifier)
save_classifier.close()
"""
