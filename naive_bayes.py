import math
import time
import sen2Vec
import json_lines
print(time.asctime( time.localtime(time.time()) ))
input_titles,input_data=zip(sen2Vec.trainVector('instances.jsonl'))
print('input_done')
test_titles,test_data=zip(sen2Vec.trainVector('test.jsonl'))
print('test_done')
output={}
input_summaries=[]
def startTrain(): #start traing 
    put_label()
    tmp=[]                  #transform traning data into prescript format
    for i in input_data[0]: #put in keyword
        for j in i:
            tmp.append(j)
   # for i in input_titles[0]:  #put in titles
   #     for j in i:
   #         tmp.append(j)
    global input_summaries
    input_summaries = summarizeByClass(tmp)
    print('traing_done')
def put_label():    #put on label by id
    labels={}
    with open('truth.jsonl', 'rb') as f:
        for label in json_lines.reader(f):
            labels[label['id']]=label['truthMode']
    for i in range(0,len(input_data[0])):
        for j in range(0,len(input_data[0][i])-1):
            input_data[0][i][j]=input_data[0][i][j].tolist()
            input_data[0][i][j].append(labels[input_data[0][i][-1]])
        input_data[0][i].pop()
    for i in range(0,len(input_titles[0])):
        for j in range(0,len(input_titles[0][i])-1):
            input_titles[0][i][j]=input_titles[0][i][j].tolist()
            input_titles[0][i][j].append(labels[input_titles[0][i][-1]])
        input_titles[0][i].pop()

    print('label_done')
def mean(numbers):
    return sum(numbers)/float(len(numbers))
def separateByClass(dataset):
    separated = {}
    for i in range(len(dataset)):
        vector = dataset[i]
        if (vector[-1] not in separated):
            separated[vector[-1]] = []
        separated[vector[-1]].append(vector)
    return separated
def stdev(numbers):
    avg = mean(numbers)
    variance = sum([pow(x-avg,2) for x in numbers])/float(len(numbers)-1)
    return math.sqrt(variance)
def summarize(dataset):
    summaries = [(mean(attribute), stdev(attribute)) for attribute in zip(*dataset)]
    del summaries[-1]
    return summaries
def summarizeByClass(dataset):
    separated = separateByClass(dataset)
    summaries = {}
    for classValue in separated:
        summaries[classValue]=summarize(separated[classValue])
    return summaries
def calculateProbability(x, mean, stdev):
    exponent = math.exp(-(math.pow(x-mean,2)/(2*math.pow(stdev,2))))
    return (1 / (math.sqrt(2*math.pi) * stdev)) * exponent

def calculateClassProbabilities(inputVector):
    probabilities = {}
    for classValue in input_summaries:
        probabilities[classValue] = 1
        for i in range(0,len(inputVector)-1):
            mean, stdev = input_summaries[classValue][i]
            x = inputVector[i]
            probabilities[classValue] *= calculateProbability(x, mean, stdev)
    return probabilities
def predict(inputVector):   #choose the biggest probability
    probabilities = calculateClassProbabilities(inputVector)
    maxVal=0
    maxClass=0
    for i in probabilities:
        if maxVal < probabilities[i]:
            maxVal = probabilities[i]
            maxClass = i
    return maxClass
def startpredict():
    for i in range(0,len(test_data[0])):    #delete id
        test_data[0][i].pop()
    for i in range(0,len(test_titles[0])):
        test_titles[0][i].pop()
    answer=[]
    avg=[]
    for i,j in zip(test_data[0],test_titles[0]):
        tmp={}
        for k in i:
            ans=predict(k)
            if ans in tmp:
                tmp[ans]+=1
            else:
                tmp[ans]=1
        #for k in j:
        #    ans=predict(k)
        #    if ans in tmp:
        #        tmp[ans]+=1
        #    else:
        #        tmp[ans]=1
        #tmp=sorted(tmp.items(),key=lambda x:(x[1],x[0]),reverse=True)[:5]
        x=0
        count=0
        for k in tmp:
            x+=tmp[k]*k
            count+=tmp[k]
        try:                        #may have no keyword
            avg.append(x/count)
        except:
            avg.append(-1)        
        try:                        #may have no titles
            answer.append(tmp[0][0])
        except:
            answer.append(0)
    print('predict_done')
    with open('test.jsonl', 'rb') as f:
        global output
        output={}
        i=0
        for item in json_lines.reader(f):
            output[item['id']]=answer[i]
            i+=1
    f1=open('truthMode.txt','w')
    f2=open('avg.txt','w')
    with open('test_truth.jsonl', 'rb') as f:
        i=0
        for item in json_lines.reader(f):
            f1.write(str(item['truthMode']))
            f1.write('\n')
            f2.write(str(avg[i]))
            f2.write('\n')
            i+=1
    f1.close()
    f2.close()
    print('done')
    print(time.asctime( time.localtime(time.time()) ))
    
startTrain()
startpredict()
"""
input format:
train_data=[[1,1,1,A],[2,2,2,A],[3,3,3,A],[1,1,1,B],[2,2,2,B],[3,3,3,B]]
test_data=[[4,4,4],[0,0,0,]]
"""
