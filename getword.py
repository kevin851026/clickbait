import jsonlines
import gensim
import numpy as np

def trainVector(file):
    data=[]
    with jsonlines.open(file) as jf:
        for i in jf:
            del i['postTimestamp']
            del i['postMedia']
            del i['targetCaptions']
            data.append(i)
    model=gensim.models.Word2Vec.load('model.txt')
    stopWord=[]
    with open('stopWord.txt','r') as f:
        for i in f.readlines():
            i=i[:-1]
            stopWord.append(i)
    keyword=[]
    #f=open("Dict.txt",'w',encoding='utf-8')###############################
    f=open('xxx.txt','w')
    for i in data:
        keyDict={}
        words=[]
        for j in i['targetKeywords'].split(','):
            for k in j.split():
                words.append(k)
        keyDict=keyDictionary(keyDict,words,stopWord)

        paraDict={}
        words=[]
        for j in i['targetParagraphs']:
            for k in j.split():
                words.append(k)
        paraDict=keyDictionary(paraDict,words,stopWord)
        paraDict=sorted(paraDict.items(),key=lambda x:(x[1],x[0]),reverse=True)[:10]
        paraDict=dict(paraDict)
        #print(paraDict)#########################
        keyDict.update(paraDict) #count important
        #f.write(' '.join(keyDict.keys())+'\n')######################
        tmp=[]
        for j in keyDict:
            try:
                x=model[j]
                f.write(str(j))
                f.write('\n')
                tmp.append(j)
            except:
                continue

        tmp.append(i['id'])
        keyword.append(tmp)
    f.close()
    return keyword
def keyDictionary(keyDict,words,stopWord):
    for i in words:
        i=wordArrange(i)
        if not checkStopword(i,stopWord):
            if i in keyDict:
                keyDict[i]+=1
            else:
                keyDict[i]=1
    return keyDict

def wordArrange(words):
    if '.' in words:
        words=words[:words.index('.')]
    if ',' in words:
        words=words[:words.index(',')]
    if ':' in words:
        words=words[:words.index(':')]
    if ';' in words:
        words=words[:words.index(';')]
    if '!' in words:
        words=words[:words.index('!')]
    if '|' in words:
        words=words[:words.index('|')]
    if '?' in words:
        words=words[:words.index('?')]
    if '’' in words:
        words=words[:words.index('’')]
    if '—' in words:
        words=words[:words.index('—')]
    if '-' in words:
        words=words[:words.index('-')]
    if '[' in words:
        words=words[1:]
    if ']' in words:
        words=words[:words.index(']')]
    if '(' in words:
        words=words[1:]
    if ')' in words:
        words=words[:words.index(')')]
    if '“' in words:
        words=words[1:]
    if '”' in words:
        words=words[:words.index('”')]
    if '\"' in words:
        if words.index('\"')==0:
            words=words[1:]
    if '\"' in words:
        if words.index('\"')!=0:
            words=words[:words.index('\"')]
    if '\'' in words:
        if words.index('\'')==0:
            words=words[1:]
    if '\'' in words:
        if words.index('\'')!=0:
            words=words[:words.index('\'')]
    words=words.lower()
    return words


def sen2v(words,vec,stopWord,model):
    for j in words:
        #j=j.lower()
        j=wordArrange(j)
        if not checkStopword(j,stopWord):
            try:
                vec+=model[j]
            except:
                continue
    return vec

def checkStopword(word,stopWord):
    low=0
    upper=len(stopWord)-1
    while low<=upper:
        mid=(low+upper)//2
        if word>stopWord[mid]:
            low=mid+1
        elif word<stopWord[mid]:
            upper=mid-1
        else:
            return True
    return False
