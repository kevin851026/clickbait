import jsonlines
import json
import warnings
warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')
import gensim
from nltk.corpus import wordnet
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tag import pos_tag
from nltk import word_tokenize
import numpy as np

def thirdVer_para(x,y):
	data=[]
	with jsonlines.open(x) as jf:
		for i in jf:
			del i['postTimestamp']
			del i['postMedia']
			del i['targetCaptions']
			data.append(i)

	truth=[]
	with jsonlines.open(y) as jf:
		for i in jf:
			truth.append(i)

	model=gensim.models.Word2Vec.load('model.txt')
	stopWord=[]
	with open('stopWord.txt','r') as f:
		for i in f.readlines():
			i=i[:-1]
			stopWord.append(i)
			print(i)
	title_x_train=[]
	title_y_train=[]

	dictionary=[{},{},{},{}]
	for i in truth:
		if i['truthMode']==0.0:
			post={}
			for j in data:
				if j['id']==i['id']:
					post=j
			for j in post['targetParagraphs']:
				words=j.split()
				dictionary[0]=keyDictionary(dictionary[0],words,stopWord)
		elif i['truthMode']-0.33<0.01:
			post={}
			for j in data:
				if j['id']==i['id']:
					post=j
			for j in post['targetParagraphs']:
				words=j.split()
				dictionary[1]=keyDictionary(dictionary[1],words,stopWord)
		elif i['truthMode']-0.66<0.01:
			post={}
			for j in data:
				if j['id']==i['id']:
					post=j
			for j in post['targetParagraphs']:
				words=j.split()
				dictionary[2]=keyDictionary(dictionary[2],words,stopWord)
		else:
			post={}
			for j in data:
				if j['id']==i['id']:
					post=j
			for j in post['targetParagraphs']:
				words=j.split()
				dictionary[3]=keyDictionary(dictionary[3],words,stopWord)
	dictSum=[0,0,0,0]
	for i in range(4):
		dictTmp={}
		for j in dictionary[i].keys():
			if dictionary[i][j]>4:
				dictTmp[j]=dictionary[i][j]
		dictionary[i]=dictTmp		
		dictSum[i]=sum(dictionary[i].values())
	totalWord=sum(dictSum)

	chiSquareDict=getChiSquare(dictionary,dictSum,totalWord)

	for i in range(4):
		dictTmp={}
		for j in chiSquareDict[i].keys():
			if chiSquareDict[i][j]>2:
				dictTmp[j]=chiSquareDict[i][j]
		chiSquareDict[i]=dictTmp		
		chiSquareDict[i]=sorted(chiSquareDict[i].items(),key=lambda x:(x[1],x[0]),reverse=True)
		chiSquareDict[i]=dict(chiSquareDict[i])

#	for i in range(4):
#		s='dictt'+str(i)+'.json'
#		with open(s,'w') as file:
#			file.write(json.dumps(chiSquareDict[i]))
	ans=[np.array([1,0,0,0]),np.array([0,1,0,0]),np.array([0,0,1,0]),np.array([0,0,0,1])]
	for i in range(4):
		for key in chiSquareDict[i].keys():
			try:
				title_x_train.append(model[key])
				title_y_train.append(ans[i])
			except:
				continue
	title_x_train=np.array(title_x_train)
	title_y_train=np.array(title_y_train)

	return title_x_train,title_y_train


def getChiSquare(dictionary,dictSum,totalWord):
	chiSquare=[{},{},{},{}]
	for i in range(4):
		for key in dictionary[i].keys():
			wordSum=0
			for j in range(4):
				wordSum+=dictionary[j].get(key) if dictionary[j].get(key) else 0
			exVal=wordSum*dictSum[i]/totalWord
			cs=(dictionary[i][key]-exVal)**2/exVal
			if dictionary[i][key]-exVal<0:
				cs*=-1
			chiSquare[i][key]=cs
	return chiSquare

def keyDictionary(keyDict,words,stopWord):
	lm=WordNetLemmatizer()
	sen=[]
	for i in words:
		sen.append(wordArrange(i))
	sen=' '.join(sen)
	words=[]
	for i,pos in pos_tag(word_tokenize(sen)):
		wordnet_pos=get_wordnet_pos(pos) or wordnet.NOUN
		words.append(lm.lemmatize(i,pos=wordnet_pos))

	for i in words:
		if not checkStopword(i,stopWord):
			if i in ['','@','$','&','–','%']:
				continue
			if i in keyDict:
				keyDict[i]+=1
			else:
				keyDict[i]=1
	return keyDict

def get_wordnet_pos(tag):
    if tag.startswith('J'):
        return wordnet.ADJ
    elif tag.startswith('V'):
        return wordnet.VERB
    elif tag.startswith('N'):
        return wordnet.NOUN
    elif tag.startswith('R'):
        return wordnet.ADV
    else:
        return None

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
	if '•' in words:
		words=words[1:]
	if '—' in words:
		words=words[:words.index('—')]
	#if '–' in words:
	#	words=words[:words.index('–')]
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
	if '‘' in words:
		words=words[1:]
	if '’' in words:
		words=words[:words.index('’')]
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
	if '#' in words:
		if words.index('#')==0:
			words=words[1:]
	words=words.lower()
	return words

#test,test2=thirdVer_para('instances.jsonl','truth.jsonl')