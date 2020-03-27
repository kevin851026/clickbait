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

	trainVec=[]
	preTrainVec=[]
	#f=open("Dict.txt",'w',encoding='utf-8')###############################
	for i in data:
		artInform=[]
		words=i['postText'][0].split()
		postText=np.zeros(300)
		postText=sen2v(words,postText,stopWord,model)
		artInform.append(postText)

		words=i['targetTitle'].split()
		targetTitle=np.zeros(300)
		targetTitle=sen2v(words,targetTitle,stopWord,model)
		artInform.append(targetTitle)

		words=i['targetDescription'].split()
		targetDescript=np.zeros(300)
		targetDescript=sen2v(words,targetDescript,stopWord,model)
		artInform.append(targetDescript)
		artInform.append(i['id'])
		trainVec.append(artInform)

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
		paraDict=sorted(paraDict.items(),key=lambda x:(x[1],x[0]),reverse=True)[:15]
		paraDict=dict(paraDict)
		#print(paraDict)#########################


		keyDict.update(paraDict) #count important
		#f.write(' '.join(keyDict.keys())+'\n')######################
		mostWord=[]
		for j in keyDict.keys():
			try:
				mostWord.append(model[j])
			except:
				continue
		mostWord.append(i['id'])

		preTrainVec.append(mostWord)
	return trainVec,preTrainVec
	#f.close()
	#output
	# f=open('train.txt','w')
	# for i in trainVec,:
	# 	for j in i:
	# 		f.write(str(j))
	# 	f.write('\n----------------------------------------------------\n\n')
	# f.close()
	#f=open('pretrain.txt','w')
	#for i in preTrainVec:
	#	for j in i:
	#		f.write(str(j))
	#	f.write('\n-------------------------------------------------------\n\n')
	#f.close()


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