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
import pickle
import random

def test(x,y):
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
	mdl=pickle.load(open('linear_model','rb'))
	stopWord=[]
	with open('stopWord.txt','r') as f:
		for i in f.readlines():
			i=i[:-1]
			stopWord.append(i)
			print(i)

	for i in truth:
		for j in data:
			if i['id']==j['id']:
				j['truthMode']=i['truthMode']

	final=[]
	crt1=0
	crt2=0
	total=0
	for i in data:
		words=[]
		for j in i['targetParagraphs']:
			sen=j.split()
			words=lemma(words,sen,stopWord)
		count=0
		summ=0
		answer=[0,0,0,0]
		for j in words:
			j=wordArrange(j)
			if not checkStopword(j,stopWord):
				try:
					vec=model[j]
				except:
					continue
				pre=get_predict(vec,mdl)
				answer[classify(pre)]+=1
				summ+=pre
				count+=1
		maxindex=0
		for j in range(0,4):
			if answer[maxindex]<answer[j]:
				maxindex=j
		ann=0
		if count>0:
			ann=summ/count
		else:
			ann=random.randint(0,1)
			maxindex==random.randint(0,3)
		print(maxindex)
		print(ann)
		ans={}
		ans['id']=i['id']
		ans['truthMode']=i['truthMode']
		if i['truthMode']>0.5:
			ans['tru']=1
		else:
			ans['tru']=0
		if ann>0.5:
			ans['avg_pre']=1
		else:
			ans['avg_pre']=0
		if maxindex>1:
			ans['mode_pre']=1
		else:
			ans['mode_pre']=0
		if ans['tru']==ans['avg_pre']:
			crt1+=1
		if ans['tru']==ans['mode_pre']:
			crt2+=1
		total+=1
		final.append(ans)
	print('avg',crt1/total)
	print('mode',crt2/total)
	with open('anss2.json','w') as f:
		f.write(json.dumps(final))

def get_predict(vec,mdl):
	test=[]
	test.append(vec)
	test=np.array(test)
	pre=mdl.predict(test)
	return pre
def classify(i):
	tmp=[abs(0-i),abs(0.33333334-i),abs(0.6666667-i),abs(1-i)]
	return tmp.index(min(tmp))
def lemma(words,sen,stopWord):
	lm=WordNetLemmatizer()
	tmpsen=[]
	for i in sen:
		tmpsen.append(wordArrange(i))
	tmpsen=' '.join(tmpsen)
	sen=[]
	for i,pos in pos_tag(word_tokenize(tmpsen)):
		wordnet_pos=get_wordnet_pos(pos) or wordnet.NOUN
		sen.append(lm.lemmatize(i,pos=wordnet_pos))

	for i in sen:
		if not checkStopword(i,stopWord):
			if i in ['','@','$','&','–','%']:
				continue
			else:
				words.append(i)
	return words

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
test('test.jsonl','test_truth.jsonl')