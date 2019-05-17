#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 12 21:54:08 2019

"""
import pip

import numpy as np
import pandas as pd

from twitter import Twitter
from twitter import OAuth


apikey=''
apisecretkey=''
accesstoken=''
accesstokensecret=''

oauth = OAuth(accesstoken,accesstokensecret,apikey,apisecretkey)
api = Twitter(auth=oauth)

#lets look at whats trending around the world
t_loc = api.trends.available()
print(t_loc)

from pandas.io.json import json_normalize

df_loc=json_normalize(t_loc)
df_loc.country.value_counts()

dfNew=df_loc[df_loc['name'].str.contains('New')]
ny=dfNew.loc[dfNew.name=='New York','woeid']

ny_trend = api.trends.place(_id=ny)

type(ny)
ny.values
ny.values[0]
ny_trend = api.trends.place(_id=ny.values[0])

########## Saving and Reading Objects ######################
import json
with open('ny_trend.json', 'w') as outfile:
    json.dump(ny_trend, outfile)

# Getting back the objects:
with open('ny_trend.json') as json_data:
    ny_trend_example = json.load(json_data)

############################################################

dfny=json_normalize(ny_trend)
type(dfny.trends)
dfny.trends.shape

dftrends=json_normalize(dfny.trends.values[0])
dftrends.to_pickle('dftrends.pkl')
#dftrends = pd.read_pickle('dftrends.pkl')

api.statuses.update(status="Their is an invasion at the border, someone get Jon Snow!!!")
mytweets=api.statuses.home_timeline()



dfmyt=json_normalize(mytweets)
dfmyt.to_pickle('dfmyt.pkl')

mytweets1=api.statuses.home_timeline(count=1)
dfmyt1=json_normalize(mytweets1)

#Searching tweets on prarticular trending topics
dftrends.columns
dftrends.nlargest(5,'tweet_volume')[['name','tweet_volume']]

search_result = api.search.tweets(q='Trump',count = 100,tweet_mode='extended')

dfsr=json_normalize(search_result)
dfsr.to_pickle('dfsr.pkl')

dfst=json_normalize(dfsr.statuses.values[0])
df0=pd.DataFrame({'Value':dfst.loc[0]})

tjson=api.statuses.user_timeline(screen_name="realDonaldTrump",tweet_mode='extended')
dftrump=json_normalize(tjson)

tfollow=api.followers.ids(screen_name="realDonaldTrump")
dffol=json_normalize(tfollow)
dffol2=json_normalize(tfollow,'ids')
dffol2.to_pickle('dffol2.pkl')

dfst2=json_normalize(search_result,'statuses')

u0=api.users.lookup(user_id=dffol2.loc[0,0])
dfu0=json_normalize(u0)

mcuban=api.statuses.user_timeline(id=963973915256246272)

#import TextBlob
import pip

!pip install textblob
!python -m textblob.download_corpora

from textblob import TextBlob
#import nltk
#nltk.download()


tx = df.loc[0,'full_text']
blob = TextBlob(tx)
blob.tags
blob.sentences[0].words
blob.noun_phrases
blob.ngrams(3)
blob.correct()
blob.words[3].spellcheck()
blob.detect_language()
blob.translate(to= 'ar') 

verbs = list()
for word, tag in blob.tags:
  if tag == 'VB':
    verbs.append(word.lemmatize())

nouns = list()
for word, tag in blob.tags:
	if tag == 'NN':
		nouns.append(word.lemmatize())

blob.sentiment.polarity
blob.sentiment.subjectivity

#Create 2 arrays
polarity=[]
subj=[]

#Get polarity and sentiment for each row and put it in either polarity or sentiment 
for t in df.full_text:
    tx=TextBlob(t)
    polarity.append(tx.sentiment.polarity)
    subj.append(tx.sentiment.subjectivity)

#Put in dataframe polsubj which has a column of polarity values and a column of subjectivity values
polsubj = pd.DataFrame({'polarity': polarity,'subjectivity': subj})

#Plot the line graph
polsubj.plot(title='Polarity and Subjectivity')



import numpy as np
import pandas as pd


#!pip install twitter

from twitter import Twitter
from twitter import OAuth

from pandas.io.json import json_normalize


apikey='LmWo260maj5KsmP3wnigGiymR'
apisecretkey='wdFhGB3XV79csLvSI57R1OFavsXNntbdtmlJzy2spNdMIFbnxn'
accesstoken='4012083173-pdPffs50tApeBURWR9QQt22rlhEp0sdEaCFwBvR'
accesstokensecret='PVwr62zsdUF0QQpcRMkPDBLxJ4HhAG4Cjccy49GwPv8pK'

oauth = OAuth(accesstoken,accesstokensecret,apikey,apisecretkey)
api = Twitter(auth=oauth)

#####  new imports ##################
import pip

#!pip install wordcloud

from nltk.corpus import stopwords

import matplotlib.pyplot as plt
from wordcloud import WordCloud

stop =stopwords.words('english')

wordcloud = WordCloud().generate(df.full_text)
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()

wordcloud2 = WordCloud(background_color="white",stopwords=stop).generate(tx)
plt.imshow(wordcloud2)
plt.axis("off")
plt.show()

# Display the generated image:
tx2=df.full_text.str.cat(sep=' ')

wordcloud3 = WordCloud(stopwords=stop).generate(tx2)
plt.imshow(wordcloud3, interpolation='bilinear')
plt.axis("off")
plt.show()

stop.append('RT')
stop.append('co')
stop.append('https')
stop.append('amp')

wordcloud4 = WordCloud(background_color="white",stopwords=stop,max_words=1000).generate(tx2)
plt.imshow(wordcloud4, interpolation='bilinear')
plt.axis("off")
plt.show()


#http://www.cherryblossom.org/results/2011/2011cucb10m-m.htm

#http://www.cherryblossom.org/results/2010/2010cucb10m-m.htm

#http://www.cherryblossom.org/results/2009/09cucb-M.htm

import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup


res = requests.get('http://www.cherryblossom.org/results/2012/2012cucb10m-m.htm')
soup = BeautifulSoup(res.content,'lxml')
print(soup)

table = soup.find("pre").contents[0]
type(table)
table[0:5]
table[0:100]
tablerows=table.split('\r\n')
print(tablerows)
tablerows[0:5]

#example
x="the cat in the hat is fat"
y=x.split('at')

webdf=pd.DataFrame({'raw':tablerows})
rownumber=webdf.loc[webdf.raw.str.contains("=")]
type(rownumber)
rownumber=webdf.loc[webdf.raw.str.contains("=")].index.values[0]
dash=webdf.loc[rownumber,'raw']
type(dash)
print(dash)

sampletext=webdf.loc[rownumber+1,'raw']
colraw=webdf.loc[rownumber-1,'raw']

vv='batman'
print(list(vv))
da=np.array(list(dash))
print(da)
dashpos=np.argwhere(da==' ').flatten().tolist()
print(dashpos)

ww="I love dark chocolate mousse from Mara's"
wwa=np.array(list(ww))
wwl=np.split(wwa,[6,21])
print(wwl)
''.join(wwl[0])

webdf['rawa']=webdf.raw.apply(lambda x: np.array(list(x)))
webdf['rawal']=webdf.rawa.apply(lambda x: np.split(x, dashpos))

#Example: Splitting Lists into New Columns
expl=[['toyota','black'],['bmw','red'],['ford','grey'],['audi','white']]
dfexp = pd.DataFrame({'First':expl})
dfexp[['Second','Third']]=pd.DataFrame(dfexp.First.values.tolist())


newcol=list(range(len(dashpos)+1))

webdf[newcol] = pd.DataFrame(webdf.rawal.values.tolist(), index= webdf.index)
newrows=[rownumber-1]+list(range(rownumber+1,webdf.shape[0]))
webdf2=webdf.loc[newrows,newcol]

webdf3=webdf2.applymap(lambda r: ''.join(r))
webdf3=webdf3.applymap(lambda r: r.strip())

datadf = webdf3.iloc[1:]
datadf.columns = webdf3.iloc[0].tolist()
datadf=datadf.reset_index(drop=True)











