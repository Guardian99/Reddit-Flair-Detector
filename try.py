# ====================================
import math
import praw
import pprint
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
import warnings
warnings.filterwarnings("ignore")
# ====================================


reddit = praw.Reddit(client_id='qftDPmjb1sdKrg',client_secret='w5JbV-0irh1FaTOqRYeU2yZDeXc', user_agent='SC')
subreddit = reddit.subreddit('india')
target=['AskIndia','Non-Political','[R]eddiquette','Scheduled','Photography','Science/Technology','Politics','Business/Finance','Policy/Economy','Sports','Food']
dataframe={"target":[],"Engagement": [], "Upvotes":[],  "title":[],'url':[]}


# try:
# 	for target in target:

# 		temp1=subreddit.search(target,limit=150)

# 		for i in temp1:
# 			dataframe['target'].append(target)			
# 			dataframe['Upvotes'].append(i.score)
# 			dataframe['Engagement'].append(i.num_comments)
# 			dataframe['title'].append(i.title)
# 			dataframe['url'].append(i.url)

			
# except KeyError:
# 	print('Error')
		
# =================================================
		# Feature Engineering
# df=pd.DataFrame(dataframe)
df=pd.read_csv('taskfinal.csv')	
# df=df[(df != 0).all(1)]
# df['Ratio']=((df.Engagement/df.Upvotes)+(df.Upvotes/df.Engagement))/2
# df['Ratio']=df['Ratio'].astype(int)
# combinations1 = df["title"] + df["url"]
# df = df.assign(feature_combine = combinations1)
# df.to_csv('taskfinal.csv')
# ====================================

# debugging

# print("Top Posts")
# post=subreddit.top(limit=5)
# for i in post:
# 	print(i.title)

# print("Hot Posts")
# post=subreddit.hot(limit=5)
# for i in post:
# 	print(i.title)

# print("controversial Posts")
# post=subreddit.controversial(limit=5)
# for i in post:
# 	print(i.title)

# print("new Posts")
# post=subreddit.new(limit=5)
# for i in post:
# 	print(i.title)

# print(df.shape)
# print(df.info())
# print(df.head(5))
# print(df.info())
# ===================================

def linear_svm(X_train, X_test, y_train, y_test):
  
  

  sgd = Pipeline([('vect', CountVectorizer()),('tfidf', TfidfTransformer()),('clf', SGDClassifier(loss='hinge', penalty='l2',alpha=0.0001, random_state=42, max_iter=100, tol=None))])
  sgd.fit(X_train, y_train)
  y_pred = sgd.predict(X_test)
  print('accuracy %s' % accuracy_score(y_pred, y_test))
  # print(classification_report(y_test, y_pred,target_names=target))

def randomforest(X_train, X_test, y_train, y_test):
  
  
  
  ranfor = Pipeline([('vect', CountVectorizer()),('tfidf', TfidfTransformer()),('clf', RandomForestClassifier(n_estimators = 1000, random_state = 42))])
  ranfor.fit(X_train, y_train)

  y_pred = ranfor.predict(X_test)

  print('accuracy %s' % accuracy_score(y_pred, y_test))
  # print(classification_report(y_test, y_pred,target_names=target))

# ====================================
def train_test(X,y):
 
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state = 42)

  
  print("Results of Linear Support Vector Machine")
  linear_svm(X_train, X_test, y_train, y_test)
  
  print("Results of Random Forest")
  randomforest(X_train, X_test, y_train, y_test)

  


print("Flair Detection using Combine as Feature")
train_test(df.feature_combine,df.target)
print('\n')
print("Flair Detection using Title as Feature")
train_test(df.title,df.target)
print('\n')
print("Flair Detection using URL as Feature")
train_test(df.url,df.target)

# ===============================================

# For plotting

# def plot_cat(cat_var):
#   sns.barplot(x=cat_var, y='target', data=df)
#   plt.show()


# plot_cat('Ratio')
# plot_cat('Engagement')
# plot_cat('Upvotes')
# ================================================
