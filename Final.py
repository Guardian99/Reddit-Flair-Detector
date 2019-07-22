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


df=pd.read_csv('taskfinal.csv')	

X_train, x_test, y_train, y_test = train_test_split(df.feature_combine,df.target, test_size=0.25, random_state = 7)
sgd = Pipeline([('vect', CountVectorizer()),('tfidf', TfidfTransformer()),('clf', SGDClassifier(loss='hinge', penalty='l2',alpha=0.0001, random_state=42, max_iter=100, tol=None))])
sgd.fit(X_train, y_train)
# y_pred = sgd.predict(X_test)


import pickle 
saved_model = pickle.dumps(sgd)  
sgd = pickle.loads(saved_model)

inputfromuser=int(input("Enter the number of posts you want to predict: "))

print("The types of posts are available are: ")
print("1.Top ")
print("2.Hot ")
print("3.Controversial ")
print("4.Rising ")
print("5.New ")

option=int(input("Choose your option: " ))


def detect_flair(id):

	submission = reddit.submission(id=id)
	data = {}
	data['title'] = submission.title
	data['url'] = submission.url
	data['combine'] = data['title'] + data['url']
	p=sgd.predict([data['combine']])
	return p[0]
reddit = praw.Reddit(client_id='qftDPmjb1sdKrg',client_secret='w5JbV-0irh1FaTOqRYeU2yZDeXc', user_agent='SC')
subreddit = reddit.subreddit('india')

if option==1:

	for submission in subreddit.top(limit=inputfromuser):
		print(detect_flair(submission.id))
		print("Cross Check:",'www.reddit.com'+submission.permalink)

elif option==2:

	for submission in subreddit.hot(limit=inputfromuser):
		print(detect_flair(submission.id))
		print("Cross Check:",'www.reddit.com'+submission.permalink)
elif option==3:

	for submission in subreddit.controversial(limit=inputfromuser):
		print(detect_flair(submission.id))
		print("Cross Check:",'www.reddit.com'+submission.permalink)
elif option==4:

	for submission in subreddit.rising(limit=inputfromuser):
		print(detect_flair(submission.id))
		print("Cross Check:",'www.reddit.com'+submission.permalink)

if option==5:

	for submission in subreddit.new(limit=inputfromuser):
		print(detect_flair(submission.id))
		print("Cross Check:",'www.reddit.com'+submission.permalink)