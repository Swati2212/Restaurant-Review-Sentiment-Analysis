# Importing essential libraries
import pandas as pd
import pickle

# Loading the dataset
df = pd.read_csv('Restaurant_Reviews.tsv.txt', delimiter='\t', quoting=3)
df.head()

# Importing essential libraries for performing the NLP

import nltk
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# cleaning the reviews
corpus = []
for i in range(0, 1000):
    # Cleaning special character from the reviews
    review = re.sub(pattern='[^a-zA-Z]', repl=' ', string=df['Review'][i])

    # Converting the entire review into lower case
    review = review.lower()

    # Tokenizing the review by words
    review_words = review.split()

    # Removing the stop words
    review_words = [word for word in review_words if not word in set(stopwords.words('english'))]

    # Stemming the words
    ps = PorterStemmer()
    review = [ps.stem(word) for word in review_words]

    # Joining the stemmed words
    review = ' '.join(review)

    # Creating a corpus
    corpus.append(review)

# Creating the bag of model model
from sklearn.feature_extraction.text import CountVectorizer

cv = CountVectorizer(max_features=1500)
X = cv.fit_transform(corpus).toarray()
y = df.iloc[:, 1].values

# Creating the pickle file for the CountVectorize
pickle.dump(cv, open('cv-transform.pkl', 'wb'))

# Model Building
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Fitting Naive Bayes to the Training set
from sklearn.naive_bayes import MultinomialNB
classifier = MultinomialNB(alpha=0.2)
classifier.fit(X_train, y_train)

# Creating the pickle file for the Multinomial Naive Bayes model
filename = 'restaurant-review-sentiment-model.pkl'
pickle.dump(classifier, open(filename, 'wb'))
