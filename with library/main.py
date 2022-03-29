# Import libraries
import numpy as np
import pandas as pd
import string
#loading files
train = pd.read_csv("train.csv", encoding="utf8")
test = pd.read_csv("test.csv", encoding="utf8")

# filling na nan with 0
train = train.replace(np.nan, 0)
test = test.replace(np.nan, 0)
#creating full vocab
vocab =pd.concat([train,test],sort=True)


# removing Punctuations
# removing Stop words
def process_text(text):
    persianStopWords = open("stopwords-fa.txt", "r", encoding="utf-8").read()
    text=str(text) #there are sum numbers and we should chnage them to string
    nopunc = [char for char in text if char not in string.punctuation]
    nopunc = ''.join(nopunc)
    clean_words = [word for word in nopunc.split() if word not in persianStopWords]
    return clean_words

# Convert a collection of text documents to a matrix
from sklearn.feature_extraction.text import CountVectorizer
# vocab bow doros kardan
vocab_bow=CountVectorizer(analyzer=process_text).fit_transform(vocab['comment'])
#160k aval ro midim be train,20k baghimande ke tag nadaran be test
train_bow= vocab_bow[:vocab.shape[0] - 20000 , :]
test_bow= vocab_bow[vocab.shape[0] - 20000:, :]

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(train_bow, train['verification_status'],test_size = 0.20,random_state = 0)

#create naive bayse
from sklearn.naive_bayes import MultinomialNB
classifier=MultinomialNB().fit(X_train, y_train)

#predict
#print(classifier.predict(X_train))
#print(y_train.values)


#Evaluate the model on the training data set
from sklearn.metrics import classification_report,confusion_matrix, accuracy_score
pred = classifier.predict(X_test)
print(classification_report(y_test ,pred ))
print('Confusion Matrix: \n',confusion_matrix(y_test,pred))
print()
print('Accuracy: ', accuracy_score(y_test,pred))

#test finalyy
#test_bow=CountVectorizer(analyzer=process_text).transform(test['comment'])
print(classifier.predict(test_bow))
prediction=classifier.predict(test_bow)
print(classifier.predict(test_bow).shape)
#output
output= test.assign(verification_status = prediction)
output=output.drop(['title','comment','rate'],axis=1)
output.to_csv('output.csv',index=False)
