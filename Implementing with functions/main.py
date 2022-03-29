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

#dic
verified_words = dict()
verified_words_size = 0
unverified_words = dict()
unverified_words_size = 0
verified_comments = 0
unverified_comments = 0

#loading stopwords
persianStopWords = open("stopwords-fa.txt", "r", encoding="utf-8").read()

#counting verified and unverified words
for index,row in train.iterrows():
    if type(row["comment"]) is str:
        nopunc = [char for char in row["comment"] if (char not in string.punctuation) and (char not in persianStopWords)]
        if row["verification_status"] == 0:
            verified_comments += 1
            for word in nopunc:
                verified_words[word] = verified_words.get(word, 0) + 1
                verified_words_size += 1
        else:
            unverified_comments += 1
            for word in nopunc:
                unverified_words[word] = unverified_words.get(word, 0) + 1
                unverified_words_size += 1
#naive bayse using above code
def check_verification(row):

    if type(row["comment"]) is str:
        nopunc = [char for char in row["comment"] if (char not in string.punctuation)  and (char not in persianStopWords)]
        p_verified = verified_comments / (verified_comments + unverified_comments)
        p_unverified = unverified_comments / (verified_comments + unverified_comments)
        for word in nopunc:
            p_verified *= ((verified_words.get(word, 0) + 1) / verified_words_size)
            p_unverified *= ((unverified_words.get(word, 0) + 1) / unverified_words_size)
        if p_verified > p_unverified:
            return 0
        else:
            return 1
    else:
        return 1

test['verification_status'] = test.apply(check_verification, axis = 1)
output = test.drop(["title", "comment", "rate"], axis=1)
output.to_csv('ansedit1.csv', index=False)