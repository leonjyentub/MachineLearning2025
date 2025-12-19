import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
count = CountVectorizer()
docs = np.array(['The sun is shining',
                 'The weather is sweet',
                 'The sun is shining, the weather is sweet, and one and one is two'])
bag = count.fit_transform(docs)
print(count.vocabulary_)
print(bag.toarray())


from sklearn.feature_extraction.text import TfidfTransformer
tfidf = TfidfTransformer()
np.set_printoptions(precision=2)
print(tfidf.fit_transform(count.fit_transform(docs)).toarray())

#使用這個詞彙集取得新原文的表示
temp = tfidf.transform(['The sun is nothing'])
print(temp.toarray())

temp = tfidf.transform(['The sun is big'])
print(temp.toarray())