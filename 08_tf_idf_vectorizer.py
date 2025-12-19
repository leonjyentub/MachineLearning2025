import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = TfidfVectorizer()
docs = np.array(['The sun is shining', 
                 'The weather is sweet',
                 'The sun is shining, the weather is sweet, and one and one is two'])
tfidf_result = tfidf.fit_transform(docs)
print(tfidf_result.toarray())
print(tfidf.get_feature_names_out()) #詞彙裡的所有單字
print(tfidf.idf_) #詞彙裡的所有單字的idf
#使用這個詞彙集取得新原文的表示
temp = tfidf.transform(['The sun is nothing'])
print(temp.toarray())

temp = tfidf.transform(['The sun is big'])
print(temp.toarray())