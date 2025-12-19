import nltk
text = "AA BB CC AA CC AA BD DD CC BB BB BB EE CE"
token = nltk.tokenize.word_tokenize(text)
frequency_distribution = nltk.FreqDist(token)
print(frequency_distribution.most_common(10)) #顯示前10個最常出現的字
frequency_distribution.plot(10) #畫出前10個最常出現的字