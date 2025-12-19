from textblob import TextBlob

text = '''Basically there's a family where a little boy (Jake) thinks there's a zombie in his closet & his parents are fighting all the time.<br /><br />This movie is slower than a soap opera... and suddenly, Jake decides to become Rambo and kill the zombie.<br /><br />OK, first of all when you're going to make a film you must Decide if its a thriller or a drama! As a drama the movie is watchable. Parents are divorcing & arguing like in real life. And then we have Jake with his closet which totally ruins all the film! I expected to see a BOOGEYMAN similar movie, and instead i watched a drama with some meaningless thriller spots.<br /><br />3 out of 10 just for the well playing parents & descent dialogs. As for the shots with Jake: just ignore them.'''

blob = TextBlob(text)
print('blob.words:', blob.words) #所有單詞
print('blob.word_counts:', blob.word_counts) #出現次數
print('blob.tags:', blob.tags) #詞性
print(blob.noun_phrases)
for sentence in blob.sentences:
    if sentence.sentiment.polarity > 0.3:
        print("Positive")
    elif sentence.sentiment.polarity < -0.3:
        print("Negative")
    else:
        print("Neutral")