import nltk

text = '''As we discussed a for-profit structure in order to further the mission, Elon wanted us to merge with Tesla or he wanted full control. Elon left OpenAI, saying there needed to be a relevant competitor to Google/DeepMind and that he was going to do it himself. He said he’d be supportive of us finding our own path.
In late 2017, we and Elon decided the next step for the mission was to create a for-profit entity. Elon wanted majority equity, initial board control, and to be CEO. In the middle of these discussions, he withheld funding. Reid Hoffman bridged the gap to cover salaries and operations.
We couldn’t agree to terms on a for-profit with Elon because we felt it was against the mission for any individual to have absolute control over OpenAI. He then suggested instead merging OpenAI into Tesla. In early February 2018, Elon forwarded us an email suggesting that OpenAI should “attach to Tesla as its cash cow”, commenting that it was “exactly right… Tesla is the only path that could even hope to hold a candle to Google. Even then, the probability of being a counterweight to Google is small. It just isn’t zero”.
Elon soon chose to leave OpenAI, saying that our probability of success was 0, and that he planned to build an AGI competitor within Tesla. When he left in late February 2018, he told our team he was supportive of us finding our own path to raising billions of dollars. In December 2018, Elon sent us an email saying “Even raising several hundred million won’t be enough. This needs billions per year immediately or forget it.”
''' #執行
sentences = nltk.sent_tokenize(text) #sentences is list
for sentence in sentences: #用迴圈輸出所有句子
    print(sentence)
