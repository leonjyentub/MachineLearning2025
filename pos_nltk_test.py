import nltk
from nltk import word_tokenize, pos_tag

# Sample sentence
sentence = "The quick brown fox jumps over the lazy dog."

# Tokenize the sentence
tokens = word_tokenize(sentence)

# Perform POS tagging
pos_tags = pos_tag(tokens)

# Display the POS tags
for tag in pos_tags:
    print(f'{tag[0]}: {tag[1]}') 