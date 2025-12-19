import spacy
 
# Load the English language model
nlp = spacy.load("en_core_web_sm")
 
# Sample text
text = 'The quick brown fox jumps over the lazy dog.'
# Process the text with SpaCy
doc = nlp(text)
 
# Display the PoS tagged result
print("Original Text: ", text)
print("PoS Tagging Result:")
for token in doc:
    print(f"{token.text}: {token.pos_}")