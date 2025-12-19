import nltk
'''
nltk.download("punkt")
nltk.download("averaged_perceptron_tagger")
nltk.download("wordnet")
nltk.download("stopwords")
nltk.download("maxent_ne_chunker")
nltk.download("words")
'''
print(nltk.corpus.wordnet.ADJ)
text = '''ChatGPT-maker OpenAI is looking to fuse its artificial intelligence (AI) systems into the bodies of humanoid robots as part of a new deal with robotics startup Figure.
Sunnyvale, California-based Figure announced the partnership on Thursday along with $675 million (€623.5 million) in venture capital funding from a group that includes Amazon founder Jeff Bezos as well as Microsoft, chipmaker Nvidia and the startup-funding divisions of Amazon, Intel and OpenAI.
Figure is less than two years old and doesn't have a commercial product but is persuading influential tech industry backers to support its vision of shipping billions of human-like robots to the world's workplaces and homes.
“If we can just get humanoids to do work that humans are not wanting to do because there’s a shortfall of humans, we can sell millions of humanoids, billions maybe,” Figure CEO Brett Adcock told The Associated Press last year.
ChatGPT maker OpenAI teams up with children’s safety non-profit to create AI guidelines
For OpenAI, which dabbled in robotics research before pivoting to a focus on the AI large language models that power ChatGPT, the partnership will "open up new possibilities for how robots can help in everyday life,” said Peter Welinder, the San Francisco company's vice president of product and partnerships, in a written statement.
The financial terms of the deal between Figure and OpenAI weren't disclosed. The collaboration will have OpenAI building specialized AI models for Figure’s humanoid robots, likely based on OpenAI's existing technology such as GPT language models, the image-generator DALL-E and the new video-generator Sora.
That will help “accelerate Figure’s commercial timeline” by enabling its robots to “process and reason from language,” according to Figure's announcement. The company announced in January an agreement with BMW to put its robots to work at a car plant in Spartanburg, South Carolina, but hadn't yet determined exactly how or when they would be used.
OpenAI claims New York Times 'hacked' ChatGPT in court filing
Robotics experts differ on the usefulness of robots shaped in human form. Most robots employed in factory and warehouse tasks might have some animal-like features — a robotic arm, finger-like grippers or even legs — but aren't truly humanoid. That's in part because it's taken decades for robotics engineers to develop effective robotic legs and arms.
OpenAI CEO Sam Altman hinted at a renewed interest in robotics in a podcast hosted by Microsoft co-founder Bill Gates and released early this year in which Altman said the company was starting to invest in promising robotics hardware platforms after having earlier abandoned its own research.
“We started robots too early and so we had to put that project on hold," Altman told Gates, noting that “we were dealing with bad simulators and breaking tendons" that were distracting from the company's other work.
“We realised more and more over time that what we really first needed was intelligence and cognition and then we could figure out how we could adapt it to physicality,” he said.'''
# Sentences
sentences = nltk.sent_tokenize(text)
# Tokenize
tokens = [nltk.tokenize.word_tokenize(sent) for sent in sentences]
tokens = [[token.lower() for token in tokens_line if token.isalpha()] for tokens_line in tokens]
print(f'token: {tokens}')
# POS
pos = [nltk.pos_tag(token) for token in tokens]
print(f'pos: {pos}')
# Lemmatization
wordnet_pos = [] #記錄詞性，加強後面的lemmatize
for p in pos:
    for word, tag in p:
        if tag.startswith('J'):
            wordnet_pos.append(nltk.corpus.wordnet.ADJ)
        elif tag.startswith('V'):
            wordnet_pos.append(nltk.corpus.wordnet.VERB)
        elif tag.startswith('N'):
            wordnet_pos.append(nltk.corpus.wordnet.NOUN)
        elif tag.startswith('R'):
            wordnet_pos.append(nltk.corpus.wordnet.ADV)
        else:
            wordnet_pos.append(nltk.corpus.wordnet.NOUN)

# Lemmatizer
lemmatizer = nltk.stem.wordnet.WordNetLemmatizer()
tokens = [lemmatizer.lemmatize(p[n][0], pos=wordnet_pos[n]) for p in pos for n in range(len(p))]

# Stopwords 根據nltk.corpus.stopwords.words("english")獲取所有停用字，用來把不需要的token刪除
nltk_stopwords = nltk.corpus.stopwords.words("english")
tokens = [token for token in tokens if token not in nltk_stopwords]
print(f'remove nltk_stopwords token: {tokens}')
# NER
ne_chunked_sents = [nltk.ne_chunk(tag) for tag in pos]
named_entities = []

for ne_tagged_sentence in ne_chunked_sents:
    for tagged_tree in ne_tagged_sentence:
        if hasattr(tagged_tree, 'label'):
            entity_name = ' '.join(c[0] for c in tagged_tree.leaves())
            entity_type = tagged_tree.label()
            named_entities.append((entity_name, entity_type))
            named_entities = list(set(named_entities))

print(f'named_entities: {named_entities}')

from nltk import FreqDist

frequency_distribution = FreqDist(tokens)
print(frequency_distribution.most_common(10))
import matplotlib.pyplot as plt
plt.tight_layout()
frequency_distribution.plot(10)


