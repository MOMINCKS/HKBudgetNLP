import numpy as np
import string
import nltk

text = np.loadtxt('HKBudgetNLP/text.txt', encoding='utf-8', dtype='str', delimiter='\t')

### text normalization ###

# 1. lower case, remove punctuations and trimming
trans = string.punctuation
trans = trans.replace('-','')
trans = str.maketrans('','',trans)
for i in range(len(text)):
      text[i] = text[i].lower()
      text[i] = text[i].strip()
      text[i] = text[i].translate(trans)

# 2. tokenization
tokenized = []
for i in range(len(text)):
      tokenized.append(nltk.word_tokenize(text[i]))

# 3. lemmatization
def get_pos_tag(word):
      tag_dict = {"J": nltk.corpus.wordnet.ADJ,
            "N": nltk.corpus.wordnet.NOUN,
            "V": nltk.corpus.wordnet.VERB,
            "R": nltk.corpus.wordnet.ADV}
      tag = nltk.pos_tag([word])[0][1][0]
      return tag_dict.get(tag, nltk.corpus.wordnet.NOUN)

wnl = nltk.stem.WordNetLemmatizer()

for sents in tokenized:
      for i in range(len(sents)):
            sents[i] = wnl.lemmatize(sents[i], get_pos_tag(sents[i]))






