import numpy as np
import string
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer

###
# This text summarization aims to extract the most important sentences and words from 
# the Hong Kong Financial Budget 2020-21, which has 84 sentences and 822 different words
# (including a few kinds of punctuations), using TFIDF score. After this algorithm,
# there will be 5 sentences left as the summary for the whole text and 81 important
# words will be selected.
###

# 0. import

text = np.loadtxt('HKBudgetNLP/text.txt', encoding='utf-8', dtype='str', delimiter='\t')

# 1. removing punctuations and trimming

def norm(text):
      trans = string.punctuation
      trans = trans.replace('-','') + '‘’“”'
      trans = str.maketrans('','',trans)
      for i in range(len(text)):
            text[i] = text[i].strip()
            text[i] = text[i].translate(trans)
      return text

text = norm(text)

# 2. tokenization and lemmatization

def get_pos_tag(word): # determine parts of speech
      tag_dict = {"J": nltk.corpus.wordnet.ADJ,
            "N": nltk.corpus.wordnet.NOUN,
            "V": nltk.corpus.wordnet.VERB,
            "R": nltk.corpus.wordnet.ADV}
      tag = nltk.pos_tag([word])[0][1][0]
      return tag_dict.get(tag, nltk.corpus.wordnet.NOUN)

def tokenize(text):
            tokenized = []
            for i in range(len(text)):
                  tokenized.append(nltk.word_tokenize(text[i]))
            return tokenized

def lemmatize(text):
      wnl = nltk.stem.WordNetLemmatizer()
      for sents in text:
            for words in range(len(sents)):
                  sents[words] = wnl.lemmatize(sents[words], get_pos_tag(sents[words]))
            sents[0] = sents[0].lower()
      return text

text = tokenize(text)
text = lemmatize(text)

# 3. tfidf vectorization

def dummy(x):
      return x

def vectorize(text):
      tfidf = TfidfVectorizer(analyzer=dummy,stop_words='english')
      text_sparse = tfidf.fit_transform(text)
      return text_sparse, tfidf

text_sparse, tfidf = vectorize(text)

# 4. tfidf score mean masking

def mean_mask_sents(text):
      mean_thresold = 0.008
      mean_score = text.mean(axis=1)
      mean_score_mask_sents = mean_score>=mean_thresold
      return mean_score_mask_sents

mean_score_mask_sents = mean_mask_sents(text_sparse)

def mean_mask_words(text):
      mean_thresold = 0.01
      mean_score = text.mean(axis=-2)
      mean_score_mask_sents = mean_score>=mean_thresold
      return mean_score_mask_sents

mean_score_mask_sents_words = mean_mask_words(text_sparse)

# 5. text summarization

def sents_filtering(text,mask):
      output = []
      for i in range(len(mask)):
            if mask[i]:
                  output.append(text[i])
      return output

def rejoin_tokens(text):
      for sents in range(len(text)):
            text[sents] = ' '.join(text[sents])
      return text

text = sents_filtering(text,mean_score_mask_sents)
result_text = rejoin_tokens(text)

# 6. important words analysis

def important_words(text,mask):
      list_of_words = np.array(text)
      mask = np.squeeze(np.asarray(mask))
      words = list_of_words[mask]
      return words

important_words = important_words(tfidf.get_feature_names(),mean_score_mask_sents_words)
