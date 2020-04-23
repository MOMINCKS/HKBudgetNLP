### HKBudgetNLP

This text summarization aims to extract the most important sentences and words from the Hong Kong Financial Budget 2020-21, which has 84 sentences and 822 different words (including a few kinds of punctuations), using TFDIF score. After this algorithm, there will be 5 sentences left as the summary for the whole text and and 81 important words will be selected.

### Process:

#### 1. Data cleansing:

The Budget Report can be downloaded here:
    
    https://www.budget.gov.hk/2020/eng/pdf/2020-21%20Media%20Sheets.pdf

Data is cleaned into the test.txt file. It's time-saving to clean it manually rather than to use pdf2txt modules instead.

#### 2. Text Normalization

a. removing punctuations and trimming

b. tokenization and lemmatization

The list of part-of-speech tags used in NLTK used for lemmatization:

        https://www.ling.upenn.edu/courses/Fall_2003/ling001/penn_treebank_pos.html

#### 3. tfidf vectorization

Caluculate the tfidf score for each of the words in the whole text. 

#### 4. tfidf score mean masking

Create masking for tfidf score by their mean.

#### 5. text summarization

For the final output please view *result_text*.

#### 6. important words analysis

For the final output please view *important_words*.



