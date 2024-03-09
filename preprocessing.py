import numpy as np

import re
import string

import nltk
from nltk import pos_tag
from nltk.corpus import stopwords
nltk.download('wordnet')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('stopwords')

translator = str.maketrans("","",string.punctuation)
lemmatizer = nltk.stem.wordnet.WordNetLemmatizer()
STOP_WORDS = stopwords.words('english')


# create removeNewline function
def removeNewline(content):
    return re.sub("\n", "", content)

# convert to lowercase
def toLower(content):
    return content.lower()

# remove punctuation
def removePunctuation(content):
    return content.translate(translator)

# word lemmatization
def wordLemmatization(content):
    words = nltk.word_tokenize(content)
    tagged_words = pos_tag(words)
    lemma = [lemmatizer.lemmatize(word, "v") if pos in ['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ'] else word for word, pos in tagged_words]
    return " ".join(lemma)

# remove single character
def removeSingleChar(content):
    return re.sub(r'\b\w{1}\b', '', content)

# remove pronounce
def removePronouns(content):
    words = nltk.word_tokenize(content)
    tagged_words = pos_tag(words)
    non_pronounces = [word for word, pos in tagged_words if pos not in ['PRP', 'PRP$', 'WP', 'WP$']]
    return " ".join(non_pronounces)

# remove stopwords
def removeStopwords(content):
    words = []
    words = content.split(" ")
    all_words = set(words)
    common_words = set(words).intersection(set(STOP_WORDS))
    uncommon_words = list(all_words - common_words)
    return " ".join(uncommon_words)

# remove common
def removeCommon(content):
    with open("./data/common-word.txt", "r") as f:
        temp_common_words = f.readlines()
        f.close()
    common_words = [removeNewline(word) for word in temp_common_words]
    words = set(content.split(" "))
    common_words = set(common_words)
    found_common_words = list(common_words.intersection(words))
    uncommon_words = [word for word in content.split(" ") if word not in found_common_words]
    return " ".join(uncommon_words)

# count positive words
def pos_count(content):
    with open("./data/positive-reviews.txt") as f:
        temp_pos_words = f.readlines()
        f.close()
    postitive_words = [removeNewline(word) for word in temp_pos_words]
    return len([word for word in content.split() if word in postitive_words])

# count negative words
def neg_count(content):
    with open("./data/negative-reviews.txt") as f:
        temp_neg_words = f.readlines()
        f.close()
    negative_words = [removeNewline(word) for word in temp_neg_words]
    return len([word for word in content.split() if word in negative_words])

# check if contain 'no'
def isContainNo(content):
    return 1 if 'no' in content.split() else 0

# check if contain '!'
def isContainExclamation(content):
    return 1 if '!' in content.split() else 0

# check if contain pronouns
def pron_count(content):
    return len([word for word in content.split() if word in ['i', 'me', 'my', 'you', 'your']])

# get content length
def getLength(content):
    return np.log((len(content.split()))+1)

# preprocessing
def preprocessing_text(content):
    content = toLower(content)
    content = removePunctuation(content)
    content = wordLemmatization(content)
    content = removePronouns(content)
    content = removeCommon(content)
    return content

# feature engineering
def feature_engineering(content):
    content = toLower(content)
    content = removeCommon(content)
    return np.array([[
        pos_count(content),
        neg_count(content),
        isContainNo(content),
        pron_count(content),
        isContainExclamation(content),
        getLength(content)
    ]])