import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.porter import *
stemmer = PorterStemmer()

nltk.download('stopwords')
nltk.download('punkt')   
nltk.download('words')
nltk.download('wordnet')
nltk.download('stopwords')


def read_pos_and_neg_words(positive_filepath, negative_filepath):
    with open(positive_filepath, 'r',) as positive_file:
        positive_words = positive_file.read().splitlines()
    with open(negative_filepath, 'r',encoding='latin-1') as negative_file:
        negative_words = negative_file.read().splitlines()
    return positive_words, negative_words


positive_words, negative_words = read_pos_and_neg_words("./data/positive-words.txt", "./data/negative-words.txt")


# teachers
def count_pos(review):
    return sum(word in review for word in positive_words)

def count_neg(review):
    return sum(word in review for word in negative_words)

def count_word_no(review):
    return (1 if "no" in review.lower().split() else 0)

def count_pronoun(review):
    return sum(word in ["i", "me", "my", "you", "your"] for word in review.lower().split())

def count_exclaimation(review):
    return (1 if "!" in review else 0)

def count_chars(text):
    return len(text)

def count_words(text):
    return len(text.split())

def count_capital_chars(text):
    count=0
    for i in text:
        if i.isupper():
            count+=1
    return count

def count_capital_words(text):
    return sum(map(str.isupper,text.split()))
    
def count_unique_words(text):
    return len(set(text.split()))

def count_htags(text):
    return len(re.findall(r'(#w[A-Za-z0-9]*)', text))

def count_mentions(text):
    x = re.findall(r'(@w[A-Za-z0-9]*)', text)
    return len(x)

def count_stopwords(text):
    stop_words = set(stopwords.words('english'))  
    word_tokens = word_tokenize(text)
    stopwords_x = [w for w in word_tokens if w in stop_words]
    return len(stopwords_x)


contraction_dict = {"ain't": "is not", "aren't": "are not","can't": "cannot", "'cause": "because", "could've": "could have", "couldn't": "could not", "didn't": "did not",  "doesn't": "does not", "don't": "do not", "hadn't": "had not", "hasn't": "has not", "haven't": "have not", "he'd": "he would","he'll": "he will", "he's": "he is", "how'd": "how did", "how'd'y": "how do you", "how'll": "how will", "how's": "how is",  "I'd": "I would", "I'd've": "I would have", "I'll": "I will", "I'll've": "I will have","I'm": "I am", "I've": "I have", "i'd": "i would", "i'd've": "i would have", "i'll": "i will",  "i'll've": "i will have","i'm": "i am", "i've": "i have", "isn't": "is not", "it'd": "it would", "it'd've": "it would have", "it'll": "it will", "it'll've": "it will have","it's": "it is", "let's": "let us", "ma'am": "madam", "mayn't": "may not", "might've": "might have","mightn't": "might not","mightn't've": "might not have", "must've": "must have", "mustn't": "must not", "mustn't've": "must not have", "needn't": "need not", "needn't've": "need not have","o'clock": "of the clock", "oughtn't": "ought not", "oughtn't've": "ought not have", "shan't": "shall not", "sha'n't": "shall not", "shan't've": "shall not have", "she'd": "she would", "she'd've": "she would have", "she'll": "she will", "she'll've": "she will have", "she's": "she is", "should've": "should have", "shouldn't": "should not", "shouldn't've": "should not have", "so've": "so have","so's": "so as", "this's": "this is","that'd": "that would", "that'd've": "that would have", "that's": "that is", "there'd": "there would", "there'd've": "there would have", "there's": "there is", "here's": "here is","they'd": "they would", "they'd've": "they would have", "they'll": "they will", "they'll've": "they will have", "they're": "they are", "they've": "they have", "to've": "to have", "wasn't": "was not", "we'd": "we would", "we'd've": "we would have", "we'll": "we will", "we'll've": "we will have", "we're": "we are", "we've": "we have", "weren't": "were not", "what'll": "what will", "what'll've": "what will have", "what're": "what are",  "what's": "what is", "what've": "what have", "when's": "when is", "when've": "when have", "where'd": "where did", "where's": "where is", "where've": "where have", "who'll": "who will", "who'll've": "who will have", "who's": "who is", "who've": "who have", "why's": "why is", "why've": "why have", "will've": "will have", "won't": "will not", "won't've": "will not have", "would've": "would have", "wouldn't": "would not", "wouldn't've": "would not have", "y'all": "you all", "y'all'd": "you all would","y'all'd've": "you all would have","y'all're": "you all are","y'all've": "you all have","you'd": "you would", "you'd've": "you would have", "you'll": "you will", "you'll've": "you will have", "you're": "you are", "you've": "you have"}
stop_words = set(stopwords.words('english'))

def _get_contractions(contraction_dict):
    contraction_re = re.compile('(%s)' % '|'.join(contraction_dict.keys()))
    return contraction_dict, contraction_re

contractions, contractions_re = _get_contractions(contraction_dict)

def replace_contractions(text):
    def replace(match):
        return contractions[match.group(0)]
    return contractions_re.sub(replace, text)


def filter_non_english(text):
    from nltk.corpus import words
    english_word_set = set(words.words())
    words = text.split()
    english_words = [word for word in words if word.lower() in english_word_set]
    return ' '.join(english_words)

def remove_stop_words_and_stem(text):
    word_tokens = word_tokenize(text)
    filtered_sentence = [stemmer.stem(word) for word in word_tokens]
    return " ".join(filtered_sentence)

                                
def preprocess_sentence(sentence):
    
    sentence = replace_contractions(sentence)
    sentence = sentence.lower()
    sentence = sentence.translate(str.maketrans('', '', string.punctuation))
    sentence = re.sub(r'\w*\d\w*', '', sentence)
    sentence = remove_stop_words_and_stem(sentence)
    return sentence