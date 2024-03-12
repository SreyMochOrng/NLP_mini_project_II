import math
import tensorflow as tf
from flask import Flask, request, render_template
from utils import count_pos, count_neg, count_chars, count_words, count_capital_chars, count_stopwords, count_unique_words, count_word_no, count_pronoun, count_exclaimation, preprocess_sentence
import numpy as np
import pandas as pd
import pickle

app = Flask(__name__, static_url_path='/static')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict/', methods=['POST'])
def predict():
    text = request.form['textInput']

    label,pos_word_count,neg_word_count,word_no,exclaimation = text_classify(text=text)
    
    return render_template('index.html', label = label, is_contain_no = word_no, is_contain_exclamation=exclaimation, positive_word_count= pos_word_count, negative_word_count= neg_word_count, input_text=text)


model = tf.keras.models.load_model('models/model')

with open('models/encoder', 'rb') as file:
    encoder = pickle.load(file)

def text_classify(text):
    
    data = []
    data.append(text)


    numerical_columns = ["positive_word_count", 
                     "negative_word_count", 
                     "char_count", 
                     "word_count", 
                     "capital_char_count", 
                     "stopword_count", 
                     "unique_word_count", 
                     "pronounce_count"]
    

    test_data = pd.DataFrame(data, columns=['reviews'])

    test_data['positive_word_count'] = test_data['reviews'].apply(count_pos)
    test_data['negative_word_count'] = test_data['reviews'].apply(count_neg)
    test_data['char_count'] = test_data["reviews"].apply(count_chars)
    test_data['word_count'] = test_data["reviews"].apply(count_words)
    test_data['capital_char_count'] = test_data["reviews"].apply(count_capital_chars)
    test_data['stopword_count'] = test_data["reviews"].apply(count_stopwords)
    test_data['unique_word_count'] = test_data["reviews"].apply(count_unique_words)
    test_data['word_no'] = test_data['reviews'].apply(count_word_no)
    test_data['pronounce_count'] = test_data['reviews'].apply(count_pronoun)
    test_data['exclaimation'] = test_data['reviews'].apply(count_exclaimation)

    test_data['cleaned_reviews'] = test_data['reviews'].apply(preprocess_sentence)
    enc_data = pd.DataFrame(encoder.transform( 
    test_data[['word_no', 'exclaimation']]).toarray()) 

    # Merge with main 
    test_data = test_data.join(enc_data)
    pos_word_count = test_data['positive_word_count'].sum()
    neg_word_count = test_data['negative_word_count'].sum()
    word_no = test_data['word_no'].sum() == 1
    exclaimation = test_data['exclaimation'].sum() == 1
    print(word_no, exclaimation)
    test_data.drop(columns=["word_no", "exclaimation"], inplace=True)

    for header in numerical_columns:
        test_data[header] = test_data[header].apply(lambda x: math.log(x+1))
        
    y_predicted = model.predict([test_data.cleaned_reviews,test_data.drop(columns=['cleaned_reviews', 'reviews'])])

    print('y_predicted', y_predicted)
    y_predict_norm = np.where(y_predicted>=0.5,1,0)
    if(y_predict_norm[0][0] == 1):
        label = "Positive"
    else:
        label = "Negative"
    
    return label,pos_word_count,neg_word_count,word_no,exclaimation

if __name__ == "__main__":
    app.run(debug=True)
