# import tensorflow as tf
# from flask import Flask, request, render_template, jsonify
# from pydantic import BaseModel
# from preprocessing import feature_engineering, preprocessing_text
# from keras.layers.experimental.preprocessing import TextVectorization
# import numpy as np
# import pickle

# app = Flask(__name__, static_url_path='/static')

# class TextAnalytic(BaseModel):
#     text: str

# # Load the vectorizer
# vectorizer_config = pickle.load(open('vectorizer.pkl', 'rb'))
# vectorizer = TextVectorization.from_config(vectorizer_config['config'])
# vectorizer.set_weights(vectorizer_config['weights'])

# @app.route('/')
# def home():
#     return render_template('index.html')

# @app.route('/predict/', methods=['POST'])
# def predict():
#     text = request.form['textInput']
#     label, pronount_count, is_contain_no, is_contain_exclamation, positive_word_count, negative_word_count= make_prediction(text)

#     return render_template('index.html', label = label, pronount_count = pronount_count, is_contain_no = is_contain_no, is_contain_exclamation=is_contain_exclamation, positive_word_count= positive_word_count, negative_word_count= negative_word_count, input_text=text)

# LOADED_MODEL = tf.keras.models.load_model('sentimental.h5')

# def text_vectorize(text):
#     cleaned_text = preprocessing_text(text)
#     from_file = pickle.load(open("vectorizer.pkl", "rb"))
#     vectorizer = TextVectorization.from_config(from_file['config'])
#     vectorizer.set_weights(from_file['weights'])
#     text = vectorizer([cleaned_text])
#     return text

# def make_prediction(text):
#     X_text = text_vectorize(text)
#     X_numerical = feature_engineering(text)

#     pronoun_count = int(X_numerical[0][5])
#     positive_word_count = int(X_numerical[0][0])
#     negative_word_count = int(X_numerical[0][1])
#     is_contain_no = bool(X_numerical[0][2])
#     is_contain_exlamaition = bool(X_numerical[0][6])

#     # X_numerical = np.pad(X_numerical, ((0, 0), (0, 2)), 'constant', constant_values=(0, 0))

#     result = LOADED_MODEL.predict([X_text, X_numerical])
#     result = np.round(result[0][0], 5)

#     if(result > 0.5):
#         label = 'Positive'
#     else:
#         label = 'Negative'

#     return label, pronoun_count, is_contain_no, is_contain_exlamaition, positive_word_count, negative_word_count


# if __name__ == "__main__":
#     app.run(debug=True)

