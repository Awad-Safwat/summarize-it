from urllib import request
from flask import Flask, request
import numpy as np
import pandas as pd 
import re
import tensorflow
from tensorflow.keras.layers import Input, LSTM, Embedding, Dense, Concatenate, TimeDistributed
from tensorflow.keras.models import Model
from bs4 import BeautifulSoup
from keras.preprocessing.text import Tokenizer 
from keras.preprocessing.sequence import pad_sequences
from nltk.corpus import stopwords
import pickle
import nltk
nltk.download('stopwords')
import warnings
pd.set_option("display.max_colwidth", 200)
warnings.filterwarnings("ignore")
from keras.preprocessing.text import Tokenizer 
from keras.preprocessing.sequence import pad_sequences
from attention import AttentionLayer
import nltk
import os


#http://127.0.0.1:5000/api?query=
app = Flask(__name__)

x_tr_data = pd.read_csv('F:\AOU\summarizer\\flask_app\\x_df.csv')

my_list= x_tr_data['x_list'].tolist()


contraction_mapping = {"ain't": "is not", "aren't": "are not","can't": "cannot", "'cause": "because", "could've": "could have", "couldn't": "could not",
                           "didn't": "did not",  "doesn't": "does not", "don't": "do not", "hadn't": "had not", "hasn't": "has not", "haven't": "have not",
                           "he'd": "he would","he'll": "he will", "he's": "he is", "how'd": "how did", "how'd'y": "how do you", "how'll": "how will", "how's": "how is",
                           "I'd": "I would", "I'd've": "I would have", "I'll": "I will", "I'll've": "I will have","I'm": "I am", "I've": "I have", "i'd": "i would",
                           "i'd've": "i would have", "i'll": "i will",  "i'll've": "i will have","i'm": "i am", "i've": "i have", "isn't": "is not", "it'd": "it would",
                           "it'd've": "it would have", "it'll": "it will", "it'll've": "it will have","it's": "it is", "let's": "let us", "ma'am": "madam",
                           "mayn't": "may not", "might've": "might have","mightn't": "might not","mightn't've": "might not have", "must've": "must have",
                           "mustn't": "must not", "mustn't've": "must not have", "needn't": "need not", "needn't've": "need not have","o'clock": "of the clock",
                           "oughtn't": "ought not", "oughtn't've": "ought not have", "shan't": "shall not", "sha'n't": "shall not", "shan't've": "shall not have",
                           "she'd": "she would", "she'd've": "she would have", "she'll": "she will", "she'll've": "she will have", "she's": "she is",
                           "should've": "should have", "shouldn't": "should not", "shouldn't've": "should not have", "so've": "so have","so's": "so as",
                           "this's": "this is","that'd": "that would", "that'd've": "that would have", "that's": "that is", "there'd": "there would",
                           "there'd've": "there would have", "there's": "there is", "here's": "here is","they'd": "they would", "they'd've": "they would have",
                           "they'll": "they will", "they'll've": "they will have", "they're": "they are", "they've": "they have", "to've": "to have",
                           "wasn't": "was not", "we'd": "we would", "we'd've": "we would have", "we'll": "we will", "we'll've": "we will have", "we're": "we are",
                           "we've": "we have", "weren't": "were not", "what'll": "what will", "what'll've": "what will have", "what're": "what are",
                           "what's": "what is", "what've": "what have", "when's": "when is", "when've": "when have", "where'd": "where did", "where's": "where is",
                           "where've": "where have", "who'll": "who will", "who'll've": "who will have", "who's": "who is", "who've": "who have",
                           "why's": "why is", "why've": "why have", "will've": "will have", "won't": "will not", "won't've": "will not have",
                           "would've": "would have", "wouldn't": "would not", "wouldn't've": "would not have", "y'all": "you all",
                           "y'all'd": "you all would","y'all'd've": "you all would have","y'all're": "you all are","y'all've": "you all have",
                           "you'd": "you would", "you'd've": "you would have", "you'll": "you will", "you'll've": "you will have",
                           "you're": "you are", "you've": "you have"}





def text_cleaner(text,num):
    stop_words = set(stopwords.words('english')) 
    newString = text.lower()
    newString = BeautifulSoup(newString, "html.parser").text
    newString = re.sub(r'\([^)]*\)', '', newString)
    newString = re.sub('"','', newString)
    newString = ' '.join([contraction_mapping[t] if t in contraction_mapping else t for t in newString.split(" ")])    
    newString = re.sub(r"'s\b","",newString)
    newString = re.sub("[^a-zA-Z]", " ", newString) 
    newString = re.sub('[m]{2,}', 'mm', newString)
    if(num==0):
        tokens = [w for w in newString.split() if not w in stop_words]
    else:
        tokens=newString.split()
    long_words=[]
    for i in tokens:
        if len(i)>1:                                                 #removing short word
            long_words.append(i)   
    return (" ".join(long_words)).strip()


@app.route('/api',methods =['GET'])
def my_app():

    my_text=str(request.args['query'])
    latent_dim = 300
    embedding_dim=200
    x_voc=8440
    y_voc=1989
    max_text_len=30
    tot_cnt_cnt=8439
    max_summary_len=8
    my_cleand_text=text_cleaner(my_text,0) 


    
    my_list.append(my_cleand_text)
    #prepare a tokenizer for reviews on training data
    x_tokenizer = Tokenizer(num_words=tot_cnt_cnt) 
    x_tokenizer.fit_on_texts(my_list)

    #convert text sequences into integer sequences
    x_tr_seq    =   x_tokenizer.texts_to_sequences(my_list) 
    #padding zero upto maximum length
    x_tr    =   pad_sequences(x_tr_seq,  maxlen=max_text_len, padding='post')
    
    #basedir = os.path.abspath(os.path.dirname(__file__))
    #de_embedding_file = os.path.join(basedir, 'static\npz\de_embedding.npz')

    tokenizer = Tokenizer()
    # load the tokenizer
    try:
        with open('F:\AOU\summarizer\\flask_app\static\\npz\\tokenizer.pickle','rb') as handle:
            tokenizer = pickle.load(handle)
    except:
        print('not found----------------------')
    
     

    # load the weights
    w_encoder_embeddings = np.load('F:\AOU\summarizer\\flask_app\static\\npz\encoder_embedding.npz', allow_pickle=True)
    w_decoder_embeddings = np.load('F:\AOU\summarizer\\flask_app\static\\npz\de_embedding.npz', allow_pickle=True)
    w_encoder_lstm_1 = np.load('F:\AOU\summarizer\\flask_app\static\\npz\en_lstm_1.npz', allow_pickle=True)
    w_encoder_lstm_2 = np.load('F:\AOU\summarizer\\flask_app\static\\npz\en_lstm_2.npz', allow_pickle=True)
    w_encoder_lstm_3 = np.load('F:\AOU\summarizer\\flask_app\static\\npz\en_lstm_3.npz', allow_pickle=True)
    w_decoder_lstm = np.load('F:\AOU\summarizer\\flask_app\static\\npz\de_lstm.npz', allow_pickle=True)
    w_dense = np.load('F:\AOU\summarizer\\flask_app\static\\npz\\time_distributed.npz', allow_pickle=True)
    w_attention_layer = np.load('F:\AOU\summarizer\\flask_app\static\\npz\\new_attention_layer.npz', allow_pickle=True)

    
    from keras import backend as K 
    K.clear_session()

    # Encoder
    encoder_inputs = Input(shape=(max_text_len,))

    #embedding layer
    enc_emb =  Embedding(x_voc, embedding_dim,trainable=True, name="encoder_embedding")(encoder_inputs)

    #encoder lstm 1
    encoder_lstm1 = LSTM(latent_dim,return_sequences=True,return_state=True,dropout=0.4,recurrent_dropout=0.4, name="en_lstm_1")
    encoder_output1, state_h1, state_c1 = encoder_lstm1(enc_emb)

    #encoder lstm 2
    encoder_lstm2 = LSTM(latent_dim,return_sequences=True,return_state=True,dropout=0.4,recurrent_dropout=0.4, name="en_lstm_2")
    encoder_output2, state_h2, state_c2 = encoder_lstm2(encoder_output1)

    #encoder lstm 3
    encoder_lstm4=LSTM(latent_dim, return_state=True, return_sequences=True,dropout=0.4,recurrent_dropout=0.4, name="en_lstm_3")
    encoder_outputs, state_h, state_c= encoder_lstm4(encoder_output2)

    # Set up the decoder, using `encoder_states` as initial state.
    decoder_inputs = Input(shape=(None,), name="de_inputs")

    #embedding layer
    dec_emb_layer = Embedding(y_voc, embedding_dim,trainable=True, name="de_embedding")
    dec_emb = dec_emb_layer(decoder_inputs)

    decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True,dropout=0.4,recurrent_dropout=0.2, name="de_lstm")
    decoder_outputs,decoder_fwd_state, decoder_back_state = decoder_lstm(dec_emb,initial_state=[state_h, state_c])

    # Attention layer
    attn_layer = AttentionLayer(name='attention_layer')
    attn_out, attn_states = attn_layer([encoder_outputs, decoder_outputs])
    
    # Concat attention input and decoder LSTM output
    decoder_concat_input = Concatenate(axis=-1, name='concat_layer')([decoder_outputs, attn_out])

    #dense layer
    decoder_dense =  TimeDistributed(Dense(y_voc, activation='softmax', name="dense_layer"))
    decoder_outputs = decoder_dense(decoder_concat_input)

    # Define the model 
    model = Model([encoder_inputs, decoder_inputs], decoder_outputs)


    #model.summary() 

      # set the weights of the model

    model.layers[1].set_weights(w_encoder_embeddings['arr_0'])
    model.layers[5].set_weights(w_decoder_embeddings['arr_0'])
    model.layers[2].set_weights(w_encoder_lstm_1['arr_0'])
    model.layers[4].set_weights(w_encoder_lstm_2['arr_0'])
    model.layers[6].set_weights(w_encoder_lstm_3['arr_0'])
    model.layers[7].set_weights(w_decoder_lstm['arr_0'])
    model.layers[8].set_weights(w_attention_layer['arr_0'])
    model.layers[10].set_weights(w_dense['arr_0'])

    reverse_target_word_index=tokenizer.index_word
    target_word_index=tokenizer.word_index

    
        # Encode the input sequence to get the feature vector
    encoder_model = Model(inputs=encoder_inputs,outputs=[encoder_outputs, state_h, state_c])

    # Decoder setup
    # Below tensors will hold the states of the previous time step
    decoder_state_input_h = Input(shape=(latent_dim,))
    decoder_state_input_c = Input(shape=(latent_dim,))
    decoder_hidden_state_input = Input(shape=(max_text_len,latent_dim))

    # Get the embeddings of the decoder sequence
    dec_emb2= dec_emb_layer(decoder_inputs) 
    # To predict the next word in the sequence, set the initial states to the states from the previous time step
    decoder_outputs2, state_h2, state_c2 = decoder_lstm(dec_emb2, initial_state=[decoder_state_input_h, decoder_state_input_c])

    #attention inference
    attn_out_inf, attn_states_inf = attn_layer([decoder_hidden_state_input, decoder_outputs2])
    decoder_inf_concat = Concatenate(axis=-1, name='concat')([decoder_outputs2, attn_out_inf])

    # A dense softmax layer to generate prob dist. over the target vocabulary
    decoder_outputs2 = decoder_dense(decoder_inf_concat) 

    # Final decoder model
    decoder_model = Model(
        [decoder_inputs] + [decoder_hidden_state_input,decoder_state_input_h, decoder_state_input_c],
        [decoder_outputs2] + [state_h2, state_c2])
    

    def decode_sequence(input_seq):

            # Encode the input as state vectors.
            e_out, e_h, e_c = encoder_model.predict(input_seq)
            
            # Generate empty target sequence of length 1.
            target_seq = np.zeros((1,1))
            
            # Populate the first word of target sequence with the start word.
            target_seq[0, 0] = target_word_index['sostok']

            stop_condition = False
            decoded_sentence = ''
            while not stop_condition:
            
                output_tokens, h, c = decoder_model.predict([target_seq] + [e_out, e_h, e_c])

                # Sample a token
                sampled_token_index = np.argmax(output_tokens[0, -1, :])
                sampled_token = reverse_target_word_index[sampled_token_index]
                
                if(sampled_token!='eostok'):
                    decoded_sentence += ' '+sampled_token

                # Exit condition: either hit max length or find stop word.
                if (sampled_token == 'eostok'  or len(decoded_sentence.split()) >= (max_summary_len-1)):
                    stop_condition = True

                # Update the target sequence (of length 1).
                target_seq = np.zeros((1,1))
                target_seq[0, 0] = sampled_token_index

                # Update internal states
                e_h, e_c = h, c

            return decoded_sentence
    



    
    
    
    return decode_sequence(x_tr[-1].reshape(1,max_text_len))





if __name__ == '__main__':
    app.run()