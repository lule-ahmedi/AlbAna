
import warnings
warnings.filterwarnings("ignore")


import numpy as np
import pandas as pd
from keras import layers
from keras.models import Model, Sequential
from keras.layers import Dense, Dropout, Activation, LSTM,  SpatialDropout1D
from keras.layers import Input, Flatten, merge, Lambda, Dropout
from keras.wrappers.scikit_learn import KerasClassifier
from keras.layers.wrappers import TimeDistributed, Bidirectional
from keras.utils import np_utils, to_categorical
from keras.optimizers import Adam, RMSprop
from keras.layers import Conv1D, MaxPooling1D, Embedding
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Conv1D, MaxPooling1D, Embedding
from keras.layers.normalization import BatchNormalization


from sklearn.model_selection import train_test_split, KFold, GridSearchCV, StratifiedShuffleSplit,StratifiedKFold
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer



import gensim
from gensim.models import Word2Vec
from gensim.utils import simple_preprocess

from gensim.models.keyedvectors import KeyedVectors

# import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import itertools


from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import RegexpTokenizer



from sklearn.utils import shuffle



'''
#%%
#==============================================================================
# Load the dataset
#==============================================================================


from xlrd import open_workbook

book = open_workbook("Class1_Original_Dataset.xlsx","utf-8")
sheet = book.sheet_by_index(0) #If your data is on sheet 1

y = []
docs = []
lastrow=12033


for row in range(1, lastrow): #start from 1, to leave out row 0
    y.append(sheet.cell_value(row, 0)) #extract from first col
    docs.append(sheet.cell_value(row, 1))


'''
#%%
#==============================================================================
# Encode class values as integers 
#==============================================================================
data = open("Dataset_Fiek_5.0.txt", encoding="utf-8").read()
y, docs = [], []
for i, line in enumerate(data.split("\n")):
    content = line.split("\t")
    docs.append(content[0])
    y.append(content[1])

average = sum(len(doc) for doc in docs) / len(docs)
print (average)

#%%
#==============================================================================
# Encode class values as integers 
#==============================================================================
encoder = LabelEncoder()

encoder.fit(y)

encoded_y = encoder.transform(y)

# convert integers to dummy variables (i.e. one hot encoded)
dummy_y = np_utils.to_categorical(encoded_y)



#%%
#==============================================================================
# Define plot_history function
#==============================================================================
def plot_history(history):
    loss_list = [s for s in history.history.keys() if 'loss' in s and 'val' not in s]
    val_loss_list = [s for s in history.history.keys() if 'loss' in s and 'val' in s]
    acc_list = [s for s in history.history.keys() if 'acc' in s and 'val' not in s]
    val_acc_list = [s for s in history.history.keys() if 'acc' in s and 'val' in s]
    
    if len(loss_list) == 0:
        print('Loss is missing in history')
        return 
    
    ## As loss always exists
    epochs = range(1,len(history.history[loss_list[0]]) + 1)
    
    ## Loss
    plt.figure(1)
    for l in loss_list:
        plt.plot(epochs, history.history[l], 'b', label='Training loss (' + str(str(format(history.history[l][-1],'.5f'))+')'))
    for l in val_loss_list:
        plt.plot(epochs, history.history[l], 'g', label='Validation loss (' + str(str(format(history.history[l][-1],'.5f'))+')'))
    
    plt.title('Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    
    ## Accuracy
    plt.figure(2)
    for l in acc_list:
        plt.plot(epochs, history.history[l], 'b', label='Training accuracy (' + str(format(history.history[l][-1],'.5f'))+')')
    for l in val_acc_list:    
        plt.plot(epochs, history.history[l], 'g', label='Validation accuracy (' + str(format(history.history[l][-1],'.5f'))+')')

    plt.title('Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()




#==============================================================================
# plot confusion_matrix function
#==============================================================================

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        title='Normalized confusion matrix'
    else:
        title='Confusion matrix'

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()
    
   

#==============================================================================
# Define full_multiclass_report which prints classification report
#==============================================================================    
## If binary (sigmoid output), set binary parameter to True
def full_multiclass_report(model,
                           x,
                           y_true,
                           classes,
                           batch_size=64,
                           binary=False):

    # 1. Transform one-hot encoded y_true into their class number
    if not binary:
        y_true = np.argmax(y_true,axis=1)
    
    # 2. Predict classes and stores in y_pred
    y_pred = model.predict_classes(x, batch_size=batch_size)
    
    # 3. Print accuracy score
    print("Accuracy : "+ str(accuracy_score(y_true,y_pred)))
    
    print("")
    
    # 4. Print classification report
    print("Classification Report")
    print(classification_report(y_true,y_pred,digits=4))    
    
  
    # 5. Plot confusion matrix
    cnf_matrix = confusion_matrix(y_true,y_pred)
    print(cnf_matrix)
    plot_confusion_matrix(cnf_matrix,classes=classes)

#==============================================================================
# Input parameters
#==============================================================================
MAX_SEQUENCE_LENGTH = 20
MAX_NB_WORDS = 2000
EMBEDDING_DIM = 300


#==============================================================================
# Create a tokenizer
#==============================================================================
tokenizer = Tokenizer(nb_words=MAX_NB_WORDS, lower=True )

tokenizer.fit_on_texts(docs)

sequences = tokenizer.texts_to_sequences(docs)

word_index = tokenizer.word_index

print('Found %s unique tokens.' % len(word_index))

# convert text to sequence of tokens and pad them to ensure equal length vectors 
x = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)




'''
vec = CountVectorizer(max_features=20000)
x = vec.fit_transform(docs).toarray()
print(type(x),x.shape)  # 5572 doc, tfidf 100 dim

vec = TfidfVectorizer(max_features=20000)
x = vec.fit_transform(docs).toarray()
print(type(x),x.shape)  # 5572 doc, tfidf 100 dim
'''

#==============================================================================
# Training, testing and validation
#==============================================================================
seed =1000

x_train, x_test, y_train, y_test = train_test_split(x, dummy_y, train_size=0.7, random_state=seed)

x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, train_size=0.7, random_state=seed)






#==============================================================================
# Pretrained FastText embeddings
#==============================================================================
print('loading FastText word embeddings...')

embeddings_index = {}
words_not_found=[]
f=open ("cc.sq.300.vec", "r", encoding="utf-8")
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()

print('Found %s word vectors.' % len(embeddings_index))


embedding_matrix = np.zeros((len(word_index) + 1, EMBEDDING_DIM))
for word, i in word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        # words not found in embedding index will be all-zeros.
        embedding_matrix[i] = embedding_vector
    else:
        words_not_found.append(word)
print('Number of null word embeddings: %d' % np.sum(np.sum(embedding_matrix, axis=1) == 0))

'''


#==============================================================================
# Pretrained word2vec embeddings
#==============================================================================
word_vectors = KeyedVectors.load_word2vec_format('/Users/zekaaa/Documents/WIMS_2019/GoogleNews-vectors-negative300.bin', binary=True)

embedding_matrix = np.zeros((len(word_index) + 1, EMBEDDING_DIM))
for word, i in word_index.items():
    if i>=MAX_NB_WORDS:
        continue
    try:
        embedding_vector = word_vectors[word]
        embedding_matrix[i] = embedding_vector
    except KeyError:
        embedding_matrix[i]=np.random.normal(0,np.sqrt(0.25),EMBEDDING_DIM)
print('Found %s word vectors.' % len(word_vectors.vocab))
del(word_vectors)



#==============================================================================
# Pretrained Glove embeddings
#==============================================================================
embeddings_index = {}
words_not_found=[]
f=open ("/Users/zekaaa/Documents/WIMS_2019/glove.6B/glove.6b.300d.txt", "r", encoding="utf-8")
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()

print('Found %s word vectors.' % len(embeddings_index))


embedding_matrix = np.zeros((len(word_index) + 1, EMBEDDING_DIM))
for word, i in word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        # words not found in embedding index will be all-zeros.
        embedding_matrix[i] = embedding_vector
    else:
        words_not_found.append(word)
print('Number of null word embeddings: %d' % np.sum(np.sum(embedding_matrix, axis=1) == 0)) 

'''

#==============================================================================
# Build CNN model
#==============================================================================
from keras_self_attention import SeqSelfAttention
import keras

model = Sequential()


model.add(layers.Embedding(len(word_index) + 1,
                            EMBEDDING_DIM,
                            weights=[embedding_matrix],
                            input_length=MAX_SEQUENCE_LENGTH,
                            trainable=False))
'''
model.add(layers.Embedding(len(word_index),  EMBEDDING_DIM, input_length=MAX_SEQUENCE_LENGTH))

'''


model.add(SpatialDropout1D(0.3))

model.add(Bidirectional(LSTM(32, return_sequences=True)))

model.add(SeqSelfAttention(attention_width=8, attention_activation='sigmoid', name='Attention',))

model.add(Flatten())

model.add(Dense(3, activation='softmax'))


model.summary()



model.compile(loss='categorical_crossentropy',optimizer='adam', metrics=['accuracy'])



#==============================================================================
# Evaluate model and print results
#==============================================================================

CNN_History=model.fit(x_train, y_train, epochs = 35, batch_size = 256,verbose=1, validation_data=(x_val,y_val), shuffle=True)

plot_history(CNN_History)

full_multiclass_report(model, x_val, y_val, encoder.inverse_transform(np.arange(3)))



