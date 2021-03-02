
import warnings
warnings.filterwarnings("ignore")

from keras.utils import np_utils, to_categorical
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier  
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import AdaBoostClassifier
from sklearn.svm import SVC
from numpy import loadtxt
from sklearn.ensemble import RandomForestClassifier
#clf_RF = RandomForestClassifier(max_depth=2, random_state=0)



# import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import itertools


from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import RegexpTokenizer

#%%
#==============================================================================
# Load the dataset
#==============================================================================
data = open("Dataset_Fiek_5.0.txt", encoding="utf-8").read()
y, docs = [], []
for i, line in enumerate(data.split("\n")):
    content = line.split("\t")
    docs.append(content[0])
    y.append(content[1])


#%%
#==============================================================================
# Document preprocessing
#==============================================================================
'''
stop_words = set(line.strip() for line in open('stopwords.txt'))
exclude = set(string.punctuation) 

def docs_preprocessor(docs):
    tokenizer = RegexpTokenizer(r'\w+')
    for idx in range(len(docs)):
        docs[idx] = docs[idx].lower()  # Convert to lowercase.
        docs[idx] = tokenizer.tokenize(docs[idx])  # Split into words.

    # Remove numbers, but not words that contain numbers.
    docs = [[token for token in doc if not token.isdigit()] for doc in docs]

    #Remove stop words in documents
    docs = [[token for token in doc if not token in stop_words] for doc in docs]

    #Remove punctuations in documents
    docs = [[token for token in doc if not token in exclude] for doc in docs]

    # Remove words that are only one character.
    docs = [[token for token in doc if len(token) > 2] for doc in docs]
    
    # Lemmatize all words in documents.
    lemmatizer = WordNetLemmatizer()
    docs = [[lemmatizer.lemmatize(token) for token in doc] for doc in docs]
  
    return docs

# Perform function on our document
documents = docs_preprocessor(docs)



#==============================================================================
# dummy_fun simply returns what it receives
#==============================================================================


# dummy_fun simply returns what it receives
def dummy_fun(doc):
    return doc

tfidf = TfidfVectorizer(
    analyzer='word',
    tokenizer=dummy_fun,
    preprocessor=dummy_fun,
    token_pattern=None) 
tfidf.fit(documents)

x=tfidf.transform(documents)

'''


#==============================================================================
# TF-idf vectors as feature vectors (word level tf-idf)
#==============================================================================
'''
# word level tf-idf

my_stop_words = set(line.strip() for line in open('stopwords.txt'))

tfidf_vect = TfidfVectorizer(analyzer='word', lowercase=True, token_pattern=r'\w{2,}', stop_words=my_stop_words, max_features=2000)
tfidf_vect.fit(docs)
x = tfidf_vect.transform(docs)


'''
#==============================================================================
# Count Vectors as features
#==============================================================================
my_stop_words = set(line.strip() for line in open('stopwords.txt'))

# create a count vectorizer object 
count_vect = CountVectorizer(analyzer='word', lowercase=True, token_pattern=r'\w{2,}',stop_words=my_stop_words, max_features=2000)
count_vect.fit(docs)
x=count_vect.transform(docs)




#==============================================================================
# Encode class values as integers 
#==============================================================================
encoder = LabelEncoder()
encoder.fit(y)
encoded_y = encoder.transform(y)

# convert integers to dummy variables (i.e. one hot encoded)
dummy_y = np_utils.to_categorical(encoded_y)



'''
from collections import Counter
from sklearn.datasets import make_classification
from imblearn.over_sampling import SMOTE # doctest: +NORMALIZE_WHITESPACE     
from imblearn.under_sampling import OneSidedSelection


#sm = SMOTE(random_state=42)
#x, encoded_y = sm.fit_resample(x, encoded_y)


undersample = OneSidedSelection(n_neighbors=1, n_seeds_S=200)
# transform the dataset
x, encoded_y = undersample.fit_resample(x, encoded_y)

'''

#==============================================================================
# Training and testing 
#==============================================================================
seed =1000

x_train, x_test, y_train, y_test = train_test_split(x, encoded_y, train_size=0.7, random_state=seed, shuffle=True)
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, train_size=0.7, random_state=seed)


#==============================================================================
# SVM classifier
#==============================================================================
print("\nResults from SVM\n")

clf_SVM=SVC(kernel='linear')

clf_SVM.fit(x_train, y_train)

accuracy=clf_SVM.score(x_test, y_test)

y_pred = clf_SVM.predict(x_val)

print(confusion_matrix(y_val, y_pred))
  
print(classification_report(y_val, y_pred,digits=4)) 

print("SVM accuracy:", accuracy)




#==============================================================================
# Decision Tree classifier
#==============================================================================
print("\n\n\nResults from Decision Tree\n")

clf_DT = DecisionTreeClassifier() 
 
clf_DT.fit(x_train, y_train)

accuracy=clf_DT.score(x_test, y_test)

y_pred = clf_DT.predict(x_test)

print(confusion_matrix(y_test, y_pred))  

print(classification_report(y_test, y_pred,digits=4))
   
print("DT accuracy:", accuracy)



#==============================================================================
# Naive Bayes classifier
#==============================================================================
print("\n\n\nResults from Naive Bayes\n")

clf_NB = MultinomialNB() 
 
clf_NB.fit(x_train, y_train)

accuracy=clf_NB.score(x_test, y_test)

y_pred = clf_NB.predict(x_test)

print(confusion_matrix(y_test, y_pred))  

print(classification_report(y_test, y_pred, digits=4)) 
  
print("NB accuracy:", accuracy)



#==============================================================================
# Boosting classifier - AdaBoost
#==============================================================================
print("\n\n\nResults from Boosting\n")

clf_Ada = AdaBoostClassifier()
 
clf_Ada.fit(x_train, y_train)

accuracy=clf_Ada.score(x_test, y_test)

y_pred = clf_Ada.predict(x_test)

print(confusion_matrix(y_test, y_pred))  

print(classification_report(y_test, y_pred, digits=4)) 
  
print("AdaBoost accuracy:", accuracy)



#==============================================================================
# Random Forest Classifier
#==============================================================================
print("\n\n\nResults from Random Forest\n")


from sklearn.ensemble import RandomForestRegressor

clf_RF = RandomForestClassifier(max_depth=200, random_state=0)
 
clf_RF.fit(x_train, y_train)

accuracy=clf_RF.score(x_test, y_test)

y_pred = clf_RF.predict(x_test)

print(confusion_matrix(y_test, y_pred))  

print(classification_report(y_test, y_pred, digits=4)) 
  
print("Random Forest accuracy:", accuracy)




