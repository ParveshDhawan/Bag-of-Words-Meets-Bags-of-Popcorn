
# coding: utf-8

# # Bag of Words Meets Bags of Popcorn

# #### Importing libraries

# In[1]:


#Librarises
import pandas as pd  #data processing and data level operation
import numpy as np  # Linear Algebra
import os, re
import string
import nltk 
from nltk.corpus import stopwords

#Visualization libraries
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud,STOPWORDS

get_ipython().run_line_magic('matplotlib', 'inline')
###For downlaod the nltk
########nltk.download()


# In[2]:


#Current working directory
print('current workind directory ==== ',os.getcwd())

#Loading data
train = pd.read_csv('Input/labeledTrainData.tsv',delimiter = '\t')
test = pd.read_csv('Input/testData.tsv',delimiter = '\t')

train.shape, test.shape


# In[3]:


train.head()


# In[4]:


train['review'][0]


# In[5]:


test.head()


# In[6]:


test['review'][0]


# In[7]:


print ("number of rows for sentiment 1: {}".format(len(train[train.sentiment == 1])))
print ( "number of rows for sentiment 0: {}".format(len(train[train.sentiment == 0])))


# There is equal distribution of data

# In[8]:


train.groupby('sentiment').describe().transpose()


# In[9]:


#Creating a new col
train['length'] = train['review'].apply(len)
train.head()


# ### Data Visualization

# In[10]:


#Histogram of count of letters
train['length'].plot.hist(bins = 100)


# In[11]:


train.length.describe()


# In[12]:


#train[train['length'] == 13708]['review']
train[train['length'] == 13708]['review'].iloc[0]


# In[13]:


train.hist(column='length', by='sentiment', bins=100,figsize=(12,4))


# ### Text Preprocessing

# In[14]:


from bs4 import BeautifulSoup

#Creating a function for cleaning of data
def clean_text(raw_text):
    # 1. remove HTML tags
    raw_text = BeautifulSoup(raw_text).get_text() 
    
    # 2. removing all non letters from text
    letters_only = re.sub("[^a-zA-Z]", " ", raw_text) 
    
    # 3. Convert to lower case, split into individual words
    words = letters_only.lower().split()                           
    
    # 4. Create variable which contain set of stopwords
    stops = set(stopwords.words("english"))                  
    
    # 5. Remove stop word & returning   
    return [w for w in words if not w in stops]


# In[15]:


#Cleaning review and also adding a new col as its len count of words
train['clean_review'] = train['review'].apply(clean_text)
train['length_clean_review'] = train['clean_review'].apply(len)
train.head()


# In[16]:


train.describe()


# In[17]:


#Checking the smallest review
print(train[train['length_clean_review'] == 4]['review'].iloc[0])
print('------After Cleaning------')
print(train[train['length_clean_review'] == 4]['clean_review'].iloc[0])


# ### Word CLoud

# Wordcloud before cleaning

# In[18]:


#Plot wordcloud
word_cloud = WordCloud(width = 1000, height = 500, stopwords = STOPWORDS, background_color = 'red').generate(
                        ''.join(train['review']))

plt.figure(figsize = (15,8))
plt.imshow(word_cloud)
plt.axis('off')
plt.show()

#word_cloud.to_file('aa.png')   #for saving file


# ### Vectorization

# Now we'll convert each review, represented as a list of tokens into a vector that machine learning models can understand.
# 
# We'll do that in three steps using the bag-of-words model:
# 
# 1. Count how many times does a word occur in each message (Known as term frequency)
# 
# 2. Weigh the counts, so that frequent tokens get lower weight (inverse document frequency)
# 
# 3. Normalize the vectors to unit length, to abstract from the original text length (L2 norm)
# 
# Let's begin the first step:

# In[19]:


from sklearn.feature_extraction.text import CountVectorizer


# There are a lot of arguments and parameters that can be passed to the CountVectorizer. In this case we will just specify the **analyzer** to be our own previously defined function:

# In[20]:


# Might take awhile...
bow_transform = CountVectorizer(analyzer=clean_text).fit(train['review'])  #bow = bag of word

# Print total number of vocab words
print(len(bow_transform.vocabulary_))


# Let's take one text message and get its bag-of-words counts as a vector, putting to use our new **bow_transformer**

# In[21]:


review1 = train['review'][1]
print(review1)


# Now let's see its vector representation:

# In[22]:


bow1 = bow_transform.transform([review1])
print(bow1)
print(bow1.shape)


# Here 2 and 1 represent the occurance of word in the text phase & Number like  '73396' are index of that particular word

# In[23]:


print(bow_transform.get_feature_names()[71821])
print(bow_transform.get_feature_names()[72911])


# Above both words stores on that particular index

# In[24]:


#Creating bag of words for our review variable
review_bow = bow_transform.transform(train['review'])


# Checking the shape and non zero occurances of our sparse matrix that we have just generated

# In[25]:


print('Shape of Sparse Matrix: ', review_bow.shape)
print('Amount of Non-Zero occurences: ', review_bow.nnz)


# Checking the sparsity of sparse matrix

# In[26]:


sparsity = (100.0 * review_bow.nnz / (review_bow.shape[0] * review_bow.shape[1]))
print('sparsity: {}'.format(sparsity))


# ### TF-IDF
# Term Frequency - Inverse Document Frequency.
# 
# After the counting, the term weighting and normalization can be done with tf-idf
# 
# **TF: Term Frequency**, which measures how frequently a term occurs in a document. Since every document is different in length, it is possible that a term would appear much more times in long documents than shorter ones. Thus, the term frequency is often divided by the document length (aka. the total number of terms in the document) as a way of normalization: 
# 
# *TF(t) = (Number of times term t appears in a document) / (Total number of terms in the document).*
# 
# **IDF: Inverse Document Frequency**, which measures how important a term is. While computing TF, all terms are considered equally important. However it is known that certain terms, such as "is", "of", and "that", may appear a lot of times but have little importance. Thus we need to weigh down the frequent terms while scale up the rare ones, by computing the following: 
# 
# *IDF(t) = log_e(Total number of documents / Number of documents with term t in it).*

# In[27]:


from sklearn.feature_extraction.text import TfidfTransformer

tfidf_transformer = TfidfTransformer().fit(review_bow)
tfidf1 = tfidf_transformer.transform(bow1)
print(tfidf1)


# We'll go ahead and check what is the IDF (inverse document frequency) of the word `"war"` and of word `"book"`?

# In[28]:


print(tfidf_transformer.idf_[bow_transform.vocabulary_['war']])
print(tfidf_transformer.idf_[bow_transform.vocabulary_['book']])


# To transform the entire bag-of-words corpus into TF-IDF corpus at once:

# In[29]:


review_tfidf = tfidf_transformer.transform(review_bow)
print(review_tfidf.shape)


# ## Modeling Part

# #### Train Test Split

# In[30]:


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(train['review'], train['sentiment'], test_size=0.22, random_state=101)

len(X_train), len(X_test), len(X_train) + len(X_test)


# #### Result Function

# In[31]:


from sklearn.metrics import classification_report
#Predicting & Stats Function
def pred(predicted,compare):
    cm = pd.crosstab(compare,predicted)
    TN = cm.iloc[0,0]
    FN = cm.iloc[1,0]
    TP = cm.iloc[1,1]
    FP = cm.iloc[0,1]
    print("CONFUSION MATRIX ------->> ")
    print(cm)
    print()
    
    ##check accuracy of model
    print('Classification paradox :------->>')
    print('Accuracy :- ', round(((TP+TN)*100)/(TP+TN+FP+FN),2))
    print()
    print('False Negative Rate :- ',round((FN*100)/(FN+TP),2))
    print()
    print('False Postive Rate :- ',round((FP*100)/(FP+TN),2))
    print()
    print(classification_report(compare,predicted))


# ### Training Model

# #### Logistic Regression

# In[32]:


from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

pipeline = Pipeline([
    ('bow', CountVectorizer(analyzer=clean_text)),  # strings to token integer counts
    ('tfidf', TfidfTransformer()),  # integer counts to weighted TF-IDF scores
    ('classifier', LogisticRegression(random_state=101)),  # train on TF-IDF vectors w/ Naive Bayes classifier
])

pipeline.fit(X_train,y_train)
predictions = pipeline.predict(X_train)
pred(predictions,y_train)


# In[33]:


#Test Set Result
predictions = pipeline.predict(X_test)
pred(predictions,y_test)


# ##### Naive Bayes Model

# In[34]:


from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB

pipeline = Pipeline([
    ('bow', CountVectorizer(analyzer=clean_text)),  # strings to token integer counts
    ('tfidf', TfidfTransformer()),  # integer counts to weighted TF-IDF scores
    ('classifier', MultinomialNB()),  # train on TF-IDF vectors w/ Naive Bayes classifier
])

pipeline.fit(X_train,y_train)
predictions = pipeline.predict(X_train)
pred(predictions,y_train)


# In[35]:


#Result on Test Case
predictions = pipeline.predict(X_test)
pred(predictions,y_test)


# ### Random Forest

# In[36]:


from sklearn.ensemble import RandomForestClassifier

pipeline = Pipeline([
    ('bow', CountVectorizer(analyzer=clean_text)),  # strings to token integer counts
    ('tfidf', TfidfTransformer()),  # integer counts to weighted TF-IDF scores
    ('classifier', RandomForestClassifier(n_estimators = 500)),  # train on TF-IDF vectors w/ Naive Bayes classifier
])

pipeline.fit(X_train,y_train)
predictions = pipeline.predict(X_train)
pred(predictions,y_train)


# In[37]:


#Test Set Result
predictions = pipeline.predict(X_test)
pred(predictions,y_test)


# ### Final Model Will be Logistic Regression

# In[38]:


#Saving Output
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

pipeline_logit = Pipeline([
    ('bow', CountVectorizer(analyzer=clean_text)),  # strings to token integer counts
    ('tfidf', TfidfTransformer()),  # integer counts to weighted TF-IDF scores
    ('classifier', LogisticRegression(random_state=101)),  # train on TF-IDF vectors w/ Naive Bayes classifier
])

pipeline_logit.fit(train['review'],train['sentiment'])
test['sentiment'] = pipeline_logit.predict(test['review'])


# In[39]:


test.head(5)


# In[40]:


output = test[['id','sentiment']]
print(output)


# In[41]:


output.to_csv( "output.csv", index=False, quoting=3 )

