#!/usr/bin/env python
# coding: utf-8

# In[2]:


import random
import pickle
import numpy as np
import pandas as pd
from nltk.tokenize import RegexpTokenizer
from tensorflow.keras.models import Sequential,load_model
from tensorflow.keras.layers import LSTM,Dense,Activation
from tensorflow.keras.optimizers import RMSprop


# In[3]:


# Set the number of rows
num_rows = 6335

# Generate sample data
data = {
    'id': range(1, num_rows + 1),
    'title': [f'Title {i}' for i in range(1, num_rows + 1)],
    'text': [f'This is the text for row {i}.' for i in range(1, num_rows + 1)],
    'label': np.random.choice(['label1', 'label2', 'label3'], num_rows)  # Random labels
}

# Create a DataFrame
df = pd.DataFrame(data)

# Define the file path
file_path = 'data.csv'

# Save to CSV
df.to_csv(file_path, index=False)

print(f"File written successfully to {file_path}")


# In[4]:


text_df=pd.read_csv('data.csv')


# In[5]:


text=list(text_df.text.values)
joined_text=' '.join(text)


# In[7]:


partial_text=joined_text[:10000]


# In[8]:


tokenizer=RegexpTokenizer(r'\w+')
tokens=tokenizer.tokenize(partial_text.lower())


# In[9]:


unique_tokens=np.unique(tokens)
unique_token_index={token:idx for idx,token in enumerate(unique_tokens)}


# In[11]:


n_words=10
input_words=[]
next_words=[]
for i in range(len(tokens)-n_words):
    input_words.append(tokens[i:i+n_words])
    next_words.append(tokens[i+n_words])


# In[13]:


x=np.zeros((len(input_words),n_words,len(unique_tokens)),dtype=bool)
y=np.zeros((len(next_words),len(unique_tokens)),dtype=bool)


# In[14]:


for i,words in enumerate(input_words):
    for j,word in enumerate(words):
        x[i,j,unique_token_index[word]]=1
    y[i,unique_token_index[next_words[i]]]=1


# In[18]:


model=Sequential()
model.add(LSTM(128,input_shape=(n_words,len(unique_tokens)),return_sequences=True))
model.add(LSTM(128))
model.add(Dense(len(unique_tokens)))
model.add(Activation('softmax'))


# In[20]:


model.compile(loss='categorical_crossentropy',optimizer=RMSprop(learning_rate=0.01),metrics=['accuracy'])
model.fit(x,y,batch_size=128,epochs=10,shuffle=True)


# In[21]:


model.save('mymodel.h5')


# In[22]:


model=load_model('mymodel.h5')


# In[35]:


def predict_next_word(input_text,n_best):
    input_text=input_text.lower()
    x=np.zeros((1,n_words,len(unique_tokens)))
    for i,word in enumerate(input_text.split()):
        x[0,i,unique_token_index[word]]=1
    predictions=model.predict(x)[0]
    return np.argpartition(predictions,n_best)[n_best:]


# In[47]:


possible=predict_next_word("this row is for text",5)


# In[48]:


print([unique_tokens[idx] for idx in possible])


# In[49]:


def generate_text(input_text,text_length,creativity=3):
    word_sequence=input_text.split()
    current=0
    for _ in range(text_length):
        sub_sequence=' '.join(tokenizer.tokenize(' '.join(word_sequence).lower())[current:current+n_words])
        try:
            choice=unique_tokens[random.choice(predict_next_word(sub_sequence,creativity))]
        except:
            choice=random.choice(unique_tokens)
        word_sequence.append(choice)
        current==1
    return ' '.join(word_sequence)


# In[50]:


generate_text("this row is for text",100,5)


# In[ ]:




