import pandas as pd
import os
import re, string
from nltk.corpus import words, wordnet, stopwords
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag
from nltk.tokenize import word_tokenize

def import_data(path):
    
    folders_path_list = []
    folders = os.listdir(path)
    
    for folder in folders:
        folders_path_list.append(os.path.join(path, folder))
    
    conversion = {'religion' : [folders[i] for i in [0, 15, 19]],
                  'computer': [folders[i] for i in range(1, 6)],
                  'science': [folders[i] for i in range(11, 15)],
                  'politics': [folders[i] for i in range(16, 19)],
                  'misc': [folders[6]],
                  'recreation': [folders[i] for i in range(7, 11)]}
    
    topics = folders.copy()
    
    for j, folder in enumerate(folders):
        for topic, values in conversion.items():
            
            if folder in values:
                topics[j] = topic
    
    
    column_names = ['File_Name', 'Content', 'Folder', 'Topic']
    df = pd.DataFrame(columns=column_names)

    for folder, folder_path, topic in zip(folders, folders_path_list, topics):
        for file_name in os.listdir(folder_path):
            file_path = os.path.join(folder_path, file_name)
            
            with open(file_path, 'r', encoding='latin-1') as file:
                content = file.read()
                df = pd.concat([df,
                                pd.DataFrame({'File_Name': [file_name], 'Content': [content],
                                              'Folder' : [folder], 'Topic': [topic]})],
                               ignore_index=True)
                
    df['Content'] = df['Content'].astype("string")
    
    return df

def preprocess(doc):
    # Remove email addresses
    doc = re.sub(r'\b\S*@\S*\.\S*\b', '', doc)
    
    # Remove special characters and digits, retain only words with letters
    doc = re.sub(r'[^\w\s]', '', doc)
    
    # Lowercase and strip
    doc = doc.lower().strip()
    
    # Remove brackets of any kind
    doc = re.sub(r'[(){}[\]]', '', doc)
    
    # Remove punctuation
    doc = doc.translate(str.maketrans("", "", string.punctuation))
    
    # Tokenize document
    tokens = word_tokenize(doc)
    
    # POS tagging
    pos_tags = pos_tag(tokens)
    
     # map POS tags to WordNet POS tags
    tag_map = {
        'N': wordnet.NOUN,
        'V': wordnet.VERB,
        'R': wordnet.ADV,
        'J': wordnet.ADJ
    }

    # Lemmatize tokens
    lemmatizer = WordNetLemmatizer()
    lemmatized_tokens = [lemmatizer.lemmatize(token, tag_map.get(pos[0], wordnet.NOUN)) for token, pos in pos_tags]
    
    
    # Filter stopwords out of lemmatized tokens
    stop_words = stopwords.words('english')
    
    stop_words.extend(['hi', 'thanks', 'lot', 'dont', 'article', 'everyone', 'anyone', 'someone', 'nothing',
                       'something', 'anything', 'everybody', 'somebody', 'anybody', 'please', 'ask', 'people', 'university',
                       'question', 'yeah', 'shouldnt', 'theyre', 'thing', 'theyll', 'didnt', 'sorry', 'hey',
                       'oh', 'thats', 'thank', 'cannot', 'right', 'would', 'one', 'get', 'know', 'like', 'use', 'go',
                       'think', 'make', 'say', 'see', 'also', 'could', 'well', 'want', 'way', 'take', 'find', 'need', 'try',
                       'much', 'come', 'many', 'may', 'give', 'really', 'tell', 'two', 'still', 'read', 'might', 'write',
                       'never', 'look', 'sure', 'day', 'even', 'new', 'time', 'good', 'first', 'keep', 'since', 'last', 
                       'long', 'fact', 'must', 'cant', 'another', 'little', 'without', 'csutexasedu', 'nntppostinghost',
                       'im', 'seem', 'replyto', 'let', 'group', 'call', 'seem', ])
    
    filtered_tokens = [token for token in lemmatized_tokens if token not in stop_words]
    
    # Recreate the document
    doc = ' '.join(filtered_tokens)
    
    return doc

def check_empty_docs(df):
    # Check for empty documents
    empty_documents = df[df['Clean_Content'].str.strip() == '']

    # Count the number of empty documents
    num_empty_documents = len(empty_documents)

    if num_empty_documents > 0:
        print(f"Number of empty documents: {num_empty_documents}")
        print("Indices of empty documents:")
        print(empty_documents.index)
    else:
        print("No empty documents found.")

path = R"C:\Users\sofch\Python\PR1.20\PR1.20\data\data-train"

df = import_data(path)

df['Clean_Content'] = df['Content'].apply(preprocess)