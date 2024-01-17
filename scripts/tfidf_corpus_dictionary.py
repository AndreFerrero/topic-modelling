from sklearn.feature_extraction.text import TfidfVectorizer
from gensim import corpora, matutils
from nltk.tokenize import word_tokenize

def get_tfidf_tokendocs_corpus_dict(df, max_df, min_df, max_features):
                    #0.5, 5, 5000
    # convert text into lists
    documents = df['Clean_Content'].tolist()
    documents

    # TF-IDF Vectorization
    vectorizer = TfidfVectorizer(max_df = max_df, min_df = min_df, norm = 'l2', max_features=max_features)
    tfidf_matrix = vectorizer.fit_transform(documents)
    feature_names = vectorizer.get_feature_names_out()
    
    # this is a list of documents with tokens
    # it's needed for the coherence function
    tokenized_docs = [word_tokenize(document) for document in documents]
    
    # Convert TF-IDF matrix to Gensim corpus
    corpus = matutils.Sparse2Corpus(tfidf_matrix.transpose())
    # Convert the document-term matrix to a gensim Dictionary
    dictionary = corpora.Dictionary.from_corpus(corpus,
                                            id2word=dict((id, word) for word, id in vectorizer.vocabulary_.items()))
    
    return tfidf_matrix, feature_names, tokenized_docs, corpus, dictionary

    