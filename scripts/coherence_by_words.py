from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import word_tokenize
from gensim import corpora, matutils
from gensim.models import LdaModel, LsiModel, CoherenceModel
from sklearn.decomposition import NMF, PCA
from sklearn.random_projection import GaussianRandomProjection
from scipy import sparse
import numpy as np

def coherence_by_words(df, n):
    models = ['LDA', 'LSA', 'NMF', 'PCA', 'RP']

    documents = df['Clean_Content'].tolist()
    documents

    coherence = []

    for model_name in models:
        # TF-IDF Vectorization
        vectorizer = TfidfVectorizer(max_df = 0.5, min_df = 5, norm = 'l2', max_features=n)
        tfidf_matrix = vectorizer.fit_transform(documents)
        feature_names = vectorizer.get_feature_names_out()

        # this is a list of documents with tokens
        # it's needed for the coherence function
        texts = [word_tokenize(document) for document in documents]
        
        # Convert TF-IDF matrix to Gensim corpus
        corpus = matutils.Sparse2Corpus(tfidf_matrix.transpose())
        # Convert the document-term matrix to a gensim Dictionary
        dictionary = corpora.Dictionary.from_corpus(corpus,
                                                id2word=dict((id, word) for word, id in vectorizer.vocabulary_.items()))
        
        if model_name == 'LDA' or model_name == 'LSA':

            if model_name == 'LDA':
                model = LdaModel(corpus=corpus, id2word=dictionary, num_topics = 5,
                                alpha='symmetric', eta='auto', passes=5, random_state=1)
        
            elif model_name == 'LSA':
                model = LsiModel(corpus, id2word=dictionary, num_topics=5, random_seed=1)
            
            coherence_value = CoherenceModel(model=model, dictionary = dictionary, texts=texts, coherence='c_v').get_coherence()
            coherence.append(coherence_value)

        elif model_name == 'NMF' or model_name == 'PCA' or model_name == 'RP':
            
            if model_name == 'NMF':
                
                model = NMF(n_components=5, random_state=1, max_iter=600).fit(tfidf_matrix)
                
            elif model_name == 'RP':
                
                model = GaussianRandomProjection(n_components=5, random_state=1).fit(tfidf_matrix)
                
            elif model_name == 'PCA':
                # Convert sparse matrix to dense. PCA cannot be done on sparse matrixes
                tfidf_matrix_dense = tfidf_matrix.todense() if sparse.issparse(tfidf_matrix) else tfidf_matrix

                # Convert to numpy array
                tfidf_matrix_array = np.asarray(tfidf_matrix_dense)

                # Centering
                mean_tfidf = np.mean(tfidf_matrix_array, axis=0)  # Calculate the mean of each column
                centered_tfidf_matrix = tfidf_matrix_array - mean_tfidf
                
                model = PCA(n_components=5, random_state=1).fit(centered_tfidf_matrix)
                
            # Retrieve top words for each component
            components = []
            for j, component in enumerate(model.components_):
                component_words = [(feature_names[k], component[k]) for k in component.argsort()[::-1]]
                components.append(component_words)
                
            topics_for_coherence = [[word for word, _ in component] for component in components]
            
            coherence_value = CoherenceModel(topics=topics_for_coherence, texts=texts, dictionary=dictionary, coherence='c_v').get_coherence()
            
            coherence.append(coherence_value)

    coherence = [round(num, 2) for num in coherence]

    return list(zip(models, coherence))
