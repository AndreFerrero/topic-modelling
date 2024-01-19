from gensim.models import LdaModel, LsiModel, CoherenceModel
from sklearn.decomposition import NMF, PCA
from sklearn.random_projection import GaussianRandomProjection
from scipy import sparse
import numpy as np

def coherence_topics(model_name: str, corpus, dictionary, texts, feature_names, tfidf):
    n_topics = [5, 10, 15, 20, 50]
    coherence = []
    
    for i in n_topics:
        
        if model_name == 'LDA' or model_name == 'LSA':
            
            if model_name == 'LDA':
                model = LdaModel(corpus=corpus, id2word=dictionary, num_topics = i,
                                alpha='symmetric', eta='auto', passes=5, random_state=1)
        
            elif model_name == 'LSA':
                model = LsiModel(corpus, id2word=dictionary, num_topics=i)
            
            coherence_value = CoherenceModel(model=model, dictionary = dictionary, texts=texts, coherence='c_v').get_coherence()
            coherence.append(coherence_value)
            
        elif model_name == 'NMF' or model_name == 'PCA' or model_name == 'RP':
            
            if model_name == 'NMF':
                
                model = NMF(n_components=i, random_state=1, max_iter=600).fit(tfidf)
                
            elif model_name == 'RP':
                
                model = GaussianRandomProjection(n_components=i, random_state=1).fit(tfidf)
                
            elif model_name == 'PCA':
                # Convert sparse matrix to dense. PCA cannot be done on sparse matrixes
                tfidf_matrix_dense = tfidf.todense() if sparse.issparse(tfidf) else tfidf

                # Convert to numpy array
                tfidf_matrix_array = np.asarray(tfidf_matrix_dense)
                
                model = PCA(n_components=i).fit(tfidf_matrix_array)
                
            # Retrieve top words for each component
            topics = []
            for j, component in enumerate(model.components_):
                component_words = [(feature_names[k], component[k]) for k in component.argsort()[::-1]]
                topics.append(component_words)
                
            topics_for_coherence = [[word for word, _ in topic] for topic in topics]
            
            coherence_value = CoherenceModel(topics=topics_for_coherence, texts=texts, dictionary=dictionary, coherence='c_v').get_coherence()
            
            coherence.append(coherence_value)
        
    return list(zip(n_topics, coherence))