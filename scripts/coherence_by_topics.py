from gensim.models import LdaModel, LsiModel, CoherenceModel
from sklearn.decomposition import NMF, PCA
from sklearn.random_projection import GaussianRandomProjection
from scipy import sparse
import numpy as np

def coherence_by_topics(n: int, corpus, dictionary, texts, feature_names, tfidf):
    models = ['LDA', 'LSA', 'NMF', 'PCA', 'RP']

    coherence = []

    for model_name in models:
        if model_name == 'LDA' or model_name == 'LSA':
            
            if model_name == 'LDA':
                model = LdaModel(corpus=corpus, id2word=dictionary, num_topics = n,
                                alpha='auto', eta='auto', passes=5, random_state=1)
        
            elif model_name == 'LSA':
                model = LsiModel(corpus, id2word=dictionary, num_topics=n, random_seed=1)
            
            coherence_value = CoherenceModel(model=model, dictionary = dictionary, texts=texts, coherence='c_v').get_coherence()
            coherence.append(coherence_value)

        elif model_name == 'NMF' or model_name == 'PCA' or model_name == 'RP':
            
            if model_name == 'NMF':
                
                model = NMF(n_components=n, random_state=1, max_iter=600).fit(tfidf)
                
            elif model_name == 'RP':
                
                model = GaussianRandomProjection(n_components=n, random_state=1).fit(tfidf)
                
            elif model_name == 'PCA':
                # Convert sparse matrix to dense. PCA cannot be done on sparse matrixes
                tfidf_matrix_dense = tfidf.todense() if sparse.issparse(tfidf) else tfidf

                # Convert to numpy array
                tfidf_matrix_array = np.asarray(tfidf_matrix_dense)

                # Centering
                mean_tfidf = np.mean(tfidf_matrix_array, axis=0)  # Calculate the mean of each column
                centered_tfidf_matrix = tfidf_matrix_array - mean_tfidf
                
                model = PCA(n_components=n, random_state=1).fit(centered_tfidf_matrix)
                
            # Retrieve top words for each component
            topics = []
            for j, component in enumerate(model.components_):
                component_words = [(feature_names[k], component[k]) for k in component.argsort()[::-1]]
                topics.append(component_words)
                
            topics_for_coherence = [[word for word, _ in topic] for topic in topics]
            
            coherence_value = CoherenceModel(topics=topics_for_coherence, texts=texts, dictionary=dictionary, coherence='c_v').get_coherence()
            
            coherence.append(coherence_value)
        
    coherence = [round(num, 4) for num in coherence]

    return list(zip(models, coherence))