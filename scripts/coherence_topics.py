from gensim.models import LdaModel, LsiModel, CoherenceModel
from sklearn.decomposition import NMF

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
            
        elif model_name == 'NMF':
            model = NMF(n_components=i, random_state=42, max_iter=600).fit(tfidf)
            
            topics = [list(zip(feature_names, weights)) for weights in model.components_]
            
            topics_for_coherence = [[word for word, _ in topic] for topic in topics]
            
            coherence_value = CoherenceModel(topics=topics_for_coherence, texts=texts, dictionary=dictionary, coherence='c_v').get_coherence()
            
            coherence.append(coherence_value)
        
    return list(zip(n_topics, coherence))