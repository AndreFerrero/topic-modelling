def display_topics(model_name, model, feature_names):
    if model_name == 'LDA' or model_name == 'LSA':
        
        for topic in model.print_topics(num_words=15):
            topic_index, words = topic
            topic_words = [word.split("*")[1].strip().strip('"') for word in words.split(" + ")]
            print(f"Topic {topic_index + 1}: {', '.join(topic_words)}")
        
    elif model_name == 'NMF' or model_name == 'PCA' or model_name == 'RP':
        
        for topic_index, component in enumerate(model.components_):
            component_words = [feature_names[k] for k in component.argsort()[::-1]]
            print(f"Topic {topic_index + 1}: {', '.join(component_words[:15])}")
            
            