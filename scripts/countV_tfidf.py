from sklearn.feature_extraction.text import CountVectorizer

cv = CountVectorizer(min_df=0., max_df=1.)
cv_matrix = cv.fit_transform(df_clean)
cv_matrix

cv_matrix = cv_matrix.toarray()
cv_matrix

# get all unique words in the corpus
vocab = cv.get_feature_names_out()
# show document feature vectors
pd.DataFrame(cv_matrix, columns=vocab)

from sklearn.feature_extraction.text import TfidfTransformer

tt = TfidfTransformer(norm='l2', use_idf=True)
tt_matrix = tt.fit_transform(cv_matrix)


tt_matrix = tt_matrix.toarray()
vocab = cv.get_feature_names_out()
tfidf = pd.DataFrame(np.round(tt_matrix, 2), columns=vocab)