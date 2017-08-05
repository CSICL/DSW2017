from sklearn.feature_extraction.text import CountVectorizer

mydoclist = ['Julie loves me more than Linda loves me', 
			'Jane likes me more than Julie loves me',
			'He likes basketball more than baseball']
			
count_vectorizer = CountVectorizer(min_df=1)
term_freq_matrix = count_vectorizer.fit_transform(mydoclist)
print "Vocabulary:", count_vectorizer.vocabulary_

from sklearn.feature_extraction.text import TfidfTransformer

tfidf = TfidfTransformer(norm="l2")
tfidf.fit(term_freq_matrix)

tf_idf_matrix = tfidf.transform(term_freq_matrix)
print tf_idf_matrix.todense()
