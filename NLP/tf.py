from collections import Counter
mydoclist = ['Julie loves me more than Linda loves me', 
			'Jane likes me more than Julie loves me',
			'He likes basketball more than baseball']
for doc in mydoclist:
    tf = Counter()
    for word in doc.split():
        tf[word] +=1
    print tf.items()
