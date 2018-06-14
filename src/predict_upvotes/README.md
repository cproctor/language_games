# Predict Upvotes

Previously, I was working on Danescu-Niculescu-Mizil et al's (2013) task of predicting users' futures. 
I'd like to switch tasks to predicting upvotes. 

1. Do comment upvotes predict user futures?
2. Predict upvotes
    1. Get upvotes (labels) for train, dev, test sets.
    2. Baseline: static GloVe embedding with SVM regression
    3. Baseline: TF/IDF on comments with SVM regression
    4. Word2Vec monthly embeddings (BoW) with SVM regression
    5. Word2Vec monthly embeddings (BoW) with SVM regression, truncated to 20k vocab
    5. GloVe monthly embeddings (BoW) with SVM regression
