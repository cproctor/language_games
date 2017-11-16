# Preprocessing

These scripts process comments in csv form, stripping out non-ascii, stripping out html, 
lowercasing, and tokenizing (words and punctuation separated by spaces). Then comments
are saved in separate files corresponding to each month of the community. Finally, 
for each month, we generate a tokens file suitable for training with word2vec and a 
bigram language model which will be our baseline. 
