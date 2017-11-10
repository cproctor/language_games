# GET AVERAGE TOKENS

import csv
import numpy as np
import matplotlib.pyplot as plt

with open("hn_comments.csv") as inf:
    reader = csv.DictReader(inf)
    comments = [len(row['comment_text'].split()) for row in reader]

print(len(comments), np.mean(comments), np.std(comments))
plt.hist(comments, log=True)
plt.title("Comment length for 12 million hacker news comments")
plt.xlabel("Comment length")
plt.ylabel("Frequency")
plt.show()
        
