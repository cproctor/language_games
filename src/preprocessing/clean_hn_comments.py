# Clean HN Comments
# author: cp

# This script does a preliminary cleaning of the comments.
# The comment data pulled from the HN api was messy. For example, some object IDs 
# are represented as ints, others as floats; when there are 0 points, it's an empty
# string, etc. It seems the data structure may have changed over time?

# It took about an hour to run on my computer. Originally, I was cleaning up the comment
# text with beautifulsoup, but that was going to take an eternity, so I deferred that to later.


from csv import DictReader, DictWriter
from bs4 import BeautifulSoup as BS
from tqdm import tqdm
from settings import *

HN_DATA = "../../data/hn_comments_clean.csv"
HN_CLEAN_DATA = "../../data/hn_comments_utf8_text.csv"

fields = ["comment_text","points","author","created_at","objectID", "parent_id"]

def is_header(comment):
    return comment['objectID'] == 'objectID'

progress = tqdm(total=12166758)
progress.set_description("Cleaning HN comments")
granularity = 100
with open(HN_DATA) as infile:
    with open(HN_CLEAN_DATA, 'wb') as outfile:
        reader = DictReader(infile, fieldnames=fields)
        writer = DictWriter(outfile, fieldnames=fields)
        for i, comment in enumerate(reader):
            #if is_header(comment): continue
            try: 
                if i % granularity == 0: progress.update(granularity)
                writer.writerow({
                    'comment_text': BS(comment['comment_text'], 'html.parser').get_text().encode('utf8'),
                    #'comment_text': comment['comment_text'],
                    'points': float(comment['points']) if comment['points'] else 0,
                    'author': comment['author'],
                    'created_at': comment['created_at'],
                    'objectID': int(float(comment['objectID'])),
                    'parent_id': int(float(comment['parent_id']))
                })
            except ValueError:
                print("ERROR: {}".format(comment))
                continue
