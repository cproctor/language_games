# Comment Graph

The scripts in this directory are utilities for processing hacker news comments into a comment graph.

- **clean_hn_data** cleans up the CSV. Apparently the API returned some wonky stuff--like at some point
  user ids start being floats...
- **build_comment_graph** builds the user graph. 

Output files are not stored in the repo, but at http://chrisproctor.net/research/language_games/ 
Scripts expect to find these files in the /data folder of this project.



