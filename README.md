# Language games

cs224w & cs229 final project. 

## Setup
- Clone this repo. 
- Download data files to `data` 
    - [Hacker news comment corpus](http://chrisproctor.net/research/language_games/hn_comments.csv.zip)
    - [text8](http://mattmahoney.net/dc/text8.zip), the first 10e8 words from wikipedia
    - [Google pretrained word vectors](https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit?usp=sharing)
    - Analogies task questions (See word2vec/README.md)
- Install dependencies
    - `python3 -m venv env`
    - `source env/bin/activate`
    - `pip install -r requirements.txt`
    - Download and compile [kenlm](https://kheafield.com/code/kenlm/)
- Compile tensorflow ops
    - Follow instructions in word2vec/README.md and word2vec/NOTES.md
- Train the default word2vec implementation
    - `python word2vec/word2vec_optimized.py --train_data data/text8 --eval_data data/question-words.txt --save_path train`

## Directory structure

- **src** is for code we write.
- **lib** is for libraries written by others. 
- **data** is where big source files (not part of this repo) live. 
- **train** is for embeddings we trained.
- **results** is for outputs, like images and reports. 

## TODO specify data structures for experiments, etc. 
