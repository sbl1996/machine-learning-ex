import re
import nltk
from nltk.stem import PorterStemmer

from get_vocab_list import get_vocab_list

def process_email(email_contents):
  vocab = get_vocab_list()
  indices = [] 

  email_contents = email_contents.lower()
  email_contents = re.sub(r'<[^<>]+>', ' ', email_contents)
  email_contents = re.sub(r'[0-9]+', 'number', email_contents)
  email_contents = re.sub(r'(http|https)://[^\s]*', 'httpaddr', email_contents)
  email_contents = re.sub(r'[^\s]+@[^\s]+', 'emailaddr', email_contents)
  email_contents = re.sub(r'[$]+', 'dollar', email_contents)

  ps = PorterStemmer()

  for w in nltk.word_tokenize(email_contents):
    w = re.sub(r'[^a-zA-Z0-9]', '', w)
    w = ps.stem(w)
    if len(w) < 1:
      continue
    i = vocab.get(w)
    if i: indices.append(i)

  return indices