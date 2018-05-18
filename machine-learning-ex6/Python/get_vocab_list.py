def get_vocab_list():
  vocab = {}
  with open('vocab.txt') as f:
    for l in f:
      k, v = l.split()
      vocab[v] = int(k)
  return vocab