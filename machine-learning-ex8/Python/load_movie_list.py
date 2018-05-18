def load_movie_list():
  movies = []
  with open('movie_ids.txt', encoding='ISO-8859-1') as f:
    for l in f:
      movies.append(l[:-1].split(' ', 1)[1])
  return movies