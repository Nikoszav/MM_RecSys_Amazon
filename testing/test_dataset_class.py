import datasets
from datasets import Movies
import matplotlib.pyplot as plt


movie_dataset = Movies(pkl_file='data.pkl',
                                    root_dir="poster_downloads/")

w = 10
h = 10
fig = plt.figure(figsize=(8, 8))
columns = 4
rows = 5

for i in range(1,20):
    movie_id, movie_descr, poster = movie_dataset[i]

    print(i, movie_id, movie_descr)

    # ax = plt.subplot(1, 4, i + 1)
    
    fig.add_subplot(rows, columns, i)
    
    plt.imshow(poster)
plt.show()