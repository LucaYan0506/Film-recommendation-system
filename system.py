from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD
import pandas as pd
import json 
import re
import numpy as np 

"""
#calculate lsa values

df = pd.read_csv('data/final_dataset.csv')
body = df['text']

vectorizer = TfidfVectorizer(min_df=1,stop_words='english')
bag_of_words = vectorizer.fit_transform(body)

svd = TruncatedSVD(n_components=2000)
lsa = svd.fit_transform(bag_of_words)

# Create a DataFrame to store LSA values with corresponding IDs
lsa_df = pd.DataFrame(subtitle_ids, columns=['Subtitle_ID'])
lsa_df = pd.concat([lsa_df, pd.DataFrame(lsa, columns=[f'LSA_{i+1}' for i in range(2000)])], axis=1)

# Save the DataFrame to a CSV file
lsa_df.to_csv('data/lsa_values.csv', index=False)
"""

print('Loading data, please wait...')
#read data
df = pd.read_csv('data/final_dataset.csv')

#input film
film_name = input("Insert the name of the film:")
film_row = df.loc[df['title'] == film_name]

#make sure that film_row is not empty
while film_row.index.empty:
    print('invalid film name')
    film_name = input("Insert another name of the film:")
    film_row = df.loc[df['title'] == film_name]

print('Calculating, please wait...')
#read data
lsa_df_loaded = pd.read_csv('data/lsa_values.csv')

#save lsa values into a variable
lsa_columns = [f'LSA_{i+1}' for i in range(2000)]
lsa_values = lsa_df_loaded[lsa_columns].values

#calculate the similarity between film_row and other film 
ls = []
for i in range(0, len(lsa_values)):
    if film_row.index[0] == i:
        continue
    cosine_similarity_value = cosine_similarity(lsa_values[film_row.index[0]].reshape(1, -1), lsa_values[i].reshape(1, -1))
    ls.append((lsa_df_loaded['id'][i],cosine_similarity_value[0][0]))

#sort in desceding order (so that ls[0] is the most similar film)
ls = sorted(ls, key=lambda x: x[1], reverse=True)

#print the first ten films 
print('I would recommend the follwing films')
for i in range(10):
    print(f"{i + 1}: {df.loc[df['id']==ls[i][0]]['title'].values[0]}")


#deserializing keywords
json_data = []
index = 0
for x in df['keywords']:
    index += 1
    x = x.replace("'",'"')
    x = x.replace('children"s book',"children's book")
    x = x.replace('women"s prison',"women's prison")
    x = x.replace('new year"s eve',"new year's eve")
    x = x.replace('côte d"azur',"côte d'azur")
    x = x.replace('family"s daily life',"family's daily life")
    x = x.replace("""\\xa0""", " ")
    try:
        json_data.append(json.loads(x))
    except:
        print(index,x)
        break