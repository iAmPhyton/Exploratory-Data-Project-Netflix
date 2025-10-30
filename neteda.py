import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px 
from collections import Counter 

neteda = pd.read_csv('netflix_titles.csv')
neteda.head() 

#inspecting the structure of the dataset
neteda.info()
neteda.describe(include='all')

#checking for missing values
neteda.isnull().sum()

#dropping duplicate values
neteda.drop_duplicates(inplace=True)

#replacing missing values with 'Unknown', 'Not Specified'
neteda['country'].fillna('Unknown', inplace=True)
neteda['cast'].fillna('Not Specified', inplace=True)
neteda['director'].fillna('Not Specified', inplace=True) 

#converting date fields 
neteda['date_added'] = neteda['date_added'].str.strip() 
neteda['date_added'] = pd.to_datetime(neteda['date_added'], errors='coerce', format='mixed') 

#exploratory analysis
#basic distribution of content type: count of movies vs TV shows
sns.countplot(data=neteda, x='type', palette='coolwarm')
plt.title('Distribution of Content Type')
plt.show() 

#content added over time
titles_per_year = neteda['release_year'].value_counts().sort_index().reset_index()
titles_per_year.columns = ['release_year', 'count']

#keeping only years from 1974 onward because from earlier tested charts, movies added before 1974 were too little and made the chart too clustered.
titles_per_year = titles_per_year[titles_per_year['release_year'] >= 1974]

plt.figure(figsize=(12,6))
sns.barplot(
    data=titles_per_year,
    x='release_year',
    y='count',
    hue='release_year',
    palette='crest',
    legend=False
)

plt.title('Number of Titles Added to Netflix Each Year (1974–Present)', fontsize=14, fontweight='bold')
plt.xlabel('Year Added', fontsize=12, fontweight= 'bold')
plt.ylabel('Number of Titles', fontsize=12, fontweight='bold' )
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show() 

#showing most represented countries
top_countries = neteda['country'].value_counts().head(10)
sns.barplot(x=top_countries.values, y=top_countries.index, palette='magma',
            legend=False)
plt.title('Top 10 Countries by Number of Titles', fontsize=12, fontweight='bold')
plt.ylabel('Countries', fontsize=12, fontweight= 'bold')
plt.show()

#showing top 10 genres on Netflix
genre_list = neteda['listed_in'].dropna().apply(lambda x: [i.strip() for i in x.split(',')])
all_genres = [genre for sublist in genre_list for genre in sublist]
top_genres = pd.Series(Counter(all_genres)).nlargest(10)

top_genres.plot(kind='barh', color='teal')
plt.title('Top 10 Genres on Netflix', fontweight='bold')
plt.xlabel('Number of Titles', fontweight='bold')
plt.show()

#content rating distribution
sns.countplot(data=neteda, y='rating', order=neteda['rating'].value_counts().index, palette='viridis')
plt.title('Content Ratings Distribution', fontweight='bold')
plt.ylabel('Rating', fontsize=12, fontweight='bold')
plt.show()

#saving the newly cleaned dataset to local
neteda.to_csv('netflix_cleaned.csv', index=False)
