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

#building a predictive ML
#selecting relevant features
features = ['release_year', 'duration', 'country', 'rating', 'listed_in']
target = 'type'

neteda_ml = neteda[features + [target]].dropna()
neteda_ml.head()

#preprocessing the data
def clean_duration(x):
    if pd.isnull(x):
        return np.nan
    x = str(x).strip()
    
    # Handle minutes
    if 'min' in x:
        try:
            return int(x.replace('min', '').strip())
        except:
            return np.nan
    
    # Handle seasons
    if 'Season' in x:
        try:
            return int(x.split()[0]) * 60
        except:
            return np.nan
    
    return np.nan

neteda_ml['duration_clean'] = neteda_ml['duration'].apply(clean_duration)


neteda_ml = neteda[features + [target]].copy()

#only drop rows where 'type' or 'duration' is missing
netneteda_ml =neteda_ml.dropna(subset=['type', 'duration'])
print(netneteda_ml.shape)

#converting values into numeric values representing minutes for movies and number of seasons for TV shows.
def clean_duration(x):
    if pd.isnull(x):
        return np.nan
    x = str(x).strip()
    if 'min' in x:
        try:
            return int(x.replace('min', '').strip())
        except:
            return np.nan
    if 'Season' in x:
        try:
            return int(x.split()[0]) * 60
        except:
            return np.nan
    return np.nan

neteda_ml['duration_clean'] = neteda_ml['duration'].apply(clean_duration)

print("Total rows before dropping:", len(neteda_ml))
print("Rows with valid duration:", neteda_ml['duration_clean'].notna().sum()) 

#replacing 'duration' with a new numeric column
neteda_ml['duration'] = neteda_ml['duration_clean']
neteda_ml = neteda_ml.drop(columns=['duration_clean'])

#cleaning 'duration'
#preparation of training models
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, classification_report
import plotly.figure_factory as ff 
import plotly.graph_objects as go 

def clean_duration(x):
    if pd.isnull(x):
        return np.nan
    x = str(x).strip()
    if 'min' in x:
        try:
            return int(x.replace('min', '').strip())
        except:
            return np.nan
    if 'Season' in x:
        try:
            return int(x.split()[0]) * 60
        except:
            return np.nan
    return np.nan

neteda_ml = neteda[['release_year', 'duration', 'country', 'rating', 'listed_in', 'type']].copy()
neteda_ml = neteda_ml.dropna(subset=['type', 'duration'])
neteda_ml['duration_clean'] = neteda_ml['duration'].apply(clean_duration)
neteda_ml['duration'] = neteda_ml['duration_clean']
neteda_ml = neteda_ml.drop(columns=['duration_clean'])

#encoding target
label_encoder = LabelEncoder()
neteda_ml['type_encoded'] = label_encoder.fit_transform(neteda_ml['type'])

#defining features and target
X = neteda_ml[['release_year', 'duration', 'country', 'rating', 'listed_in']]
y = neteda_ml['type_encoded']

#defining categorical and numeric columns
cat_features = ['country', 'rating', 'listed_in']
num_features = ['release_year', 'duration']

#creating preprocessor
preprocessor = ColumnTransformer([
    ('cat', Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore'))
    ]), cat_features),
    ('num', Pipeline([
        ('imputer', SimpleImputer(strategy='median'))
    ]), num_features)
])

#spliting test data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#building full model pipeline
model = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
])

#training the model
model.fit(X_train, y_train)

#checking if the models got trained
from sklearn.metrics import accuracy_score

y_pred = model.predict(X_test)
print("Model trained successfully!")
print("Accuracy:", round(accuracy_score(y_test, y_pred), 3))
print("\nClassification Report:\n", classification_report(y_test, y_pred, target_names=label_encoder.classes_))

#computing confusion matrix
cm = confusion_matrix(y_test, y_pred)
labels = sorted(y_test.unique())

#computing normalized confusion matrix (row-wise)
cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
cm_normalized = np.nan_to_num(cm_normalized)

#combining raw and normalized values for annotation
annot_text = [
    [f"{cm[i][j]}<br>({cm_normalized[i][j]*100:.1f}%)" for j in range(len(labels))]
    for i in range(len(labels))
]

#creating an annotated heatmap
fig = ff.create_annotated_heatmap(
    z=cm_normalized,
    x=labels,
    y=labels,
    annotation_text=annot_text,
    colorscale='Blues',
    showscale=True,
    hoverinfo='z'
)

#generating classification metrics
report = classification_report(y_test, y_pred, target_names=[str(l) for l in labels], output_dict=True)
metrics = pd.DataFrame(report).T.iloc[:-3, :] 

#creating a side panel for metrics
metrics_text = [
    f"<b>{label}</b><br>"
    f"Precision: {metrics.loc[str(label), 'precision']:.2f}<br>"
    f"Recall: {metrics.loc[str(label), 'recall']:.2f}<br>"
    f"F1-score: {metrics.loc[str(label), 'f1-score']:.2f}"
    for label in labels
]

#adding metric boxes next to the heatmap
for i, text in enumerate(metrics_text):
    fig.add_trace(go.Scatter(
        x=[len(labels) + 0.5],
        y=[i],
        mode="text",
        text=[text],
        textposition="middle left",
        showlegend=False
    ))

#updating the layout for better readability
fig.update_layout(
    title=dict(
        text=" Interactive Confusion Matrix (Counts + Normalized %)",
        x=0.35,
        font=dict(size=18)
    ),
    xaxis_title="Predicted Label",
    yaxis_title="True Label",
    xaxis=dict(showgrid=False),
    yaxis=dict(showgrid=False, autorange="reversed"),
    width=1000,
    height=750,
    margin=dict(l=80, r=300, t=120, b=80)
)

fig.show()

