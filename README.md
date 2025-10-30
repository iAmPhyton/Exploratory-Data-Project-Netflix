Netflix Content Analysis & Machine Learning Project
Overview:
This project explores the Netflix Movies and TV Shows dataset, performing data cleaning, exploratory data analysis (EDA), and machine learning to understand content trends and predict content types. The analysis aims to uncover insights such as content growth over time, genre distribution, and factors influencing whether a title is a Movie or TV Show.

Project Objectives:
Data Cleaning — Handle missing values, parse dates, and standardize duration formats.
Exploratory Data Analysis (EDA) — Visualize trends such as content growth by year, ratings, and country distribution.
Feature Engineering — Extract key features from date and text fields (e.g., release year, duration, and type).
Machine Learning Model — Train a classification model to predict whether a title is a Movie or a TV Show.
Visualization — Used Plotly for a rich, interactive dashboard and confusion matrix visualization with performance metrics.
Dataset Source: Netflix Titles Dataset on Kaggle

Key Insights:
Netflix's content catalog has grown rapidly since 2014, with the largest surge in original programming after 2016.
The most common content category is International Movies, followed by Dramas and Comedies.
Movies dominate Netflix’s library, though TV Shows have increased steadily in recent years.
The United States and India remain the top-producing countries.

Machine Learning Implementation Model:
Random Forest Classifier

Tech Stack:
Category	Tools / Libraries
Language	Python 3.x
Data Handling	pandas, numpy
Visualization: matplotlib, seaborn, plotly
Machine Learning	scikit-learn
Environment	Jupyter Notebook

📁 Project Structure
Netflix-Analysis/
│
├── data/
│   └── netflix_titles.csv
│
├── notebooks/
│   └── 01_data_cleaning_and_eda.ipynb
│   └── 02_model_training_and_evaluation.ipynb
│
├── visuals/
│   └── confusion_matrix_plotly.html
│   └── content_growth_plot.html
│
├── README.md
└── requirements.txt

Author:
Chukwuemeka Eugene Obiyo
Data Science | Machine Learning | Visualization Enthusiast
praise609@gmail.com
