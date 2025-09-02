import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ================== Load Dataset ==================
df = pd.read_csv(r"C:\Users\ABBAS COMPUTERS\Desktop\IMDB-Movie-Data.csv")

# ================== Basic Info ==================
print(df.head(10))
print(df.tail(10))
print("Number of Rows:", df.shape[0])
print("Number of Columns:", df.shape[1])
df.info()

print("Any missing value?", df.isnull().values.any())
print(df.isnull().sum())

# ================== Missing Values Heatmap ==================
plt.figure(figsize=(10, 6))
sns.heatmap(df.isnull(), cbar=False, cmap="viridis")
plt.title("Missing Values Heatmap")
plt.show()

# Missing values percentage
per_missing = df.isnull().sum() * 100 / len(df)
print("Missing values percentage per column:\n", per_missing)

# Drop missing values
df.dropna(axis=0, inplace=True)

# Check duplicates
print("Are there duplicate values?", df.duplicated().any())

# ================== Descriptive Stats ==================
print(df.describe())
print(df.describe(include='all'))
print("Columns List:", list(df.columns))

# ================== Long Movies ==================
long_movies = df.loc[df['Runtime (Minutes)'] >= 180, 'Title']
print("Movies >= 180 minutes:\n", long_movies)

# ================== Votes Analysis ==================
avg_votes_per_year = df.groupby('Year')['Votes'].mean().reset_index()
best_year = avg_votes_per_year.loc[avg_votes_per_year['Votes'].idxmax()]

print("Year with highest average votes:", best_year['Year'])
print("Average votes in that year:", best_year['Votes'])

plt.figure(figsize=(10, 6))
sns.barplot(x='Year', y='Votes', data=avg_votes_per_year, color='orange')
plt.title("Average Votes per Year")
plt.xticks(rotation=45)
plt.show()

# ================== Revenue Analysis ==================
avg_revenue_per_year = df.groupby('Year')['Revenue (Millions)'].mean()
best_year = avg_revenue_per_year.idxmax()
best_revenue = avg_revenue_per_year.max()

print("Year with highest avg revenue:", best_year)
print("Average revenue in that year (Millions):", best_revenue)

plt.figure(figsize=(10, 6))
sns.barplot(x=avg_revenue_per_year.index, y=avg_revenue_per_year.values, palette="viridis")
plt.title("Average Revenue per Year")
plt.xticks(rotation=45)
plt.show()

# ================== Top Directors by Rating ==================
avg_rating_per_director = df.groupby('Director')['Rating'].mean().reset_index()
avg_rating_per_director = avg_rating_per_director.sort_values(by='Rating', ascending=False)

print("Top 10 Directors:\n", avg_rating_per_director.head(10))

plt.figure(figsize=(12, 6))
sns.barplot(x='Rating', y='Director', data=avg_rating_per_director.head(10), palette='coolwarm')
plt.title("Top 10 Directors by Average Rating")
plt.show()

# ================== Movies per Year ==================
movies_per_year = df['Year'].value_counts().sort_index()
plt.figure(figsize=(10, 6))
sns.barplot(x=movies_per_year.index, y=movies_per_year.values, palette="crest")
plt.title("Number of Movies per Year")
plt.xticks(rotation=45)
plt.show()

# ================== Revenue Top Movies ==================
max_revenue_movie = df.loc[df['Revenue (Millions)'].idxmax(), ['Title', 'Revenue (Millions)']]
print("Most popular movie by revenue:\n", max_revenue_movie)

top10_revenue = df[['Title', 'Revenue (Millions)']].sort_values(by='Revenue (Millions)', ascending=False).head(10)
print("Top 10 movies by revenue:\n", top10_revenue)

plt.figure(figsize=(12, 6))
sns.barplot(x='Revenue (Millions)', y='Title', data=top10_revenue, palette='mako')
plt.title("Top 10 Movies by Revenue")
plt.show()

# ================== Top Rated Movies ==================
top10_rated = df[['Title', 'Director', 'Rating']].sort_values(by='Rating', ascending=False).head(10)
print("Top 10 Highest Rated Movies:\n", top10_rated)

plt.figure(figsize=(12, 6))
sns.barplot(x='Rating', y='Title', data=top10_rated, hue='Director', dodge=False, palette='Set2')
plt.title("Top 10 Highest Rated Movies & Directors")
plt.show()

# ================== Rating per Year ==================
avg_rating_per_year = df.groupby('Year')['Rating'].mean()
print("Average Rating per Year:\n", avg_rating_per_year)

plt.figure(figsize=(10, 6))
sns.barplot(x=avg_rating_per_year.index, y=avg_rating_per_year.values, palette="coolwarm")
plt.title("Average Movie Rating per Year")
plt.xticks(rotation=45)
plt.show()

# ================== Correlation Rating vs Revenue ==================
correlation = df['Rating'].corr(df['Revenue (Millions)'])
print("Correlation between Rating & Revenue:", correlation)

plt.figure(figsize=(10, 6))
sns.scatterplot(x='Rating', y='Revenue (Millions)', data=df, alpha=0.7)
plt.title("Does Rating Affect Revenue?")
plt.show()

# ================== Rating Categories ==================
def classify_rating(r):
    if r >= 8: return "Excellent"
    elif r >= 6: return "Good"
    else: return "Average"

df['Rating_Category'] = df['Rating'].apply(classify_rating)
print(df[['Title', 'Rating', 'Rating_Category']].head(15))

plt.figure(figsize=(8, 5))
sns.countplot(x='Rating_Category', data=df, palette="Set2")
plt.title("Movies Classification Based on Rating")
plt.show()

# ================== Genre Analysis ==================
action_movies_count = df['Genre'].str.contains("Action", case=False, na=False).sum()
print("Number of Action Movies:", action_movies_count)

all_genres = df['Genre'].str.split(',').sum()
genre_counts = pd.Series([g.strip() for g in all_genres]).value_counts()

print("Unique Genres:\n", set(genre_counts.index))
print("Number of films per genre:\n", genre_counts)

plt.figure(figsize=(12, 6))
sns.barplot(x=genre_counts.index, y=genre_counts.values, palette="viridis")
plt.xticks(rotation=45)
plt.title("Number of Films per Genre")
plt.show()
