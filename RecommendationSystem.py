import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

class MovieRecommendationSystem:
    def __init__(self):
        self.movies_df = None
        self.ratings_df = None
        self.movie_similarity = None
        self.user_movie_ratings = None

    def load_data(self, movies_file, ratings_file):
        """Load movie and rating data from CSV files."""
        self.movies_df = pd.read_csv(movies_file)
        self.ratings_df = pd.read_csv(ratings_file)

    def preprocess_data(self):
        """Preprocess the data to create a user-movie rating matrix."""
        self.user_movie_ratings = self.ratings_df.pivot(
            index='userId', columns='movieId', values='rating'
        ).fillna(0)

    def compute_similarity(self):
        """Compute similarity between movies using cosine similarity."""
        self.movie_similarity = cosine_similarity(self.user_movie_ratings.T)

    def get_movie_recommendations(self, movie_id, top_n=5):
        """Get top N movie recommendations based on a given movie."""
        movie_index = self.user_movie_ratings.columns.get_loc(movie_id)
        similar_movies = list(enumerate(self.movie_similarity[movie_index]))
        similar_movies = sorted(similar_movies, key=lambda x: x[1], reverse=True)
        similar_movies = similar_movies[1:top_n+1]  # Exclude the movie itself

        recommended_movies = []
        for i, score in similar_movies:
            movie_id = self.user_movie_ratings.columns[i]
            title = self.movies_df.loc[self.movies_df['movieId'] == movie_id, 'title'].iloc[0]
            recommended_movies.append((title, score))

        return recommended_movies

    def get_user_recommendations(self, user_id, top_n=5):
        """Get top N movie recommendations for a given user."""
        user_ratings = self.user_movie_ratings.loc[user_id]
        user_rated_movies = user_ratings[user_ratings > 0].index

        recommendations = []
        for movie_id in user_rated_movies:
            movie_recs = self.get_movie_recommendations(movie_id, top_n=top_n)
            recommendations.extend(movie_recs)

        # Remove duplicates and sort by similarity score
        recommendations = list(set(recommendations))
        recommendations.sort(key=lambda x: x[1], reverse=True)

        return recommendations[:top_n]

def main():
    # Example usage
    recommender = MovieRecommendationSystem()
    recommender.load_data('movies.csv', 'ratings.csv')
    recommender.preprocess_data()
    recommender.compute_similarity()

    # Get recommendations for a specific movie
    movie_id = 1  # Example movie ID
    movie_recs = recommender.get_movie_recommendations(movie_id)
    print(f"Recommendations based on movie {movie_id}:")
    for title, score in movie_recs:
        print(f"{title}: {score:.2f}")

    # Get recommendations for a specific user
    user_id = 1  # Example user ID
    user_recs = recommender.get_user_recommendations(user_id)
    print(f"\nRecommendations for user {user_id}:")
    for title, score in user_recs:
        print(f"{title}: {score:.2f}")

if __name__ == "__main__":
    main()