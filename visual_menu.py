import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from nltk.sentiment import SentimentIntensityAnalyzer

def visual():
    # Load your dataset
    df = pd.read_csv('Datafiniti_Amazon_Consumer_Reviews_of_Amazon_Products.csv')
    df['reviews.date'] = pd.to_datetime(df['reviews.date'])
    df['year_month'] = df['reviews.date'].dt.to_period('M')

    # Function to plot Number of Reviews Over Time
    def plot_reviews_over_time():
        fig, ax = plt.subplots(figsize=(14, 6))
        sns.countplot(x='year_month', data=df, palette='viridis', ax=ax)
        plt.title('Number of Reviews Over Time')
        plt.xlabel('Year-Month')
        plt.ylabel('Count')
        plt.xticks(rotation=45, ha='right')
        st.pyplot(fig)

    # Function to plot Recommendation Percentage by Rating
    def plot_recommendation_percentage():
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(x='ratings', y='reviews.doRecommend', data=df, estimator=lambda x: sum(x) / len(x) * 100, ci=None, ax=ax)
        plt.title('Recommendation Percentage by Rating')
        plt.xlabel('Rating')
        plt.ylabel('Recommendation Percentage')
        st.pyplot(fig)

    # Function to plot Correlation Matrix
    def plot_correlation_matrix():
        #correlation_matrix = df.corr()
        correlation_matrix = df.select_dtypes(include='number').corr()
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', ax=ax)
        plt.title('Correlation Matrix')
        st.pyplot(fig)

    # Function to plot Distribution of Ratings
    def plot_distribution_of_ratings():
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.histplot(df['ratings'], bins=5, kde=True, ax=ax)
        plt.title('Distribution of Ratings')
        st.pyplot(fig)

    # Function to plot Categories Distribution
    def plot_categories_distribution():
        fig, ax = plt.subplots(figsize=(16, 12))
        sns.countplot(y='categories', data=df, order=df['categories'].value_counts().index, ax=ax)
        plt.title('Categories Distribution')
        plt.xlabel('Count')
        plt.ylabel('Categories')
        plt.xticks(rotation=45, ha='right')
        st.pyplot(fig)

    # Function to plot Count of Recommendations
    def plot_count_of_recommendations():
        plt.figure(figsize=(10, 6))
        sns.countplot(x='reviews.doRecommend', data=df)
        plt.title('Count of Recommendations')
        st.pyplot(plt)

    # Function to plot Rating vs. Recommendation
    def plot_rating_vs_recommendation():
        plt.figure(figsize=(10, 6))
        sns.countplot(x='ratings', hue='reviews.doRecommend', data=df)
        plt.title('Rating vs. Recommendation')
        plt.xlabel('Rating')
        plt.ylabel('Count')
        st.pyplot(plt)

    # Function to plot Compound Score by Amazon Reviews
    def plot_compound_score_by_reviews():
        # Create a SentimentIntensityAnalyzer object
        sia = SentimentIntensityAnalyzer()

        # Apply sentiment analysis to each review and create a new column 'compound'
        df['compound'] = df['reviewsText'].apply(lambda x: sia.polarity_scores(x)['compound'])

        # Plot barplot
        ax = sns.barplot(data=df, x='ratings', y='compound')
        ax.set_title('Compound score by Amazon Reviews')
        st.pyplot(ax.figure)

    # Function to plot Sentiment Distribution by Ratings
    def plot_sentiment_distribution_by_ratings():
        # Create a SentimentIntensityAnalyzer object
        sia = SentimentIntensityAnalyzer()

        # Apply sentiment analysis to each review and create new columns 'pos', 'neu', 'neg'
        df['pos'] = df['reviewsText'].apply(lambda x: sia.polarity_scores(x)['pos'])
        df['neu'] = df['reviewsText'].apply(lambda x: sia.polarity_scores(x)['neu'])
        df['neg'] = df['reviewsText'].apply(lambda x: sia.polarity_scores(x)['neg'])

        # Plot subplots
        fig, axs = plt.subplots(1, 3, figsize=(15, 5))
        sns.barplot(data=df, x='ratings', y='pos', ax=axs[0])
        sns.barplot(data=df, x='ratings', y='neu', ax=axs[1])
        sns.barplot(data=df, x='ratings', y='neg', ax=axs[2])

        axs[0].set_title('Positive')
        axs[1].set_title('Neutral')
        axs[2].set_title('Negative')
        plt.tight_layout()
        st.pyplot(fig)

    # Streamlit app
    st.title('Interactive Visualizations')

    # Define the visualizations
    visualizations = {
        'Number of Reviews Over Time': plot_reviews_over_time,
        'Recommendation Percentage by Rating': plot_recommendation_percentage,
        'Correlation Matrix': plot_correlation_matrix,
        'Distribution of Ratings': plot_distribution_of_ratings,
        'Categories Distribution': plot_categories_distribution,
        'Count of Recommendations': plot_count_of_recommendations,
        'Rating vs. Recommendation': plot_rating_vs_recommendation,
        'Compound Score by Amazon Reviews': plot_compound_score_by_reviews,
        'Sentiment Distribution by Ratings': plot_sentiment_distribution_by_ratings,  # Added new visualization
    }

    # Select a visualization from the dropdown
    selected_visualization = st.selectbox('Select Visualization', list(visualizations.keys()))

    # Display the selected visualization
    visualizations[selected_visualization]()
