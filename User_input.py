import streamlit as st
import threading
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer


def input():
    nltk.download('vader_lexicon')  # Download the VADER lexicon for sentiment analysis


    def analyze_sentiment_with_rating(review, rating):
        sia = SentimentIntensityAnalyzer()
        sentiment_score = sia.polarity_scores(review)['compound']
        
        # Adjust sentiment score based on rating
        adjusted_score = sentiment_score + (rating - 3) * 0.1
        
        if adjusted_score >= 0.05:
            return 'Positive'
        elif adjusted_score <= -0.05:
            return 'Negative'
        else:
            return 'Neutral'

    def perform_sentiment_analysis(review, rating, output):
        sentiment_result = analyze_sentiment_with_rating(review, rating)
        output.append(sentiment_result)

    # Streamlit app
    st.title("Sentiment Analysis with Rating")

    # Text input for reviews
    user_review = st.text_area("Enter your review:")

    # Dropdown menu for ratings
    user_rating = st.selectbox("Select your rating:", options=[1, 2, 3, 4, 5], index=2)  # Default index is set to 2 (3rd option)

    # Button to trigger sentiment analysis asynchronously
    if st.button("Submit"):
        if user_review.lower() != 'exit':
            st.write("Performing sentiment analysis... Please wait.")
            
            # Asynchronous execution using threading
            output = []
            thread = threading.Thread(target=perform_sentiment_analysis, args=(user_review, user_rating, output))
            thread.start()
            
            # Wait for the thread to finish
            thread.join()

            sentiment_result = output[0]
            st.write(f"Your review: '{user_review}' | Your rating: {user_rating} | Sentiment: {sentiment_result}")
        else:
            st.write("Exiting the program.")
