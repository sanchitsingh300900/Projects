import streamlit as st
import visual_menu # type: ignore
import User_input # type: ignore
import final_4_models # type: ignore

# Set page configuration
st.set_page_config(
    page_title="Sentiment Analysis App",
    page_icon=":smiley:",
    layout="wide"
)

def home():
    st.write("""
    Sentiment analysis is a technique that detects the underlying sentiment in a piece of text. It is the process of classifying text as either positive, negative, or neutral. Machine learning techniques are used to evaluate a piece of text and determine the sentiment behind it

    This app provides three main features:

    1. **Visualization:** Explore interactive visualizations related to the dataset.
    2. **Models Comparison:** Compare the performance of four machine learning models.
    3. **User Input:** Input your own text and see the sentiment prediction.

    Use the sidebar to navigate between different features. Enjoy exploring sentiment analysis!
    """)

def main():
    #st.title("Sentiment Analysis App")  # Updated title

    # Create a sidebar with options
    menu_option = st.sidebar.radio("Select an option", ("Home", "Visualization", "Models Comparison", "User Input"))

    if menu_option == "Home":
        st.title("Sentiment Analysis App")  # Updated title
        home()

    elif menu_option == "Visualization":
        visual_menu.visual()

    elif menu_option == "Models Comparison":
        final_4_models.run_models()

    elif menu_option == "User Input":
        User_input.input()

if __name__ == "__main__":
    main()
