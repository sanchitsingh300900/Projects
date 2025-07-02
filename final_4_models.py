import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.model_selection import train_test_split, GridSearchCV
import seaborn as sns
import matplotlib.pyplot as plt


def run_models():
    # Load your dataset
    data = pd.read_csv('Datafiniti_Amazon_Consumer_Reviews_of_Amazon_Products.csv')
    data = data[['reviewsText', 'ratings']]
    data["ratings"] = data["ratings"].apply(lambda x: 1 if x < 3 else 0)

    # Data split with fixed random seed for reproducibility
    xtrain, xtest, ytrain, ytest = train_test_split(data['reviewsText'], data['ratings'], test_size=0.3, random_state=42)

    # Vectorize using BOW
    vectorizer = CountVectorizer()
    xtrain_bow = vectorizer.fit_transform(xtrain).toarray()
    xtest_bow = vectorizer.transform(xtest).toarray()

    # Train Naive Bayes model
    clf_bow = GaussianNB().fit(xtrain_bow, ytrain)

    # Initialize Logistic Regression classifier
    logistic_reg = LogisticRegression()

    # Define the parameter grid for GridSearchCV
    param_grid_logistic = {'C': [0.1, 1, 10, 100], 'max_iter': [100, 500, 1000]}

    # Initialize GridSearchCV with 5-fold cross-validation
    grid_search_logistic = GridSearchCV(logistic_reg, param_grid_logistic, cv=5, scoring='accuracy')

    # Fit the model
    grid_search_logistic.fit(xtrain_bow, ytrain)

    # Get the best parameters from the grid search
    best_params_logistic = grid_search_logistic.best_params_

    # Train Logistic Regression model with the best parameters
    clf_logistic = LogisticRegression(**best_params_logistic)
    clf_logistic.fit(xtrain_bow, ytrain)

    # Streamlit app
    st.title("Model Comparison App")

    # Naive Bayes Model Section
    with st.expander("Naive Bayes Model"):
        # Predict using BOW
        prediction_bow = clf_bow.predict(xtest_bow)

        # Display confusion matrix
        st.subheader("Confusion Matrix:")
        st.write(confusion_matrix(ytest, prediction_bow))

        # Display accuracy
        accuracy = accuracy_score(ytest, prediction_bow)
        st.subheader("Accuracy:")
        st.write(f"{accuracy * 100:.2f}%")

        # Display classification report
        st.subheader("Classification Report:")
        classification_df = pd.DataFrame.from_dict(classification_report(ytest, prediction_bow, output_dict=True))
        st.dataframe(classification_df)

    # Logistic Regression Model Section
    with st.expander("Logistic Regression Model"):
        # Predict using BOW
        prediction_logistic = clf_logistic.predict(xtest_bow)

        # Display confusion matrix (smaller size)
        st.subheader("Confusion Matrix:")
        cm_logistic = confusion_matrix(ytest, prediction_logistic)
        st.write(cm_logistic)

        # Display accuracy
        accuracy_logistic = accuracy_score(ytest, prediction_logistic)
        st.subheader("Accuracy:")
        st.write(f"{accuracy_logistic * 100:.2f}%")

        # Display classification report (in table format)
        st.subheader("Classification Report:")
        classification_report_logistic = classification_report(ytest, prediction_logistic, output_dict=True)
        st.dataframe(pd.DataFrame.from_dict(classification_report_logistic))

        # Display best parameters
        st.subheader("Best Parameters:")
        st.text(best_params_logistic)

        # Plot confusion matrix as a heatmap (smaller size)
        st.subheader("Confusion Matrix Heatmap:")
        fig_logistic, ax_logistic = plt.subplots(figsize=(3, 2))
        sns.heatmap(cm_logistic, annot=True, fmt="d", cmap="Blues", cbar=False, annot_kws={"size": 8}, ax=ax_logistic)
        plt.title("Confusion Matrix (Logistic Regression)")
        plt.xlabel("Predicted")
        plt.ylabel("True")
        st.pyplot(fig_logistic)

    # Support Vector Machine (SVM) Model Section
    with st.expander("Support Vector Machine (SVM) Model"):
        # Parameter grid for GridSearchCV
        param_grid_svm = {'C': [0.1, 1, 10, 100], 'kernel': ['linear', 'rbf', 'poly'], 'gamma': ['scale', 'auto']}

        # Initialize SVM classifier
        svc = SVC()

        # Initialize GridSearchCV with 5-fold cross-validation and parallelize
        grid_search_svm = GridSearchCV(svc, param_grid_svm, cv=5, scoring='accuracy', n_jobs=-1)

        # Fit the model
        grid_search_svm.fit(xtrain_bow, ytrain)

        # Get the best parameters from the grid search
        best_params_svm = grid_search_svm.best_params_
        st.write("Best Parameters (SVM):", best_params_svm)

        # Train Support Vector Machine (SVM) model with the best parameters
        clf_svm = SVC(**best_params_svm)
        clf_svm.fit(xtrain_bow, ytrain)

        # Predict using BOW
        prediction_svm = clf_svm.predict(xtest_bow)

        # Display accuracy
        accuracy_svm = accuracy_score(ytest, prediction_svm)
        st.subheader("Accuracy (SVM):")
        st.write(f"Accuracy: {accuracy_svm * 100:.2f}%")

        # Display classification report
        st.subheader("Classification Report (SVM):")
        classification_report_svm = classification_report(ytest, prediction_svm, output_dict=True)
        st.dataframe(pd.DataFrame.from_dict(classification_report_svm))

        # Generate confusion matrix
        cm_svm = confusion_matrix(ytest, prediction_svm)

        # Plot confusion matrix as a heatmap
        st.subheader("Confusion Matrix Heatmap (SVM):")
        fig_svm, ax_svm = plt.subplots(figsize=(6, 4))
        sns.heatmap(cm_svm, annot=True, fmt="d", cmap="Blues", cbar=False, annot_kws={"size": 10}, ax=ax_svm)
        plt.title("Confusion Matrix (SVM)")
        plt.xlabel("Predicted")
        plt.ylabel("True")
        st.pyplot(fig_svm)

        # Display confusion matrix as a table
        st.subheader("Confusion Matrix Table (SVM):")
        cm_df_svm = pd.crosstab(ytest, prediction_svm, rownames=['Actual'], colnames=['Predicted'], margins=True)
        st.dataframe(cm_df_svm)

    # RandomForestClassifier Model Section
    with st.expander("RandomForestClassifier Model"):
        # Load dataset
        df = pd.read_csv('Datafiniti_Amazon_Consumer_Reviews_of_Amazon_Products.csv')

        # Convert 'reviews.rating' to numeric (replace non-convertible values with -1)
        df['ratings'] = pd.to_numeric(df['ratings'], errors='coerce', downcast='integer').fillna(-1)

        # Select features for training
        features = ['reviews.numHelpful', 'ratings']  # Add more features as needed

        # Use the selected features for X
        X = df[features]

        # Target variable
        y = df['reviews.doRecommend']

        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42) 

        # Define the parameter grid for GridSearchCV
        param_grid_rf = {
            'n_estimators': [50, 100, 150],
            'max_depth': [None, 10, 20],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }

        # Initialize RandomForestClassifier
        rf_classifier = RandomForestClassifier()

        # Initialize GridSearchCV with 5-fold cross-validation
        grid_search_rf = GridSearchCV(rf_classifier, param_grid_rf, cv=5, scoring='accuracy')

        # Fit the model
        grid_search_rf.fit(X_train, y_train)

        # Get the best parameters from the grid search
        best_params_rf = grid_search_rf.best_params_
        st.write("Best Parameters (RandomForest):", best_params_rf)

        # Train RandomForestClassifier with the best parameters
        best_rf_classifier = RandomForestClassifier(**best_params_rf)
        best_rf_classifier.fit(X_train, y_train)

        # Make predictions on the test set
        y_pred_rf = best_rf_classifier.predict(X_test)

        # Evaluate accuracy
        accuracy_rf = accuracy_score(y_test, y_pred_rf)
        st.write(f'Accuracy (RandomForest): {accuracy_rf * 100:.2f}%')

        # Display classification report
        st.subheader("Classification Report (RandomForest):")
        classification_report_rf = classification_report(y_test, y_pred_rf, output_dict=True)
        st.dataframe(pd.DataFrame.from_dict(classification_report_rf))

        # Generate confusion matrix
        cm_rf = confusion_matrix(y_test, y_pred_rf)

        # Plot confusion matrix as a heatmap
        st.subheader("Confusion Matrix Heatmap (RandomForest):")
        fig_rf, ax_rf = plt.subplots(figsize=(6, 4))
        sns.heatmap(cm_rf, annot=True, fmt="d", cmap="Blues", cbar=False, annot_kws={"size": 10}, ax=ax_rf)
        plt.title("Confusion Matrix (RandomForest)")
        plt.xlabel("Predicted")
        plt.ylabel("True")
        st.pyplot(fig_rf)

        # Display confusion matrix as a table
        st.subheader("Confusion Matrix Table (RandomForest):")
        cm_df_rf = pd.crosstab(y_test, y_pred_rf, rownames=['Actual'], colnames=['Predicted'], margins=True)
        st.dataframe(cm_df_rf)
