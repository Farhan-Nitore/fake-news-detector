import joblib

# Ensure 'vectorizer' and 'best_models' exist in your workspace before running
try:
    joblib.dump(vectorizer, 'tfidf_vectorizer.pkl')
    # Save your choice of trained model (e.g., Logistic Regression or Voting Classifier)
    joblib.dump(best_models['Logistic Regression'], 'fake_news_model.pkl')
    print("Successfully exported 'tfidf_vectorizer.pkl' and 'fake_news_model.pkl'")
except NameError as e:
    print(f"Error: Ensure your notebook model training cells have been executed. Details: {e}")