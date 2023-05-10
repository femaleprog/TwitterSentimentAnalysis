
# TweetSentimentClassifier

TweetSentimentClassifier is a project that aims to analyze sentiment on Twitter by classifying tweets into sports and politics categories. The project utilizes a carefully curated dataset from Twitter, focusing on sports and politics-related tweets. The goal is to accurately classify tweets based on their content and provide valuable insights into public opinion tracking, trend analysis, and sentiment monitoring around topical issues.

## Features
- Preprocessing: The text data is preprocessed to remove special characters and normalize the text, ensuring consistency and improving the quality of the classification task.
- Feature Extraction: Various features are extracted from the tweets, including hashtags, mentions, and targeted terms such as political figures' names. These features provide additional context and help in better classification.
- CountVectorizer: The project employs the CountVectorizer from the scikit-learn library to convert the text data into numerical representations suitable for machine learning algorithms.
- Naive Bayes Classifier: The project utilizes the Multinomial Naive Bayes classifier to train and classify the tweets into sports and politics categories.
- Performance Evaluation: The model's performance is evaluated using classification metrics such as precision, recall, and F1-score to assess its effectiveness in accurately categorizing tweets.
- Predictions: Once the model is trained, it can be used to predict the category of new, unseen tweets. The predictions are saved in a CSV file that includes the tweet ID and the predicted label.

## Usage
1. Install the required dependencies listed in the `requirements.txt` file.
2. Prepare your training and test datasets, ensuring they are in the appropriate CSV format.
3. Preprocess the text data by removing special characters and normalizing the text using the provided preprocessing function.
4. Split the training data into training and validation sets for model evaluation.
5. Extract features from the tweets using hashtags, mentions, and targeted terms.
6. Convert the text data into numerical representations using the CountVectorizer.
7. Train the Naive Bayes classifier using the training data.
8. Evaluate the model's performance using the validation data.
9. Use the trained model to predict the category of the test tweets and save the predictions to a CSV file.

## Conclusion
TweetSentimentClassifier offers a powerful solution for sentiment analysis on Twitter, specifically focusing on classifying tweets into sports and politics categories. Through effective preprocessing, feature extraction, and machine learning techniques, the project achieves impressive performance in accurately categorizing tweets. Its potential applications include public opinion tracking, trend analysis, and sentiment monitoring around topical issues. By exploring natural language processing, text classification, and social media sentiment analysis, TweetSentimentClassifier provides an exciting opportunity to gain insights from Twitter data and understand the sentiments expressed in the online world.

Feel free to customize the README file based on your specific project details and requirements.
