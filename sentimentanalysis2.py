import pandas as pd
from textblob import TextBlob
import re
import matplotlib.pyplot as plt
import seaborn as sns
import langid
from googletrans import Translator

# Read tweets from CSV file
def read_tweets_from_csv(file_path):
    df = pd.read_csv(file_path)
    
    # Handle NaN values in the 'text' column
    df['text'] = df['text'].astype(str)
    
    return df['text'].tolist()

# Preprocess the text data
def preprocess_text(text):
    text = re.sub(r'http\S+', '', text)  # Remove URLs
    text = re.sub(r'@[A-Za-z0-9]+', '', text)  # Remove mentions
    text = re.sub(r'#', '', text)  # Remove hashtags
    text = re.sub(r'RT[\s]+', '', text)  # Remove retweets
    text = re.sub(r'\n', '', text)  # Remove newlines
    text = re.sub(r'[^A-Za-z\s]', '', text)  # Remove non-alphanumeric characters
    text = text.lower()  # Convert to lowercase
    return text

# Translate text to English based on language detection
def translate_to_english(text):
    lang, confidence = langid.classify(text)

    if lang == 'ta':
        # Translate pure Tamil to English
        translator = Translator()
        translation = translator.translate(text, src='ta', dest='en')
        return translation.text
    elif lang == 'en':
        # Translate English-written Tamil (Thanglish) to English
        # This assumes that Thanglish contains Tamil characters in an English script
        return text
    else:
        # Leave as is for other languages (including English)
        return text

# Perform sentiment analysis using TextBlob
def analyze_sentiment(text):
    analysis = TextBlob(text)
    return analysis.sentiment.polarity

# Categorize sentiment as positive, negative, or neutral
def categorize_sentiment(score):
    if score > 0:
        return 'positive'
    elif score < 0:
        return 'negative'
    else:
        return 'neutral'

# Visualize sentiment distribution
def visualize_sentiment(sentiments, party_name):
    plt.figure(figsize=(12, 6))

    # Plot stacked bar chart
    sns.histplot(sentiments, bins=[-1, -0.5, 0, 0.5, 1], kde=True, color='skyblue', multiple="stack")
    plt.xlabel(f'{party_name} Sentiment Score')
    plt.ylabel('Number of Tweets')
    plt.title(f'Sentiment Analysis - {party_name}')

    plt.tight_layout()
    plt.show()

# Visualize overall sentiment distribution as a pie chart
def visualize_pie_chart(sentiments, party_name):
    plt.figure(figsize=(6, 6))
    
    labels = ['Positive', 'Neutral', 'Negative']
    sentiment_distribution = [sentiments.count('positive'), sentiments.count('neutral'), sentiments.count('negative')]
    
    plt.pie(sentiment_distribution, labels=labels, autopct='%1.1f%%', colors=['lightgreen', 'lightgrey', 'tomato'])
    plt.title(f'Overall Sentiment Distribution - {party_name}')

    plt.show()

# Main function for sentiment analysis and comparison
def main():
    # Replace with your provided file paths
    file_paths = [
        'C:\\Users\\INIYA VASANTHAN\\Desktop\\election\\aiadmk\\aiadmknew.csv',
        'C:\\Users\\INIYA VASANTHAN\\Desktop\\election\\dmk\\dmknew.csv',
        'C:\\Users\\INIYA VASANTHAN\\Desktop\\election\\tvk\\tvkvijaynew.csv'
    ]
    
    all_sentiments = []

    highest_positive_percentage = 0
    winning_party = ""

    party_names = ["AIADMK", "DMK", "TVKVijay"]

    for i, file_path in enumerate(file_paths):
        tweets = read_tweets_from_csv(file_path)

        sentiments = []
        for tweet in tweets:
            processed_text = preprocess_text(tweet)

            # Translate text to English based on language detection
            processed_text = translate_to_english(processed_text)

            sentiment_score = analyze_sentiment(processed_text)
            sentiments.append(categorize_sentiment(sentiment_score))

        all_sentiments.append(sentiments)

        # Calculate overall sentiment percentage for each party
        overall_sentiments = sentiments
        positive_percentage = overall_sentiments.count('positive') / len(overall_sentiments) * 100
        negative_percentage = overall_sentiments.count('negative') / len(overall_sentiments) * 100
        neutral_percentage = overall_sentiments.count('neutral') / len(overall_sentiments) * 100

        print(f"\nSentiment Analysis Results for {party_names[i]}")
        print(f"Positive Percentage: {positive_percentage:.2f}%")
        print(f"Negative Percentage: {negative_percentage:.2f}%")
        print(f"Neutral Percentage: {neutral_percentage:.2f}%")

        # Output final sentiment category for each party
        if positive_percentage > negative_percentage and positive_percentage > neutral_percentage:
            print(f"Overall Sentiment for {party_names[i]}: Positive")
        elif negative_percentage > positive_percentage and negative_percentage > neutral_percentage:
            print(f"Overall Sentiment for {party_names[i]}: Negative")
        else:
            print(f"Overall Sentiment for {party_names[i]}: Neutral")

        # Update highest positive percentage and winning party
        if positive_percentage > highest_positive_percentage:
            highest_positive_percentage = positive_percentage
            winning_party = party_names[i]

        # Visualize sentiment distribution for each party
        visualize_sentiment(sentiments, party_names[i])

        # Visualize overall sentiment distribution as a pie chart
        visualize_pie_chart(sentiments, party_names[i])

    # Output final winner based on sentiment analysis
    if winning_party:
        print(f"\nThe party with the highest chance of winning the election is {winning_party} with the highest positive sentiment percentage.")
    else:
        print("All parties have a higher neutral sentiment. The election outcome is neutral.")


if __name__ == "__main__":
    main()
