import googleapiclient.discovery
import pandas as pd
import streamlit as st
from textblob import TextBlob
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle

tfidf = TfidfVectorizer()

file_path = 'knn_model.pkl'
file_path2 = 'tfidf_model.pkl'

with open(file_path, 'rb') as file:
    loaded_model = pickle.load(file)

with open(file_path2, 'rb') as file2:
    tfidf_model = pickle.load(file2)

api_service_name = "youtube"
api_version = "v3"
DEVELOPER_KEY = "AIzaSyAw1-i3tNmJR3RkPE3arIbXVIUwrW-e9xA"

youtube = googleapiclient.discovery.build(
    api_service_name, api_version, developerKey=DEVELOPER_KEY)

def get_video_comments(video_id, comment_limit=20):
    comments = []

    request = youtube.commentThreads().list(
        part="snippet",
        videoId=video_id,
        maxResults=100
    )
    i = 0

    while request and i < comment_limit:
        response = request.execute()

        for item in response['items']:
            comment = item['snippet']['topLevelComment']['snippet']
            comments.append(comment['textDisplay'])
            i += 1

        request = youtube.commentThreads().list_next(request, response)

    sentiment = []
    for comment in comments:
        this = tfidf_model.transform([comment])

        prediction = loaded_model.predict(this)

        if (prediction[0]==0):
            sentiment_label = 'neutral'
            
        elif (prediction[0]==1):
            sentiment_label = 'positive'
            
        elif (prediction[0]==-1):
            sentiment_label = 'negative'

        sentiment.append(sentiment_label)

    return comments, sentiment

st.title("Sentiment Analysis")

analysis_option = st.sidebar.selectbox(
    "Select analysis option:",
    ("Input Text", "YouTube Video Comments")
)

if analysis_option == "Input Text":
    st.title('Sentiment Analysis of Tweets with kNN')
    st.write("Enter a tweet, and we'll predict its sentiment!")

    input_tweet = st.text_input('Enter Tweet')

    if input_tweet:
        input = tfidf_model.transform([input_tweet])

        prediction = loaded_model.predict(input)

        if (prediction[0]==0):
            sentiment_label = 'neutral'
            
        elif (prediction[0]==1):
            sentiment_label = 'positive'
            
        elif (prediction[0]==-1):
            sentiment_label = 'negative'

        confidence_score = loaded_model.predict_proba(input)
            
        st.write(f'The sentiment of "{input_tweet}" is: {sentiment_label}')

        if confidence_score is not None:
            st.write(f'Confidence Score: {max(confidence_score[0]):.2f}')

elif analysis_option == "YouTube Video Comments":
    st.header("YouTube Comments Sentiment Analysis")

    video_id = st.text_input("Enter YouTube Video ID:")
    comment_limit = st.number_input(
        "Enter the comment limit (0 for all comments):", min_value=0, value=20)

    if video_id:
        if comment_limit == 0:
            comments, sentiment = get_video_comments(video_id)
        else:
            comments, sentiment = get_video_comments(video_id, comment_limit)

        if not comments:
            st.error("No comments found for the provided video ID.")
        else:
            st.subheader("Sentiment Distribution of Comments:")
            sentiment_counts = pd.Series(sentiment).value_counts()
            labels = sentiment_counts.index
            sizes = sentiment_counts.values

            fig1, ax1 = plt.subplots()
            ax1.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
            ax1.axis('equal')
            st.pyplot(fig1)
