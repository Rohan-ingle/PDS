import googleapiclient.discovery
import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import KNeighborsClassifier
import pickle
import re
import matplotlib.pyplot as plt
from textblob import TextBlob 

with open('knn_model.pkl', 'rb') as knn_file:
    loaded_knn_model = pickle.load(knn_file)

with open('tfidf_model.pkl', 'rb') as tfidf_file:
    tfidf_vectorizer = pickle.load(tfidf_file)

api_service_name = "youtube"
api_version = "v3"
DEVELOPER_KEY = "AIzaSyAw1-i3tNmJR3RkPE3arIbXVIUwrW-e9xA"
youtube = googleapiclient.discovery.build(api_service_name, api_version, developerKey=DEVELOPER_KEY)

def filter(text):
    whitelist = "1234567890abcdefghijklmnopqrstuvwxyz .,"
    clr_str = "".join(c for c in text if c in whitelist)
    return clr_str

def get_video_id_from_url(video_url):
    video_id_match = re.search(r'(?<=v=)[^&]+', video_url)
    return video_id_match.group(0) if video_id_match else None

def get_sentiment_textblob(text):
    analysis = TextBlob(text)
    sentiment = analysis.sentiment.polarity

    if sentiment > 0:
        return 'positive'
    elif sentiment < 0:
        return 'negative'
    else:
        return 'neutral'
    
def get_related_videos(video_url, num_recommendations=3):
    video_id = get_video_id_from_url(video_url)
    if video_id:
        video_info = youtube.videos().list(
            part="snippet",
            id=video_id
        ).execute()

        if video_info.get("items"):
            video_info = video_info["items"][0]["snippet"]
            video_title = video_info["title"]
            video_description = video_info.get("description", "")
            search_query = f"{video_title} {video_description}"

            search_response = youtube.search().list(
                q=search_query,
                type="video",
                maxResults=num_recommendations,
                part="snippet"
            ).execute()

            recommended_videos = []
            for item in search_response.get("items", []):
                video_info = {
                    "title": item["snippet"]["title"],
                    "channel_name": item["snippet"]["channelTitle"],
                    "video_url": f"https://www.youtube.com/watch?v={item['id']['videoId']}"
                }
                recommended_videos.append(video_info)

            return recommended_videos


def get_video_comments(video_url, comment_limit=0, use_textblob=False):
    video_id = get_video_id_from_url(video_url)
    if not video_id:
        st.error("Invalid YouTube video URL. Please provide a valid URL.")
        return [], []

    comments = []

    request = youtube.commentThreads().list(
        part="snippet",
        videoId=video_id,
        maxResults=100
    )

    while request:
        response = request.execute()
        if comment_limit == 0 or len(comments) >= comment_limit:
            break  # Fetch all comments or reach the specified limit
        else:
            a = 0
            for item in response['items']:
                if a < comment_limit:
                    comment = item['snippet']['topLevelComment']['snippet']
                    comments.append(filter(comment['textDisplay']))
                    a = a + 1

        request = youtube.commentThreads().list_next(request, response)

    sentiments = []
    if use_textblob:
        for comment in comments:
            sentiment = get_sentiment_textblob(comment)
            sentiments.append(sentiment)
    else:
        for comment in comments:
            tfidf_text = tfidf_vectorizer.transform([comment])
            prediction = loaded_knn_model.predict(tfidf_text)
            sentiment_label = 'neutral' if prediction[0] == 0 else 'positive' if prediction[0] == 1 else 'negative'
            sentiments.append(sentiment_label)
    st.write(f"Total comments tokanized: {len(comments)}")
    if comment_limit==0:
        return comments, sentiments
    else:
        return comments[:comment_limit], sentiments[:comment_limit]


# st.title("Sentiment Analysis")

analysis_option = st.sidebar.selectbox(
    "Select analysis option:",
    ("Input Text", "YouTube Video Comments",)
)

if analysis_option == "Input Text":
    st.title('Sentiment Analysis of Text with kNN/TextBlob')
    st.write("Enter a tweet, and we'll predict its sentiment!")

    input_tweet = st.text_input('Enter Text')
    sentiment_analysis_method = st.selectbox(
        "Select Sentiment Analysis Method:",
        ("kNN", "TextBlob")
    )

    if input_tweet:
        if sentiment_analysis_method == "kNN":
            tfidf_input = tfidf_vectorizer.transform([filter(input_tweet)])
            prediction = loaded_knn_model.predict(tfidf_input)
            sentiment_label = 'neutral' if prediction[0] == 0 else 'positive' if prediction[0] == 1 else 'negative'
            confidence_score = loaded_knn_model.predict_proba(tfidf_input)
        else:
            sentiment_label = get_sentiment_textblob(input_tweet)
            confidence_score = None

        st.write(f'The sentiment of "{input_tweet}" is: {sentiment_label}')

        if confidence_score is not None:
            st.write(f'Confidence Score: {max(confidence_score[0]):.2f}')

elif analysis_option == "YouTube Video Comments":
    st.header("YouTube Comments Sentiment Analysis")

    video_url = st.text_input("Enter YouTube Video URL:")
    comment_limit = st.number_input(
        "Enter the comment limit (0 for all comments):", min_value=0, value=20)

    sentiment_analysis_method = st.selectbox(
        "Select Sentiment Analysis Method:",
        ("TextBlob", "kNN")
    )

    use_textblob = sentiment_analysis_method == "TextBlob"

    if video_url:
        if comment_limit == 0:
            comments, sentiment = get_video_comments(video_url, 1000000000,use_textblob=use_textblob)
        else:
            comments, sentiment = get_video_comments(video_url, comment_limit, use_textblob)
        st.write(f"Total comments analyzed : {len(comments)}")
        
        if st.button("Get Recommendations"):
            recommendations = get_related_videos(video_url)
            if recommendations:
                st.subheader("Recommended Videos:")
                for video in recommendations:
                    st.write(f"Title: [{video['title']}]({video['video_url']})")
                    st.write(f"Channel: {video['channel_name']}")

        if not comments:
            st.error("No comments found for the provided video URL.")
        else:
            st.subheader("Sentiment Distribution of Comments:")
            if len(comments) > 0:
                sentiment_counts = pd.Series(sentiment).value_counts()
                labels = sentiment_counts.index
                sizes = sentiment_counts.values

                fig1, ax1 = plt.subplots()
                ax1.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
                ax1.axis('equal')
                st.pyplot(fig1)
            else:
                st.write("Only one comment available. Sentiment analysis not performed.")

