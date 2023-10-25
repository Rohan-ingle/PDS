import googleapiclient.discovery
import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import KNeighborsClassifier
import pickle
import re
import matplotlib.pyplot as plt
from textblob import TextBlob 

st.set_page_config(
    page_title="Sentiment Analysis App"
)

with open('knn_model1.pkl', 'rb') as knn_file1:
    loaded_knn_model1 = pickle.load(knn_file1)

with open('knn_model2.pkl', 'rb') as knn_file2:
    loaded_knn_model2 = pickle.load(knn_file2)

with open('tfidf_model1.pkl', 'rb') as tfidf_file1:
    tfidf_vectorizer1 = pickle.load(tfidf_file1)

with open('tfidf_model2.pkl', 'rb') as tfidf_file2:
    tfidf_vectorizer2 = pickle.load(tfidf_file2)

api_service_name = "youtube"
api_version = "v3"
DEVELOPER_KEY = "AIzaSyDrO-T21_wPDt6bTUAYq8JVcay9rXuQ4Ro"
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
        sentiment_analysis_method2 = st.selectbox(
                "Select Sentiment Analysis Method:",
                ("accuracy = 0.40", "accuracy = 0.67")
            )

        if sentiment_analysis_method2 == "accuracy = 0.40" :
            for comment in comments:
                tfidf_text = tfidf_vectorizer1.transform([comment])
                prediction = loaded_knn_model1.predict(tfidf_text)
                sentiment_label = 'neutral' if prediction[0] == 0 else 'positive' if prediction[0] == 1 else 'negative'
                sentiments.append(sentiment_label)
        else:
            for comment in comments:
                tfidf_text = tfidf_vectorizer2.transform([comment])
                prediction = loaded_knn_model2.predict(tfidf_text)
                sentiment_label = 'neutral' if prediction[0] == 0 else 'positive' if prediction[0] == 1 else 'negative'
                sentiments.append(sentiment_label)
    st.write(f"Total comments tokenized: {len(comments)}")
    if comment_limit==0:
        return comments, sentiments
    else:
        return comments[:comment_limit], sentiments[:comment_limit]


# st.title("Sentiment Analysis")

analysis_option = st.sidebar.selectbox(
    "Select analysis option:",
    ("Input Text", "YouTube Video Comments", "About")
)

if analysis_option == "Input Text":
    st.title('Sentiment Analysis of Text')
    st.write("Enter a text, and we'll predict its sentiment!")

    input_tweet = st.text_input('Enter Text')
    sentiment_analysis_method = st.selectbox(
        "Select Sentiment Analysis Method:",
        ("kNN", "TextBlob")
    )

    if input_tweet:
        if sentiment_analysis_method == "kNN":

            sentiment_analysis_method2 = st.selectbox(
                "Select model:",
                ( "Model 1 : accuracy = 0.67","Model 1 : accuracy = 0.40")
            )

            if sentiment_analysis_method2 == "Model 1 : accuracy = 0.40" :

                tfidf_input = tfidf_vectorizer1.transform([filter(input_tweet)])
                prediction = loaded_knn_model1.predict(tfidf_input)
                sentiment_label = 'neutral' if prediction[0] == 0 else 'positive' if prediction[0] == 1 else 'negative'
                confidence_score = loaded_knn_model1.predict_proba(tfidf_input)

            else:
                tfidf_input = tfidf_vectorizer2.transform([filter(input_tweet)])
                prediction = loaded_knn_model2.predict(tfidf_input)
                sentiment_label = 'neutral' if prediction[0] == 0 else 'positive' if prediction[0] == 1 else 'negative'
                confidence_score = loaded_knn_model2.predict_proba(tfidf_input)

        else:
            sentiment_label = get_sentiment_textblob(input_tweet)
            confidence_score = None

        st.write(f'The sentiment of "{input_tweet}" is: {sentiment_label}')

        if confidence_score is not None:
            st.write(f'Confidence Score: {max(confidence_score[0]):.2f}')

elif analysis_option == "YouTube Video Comments":
    st.header("YouTube Comments Sentiment Analysis")

    video_url = st.text_input("Enter YouTube Video URL:")

    if st.button("Get Recommendations"):
            recommendations = get_related_videos(video_url)
            if recommendations:
                st.subheader("Recommended Videos:")
                for video in recommendations:
                    st.write(f"Title: [{video['title']}]({video['video_url']})")
                    st.write(f"Channel: {video['channel_name']}")

    comment_limit = st.number_input(
        "Enter the comment limit (0 for all comments):", min_value=0, value=20)

    sentiment_analysis_method = st.selectbox(
        "Select Sentiment Analysis Method:",
        ("kNN", "TextBlob")
    )

    use_textblob = sentiment_analysis_method == "TextBlob"

    if video_url:
        if comment_limit == 0:
            comments, sentiment = get_video_comments(video_url, 1000000000,use_textblob=use_textblob)
        else:
            comments, sentiment = get_video_comments(video_url, comment_limit, use_textblob)
        st.write(f"Total comments analyzed : {len(comments)}")

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

elif analysis_option == "About":
    st.write("This is a sentiment analysis tool using kNN and TextBlob.")
    about_option = st.selectbox("Select an option:", ("About Model 1", "About Model 2"))
    
    if about_option == "About Model 1":
        with open('metrics1.txt', 'r') as metrics_file:
            metrics_data = metrics_file.read()
        st.download_button(label="Download Model 1 Metrics", data=metrics_data, key="model1_metrics")

    elif about_option == "About Model 2":
        with open('metrics2.txt', 'r') as metrics_file:
            metrics_data = metrics_file.read()
        st.download_button(label="Download Model 2 Metrics", data=metrics_data, key="model2_metrics")
