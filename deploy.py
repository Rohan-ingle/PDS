import googleapiclient.discovery
import pandas as pd
import streamlit as st
import pickle
import re
from textblob import TextBlob
import plotly.express as px
import plotly.graph_objects as go

##################### INITIALIZE UI #####################

st.set_page_config(
    page_title="Sentiment Analysis App",
    page_icon="youtube.png"
)
##################### LOAD MODELS #####################

with open(r'tfidf_model.pkl', 'rb') as temp_tfidf:
    tfidf = pickle.load(temp_tfidf)

with open(r'knn_model.pkl', 'rb') as temp_knn_model:
    knn_model = pickle.load(temp_knn_model)

with open(r'dt_model.pkl', 'rb') as temp_dt_model:
    dt_model = pickle.load(temp_dt_model)

with open(r'lr_model.pkl', 'rb') as temp_lr_model:
    lr_model = pickle.load(temp_lr_model)

with open(r'nb_model.pkl', 'rb') as temp_nb_model:
    nb_model = pickle.load(temp_nb_model)
####################### MODEL DICTIONARY #######################

model_dict = {"KNN" : knn_model,
            "Decision Tree": dt_model,
            "Logistic Regression" : lr_model,
            "Naive Bayes" : nb_model}


####################### UI STUFF #######################

# Define main options and sub-options
main_options = (
    "Input Text",
    "YouTube Video Comments"
)

input_text_sub_options = (
    "KNN",
    "Decision Tree",
    "Logistic Regression",
    "Naive Bayes"
)

##################### INITIALIZE YOUTUBE API AND OTHER FUNCTIONS #####################

api_service_name = "youtube"
api_version = "v3"
DEVELOPER_KEY = "AIzaSyDrO-T21_wPDt6bTUAYq8JVcay9rXuQ4Ro"
youtube = googleapiclient.discovery.build(api_service_name, api_version, developerKey=DEVELOPER_KEY)


# Define a function to get video ID from URL

def get_video_id_from_url(video_url):
    video_id_match = re.search(r'(?<=v=)[^&]+', video_url)
    return video_id_match.group(0) if video_id_match else None


# Define a function to get related videos

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

##################### PREPROCESSING FUNCTION #####################

def filter(text):
    if len(text)==1:
        return ""
    try :
        int(text)
        return ""
    except:
        cleaned_text = re.sub(r'[^\w\s]', '', str(text))
        cleaned_text = re.sub('_', '', cleaned_text)
        return cleaned_text

def cat_sentiment(pred):
    return'neutral' if pred[0] == 0 else 'positive' if pred[0] == 1 else 'negative'

# def filter(text):
#     whitelist = "1234567890abcdefghijklmnopqrstuvwxyz .,ABCDEFGHIJKLMNOPQRSTUVWXYZ"
#     clr_str = "".join(c for c in text if c in whitelist)
#     return clr_str

##################### SCRAPE COMMENTS -- FILTER -- APPEND SENTIMENTS #####################

def get_video_comments(video_url, comment_limit=0, run_all_models=False):
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
        for item in response['items']:
            if comment_limit == 0 or len(comments) < comment_limit:
                comment = item['snippet']['topLevelComment']['snippet']
                comments.append(filter(comment['textDisplay']))
            else:
                break

        if len(comments) >= comment_limit:
            break

        request = youtube.commentThreads().list_next(request, response)

    sentiment_by_model = {"KNN": [], "Decision Tree": [], "Logistic Regression": [], "Naive Bayes": []}

    if run_all_models:
        for model in input_text_sub_options:
            model_ = model_dict[model]
            for comment in comments:
                sentiment, _ = process_by(model_, comment)
                sentiment_by_model[model].append(sentiment)

        st.write(f"Total comments tokenized: {len(comments)}") 
        return comments, sentiment_by_model       

    else:
        for comment in comments:
            sentiment, _ = process_by(model_dict[sentiment_analysis_method2], comment)
            sentiment_by_model[sentiment_analysis_method2].append(sentiment)
        st.write(f"Total comments tokenized: {len(comments)}")
        return comments, sentiment_by_model

    

# Create the Streamlit app interface
st.title("Sentiment Analysis")

analysis_option = st.sidebar.selectbox(
    "Select analysis option:",
    main_options
)

############################## DEFINE SENTIME PROCESSING BY MODEL FUNCTION ##############################

def process_by(model, input_text):
    tfidf_input = tfidf.transform([filter(input_text)])
    prediction = model.predict(tfidf_input)
    sentiment_label = 'negative' if prediction[0] == -1 else 'positive' if prediction[0] == 1 else 'neutral'
    confidence_score = model.predict_proba(tfidf_input)

    return sentiment_label, confidence_score

# x,yp1 = process_by("KNN", "Hello")
# print(x)
if analysis_option == "Input Text":
    st.title('Sentiment Analysis of Text')
    st.write("Enter a text, and we'll predict its sentiment!")

    input_text = st.text_input('Enter Text')
    sentiment_analysis_method = st.selectbox(
        "Select Sentiment Analysis Method:",
        input_text_sub_options
    )

    if st.button("Run Analysis"):
        if input_text:
            sentiment_label, confidence_score = process_by(model_dict[sentiment_analysis_method], input_text)

            st.write(f'The sentiment of "{input_text}" is: {sentiment_label}')

            if confidence_score is not None:
                st.write(f'Confidence Score: {max(confidence_score[0]):.2f}')

elif analysis_option == "YouTube Video Comments":
    st.header("YouTube Comments Sentiment Analysis")

    video_url = st.text_input("Enter YouTube Video URL:")

    sentiment_analysis_method = "kNN" 

    if st.button("Get Recommendations"):
        recommendations = get_related_videos(video_url)
        if recommendations:
            st.subheader("Recommended Videos:")
            for video in recommendations:
                st.write(f"Title: [{video['title']}]({video['video_url']})")
                st.write(f"Channel: {video['channel_name']}")

    comment_limit = st.number_input(
        "Enter the comment limit (0 for all comments):", min_value=0, value=20)

    sentiment_analysis_method2 = st.selectbox(
                "Select Sentiment Analysis Method:",
                input_text_sub_options
            )

    run_all_models = st.checkbox("Run all models")

    if st.button("Run Analysis"):
        if video_url:
            if comment_limit == 0:
                comments, sentiments = get_video_comments(video_url, 1000000000, run_all_models=run_all_models)
            else:
                comments, sentiments = get_video_comments(video_url, comment_limit, run_all_models=run_all_models)
            st.write(f"Total comments analyzed: {len(comments)}")

            if not comments:
                st.error("No comments found for the provided video URL.")
            else:
                st.subheader("Sentiment Distribution of Comments:")
                # if len(comments) > 0:
                #     df = pd.DataFrame()
                #     if run_all_models:
                #         for col in input_text_sub_options:
                #             df[col]=sentiments[col]

                #         fig = px.pie(df, values=input_text_sub_options)
                #         st.plotly_chart(fig)
                #         # st.dataframe(df)
                        
                #     else:
                #         df[sentiment_analysis_method2] = sentiments[sentiment_analysis_method2]
                #         # st.dataframe(df)
                #         fig = px.pie(df, values=sentiment_analysis_method2)
                #         st.plotly_chart(fig)
                #         # st.write(sentiments)
                # # else:
                # #     sentiment_counts = pd.Series(sentiments).value_counts()

                # #     fig1 = px.pie(sentiment_counts, names=sentiment_counts.index, title="Sentiment Distribution of Comments")
                # #     st.plotly_chart(fig1)

            if len(comments) > 0:
                df = pd.DataFrame()
                fig_list = []  # Initialize an empty list to store the figures

                if run_all_models:
                    for model_name in input_text_sub_options:
                        df[model_name] = sentiments[model_name]
                        fig = px.pie(df, names=model_name)
                        fig.update_layout(title=f"Sentiment Distribution for {model_name}")
                        fig_list.append(fig)  # Append each figure to the list
                else:
                    df[sentiment_analysis_method2] = sentiments[sentiment_analysis_method2]
                    fig = px.pie(df, names=sentiment_analysis_method2)
                    fig.update_layout(title=f"Sentiment Distribution for {sentiment_analysis_method2}")
                    fig_list.append(fig)  # Append the single figure to the list

                # Now, loop through the list and display each figure
                for fig in fig_list:
                    st.plotly_chart(fig)



        # else:
        #     st.write("No comments analyzed for sentiment.")