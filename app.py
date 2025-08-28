import streamlit as st
import pickle
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import pandas as pd
import altair as alt

nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('wordnet')
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))


def preprocess_tweet(text):
    # 1. Lowercase
    text = text.lower()

    # 2. Remove URLs
    text = re.sub(r"http\S+|www\S+|https\S+", '', text, flags=re.MULTILINE)

    # 3. Remove @mentions
    text = re.sub(r'@\w+', '', text)

    # 4. Remove special characters, numbers, punctuation (keep only alphabets)
    text = re.sub(r"[^a-z\s]", '', text)

    # 5. Tokenization
    tokens = word_tokenize(text)

    # 6. Remove stopwords
    tokens = [word for word in tokens if word not in stop_words and len(word) > 1]

    # 7. Lemmatization
    tokens = [lemmatizer.lemmatize(word) for word in tokens]

    return " ".join(tokens)


st.set_page_config(page_title="Tweet Sentiment Analyzer", page_icon="🐦", layout="centered")

# Title
st.title("🐦 Tweet Sentiment Analyzer")
st.write("Analyze the sentiment of tweets in real-time using NLP models.")

# Sidebar
st.sidebar.header("Upload / Input Options")
option = st.sidebar.radio("Choose Input Method:", ("Single Tweet", "CSV Upload"))


# Function for sentiment prediction (mock using TextBlob here)
def analyze_sentiment(text):
    print(f"here=>{text}")
    # Preprocessing
    preprocess = preprocess_tweet(text)
    print(f"preprocess=>{preprocess}")
    # Vectorization
    vect = vectorizer.transform([preprocess])
    print(f"vect=>{vect}")
    # prediction
    prediction = model.predict(vect)[0]

    if prediction == 1:
        return "😊 Positive"
    else:
        return "😠 Negative"


# Single tweet input
if option == "Single Tweet":
    tweet_text = st.text_area("Enter Tweet:", placeholder="Type or paste a tweet here...")
    if st.button("Analyze"):
        if tweet_text.strip() != "":
            print(f"tweet=>{tweet_text}")
            sentiment = analyze_sentiment(tweet_text)
            st.subheader("Result:")
            if "Positive" in sentiment:
                st.success(f"Sentiment: {sentiment}")  # Green box
            elif "Negative" in sentiment:
                st.error(f"Sentiment: {sentiment}")  # Red box
        else:
            st.warning("Please enter a tweet before analyzing.")

# CSV upload option
elif option == "CSV Upload":
    uploaded_file = st.file_uploader("Upload a CSV file containing tweets", type=["csv"])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)

        if "tweet" not in df.columns:
            st.error("CSV must contain a column named 'tweet'")
        else:
            st.write("### Uploaded Data Preview")
            st.dataframe(df.head())

            if st.button("Analyze All"):
                df["Sentiment"] = df["tweet"].apply(analyze_sentiment)


                # Apply color styling
                def sentiment_color(val):
                    if "Positive" in val:
                        return "color: green; font-weight: bold;"
                    elif "Negative" in val:
                        return "color: red; font-weight: bold;"
                    else:
                        return "color: gray; font-weight: bold;"


                styled_df = df.style.applymap(sentiment_color, subset=["Sentiment"])

                st.write("### Results")
                st.dataframe(styled_df, use_container_width=True)

                # Sentiment distribution chart with custom colors
                st.write("### Sentiment Distribution")
                sentiment_counts = df["Sentiment"].value_counts().reset_index()
                sentiment_counts.columns = ["Sentiment", "Count"]

                chart = (
                    alt.Chart(sentiment_counts)
                    .mark_bar()
                    .encode(
                        x=alt.X("Sentiment", sort=["😊 Positive", "😠 Negative"]),
                        y="Count",
                        color=alt.Color(
                            "Sentiment",
                            scale=alt.Scale(
                                domain=["😊 Positive", "😠 Negative"],
                                range=["green", "red"],
                            ),
                        ),
                        tooltip=["Sentiment", "Count"]
                    )
                )
                st.altair_chart(chart, use_container_width=True)

st.markdown("---")
st.caption("Built with ❤️ using Streamlit")
