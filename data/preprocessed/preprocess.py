import pandas as pd
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# nltk.download("stopwords")
# nltk.download("wordnet")

stop_words = set(stopwords.words("english"))
negation_words = {"not", "nor", "never", "no"}
stop_words = stop_words - negation_words
lemmatizer = WordNetLemmatizer()


def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)
    text = re.sub(r"\d+", "", text)
    text = text.translate(str.maketrans("", "", string.punctuation))

    words = text.split()
    words = [
        lemmatizer.lemmatize(word)
        for word in words
        if word not in stop_words and word.isalpha()
    ]

    return " ".join(words)


def preprocess_and_save():
    fake = pd.read_csv("../raw/Fake.csv", engine="python", encoding="latin1")
    real = pd.read_csv("../raw/True.csv", engine="python", encoding="latin1")

    fake["label"] = 0
    real["label"] = 1

    df = pd.concat([fake, real], ignore_index=True)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    # Handle missing values
    df["title"] = df["title"].fillna("")
    df["text"] = df["text"].fillna("")

    df["content"] = df["title"] + " " + df["text"]
    df = df[df["content"].str.strip() != ""]

    df = df[["content", "label"]]
    df = df.drop_duplicates(subset=["content"])

    df["content"] = df["content"].apply(clean_text)
    df = df[df["content"].str.strip() != ""]

    df.to_csv("cleaned_dataset.csv", index=False)

    print("Preprocessing completed successfully.")
    print("Final dataset shape:", df.shape)


if __name__ == "__main__":
    preprocess_and_save()