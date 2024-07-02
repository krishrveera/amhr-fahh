import pandas as pd
from gensim.models import KeyedVectors
import gensim.downloader as api

# Load the Word2Vec model
model_path = api.load("word2vec-google-news-300", return_path=True)
model = KeyedVectors.load_word2vec_format(model_path, binary=True)


def compute_similarity(model, words):
    """
    Compute the cosine similarity between consecutive words in the list.

    Parameters:
    model (KeyedVectors): Pre-trained Word2Vec model.
    words (list of str): List of words to compute similarity.

    Returns:
    list of float: List of similarity scores between consecutive words.
    """
    similarities = []
    for i in range(len(words) - 1):
        word1 = words[i]
        word2 = words[i + 1]
        if word1 in model.key_to_index and word2 in model.key_to_index:
            similarity = model.similarity(word1, word2)
        else:
            similarity = None  # Handle words not in the model
        similarities.append(similarity)
    return similarities


# Path to your input TSV file
input_file_path = "C:/Users/krish/Desktop/AMHR Research/amhr-fahh/current_fals.tsv"

# Load the TSV file into a DataFrame
df = pd.read_csv(input_file_path, delimiter="\t")


# Function to apply compute_similarity to each row
def process_row(row):
    words = row[2:]  # Skip the first two columns
    similarities = compute_similarity(model, words)
    return pd.Series(similarities)


# Apply the function to each row and create new columns for the similarities
similarity_df = df.apply(process_row, axis=1)

# Combine the PROLIFIC_PID, Task, and similarity DataFrames
result_df = pd.concat([df[["PROLIFIC_PID", "Task"]], similarity_df], axis=1)

# Save the result to a new TSV file
result_df.to_csv("similarities.tsv", sep="\t", index=False)

# what is the difference between extend and append: if you use
