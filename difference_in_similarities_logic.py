import pandas as pd
from gensim.models import KeyedVectors
import gensim.downloader as api
import argparse
import os

USER = None
USER = os.getenv("USER")
USER = USER.split("@")[0]


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


# Function to apply compute_similarity to each row
def process_row(row):
    words = row[2:]  # Skip the first two columns
    similarities = compute_similarity(model, words)
    return pd.Series(similarities, name="similarity")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--onethirdm", action="store_true", help="Process oneThirdM results"
    )
    parser.add_argument(
        "--context_sizes",
        type=str,
        default="0,1,2,3,4,5,6,7,8,9,10",
        help="Comma-separated list of context sizes for W2V and GloVe",
    )
    parser.add_argument(
        "--f_out",
        type=str,
        default="test",
        help="File name selection",
    )
    parser.add_argument(
        "--f_in",
        type=str,
        default="DAC-HH-1.1_February+21,+2024_19.21.tsv",
        help="File name selection",
    )
    parser.add_argument("--task", type=str, default="CFA", help="Task identifier")
    parser.add_argument("--n_proc", type=int, default=16, help="Number of processes")
    args = parser.parse_args()

    # Load the Word2Vec model
    model_path = api.load("word2vec-google-news-300", return_path=True)
    model = KeyedVectors.load_word2vec_format(model_path, binary=True)

    # Path to your input TSV file
    input_file_path = f"/data/{USER}/fahh/processed_fals/{args.f_in}"

    # Load the TSV file into a DataFrame
    df = pd.read_csv(input_file_path, delimiter="\t")

    # Apply the function to each row and create new columns for the similarities
    similarity_df = df.apply(process_row, axis=1)

    # Combine the PROLIFIC_PID, Task, and similarity DataFrames
    result_df = pd.concat([df[["PROLIFIC_PID", "Task"]], similarity_df], axis=1)

    # Rename columns to similarity_1, similarity_2, ..., similarity_n
    result_df.columns = ["PROLIFIC_PID", "Task"] + [
        f"similarity_{i+1}" for i in range(len(similarity_df.columns))
    ]

    # Specify the output file path for similarities.tsv
    output_file_path = f"/data/{USER}/fahh/similarities/{args.f_out}.tsv"

    # Save the result to similarities.tsv
    result_df.to_csv(output_file_path, sep="\t", index=False)
    print(f"Similarities saved to {output_file_path}.")

    # Calculate absolute differences between consecutive similarities
    differences_df = pd.DataFrame()
    for i in range(1, len(similarity_df.columns)):
        diff_col_name = f"absolute_difference_similarity_{i}"
        differences_df[diff_col_name] = (
            result_df[f"similarity_{i+1}"].sub(result_df[f"similarity_{i}"]).abs()
        )

    # Specify the output file path for difference_in_similarity.tsv
    diff_output_file_path = (
        f"/data/{USER}/fahh/similarities/difference_in_similarity.tsv"
    )

    # Save the absolute differences to difference_in_similarity.tsv
    differences_df.to_csv(diff_output_file_path, sep="\t", index=False)
    print(f"Absolute differences saved to {diff_output_file_path}.")
