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


def process_row(row, model):
    words = row[2:]  # Skip the first two columns
    similarities = compute_similarity(model, words)
    return pd.Series(similarities)


def find_groups(similarities, threshold=0.5):
    groups = []
    current_group = []

    for i in range(len(similarities) - 1):
        if abs(similarities[i + 1] - similarities[i]) > threshold:
            current_group.append(i)
            groups.append(current_group)
            current_group = []
        else:
            current_group.append(i)

    # Add the last group
    if current_group:
        groups.append(current_group)

    return groups


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
    similarity_df = df.apply(lambda row: process_row(row, model), axis=1)

    # Combine the PROLIFIC_PID, Task, and similarity DataFrames
    result_df = pd.concat([df[["PROLIFIC_PID", "Task"]], similarity_df], axis=1)

    # Save the result to a new TSV file
    similarity_output_path = (
        f"/data/{USER}/fahh/similarities/{args.f_out}_similarities.tsv"
    )
    result_df.to_csv(similarity_output_path, sep="\t", index=False)
    print(f"Similarities saved to {similarity_output_path}.")

    # Calculate differences in similarities
    difference_df = similarity_df.apply(lambda x: x.diff().abs(), axis=1)

    # Save differences to a new TSV file
    difference_output_path = (
        f"/data/{USER}/fahh/similarities/{args.f_out}_difference_in_similarity.tsv"
    )
    difference_df.to_csv(difference_output_path, sep="\t", index=False)
    print(f"Differences in similarities saved to {difference_output_path}.")

    # Find groups based on differences in similarities
    groups = difference_df.apply(find_groups, axis=1)

    # Save groups to a new TSV file
    groups_output_path = f"/data/{USER}/fahh/similarities/{args.f_out}_groups.tsv"
    with open(groups_output_path, "w", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        for row_groups in groups:
            writer.writerow(row_groups)
    print(f"Groups saved to {groups_output_path}.")
