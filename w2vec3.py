from gensim.models import KeyedVectors
import gensim.downloader as api
import pandas as pd
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
    list: List of similarity scores between consecutive words.
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--onethirdm", action="store_true", help="Process oneThirdM results"
    )
    parser.add_argument("--usf", action="store_true", help="Process USF results")
    parser.add_argument("--swow", action="store_true", help="Process SWOW results")
    parser.add_argument("--w2v", action="store_true", help="Process W2V results")
    parser.add_argument("--glove", action="store_true", help="Process GloVe results")
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

    # Path to your pre-trained Word2Vec model
    model_path = api.load("word2vec-google-news-300", return_path=True)

    # Load the Word2Vec model
    model = KeyedVectors.load_word2vec_format(model_path, binary=True)

    # Path to your input TSV file
    input_file_path = f"/data/{USER}/fahh/processed_fals/{args.f_in}"

    # Load the TSV file into a DataFrame
    df = pd.read_csv(input_file_path, delimiter="\t")

    # Compute similarities for each row
    for index, row in df.iterrows():
        words = row[2:].tolist()
        similarities = compute_similarity(model, words)
        for i, sim in enumerate(similarities):
            col_name = f"similarity_{i+1}"
            df.at[index, col_name] = sim

    similarity_cols = [f"similarity_{i+1}" for i in range(19)]
    columns_to_keep = ["PROLIFIC_PID", "Task"] + similarity_cols
    df = df[columns_to_keep]

    # Save the updated DataFrame to a TSV file
    output_file_path = f"/data/{USER}/fahh/processed_fals/{args.f_out}.tsv"
    df.to_csv(output_file_path, sep="\t", index=False)
