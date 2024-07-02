from gensim.models import KeyedVectors
import gensim.downloader as api
import csv
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
    list of tuple: List of tuples containing word pairs and their similarity score.
    """
    similarities = []
    for i in range(len(words) - 1):
        word1 = words[i]
        word2 = words[i + 1]
        if word1 in model.key_to_index and word2 in model.key_to_index:
            similarity = model.similarity(word1, word2)
        else:
            similarity = None  # Handle words not in the model
        similarities.append(((word1, word2), similarity))
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
    # Path to your pre-trained Word2Vec model (e.g., 'GoogleNews-vectors-negative300.bin')
    model_path = api.load("word2vec-google-news-300", return_path=True)

    # Load the Word2Vec model
    model = KeyedVectors.load_word2vec_format(model_path, binary=True)

    # Path to your input TSV file
    input_file_path = f"/data/{USER}/fahh/processed_fals/{args.f_in}"

    # Open the input TSV file and read the words
    words_list = []
    with open(input_file_path, "r", newline="") as file:
        reader = csv.reader(file, delimiter="\t")
        for row in reader:
            words_list.extend(row)  # Assuming words are in a single column

    # Compute similarities
    similarities = compute_similarity(model, words_list)

    # Open a file in write mode
    with open("similarities.tsv", "w", newline="") as file:
        writer = csv.writer(file, delimiter="\t")

        # Write the header (optional)
        writer.writerow(["Word1", "Word2", "Similarity"])

        # Loop through the pairs and their similarities
        for pair, sim in similarities:
            word1, word2 = pair
            if sim is not None:
                # Write the pair and similarity to the file
                writer.writerow([word1, word2, f"{sim:.4f}"])
            else:
                # Handle the case where one or both words are not in the model vocabulary
                writer.writerow([word1, word2, "N/A"])
