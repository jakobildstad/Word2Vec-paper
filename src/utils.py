from collections import Counter
import math
import random



def tokenize(text):
    # Simple whitespace tokenizer
    return text.strip().split()


def build_vocab(tokens, min_count=10):
    """
    Build vocabulary from tokenized text.

    Args:
        tokens (list of str): list of word tokens
        min_count (int): drop words with frequency < min_count

    Returns:
        string_to_index (dict): string → index
        index_to_string (list): index → string
        word_counts (dict): word index → count
    """
    # Count frequencies
    vocab = Counter(tokens)

    # Remove low-frequency words
    vocab = {word: count for word, count in vocab.items() if count >= min_count}

    # Sort by frequency (most frequent first)
    sorted_vocab = sorted(vocab.items(), key=lambda x: -x[1])

    # Build mappings
    index_to_string = [w for w, _ in sorted_vocab]
    string_to_index = {w: i for i, w in enumerate(index_to_string)}

    # Keep counts aligned with indices
    token_counts = {string_to_index[w]: c for w, c in sorted_vocab}

    return string_to_index, index_to_string, token_counts


def build_freqs(token_counts, total_token_count):
    """
    Compute relative frequencies for each word in the vocabulary.

    Returns:
        dict[int -> float]: mapping from word_id to relative frequency in (0,1]
    """
    return {token_id: count / total_token_count for token_id, count in token_counts.items()}


def should_keep(token_id, token_freqs, t=1e-5, rng=random.random):
    """
    Decide whether to keep or drop a token according to Mikolov's subsampling trick.
    """
    token_freq = token_freqs[token_id]

    # Compute probability of keeping this word
    p_keep = math.sqrt(t / token_freq) + (t / token_freq)
    if p_keep > 1.0:
        p_keep = 1.0

    return rng() < p_keep


