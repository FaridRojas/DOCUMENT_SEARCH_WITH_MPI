#!/usr/bin/env python3
"""
generate_corpus.py — Synthetic Document Corpus Generator

Generates a configurable number of text documents in a corpus/ directory
for benchmarking the MPI document search engine.

Usage:
    python3 generate_corpus.py [--num-docs N] [--vocab-size V]
                               [--min-words W1] [--max-words W2]
                               [--output-dir DIR] [--query-file FILE]
                               [--num-query-terms Q] [--seed S]
"""

import argparse
import os
import random
import string


def generate_vocabulary(size: int, rng: random.Random) -> list[str]:
    """Generate a list of unique pseudo-words."""
    vocab = set()
    # Use common English-like syllable patterns for readability
    consonants = "bcdfghjklmnpqrstvwxyz"
    vowels = "aeiou"

    while len(vocab) < size:
        # Generate words of 3-8 characters
        word_len = rng.randint(3, 8)
        word = []
        for i in range(word_len):
            if i % 2 == 0:
                word.append(rng.choice(consonants))
            else:
                word.append(rng.choice(vowels))
        vocab.add("".join(word))

    return sorted(vocab)


def generate_document(vocab: list[str], min_words: int, max_words: int,
                      rng: random.Random) -> str:
    """Generate a single document with random words from the vocabulary."""
    num_words = rng.randint(min_words, max_words)

    # Use Zipf-like distribution: some words are much more frequent
    # This creates more realistic TF-IDF distributions
    weights = [1.0 / (i + 1) ** 0.5 for i in range(len(vocab))]

    words = rng.choices(vocab, weights=weights, k=num_words)

    # Break into sentences (5-15 words each)
    sentences = []
    i = 0
    while i < len(words):
        sent_len = rng.randint(5, 15)
        sentence_words = words[i:i + sent_len]
        if sentence_words:
            sentence_words[0] = sentence_words[0].capitalize()
            sentences.append(" ".join(sentence_words) + ".")
        i += sent_len

    # Break into paragraphs (2-4 sentences each)
    paragraphs = []
    i = 0
    while i < len(sentences):
        para_len = rng.randint(2, 4)
        para = " ".join(sentences[i:i + para_len])
        paragraphs.append(para)
        i += para_len

    return "\n\n".join(paragraphs)


def generate_query(vocab: list[str], num_terms: int,
                   rng: random.Random) -> str:
    """Generate a query string from the vocabulary."""
    # Pick query terms with bias towards more common words (Zipf)
    weights = [1.0 / (i + 1) ** 0.5 for i in range(len(vocab))]
    terms = rng.choices(vocab, weights=weights, k=num_terms)
    return " ".join(terms)


def main():
    parser = argparse.ArgumentParser(
        description="Generate a synthetic text corpus for MPI search benchmarks"
    )
    parser.add_argument("--num-docs", type=int, default=1000,
                        help="Number of documents to generate (default: 1000)")
    parser.add_argument("--vocab-size", type=int, default=5000,
                        help="Vocabulary size (default: 5000)")
    parser.add_argument("--min-words", type=int, default=50,
                        help="Minimum words per document (default: 50)")
    parser.add_argument("--max-words", type=int, default=500,
                        help="Maximum words per document (default: 500)")
    parser.add_argument("--output-dir", type=str, default="corpus",
                        help="Output directory for documents (default: corpus)")
    parser.add_argument("--query-file", type=str, default="query.txt",
                        help="Output query file (default: query.txt)")
    parser.add_argument("--num-query-terms", type=int, default=5,
                        help="Number of query terms (default: 5)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility (default: 42)")

    args = parser.parse_args()

    rng = random.Random(args.seed)

    print(f"Generating vocabulary ({args.vocab_size} words)...")
    vocab = generate_vocabulary(args.vocab_size, rng)

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Generating {args.num_docs} documents in '{args.output_dir}/'...")
    for i in range(args.num_docs):
        doc_text = generate_document(vocab, args.min_words, args.max_words, rng)
        doc_path = os.path.join(args.output_dir, f"doc_{i:05d}.txt")
        with open(doc_path, "w") as f:
            f.write(doc_text)

        if (i + 1) % 500 == 0 or i == args.num_docs - 1:
            print(f"  ... {i + 1}/{args.num_docs}")

    # Generate query
    query = generate_query(vocab, args.num_query_terms, rng)
    with open(args.query_file, "w") as f:
        f.write(query + "\n")
    print(f"Query written to '{args.query_file}': \"{query}\"")

    # Print summary
    total_size = sum(
        os.path.getsize(os.path.join(args.output_dir, f))
        for f in os.listdir(args.output_dir)
        if f.endswith(".txt")
    )
    print(f"\nCorpus summary:")
    print(f"  Documents:  {args.num_docs}")
    print(f"  Vocabulary: {args.vocab_size} words")
    print(f"  Total size: {total_size / 1024 / 1024:.2f} MB")
    print(f"  Avg size:   {total_size / args.num_docs / 1024:.1f} KB/doc")


if __name__ == "__main__":
    main()
