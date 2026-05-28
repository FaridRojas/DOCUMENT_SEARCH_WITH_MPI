/*
 * tfidf.h — TF-IDF Document Search Utilities
 *
 * Provides tokenization, vocabulary construction, TF-IDF vector computation,
 * cosine similarity, and top-K merge operations for the MPI document search engine.
 */

#ifndef TFIDF_H
#define TFIDF_H

#include <stdio.h>
#include <stdlib.h>

/* ── Constants ─────────────────────────────────────────────────────────── */

#define MAX_WORD_LEN    64      /* maximum length of a single token          */
#define MAX_LINE_LEN    4096    /* maximum length of a line read from file   */
#define MAX_PATH_LEN    512     /* maximum file-path length                  */
#define INITIAL_CAP     256     /* initial capacity for dynamic arrays       */

/* ── Data Structures ───────────────────────────────────────────────────── */

/* A single search result: document ID + similarity score */
typedef struct {
    int    doc_id;
    double score;
} SearchResult;

/* Vocabulary: maps words → integer IDs */
typedef struct {
    char **words;       /* array of word strings                  */
    int    size;        /* number of unique words in vocabulary   */
    int    capacity;    /* allocated capacity                     */
} Vocabulary;

/* A single document (raw text) */
typedef struct {
    int    id;          /* document identifier                    */
    char  *text;        /* raw text content (heap-allocated)      */
} Document;

/* ── Tokenization ──────────────────────────────────────────────────────── */

/*
 * Tokenize a string into lowercase words. Strips punctuation.
 * Returns a NULL-terminated array of heap-allocated word strings.
 * The caller must free each word and the array itself.
 *
 * *out_count receives the number of tokens produced.
 */
char **tokenize(const char *text, int *out_count);

/*
 * Free a token array returned by tokenize().
 */
void free_tokens(char **tokens, int count);

/* ── Vocabulary ────────────────────────────────────────────────────────── */

/*
 * Create an empty vocabulary.
 */
Vocabulary *vocab_create(void);

/*
 * Add a word to the vocabulary (if not already present).
 * Returns the word's integer ID.
 */
int vocab_add(Vocabulary *v, const char *word);

/*
 * Look up a word in the vocabulary.
 * Returns the word's ID, or -1 if not found.
 */
int vocab_lookup(const Vocabulary *v, const char *word);

/*
 * Free all memory associated with a vocabulary.
 */
void vocab_free(Vocabulary *v);

/*
 * Add words from a newline-delimited buffer into a vocabulary.
 */
void vocab_add_from_buffer(Vocabulary *v, const char *buf, int len);

/* ── Document I/O ──────────────────────────────────────────────────────── */

/*
 * Load all .txt files from `corpus_dir` into an array of Document structs.
 * Returns heap-allocated array; *out_count receives the number of documents.
 * The caller must free each Document's text and the array itself.
 */
Document *load_corpus(const char *corpus_dir, int *out_count);

/*
 * List all .txt filenames in `corpus_dir` (sorted lexicographically).
 * Returns heap-allocated array of strings; *out_count receives count.
 */
char **list_corpus_files(const char *corpus_dir, int *out_count);

/*
 * Free a list returned by list_corpus_files().
 */
void free_corpus_files(char **files, int count);

/*
 * Load a subset of files given a sorted filename list.
 * `start` is the global index into `files`, and `count` is how many to load.
 */
Document *load_corpus_subset(const char *corpus_dir, char **files,
                             int start, int count);

/*
 * Free an array of documents.
 */
void free_corpus(Document *docs, int count);

/* ── Parallel TF-IDF Helpers ───────────────────────────────────────────── */

/*
 * Compute document frequency counts for local documents.
 * Returns an int array of size vocab->size.
 */
int *compute_local_df(Document *docs, int num_docs, const Vocabulary *vocab);

/*
 * Compute IDF vector from a global DF array.
 */
double *compute_idf_from_df(const int *df, int vocab_size, int num_docs);

/*
 * Pack a vocabulary into a newline-delimited buffer for MPI exchange.
 * Returns heap-allocated buffer; out_len receives byte length.
 */
char *vocab_pack(const Vocabulary *v, int *out_len);

/*
 * Build a vocabulary from a newline-delimited buffer.
 */
Vocabulary *vocab_from_buffer(const char *buf, int len);

/* ── TF-IDF Computation ───────────────────────────────────────────────── */

/*
 * Compute TF (term frequency) vector for a single document.
 * Returns a heap-allocated array of size vocab->size (doubles).
 * tf[i] = (count of word i in doc) / (total words in doc)
 */
double *compute_tf(const char *text, const Vocabulary *vocab);

/*
 * Compute IDF (inverse document frequency) vector across all documents.
 * Returns a heap-allocated array of size vocab->size.
 * idf[i] = log( N / (1 + df_i) )   where df_i = nr of docs containing word i.
 */
double *compute_idf(Document *docs, int num_docs, const Vocabulary *vocab);

/*
 * Compute the full TF-IDF matrix for all documents.
 * Returns a flat row-major array of size (num_docs × vocab_size).
 * tfidf[d * vocab_size + t] = tf[d][t] * idf[t]
 */
double *compute_tfidf_matrix(Document *docs, int num_docs,
                             const Vocabulary *vocab, const double *idf);

/*
 * Compute TF-IDF vector for a single query string.
 * Returns a heap-allocated array of size vocab->size.
 */
double *compute_query_tfidf(const char *query, const Vocabulary *vocab,
                            const double *idf);

/* ── Similarity ────────────────────────────────────────────────────────── */

/*
 * Cosine similarity between two vectors of length `len`.
 * Returns dot(a,b) / (||a|| * ||b||),  or 0.0 if either norm is 0.
 */
double cosine_similarity(const double *a, const double *b, int len);

/* ── Top-K Merge ───────────────────────────────────────────────────────── */

/*
 * Sort an array of SearchResult by score (descending).
 */
void sort_results(SearchResult *results, int count);

/*
 * Merge `num_arrays` sorted SearchResult arrays (each of length `k`)
 * into a single sorted array and return the top `k` results.
 * The returned array is heap-allocated with exactly `k` elements.
 */
SearchResult *merge_topk(SearchResult **arrays, int num_arrays, int k);

#endif /* TFIDF_H */
