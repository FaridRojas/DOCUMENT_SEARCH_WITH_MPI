/*
 * tfidf.c — TF-IDF Document Search Utilities (implementation)
 */

#include "tfidf.h"

#include <ctype.h>
#include <dirent.h>
#include <math.h>
#include <string.h>
#include <sys/stat.h>

/* ═══════════════════════════════════════════════════════════════════════
 *  Tokenization
 * ═══════════════════════════════════════════════════════════════════════ */

char **tokenize(const char *text, int *out_count)
{
    int capacity = INITIAL_CAP;
    int count    = 0;
    char **tokens = malloc(capacity * sizeof(char *));
    if (!tokens) { *out_count = 0; return NULL; }

    const char *p = text;
    while (*p) {
        /* skip non-alpha characters */
        while (*p && !isalpha((unsigned char)*p)) p++;
        if (!*p) break;

        /* start of a word */
        char word[MAX_WORD_LEN];
        int  wlen = 0;
        while (*p && isalpha((unsigned char)*p) && wlen < MAX_WORD_LEN - 1) {
            word[wlen++] = tolower((unsigned char)*p);
            p++;
        }
        word[wlen] = '\0';

        /* grow array if needed */
        if (count >= capacity) {
            capacity *= 2;
            tokens = realloc(tokens, capacity * sizeof(char *));
            if (!tokens) { *out_count = 0; return NULL; }
        }
        tokens[count] = strdup(word);
        count++;
    }

    *out_count = count;
    return tokens;
}

void free_tokens(char **tokens, int count)
{
    if (!tokens) return;
    for (int i = 0; i < count; i++) free(tokens[i]);
    free(tokens);
}

/* ═══════════════════════════════════════════════════════════════════════
 *  Vocabulary
 * ═══════════════════════════════════════════════════════════════════════ */

Vocabulary *vocab_create(void)
{
    Vocabulary *v = malloc(sizeof(Vocabulary));
    if (!v) return NULL;
    v->capacity = INITIAL_CAP;
    v->size     = 0;
    v->words    = malloc(v->capacity * sizeof(char *));
    return v;
}

int vocab_add(Vocabulary *v, const char *word)
{
    /* check if already present */
    int id = vocab_lookup(v, word);
    if (id >= 0) return id;

    /* grow if needed */
    if (v->size >= v->capacity) {
        v->capacity *= 2;
        v->words = realloc(v->words, v->capacity * sizeof(char *));
    }
    v->words[v->size] = strdup(word);
    return v->size++;
}

int vocab_lookup(const Vocabulary *v, const char *word)
{
    for (int i = 0; i < v->size; i++) {
        if (strcmp(v->words[i], word) == 0) return i;
    }
    return -1;
}

void vocab_free(Vocabulary *v)
{
    if (!v) return;
    for (int i = 0; i < v->size; i++) free(v->words[i]);
    free(v->words);
    free(v);
}

/* ═══════════════════════════════════════════════════════════════════════
 *  Document I/O
 * ═══════════════════════════════════════════════════════════════════════ */

/* Helper: check if filename ends with ".txt" */
static int ends_with_txt(const char *name)
{
    size_t len = strlen(name);
    if (len < 4) return 0;
    return strcmp(name + len - 4, ".txt") == 0;
}

/* Comparison for qsort — sort documents by ID ascending */
static int cmp_doc_id(const void *a, const void *b)
{
    return ((const Document *)a)->id - ((const Document *)b)->id;
}

Document *load_corpus(const char *corpus_dir, int *out_count)
{
    DIR *dir = opendir(corpus_dir);
    if (!dir) {
        fprintf(stderr, "Error: cannot open corpus directory '%s'\n", corpus_dir);
        *out_count = 0;
        return NULL;
    }

    int capacity = INITIAL_CAP;
    int count    = 0;
    Document *docs = malloc(capacity * sizeof(Document));

    struct dirent *entry;
    while ((entry = readdir(dir)) != NULL) {
        if (!ends_with_txt(entry->d_name)) continue;

        /* build full path */
        char path[MAX_PATH_LEN];
        snprintf(path, sizeof(path), "%s/%s", corpus_dir, entry->d_name);

        /* read file contents */
        FILE *fp = fopen(path, "r");
        if (!fp) continue;

        fseek(fp, 0, SEEK_END);
        long fsize = ftell(fp);
        rewind(fp);

        char *text = malloc(fsize + 1);
        if (!text) { fclose(fp); continue; }
        size_t nread = fread(text, 1, fsize, fp);
        text[nread] = '\0';
        fclose(fp);

        /* grow array if needed */
        if (count >= capacity) {
            capacity *= 2;
            docs = realloc(docs, capacity * sizeof(Document));
        }

        docs[count].id   = count;  /* sequential ID */
        docs[count].text = text;
        count++;
    }
    closedir(dir);

    /* sort by ID for deterministic ordering */
    qsort(docs, count, sizeof(Document), cmp_doc_id);

    *out_count = count;
    return docs;
}

void free_corpus(Document *docs, int count)
{
    if (!docs) return;
    for (int i = 0; i < count; i++) free(docs[i].text);
    free(docs);
}

/* ═══════════════════════════════════════════════════════════════════════
 *  TF-IDF Computation
 * ═══════════════════════════════════════════════════════════════════════ */

double *compute_tf(const char *text, const Vocabulary *vocab)
{
    double *tf = calloc(vocab->size, sizeof(double));
    if (!tf) return NULL;

    int num_tokens = 0;
    char **tokens = tokenize(text, &num_tokens);

    for (int i = 0; i < num_tokens; i++) {
        int id = vocab_lookup(vocab, tokens[i]);
        if (id >= 0) tf[id] += 1.0;
    }
    free_tokens(tokens, num_tokens);

    /* normalize by total number of tokens */
    if (num_tokens > 0) {
        for (int i = 0; i < vocab->size; i++) {
            tf[i] /= (double)num_tokens;
        }
    }
    return tf;
}

double *compute_idf(Document *docs, int num_docs, const Vocabulary *vocab)
{
    double *idf = calloc(vocab->size, sizeof(double));
    if (!idf) return NULL;

    /* df[i] = number of documents containing word i */
    int *df = calloc(vocab->size, sizeof(int));

    for (int d = 0; d < num_docs; d++) {
        /* get unique words in this document */
        int num_tokens = 0;
        char **tokens = tokenize(docs[d].text, &num_tokens);

        /* use a boolean array to avoid counting duplicates */
        int *seen = calloc(vocab->size, sizeof(int));
        for (int t = 0; t < num_tokens; t++) {
            int id = vocab_lookup(vocab, tokens[t]);
            if (id >= 0 && !seen[id]) {
                df[id]++;
                seen[id] = 1;
            }
        }
        free(seen);
        free_tokens(tokens, num_tokens);
    }

    /* idf[i] = log(N / (1 + df[i])) */
    for (int i = 0; i < vocab->size; i++) {
        idf[i] = log((double)num_docs / (1.0 + (double)df[i]));
    }

    free(df);
    return idf;
}

double *compute_tfidf_matrix(Document *docs, int num_docs,
                             const Vocabulary *vocab, const double *idf)
{
    int vs = vocab->size;
    double *matrix = malloc((size_t)num_docs * vs * sizeof(double));
    if (!matrix) return NULL;

    for (int d = 0; d < num_docs; d++) {
        double *tf = compute_tf(docs[d].text, vocab);
        for (int t = 0; t < vs; t++) {
            matrix[d * vs + t] = tf[t] * idf[t];
        }
        free(tf);
    }
    return matrix;
}

double *compute_query_tfidf(const char *query, const Vocabulary *vocab,
                            const double *idf)
{
    double *tf = calloc(vocab->size, sizeof(double));
    if (!tf) return NULL;

    int num_tokens = 0;
    char **tokens = tokenize(query, &num_tokens);

    for (int i = 0; i < num_tokens; i++) {
        int id = vocab_lookup(vocab, tokens[i]);
        if (id >= 0) tf[id] += 1.0;
    }
    free_tokens(tokens, num_tokens);

    /* normalize and multiply by IDF */
    if (num_tokens > 0) {
        for (int i = 0; i < vocab->size; i++) {
            tf[i] = (tf[i] / (double)num_tokens) * idf[i];
        }
    }
    return tf;
}

/* ═══════════════════════════════════════════════════════════════════════
 *  Cosine Similarity
 * ═══════════════════════════════════════════════════════════════════════ */

double cosine_similarity(const double *a, const double *b, int len)
{
    double dot   = 0.0;
    double norm_a = 0.0;
    double norm_b = 0.0;

    for (int i = 0; i < len; i++) {
        dot    += a[i] * b[i];
        norm_a += a[i] * a[i];
        norm_b += b[i] * b[i];
    }

    double denom = sqrt(norm_a) * sqrt(norm_b);
    if (denom < 1e-12) return 0.0;
    return dot / denom;
}

/* ═══════════════════════════════════════════════════════════════════════
 *  Top-K Merge
 * ═══════════════════════════════════════════════════════════════════════ */

static int cmp_results_desc(const void *a, const void *b)
{
    double diff = ((const SearchResult *)b)->score -
                  ((const SearchResult *)a)->score;
    if (diff > 0.0) return 1;
    if (diff < 0.0) return -1;
    return 0;
}

void sort_results(SearchResult *results, int count)
{
    qsort(results, count, sizeof(SearchResult), cmp_results_desc);
}

SearchResult *merge_topk(SearchResult **arrays, int num_arrays, int k)
{
    int total = num_arrays * k;
    SearchResult *merged = malloc(total * sizeof(SearchResult));
    if (!merged) return NULL;

    int idx = 0;
    for (int i = 0; i < num_arrays; i++) {
        for (int j = 0; j < k; j++) {
            merged[idx++] = arrays[i][j];
        }
    }

    sort_results(merged, total);

    /* shrink to top-k */
    SearchResult *topk = malloc(k * sizeof(SearchResult));
    int result_count = (total < k) ? total : k;
    memcpy(topk, merged, result_count * sizeof(SearchResult));
    free(merged);

    return topk;
}
