/*
 * search_engine.c — MPI Parallel Document Search Engine
 *
 * Usage:
 *   mpirun -np <P> ./search_engine <corpus_dir> "<query>" <top_k> [--csv]
 *
 * Architecture:
 *   All ranks:
 *     1. List corpus files and agree on global ordering
 *     2. Load a local subset of documents
 *     3. Build local vocabulary and gather to rank 0
 *     4. Rank 0 builds global vocabulary and broadcasts it
 *     5. Each rank computes local DF → MPI_Allreduce to global DF
 *     6. Each rank computes IDF, TF-IDF for its docs, and query vector
 *     7. Each rank computes cosine similarity and keeps local top-K
 *     8. Rank 0 gathers and merges top-K results
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mpi.h>

#include "tfidf.h"

/* ── Print helpers ─────────────────────────────────────────────────────── */

static void print_usage(const char *prog)
{
    fprintf(stderr,
        "Usage: mpirun -np <P> %s <corpus_dir> \"<query>\" <top_k> [--csv]\n"
        "\n"
        "  corpus_dir  Directory containing .txt document files\n"
        "  query       Search query (quoted string)\n"
        "  top_k       Number of top results to return\n"
        "  --csv       Optional: output timing as CSV line\n",
        prog);
}

/* ══════════════════════════════════════════════════════════════════════
 *  main
 * ══════════════════════════════════════════════════════════════════════ */

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);

    int world_size, my_rank;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

    /* ── Argument parsing (all ranks need these) ────────────────────── */
    if (argc < 4) {
        if (my_rank == 0) print_usage(argv[0]);
        MPI_Finalize();
        return 1;
    }

    const char *corpus_dir = argv[1];
    const char *query_str  = argv[2];
    int         top_k      = atoi(argv[3]);
    int         csv_mode   = (argc >= 5 && strcmp(argv[4], "--csv") == 0);

    if (top_k <= 0) {
        if (my_rank == 0) fprintf(stderr, "Error: top_k must be > 0\n");
        MPI_Finalize();
        return 1;
    }

    /* ── Timing variables ───────────────────────────────────────────── */
    double t_start, t_index_done, t_scatter_done, t_search_done, t_end;

    /* ── Variables shared across ranks ──────────────────────────────── */
    int num_docs   = 0;
    int vocab_size = 0;

    double *query_vec = NULL;
    double *my_tfidf  = NULL;
    double *idf       = NULL;
    int    *local_df  = NULL;
    int    *global_df = NULL;

    /* ══════════════════════════════════════════════════════════════════
     *  PHASE 1: Parallel Index Building
     * ══════════════════════════════════════════════════════════════════ */

    t_start = MPI_Wtime();

    /* List corpus files on all ranks (sorted) */
    char **files = list_corpus_files(corpus_dir, &num_docs);
    if (!files || num_docs == 0) {
        if (my_rank == 0)
            fprintf(stderr, "Error: no documents found in '%s'\n", corpus_dir);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    int min_docs = 0, max_docs = 0;
    MPI_Allreduce(&num_docs, &min_docs, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
    MPI_Allreduce(&num_docs, &max_docs, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
    if (min_docs != max_docs || min_docs == 0) {
        if (my_rank == 0)
            fprintf(stderr, "Error: inconsistent corpus view across ranks\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    num_docs = min_docs;

    if (!csv_mode && my_rank == 0) {
        printf("=== MPI Document Search Engine ===\n");
        printf("Corpus: %s (%d documents)\n", corpus_dir, num_docs);
        printf("Query:  \"%s\"\n", query_str);
        printf("Top-K:  %d\n", top_k);
        printf("Ranks:  %d\n\n", world_size);
    }

    /* Clamp top_k on master and broadcast */
    if (my_rank == 0 && top_k > num_docs) top_k = num_docs;
    MPI_Bcast(&top_k, 1, MPI_INT, 0, MPI_COMM_WORLD);

    /* ── Compute partitioning ───────────────────────────────────────── */
    int base_count = num_docs / world_size;
    int remainder  = num_docs % world_size;
    int my_num_docs = base_count + (my_rank < remainder ? 1 : 0);
    int my_offset = my_rank * base_count + (my_rank < remainder ? my_rank : remainder);

    /* Load only local documents */
    Document *docs = NULL;
    if (my_num_docs > 0) {
        docs = load_corpus_subset(corpus_dir, files, my_offset, my_num_docs);
        if (!docs) {
            fprintf(stderr, "Rank %d: failed to load corpus subset\n", my_rank);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    }
    free_corpus_files(files, num_docs);

    /* Build local vocabulary */
    Vocabulary *local_vocab = vocab_create();
    if (!local_vocab) {
        fprintf(stderr, "Rank %d: failed to allocate local vocabulary\n", my_rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    for (int d = 0; d < my_num_docs; d++) {
        int ntok = 0;
        char **tokens = tokenize(docs[d].text, &ntok);
        for (int t = 0; t < ntok; t++) {
            vocab_add(local_vocab, tokens[t]);
        }
        free_tokens(tokens, ntok);
    }

    int local_vocab_len = 0;
    char *local_vocab_buf = vocab_pack(local_vocab, &local_vocab_len);

    int *vocab_counts = NULL;
    int *vocab_displs = NULL;
    char *all_vocab_buf = NULL;
    if (my_rank == 0) {
        vocab_counts = malloc(world_size * sizeof(int));
    }
    MPI_Gather(&local_vocab_len, 1, MPI_INT,
               vocab_counts, 1, MPI_INT, 0, MPI_COMM_WORLD);

    int total_vocab_len = 0;
    if (my_rank == 0) {
        vocab_displs = malloc(world_size * sizeof(int));
        int off = 0;
        for (int r = 0; r < world_size; r++) {
            vocab_displs[r] = off;
            off += vocab_counts[r];
        }
        total_vocab_len = off;
        if (total_vocab_len > 0) {
            all_vocab_buf = malloc((size_t)total_vocab_len);
        }
    }

    MPI_Gatherv(local_vocab_buf, local_vocab_len, MPI_CHAR,
                all_vocab_buf, vocab_counts, vocab_displs, MPI_CHAR,
                0, MPI_COMM_WORLD);

    vocab_free(local_vocab);
    free(local_vocab_buf);

    Vocabulary *vocab = NULL;
    char *global_vocab_buf = NULL;
    int global_vocab_len = 0;

    if (my_rank == 0) {
        vocab = vocab_create();
        if (!vocab) {
            fprintf(stderr, "Rank 0: failed to allocate global vocabulary\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        for (int r = 0; r < world_size; r++) {
            if (vocab_counts[r] > 0) {
                vocab_add_from_buffer(vocab,
                                      all_vocab_buf + vocab_displs[r],
                                      vocab_counts[r]);
            }
        }
        free(all_vocab_buf);
        free(vocab_counts);
        free(vocab_displs);

        global_vocab_buf = vocab_pack(vocab, &global_vocab_len);
    }

    MPI_Bcast(&global_vocab_len, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (global_vocab_len > 0) {
        if (my_rank != 0) {
            global_vocab_buf = malloc((size_t)global_vocab_len);
        }
        MPI_Bcast(global_vocab_buf, global_vocab_len, MPI_CHAR, 0, MPI_COMM_WORLD);
    }

    if (my_rank != 0) {
        vocab = vocab_from_buffer(global_vocab_buf, global_vocab_len);
    }

    free(global_vocab_buf);

    vocab_size = vocab ? vocab->size : 0;
    if (!csv_mode && my_rank == 0)
        printf("Vocabulary size: %d words\n", vocab_size);

    /* Compute DF locally and reduce globally */
    local_df = compute_local_df(docs, my_num_docs, vocab);
    if (vocab_size > 0 && !local_df) {
        fprintf(stderr, "Rank %d: failed to compute local DF\n", my_rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    if (vocab_size > 0) {
        global_df = calloc((size_t)vocab_size, sizeof(int));
        if (!global_df) {
            fprintf(stderr, "Rank %d: failed to allocate global DF\n", my_rank);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        MPI_Allreduce(local_df, global_df, vocab_size, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    }

    idf = compute_idf_from_df(global_df ? global_df : local_df,
                              vocab_size, num_docs);
    if (vocab_size > 0 && !idf) {
        fprintf(stderr, "Rank %d: failed to compute IDF\n", my_rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    my_tfidf = compute_tfidf_matrix(docs, my_num_docs, vocab, idf);
    if (vocab_size > 0 && my_num_docs > 0 && !my_tfidf) {
        fprintf(stderr, "Rank %d: failed to compute TF-IDF matrix\n", my_rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    query_vec = compute_query_tfidf(query_str, vocab, idf);
    if (vocab_size > 0 && !query_vec) {
        fprintf(stderr, "Rank %d: failed to compute query vector\n", my_rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    free_corpus(docs, my_num_docs);

    t_index_done = MPI_Wtime();
    t_scatter_done = t_index_done;

    /* ══════════════════════════════════════════════════════════════════
     *  PHASE 3: Local Search (all ranks)
     * ══════════════════════════════════════════════════════════════════ */

    /* Compute cosine similarity for each local document */
    SearchResult *local_results = malloc(my_num_docs * sizeof(SearchResult));
    for (int d = 0; d < my_num_docs; d++) {
        local_results[d].doc_id = my_offset + d;
        if (vocab_size == 0) {
            local_results[d].score = 0.0;
        } else {
            local_results[d].score  = cosine_similarity(
                &my_tfidf[d * vocab_size], query_vec, vocab_size);
        }
    }

    /* Sort locally and keep top-K */
    sort_results(local_results, my_num_docs);
    int local_k = (my_num_docs < top_k) ? my_num_docs : top_k;

    t_search_done = MPI_Wtime();

    /* ══════════════════════════════════════════════════════════════════
     *  PHASE 4: Gather & Merge Top-K (master)
     * ══════════════════════════════════════════════════════════════════ */

    /*
     * Each rank sends local_k results (doc_id + score).
     * We pack them as: [doc_id_0, doc_id_1, ..., score_0, score_1, ...]
     * for simpler gather; or use two separate gathers.
     */

    /* Gather the count of results from each rank */
    int *recv_k = NULL;
    if (my_rank == 0) recv_k = malloc(world_size * sizeof(int));
    MPI_Gather(&local_k, 1, MPI_INT, recv_k, 1, MPI_INT, 0, MPI_COMM_WORLD);

    /* Separate gathers for IDs and scores */
    int *local_ids    = malloc(local_k * sizeof(int));
    double *local_scores = malloc(local_k * sizeof(double));
    for (int i = 0; i < local_k; i++) {
        local_ids[i]    = local_results[i].doc_id;
        local_scores[i] = local_results[i].score;
    }

    int *gather_displs_id  = NULL;
    int *gather_displs_sc  = NULL;
    int *all_ids    = NULL;
    double *all_scores = NULL;
    int total_gathered = 0;

    if (my_rank == 0) {
        gather_displs_id = malloc(world_size * sizeof(int));
        gather_displs_sc = malloc(world_size * sizeof(int));
        int off = 0;
        for (int r = 0; r < world_size; r++) {
            gather_displs_id[r] = off;
            gather_displs_sc[r] = off;
            off += recv_k[r];
        }
        total_gathered = off;
        all_ids    = malloc(total_gathered * sizeof(int));
        all_scores = malloc(total_gathered * sizeof(double));
    }

    MPI_Gatherv(local_ids, local_k, MPI_INT,
                all_ids, recv_k, gather_displs_id, MPI_INT,
                0, MPI_COMM_WORLD);

    MPI_Gatherv(local_scores, local_k, MPI_DOUBLE,
                all_scores, recv_k, gather_displs_sc, MPI_DOUBLE,
                0, MPI_COMM_WORLD);

    /* ── Master: merge and print results ───────────────────────────── */
    if (my_rank == 0) {
        /* Build result array and sort */
        SearchResult *all_results = malloc(total_gathered * sizeof(SearchResult));
        for (int i = 0; i < total_gathered; i++) {
            all_results[i].doc_id = all_ids[i];
            all_results[i].score  = all_scores[i];
        }
        sort_results(all_results, total_gathered);

        t_end = MPI_Wtime();

        int final_k = (total_gathered < top_k) ? total_gathered : top_k;

        if (csv_mode) {
            /* CSV header (only print if first run — caller handles dedup):
             * procs,num_docs,vocab_size,top_k,index_time,scatter_time,search_time,merge_time,total_time
             */
            printf("%d,%d,%d,%d,%.6f,%.6f,%.6f,%.6f,%.6f\n",
                   world_size, num_docs, vocab_size, top_k,
                   t_index_done - t_start,
                   t_scatter_done - t_index_done,
                   t_search_done - t_scatter_done,
                   t_end - t_search_done,
                   t_end - t_start);
        } else {
            printf("\n--- Top %d Results ---\n", final_k);
            printf("%-6s  %-10s  %s\n", "Rank", "Score", "Doc ID");
            printf("------  ----------  ------\n");
            for (int i = 0; i < final_k; i++) {
                printf("%-6d  %-10.6f  %d\n",
                       i + 1, all_results[i].score, all_results[i].doc_id);
            }

            printf("\n--- Timing ---\n");
            printf("Index build:  %.4f s\n", t_index_done - t_start);
            printf("Data scatter: %.4f s\n", t_scatter_done - t_index_done);
            printf("Local search: %.4f s\n", t_search_done - t_scatter_done);
            printf("Gather/merge: %.4f s\n", t_end - t_search_done);
            printf("Total:        %.4f s\n", t_end - t_start);
        }

        free(all_results);
        free(all_ids);
        free(all_scores);
        free(gather_displs_id);
        free(gather_displs_sc);
        free(recv_k);
    }

    /* ── Cleanup ───────────────────────────────────────────────────── */
    free(local_ids);
    free(local_scores);
    free(local_results);
    free(my_tfidf);
    free(query_vec);
    free(local_df);
    free(global_df);
    free(idf);
    if (vocab) vocab_free(vocab);

    MPI_Finalize();
    return 0;
}
