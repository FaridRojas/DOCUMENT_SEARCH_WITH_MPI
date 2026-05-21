/*
 * search_engine.c — MPI Parallel Document Search Engine
 *
 * Usage:
 *   mpirun -np <P> ./search_engine <corpus_dir> "<query>" <top_k> [--csv]
 *
 * Architecture:
 *   Rank 0 (master):
 *     1. Loads all .txt documents from corpus_dir
 *     2. Builds vocabulary from the entire corpus
 *     3. Computes IDF vector
 *     4. Computes TF-IDF matrix for all documents
 *     5. Computes TF-IDF vector for the query
 *     6. Broadcasts vocabulary size & query vector
 *     7. Scatters document TF-IDF vectors across ranks
 *     8. Each rank computes cosine similarity for its chunk
 *     9. Gathers top-K from each rank
 *    10. Merges and prints final top-K results
 *
 *   Rank 1..P-1 (workers):
 *     - Receive query vector and document chunk
 *     - Compute cosine similarity
 *     - Local sort → keep top-K
 *     - Return top-K to master via MPI_Gatherv
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

    double *tfidf_matrix = NULL;   /* num_docs × vocab_size  (master only initially) */
    double *query_vec    = NULL;   /* vocab_size              (broadcast to all)      */
    int    *doc_ids      = NULL;   /* document IDs array      (master only)           */

    /* ══════════════════════════════════════════════════════════════════
     *  PHASE 1: Index Building (master only)
     * ══════════════════════════════════════════════════════════════════ */

    t_start = MPI_Wtime();

    if (my_rank == 0) {
        /* Load documents */
        Document *docs = load_corpus(corpus_dir, &num_docs);
        if (!docs || num_docs == 0) {
            fprintf(stderr, "Error: no documents found in '%s'\n", corpus_dir);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        if (!csv_mode) {
            printf("=== MPI Document Search Engine ===\n");
            printf("Corpus: %s (%d documents)\n", corpus_dir, num_docs);
            printf("Query:  \"%s\"\n", query_str);
            printf("Top-K:  %d\n", top_k);
            printf("Ranks:  %d\n\n", world_size);
        }

        /* Clamp top_k */
        if (top_k > num_docs) top_k = num_docs;

        /* Build vocabulary from corpus */
        Vocabulary *vocab = vocab_create();
        for (int d = 0; d < num_docs; d++) {
            int ntok = 0;
            char **tokens = tokenize(docs[d].text, &ntok);
            for (int t = 0; t < ntok; t++) {
                vocab_add(vocab, tokens[t]);
            }
            free_tokens(tokens, ntok);
        }
        vocab_size = vocab->size;

        if (!csv_mode)
            printf("Vocabulary size: %d words\n", vocab_size);

        /* Compute IDF */
        double *idf = compute_idf(docs, num_docs, vocab);

        /* Compute TF-IDF matrix for all documents */
        tfidf_matrix = compute_tfidf_matrix(docs, num_docs, vocab, idf);

        /* Compute query TF-IDF vector */
        query_vec = compute_query_tfidf(query_str, vocab, idf);

        /* Store document IDs */
        doc_ids = malloc(num_docs * sizeof(int));
        for (int d = 0; d < num_docs; d++) {
            doc_ids[d] = docs[d].id;
        }

        /* Clean up corpus (texts no longer needed) */
        free_corpus(docs, num_docs);
        free(idf);
        vocab_free(vocab);
    }

    t_index_done = MPI_Wtime();

    /* ══════════════════════════════════════════════════════════════════
     *  PHASE 2: Distribute Data
     * ══════════════════════════════════════════════════════════════════ */

    /* Broadcast metadata */
    MPI_Bcast(&num_docs,   1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&vocab_size, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&top_k,      1, MPI_INT, 0, MPI_COMM_WORLD);

    /* Allocate query vector on workers and broadcast */
    if (my_rank != 0) {
        query_vec = malloc(vocab_size * sizeof(double));
    }
    MPI_Bcast(query_vec, vocab_size, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    /* ── Compute scatter distribution ──────────────────────────────── */
    int base_count = num_docs / world_size;
    int remainder  = num_docs % world_size;

    /* Each rank gets `base_count` docs; first `remainder` ranks get one extra */
    int *send_counts = malloc(world_size * sizeof(int));
    int *displs      = malloc(world_size * sizeof(int));

    int *id_send_counts = malloc(world_size * sizeof(int));
    int *id_displs      = malloc(world_size * sizeof(int));

    int offset = 0;
    for (int r = 0; r < world_size; r++) {
        int n = base_count + (r < remainder ? 1 : 0);
        send_counts[r]    = n * vocab_size;  /* TF-IDF: n rows × vocab_size cols */
        id_send_counts[r] = n;               /* IDs: n elements */
        displs[r]         = offset * vocab_size;
        id_displs[r]      = offset;
        offset += n;
    }

    int my_num_docs = base_count + (my_rank < remainder ? 1 : 0);

    /* Allocate local buffers */
    double *my_tfidf = malloc((size_t)my_num_docs * vocab_size * sizeof(double));
    int    *my_ids   = malloc(my_num_docs * sizeof(int));

    /* Scatter TF-IDF matrix rows */
    MPI_Scatterv(tfidf_matrix, send_counts, displs, MPI_DOUBLE,
                 my_tfidf, my_num_docs * vocab_size, MPI_DOUBLE,
                 0, MPI_COMM_WORLD);

    /* Scatter document IDs */
    MPI_Scatterv(doc_ids, id_send_counts, id_displs, MPI_INT,
                 my_ids, my_num_docs, MPI_INT,
                 0, MPI_COMM_WORLD);

    t_scatter_done = MPI_Wtime();

    /* ══════════════════════════════════════════════════════════════════
     *  PHASE 3: Local Search (all ranks)
     * ══════════════════════════════════════════════════════════════════ */

    /* Compute cosine similarity for each local document */
    SearchResult *local_results = malloc(my_num_docs * sizeof(SearchResult));
    for (int d = 0; d < my_num_docs; d++) {
        local_results[d].doc_id = my_ids[d];
        local_results[d].score  = cosine_similarity(
            &my_tfidf[d * vocab_size], query_vec, vocab_size);
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
    free(my_ids);
    free(query_vec);
    free(send_counts);
    free(displs);
    free(id_send_counts);
    free(id_displs);

    if (my_rank == 0) {
        free(tfidf_matrix);
        free(doc_ids);
    }

    MPI_Finalize();
    return 0;
}
