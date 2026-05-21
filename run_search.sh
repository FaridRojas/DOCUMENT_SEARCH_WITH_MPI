#!/bin/bash
#SBATCH --job-name=DocSearch_MPI
#SBATCH --partition=legacy
#SBATCH --time=04:00:00
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=16
#SBATCH --mem-per-cpu=512
#SBATCH --output=docsearch_%j.out
#SBATCH --error=docsearch_%j.err

set -e

# ── Load MPI module ───────────────────────────────────────────────────
module purge
module --ignore_cache load "mpi/openmpi-gcc11/4.1.6"

which mpicc

# Disable CUDA support (legacy nodes have no GPU)
export OMPI_MCA_mpi_cuda_support=0

# ── Configuration ─────────────────────────────────────────────────────
SRC_MAIN="search_engine.c"
SRC_LIB="tfidf.c"
EXEC="search_engine"
REPS=3

PROCS=(1 2 4 8 16 32 64)
DOC_COUNTS=(100 500 1000 5000)
QUERY_TERMS=5
VOCAB_SIZE=5000
TOP_K=10

# Working directory (NFS, visible from all nodes)
WORK_DIR="$SLURM_SUBMIT_DIR"
cd "$WORK_DIR"

echo "============================================"
echo "  MPI Document Search Engine — SC3 Cluster"
echo "============================================"
echo "Node:       $(hostname)"
echo "Directory:  $WORK_DIR"
echo "Nodes:      $SLURM_JOB_NODELIST"
echo "Date:       $(date)"
echo ""

# ── Compilation ───────────────────────────────────────────────────────
echo "Compiling $EXEC ..."
mpicc -O2 -o "$EXEC" "$SRC_MAIN" "$SRC_LIB" -lm
echo "Compilation successful."
echo ""

# ── Results directory ─────────────────────────────────────────────────
RESULTS_DIR="results"
mkdir -p "$RESULTS_DIR"

RAW_FILE="$RESULTS_DIR/raw_results.csv"
echo "procs,num_docs,vocab_size,top_k,index_time,scatter_time,search_time,merge_time,total_time" > "$RAW_FILE"

EXEC_PATH="$WORK_DIR/$EXEC"

# ── Generate corpora at different sizes ───────────────────────────────
echo "Generating test corpora..."
for N_DOCS in "${DOC_COUNTS[@]}"; do
    CORPUS_DIR="corpus_${N_DOCS}"
    QUERY_FILE="query_${N_DOCS}.txt"

    if [ ! -d "$CORPUS_DIR" ]; then
        echo "  Generating $N_DOCS documents in $CORPUS_DIR/ ..."
        python3 generate_corpus.py \
            --num-docs "$N_DOCS" \
            --vocab-size "$VOCAB_SIZE" \
            --min-words 50 \
            --max-words 500 \
            --output-dir "$CORPUS_DIR" \
            --query-file "$QUERY_FILE" \
            --num-query-terms "$QUERY_TERMS" \
            --seed 42
    else
        echo "  $CORPUS_DIR/ already exists, skipping generation."
    fi
done
echo ""

# ── Run experiments ───────────────────────────────────────────────────
total_runs=$(( ${#DOC_COUNTS[@]} * ${#PROCS[@]} * REPS ))
run=0

echo "Starting $total_runs experiment runs..."
echo ""

for N_DOCS in "${DOC_COUNTS[@]}"; do
    CORPUS_DIR="corpus_${N_DOCS}"
    QUERY_FILE="query_${N_DOCS}.txt"
    QUERY=$(cat "$QUERY_FILE" | tr -d '\n')

    echo "--- Corpus: $N_DOCS documents ---"

    for P in "${PROCS[@]}"; do
        for r in $(seq 1 $REPS); do
            run=$((run + 1))
            echo "  Run $run/$total_runs  (P=$P, docs=$N_DOCS, rep=$r)"
            mpirun -np "$P" "$EXEC_PATH" "$CORPUS_DIR" "$QUERY" "$TOP_K" --csv >> "$RAW_FILE"
        done
    done
    echo ""
done

echo "Raw results written to $RAW_FILE"

# ── Generate performance tables ───────────────────────────────────────
echo ""
echo "Generating performance tables..."
python3 generate_tables.py --input "$RAW_FILE" --output-dir "$RESULTS_DIR"

echo ""
echo "============================================"
echo "  All done! Check $RESULTS_DIR/"
echo "============================================"
