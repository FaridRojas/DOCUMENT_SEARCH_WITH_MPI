#!/usr/bin/env python3
"""
plot_pipeline.py — Diagrama del pipeline REAL del MPI Document Search Engine.

Refleja la implementacion actual (search_engine.c):
  - Indexacion distribuida (todos los ranks cargan su subconjunto)
  - Vocabulario: Gatherv local -> fusion en Rank 0 -> Bcast global
  - DF/IDF en paralelo via Allreduce
  - TF-IDF local en cada rank (NO se envia la matriz completa)
  - Merge jerarquico de Top-K en arbol (log2 P niveles)

Genera: figuras/pipeline_actual.png
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

# ── Colores (estilo coherente con el resto de figuras) ──────────────────
C_LOCAL = "#cfe2f3"   # azul: trabajo local en todos los ranks
C_COMM = "#f4cccc"    # rojo: operaciones de comunicacion MPI
C_MASTER = "#d9ead3"  # verde: trabajo del master / resultado final
C_EDGE = "#666666"

fig, ax = plt.subplots(figsize=(9.5, 12))
ax.set_xlim(0, 10)
ax.set_ylim(0, 25)
ax.axis("off")


def box(x, y, w, h, text, color, fontsize=10, bold=False):
    p = FancyBboxPatch(
        (x - w / 2, y - h / 2), w, h,
        boxstyle="round,pad=0.02,rounding_size=0.15",
        linewidth=1.2, edgecolor=C_EDGE, facecolor=color,
    )
    ax.add_patch(p)
    ax.text(
        x, y, text, ha="center", va="center",
        fontsize=fontsize, fontweight="bold" if bold else "normal",
        wrap=True,
    )


def band(y, h, text):
    """Banda de fondo que agrupa fases 'en todos los ranks'."""
    ax.add_patch(
        FancyBboxPatch(
            (0.4, y - h / 2), 9.2, h,
            boxstyle="round,pad=0.02,rounding_size=0.1",
            linewidth=1.0, edgecolor="#bbbbbb",
            facecolor="#f7f7f7", linestyle="--",
        )
    )
    ax.text(0.7, y + h / 2 - 0.45, text, ha="left", va="center",
            fontsize=9.5, style="italic", color="#555555")


def arrow(x0, y0, x1, y1, text=None, dx_text=0.25):
    a = FancyArrowPatch(
        (x0, y0), (x1, y1),
        arrowstyle="-|>", mutation_scale=16,
        linewidth=1.4, color=C_EDGE,
    )
    ax.add_patch(a)
    if text:
        ax.text((x0 + x1) / 2 + dx_text, (y0 + y1) / 2, text,
                ha="left", va="center", fontsize=8.5, color="#333333")


# ══════════════════════════════════════════════════════════════════════
#  BANDA 1: trabajo local en TODOS los ranks (indexacion distribuida)
# ══════════════════════════════════════════════════════════════════════
band(22.0, 5.2, "TODOS LOS RANKS  (en paralelo desde el inicio)")
box(5, 23.4, 6.2, 0.9, "Listar archivos del corpus (orden global identico)", C_LOCAL)
arrow(5, 22.95, 5, 22.55)
box(5, 22.1, 6.2, 0.9, "Cargar SUBCONJUNTO local de documentos", C_LOCAL)
arrow(5, 21.65, 5, 21.25)
box(5, 20.8, 6.2, 0.9, "Construir vocabulario LOCAL", C_LOCAL)

# Gatherv vocab -> Rank 0
arrow(5, 20.35, 5, 19.55, "MPI_Gatherv(vocab local)  -->  Rank 0")
box(5, 19.05, 5.0, 0.95, "Rank 0: fusiona vocabularios\n-> VOCABULARIO GLOBAL", C_MASTER, bold=True)

# Bcast vocab global
arrow(5, 18.55, 5, 17.75, "MPI_Bcast(vocabulario global)")

# ══════════════════════════════════════════════════════════════════════
#  BANDA 2: DF/IDF + TF-IDF + busqueda local (todos los ranks)
# ══════════════════════════════════════════════════════════════════════
band(15.6, 5.2, "TODOS LOS RANKS")
box(5, 17.05, 6.6, 1.0,
    "DF local   --MPI_Allreduce(SUM)-->   DF global  ->  IDF", C_LOCAL)
arrow(5, 16.5, 5, 16.1)
box(5, 15.6, 6.6, 1.0,
    "TF-IDF de documentos LOCALES  +  vector TF-IDF de la query", C_LOCAL)
arrow(5, 15.05, 5, 14.65)
box(5, 14.15, 6.6, 1.0,
    "Similitud coseno por documento  ->  Top-K LOCAL (ordenado)", C_LOCAL)

# ══════════════════════════════════════════════════════════════════════
#  Merge jerarquico en arbol
# ══════════════════════════════════════════════════════════════════════
arrow(5, 13.65, 5, 12.55,
      "Merge jerarquico en ARBOL\n(log2 P niveles, Send/Recv por pares)")

# Pequena ilustracion del arbol de reduccion
def node(x, y, label):
    box(x, y, 0.95, 0.6, label, C_COMM, fontsize=8.5)

ax.text(5, 12.0, "Ejemplo de reduccion en arbol con P = 4:",
        ha="center", va="center", fontsize=8.5, style="italic", color="#555555")
# nivel 0
node(2.0, 11.0, "R0"); node(4.0, 11.0, "R1"); node(6.0, 11.0, "R2"); node(8.0, 11.0, "R3")
# nivel 1 (R1->R0, R3->R2)
arrow(4.0, 10.7, 2.45, 10.2)
arrow(8.0, 10.7, 6.45, 10.2)
node(2.0, 9.85, "R0"); node(6.0, 9.85, "R2")
# nivel 2 (R2->R0)
arrow(6.0, 9.55, 2.45, 9.05)
node(2.0, 8.7, "R0")
ax.text(8.4, 9.85, "cada par fusiona\ny conserva solo K",
        ha="left", va="center", fontsize=8, color="#777777")

arrow(2.0, 8.4, 5, 7.7)

box(5, 7.25, 5.4, 1.0, "Rank 0: TOP-K GLOBAL  (ranking final)", C_MASTER, bold=True)

# ── Leyenda ─────────────────────────────────────────────────────────────
ax.add_patch(FancyBboxPatch((0.6, 4.4), 0.5, 0.4, boxstyle="round,pad=0.02",
                            facecolor=C_LOCAL, edgecolor=C_EDGE))
ax.text(1.25, 4.6, "Trabajo local (todos los ranks)", va="center", fontsize=9)
ax.add_patch(FancyBboxPatch((0.6, 3.8), 0.5, 0.4, boxstyle="round,pad=0.02",
                            facecolor=C_COMM, edgecolor=C_EDGE))
ax.text(1.25, 4.0, "Comunicacion MPI", va="center", fontsize=9)
ax.add_patch(FancyBboxPatch((0.6, 3.2), 0.5, 0.4, boxstyle="round,pad=0.02",
                            facecolor=C_MASTER, edgecolor=C_EDGE))
ax.text(1.25, 3.4, "Trabajo del master / resultado", va="center", fontsize=9)

ax.text(5, 2.3,
        "Nota: NO existe MPI_Scatterv de la matriz TF-IDF (eliminado).\n"
        "Solo se transfieren el vocabulario global y las listas Top-K.",
        ha="center", va="center", fontsize=8.5, color="#444444",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#fff7e6",
                  edgecolor="#e0c060"))

ax.set_title("Pipeline del MPI Document Search Engine (implementacion actual)",
             fontsize=13, fontweight="bold", pad=14)

plt.tight_layout()
import os
os.makedirs("figuras", exist_ok=True)
out = "figuras/pipeline_actual.png"
plt.savefig(out, dpi=150, bbox_inches="tight")
print(f"Diagrama guardado en {out}")
