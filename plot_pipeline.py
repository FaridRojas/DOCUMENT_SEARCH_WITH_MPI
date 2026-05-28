#!/usr/bin/env python3
"""
plot_pipeline.py — Diagrama profesional del pipeline del MPI Document Search Engine.

Estilo "swimlane": cada proceso MPI (rank) es un carril vertical, de modo que el
PARALELISMO se ve graficamente. Las operaciones colectivas (Gatherv, Bcast,
Allreduce) son bandas que cruzan todos los carriles, y el merge de Top-K se
dibuja como un arbol de reduccion.

Refleja la implementacion real (search_engine.c):
  - Indexacion distribuida (cada rank carga su subconjunto)
  - Vocabulario: Gatherv local -> fusion en Rank 0 -> Bcast global
  - DF/IDF en paralelo via Allreduce
  - TF-IDF local en cada rank (NO se envia la matriz completa)
  - Merge jerarquico de Top-K en arbol (log2 P niveles)

Genera: figuras/pipeline_actual.png
"""

import os
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle

# ── Paleta profesional ──────────────────────────────────────────────────
NAVY = "#1F3B57"        # cabeceras / texto fuerte
LANE_A = "#F4F7FB"      # fondo carril (par)
LANE_B = "#EAF0F7"      # fondo carril (impar)
LOCAL = "#4F81BD"       # cajas de trabajo local
LOCAL_LT = "#DCE6F4"    # relleno claro trabajo local
COMM = "#E8943A"        # operaciones colectivas MPI
COMM_LT = "#FBE6CD"
MASTER = "#6FA34F"      # trabajo del master / resultado
MASTER_LT = "#DCE9CF"
GREY = "#8A98A6"
SHADOW = "#00000018"

fig, ax = plt.subplots(figsize=(14.5, 14))
ax.set_xlim(0, 16)
ax.set_ylim(0, 24.5)
ax.axis("off")
fig.patch.set_facecolor("white")

# ── Geometria de carriles ───────────────────────────────────────────────
LANE_X = [4.2, 7.2, 10.2, 13.2]          # centros de los 4 ranks
LANE_W = 2.55
BOX_W = 2.3
PHASE_X = 1.55                            # columna de etiquetas de fase
LANE_TOP = 23.0
LANE_BOT = 10.6


def shadow_box(x, y, w, h, fc, ec, text, fontsize=8.6, bold=False,
               text_color="#1c2733", rounding=0.12):
    """Caja redondeada con sombra sutil."""
    ax.add_patch(FancyBboxPatch(
        (x - w / 2 + 0.05, y - h / 2 - 0.07), w, h,
        boxstyle=f"round,pad=0.02,rounding_size={rounding}",
        linewidth=0, facecolor=SHADOW, zorder=2))
    ax.add_patch(FancyBboxPatch(
        (x - w / 2, y - h / 2), w, h,
        boxstyle=f"round,pad=0.02,rounding_size={rounding}",
        linewidth=1.6, edgecolor=ec, facecolor=fc, zorder=3))
    ax.text(x, y, text, ha="center", va="center", fontsize=fontsize,
            fontweight="bold" if bold else "normal", color=text_color,
            zorder=4, linespacing=1.25)


def varrow(x, y0, y1, color=NAVY, lw=2.0, z=2.5):
    ax.add_patch(FancyArrowPatch(
        (x, y0), (x, y1), arrowstyle="-|>", mutation_scale=15,
        linewidth=lw, color=color, zorder=z,
        shrinkA=0, shrinkB=0))


def phase_label(y, text, accent):
    ax.add_patch(FancyBboxPatch(
        (0.35, y - 0.5), 2.45, 1.0,
        boxstyle="round,pad=0.02,rounding_size=0.12",
        linewidth=0, facecolor=accent, zorder=3, alpha=0.18))
    ax.add_patch(Rectangle((0.35, y - 0.5), 0.12, 1.0, facecolor=accent,
                           edgecolor="none", zorder=4))
    ax.text(PHASE_X, y, text, ha="center", va="center", fontsize=9,
            fontweight="bold", color=NAVY, zorder=5, linespacing=1.1)


# ── Fondos de carriles + cabeceras de rank ──────────────────────────────
for i, cx in enumerate(LANE_X):
    ax.add_patch(Rectangle((cx - LANE_W / 2, LANE_BOT - 0.2), LANE_W,
                           LANE_TOP - LANE_BOT + 0.4,
                           facecolor=LANE_A if i % 2 == 0 else LANE_B,
                           edgecolor="#D5DEE9", linewidth=1.0, zorder=1))
    # cabecera
    ax.add_patch(FancyBboxPatch(
        (cx - LANE_W / 2 + 0.1, LANE_TOP - 0.05), LANE_W - 0.2, 0.9,
        boxstyle="round,pad=0.02,rounding_size=0.15",
        facecolor=NAVY, edgecolor="none", zorder=4))
    label = "Rank 0  (master)" if i == 0 else (
        f"Rank {i}" if i < 3 else "Rank P-1")
    ax.text(cx, LANE_TOP + 0.4, label, ha="center", va="center",
            fontsize=10, fontweight="bold", color="white", zorder=5)
    # icono de proceso (CPU)
    ax.text(cx - LANE_W / 2 + 0.45, LANE_TOP + 0.4, "▣",
            ha="center", va="center", fontsize=11, color="#9DBBDD", zorder=6)

# puntos suspensivos entre rank 2 y rank P-1 (indican mas ranks)
ax.text((LANE_X[2] + LANE_X[3]) / 2, (LANE_TOP + LANE_BOT) / 2, "· · ·",
        ha="center", va="center", fontsize=18, color=GREY, zorder=2)

# ── Filas de fase (centros y) ───────────────────────────────────────────
Y_LOAD = 21.4
Y_VOCAB = 19.7
Y_GATHER = 18.1
Y_MERGE = 16.5
Y_BCAST = 14.9
Y_DF = 13.2
Y_ALLRED = 11.6
Y_TFIDF = 10.0     # (debajo de los carriles -> ultima fase local)

# Para mantener TF-IDF y coseno dentro de carriles, extiendo carriles:
# (ya definidos hasta LANE_BOT=10.6; ajusto TFIDF y coseno dentro)
Y_TFIDF = 12.0
# reordeno: DF (13.2) -> Allreduce(11.6)? necesito espacio. Recoloco:
# Definicion final de filas:
Y_LOAD = 21.5
Y_VOCAB = 19.9
Y_GATHER = 18.4
Y_MERGE = 16.9
Y_BCAST = 15.4
Y_DF = 13.9
Y_ALLRED = 12.5
Y_TFIDF = 11.1
# carriles terminan en LANE_BOT=10.6, coseno/topk va justo arriba del borde
Y_COSINE = 11.1  # combinaremos tfidf y coseno? -> mejor separamos extendiendo

# ---- Re-extiendo carriles para que quepan TODAS las fases locales ----
# (rehago los fondos con limites correctos)

# ══════════════════════════════════════════════════════════════════════
#  Para simplicidad y limpieza, redibujo todo con un set de filas definitivo
# ══════════════════════════════════════════════════════════════════════
ax.clear()
ax.set_xlim(0, 16)
ax.set_ylim(0, 25.4)
ax.axis("off")

LANE_TOP = 23.0
LANE_BOT = 9.0

ROWS = {
    "load":   21.5,
    "vocab":  20.0,
    "gather": 18.5,
    "merge":  17.0,
    "bcast":  15.5,
    "df":     14.0,
    "allred": 12.6,
    "tfidf":  11.2,
    "cosine":  9.8,
}

# fondos de carriles + cabeceras
for i, cx in enumerate(LANE_X):
    ax.add_patch(Rectangle((cx - LANE_W / 2, LANE_BOT - 0.2), LANE_W,
                           LANE_TOP - LANE_BOT + 0.4,
                           facecolor=LANE_A if i % 2 == 0 else LANE_B,
                           edgecolor="#D5DEE9", linewidth=1.0, zorder=1))
    ax.add_patch(FancyBboxPatch(
        (cx - LANE_W / 2 + 0.1, LANE_TOP - 0.05), LANE_W - 0.2, 0.9,
        boxstyle="round,pad=0.02,rounding_size=0.15",
        facecolor=NAVY, edgecolor="none", zorder=4))
    label = "Rank 0 (master)" if i == 0 else (
        f"Rank {i}" if i < 3 else "Rank P-1")
    ax.text(cx, LANE_TOP + 0.4, label, ha="center", va="center",
            fontsize=10, fontweight="bold", color="white", zorder=5)
    ax.text(cx - LANE_W / 2 + 0.42, LANE_TOP + 0.4, "▣",
            ha="center", va="center", fontsize=11, color="#9DBBDD", zorder=6)

ax.text((LANE_X[2] + LANE_X[3]) / 2, (LANE_TOP + LANE_BOT) / 2, "· · ·",
        ha="center", va="center", fontsize=20, color=GREY, zorder=2)

# ── etiquetas de fase ───────────────────────────────────────────────────
phase_label(ROWS["load"],  "1. Carga\ndistribuida", LOCAL)
phase_label(ROWS["vocab"], "2. Vocabulario\nlocal", LOCAL)
phase_label(ROWS["gather"],"3. Reunir\nvocabularios", COMM)
phase_label(ROWS["merge"], "4. Vocabulario\nglobal", MASTER)
phase_label(ROWS["bcast"], "5. Difundir\nvocabulario", COMM)
phase_label(ROWS["df"],    "6. DF local", LOCAL)
phase_label(ROWS["allred"],"7. DF global\n(reduccion)", COMM)
phase_label(ROWS["tfidf"], "8. TF-IDF\nlocal + query", LOCAL)
phase_label(ROWS["cosine"],"9. Coseno +\nTop-K local", LOCAL)

# ── conectores verticales dentro de cada carril ─────────────────────────
def lane_connect(y_from, y_to):
    for cx in LANE_X:
        varrow(cx, y_from - 0.45, y_to + 0.45, color="#A9B6C4", lw=1.6)

# ── Fase 1: carga local (caja en cada carril) ───────────────────────────
for i, cx in enumerate(LANE_X):
    sub = ("docs 0..m" if i == 0 else
           ("docs m..2m" if i == 1 else
            ("docs 2m.." if i == 2 else "ultimo bloque")))
    shadow_box(cx, ROWS["load"], BOX_W, 0.95, LOCAL_LT, LOCAL,
               f"Cargar subset\n({sub})")

lane_connect(ROWS["load"], ROWS["vocab"])

# ── Fase 2: vocabulario local ───────────────────────────────────────────
for cx in LANE_X:
    shadow_box(cx, ROWS["vocab"], BOX_W, 0.85, LOCAL_LT, LOCAL,
               "Vocabulario\nLOCAL")


def comm_band(y, text, sub=""):
    """Banda colectiva que cruza todos los carriles."""
    x0 = LANE_X[0] - LANE_W / 2 - 0.1
    x1 = LANE_X[-1] + LANE_W / 2 + 0.1
    ax.add_patch(FancyBboxPatch(
        (x0 + 0.06, y - 0.42 - 0.07), x1 - x0, 0.84,
        boxstyle="round,pad=0.02,rounding_size=0.12",
        facecolor=SHADOW, linewidth=0, zorder=2))
    ax.add_patch(FancyBboxPatch(
        (x0, y - 0.42), x1 - x0, 0.84,
        boxstyle="round,pad=0.02,rounding_size=0.12",
        facecolor=COMM_LT, edgecolor=COMM, linewidth=1.8, zorder=3))
    ax.text((x0 + x1) / 2, y + 0.08, text, ha="center", va="center",
            fontsize=9.5, fontweight="bold", color="#8a4a00", zorder=5)
    if sub:
        ax.text((x0 + x1) / 2, y - 0.22, sub, ha="center", va="center",
                fontsize=7.8, color="#8a5a20", zorder=5, style="italic")


# ── Fase 3: Gatherv (convergen hacia rank 0) ────────────────────────────
# flechas de cada carril a la banda
for cx in LANE_X:
    varrow(cx, ROWS["vocab"] - 0.45, ROWS["gather"] + 0.45,
           color=COMM, lw=1.8)
comm_band(ROWS["gather"], "MPI_Gatherv",
          "cada rank envia su vocabulario al Rank 0")
# convergencia visual hacia rank 0
for cx in LANE_X[1:]:
    ax.add_patch(FancyArrowPatch(
        (cx, ROWS["gather"] - 0.45), (LANE_X[0], ROWS["merge"] + 0.5),
        arrowstyle="-|>", mutation_scale=13, linewidth=1.5,
        color=COMM, zorder=2.5, connectionstyle="arc3,rad=0.12"))
varrow(LANE_X[0], ROWS["gather"] - 0.45, ROWS["merge"] + 0.5, color=COMM, lw=1.8)

# ── Fase 4: fusion en rank 0 (solo master) ──────────────────────────────
shadow_box(LANE_X[0], ROWS["merge"], BOX_W + 0.3, 0.95, MASTER_LT, MASTER,
           "Fusionar ->\nVOCAB GLOBAL", bold=True, text_color="#2f4d1f")

# ── Fase 5: Bcast (se difunde a todos) ──────────────────────────────────
varrow(LANE_X[0], ROWS["merge"] - 0.5, ROWS["bcast"] + 0.45, color=COMM, lw=1.8)
comm_band(ROWS["bcast"], "MPI_Bcast",
          "el Rank 0 difunde el vocabulario global a todos")
for cx in LANE_X:
    if cx == LANE_X[0]:
        varrow(cx, ROWS["bcast"] - 0.45, ROWS["df"] + 0.45, color=COMM, lw=1.8)
    else:
        ax.add_patch(FancyArrowPatch(
            (LANE_X[0], ROWS["bcast"] - 0.45), (cx, ROWS["df"] + 0.45),
            arrowstyle="-|>", mutation_scale=13, linewidth=1.5,
            color=COMM, zorder=2.5, connectionstyle="arc3,rad=-0.12"))

# ── Fase 6: DF local ────────────────────────────────────────────────────
for cx in LANE_X:
    shadow_box(cx, ROWS["df"], BOX_W, 0.85, LOCAL_LT, LOCAL, "DF local")

# ── Fase 7: Allreduce (todos <-> todos) ─────────────────────────────────
for cx in LANE_X:
    varrow(cx, ROWS["df"] - 0.45, ROWS["allred"] + 0.45, color=COMM, lw=1.8)
comm_band(ROWS["allred"], "MPI_Allreduce (SUM)",
          "DF global identico en todos los ranks  ->  IDF")
for cx in LANE_X:
    varrow(cx, ROWS["allred"] - 0.45, ROWS["tfidf"] + 0.45, color=COMM, lw=1.8)

# ── Fase 8: TF-IDF local + query ────────────────────────────────────────
for cx in LANE_X:
    shadow_box(cx, ROWS["tfidf"], BOX_W, 0.95, LOCAL_LT, LOCAL,
               "TF-IDF local\n+ vector query")
lane_connect(ROWS["tfidf"], ROWS["cosine"])

# ── Fase 9: coseno + Top-K local ────────────────────────────────────────
for cx in LANE_X:
    shadow_box(cx, ROWS["cosine"], BOX_W, 0.95, LOCAL_LT, LOCAL,
               "Coseno ->\nTop-K LOCAL")

# ══════════════════════════════════════════════════════════════════════
#  Merge jerarquico en arbol (debajo de los carriles)
# ══════════════════════════════════════════════════════════════════════
ax.text(8.0, 8.2, "10.  Merge jerarquico de Top-K  —  reduccion en arbol  (log₂ P niveles)",
        ha="center", va="center", fontsize=10, fontweight="bold", color=NAVY)

def tnode(x, y, label, fc=COMM_LT, ec=COMM, tc="#8a4a00"):
    shadow_box(x, y, 1.15, 0.62, fc, ec, label, fontsize=8.6, bold=True,
               text_color=tc, rounding=0.1)

# nivel 0: las 4 listas Top-K locales (alineadas con los carriles)
yL0 = 7.2
for cx in LANE_X:
    varrow(cx, ROWS["cosine"] - 0.5, yL0 + 0.35, color="#A9B6C4", lw=1.6)
tnode(LANE_X[0], yL0, "Top-K\nR0")
tnode(LANE_X[1], yL0, "Top-K\nR1")
tnode(LANE_X[2], yL0, "Top-K\nR2")
tnode(LANE_X[3], yL0, "Top-K\nR3")

# nivel 1: R1->R0, R3->R2
yL1 = 5.6
ax.add_patch(FancyArrowPatch((LANE_X[1], yL0 - 0.35), (LANE_X[0] + 0.2, yL1 + 0.35),
             arrowstyle="-|>", mutation_scale=13, lw=1.7, color=COMM, zorder=2.5,
             connectionstyle="arc3,rad=0.0"))
varrow(LANE_X[0], yL0 - 0.35, yL1 + 0.35, color=COMM, lw=1.7)
ax.add_patch(FancyArrowPatch((LANE_X[3], yL0 - 0.35), (LANE_X[2] + 0.2, yL1 + 0.35),
             arrowstyle="-|>", mutation_scale=13, lw=1.7, color=COMM, zorder=2.5))
varrow(LANE_X[2], yL0 - 0.35, yL1 + 0.35, color=COMM, lw=1.7)
tnode(LANE_X[0], yL1, "merge\nR0")
tnode(LANE_X[2], yL1, "merge\nR2")
ax.text(LANE_X[1], yL1, "paso 1", ha="center", va="center", fontsize=8,
        style="italic", color=GREY)

# nivel 2: R2->R0
yL2 = 4.0
ax.add_patch(FancyArrowPatch((LANE_X[2], yL1 - 0.35), (LANE_X[0] + 0.25, yL2 + 0.4),
             arrowstyle="-|>", mutation_scale=14, lw=1.9, color=COMM, zorder=2.5,
             connectionstyle="arc3,rad=0.05"))
varrow(LANE_X[0], yL1 - 0.35, yL2 + 0.45, color=COMM, lw=1.9)
ax.text((LANE_X[0] + LANE_X[2]) / 2, yL1 - 1.0, "paso 2", ha="center",
        va="center", fontsize=8, style="italic", color=GREY)

# ── Resultado final ─────────────────────────────────────────────────────
shadow_box(LANE_X[0], yL2, 3.0, 1.0, MASTER_LT, MASTER,
           "Rank 0:\nTOP-K GLOBAL", bold=True, text_color="#2f4d1f")
ax.add_patch(FancyArrowPatch((LANE_X[0], yL2 - 0.55), (7.3, 2.6),
             arrowstyle="-|>", mutation_scale=16, lw=2.0, color=MASTER, zorder=2.5))
shadow_box(8.4, 2.2, 4.6, 0.95, "white", MASTER,
           "Ranking final de documentos\n(salida / CSV de tiempos)",
           fontsize=9, bold=True, text_color="#2f4d1f")

# ── Leyenda ─────────────────────────────────────────────────────────────
lx = 11.0
ly = 6.8
items = [(LOCAL_LT, LOCAL, "Trabajo local (todos los ranks en paralelo)"),
         (COMM_LT, COMM, "Comunicacion MPI (colectiva / por pares)"),
         (MASTER_LT, MASTER, "Trabajo del master / resultado")]
ax.text(lx, ly + 0.7, "Leyenda", fontsize=9.5, fontweight="bold", color=NAVY)
for k, (fc, ec, txt) in enumerate(items):
    yy = ly - k * 0.6
    ax.add_patch(FancyBboxPatch((lx, yy - 0.18), 0.5, 0.36,
                 boxstyle="round,pad=0.02,rounding_size=0.06",
                 facecolor=fc, edgecolor=ec, linewidth=1.4))
    ax.text(lx + 0.7, yy, txt, va="center", fontsize=8.6, color="#33414f")

# nota clave
ax.add_patch(FancyBboxPatch((10.7, 3.2), 4.7, 1.25,
             boxstyle="round,pad=0.1,rounding_size=0.1",
             facecolor="#FFF7E6", edgecolor="#E0C060", linewidth=1.4))
ax.text(13.05, 3.82,
        "Sin MPI_Scatterv de la matriz TF-IDF.\n"
        "Solo se transfieren el vocabulario global\ny las listas Top-K.",
        ha="center", va="center", fontsize=8.4, color="#6b5410")

# ── Titulo ──────────────────────────────────────────────────────────────
ax.text(8.0, 24.9, "Pipeline del MPI Document Search Engine",
        ha="center", va="center", fontsize=15.5, fontweight="bold", color=NAVY)
ax.text(8.0, 24.4, "Indexacion distribuida + busqueda Top-K paralela (implementacion actual)",
        ha="center", va="center", fontsize=10, color=GREY, style="italic")

plt.tight_layout()
os.makedirs("figuras", exist_ok=True)
out = "figuras/pipeline_actual.png"
plt.savefig(out, dpi=160, bbox_inches="tight", facecolor="white")
print(f"Diagrama guardado en {out}")
