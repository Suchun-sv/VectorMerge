"""Collect the reproduced tables and figures into one PDF laid out like the paper."""
import argparse
import datetime
import json
import re
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

plt.rcParams.update({"font.family": "serif", "mathtext.fontset": "dejavuserif", "pdf.fonttype": 42})
PAGE = (11.69, 8.27)
DATASETS = ["scifact", "nfcorpus", "arguana", "scidocs", "fiqa"]
NAMES = {"scifact": "SciFact", "nfcorpus": "NFCorpus", "arguana": "ArguAna",
         "scidocs": "SciDocs", "fiqa": "FiQA-2018"}
ROWS = [("union", "Union"), ("procrustes", "A2M"), ("gromov_wasserstein", "GW"), ("cca", "CCA"),
        ("nonlinear", r"A2M$_{\mathrm{MLP}}$"), ("la2m", "LA2M")]
CAPTIONS = {4: "Integrating Mistral into OpenAI-Ada vectors",
            5: "Integrating NV-Embed-V2 database into OpenAI-Ada database",
            6: "Integrating FastText database into OpenAI-Ada database",
            7: "Integrating FastText database into NV-Embed-V2",
            8: "Integrating GTE-Qwen2 database into NV-Embed-V2 database",
            9: "Integrating GloVe into FastText vectors"}
TABLE1_PAIRS = [(7, "Fast Text vs\nNV-embed V2"), (5, "NV-embed vs\nOpenai Ada"),
                (8, "gte-Qwen2 vs\nNV-embed V2"), (4, "Mistral vs\nOpenai Ada")]
PAIRS3 = [("FastText v.s. NV-Embed V2", "blue", 42), ("GTE-Qwen2 v.s. NV-Embed V2", "red", 55),
          ("Mistral v.s. OpenAI Ada", "green", 110)]


def layout(fig, y, segments, x0=0.06, x1=0.94, center=False, fontsize=11, step=0.03):
    """Lay out (text, bold) segments word by word between x0 and x1, wrapping on the measured width, as the paper sets a caption."""
    renderer = fig.canvas.get_renderer()
    words = [(w, bold) for text, bold in segments for w in re.findall(r"(?:\$[^$]*\$|\S)+", text)]
    space = fig.text(0, 0, "a a", fontsize=fontsize).get_window_extent(renderer).width \
        - fig.text(0, 0, "aa", fontsize=fontsize).get_window_extent(renderer).width
    for t in fig.texts[-2:]:
        t.remove()
    widths = []
    for w, bold in words:
        t = fig.text(0, 0, w, fontsize=fontsize, fontweight="bold" if bold else "normal")
        widths.append(t.get_window_extent(renderer).width / fig.bbox.width)
        t.remove()
    gap = space / fig.bbox.width
    lines, cur, cur_w = [], [], 0.0
    for (w, bold), wd in zip(words, widths):
        if cur and cur_w + gap + wd > x1 - x0:
            lines.append((cur, cur_w)); cur, cur_w = [], 0.0
        cur.append((w, bold, wd)); cur_w += (gap if len(cur) > 1 else 0) + wd
    if cur:
        lines.append((cur, cur_w))
    for line, line_w in lines:
        x = 0.5 - line_w / 2 if center else x0
        for w, bold, wd in line:
            fig.text(x, y - 0.7 * step, w, fontsize=fontsize, fontweight="bold" if bold else "normal",
                     va="baseline")
            x += wd + gap
        y -= step
    return y


def caption(fig, y, bold, rest=""):
    single = not rest
    return layout(fig, y, [(bold, True), (rest, False)], center=single)


def sub_caption(fig, y, bold, rest=""):
    return layout(fig, y, [(bold, True), (rest, False)], x0=0.12, x1=0.88, center=True,
                  fontsize=10, step=0.026)


def draw_grid(ax, header, rows, col_widths, fontsize=8):
    """A plain ruled table drawn with text, header rows given as lists of (text, span)."""
    ax.axis("off")
    total = sum(col_widths)
    ax.set_xlim(0, total); ax.set_ylim(0, 1)
    h = 1.0 / (len(header) + len(rows))
    y = 1.0
    starts = [sum(col_widths[:i]) for i in range(len(col_widths))]
    ax.plot([0, total], [y, y], color="k", lw=1.0)
    for hdr in header:
        c = 0
        for text, span in hdr:
            x0, x1 = starts[c], starts[c] + sum(col_widths[c:c + span])
            ax.text((x0 + x1) / 2, y - h / 2, text, ha="center", va="center", fontsize=fontsize,
                    fontweight="bold")
            c += span
        y -= h
    ax.plot([0, total], [y, y], color="k", lw=0.6)
    for row in rows:
        for c, text in enumerate(row):
            ax.text(starts[c] + col_widths[c] / 2, y - h / 2, text, ha="center", va="center",
                    fontsize=fontsize)
        y -= h
    ax.plot([0, total], [y, y], color="k", lw=1.0)


def load_cell(tables, t, d, m):
    f = tables / f"table{t}_{d}_{'procrustes' if m == 'union' else m}.json"
    if not f.exists():
        return None
    j = json.loads(f.read_text())
    if j.get("status") != "ok":
        return None
    r = j["results"]
    if m == "union":
        return {"R": 100 * r["direct_concat@100"]}
    return {"R": 100 * r["recall_at_100"], "R1": 100 * r["recall_d1@100"],
            "RN": 100 * r["relative_ndcg"], "beta": r["integration_error_beta"]}


def fmt(v):
    return "-" if v is None else f"{v:.2f}"


def title_page(pdf):
    fig = plt.figure(figsize=PAGE)
    fig.text(0.08, 0.80, "Reproduced results", fontsize=22, fontweight="bold")
    fig.text(0.08, 0.73, "Integrating Vector Databases across Embedding Models", fontsize=16)
    fig.text(0.08, 0.68, "Beining Yang, Yang Cao, Yang Ren. Proc. ACM Manag. Data 3(6), Article 338 "
                         "(SIGMOD 2025). doi:10.1145/3769803", fontsize=11)
    fig.text(0.08, 0.63, f"Generated {datetime.date.today().isoformat()} by LA2M_artifacts (run_all.sh)",
             fontsize=11, color="0.3")
    fig.text(0.08, 0.56, "Note: owing to subsequent work, the original embedding vectors have changed "
                         "somewhat since publication.", fontsize=11)
    fig.text(0.08, 0.53, "This does not affect the trends or the main results of the local isometries "
                         "(LA2M).", fontsize=11)
    pdf.savefig(fig); plt.close(fig)


def table1_page(pdf, tables):
    header = [[("emb1 vs emb2", 1)] + [(name, 3) for _, name in TABLE1_PAIRS],
              [("method", 1)] + [(m, 1) for _ in TABLE1_PAIRS for m in ("Union", "A2M", "LA2M")]]
    rows = []
    for d in DATASETS[:4]:
        row = [NAMES[d]]
        for t, _ in TABLE1_PAIRS:
            for m in ("union", "procrustes", "la2m"):
                c = load_cell(tables, t, d, m)
                row.append(fmt(c["R"]) if c else "n/a")
        rows.append(row)
    fig = plt.figure(figsize=PAGE)
    fig.text(0.5, 0.93, "Table 1. Recall@100 over integrated vector databases", ha="center",
             fontsize=12, fontweight="bold")
    ax = fig.add_axes([0.06, 0.55, 0.88, 0.33])
    draw_grid(ax, header, rows, [1.6] + [1.0] * 12, fontsize=10)
    pdf.savefig(fig); plt.close(fig)


def main_table_page(pdf, tables, t):
    header = [[("method", 1)] + [(NAMES[d], 4) for d in DATASETS],
              [("", 1)] + [(m, 1) for _ in DATASETS
                           for m in ("R(↑)", "R1(↑)", "RN(↑)", r"$\mathbf{\beta}$(↓)")]]
    rows = []
    for m, label in ROWS:
        row = [label]
        for d in DATASETS:
            c = load_cell(tables, t, d, m) or {}
            row += [fmt(c.get(k)) if c else "n/a" for k in ("R", "R1", "RN", "beta")]
        rows.append(row)
    fig = plt.figure(figsize=PAGE)
    fig.text(0.5, 0.93, f"Table {t}. {CAPTIONS[t]}", ha="center", fontsize=12, fontweight="bold")
    top = 0.88
    if t == 4:
        fig.text(0.02, 0.89, "Notations:", fontsize=9, fontweight="bold")
        fig.text(0.02, 0.865, "R: recall@100 (%); R1: recall@100 [D1] (%); RN: Relative NDCG (%); "
                              "↑: higher the better; ↓: lower the better", fontsize=9)
        top = 0.84
    ax = fig.add_axes([0.02, top - 0.36, 0.96, 0.36])
    draw_grid(ax, header, rows, [1.3] + [0.85] * 20)
    pdf.savefig(fig); plt.close(fig)


def fig3b_page(pdf, figures):
    d = json.loads((figures / "fig3b.json").read_text())
    k, m, s = np.array(d["k"]), np.array(d["mean"]), np.array(d["std"])
    fig = plt.figure(figsize=PAGE)
    ax = fig.add_axes([0.3, 0.38, 0.4, 0.5])
    ax.plot(k, m, color="blue", lw=1.2)
    ax.fill_between(k, m - s, m + s, color="blue", alpha=0.2, lw=0)
    ax.set_xlabel(r"$k$", fontsize=12); ax.set_ylabel("Jaccard Index", fontsize=12)
    ax.set_ylim(0.3, 1.0)
    y = sub_caption(fig, 0.27, r"(b) Consistency of top-$k$ search over SciFact with NV-Embed V2 and "
                               r"OpenAI-Ada", "(blue line is the average of 10 random queries; shaded "
                                              "area shows the variance.)")
    caption(fig, y - 0.04, "Fig. 3. Are real-life vector databases (approximately) isometric?")
    pdf.savefig(fig); plt.close(fig)


def fig5_page(pdf, figures):
    d = json.loads((figures / "fig5.json").read_text())
    n = [r["n"] for r in d["rows"]]; a = [r["alpha"] for r in d["rows"]]; b = [r["beta"] for r in d["rows"]]
    fig = plt.figure(figsize=PAGE)
    ax = fig.add_axes([0.25, 0.42, 0.5, 0.45]); ax2 = ax.twinx()
    la, = ax.plot(n, a, "-o", color="blue", ms=4, lw=1.2)
    lb, = ax2.plot(n, b, "--x", color="red", ms=5, lw=1.2)
    i = int(np.argmin(b))
    ax2.annotate(f"({n[i]}, {b[i]:.4f})", (n[i], b[i]), xytext=(25, 20), textcoords="offset points",
                 fontsize=10, bbox=dict(boxstyle="round", fc="white", ec="0.3"))
    ax.set_title("FastText v.s. NV-Embed on SciFact", fontsize=13)
    ax.set_xlabel(r"Number of grouped clusters of $\mathcal{X}$ (n)", fontsize=12)
    ax.set_ylabel(r"Alignment error $\alpha$ ($y_1$)", fontsize=12)
    ax2.set_ylabel(r"Integration error $\beta$ ($y_2$)", fontsize=12)
    ax.legend([la, lb], [r"$\alpha$ ($y_1$)", r"$\beta$ ($y_2$)"], loc="upper center", ncol=2, fontsize=11)
    caption(fig, 0.31, r"Fig. 5. #-clusters of $\mathcal{X}$ vs alignment error $\alpha$ and integration "
                       r"error $\beta$.",
            r"(alignment error $\alpha$: distance between $f^{j}_{\mathrm{iso}}(\mathbf{x})$ and $\mathbf{y}$ "
            r"when $\mathbf{x} \in X_j$ corresponds to $\mathbf{y} \in Y_j$ (step (2) error); $\beta$: "
            r"distance between $\mathcal{T}(\mathbf{x})$ and the vector would have been embedded by "
            r"emb$_2$ (step (3) error). $\mathcal{X}$ is grouped with k-meaning clustering.)")
    pdf.savefig(fig); plt.close(fig)


def fig7_page(pdf, figures):
    d = json.loads((figures / "fig7_fig7-version.json").read_text())
    fig = plt.figure(figsize=PAGE)
    ax = fig.add_axes([0.15, 0.38, 0.7, 0.42])
    handles = []
    for label, colour, dhat in PAIRS3:
        r = d[label]
        h, = ax.plot(r["sizes"], r["recall"], "-o", color=colour, ms=4, lw=1.2)
        ax.axvline(dhat, color=colour, ls="--", lw=1.2)
        handles.append(matplotlib.lines.Line2D([], [], color=colour, marker="o", ms=4, lw=1.2, ls="--"))
    ax.set_xscale("log")
    ticks = [1, 10, 42, 55, 110, 200, 300, 500, 700, 1000, 1500, 2000]
    ax.set_xticks(ticks); ax.set_xticklabels([str(x) for x in ticks], rotation=45)
    ax.minorticks_off()
    ax.set_ylim(0.2, 1.0)
    ax.set_xlabel("The size |X| of reference cluster X (log scale)", fontsize=12)
    ax.set_ylabel("Recall@100", fontsize=12)
    ax.grid(alpha=0.3)
    fig.legend(handles, [p[0] for p in PAIRS3], loc="upper center", bbox_to_anchor=(0.5, 0.93),
               ncol=2, fontsize=11)
    caption(fig, 0.2, r"Fig. 7. Integration accuracy (recall@100) Vs size of local reference clusters "
                      r"$X$ over SciFact:", r"the vertical dashed lines mark the intrinsic dimensions. "
                      r"Benchmark results verify that the accuracy (recall@100) peaks when $|X|$ is close "
                      r"to the intrinsic dimension in all three integration cases.")
    pdf.savefig(fig); plt.close(fig)


def fig8_page(pdf, figures):
    g = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    style = [("scifact", "SciFact", "red"), ("arguana", "ArguAna", "green"), ("nfcorpus", "NFCorpus", "blue")]
    fig = plt.figure(figsize=PAGE)
    ar = fig.add_axes([0.08, 0.42, 0.4, 0.4]); ab = fig.add_axes([0.57, 0.42, 0.4, 0.4])
    handles, labels = [], []
    for ds, name, colour in style:
        for kind, prefix, ls in (("Ours", "strat_", "-"), ("Rand", "", "--")):
            rs = [json.loads((figures / "fig8" / f"{prefix}{ds}_g{x}.json").read_text())["results"] for x in g]
            h, = ar.plot(g, [r["relative_ndcg"] for r in rs], ls, marker="o", color=colour, ms=4, lw=1.2)
            ab.plot(g, [r["integration_error_beta"] for r in rs], ls, marker="o", color=colour, ms=4, lw=1.2)
            handles.append(h); labels.append(f"{kind}@{name}")
    ar.set_ylabel("RN", fontsize=13); ab.set_ylabel(r"$\beta$", fontsize=13)
    for ax in (ar, ab):
        ax.set_xlabel(r"Reference ratio ($\gamma$)", fontsize=12); ax.grid(alpha=0.3)
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.92), ncol=6, fontsize=10)
    caption(fig, 0.27, "Fig. 8. Impact of reference vectors (FastText Vs NV-Embed-V2)")
    pdf.savefig(fig); plt.close(fig)


def fig9_page(pdf, figures):
    da = json.loads((figures / "fig9_paperdhat.json").read_text())
    dr = json.loads((figures / "fig9_fig9-version.json").read_text())
    fig = plt.figure(figsize=PAGE)
    aa = fig.add_axes([0.08, 0.42, 0.4, 0.4]); ar = fig.add_axes([0.57, 0.42, 0.4, 0.4])
    handles = []
    for label, colour, _ in PAIRS3:
        k = [x for x in da[label]["k"] if x <= 1000]
        h, = aa.plot(k, da[label]["alpha"][: len(k)], "-o", color=colour, ms=3, lw=1.2)
        ar.plot(dr[label]["k"], dr[label]["recall"], "-o", color=colour, ms=3, lw=1.2)
        handles.append(h)
    aa.set_ylabel(r"$\alpha$", fontsize=14); ar.set_ylabel("Recall@100", fontsize=12)
    for ax in (aa, ar):
        ax.set_xscale("log"); ax.set_xlabel(r"Number of clusters $k$ (log scale)", fontsize=12)
        ax.grid(alpha=0.3)
    fig.legend(handles, [p[0] for p in PAIRS3], loc="upper center", bbox_to_anchor=(0.5, 0.92),
               ncol=3, fontsize=11)
    caption(fig, 0.27, "Fig. 9. Global Vs. local isometries")
    pdf.savefig(fig); plt.close(fig)


def fig10_page(pdf, figures, data):
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from fig10_repro import GRID_QUERIES
    root = data / "cifar10"
    z = np.load(root / "cifar10.npz")
    train_img, test_img = z["train_x"], z["test_x"]
    qtrain = np.load(root / "emb" / "q_train_idx.npy")
    tops = dict(np.load(figures / "fig10" / "tops_bob.npz"))
    rows = [([train_img[qtrain[q]] for q in GRID_QUERIES],
             "(a) Alice's query image: 10 images from the CIFAR-10 dataset"),
            ([test_img[tops["union"][q]] for q in GRID_QUERIES],
             "(b) Top-1 image with Union: each corresponds to the query image directly above"),
            ([test_img[tops["la2m (n=100)"][q]] for q in GRID_QUERIES],
             "(c) Top-1 image with LA2M: each corresponds to the query image above in (a)")]
    fig = plt.figure(figsize=PAGE)
    for r, (imgs, sub) in enumerate(rows):
        y0 = 0.74 - r * 0.21
        for c, img in enumerate(imgs):
            ax = fig.add_axes([0.16 + c * 0.069, y0, 0.06, 0.12]); ax.imshow(img); ax.axis("off")
        fig.text(0.5, y0 - 0.015, sub, ha="center", va="top", fontsize=11, fontweight="bold")
    caption(fig, 0.2, "Fig. 10. Privacy-preserving image search via vector integration")
    pdf.savefig(fig); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tables", default="/app/results/tables")
    ap.add_argument("--figures", default="/app/figures")
    ap.add_argument("--data", default="/app/data")
    ap.add_argument("--out", default="/app/output/report.pdf")
    a = ap.parse_args()
    tables, figures, data, out = Path(a.tables), Path(a.figures), Path(a.data), Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    pages = [lambda pdf: fig3b_page(pdf, figures), lambda pdf: fig5_page(pdf, figures),
             lambda pdf: fig7_page(pdf, figures), lambda pdf: fig8_page(pdf, figures),
             lambda pdf: fig9_page(pdf, figures), lambda pdf: fig10_page(pdf, figures, data)]
    with PdfPages(out) as pdf:
        title_page(pdf)
        table1_page(pdf, tables)
        for t in range(4, 10):
            main_table_page(pdf, tables, t)
        for page in pages:
            try:
                page(pdf)
            except FileNotFoundError as exc:
                print(f"skipped a figure, missing input: {exc.filename}")
    print(f"saved {out}")


if __name__ == "__main__":
    main()
