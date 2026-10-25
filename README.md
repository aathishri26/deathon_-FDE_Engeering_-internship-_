# deathon_-FDE_Engeering_-internship-_
#!/usr/bin/env python3
"""
Client Legacy Data Ingestion & Vector ETL Pipeline
--------------------------------------------------
Ingests PDFs, CSVs and markdown, does semantic chunking with metadata
preservation (department, access_tier, date), indexes into ChromaDB with
cosine similarity, supports metadata-filtered search, and benchmarks
batch ingestion.

Install:
    pip install chromadb sentence-transformers pypdf pandas numpy reportlab

Usage:
    python vector_etl.py sample                       # generate demo data in ./data
    python vector_etl.py ingest --root data           # build the index in ./chroma_db
    python vector_etl.py query "quarterly budget" --department finance \
        --max-tier internal --date-from 2025-01-01
    python vector_etl.py benchmark --root data        # batch-size + filter latency

Metadata resolution (later wins):
    1. folder layout:  <root>/<department>/<access_tier>/<file>
    2. date in filename (YYYY-MM-DD) or file modified time
    3. optional <root>/manifest.csv with columns: path,department,access_tier,date
"""
import argparse
import csv
import hashlib
import random
import re
import shutil
import time
from datetime import datetime
from pathlib import Path

import chromadb
import numpy as np
import pandas as pd
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
DB_PATH = "chroma_db"
COLLECTION = "client_docs"
TIERS = ["public", "internal", "confidential"]  # ordered low -> high
SUPPORTED = {".md", ".markdown", ".pdf", ".csv"}


# --------------------------------------------------------------------------
# Metadata
# --------------------------------------------------------------------------
def load_manifest(root: Path) -> dict:
    path = root / "manifest.csv"
    if not path.exists():
        return {}
    with open(path, newline="", encoding="utf-8") as f:
        return {row["path"]: row for row in csv.DictReader(f)}


def date_to_int(date_str: str) -> int:
    return int(date_str.replace("-", ""))


def resolve_metadata(path: Path, root: Path, manifest: dict) -> dict:
    rel = path.relative_to(root)
    meta = {"department": "general", "access_tier": "internal"}
    parts = rel.parts
    if len(parts) >= 2:
        meta["department"] = parts[0].lower()
    if len(parts) >= 3 and parts[1].lower() in TIERS:
        meta["access_tier"] = parts[1].lower()

    m = re.search(r"\d{4}-\d{2}-\d{2}", path.name)
    meta["date"] = m.group(0) if m else datetime.fromtimestamp(
        path.stat().st_mtime).strftime("%Y-%m-%d")

    override = manifest.get(rel.as_posix(), {})
    for key in ("department", "access_tier", "date"):
        if override.get(key):
            meta[key] = override[key].strip().lower() if key != "date" else override[key].strip()

    meta["date_int"] = date_to_int(meta["date"])  # numeric for range filters
    meta["source"] = rel.as_posix()
    meta["doc_type"] = path.suffix.lstrip(".").lower()
    return meta


# --------------------------------------------------------------------------
# Loaders: each returns a list of (text, extra_metadata) sections
# --------------------------------------------------------------------------
def load_markdown(path: Path):
    text = path.read_text(encoding="utf-8", errors="ignore")
    sections, heading, buf = [], "", []
    for line in text.splitlines():
        if re.match(r"^#{1,6}\s", line):
            if buf:
                sections.append(("\n".join(buf).strip(), {"heading": heading}))
            heading, buf = line.lstrip("# ").strip(), []
        else:
            buf.append(line)
    if buf:
        sections.append(("\n".join(buf).strip(), {"heading": heading}))
    return [(t, m) for t, m in sections if t]


def load_pdf(path: Path):
    reader = PdfReader(str(path))
    out = []
    for i, page in enumerate(reader.pages, start=1):
        text = (page.extract_text() or "").strip()
        if text:
            out.append((text, {"page": i}))
    return out


def load_csv(path: Path, rows_per_chunk: int = 20):
    df = pd.read_csv(path, dtype=str).fillna("")
    out = []
    for start in range(0, len(df), rows_per_chunk):
        block = df.iloc[start:start + rows_per_chunk]
        lines = [" | ".join(f"{c}: {v}" for c, v in row.items() if v)
                 for _, row in block.iterrows()]
        out.append(("\n".join(lines),
                    {"row_start": start, "row_end": start + len(block) - 1}))
    return out


LOADERS = {".md": load_markdown, ".markdown": load_markdown,
           ".pdf": load_pdf, ".csv": load_csv}


# --------------------------------------------------------------------------
# Semantic chunking
# --------------------------------------------------------------------------
def split_sentences(text: str):
    parts = re.split(r"(?<=[.!?])\s+|\n{2,}", text)
    return [p.strip() for p in parts if p and p.strip()]


def word_count(s: str) -> int:
    return len(s.split())


def semantic_chunk(text, model, max_words=300, min_words=60,
                   percentile=25, overlap_sentences=1):
    """Split where adjacent-sentence similarity dips (below the given
    percentile), bounded by min/max chunk size, with sentence overlap."""
    sents = split_sentences(text)
    if len(sents) <= 2:
        return [" ".join(sents)] if sents else []

    embs = model.encode(sents, batch_size=64, normalize_embeddings=True,
                        show_progress_bar=False)
    sims = (embs[:-1] * embs[1:]).sum(axis=1)
    threshold = np.percentile(sims, percentile)

    chunks, cur, cur_w = [], [sents[0]], word_count(sents[0])
    for i, s in enumerate(sents[1:]):
        w = word_count(s)
        topic_break = sims[i] < threshold and cur_w >= min_words
        too_big = cur_w + w > max_words
        if topic_break or too_big:
            chunks.append(" ".join(cur))
            cur = cur[-overlap_sentences:] if overlap_sentences else []
            cur_w = sum(word_count(x) for x in cur)
        cur.append(s)
        cur_w += w
    if cur:
        chunks.append(" ".join(cur))
    return chunks


def chunk_id(source: str, idx: int, text: str) -> str:
    h = hashlib.sha1(f"{source}:{idx}:{text}".encode()).hexdigest()[:16]
    return f"{Path(source).stem}-{idx}-{h}"


def build_chunks(root: Path, model) -> list:
    manifest = load_manifest(root)
    chunks = []
    files = [p for p in sorted(root.rglob("*")) if p.suffix.lower() in SUPPORTED]
    for path in files:
        try:
            base = resolve_metadata(path, root, manifest)
            sections = LOADERS[path.suffix.lower()](path)
        except Exception as e:  # keep going on messy legacy files
            print(f"  ! skipped {path}: {e}")
            continue
        idx = 0
        for text, extra in sections:
            pieces = ([text] if path.suffix.lower() == ".csv"
                      else semantic_chunk(text, model))
            for piece in pieces:
                meta = {**base, **extra, "chunk_index": idx}
                meta = {k: v for k, v in meta.items() if v is not None}
                chunks.append({"id": chunk_id(base["source"], idx, piece),
                               "text": piece, "metadata": meta})
                idx += 1
        print(f"  {path.relative_to(root)}: {idx} chunks")
    return chunks


# --------------------------------------------------------------------------
# Vector DB (ChromaDB, cosine)
# --------------------------------------------------------------------------
def get_collection(reset=False, path=DB_PATH, name=COLLECTION):
    client = chromadb.PersistentClient(path=path)
    if reset:
        try:
            client.delete_collection(name)
        except Exception:
            pass
    return client.get_or_create_collection(
        name=name, metadata={"hnsw:space": "cosine"}, embedding_function=None)


def ingest_chunks(col, model, chunks, batch_size=64):
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]
        texts = [c["text"] for c in batch]
        embs = model.encode(texts, batch_size=batch_size,
                            normalize_embeddings=True, show_progress_bar=False)
        col.upsert(ids=[c["id"] for c in batch],
                   embeddings=embs.tolist(),
                   documents=texts,
                   metadatas=[c["metadata"] for c in batch])


def build_where(department=None, max_tier=None, date_from=None, date_to=None):
    clauses = []
    if department:
        clauses.append({"department": department.lower()})
    if max_tier:
        allowed = TIERS[:TIERS.index(max_tier.lower()) + 1]
        clauses.append({"access_tier": {"$in": allowed}})
    if date_from:
        clauses.append({"date_int": {"$gte": date_to_int(date_from)}})
    if date_to:
        clauses.append({"date_int": {"$lte": date_to_int(date_to)}})
    if not clauses:
        return None
    return clauses[0] if len(clauses) == 1 else {"$and": clauses}


def search(col, model, query, k=5, **filters):
    emb = model.encode([query], normalize_embeddings=True,
                       show_progress_bar=False)
    res = col.query(query_embeddings=emb.tolist(), n_results=k,
                    where=build_where(**filters))
    return list(zip(res["documents"][0], res["metadatas"][0],
                    res["distances"][0]))


# --------------------------------------------------------------------------
# Sample data
# --------------------------------------------------------------------------
POOL = {
    "finance": [
        "The quarterly budget was revised to reflect higher cloud spending.",
        "Accounts receivable aging shows several invoices past 60 days.",
        "Forecasted revenue growth is 8 percent for the next fiscal year.",
        "Expense reports must be approved by a department head before payout.",
        "The audit committee reviewed internal controls over financial reporting.",
        "Cash flow projections assume stable payment terms with major clients.",
        "Capital expenditure requests above the threshold need CFO sign-off.",
        "Tax provisions were updated after the latest regulatory guidance.",
    ],
    "hr": [
        "New employees complete onboarding within their first two weeks.",
        "The leave policy allows carry-over of up to five vacation days.",
        "Performance reviews are conducted twice a year with manager input.",
        "Recruiting targets focus on engineering and data roles this quarter.",
        "Employee benefits enrollment opens every November.",
        "Remote work arrangements require written manager approval.",
        "Compensation bands are reviewed annually against market data.",
        "Harassment complaints are handled confidentially by the HR team.",
    ],
    "engineering": [
        "The service migrated from a monolith to containerized microservices.",
        "Database indexes were tuned to reduce query latency on reports.",
        "Incident response requires a postmortem within five business days.",
        "The CI pipeline runs unit tests and static analysis on every commit.",
        "API rate limits protect downstream services from traffic spikes.",
        "Legacy batch jobs will be replaced by event-driven pipelines.",
        "Secrets are stored in a managed vault and rotated every 90 days.",
        "Load tests showed the cache layer handles 10k requests per second.",
    ],
}


def make_sample(root: Path, n_md=24, n_csv=8, n_pdf=6):
    random.seed(7)
    if root.exists():
        shutil.rmtree(root)
    depts = list(POOL)

    def rand_path(prefix, ext):
        d = random.choice(depts)
        tier = random.choice(TIERS)
        date = f"{random.choice([2024, 2025, 2026])}-{random.randint(1, 12):02d}-{random.randint(1, 28):02d}"
        p = root / d / tier / f"{prefix}_{date}{ext}"
        p.parent.mkdir(parents=True, exist_ok=True)
        return d, p

    def body(dept, n):
        return " ".join(random.choices(POOL[dept], k=n))

    for i in range(n_md):
        d, p = rand_path(f"memo{i}", ".md")
        parts = [f"# {d.title()} Memo {i}"]
        for s in range(random.randint(2, 4)):
            parts.append(f"## Section {s + 1}\n\n{body(d, random.randint(8, 14))}")
        p.write_text("\n\n".join(parts), encoding="utf-8")

    for i in range(n_csv):
        d, p = rand_path(f"ledger{i}", ".csv")
        rows = [{"id": r, "owner": random.choice(["ana", "ben", "chi"]),
                 "amount": round(random.uniform(10, 5000), 2),
                 "note": random.choice(POOL[d])} for r in range(random.randint(40, 120))]
        pd.DataFrame(rows).to_csv(p, index=False)

    try:
        from reportlab.lib.pagesizes import letter
        from reportlab.pdfgen import canvas
        import textwrap
        for i in range(n_pdf):
            d, p = rand_path(f"report{i}", ".pdf")
            c = canvas.Canvas(str(p), pagesize=letter)
            for _page in range(random.randint(1, 3)):
                y = 750
                for line in textwrap.wrap(body(d, 25), 90):
                    c.drawString(50, y, line)
                    y -= 16
                c.showPage()
            c.save()
    except ImportError:
        print("reportlab not installed; skipping sample PDFs")
    print(f"Sample data written to {root}/")


# --------------------------------------------------------------------------
# Benchmark
# --------------------------------------------------------------------------
def run_benchmark(root: Path, batch_sizes, n_queries=50):
    model = SentenceTransformer(MODEL_NAME)
    print("Chunking once (excluded from ingestion timings)...")
    chunks = build_chunks(root, model)
    n_docs = len({c["metadata"]["source"] for c in chunks})
    results = []

    print(f"\n{n_docs} docs, {len(chunks)} chunks\n")
    print(f"{'batch':>6} {'secs':>8} {'docs/s':>9} {'chunks/s':>10}")
    col = None
    for bs in batch_sizes:
        col = get_collection(reset=True, path=f"{DB_PATH}_bench")
        t0 = time.perf_counter()
        ingest_chunks(col, model, chunks, batch_size=bs)
        dt = time.perf_counter() - t0
        row = {"batch_size": bs, "seconds": round(dt, 3),
               "docs_per_s": round(n_docs / dt, 2),
               "chunks_per_s": round(len(chunks) / dt, 2)}
        results.append(row)
        print(f"{bs:>6} {dt:>8.2f} {row['docs_per_s']:>9.1f} {row['chunks_per_s']:>10.1f}")

    # Query latency: DB time only (query embeddings precomputed)
    random.seed(1)
    sample = random.sample(chunks, min(n_queries, len(chunks)))
    q_embs = model.encode([" ".join(c["text"].split()[:12]) for c in sample],
                          normalize_embeddings=True, show_progress_bar=False)
    scenarios = {
        "no filter": None,
        "department": build_where(department="finance"),
        "tier<=internal": build_where(max_tier="internal"),
        "dept+tier+date": build_where(department="finance", max_tier="internal",
                                      date_from="2025-01-01"),
    }
    print(f"\n{'query scenario':<18} {'avg ms':>8} {'p95 ms':>8}")
    for name, where in scenarios.items():
        times = []
        for e in q_embs:
            t0 = time.perf_counter()
            col.query(query_embeddings=[e.tolist()], n_results=5, where=where)
            times.append((time.perf_counter() - t0) * 1000)
        print(f"{name:<18} {np.mean(times):>8.2f} {np.percentile(times, 95):>8.2f}")
        results.append({"scenario": name, "avg_ms": round(float(np.mean(times)), 2),
                        "p95_ms": round(float(np.percentile(times, 95)), 2)})

    pd.DataFrame(results).to_csv("benchmark_results.csv", index=False)
    print("\nSaved benchmark_results.csv")


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("sample")
    s.add_argument("--root", default="data")

    i = sub.add_parser("ingest")
    i.add_argument("--root", default="data")
    i.add_argument("--batch-size", type=int, default=64)
    i.add_argument("--reset", action="store_true")

    q = sub.add_parser("query")
    q.add_argument("text")
    q.add_argument("-k", type=int, default=5)
    q.add_argument("--department")
    q.add_argument("--max-tier", choices=TIERS)
    q.add_argument("--date-from")
    q.add_argument("--date-to")

    b = sub.add_parser("benchmark")
    b.add_argument("--root", default="data")
    b.add_argument("--batch-sizes", type=int, nargs="+", default=[32, 64, 128, 256])

    a = ap.parse_args()

    if a.cmd == "sample":
        make_sample(Path(a.root))
    elif a.cmd == "ingest":
        model = SentenceTransformer(MODEL_NAME)
        chunks = build_chunks(Path(a.root), model)
        col = get_collection(reset=a.reset)
        t0 = time.perf_counter()
        ingest_chunks(col, model, chunks, batch_size=a.batch_size)
        print(f"Indexed {len(chunks)} chunks in {time.perf_counter() - t0:.1f}s "
              f"(collection size: {col.count()})")
    elif a.cmd == "query":
        model = SentenceTransformer(MODEL_NAME)
        col = get_collection()
        hits = search(col, model, a.text, k=a.k, department=a.department,
                      max_tier=a.max_tier, date_from=a.date_from, date_to=a.date_to)
        for rank, (doc, meta, dist) in enumerate(hits, 1):
            print(f"\n#{rank}  cosine sim={1 - dist:.3f}  "
                  f"{meta['source']}  [{meta['department']}/{meta['access_tier']}/{meta['date']}]")
            print("   " + doc[:300].replace("\n", " "))
    elif a.cmd == "benchmark":
        run_benchmark(Path(a.root), a.batch_sizes)


if __name__ == "__main__":
    main()
