from __future__ import annotations

import argparse
import hashlib
import json
from importlib.metadata import version
from pathlib import Path
from urllib.request import HTTPRedirectHandler, ProxyHandler, Request, build_opener

import chromadb
from langchain_chroma import Chroma
from langchain_core.embeddings import Embeddings

MODEL = "nomic-embed-text:latest"
COLLECTION = "bidmate_replay_v1"
BATCH = 96


class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


class LocalEmbeddings(Embeddings):
    def __init__(self):
        self.client = build_opener(ProxyHandler({}), NoRedirect())

    def _embed(self, texts):
        data = json.dumps(
            {"model": MODEL, "input": texts, "truncate": False, "keep_alive": "15m"},
            ensure_ascii=False,
        ).encode()
        request = Request(
            "http://127.0.0.1:11434/api/embed",
            data=data,
            headers={"Content-Type": "application/json"},
        )
        with self.client.open(request, timeout=180) as response:
            result = json.load(response)
        embeddings = result["embeddings"]
        if len(embeddings) != len(texts) or any(not item for item in embeddings):
            raise ValueError("Embedding count or dimension mismatch.")
        return embeddings

    def embed_documents(self, texts):
        return self._embed([f"search_document: {text}" for text in texts])

    def embed_query(self, text):
        return self._embed([f"search_query: {text}"])[0]


def digests(corpus, questions, projects):
    limit = ("corpus", "questions", "projects")
    return {
        name: hashlib.sha256(path.read_bytes()).hexdigest()
        for name, path in zip(limit, (corpus, questions, projects), strict=True)
    }


def open_database(root: Path, embedding=None):
    return Chroma(
        client=chromadb.PersistentClient(path=str(root)),
        collection_name=COLLECTION,
        embedding_function=embedding or LocalEmbeddings(),
    )


def main():
    parser = argparse.ArgumentParser()
    for name in ("corpus", "reference", "catalog", "output"):
        parser.add_argument(name, type=Path)
    args = parser.parse_args()
    reference = json.loads(args.reference.read_text())
    picked = {pid for row in reference["rows"] for pid in row["expected"] + row["selected"]}
    data = json.loads(args.corpus.read_text())
    chunks = [row for row in data["chunks"] if row["metadata"]["project_id"] in picked]
    wanted = {item["metadata"]["chunk_id"] for item in chunks}
    assert len(wanted) == len(chunks)
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = args.output / "manifest.json"
    keys = digests(args.corpus, args.reference, args.catalog)
    store = open_database(args.output)
    saved = set(store.get(include=[])["ids"])
    if saved and (
        not manifest.is_file()
        or json.loads(manifest.read_text())["input_sha256"] != keys
        or not saved <= wanted
    ):
        raise ValueError("Incompatible replay database")
    manifest.write_text(
        json.dumps(
            {
                "input_sha256": keys,
                "model": MODEL,
                "chromadb": version("chromadb"),
                "projects": len(picked),
                "chunks": len(chunks),
            },
            indent=2,
        )
    )
    pending = [item for item in chunks if item["metadata"]["chunk_id"] not in saved]
    print(
        "replay projects", len(picked), "chunks", len(chunks), "pending", len(pending), flush=True
    )
    for start in range(0, len(pending), BATCH):
        batch = pending[start : start + BATCH]
        store.add_texts(
            [item["page_content"] for item in batch],
            [item["metadata"] for item in batch],
            ids=[item["metadata"]["chunk_id"] for item in batch],
        )
        if (start // BATCH) % 6 == 0:
            print(
                "indexed",
                min(len(chunks), len(saved) + start + len(batch)),
                "/",
                len(chunks),
                flush=True,
            )
    assert set(store.get(include=[])["ids"]) == wanted
    print("indexed and verified", len(chunks), flush=True)


if __name__ == "__main__":
    main()
