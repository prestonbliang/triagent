import json
from pathlib import Path

from triagent.config import Settings
from triagent.ingest import cve_to_document
from triagent.rag.embeddings import HashEmbedder, get_embedder
from triagent.rag.store import MemoryStore, get_store

FIXTURE = Path(__file__).parent / "fixtures" / "nvd_sample.json"


def _docs():
    data = json.loads(FIXTURE.read_text())["vulnerabilities"]
    return [cve_to_document(v) for v in data]


def test_memory_store_retrieves_relevant_cve():
    store = MemoryStore(HashEmbedder(dim=384))
    store.add(_docs())
    assert store.count() == 3
    # A query about RDP remote code execution should surface BlueKeep first.
    hits = store.search("unauthenticated remote desktop RDP remote code execution", k=3)
    assert hits[0].document.doc_id == "CVE-2019-0708"


def test_hybrid_exact_identifier_wins():
    store = MemoryStore(HashEmbedder(dim=384))
    store.add(_docs())
    hits = store.search("Heartbleed OpenSSL TLS heartbeat buffer over-read", k=3)
    assert hits[0].document.doc_id == "CVE-2014-0160"


def test_fake_embedder_is_deterministic():
    e = HashEmbedder(dim=128)
    a = e.encode_one("JNDI LDAP log4j remote code execution")
    b = e.encode_one("JNDI LDAP log4j remote code execution")
    assert (a == b).all()
    # Normalized.
    assert abs(float((a * a).sum()) - 1.0) < 1e-5


def test_get_embedder_defaults_to_fake():
    emb = get_embedder(Settings(embedder="fake", embed_dim=384))
    assert isinstance(emb, HashEmbedder)


def test_get_store_singleton_and_fresh():
    s = Settings(store="memory")
    a = get_store(s)
    b = get_store(s)
    assert a is b
    c = get_store(s, fresh=True)
    assert c is not a
