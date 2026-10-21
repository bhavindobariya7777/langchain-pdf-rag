from langchain_core.embeddings import Embeddings
from langchain_core.messages import AIMessage
from reportlab.pdfgen.canvas import Canvas

from pdf_rag.config import Settings
from pdf_rag.service import RagService


class FakeEmbeddings(Embeddings):
    """Deterministic offline embeddings; no paid API calls in tests."""

    def embed_documents(self, texts):
        return [[float("holiday" in text.lower()), 1.0, 0.5] for text in texts]

    def embed_query(self, text):
        return self.embed_documents([text])[0]


class FakeChat:
    def invoke(self, messages):
        assert "page 1" in messages[1].content
        return AIMessage(content="There are 20 holiday days [1].")


def make_pdf(path, text):
    canvas = Canvas(str(path))
    canvas.drawString(50, 750, text)
    canvas.save()


def test_incremental_lifecycle_and_retrieval(tmp_path):
    settings = Settings(_env_file=None, documents_dir=tmp_path / "docs", data_dir=tmp_path / "data")
    service = RagService(settings, embeddings=FakeEmbeddings(), chat=FakeChat())
    pdf = settings.documents_dir / "policy.pdf"
    make_pdf(pdf, "Staff receive 20 holiday days per year.")
    assert service.sync()["indexed"] == ["policy.pdf"]
    first_ids = service.store.get()["ids"]
    assert service.sync()["skipped"] == ["policy.pdf"]
    assert service.store.get()["ids"] == first_ids
    result = service.ask("How many holiday days?")
    assert result["sources"][0]["page"] == 1
    assert result["sources"][0]["file"] == "policy.pdf"
    assert "[1]" in result["answer"]
    make_pdf(pdf, "Staff receive 25 holiday days per year.")
    assert service.sync()["indexed"] == ["policy.pdf"]
    assert not set(first_ids) & set(service.store.get()["ids"])
    # A failed replacement preserves the previous complete version.
    pdf.write_bytes(b"invalid PDF")
    assert "policy.pdf" in service.sync()["errors"]
    assert service.store.get()["ids"]
    pdf.unlink()
    assert service.sync()["deleted"] == ["policy.pdf"]
    assert service.store.get()["ids"] == []
    assert service.ask("hello")["sources"] == []


def test_settings_reject_invalid_overlap():
    import pytest

    with pytest.raises(ValueError):
        Settings(_env_file=None, chunk_size=100, chunk_overlap=100)


def test_partial_write_is_repaired(tmp_path):
    settings = Settings(
        _env_file=None,
        documents_dir=tmp_path / "docs",
        data_dir=tmp_path / "data",
        chunk_size=100,
        chunk_overlap=10,
    )
    service = RagService(settings, embeddings=FakeEmbeddings())
    make_pdf(settings.documents_dir / "long.pdf", "holiday policy " * 50)
    service.sync()
    ids = service.store.get()["ids"]
    assert len(ids) > 1
    service.store.delete(ids=[ids[0]])
    assert service.sync()["indexed"] == ["long.pdf"]
    assert set(service.store.get()["ids"]) == set(ids)
