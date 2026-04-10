from rag.ingestion import chunk_text


class FakeTokenizer:
    max_length = 512

    def split_tokens(self, text: str):
        return text.split()

    def count_tokens(self, text: str) -> int:
        return len(text.split())


def test_chunking_overlap_and_size():
    text = " ".join(f"w{i}" for i in range(1000))
    chunks = chunk_text(
        source="test.md",
        text=text,
        tokenizer=FakeTokenizer(),
        chunk_size=400,
        chunk_overlap=60,
    )
    assert chunks
    assert all(1 <= c.token_count <= 400 for c in chunks)
    if len(chunks) > 1:
        first = chunks[0].text.split()
        second = chunks[1].text.split()
        assert first[-60:] == second[:60]
