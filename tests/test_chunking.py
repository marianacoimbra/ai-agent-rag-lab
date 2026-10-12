from app.chunking import chunk_text


def test_empty_text_returns_no_chunks():
    assert chunk_text("") == []


def test_text_is_split_into_multiple_chunks():
    text = "0123456789" * 10

    chunks = chunk_text(
        text,
        chunk_size=20,
        overlap=5,
    )

    assert len(chunks) > 1
    assert all(chunks)


def test_adjacent_chunks_have_overlap():
    text = "0123456789" * 10

    chunks = chunk_text(
        text,
        chunk_size=20,
        overlap=5,
    )

    assert chunks[0][-5:] == chunks[1][:5]