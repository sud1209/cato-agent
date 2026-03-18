from langchain_core.documents import Document
from app.rag.retriever import reciprocal_rank_fusion


def _doc(id: str, content: str = "") -> Document:
    d = Document(page_content=content)
    d.id = id
    return d


def test_rrf_preserves_all_unique_docs():
    list_a = [_doc("a"), _doc("b"), _doc("c")]
    list_b = [_doc("b"), _doc("d")]
    result = reciprocal_rank_fusion([list_a, list_b])
    ids = [d.id for d in result]
    assert set(ids) == {"a", "b", "c", "d"}


def test_rrf_ranks_common_doc_higher():
    list_a = [_doc("x"), _doc("shared"), _doc("y")]
    list_b = [_doc("z"), _doc("shared")]
    result = reciprocal_rank_fusion([list_a, list_b])
    ids = [d.id for d in result]
    assert ids.index("shared") < ids.index("x")


def test_rrf_caps_output_at_15():
    big_list = [_doc(str(i)) for i in range(20)]
    result = reciprocal_rank_fusion([big_list])
    assert len(result) == 15


def test_rrf_single_list_preserves_order():
    docs = [_doc(str(i)) for i in range(5)]
    result = reciprocal_rank_fusion([docs])
    assert [d.id for d in result] == [d.id for d in docs]
