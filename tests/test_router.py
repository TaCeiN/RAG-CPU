from rag.router import route_query


def test_router_memory():
    d = route_query("Что я сказал выше?")
    assert d.route == "memory"


def test_router_direct():
    d = route_query("привет")
    assert d.route == "direct"


def test_router_rag():
    d = route_query("Как устроен деплой локального RAG?")
    assert d.route == "rag"
