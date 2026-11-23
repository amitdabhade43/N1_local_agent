import ninja_agent


class FakeIndex:
    def __init__(self, results):
        self.results = results

    def search_scored(self, query, limit):
        return self.results[:limit]


def operation(path="/v2/software-products", summary="List software products", score=2.0):
    return score, {
        "id": "getSoftwareProducts",
        "method": "GET",
        "path": path,
        "summary": summary,
        "read_only": True,
        "deprecated": False,
        "params": [],
    }


def test_catalog_router_counts_a_spec_discovered_collection(monkeypatch):
    rows = ninja_agent.FetchRows([
        {"id": 1, "name": "Product A"},
        {"id": 2, "name": "Product B"},
        {"id": 3, "name": "Product C"},
    ])
    monkeypatch.setattr(ninja_agent, "IDX", FakeIndex([operation()]))
    monkeypatch.setattr(ninja_agent, "_response_list_field", lambda op: "results")
    monkeypatch.setattr(ninja_agent, "fetch", lambda *args, **kwargs: rows)

    answer = ninja_agent.recipe_catalog_collection_read("How many software products do we have?")

    assert answer == "3 software products in NinjaOne."


def test_catalog_router_lists_names_from_a_spec_discovered_collection(monkeypatch):
    rows = ninja_agent.FetchRows([
        {"id": 1, "name": "Product A"},
        {"id": 2, "name": "Product B"},
    ])
    monkeypatch.setattr(ninja_agent, "IDX", FakeIndex([operation()]))
    monkeypatch.setattr(ninja_agent, "_response_list_field", lambda op: "results")
    monkeypatch.setattr(ninja_agent, "fetch", lambda *args, **kwargs: rows)

    answer = ninja_agent.recipe_catalog_collection_read("List software products")

    assert "2 software products; showing 2." in answer
    assert "1. Product A" in answer
    assert "2. Product B" in answer


def test_catalog_router_never_claims_a_partial_count_is_complete(monkeypatch):
    rows = ninja_agent.FetchRows(
        [{"id": 1, "name": "Product A"}],
        complete=False,
        note="The next page could not be retrieved.",
    )
    monkeypatch.setattr(ninja_agent, "IDX", FakeIndex([operation()]))
    monkeypatch.setattr(ninja_agent, "_response_list_field", lambda op: "results")
    monkeypatch.setattr(ninja_agent, "fetch", lambda *args, **kwargs: rows)

    answer = ninja_agent.recipe_catalog_collection_read("How many software products?")

    assert "At least 1" in answer
    assert "unknown" in answer.casefold()


def test_catalog_router_defers_filtered_or_relational_requests(monkeypatch):
    monkeypatch.setattr(ninja_agent, "IDX", FakeIndex([operation()]))
    assert ninja_agent.recipe_catalog_collection_read(
        "How many software products are active in each organization?"
    ) is None


def test_catalog_router_defers_mutations(monkeypatch):
    monkeypatch.setattr(ninja_agent, "IDX", FakeIndex([operation()]))
    assert ninja_agent.recipe_catalog_collection_read("Create a software product") is None


def test_catalog_router_defers_ambiguous_endpoint_matches(monkeypatch):
    monkeypatch.setattr(ninja_agent, "IDX", FakeIndex([
        operation("/v2/software-products", "List software products", 2.0),
        operation("/v2/software", "List software", 1.9),
    ]))
    monkeypatch.setattr(ninja_agent, "_response_list_field", lambda op: "results")

    assert ninja_agent.recipe_catalog_collection_read("How many software products?") is None
