from llm_evals import hello


def test_hello_smoke() -> None:
    assert hello() == "Hello from llm-evals-platform"
