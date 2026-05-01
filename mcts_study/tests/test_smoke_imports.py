def test_can_import_catan_mcts():
    import catan_mcts
    assert hasattr(catan_mcts, "__version__")


def test_can_import_dependencies():
    import open_spiel  # noqa: F401
    import pyarrow      # noqa: F401
    import pandas       # noqa: F401
    import numpy        # noqa: F401


def test_engine_module_available():
    from catan_bot import _engine
    v = _engine.engine_version()
    assert v.startswith("2."), f"expected v2 engine, got {v!r}"
