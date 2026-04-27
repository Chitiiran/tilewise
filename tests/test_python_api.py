def test_engine_module_imports():
    from catan_bot import _engine
    assert _engine.engine_version() == "0.1.0"
