def test_engine_module_imports():
    from catan_bot import _engine
    assert _engine.engine_version() == "0.1.0"


def test_random_rollout_to_terminal_returns_alphazero_format():
    from catan_bot import _engine
    e = _engine.Engine(42)
    returns = e.random_rollout_to_terminal(7)
    assert e.is_terminal()
    rs = list(returns)
    assert rs.count(1.0) == 1, f"expected 1 winner, got {rs}"
    assert rs.count(-1.0) == 3, f"expected 3 losers, got {rs}"


def test_random_rollout_deterministic():
    from catan_bot import _engine
    a = _engine.Engine(42)
    ra = a.random_rollout_to_terminal(7)
    b = _engine.Engine(42)
    rb = b.random_rollout_to_terminal(7)
    assert list(ra) == list(rb)
    assert list(a.action_history()) == list(b.action_history())
