"""One live interactive game: engine + bots + cooperative driving loop.

The session owns a CatanState and three bots, drives chance + bot turns,
and yields control to the human at their turn or when a bot's ProposeTrade
would auto-match the human (the trade-intercept; see advance()).
"""
from __future__ import annotations

import base64
import random
import tempfile
import threading
from pathlib import Path

from catan_mcts.adapter import CatanGame
from catan_mcts.web import bot_registry, board_layout, serializers, action_decode, trade_logic


class _MaskedLegalView:
    """Wraps a CatanState so legal_actions() omits one action; all else delegates.

    clone() preserves the mask (returns another _MaskedLegalView) so tree-search
    bots like OpenSpiel's MCTSBot also see the masked action set at their tree
    root, not just the top-level step() caller.
    """
    def __init__(self, state, masked_action: int):
        self._state = state
        self._masked = int(masked_action)

    def legal_actions(self):
        return [a for a in self._state.legal_actions() if int(a) != self._masked]

    def clone(self):
        return _MaskedLegalView(self._state.clone(), self._masked)

    def __getattr__(self, name):
        return getattr(self._state, name)


END_TURN = 204


class GameSession:
    def __init__(self, setup: dict) -> None:
        self.human_seat = int(setup["human_seat"])
        if not (0 <= self.human_seat <= 3):
            raise ValueError(f"human_seat must be 0..3, got {self.human_seat}")
        rules = setup.get("rules", {})
        self._vp_target = int(rules.get("vp_target", 10))
        self._bonuses = bool(rules.get("bonuses", True))
        self.seed = int(setup.get("seed") if setup.get("seed") is not None
                        else random.Random().randint(1, 2**31 - 1))
        self._game = CatanGame(vp_target=self._vp_target, bonuses=self._bonuses)
        self._state = self._game.new_initial_state(seed=self.seed)
        self._rng = random.Random(self.seed ^ 0x5EED)
        self._bots: dict[int, object] = {}
        self._seat_specs: dict[int, dict] = {}
        for seat_str, spec in setup["seats"].items():
            seat = int(seat_str)
            self._bots[seat] = bot_registry.build(spec, game=self._game, seed=self.seed + seat)
            self._seat_specs[seat] = spec
        missing = [s for s in range(4) if s != self.human_seat and s not in self._seat_specs]
        if missing:
            raise ValueError(f"missing bot spec for seat(s) {missing}")
        self._pending_trade = None
        self._last_narration = "(game start)"
        self._error = None
        self._thread = None
        # RLock (re-entrant): advance() -> apply_human_action()/respond_to_trade()
        # paths re-acquire; the server reads state_json() from the main thread
        # while advance_async()'s daemon thread mutates the engine.
        self._lock = threading.RLock()

    def seat_names(self) -> list[str]:
        names = []
        for s in range(4):
            if s == self.human_seat:
                names.append("You")
            else:
                names.append(f"P{s} {self._seat_specs[s]['type']}")
        return names

    def board_payload(self) -> dict:
        vertex_xy, _, _ = board_layout.build_layout()
        with tempfile.TemporaryDirectory() as td:
            png = Path(td) / "board.png"
            board_layout.render_board_png(self.seed, png, vertex_xy=vertex_xy)
            b64 = base64.b64encode(png.read_bytes()).decode("ascii")
        return {"layout": board_layout.layout_dict(), "png_b64": b64}

    def _status(self) -> str:
        if self._error is not None:
            return "error"
        if self._state.is_terminal():
            return "game_over"
        if self._pending_trade is not None:
            return "trade_offer"
        if self._state.is_chance_node():
            # A chance node is never the human's decision point.
            return "bot_thinking"
        if int(self._state.current_player()) == self.human_seat:
            return "your_turn"
        return "bot_thinking"

    def state_json(self) -> dict:
        with self._lock:
            eng = self._state._engine
            status = self._status()
            out = {
                "status": status,
                "human_seat": self.human_seat,
                "current_player": -1 if eng.is_terminal() else int(eng.current_player()),
                "phase": None,
                "narration": self._last_narration,
                "state": serializers.serialize_state(eng, self._last_narration),
                "seat_names": self.seat_names(),
            }
            out["phase"] = out["state"]["phase"]
            if status == "your_turn":
                out["legal_actions"] = action_decode.decode_many(self._state.legal_actions())
            if status == "trade_offer":
                out["trade_offer"] = self._trade_offer_payload()
            if status == "game_over":
                out["returns"] = self._state.returns()
            if status == "error":
                out["error"] = str(self._error)
            return out

    def _trade_offer_payload(self) -> dict:
        proposer, action = self._pending_trade
        give, get = trade_logic.decode_propose_trade(action)
        # The proposer gives `give` and wants `get`. From the human's view this
        # is mirrored: you_give = the resource the bot wants (get); you_get = the
        # resource the bot offers (give). Each side is a 1-for-1 swap.
        return {"from_seat": proposer, "you_give": [get, 1], "you_get": [give, 1]}

    def _sample_chance(self) -> int:
        outcomes = self._state.chance_outcomes()
        r = self._rng.random()
        cum = 0.0
        for v, p in outcomes:
            cum += p
            if r <= cum:
                return int(v)
        return int(outcomes[-1][0])

    def advance(self, max_steps: int = 100000) -> dict:
        """Run chance + bot turns until human turn / trade offer / terminal."""
        with self._lock:
            steps = 0
            while steps < max_steps:
                if self._error is not None:
                    return self.state_json()
                if self._state.is_terminal():
                    return self.state_json()
                if self._state.is_chance_node():
                    self._state.apply_action(self._sample_chance())
                    steps += 1
                    continue
                cp = int(self._state.current_player())
                if cp == self.human_seat:
                    return self.state_json()
                legal = self._state.legal_actions()
                if len(legal) == 1:
                    self._apply_and_narrate(int(legal[0]), cp)
                    steps += 1
                    continue
                try:
                    action = int(self._bots[cp].step(self._state))
                except Exception as e:
                    self._error = f"bot P{cp} errored: {e}"
                    return self.state_json()
                if self._maybe_intercept_trade(cp, action):
                    return self.state_json()
                self._apply_and_narrate(action, cp)
                steps += 1
            # Every normal exit is via an explicit return inside the loop; reaching
            # here means the step cap was hit without the game terminating.
            self._error = "step cap exceeded (game did not terminate within max_steps)"
            return self.state_json()

    def _apply_and_narrate(self, action: int, player: int) -> None:
        self._last_narration = f"P{player} {serializers.action_desc(action)}"
        self._state.apply_action(int(action))

    def apply_human_action(self, action: int) -> dict:
        with self._lock:
            if int(self._state.current_player()) != self.human_seat:
                raise ValueError("not your turn")
            legal = self._state.legal_actions()
            if int(action) not in legal:
                raise ValueError(f"illegal action {action}")
            self._apply_and_narrate(int(action), self.human_seat)
            return self.advance()

    def _predict_trade_acceptor(self, current_player: int, action: int) -> int:
        """Seat the engine would auto-match for this ProposeTrade, else -1.

        Assumes `action` is already in the ProposeTrade range; the entry-point
        guard lives in _maybe_intercept_trade. Delegates decode + acceptor
        scan to trade_logic so the rule mirror lives in one place.
        """
        give, get = trade_logic.decode_propose_trade(action)
        hands = [list(h) for h in self._state._engine.all_hands()]
        return trade_logic.first_acceptor(current_player, give, get, hands)

    def _maybe_intercept_trade(self, current_player: int, action: int) -> bool:
        """If this bot ProposeTrade would auto-match the human, pause. Returns
        True iff intercepted (caller must stop driving and surface trade_offer)."""
        base = trade_logic.PROPOSE_TRADE_BASE
        if not (base <= int(action) < base + 20):
            return False
        if self._predict_trade_acceptor(current_player, action) == self.human_seat:
            self._pending_trade = (current_player, int(action))
            return True
        return False

    def respond_to_trade(self, accept: bool) -> dict:
        with self._lock:
            if self._pending_trade is None:
                return self.advance()
            proposer, action = self._pending_trade
            self._pending_trade = None
            if accept:
                self._apply_and_narrate(action, proposer)
            else:
                substitute = self._requery_bot_masked(proposer, masked_action=action)
                self._apply_and_narrate(substitute, proposer)
            return self.advance()

    def _requery_bot_masked(self, seat: int, masked_action: int) -> int:
        """Ask the bot for an action with `masked_action` removed; else EndTurn."""
        legal = [a for a in self._state.legal_actions() if int(a) != int(masked_action)]
        if not legal:
            return END_TURN
        try:
            a = int(self._bots[seat].step(_MaskedLegalView(self._state, masked_action)))
            if a in legal:
                return a
        except Exception:
            pass
        return END_TURN if END_TURN in legal else int(legal[0])

    def apply_human_action_async(self, action: int) -> dict:
        """Apply the human action, then drive bots in the background.

        Returns the immediate state snapshot (status likely bot_thinking, but
        with fast bots it may already show the settled status). The `with
        self._lock:` block ends BEFORE advance_async() so the daemon thread can
        re-acquire the lock inside advance() without deadlocking (RLock is
        per-thread; the worker is a different thread)."""
        with self._lock:
            if int(self._state.current_player()) != self.human_seat:
                raise ValueError("not your turn")
            legal = self._state.legal_actions()
            if int(action) not in legal:
                raise ValueError(f"illegal action {action}")
            self._apply_and_narrate(int(action), self.human_seat)
        self.advance_async()
        return self.state_json()

    def respond_to_trade_async(self, accept: bool) -> dict:
        """Resolve the pending trade, then drive bots in the background."""
        with self._lock:
            if self._pending_trade is not None:
                proposer, action = self._pending_trade
                self._pending_trade = None
                if accept:
                    self._apply_and_narrate(action, proposer)
                else:
                    substitute = self._requery_bot_masked(proposer, masked_action=action)
                    self._apply_and_narrate(substitute, proposer)
        self.advance_async()
        return self.state_json()

    def start_async(self) -> dict:
        """Kick off the opening bot turns in the background (game creation)."""
        self.advance_async()
        return self.state_json()

    def advance_async(self) -> None:
        """Fire-and-forget: run advance() in a daemon thread; no-op if one is
        already running. Poll is_advancing()/state_json() for progress."""
        if getattr(self, "_thread", None) is not None and self._thread.is_alive():
            return
        self._thread = threading.Thread(target=self.advance, daemon=True)
        self._thread.start()

    def is_advancing(self) -> bool:
        t = getattr(self, "_thread", None)
        return bool(t is not None and t.is_alive())
