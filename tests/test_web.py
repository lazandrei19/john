import pathlib

from fastapi.testclient import TestClient

from romanian_whist.agents.checkpoint import save_checkpoint
from romanian_whist.agents.model import WhistPolicyNetwork
from romanian_whist.rules.cards import parse_card
from romanian_whist.rules.game import action_from_card
from romanian_whist.web.app import create_app
from romanian_whist.web.services import PublicSeatTracker, RecommendationService, parse_optional_int_list


def _checkpoint(tmp_path: pathlib.Path) -> pathlib.Path:
    path = tmp_path / "web-model.pt"
    save_checkpoint(path, WhistPolicyNetwork())
    return path


def test_ui_pages_load(tmp_path: pathlib.Path) -> None:
    app = create_app(default_checkpoint=str(_checkpoint(tmp_path)))
    client = TestClient(app)
    for route in ("/play", "/inspect", "/advisor"):
        response = client.get(route)
        assert response.status_code == 200
        assert "Local Game UI" in response.text


def test_root_redirect_respects_default_mode(tmp_path: pathlib.Path) -> None:
    app = create_app(default_checkpoint=str(_checkpoint(tmp_path)), default_mode="advisor")
    client = TestClient(app)
    response = client.get("/", follow_redirects=False)
    assert response.status_code == 307
    assert response.headers["location"] == "/advisor"


def test_play_session_and_human_action(tmp_path: pathlib.Path) -> None:
    checkpoint = _checkpoint(tmp_path)
    app = create_app(default_checkpoint=str(checkpoint))
    client = TestClient(app)
    response = client.post(
        "/api/sessions",
        json={
            "mode": "play",
            "players": 4,
            "seed": 0,
            "roles": ["heuristic", "human", "model", "random"],
            "checkpoint_path": str(checkpoint),
            "device": "cpu",
        },
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["human_turn"] is True
    assert payload["current_player"] == 1
    action = payload["legal_actions"][0]["action"]

    next_response = client.post(f"/api/sessions/{payload['session_id']}/action", json={"action": action})
    assert next_response.status_code == 200
    next_payload = next_response.json()
    assert next_payload["step_index"] == 1
    assert next_payload["total_steps"] >= 2


def test_inspect_session_autoplay_jump_and_export(tmp_path: pathlib.Path) -> None:
    checkpoint = _checkpoint(tmp_path)
    app = create_app(default_checkpoint=str(checkpoint))
    client = TestClient(app)
    response = client.post(
        "/api/sessions",
        json={
            "mode": "inspect",
            "players": 4,
            "seed": 0,
            "roles": ["model", "heuristic", "random", "safe"],
            "checkpoint_path": str(checkpoint),
            "device": "cpu",
        },
    )
    assert response.status_code == 200
    payload = response.json()
    session_id = payload["session_id"]

    stepped = client.post(f"/api/sessions/{session_id}/step", json={"autoplay": True, "max_steps": 3})
    assert stepped.status_code == 200
    stepped_payload = stepped.json()
    assert stepped_payload["step_index"] >= 1
    assert stepped_payload["current_recommendation"] is not None

    jumped = client.post(f"/api/sessions/{session_id}/jump", json={"step_index": 0})
    assert jumped.status_code == 200
    jumped_payload = jumped.json()
    assert jumped_payload["step_index"] == 0
    assert jumped_payload["is_live_view"] is False

    exported = client.get(f"/api/sessions/{session_id}/export")
    assert exported.status_code == 200
    replay = exported.json()
    loaded = client.post("/api/replays/load", json={"payload": replay})
    assert loaded.status_code == 200
    assert loaded.json()["total_steps"] == len(replay["snapshots"])


def test_invalid_checkpoint_is_rejected() -> None:
    app = create_app(default_checkpoint="runs/universal/best.pt")
    client = TestClient(app)
    response = client.post(
        "/api/sessions",
        json={
            "mode": "play",
            "players": 4,
            "seed": 0,
            "roles": ["human", "model", "heuristic", "random"],
            "checkpoint_path": "/does/not/exist.pt",
            "device": "cpu",
        },
    )
    assert response.status_code == 400


def test_public_tracker_follow_suit_and_recommendation(tmp_path: pathlib.Path) -> None:
    tracker = PublicSeatTracker.create(
        players=4,
        advised_seat=2,
        dealer=0,
        hand_size=2,
        hand=[parse_card("2C"), parse_card("AS")],
        trump_card=parse_card("KH"),
        scores=None,
        round_index=0,
    )
    tracker.apply_bid(1, 0)
    tracker.apply_bid(2, 0)
    tracker.apply_bid(3, 0)
    tracker.apply_bid(0, 1)
    tracker.apply_card(1, parse_card("10C"))
    legal = tracker.legal_actions()
    assert legal == [action_from_card(parse_card("2C"))]

    recommendation = RecommendationService(default_checkpoint=_checkpoint(tmp_path)).recommend(
        tracker.observe(),
        checkpoint_path=str(_checkpoint(tmp_path)),
        device="cpu",
    )
    assert recommendation.legal_actions == legal
    assert recommendation.top_actions
    assert recommendation.top_actions[0]["label"]


def test_advisor_api_flow(tmp_path: pathlib.Path) -> None:
    checkpoint = _checkpoint(tmp_path)
    app = create_app(default_checkpoint=str(checkpoint))
    client = TestClient(app)
    response = client.post(
        "/api/advisor/sessions",
        json={
            "players": 4,
            "advised_seat": 0,
            "dealer": 3,
            "hand_size": 2,
            "hand": ["AS", "2C"],
            "trump_card": "KH",
            "round_index": 0,
        },
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["current_player"] == 0
    session_id = payload["session_id"]

    recommendation = client.post(
        f"/api/advisor/sessions/{session_id}/recommend",
        params={"checkpoint_path": str(checkpoint), "device": "cpu"},
    )
    assert recommendation.status_code == 200
    recommendation_payload = recommendation.json()
    assert recommendation_payload["recommendation"]["chosen_label"]

    bid_response = client.post(
        f"/api/advisor/sessions/{session_id}/bid",
        json={"player": 0, "bid": 1},
    )
    assert bid_response.status_code == 200
    assert bid_response.json()["bids"][0] == 1


def test_parse_optional_int_list_blank_returns_empty() -> None:
    assert parse_optional_int_list("") == []
    assert parse_optional_int_list(" , ") == []


def test_public_tracker_manual_update_validates_lengths_and_is_atomic() -> None:
    tracker = PublicSeatTracker.create(
        players=4,
        advised_seat=0,
        dealer=3,
        hand_size=2,
        hand=[parse_card("AS"), parse_card("2C")],
        trump_card=parse_card("KH"),
        scores=None,
        round_index=0,
    )
    original = tracker.snapshot()

    try:
        tracker.manual_update(bids=[1, None])
    except ValueError as exc:
        assert "Bids must contain exactly 4 values." in str(exc)
    else:
        raise AssertionError("Expected invalid manual update to fail.")

    assert tracker.snapshot()["bids"] == original["bids"]


def test_public_tracker_manual_update_syncs_current_trick_cards() -> None:
    tracker = PublicSeatTracker.create(
        players=4,
        advised_seat=0,
        dealer=3,
        hand_size=2,
        hand=[parse_card("AS"), parse_card("2C")],
        trump_card=parse_card("KH"),
        scores=None,
        round_index=0,
    )
    tracker.manual_update(
        phase="play",
        current_player=1,
        current_trick=[(1, parse_card("10C"))],
    )
    snapshot = tracker.snapshot()
    assert snapshot["current_trick"][0]["card"]["label"] == "10C"
    assert snapshot["played_cards"][0]["label"] == "10C"


def test_advisor_manual_rejects_blank_length_mismatch(tmp_path: pathlib.Path) -> None:
    checkpoint = _checkpoint(tmp_path)
    app = create_app(default_checkpoint=str(checkpoint))
    client = TestClient(app)
    created = client.post(
        "/api/advisor/sessions",
        json={
            "players": 4,
            "advised_seat": 0,
            "dealer": 3,
            "hand_size": 2,
            "hand": ["AS", "2C"],
            "trump_card": "KH",
            "round_index": 0,
        },
    )
    session_id = created.json()["session_id"]
    response = client.post(
        f"/api/advisor/sessions/{session_id}/manual",
        json={"bids": ""},
    )
    assert response.status_code == 400
    assert "Bids must contain exactly 4 values." in response.json()["detail"]
