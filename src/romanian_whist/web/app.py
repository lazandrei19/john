from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field

from romanian_whist.rules.cards import card_label
from romanian_whist.web.services import (
    RecommendationService,
    SessionManager,
    parse_card,
    parse_card_list,
    parse_int_list,
    parse_optional_int_list,
    parse_trick,
)


class SessionCreateRequest(BaseModel):
    mode: str = Field(pattern="^(play|inspect)$")
    players: int = Field(ge=3, le=6)
    seed: int = 0
    roles: list[str]
    checkpoint_path: Optional[str] = None
    device: str = "cpu"


class ActionRequest(BaseModel):
    action: int


class StepRequest(BaseModel):
    autoplay: bool = False
    max_steps: int = 64


class JumpRequest(BaseModel):
    step_index: int


class LoadReplayRequest(BaseModel):
    payload: dict[str, Any]


class AdvisorCreateRequest(BaseModel):
    players: int = Field(ge=3, le=6)
    advised_seat: int
    dealer: int
    hand_size: int = Field(ge=1, le=8)
    hand: list[str | int]
    trump_card: Optional[str | int] = None
    scores: Optional[list[int]] = None
    round_index: int = 0


class AdvisorBidRequest(BaseModel):
    player: int
    bid: int


class AdvisorCardRequest(BaseModel):
    player: int
    card: str | int


class AdvisorManualRequest(BaseModel):
    hand: Optional[str] = None
    bids: Optional[str] = None
    tricks_won: Optional[str] = None
    scores: Optional[str] = None
    current_trick: Optional[str] = None
    current_player: Optional[int] = None
    phase: Optional[str] = None
    leader: Optional[int] = None
    trump_card: Optional[str] = None


class AdvisorRoundRequest(BaseModel):
    dealer: int
    hand_size: int = Field(ge=1, le=8)
    hand: list[str | int]
    trump_card: Optional[str | int] = None
    round_index: Optional[int] = None


def create_app(*, default_checkpoint: Optional[str] = None, default_mode: str = "play", default_players: int = 4, default_device: str = "cpu") -> FastAPI:
    package_dir = Path(__file__).resolve().parent
    templates = Jinja2Templates(directory=str(package_dir / "templates"))
    recommender = RecommendationService(default_checkpoint=Path(default_checkpoint) if default_checkpoint else None)
    manager = SessionManager(recommender)

    app = FastAPI(title="Romanian Whist UI")
    app.state.templates = templates
    app.state.manager = manager
    app.state.default_checkpoint = default_checkpoint or str(recommender.default_checkpoint)
    app.state.default_mode = default_mode
    app.state.default_players = default_players
    app.state.default_device = default_device
    app.mount("/static", StaticFiles(directory=str(package_dir / "static")), name="static")

    @app.get("/", response_class=HTMLResponse)
    def root() -> RedirectResponse:
        return RedirectResponse(url="/{mode}".format(mode=app.state.default_mode))

    @app.get("/play", response_class=HTMLResponse)
    def play_page(request: Request) -> HTMLResponse:
        return templates.TemplateResponse(
            request,
            "play.html",
            {
                "default_checkpoint": app.state.default_checkpoint,
                "default_players": app.state.default_players,
                "default_device": app.state.default_device,
                "default_roles": "human,model,heuristic,heuristic",
                "mode": "play",
            },
        )

    @app.get("/inspect", response_class=HTMLResponse)
    def inspect_page(request: Request) -> HTMLResponse:
        return templates.TemplateResponse(
            request,
            "inspect.html",
            {
                "default_checkpoint": app.state.default_checkpoint,
                "default_players": app.state.default_players,
                "default_device": app.state.default_device,
                "default_roles": "model,heuristic,random,safe",
                "mode": "inspect",
            },
        )

    @app.get("/advisor", response_class=HTMLResponse)
    def advisor_page(request: Request) -> HTMLResponse:
        cards = [card_label(card_id) for card_id in range(52)]
        return templates.TemplateResponse(
            request,
            "advisor.html",
            {
                "default_checkpoint": app.state.default_checkpoint,
                "default_players": app.state.default_players,
                "default_device": app.state.default_device,
                "cards_json": json.dumps(cards),
                "mode": "advisor",
            },
        )

    @app.get("/api/meta")
    def meta() -> dict[str, Any]:
        selection = recommender.resolve_checkpoint(None, device=app.state.default_device)
        return {
            "default_checkpoint": selection.resolved_path,
            "default_checkpoint_exists": selection.exists,
            "default_players": app.state.default_players,
            "cards": [{"id": card_id, "label": card_label(card_id)} for card_id in range(52)],
        }

    @app.post("/api/sessions")
    def create_session(payload: SessionCreateRequest) -> dict[str, Any]:
        try:
            session = manager.create_full_session(
                mode=payload.mode,
                players=payload.players,
                seed=payload.seed,
                roles=payload.roles,
                checkpoint_path=payload.checkpoint_path,
                device=payload.device,
            )
            return session.current_state(reveal_all=payload.mode == "inspect", include_recommendation=payload.mode == "inspect")
        except (ValueError, FileNotFoundError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.get("/api/sessions/{session_id}")
    def get_session(session_id: str, reveal_all: bool = False, include_recommendation: bool = False) -> dict[str, Any]:
        try:
            session = manager.get(session_id)
            return session.current_state(reveal_all=reveal_all, include_recommendation=include_recommendation)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="Session not found") from exc

    @app.post("/api/sessions/{session_id}/action")
    def submit_action(session_id: str, payload: ActionRequest) -> dict[str, Any]:
        try:
            session = manager.get(session_id)
            return session.submit_human_action(payload.action)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="Session not found") from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/api/sessions/{session_id}/step")
    def step_session(session_id: str, payload: StepRequest) -> dict[str, Any]:
        try:
            session = manager.get(session_id)
            if payload.autoplay:
                return session.autoplay(max_steps=payload.max_steps)
            return session.step_once()
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="Session not found") from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/api/sessions/{session_id}/recommend")
    def recommend_session(session_id: str) -> dict[str, Any]:
        try:
            session = manager.get(session_id)
            return session.recommend()
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="Session not found") from exc
        except (ValueError, FileNotFoundError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/api/sessions/{session_id}/jump")
    def jump_session(session_id: str, payload: JumpRequest) -> dict[str, Any]:
        try:
            session = manager.get(session_id)
            return session.jump(payload.step_index)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="Session not found") from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.get("/api/sessions/{session_id}/export")
    def export_session(session_id: str) -> dict[str, Any]:
        try:
            session = manager.get(session_id)
            return session.export_state()
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="Session not found") from exc

    @app.post("/api/replays/load")
    def load_replay(payload: LoadReplayRequest) -> dict[str, Any]:
        try:
            session = manager.load_replay_session(payload.payload)
            return session.current_state()
        except (KeyError, ValueError) as exc:
            raise HTTPException(status_code=400, detail="Invalid replay payload: {error}".format(error=exc)) from exc

    @app.post("/api/advisor/sessions")
    def create_advisor(payload: AdvisorCreateRequest) -> dict[str, Any]:
        try:
            session_id = manager.create_advisor_session(
                players=payload.players,
                advised_seat=payload.advised_seat,
                dealer=payload.dealer,
                hand_size=payload.hand_size,
                hand=[parse_card(item) for item in payload.hand],
                trump_card=parse_card(payload.trump_card) if payload.trump_card is not None else None,
                scores=payload.scores,
                round_index=payload.round_index,
            )
            tracker = manager.get(session_id)
            response = tracker.snapshot()
            response["session_id"] = session_id
            return response
        except (ValueError, FileNotFoundError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.get("/api/advisor/sessions/{session_id}")
    def advisor_state(session_id: str) -> dict[str, Any]:
        try:
            tracker = manager.get(session_id)
            response = tracker.snapshot()
            response["session_id"] = session_id
            return response
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="Advisor session not found") from exc

    @app.post("/api/advisor/sessions/{session_id}/bid")
    def advisor_bid(session_id: str, payload: AdvisorBidRequest) -> dict[str, Any]:
        try:
            tracker = manager.get(session_id)
            tracker.apply_bid(payload.player, payload.bid)
            response = tracker.snapshot()
            response["session_id"] = session_id
            return response
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="Advisor session not found") from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/api/advisor/sessions/{session_id}/card")
    def advisor_card(session_id: str, payload: AdvisorCardRequest) -> dict[str, Any]:
        try:
            tracker = manager.get(session_id)
            tracker.apply_card(payload.player, parse_card(payload.card))
            response = tracker.snapshot()
            response["session_id"] = session_id
            return response
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="Advisor session not found") from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/api/advisor/sessions/{session_id}/recommend")
    def advisor_recommend(session_id: str, checkpoint_path: Optional[str] = None, device: str = "cpu") -> dict[str, Any]:
        try:
            tracker = manager.get(session_id)
            recommendation = recommender.recommend(tracker.observe(), checkpoint_path=checkpoint_path, device=device)
            response = tracker.snapshot()
            response["session_id"] = session_id
            response["recommendation"] = recommendation.__dict__
            return response
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="Advisor session not found") from exc
        except (ValueError, FileNotFoundError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/api/advisor/sessions/{session_id}/manual")
    def advisor_manual(session_id: str, payload: AdvisorManualRequest) -> dict[str, Any]:
        try:
            tracker = manager.get(session_id)
            tracker.manual_update(
                hand=parse_card_list(payload.hand) if payload.hand is not None else None,
                bids=parse_optional_int_list(payload.bids) if payload.bids is not None else None,
                tricks_won=parse_int_list(payload.tricks_won) if payload.tricks_won is not None else None,
                scores=parse_int_list(payload.scores) if payload.scores is not None else None,
                current_trick=parse_trick(payload.current_trick) if payload.current_trick is not None else None,
                current_player=payload.current_player,
                phase=payload.phase,
                leader=payload.leader,
                trump_card="__unset__" if payload.trump_card is None else (parse_card(payload.trump_card) if payload.trump_card != "" else None),
            )
            response = tracker.snapshot()
            response["session_id"] = session_id
            return response
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="Advisor session not found") from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/api/advisor/sessions/{session_id}/next-round")
    def advisor_next_round(session_id: str, payload: AdvisorRoundRequest) -> dict[str, Any]:
        try:
            tracker = manager.get(session_id)
            tracker.start_next_round(
                dealer=payload.dealer,
                hand_size=payload.hand_size,
                hand=[parse_card(item) for item in payload.hand],
                trump_card=parse_card(payload.trump_card) if payload.trump_card is not None else None,
                round_index=payload.round_index,
            )
            response = tracker.snapshot()
            response["session_id"] = session_id
            return response
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="Advisor session not found") from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    return app


def run_ui(*, checkpoint: Optional[str], host: str, port: int, mode: str, players: int, device: str) -> None:
    app = create_app(
        default_checkpoint=checkpoint,
        default_mode=mode,
        default_players=players,
        default_device=device,
    )
    uvicorn.run(app, host=host, port=port)
