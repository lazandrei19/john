const state = {
  playSessionId: null,
  inspectSessionId: null,
  advisorSessionId: null,
};

async function api(path, options = {}) {
  const response = await fetch(path, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });
  const payload = await response.json();
  if (!response.ok) {
    throw new Error(payload.detail || "Request failed");
  }
  return payload;
}

function pretty(value) {
  return JSON.stringify(value, null, 2);
}

function renderEvents(targetId, events) {
  const target = document.getElementById(targetId);
  if (!target) return;
  target.innerHTML = "";
  for (const event of events || []) {
    const entry = document.createElement("div");
    entry.className = "log-entry";
    entry.textContent = event.message || pretty(event);
    target.appendChild(entry);
  }
}

function renderBoard(targetId, data) {
  const target = document.getElementById(targetId);
  if (!target) return;
  const hands = (data.hands || []).map((seat) => {
    const cards = seat.cards.map((card) => `<span class="card-pill ${seat.hidden ? "hidden" : ""}">${card.label}</span>`).join("");
    return `
      <section class="seat ${data.current_player === seat.seat ? "current" : ""}">
        <strong>P${seat.seat}</strong> · ${seat.role} · ${seat.count} cards
        <div class="cards">${cards}</div>
      </section>
    `;
  }).join("");
  const trick = (data.current_trick || []).map((item) => `<span class="card-pill">${item.card.label} · P${item.seat}</span>`).join("");
  target.innerHTML = `
    <div class="board-header">
      <div>Round ${data.round_index} · ${data.phase} · current P${data.current_player ?? "-"}</div>
      <div>Trump ${data.trump_card ? data.trump_card.label : "-"}</div>
      <div>Scores ${JSON.stringify(data.scores || [])}</div>
    </div>
    <section class="seat">
      <strong>Current Trick</strong>
      <div class="cards">${trick || "<span class='card-pill hidden'>No cards yet</span>"}</div>
    </section>
    <div class="seat-grid">${hands}</div>
  `;
}

function renderActions(targetId, data, handler) {
  const target = document.getElementById(targetId);
  if (!target) return;
  target.innerHTML = "";
  for (const action of data.legal_actions || []) {
    const button = document.createElement("button");
    button.textContent = action.label;
    button.onclick = () => handler(action.action);
    target.appendChild(button);
  }
}

function setText(id, value) {
  const target = document.getElementById(id);
  if (target) target.textContent = value;
}

function setPre(id, value) {
  const target = document.getElementById(id);
  if (target) target.textContent = typeof value === "string" ? value : pretty(value);
}

function refreshPlay(data) {
  setText("play-status", `Session ${data.session_id} · step ${data.step_index + 1}/${data.total_steps} · ${data.current_role || "finished"}`);
  renderBoard("play-board", data);
  renderEvents("play-events", data.event_log);
  setPre("play-recommendation", data.current_recommendation || "None");
  renderActions("play-actions", data, async (action) => {
    const next = await api(`/api/sessions/${state.playSessionId}/action`, {
      method: "POST",
      body: JSON.stringify({ action }),
    });
    refreshPlay(next);
  });
}

function refreshInspect(data) {
  setText("inspect-status", `Session ${data.session_id} · step ${data.step_index + 1}/${data.total_steps} · ${data.current_role || "finished"}`);
  renderBoard("inspect-board", data);
  renderEvents("inspect-events", data.event_log);
  setPre("inspect-recommendation", data.current_recommendation || "None");
  setPre("inspect-observation", data.observations ? data.observations[String(data.current_player)] : "None");
  const jumpInput = document.getElementById("replay-step");
  if (jumpInput) {
    jumpInput.max = Math.max(0, data.total_steps - 1);
    jumpInput.value = data.step_index;
  }
}

function refreshAdvisor(data) {
  setText("advisor-status", `Advisor session ${data.session_id} · phase ${data.phase} · current P${data.current_player}`);
  setPre("advisor-state", data);
  setPre("advisor-recommendation", data.recommendation || "None");
  renderEvents("advisor-events", (data.event_log || []).map((item) => ({ message: pretty(item) })));
  const manual = data.manual_state || {};
  if (document.getElementById("manual-hand")) document.getElementById("manual-hand").value = manual.hand || "";
  if (document.getElementById("manual-bids")) document.getElementById("manual-bids").value = manual.bids || "";
  if (document.getElementById("manual-tricks")) document.getElementById("manual-tricks").value = manual.tricks_won || "";
  if (document.getElementById("manual-scores")) document.getElementById("manual-scores").value = manual.scores || "";
  if (document.getElementById("manual-trick")) document.getElementById("manual-trick").value = manual.current_trick || "";
}

async function createPlay() {
  const payload = {
    mode: "play",
    players: Number(document.getElementById("players").value),
    seed: Number(document.getElementById("seed").value),
    checkpoint_path: document.getElementById("checkpoint").value || null,
    device: document.getElementById("device").value,
    roles: document.getElementById("seat-config").value.split(",").map((item) => item.trim()),
  };
  const data = await api("/api/sessions", { method: "POST", body: JSON.stringify(payload) });
  state.playSessionId = data.session_id;
  refreshPlay(data);
}

async function createInspect() {
  const payload = {
    mode: "inspect",
    players: Number(document.getElementById("players").value),
    seed: Number(document.getElementById("seed").value),
    checkpoint_path: document.getElementById("checkpoint").value || null,
    device: document.getElementById("device").value,
    roles: document.getElementById("seat-config").value.split(",").map((item) => item.trim()),
  };
  const data = await api("/api/sessions", { method: "POST", body: JSON.stringify(payload) });
  state.inspectSessionId = data.session_id;
  refreshInspect(data);
}

async function createAdvisor() {
  const payload = {
    players: Number(document.getElementById("advisor-players").value),
    advised_seat: Number(document.getElementById("advisor-seat").value),
    dealer: Number(document.getElementById("advisor-dealer").value),
    hand_size: Number(document.getElementById("advisor-hand-size").value),
    round_index: Number(document.getElementById("advisor-round-index").value),
    hand: document.getElementById("advisor-hand").value.split(",").map((item) => item.trim()).filter(Boolean),
    trump_card: document.getElementById("advisor-trump").value || null,
  };
  const data = await api("/api/advisor/sessions", { method: "POST", body: JSON.stringify(payload) });
  state.advisorSessionId = data.session_id;
  refreshAdvisor(data);
}

document.addEventListener("DOMContentLoaded", () => {
  const mode = document.body.dataset.mode;
  if (mode === "play") {
    document.getElementById("create-play").onclick = () => createPlay().catch((error) => alert(error.message));
    document.getElementById("step-play").onclick = async () => {
      const data = await api(`/api/sessions/${state.playSessionId}/step`, { method: "POST", body: JSON.stringify({ autoplay: false }) });
      refreshPlay(data);
    };
    document.getElementById("autoplay-play").onclick = async () => {
      const data = await api(`/api/sessions/${state.playSessionId}/step`, { method: "POST", body: JSON.stringify({ autoplay: true, max_steps: 64 }) });
      refreshPlay(data);
    };
    document.getElementById("recommend-play").onclick = async () => {
      const data = await api(`/api/sessions/${state.playSessionId}/recommend`, { method: "POST", body: JSON.stringify({}) });
      setPre("play-recommendation", data);
    };
  }

  if (mode === "inspect") {
    document.getElementById("create-inspect").onclick = () => createInspect().catch((error) => alert(error.message));
    document.getElementById("step-inspect").onclick = async () => {
      const data = await api(`/api/sessions/${state.inspectSessionId}/step`, { method: "POST", body: JSON.stringify({ autoplay: false }) });
      refreshInspect(data);
    };
    document.getElementById("autoplay-inspect").onclick = async () => {
      const data = await api(`/api/sessions/${state.inspectSessionId}/step`, { method: "POST", body: JSON.stringify({ autoplay: true, max_steps: 128 }) });
      refreshInspect(data);
    };
    document.getElementById("jump-inspect").onclick = async () => {
      const stepIndex = Number(document.getElementById("replay-step").value);
      const data = await api(`/api/sessions/${state.inspectSessionId}/jump`, { method: "POST", body: JSON.stringify({ step_index: stepIndex }) });
      refreshInspect(data);
    };
    document.getElementById("export-inspect").onclick = async () => {
      const data = await api(`/api/sessions/${state.inspectSessionId}/export`);
      document.getElementById("replay-json").value = pretty(data);
    };
    document.getElementById("load-replay").onclick = async () => {
      const payload = JSON.parse(document.getElementById("replay-json").value);
      const data = await api("/api/replays/load", { method: "POST", body: JSON.stringify({ payload }) });
      state.inspectSessionId = data.session_id;
      refreshInspect(data);
    };
  }

  if (mode === "advisor") {
    document.getElementById("create-advisor").onclick = () => createAdvisor().catch((error) => alert(error.message));
    document.getElementById("advisor-submit-bid").onclick = async () => {
      const data = await api(`/api/advisor/sessions/${state.advisorSessionId}/bid`, {
        method: "POST",
        body: JSON.stringify({
          player: Number(document.getElementById("advisor-bid-player").value),
          bid: Number(document.getElementById("advisor-bid-value").value),
        }),
      });
      refreshAdvisor(data);
    };
    document.getElementById("advisor-submit-card").onclick = async () => {
      const data = await api(`/api/advisor/sessions/${state.advisorSessionId}/card`, {
        method: "POST",
        body: JSON.stringify({
          player: Number(document.getElementById("advisor-card-player").value),
          card: document.getElementById("advisor-card-value").value,
        }),
      });
      refreshAdvisor(data);
    };
    document.getElementById("advisor-recommend").onclick = async () => {
      const params = new URLSearchParams({
        checkpoint_path: document.getElementById("advisor-checkpoint").value,
        device: document.getElementById("advisor-device").value,
      });
      const data = await api(`/api/advisor/sessions/${state.advisorSessionId}/recommend?${params.toString()}`, { method: "POST", body: JSON.stringify({}) });
      refreshAdvisor(data);
    };
    document.getElementById("advisor-manual").onclick = async () => {
      const data = await api(`/api/advisor/sessions/${state.advisorSessionId}/manual`, {
        method: "POST",
        body: JSON.stringify({
          hand: document.getElementById("manual-hand").value,
          bids: document.getElementById("manual-bids").value,
          tricks_won: document.getElementById("manual-tricks").value,
          scores: document.getElementById("manual-scores").value,
          current_trick: document.getElementById("manual-trick").value,
          current_player: document.getElementById("manual-player").value ? Number(document.getElementById("manual-player").value) : null,
          phase: document.getElementById("manual-phase").value || null,
          leader: document.getElementById("manual-leader").value ? Number(document.getElementById("manual-leader").value) : null,
          trump_card: document.getElementById("manual-trump").value || null,
        }),
      });
      refreshAdvisor(data);
    };
  }
});
