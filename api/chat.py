import os
import requests
from typing import List, Optional
import time
from collections import defaultdict, deque
from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field
from api.retrieval import kb

router = APIRouter()

GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODEL = os.environ.get("GROQ_MODEL", "openai/gpt-oss-120b")

SYSTEM_PROMPT = (
    "You are KrishiBee Assistant, a farming advisor for small and marginal farmers in India. "
    "Use short, plain sentences and simple bullet lists. Never use tables. "
    "Use the field details provided when they are relevant. "
    "Give practical, stage-by-stage guidance on sowing, irrigation timing, fertilizer scheduling and crop care. "
    "Never state exact pesticide or fertilizer doses from memory. Explain the general approach and tell the "
    "farmer to confirm quantities with their Soil Health Card or the local Krishi Vigyan Kendra (KVK). "
    "Do not guess the timing of crop stages or irrigations. State a number, such as days, depth or quantity, only if it appears in the reference notes provided. If you are not confident about when something should be done for this crop, say so and point the farmer to the KVK. Stay on farming topics."
)


class ChatTurn(BaseModel):
    role: str = Field(..., pattern="^(user|assistant)$")
    content: str = Field(..., max_length=1500)


class PredictionContext(BaseModel):
    crop: Optional[str] = None
    state: Optional[str] = None
    season: Optional[str] = None
    recommended_depth_cm: Optional[float] = None
    safe_range_cm: Optional[List[float]] = None
    soil_texture: Optional[str] = None
    soil_moisture_pct: Optional[float] = None


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=1000)
    history: List[ChatTurn] = []
    context: Optional[PredictionContext] = None


class SourceRef(BaseModel):
    title: str
    url: str


class ChatResponse(BaseModel):
    reply: str
    sources: List[SourceRef] = []


def describe_context(ctx: Optional[PredictionContext]) -> Optional[str]:
    if ctx is None:
        return None
    parts = []
    if ctx.crop:
        parts.append(f"crop {ctx.crop}")
    if ctx.state:
        parts.append(f"state {ctx.state}")
    if ctx.season:
        parts.append(f"season {ctx.season}")
    if ctx.recommended_depth_cm is not None:
        parts.append(f"recommended sowing depth {ctx.recommended_depth_cm} cm")
    if ctx.safe_range_cm and len(ctx.safe_range_cm) == 2:
        parts.append(f"safe depth range {ctx.safe_range_cm[0]} to {ctx.safe_range_cm[1]} cm")
    if ctx.soil_texture:
        parts.append(f"soil texture {ctx.soil_texture}")
    if ctx.soil_moisture_pct is not None:
        parts.append(f"estimated topsoil moisture {ctx.soil_moisture_pct}%")
    if not parts:
        return None
    return (
        "Field details from the farmer's latest KrishiBee prediction: "
        + ", ".join(parts)
        + ". The moisture figure is an estimate, not a measurement. The farmer may not have sown yet."
    )

def format_notes(notes: list) -> Optional[str]:
    if not notes:
        return None
    lines = [f"[{i + 1}] {n['text']} (Source: {n['source_title']})" for i, n in enumerate(notes)]
    return (
        "Reference notes from published sources. Base crop-stage timings and quantities on these notes. "
        "If they do not cover the question, say you are not sure instead of guessing:\n" + "\n".join(lines)
    )

NO_NOTES_NOTE = (
    "No reference notes were found for this question. Do not give crop-stage timings, quantities, "
    "schedules or doses. Say you do not have a sourced answer for this, share only general advice "
    "without numbers, and point the farmer to the local KVK."
)

IP_LIMIT = 10
IP_WINDOW_SECONDS = 600
DAILY_LIMIT = 80

_ip_hits = defaultdict(deque)
_daily = {"day": None, "count": 0}


def language_instruction(message: str) -> str:
    if any(ord(c) > 127 and c.isalpha() for c in message):
        return "Reply in the same language and script as the farmer's latest message."
    return "Reply in English, even if the question uses Hindi words written in English letters."


def client_ip(request: Request) -> str:
    forwarded = request.headers.get("x-forwarded-for", "")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


def check_rate_limit(request: Request):
    now = time.time()
    today = time.strftime("%Y-%m-%d", time.gmtime(now))
    if _daily["day"] != today:
        _daily["day"] = today
        _daily["count"] = 0
    if _daily["count"] >= DAILY_LIMIT:
        raise HTTPException(429, "The assistant has reached its daily limit. Please try again tomorrow.")

    hits = _ip_hits[client_ip(request)]
    while hits and now - hits[0] > IP_WINDOW_SECONDS:
        hits.popleft()
    if len(hits) >= IP_LIMIT:
        raise HTTPException(429, "You are asking too quickly. Please wait a few minutes and try again.")
    hits.append(now)
    _daily["count"] += 1

def call_groq(messages: list) -> str:
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise HTTPException(503, "Chat is not configured on the server.")

    payload = {
        "model": GROQ_MODEL,
        "messages": messages,
        "temperature": 0.3,
        "max_completion_tokens": 1000,
    }
    if GROQ_MODEL.startswith("openai/gpt-oss"):
        payload["reasoning_effort"] = "low"

    try:
        resp = requests.post(
            GROQ_URL,
            headers={"Authorization": f"Bearer {api_key}"},
            json=payload,
            timeout=25,
        )
    except requests.exceptions.RequestException:
        raise HTTPException(502, "Could not reach the chat service. Please try again.")

    if resp.status_code == 429:
        raise HTTPException(429, "The chat service is busy right now. Please try again in a minute.")
    if resp.status_code != 200:
        raise HTTPException(502, "The chat service returned an error. Please try again.")

    try:
        reply = resp.json()["choices"][0]["message"]["content"]
    except (KeyError, IndexError, ValueError):
        raise HTTPException(502, "Unexpected response from the chat service.")

    reply = (reply or "").strip()
    if not reply:
        raise HTTPException(502, "The chat service returned an empty answer. Please try again.")
    return reply


@router.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest, request: Request):
    check_rate_limit(request)

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    context_note = describe_context(req.context)
    if context_note:
        messages.append({"role": "system", "content": context_note})

    crop = req.context.crop if req.context else None
    notes = kb.search(req.message, crop=crop, top_k=3)
    notes_text = format_notes(notes)
    messages.append({"role": "system", "content": notes_text or NO_NOTES_NOTE})

    messages.extend({"role": t.role, "content": t.content} for t in req.history[-6:])
    messages.append({"role": "system", "content": language_instruction(req.message)})
    messages.append({"role": "user", "content": req.message})

    reply = call_groq(messages)

    seen = set()
    sources = []
    for n in notes:
        if n["source_url"] not in seen:
            seen.add(n["source_url"])
            sources.append(SourceRef(title=n["source_title"], url=n["source_url"]))
    return ChatResponse(reply=reply, sources=sources)