"""FastAPI service + minimal chat UI.  Run: uvicorn app:app --reload  then open http://127.0.0.1:8000"""
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from pydantic import BaseModel

from agent import SupportAgent
from agent.router import Session

app = FastAPI(title="Bilingual Support Agent")
agent = SupportAgent()
sessions: dict[str, Session] = {}


class ChatIn(BaseModel):
    session_id: str
    message: str


@app.post("/chat")
def chat(body: ChatIn) -> dict:
    session = sessions.setdefault(body.session_id, Session())
    return agent.handle(body.message, session)


@app.get("/health")
def health() -> dict:
    return {"ok": True, "llm_enabled": agent.llm.enabled}


@app.get("/")
def index():
    return FileResponse(Path(__file__).parent / "static/index.html")
