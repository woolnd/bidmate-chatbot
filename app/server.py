"""챗봇 웹 서버 (FastAPI). 담당: 재웅

화면(static/index.html)을 내려주고, 화면이 부르는 API로 질문·답을 주고받는다.
API 약속은 index.html 맨 위 주석과 아래 pydantic 모델이 같다. 한쪽을 바꾸면 다른 쪽도 바꾼다.

    GET  /                화면
    GET  /api/health      서버 살아 있는지 (VM 배포 확인용)
    POST /api/login       비밀번호 → 로그인 쿠키
    GET  /api/projects    사업 목록 (사이드바·추천 질문)
    POST /api/chat        질문 → 네 단계(run_chatbot) → 답변·출처·조각

비밀번호(.env의 APP_PASSWORD)가 비어 있으면 로그인 없이 열린다 (개발 모드).

실행: uv run uvicorn app.server:app --reload   → http://localhost:8000 (API 문서: /docs)
"""

import hashlib
import hmac
import time
from pathlib import Path

from fastapi import Depends, FastAPI, HTTPException, Request, Response
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from bidmate.config import APP_PASSWORD, PROJECT_LIST_PATH
from bidmate.pipeline.graph import get_projects, run_chatbot

INDEX_PATH = Path(__file__).parent / "static" / "index.html"
SESSION_COOKIE_NAME = "bidmate_session"
CHUNK_PREVIEW_LENGTH = 600  # 근거 원문 보기에 보여 줄 조각 글자 수

app = FastAPI(title="BidMate RFP 챗봇")


# ── 주고받는 모양 (index.html의 API 약속과 같다) ──


class LoginRequest(BaseModel):
    password: str


class ProjectView(BaseModel):
    """사이드바 사업 목록 한 줄."""

    project_id: str
    agency_name: str
    project_name: str


class ChatRequest(BaseModel):
    """화면이 보내는 질문. chat_history는 화면이 모아 둔 [[질문, 답변], ...]."""

    question: str = Field(min_length=1, max_length=1000)
    chat_history: list[tuple[str, str]] = []


class ChunkView(BaseModel):
    """근거 원문 보기에 보여 줄 조각 하나."""

    chunk_id: str
    project_name: str
    text: str
    score: float  # Chroma 거리. 작을수록 질문과 가깝다


class ChatResponse(BaseModel):
    """화면이 받는 답. v0.2 칸(next_questions 등)은 아직 없다."""

    answer: str
    source_project_names: list[str]
    has_evidence: bool
    is_rfp_question: bool  # false면 인사·챗봇 소개 → 화면이 근거 안내를 숨긴다
    rewritten_question: str
    chunks: list[ChunkView]
    seconds: float


# ── 로그인 ──


def make_session_token() -> str:
    """로그인 쿠키에 넣을 값. 비밀번호로 만든 지문이라 서버에 따로 저장할 필요가 없다.

    받는 것: 없음 (APP_PASSWORD를 쓴다)
    넘기는 것: 64글자 문자열. 비밀번호가 같으면 항상 같은 값
    """
    # 1. 비밀번호를 열쇠로 고정 문장에 지문을 찍는다 (비밀번호를 모르면 같은 값을 못 만든다)
    return hmac.new(APP_PASSWORD.encode(), b"bidmate_session", hashlib.sha256).hexdigest()


def require_login(request: Request) -> None:
    """로그인이 필요한 API 앞에 붙이는 문지기 (Depends로 쓴다). 통과 못 하면 401.

    받는 것: 요청 (쿠키를 본다)
    넘기는 것: 없음. 로그인 안 됐으면 HTTPException(401)을 던진다
    """
    # 1. 비밀번호가 비어 있으면 개발 모드라 그냥 통과 (로컬·CI에서 로그인 없이 쓰려고)
    if not APP_PASSWORD:
        return

    # 2. 쿠키가 로그인 때 준 토큰과 같은지 (compare_digest: 비교 시간으로 값을 못 알아내게)
    cookie = request.cookies.get(SESSION_COOKIE_NAME) or ""
    if not hmac.compare_digest(cookie, make_session_token()):
        raise HTTPException(status_code=401, detail="로그인이 필요합니다.")


@app.post("/api/login")
def login(login_request: LoginRequest, response: Response) -> dict:
    """비밀번호가 맞으면 로그인 쿠키를 준다. 틀리면 401."""
    # 1. 비밀번호 비교 (bytes로 바꿔야 한글을 쳐도 에러가 안 난다)
    if not hmac.compare_digest(login_request.password.encode(), APP_PASSWORD.encode()):
        raise HTTPException(status_code=401, detail="비밀번호가 다릅니다.")

    # 2. 로그인 쿠키 주기 (httponly: 브라우저 JS가 쿠키를 못 읽게 해서 쿠키 훔치기를 막는다)
    response.set_cookie(SESSION_COOKIE_NAME, make_session_token(), httponly=True, samesite="lax")
    return {"ok": True}


# ── 화면과 상태 확인 ──


@app.get("/")
def index() -> FileResponse:
    """화면 파일을 내려준다. 로그인은 화면 안에서 처리한다."""
    return FileResponse(INDEX_PATH)


@app.get("/api/health")
def health() -> dict:
    """서버가 살아 있는지. VM 배포 뒤 curl로 확인한다."""
    return {"status": "ok"}


# ── 챗봇 API (로그인 필요) ──


@app.get("/api/projects", dependencies=[Depends(require_login)])
def projects() -> list[ProjectView]:
    """사업 목록. 목록표 파일이 없으면(CI, 데이터 받기 전) 빈 목록."""
    # 1. 목록표가 없으면 빈 목록 (CI에는 data_list.csv가 없다)
    if not PROJECT_LIST_PATH.exists():
        return []

    # 2. 사업마다 화면에 필요한 세 칸만 담기 (예산·마감일은 사이드바에 안 쓴다)
    project_views = []
    for project in get_projects():
        project_views.append(
            ProjectView(
                project_id=project.project_id,
                agency_name=project.agency_name,
                project_name=project.project_name,
            )
        )
    return project_views


@app.post("/api/chat", dependencies=[Depends(require_login)])
def chat(chat_request: ChatRequest) -> ChatResponse:
    """질문 하나를 네 단계에 통과시켜 답을 돌려준다."""
    # 1. 네 단계 실행 (응답 시간 목표 10초라 걸린 시간을 같이 잰다)
    start_time = time.time()
    state = run_chatbot(chat_request.question, chat_request.chat_history)
    seconds = time.time() - start_time

    # 2. 조각을 화면용 모양으로 (원문은 앞부분만 잘라 응답을 가볍게)
    chunk_views = [
        ChunkView(
            chunk_id=document.metadata.get("chunk_id", ""),
            project_name=document.metadata.get("project_name", ""),
            text=document.page_content[:CHUNK_PREVIEW_LENGTH],
            score=float(score),
        )
        for document, score in state["chunks"]
    ]

    # 3. 4단계 답변 결과에서 칸을 꺼내 응답 만들기
    answer_result = state["answer_result"]
    return ChatResponse(
        answer=answer_result.answer,
        source_project_names=answer_result.source_project_names,
        has_evidence=answer_result.has_evidence,
        is_rfp_question=answer_result.is_rfp_question,
        rewritten_question=state["rewritten_question"],
        chunks=chunk_views,
        seconds=round(seconds, 2),
    )
