"""웹 서버(app/server.py) 테스트. OpenAI를 부르지 않는다.

TestClient로 서버를 띄우지 않고 요청을 보낸다.
run_chatbot은 가짜로 바꿔서 "서버가 결과를 화면 약속(ChatResponse) 모양으로 바꾸는지"만 본다.
"""

from pathlib import Path

from fastapi.testclient import TestClient
from langchain_core.documents import Document

from app import server
from bidmate.schemas import AnswerResult

TEST_PASSWORD = "test-pass"
FAKE_STATE = {
    "rewritten_question": "가나다공단 이러닝 사업 예산은?",
    "chunks": [
        (
            Document(
                page_content="[가나다공단 / 이러닝] " + "예산 1억 " * 200,
                metadata={"chunk_id": "가나다공단_이러닝.hwp-0001", "project_name": "이러닝"},
            ),
            0.25,
        )
    ],
    "answer_result": AnswerResult(
        answer="1억입니다.",
        source_project_names=["이러닝"],
        has_evidence=True,
        is_rfp_question=True,
    ),
}


def test_login_required_when_password_set(monkeypatch):
    # 비밀번호가 있으면: 로그인 전 401 → 틀린 비밀번호 401 → 맞는 비밀번호로 쿠키 받기 → 통과
    monkeypatch.setattr(server, "APP_PASSWORD", TEST_PASSWORD)
    client = TestClient(server.app)

    assert client.get("/api/projects").status_code == 401
    assert client.post("/api/login", json={"password": "wrong"}).status_code == 401
    # 한글 비밀번호도 500이 아니라 401 (compare_digest에 bytes를 넣어서)
    assert client.post("/api/login", json={"password": "비밀"}).status_code == 401

    assert client.post("/api/login", json={"password": TEST_PASSWORD}).status_code == 200
    assert client.get("/api/projects").status_code == 200


def test_forged_cookie_is_rejected(monkeypatch):
    # 아무 값이나 넣은 쿠키로는 통과하지 못한다
    monkeypatch.setattr(server, "APP_PASSWORD", TEST_PASSWORD)
    client = TestClient(server.app, cookies={server.SESSION_COOKIE_NAME: "fake-token"})

    assert client.get("/api/projects").status_code == 401


def test_chat_returns_answer_in_promised_shape(monkeypatch):
    # 개발 모드(비밀번호 없음)에서 run_chatbot 결과가 ChatResponse 모양으로 나온다
    monkeypatch.setattr(server, "APP_PASSWORD", "")
    monkeypatch.setattr(server, "run_chatbot", lambda question, chat_history: FAKE_STATE)
    client = TestClient(server.app)

    response = client.post(
        "/api/chat", json={"question": "예산은?", "chat_history": [["이전 질문", "이전 답"]]}
    )

    assert response.status_code == 200
    result = response.json()
    assert result["answer"] == "1억입니다."
    assert result["source_project_names"] == ["이러닝"]
    assert result["has_evidence"] is True
    assert result["rewritten_question"] == "가나다공단 이러닝 사업 예산은?"
    assert result["chunks"][0]["chunk_id"] == "가나다공단_이러닝.hwp-0001"
    assert len(result["chunks"][0]["text"]) == server.CHUNK_PREVIEW_LENGTH  # 원문은 앞부분만
    assert result["chunks"][0]["score"] == 0.25


def test_chat_rejects_empty_question(monkeypatch):
    # 빈 질문은 run_chatbot까지 가지 않고 422 (ChatRequest의 min_length=1)
    monkeypatch.setattr(server, "APP_PASSWORD", "")
    client = TestClient(server.app)

    assert client.post("/api/chat", json={"question": ""}).status_code == 422


def test_projects_empty_without_project_list(monkeypatch):
    # 목록표 파일이 없으면(CI) 에러 대신 빈 목록
    monkeypatch.setattr(server, "APP_PASSWORD", "")
    monkeypatch.setattr(server, "PROJECT_LIST_PATH", Path("없는_폴더/data_list.csv"))
    client = TestClient(server.app)

    assert client.get("/api/projects").json() == []
