"""프롬프트와 메시지 순서. 담당: 재웅

프롬프트 캐싱 규칙:
    고정 부분(시스템 프롬프트 → 답변 예시 → 출력 형식)을 앞에,
    바뀌는 부분(대화 기록 → 찾은 조각 → 질문)을 뒤에 둔다.
    앞부분이 한 글자라도 달라지면 캐시가 깨지므로, 고정 부분에 날짜·이름 같은 값을 넣지 않는다.

답변 품질 원칙:
    - 시스템 프롬프트에 "RFP 분석 도우미"라는 역할을 준다.
    - 짧게 시작하고, 평가에서 틀린 경우가 나올 때만 규칙을 하나씩 더한다.
    - 긴 규칙 대신 좋은 답변 예시 2개를 넣는다.
    - 대화 기록은 최근 3번의 질문·답변만 넣는다.
    - AI 채점용 프롬프트는 답변용과 따로 둔다.
"""

from langchain_core.documents import Document
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage

from bidmate.config import REWRITE_ANSWER_PREVIEW_LENGTH
from bidmate.schemas import ChatHistory

# 1단계 질문 다듬기 지시문 (고정 부분). 날짜·이름 같은 바뀌는 값을 넣지 않는다.
# 예시는 실제 RFP 이름이 아니라 지어낸 기관명으로 쓴다 (이 파일은 Git에 올라간다).
# 짧게 시작하고, 평가에서 틀린 경우가 나올 때만 규칙을 하나씩 더한다.
REWRITE_SYSTEM_PROMPT = """너는 공공기관 RFP 문서 검색을 돕는 질문 다듬기 도우미다.
대화 기록과 새 질문을 보고, 새 질문을 혼자 읽어도 뜻이 통하는 질문으로 바꾼다.

규칙:
- 새 질문이 앞 대화를 봐야 뜻이 통하면, 앞 대화에 나온 기관명·사업명을 넣어 바꾼다.
- 새 질문이 이미 혼자 뜻이 통하거나 다른 사업으로 주제가 바뀌었으면 그대로 둔다.
- 질문에 답하지 않는다. 대화 기록에 없는 기관명·사업명을 지어내지 않는다.
- 다듬은 질문 한 줄만 쓴다. 설명이나 따옴표를 붙이지 않는다.

예시 1)
대화 기록:
질문: 가나다공단 이러닝 사업 요구사항 알려줘
답변: 가나다공단 통합 이러닝 플랫폼 구축 사업의 요구사항은 강의 수강, 진도 관리 …

새 질문: 그럼 예산은?
출력: 가나다공단 통합 이러닝 플랫폼 구축 사업 예산은?

예시 2)
대화 기록:
질문: 가나다공단 이러닝 사업 알려줘
답변: 가나다공단 통합 이러닝 플랫폼 구축 사업은 직원 교육을 하나로 통합하는 사업으로 …

새 질문: 라마대학교 학사 시스템 사업 마감일은?
출력: 라마대학교 학사 시스템 사업 마감일은?"""


def build_rewrite_messages(question: str, recent_history: ChatHistory) -> list[BaseMessage]:
    """1단계 다듬기용 메시지를 만든다.

    받는 것: 새 질문, 최근 대화 기록 [(질문, 답변), ...] (이미 최근 3번으로 잘린 것)
    넘기는 것: [시스템 메시지(고정), 사람 메시지(대화 기록 + 새 질문)]
    고정 부분을 앞에, 바뀌는 부분을 뒤에 둔다 (프롬프트 캐싱 규칙).
    """

    # 1. 대화 기록을 "질문: …" / "답변: …" 줄로 바꾼다
    #    답변은 앞 REWRITE_ANSWER_PREVIEW_LENGTH자(200자)만 넣는다 (기관명·사업명만 있으면 된다)
    history_lines: list[str] = []
    for previous_question, previous_answer in recent_history:
        answer_preview = previous_answer[:REWRITE_ANSWER_PREVIEW_LENGTH]
        history_lines.append(f"질문: {previous_question}")
        history_lines.append(f"답변: {answer_preview}")

    # 2. 줄들을 한 덩어리 글로 잇는다
    history_text = "\n".join(history_lines)

    # 3. 사람 메시지 글 만들기: 시스템 프롬프트 예시와 같은 모양 ("대화 기록:" → "새 질문:")
    human_message_text = "대화 기록:\n" + history_text + "\n\n새 질문: " + question

    # 4. 고정 부분(시스템 메시지)을 앞에, 바뀌는 부분(사람 메시지)을 뒤에 둔다 (프롬프트 캐싱)
    messages: list[BaseMessage] = [
        SystemMessage(REWRITE_SYSTEM_PROMPT),
        HumanMessage(human_message_text),
    ]

    return messages


# 4단계 답변 지시문 (고정 부분). 날짜·이름 같은 바뀌는 값을 넣지 않는다.
# 예시는 지어낸 기관명으로 쓴다. 짧게 시작하고, 평가에서 틀린 경우가 나올 때만 규칙을 하나씩 더한다.
ANSWER_SYSTEM_PROMPT = """너는 입찰 컨설턴트를 돕는 RFP 분석 도우미다.
찾은 조각을 읽고 질문에 답한다.

규칙:
- 찾은 조각에 있는 내용만 쓴다. 조각에 없으면 지어내지 말고 "찾은 문서에 없습니다"라고 쓴다.
- 질문에 필요한 내용만 짧게 쓴다. 금액·날짜·숫자는 조각에 적힌 그대로 쓴다.
- 출처에는 답의 근거가 된 조각의 사업명을 쓴다. 근거가 없으면 출처를 비우고 근거 없음으로 한다.
- 대화 기록은 질문의 뜻을 이해할 때만 쓰고, 답의 근거로 쓰지 않는다.
- 질문이 RFP가 아니라 너 자신이나 인사에 관한 것이면, 공공기관 RFP 100개를 검색해
  출처와 함께 답하는 입찰메이트 RFP 분석 도우미라고 짧게 소개하고 물어볼 수 있는 질문 예시
  (사업 예산, 마감일, 요구사항, 제출 방법)를 든다. 이때 출처는 비우고 근거 없음으로 한다.

예시 1)
찾은 조각:
[1] 사업명: 통합 이러닝 플랫폼 구축
[가나다공단 / 통합 이러닝 플랫폼 구축] 사업 예산: 500,000,000원(부가세 포함)

질문: 가나다공단 이러닝 사업 예산은?
답변: 사업 예산은 500,000,000원(부가세 포함)입니다.
출처: 통합 이러닝 플랫폼 구축
근거 있음: 예

예시 2)
찾은 조각:
[1] 사업명: 학사 시스템 고도화
[라마대학교 / 학사 시스템 고도화] 수강신청, 성적 관리, 졸업 심사 기능을 개선한다.

질문: 라마대학교 학사 시스템 사업 제출 서류는?
답변: 찾은 문서에 없습니다.
출처: (없음)
근거 있음: 아니오"""


def build_answer_messages(
    question: str, chunks: list[tuple[Document, float]], recent_history: ChatHistory
) -> list[BaseMessage]:
    """4단계 답변용 메시지를 만든다.

    받는 것: 다듬은 질문, 찾은 조각 [(Document, 점수), ...],
             최근 대화 기록 (이미 최근 3번으로 잘린 것)
    넘기는 것: [시스템 메시지(고정), 사람 메시지(대화 기록 → 찾은 조각 → 질문)]
    고정 부분을 앞에, 바뀌는 부분을 뒤에 둔다 (프롬프트 캐싱 규칙).
    """

    # 1. 대화 기록을 "질문: …" / "답변: …" 줄로 바꾼다. 기록이 없으면 "(없음)" 한 줄
    #    답변은 앞 200자만 넣는다 (질문의 뜻을 이해할 때만 쓰고 답의 근거로는 쓰지 않으니까)
    history_lines: list[str] = []

    if len(recent_history) == 0:
        history_lines.append("(없음)")
    else:
        for previous_question, previous_answer in recent_history:
            answer_preview = previous_answer[:REWRITE_ANSWER_PREVIEW_LENGTH]
            history_lines.append(f"질문: {previous_question}")
            history_lines.append(f"답변: {answer_preview}")

    history_text = "\n".join(history_lines)

    # 2. 조각을 "[번호] 사업명: …" + 본문 모양으로 바꾼다 (시스템 프롬프트 예시와 같은 모양)
    #    사업명이 보여야 AI가 출처(source_project_names)를 고를 수 있다. 점수는 넘기지 않는다
    chunk_lines: list[str] = []

    for number, (document, score) in enumerate(chunks, start=1):
        chunk_lines.append(f"[{number}] 사업명: {document.metadata['project_name']}")
        chunk_lines.append(document.page_content)
        chunk_lines.append("")  # 조각 사이 빈 줄

    # 마지막 조각 뒤에 붙은 빈 줄은 지운다
    chunk_text = "\n".join(chunk_lines).strip()

    # 3. 사람 메시지 글 만들기: 대화 기록 → 찾은 조각 → 질문 순서
    human_message_text = (
        "대화 기록:\n" + history_text + "\n\n찾은 조각:\n" + chunk_text + "\n\n질문: " + question
    )

    # 4. 고정 부분(시스템 메시지)을 앞에, 바뀌는 부분(사람 메시지)을 뒤에 둔다 (프롬프트 캐싱)
    return [SystemMessage(ANSWER_SYSTEM_PROMPT), HumanMessage(human_message_text)]
