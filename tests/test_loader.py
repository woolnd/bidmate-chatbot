"""문서 읽기(loader.py) 글자 정리 테스트. 실제 파일 없이 문자열로 확인한다."""

from bidmate.data.loader import clean_text


def test_clean_text_removes_surrogates() -> None:
    """hwp 전용 기호가 남긴 반쪽 글자(surrogate)를 지워 utf-8로 바꿀 수 있게 한다."""
    cleaned = clean_text("󰂛 행정정보시스템 유지관리")
    assert cleaned == " 행정정보시스템 유지관리"
    cleaned.encode("utf-8")  # 오류가 나지 않아야 한다
