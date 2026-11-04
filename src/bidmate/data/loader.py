"""원본 파일을 읽어 정리된 문서로 만든다. 담당: 데이터 파트

확장자를 보고 hwp_loader 또는 pdf_loader를 부른 뒤,
머리말·꼬리말, 쪽번호, 연속 공백을 똑같은 규칙으로 정리한다.
hwp와 pdf 결과를 같은 형식(LoadedDocument)으로 맞춰서
다음 단계(조각 자르기)가 파일 형식을 신경 쓰지 않게 한다.

받는 것: ProjectInfo (목록표 한 줄)
넘기는 것: LoadedDocument
"""
