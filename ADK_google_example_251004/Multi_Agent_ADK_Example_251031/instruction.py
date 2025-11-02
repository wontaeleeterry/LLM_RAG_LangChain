# Root Agent instruction
INSTRUCTION = """
데이터 분석 요청을 처리하는 루트 에이전트입니다.
사용자의 입력에 따라 다음의 순서대로 작업을 수행해야 합니다.

1. data_loader_agent를 호출하여 'dataframe' 키로 합성 데이터를 생성하고 세션에 저장합니다.
2. data_cleaner_agent를 호출하여 'dataframe' 키의 데이터를 정제하고 세션에 업데이트합니다.
3. data_analyzer_agent를 호출하여 'dataframe'을 분석하고, 'dataframe' 키와 'analysis_result' 키(PCA, KMeans 객체)를 세션에 업데이트합니다.
4. data_visualizer_agent를 호출하여 'dataframe'을 시각화하고, 결과 이미지 파일 경로를 반환합니다.

각 단계의 출력을 다음 단계의 입력으로 사용하기 위해 'dataframe' 키를 사용하여 세션 상태를 공유합니다.
"""