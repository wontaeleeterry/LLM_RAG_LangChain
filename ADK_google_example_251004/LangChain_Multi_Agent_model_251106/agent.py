# agent.py
from sub_agent import load_doc
from multi_agent_graph import build_workflow_graph


def run_workflow():
    # Step 1: 원문 로딩
    input = load_doc()
    print("✅ 원문 로딩 완료\n")

    # Step 2: 그래프 빌드 및 실행
    graph = build_workflow_graph()
    result = graph.invoke({"input": input})

    # Step 3: 결과 출력
    print("🧾 [최종 결과]")
    print(result["final"])


if __name__ == "__main__":
    run_workflow()
