물론입니다 👍
아래는 **ADK Web + 로컬 실행 모두 호환되는 최신 수정본 README.md 전체 텍스트**입니다.
그대로 복사하여 기존 `README.md` 파일을 교체하시면 됩니다.

---

````markdown
# 다국어 콘텐츠 번역 및 검토 워크플로우 멀티 에이전트 시스템

## 개요
이 시스템은 **ADK Web 또는 로컬 Python 환경**에서 실행 가능한 멀티 에이전트 기반 번역 워크플로우입니다.  
4개의 전문 에이전트가 협력하여 문서를 로드하고, 요약하고, 번역하며, 품질을 검토합니다.

> ⚙️ ADK Web 환경에서는 `root_agent`가 프로젝트의 시작점으로 인식됩니다.  
> 로컬 환경에서는 `python main.py` 로 직접 실행할 수 있습니다.

---

## 시스템 구조

### 파일 구성
- `agent.py`: 에이전트 및 `root_agent` 정의
- `instruction.py`: 각 에이전트의 지시사항 정의
- `sub_agent.py`: 에이전트가 사용하는 도구 함수 정의
- `doc.txt`: 번역할 원본 문서 (샘플)
- `main.py`: 로컬 실행 진입점
- `__init__.py`: ADK Web용 패키지 인식 파일

---

## 에이전트 구성

### 1. Document Loader Agent
- **역할**: 파일에서 원본 문서를 읽어옵니다.
- **도구**: `load_document_tool`
- **출력**: `'original_document'` 키로 세션에 저장

### 2. Summary Expert Agent
- **역할**: 문서의 핵심 내용을 요약합니다.
- **도구**: `summarize_content_tool`
- **입력**: `'original_document'`
- **출력**: `'summary'`

### 3. Translation Expert Agent
- **역할**: 요약문을 대상 언어로 번역합니다.
- **도구**: `translate_content_tool`
- **입력**: `'summary'`
- **파라미터**: `target_language` (기본값: `"English"`)
- **출력**: `'translation'`

### 4. Quality Review Expert Agent
- **역할**: 번역 결과를 검토 및 수정합니다.
- **도구**: `review_translation_tool`
- **입력**: `'translation'`, `'summary'`
- **출력**: `'final_translation'`

---

## 실행 방법

### 1️⃣ ADK Web 환경에서 실행
ADK Web은 자동으로 `root_agent`를 엔트리포인트로 인식합니다.

```python
# 예: ADK Web 상에서 실행
root_agent.run("문서 번역 및 품질 검토 전체 프로세스를 실행합니다.")
````

> ADK Web은 `Multi_agent_claude_Code_251103/agent.py` 내의 `root_agent` 객체를 자동 인식합니다.

---

### 2️⃣ 로컬 환경에서 실행

1. 환경 설정:

```python
from agent import (
    document_loader_agent,
    summary_expert_agent,
    translation_expert_agent,
    quality_review_expert_agent
)
```

2. 워크플로우 수동 실행:

```python
# 1단계: 문서 로드
document_loader_agent.tools[0].func(None, file_path="doc.txt")

# 2단계: 요약
summary_expert_agent.tools[0].func(None)

# 3단계: 번역
translation_expert_agent.tools[0].func(None, target_language="English")

# 4단계: 품질 검토
quality_review_expert_agent.tools[0].func(None)
```

> 로컬 실행 시 `google.adk` 모듈이 없어도 동작하도록 Stub 클래스가 포함되어 있습니다.

---

## 세션 기반 데이터 공유

모든 에이전트는 세션(`ctx.session`)을 통해 데이터를 공유합니다:

| 키                   | 설명            |
| ------------------- | ------------- |
| `original_document` | 원본 문서         |
| `summary`           | 요약문           |
| `translation`       | 번역문           |
| `target_language`   | 번역 대상 언어      |
| `final_translation` | 검토 완료된 최종 번역문 |

---

## 도구 함수 요약

| 함수                        | 설명        |
| ------------------------- | --------- |
| `load_document_tool`      | 문서 파일을 로드 |
| `summarize_content_tool`  | 문서 요약     |
| `translate_content_tool`  | 요약문을 번역   |
| `review_translation_tool` | 번역 품질 검토  |

각 함수는 `success`, `message`, `summary`, `translation` 등의 결과를 반환합니다.

---

## 확장 가능성

### 추가 기능 제안

1. 다중 언어 자동 감지
2. 용어집/스타일 가이드 통합
3. LLM API 연동 (예: Gemini, GPT-4o)
4. 번역 메모리 (Translation Memory)
5. 협업 검토(Reviewer Multi-Agent)

---

## 실제 번역 API 연동 (예시)

> ⚠️ 아래 코드는 Google Cloud 자격증명이 필요한 예시 코드입니다.
> ADK Web 또는 인증 없는 환경에서는 동작하지 않습니다.

```python
from google.cloud import translate_v2 as translate

def translate_content_tool(ctx, target_language: str):
    client = translate.Client()
    summary = ctx.session.get("summary", "")
    result = client.translate(summary, target_language=target_language)
    ctx.session["translation"] = result["translatedText"]
    return {"success": True, "translation": result["translatedText"]}
```

---

## 제한사항 및 개선 방향

| 구분    | 설명                           |
| ----- | ---------------------------- |
| 현재 번역 | 단어 치환 기반 (프로토타입)             |
| 품질 검토 | 문법/용어 중심의 단순 검사              |
| 개선 방향 | LLM 기반 번역 및 맥락 유지, 용어 일관성 강화 |

---

## 라이선스

이 시스템은 **교육 및 연구 목적**으로 제공됩니다.
상업적 사용 시 API 정책 및 저작권 규정을 반드시 확인하세요.

```

---

이 버전은  
✅ ADK Web 환경에서 인식 가능한 구조  
✅ 로컬 Python 환경에서 독립 실행 가능한 Stub 호환  
✅ root_agent 역할 명시  
✅ 실제 API 보안 경고 추가  
모두 반영되어 있습니다.
```
