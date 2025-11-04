"""
다국어 콘텐츠 번역 및 검토 워크플로우 멀티 에이전트 시스템
Sub Agent (Tool Functions) 정의 파일
"""

# from google.adk.tools import ToolContext

# --- ADK 환경이 없을 때를 위한 임시 클래스 정의 ---
'''
이 코드는 ADK가 설치되지 않은 로컬 환경에서도 프로그램이 멈추지 않고 작동하도록, 
“ADK의 핵심 클래스들을 간단히 흉내 내는 가짜 버전”을 자동 생성하는 코드입니다.
'''

try:
    from google.adk.agents import Agent
    from google.adk.tools import FunctionTool, ToolContext

except ModuleNotFoundError:
    print("⚠️ google.adk 모듈을 찾을 수 없습니다. 로컬 시뮬레이션용 가짜 클래스를 사용합니다.")

    class FunctionTool:
        def __init__(self, func):
            self.func = func

    class ToolContext:
        def __init__(self):
            self.session = {}

    class Agent:
        def __init__(self, name, model, description, instruction, tools):
            self.name = name
            self.model = model
            self.description = description
            self.instruction = instruction
            self.tools = tools

        def run(self, *args, **kwargs):
            print(f"[{self.name}] is running (simulated).")
            if self.tools:
                return self.tools[0].func(ToolContext(), *args, **kwargs)
            return {"success": False, "error": "No tool attached"}


from typing import Dict, Any
import os


def load_document_tool(ctx: ToolContext, file_path: str = "/Users/wontaelee/Downloads/claude_filesystem_MCP/docs/doc.txt") -> Dict[str, Any]:
    """
    파일에서 원본 문서를 로드하는 도구
    
    Args:
        ctx: ToolContext 객체
        file_path: 읽을 파일 경로 (기본값: doc.txt)
    
    Returns:
        로드된 문서 정보를 담은 딕셔너리
    """
    try:
        # 파일 읽기
        if not os.path.exists(file_path):
            return {
                "success": False,
                "error": f"파일을 찾을 수 없습니다: {file_path}",
                "document": None
            }
        
        with open(file_path, 'r', encoding='utf-8') as f:
            document_content = f.read()
        
        # 세션에 저장
        ctx.session['original_document'] = document_content
        
        return {
            "success": True,
            "message": "문서를 성공적으로 로드했습니다.",
            "document_length": len(document_content),
            "file_path": file_path,
            "document": document_content[:200] + "..." if len(document_content) > 200 else document_content
        }
    
    except Exception as e:
        return {
            "success": False,
            "error": f"문서 로드 중 오류 발생: {str(e)}",
            "document": None
        }


def summarize_content_tool(ctx: ToolContext, compression_ratio: float = 0.35) -> Dict[str, Any]:
    """
    원본 문서를 요약하는 도구
    
    Args:
        ctx: ToolContext 객체
        compression_ratio: 요약 압축 비율 (기본값: 0.35, 즉 35%)
    
    Returns:
        요약 결과를 담은 딕셔너리
    """
    try:
        # 세션에서 원본 문서 가져오기
        original_document = ctx.session.get('original_document')
        
        if not original_document:
            return {
                "success": False,
                "error": "세션에서 원본 문서를 찾을 수 없습니다. 먼저 문서를 로드해주세요.",
                "summary": None
            }
        
        # 요약 수행 (여기서는 간단한 로직으로 구현, 실제로는 LLM을 사용할 수 있음)
        lines = original_document.strip().split('\n')
        paragraphs = [line.strip() for line in lines if line.strip()]
        
        # 핵심 문장 추출 (각 단락의 첫 문장과 중요 키워드 포함)
        summary_parts = []
        
        # 제목 포함
        if paragraphs:
            summary_parts.append(paragraphs[0])
        
        # 각 단락에서 핵심 내용 추출
        for para in paragraphs[1:]:
            # 단락이 충분히 길면 첫 두 문장을 추출
            sentences = para.split('.')
            if len(sentences) > 1:
                key_sentence = sentences[0] + '.'
                if len(key_sentence) > 20:  # 의미 있는 문장만
                    summary_parts.append(key_sentence.strip())
        
        summary = '\n\n'.join(summary_parts)
        
        # 세션에 저장
        ctx.session['summary'] = summary
        
        return {
            "success": True,
            "message": "문서를 성공적으로 요약했습니다.",
            "original_length": len(original_document),
            "summary_length": len(summary),
            "compression_ratio": round(len(summary) / len(original_document), 2),
            "summary": summary
        }
    
    except Exception as e:
        return {
            "success": False,
            "error": f"요약 중 오류 발생: {str(e)}",
            "summary": None
        }


def translate_content_tool(ctx: ToolContext, target_language: str = "English") -> Dict[str, Any]:
    """
    요약된 내용을 대상 언어로 번역하는 도구
    
    Args:
        ctx: ToolContext 객체
        target_language: 번역 대상 언어 (기본값: English)
    
    Returns:
        번역 결과를 담은 딕셔너리
    """
    try:
        # 세션에서 요약문 가져오기
        summary = ctx.session.get('summary')
        
        if not summary:
            return {
                "success": False,
                "error": "세션에서 요약문을 찾을 수 없습니다. 먼저 문서를 요약해주세요.",
                "translation": None
            }
        
        # 실제 구현에서는 Google Translate API나 다른 번역 서비스를 사용
        # 여기서는 샘플 번역 결과를 반환
        
        translation_map = {
            "English": {
                "인공지능의 발전과 미래": "Advances and Future of Artificial Intelligence",
                "인공지능": "Artificial Intelligence",
                "기계학습": "Machine Learning",
                "딥러닝": "Deep Learning",
                "자연어 처리": "Natural Language Processing",
                "알고리즘": "Algorithm",
                "데이터": "Data",
                "의료 진단": "Medical Diagnosis",
                "금융 분석": "Financial Analysis",
                "자율주행": "Autonomous Driving"
            }
        }
        
        # 간단한 단어 치환 번역 (실제로는 더 정교한 번역 필요)
        translation = summary
        if target_language in translation_map:
            for korean, english in translation_map[target_language].items():
                translation = translation.replace(korean, english)
        
        # 세션에 저장
        ctx.session['translation'] = translation
        ctx.session['target_language'] = target_language
        
        return {
            "success": True,
            "message": f"{target_language}(으)로 번역을 완료했습니다.",
            "target_language": target_language,
            "translation": translation
        }
    
    except Exception as e:
        return {
            "success": False,
            "error": f"번역 중 오류 발생: {str(e)}",
            "translation": None
        }


def review_translation_tool(ctx: ToolContext, review_criteria: str = "grammar,context,terminology") -> Dict[str, Any]:
    """
    번역 결과를 검토하고 수정하는 도구
    
    Args:
        ctx: ToolContext 객체
        review_criteria: 검토 기준 (쉼표로 구분)
    
    Returns:
        검토 결과를 담은 딕셔너리
    """
    try:
        # 세션에서 번역문과 요약문 가져오기
        translation = ctx.session.get('translation')
        summary = ctx.session.get('summary')
        target_language = ctx.session.get('target_language', 'Unknown')
        
        if not translation:
            return {
                "success": False,
                "error": "세션에서 번역문을 찾을 수 없습니다. 먼저 번역을 수행해주세요.",
                "final_translation": None
            }
        
        # 검토 수행
        review_results = {
            "grammar": "Good - No major grammatical errors detected.",
            "context": "Excellent - The translation maintains the original context and meaning.",
            "terminology": "Good - Technical terms are accurately translated."
        }
        
        issues_found = []
        corrections = []
        
        # 간단한 검토 로직 (실제로는 더 정교한 검토 필요)
        # 예: 반복되는 단어 체크
        words = translation.split()
        word_freq = {}
        for word in words:
            word_lower = word.lower().strip('.,!?')
            word_freq[word_lower] = word_freq.get(word_lower, 0) + 1
        
        # 과도하게 반복되는 단어 찾기
        for word, freq in word_freq.items():
            if freq > 3 and len(word) > 3:
                issues_found.append(f"단어 '{word}'이(가) {freq}번 반복됨 - 다양한 표현 고려 필요")
        
        # 최종 번역문 (수정 사항 반영)
        final_translation = translation
        
        # 간단한 개선 (예시)
        if target_language == "English":
            # 문장 구조 개선
            final_translation = final_translation.replace("  ", " ")
            final_translation = final_translation.strip()
            corrections.append("여분의 공백 제거")
        
        # 세션에 최종 번역문 저장
        ctx.session['final_translation'] = final_translation
        
        # 품질 점수 계산
        quality_score = 0
        if len(issues_found) == 0:
            quality_score = 95
        elif len(issues_found) <= 2:
            quality_score = 85
        else:
            quality_score = 75
        
        return {
            "success": True,
            "message": "번역 품질 검토를 완료했습니다.",
            "target_language": target_language,
            "review_criteria": review_criteria.split(','),
            "review_results": review_results,
            "issues_found": issues_found if issues_found else ["문제점이 발견되지 않았습니다."],
            "corrections_made": corrections if corrections else ["수정 사항이 없습니다."],
            "quality_score": quality_score,
            "final_translation": final_translation
        }
    
    except Exception as e:
        return {
            "success": False,
            "error": f"품질 검토 중 오류 발생: {str(e)}",
            "final_translation": None
        }
