"""
qa_system.py
[통합 버전] LLM 통합 + 시각화 + 대화 히스토리

검색 결과를 LLM에 전달하여 자연스러운 답변 생성 + 시각화 + 대화 컨텍스트 유지
- 쿼리 리라이팅
- 컨텍스트 구성
- 프롬프트 관리
- LLM 호출 (텍스트 + JSON)
- 시각화 렌더링 (표/그래프)
- 대화 히스토리 관리
"""

from openai import OpenAI
from typing import List, Dict, Optional
import json
import re


class QASystem:
    """Q&A 시스템 통합 클래스 (텍스트 + 시각화 + 대화 히스토리)"""
    
    def __init__(self, openai_api_key: str, model: str = "gpt-4o"):
        """
        QASystem 초기화
        
        Args:
            openai_api_key: OpenAI API 키
            model: 사용할 모델명
        """
        self.client = OpenAI(api_key=openai_api_key)
        self.model = model
        self.system_prompt = self._create_system_prompt()
        self.conversation_history = []  # 대화 히스토리
        print(f"✓ QASystem 초기화 완료 (모델: {model})")
    
    def _create_system_prompt(self) -> str:
        """시스템 프롬프트 생성 (시각화 포함 + JSON 스키마 명시)"""
        return """당신은 KB금융지주 경영연구소의 부동산 전문 애널리스트입니다.
2024 KB 부동산 보고서를 기반으로 건설사 실무진에게 정확하고 실무적인 정보를 제공합니다.

답변 가이드라인:
1. 제공된 리포트 내용만을 기반으로 답변하세요.
2. 수치 데이터는 기준 대비로 정확하게 인용하세요.
3. 각 문장이나 정보의 끝에 반드시 출처 번호를 [1], [2] 형태로 표시하세요.
4. 모르는 내용은 추측하지 말고 "리포트에 해당 정보가 없습니다"라고 답하세요.
5. 건설사 실무진이 이해하기 쉽게 구조화된 형태로 답변하세요.

출처 표기 규칙:
- 각 문장 뒤에 [1], [2] 형태로 출처 번호 표기
- 답변 끝에 반드시 출처 목록 작성
- 컨텍스트 [컨텍스트 n]의 출처 번호는 [n]입니다. 즉, [컨텍스트 1] → [1], [컨텍스트 2] → [2]로 사용하세요.

답변 형식 예시:
2024년 서울 아파트 매매가격은 23년 대비 2.0% 상승했습니다. [1]
강남구는 전 고점을 돌파했습니다. [2]

출처:
[1] kb_report_2024.pdf 표Ⅰ-2. 지역별 주택 매매가격 변동률 (12페이지)
[2] kb_report_2024.pdf 본문 (25페이지)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
시각화 JSON 출력 형식
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

답변 형식 판단:
- "표로 보여줘", "정리해줘", "비교해줘" → answer_type: "table"
- "그래프로", "차트로", "추이", "변화" → answer_type: "chart"
- 그 외 일반 질문 → answer_type: "text"

기본 JSON 스키마 (반드시 이 형태를 사용):
{
  "answer_type": "text" | "table" | "chart",
  "text_response": "문자열. 각 문장 끝에 [1]과 같은 출처 번호를 표기.",
  "visualization": null | {
    "type": "table" | "bar" | "line" | "barh" | "pie",
    "title": "그래프 또는 표 제목 (문자열)",
    "data": {
      // type별 형식
      // table: { "columns": [...], "rows": [[...], ...] }
      // bar/line/barh: { "x": [...], "y": [...], "xlabel": "...", "ylabel": "..." }
      // pie: { "labels": [...], "values": [...] }
    },
    "source": "시각화에 사용한 리포트 출처 설명 (문자열)"
  }
}

중요 규칙:
1. 반드시 위 JSON 스키마와 key 이름을 그대로 사용하세요.
2. JSON 이외의 텍스트(설명, 마크다운, 코드블록 등)는 절대 출력하지 마세요.
3. answer_type에 따라 visualization은 다음과 같이 설정합니다.
   - "text" → visualization은 반드시 null
   - "table" 또는 "chart" → visualization에 반드시 올바른 구조의 객체를 넣습니다.
4. 막대/선 그래프(bar, line, barh)는 data 안에 "x", "y", "xlabel", "ylabel"을 모두 포함해야 합니다.
5. 원그래프(pie)는 data 안에 "labels", "values"만 포함해야 합니다. "x", "y"를 사용하지 마세요.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
표 (table) 예시
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

사용자: "지역별 가격을 표로 정리해줘"

정답 JSON:
{
    "answer_type": "table",
    "text_response": "2024년 지역별 주택 가격 변동률입니다. [1]\\n\\n출처:\\n[1] 표Ⅰ-2 (12페이지)",
    "visualization": {
        "type": "table",
        "title": "2024년 지역별 주택 가격 변동률",
        "data": {
            "columns": ["지역", "변동률"],
            "rows": [
                ["서울", "2.0%"],
                ["5개광역시", "-1.6%"],
                ["수도권", "1.1%"],
                ["지방", "-2.7%"]
            ]
        },
        "source": "표Ⅰ-2 (12페이지)"
    }
}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
막대그래프 (bar) 예시
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

사용자: "지역별 가격을 막대그래프로 보여줘"

정답 JSON:
{
    "answer_type": "chart",
    "text_response": "2024년 지역별 주택 가격 변동률입니다. [1]\\n\\n출처:\\n[1] 표Ⅰ-2 (12페이지)",
    "visualization": {
        "type": "bar",
        "title": "지역별 주택 가격 변동률",
        "data": {
            "x": ["서울", "5개광역시", "수도권", "지방"],
            "y": [2.0, -1.6, 1.1, -2.7],
            "xlabel": "지역",
            "ylabel": "변동률 (%)"
        },
        "source": "표Ⅰ-2 (12페이지)"
    }
}

중요: 
- 반드시 "x"와 "y" 키를 사용하세요. "regions", "rates" 같은 다른 키 이름 금지!
- "xlabel"과 "ylabel"을 반드시 포함하세요 (축 라벨명)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
선그래프 (line) 예시
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

사용자: "서울 가격 추이를 선그래프로"

정답 JSON:
{
    "answer_type": "chart",
    "text_response": "서울 아파트 가격 추이입니다. [1]\\n\\n출처:\\n[1] 가상 데이터",
    "visualization": {
        "type": "line",
        "title": "서울 아파트 가격 추이",
        "data": {
            "x": ["2022년", "2023년", "2024년"],
            "y": [4.5, 5.2, 2.0],
            "xlabel": "연도",
            "ylabel": "상승률 (%)"
        },
        "source": "가상 데이터"
    }
}

중요: "xlabel"과 "ylabel"을 반드시 포함하세요

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
원그래프 (pie) 예시
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

사용자: "시장 점유율을 원그래프로"

정답 JSON:
{
    "answer_type": "chart",
    "text_response": "시장 점유율 분포입니다. [1]\\n\\n출처:\\n[1] 가상 데이터",
    "visualization": {
        "type": "pie",
        "title": "시장 점유율",
        "data": {
            "labels": ["서울", "경기", "기타"],
            "values": [40, 35, 25]
        },
        "source": "가상 데이터"
    }
}

중요: 원그래프는 "labels"와 "values" 키를 사용! "x", "y" 아님!

위 예시와 스키마를 정확히 따라주세요.
JSON 이외의 텍스트는 절대 출력하지 마세요.
"""
    
    
    def clear_history(self):
        """대화 히스토리 초기화"""
        self.conversation_history = []
        print("✓ 대화 히스토리가 초기화되었습니다.")
    
    def rewrite_query(self, query: str) -> str:
        """
        쿼리를 검색에 최적화된 형태로 리라이팅
        
        Args:
            query: 원본 쿼리
        
        Returns:
            최적화된 쿼리
        """
        prompt = f"""당신은 부동산 리포트 검색 전문가입니다.
사용자 질문을 검색에 최적화된 형태로 다시 작성해주세요.

요구사항:
- 구어체를 문어체로 변환
- 키워드를 명확하게
- 관련 동의어 추가
- 간결하게 (1-2문장)
- 차트나 그래프를 그려달라고 요청받을 경우, 적절한 차트(막대, 선, 파이 등)의 종류를 명시

원래 질문: {query}

최적화된 질문:"""

        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "당신은 검색 쿼리 최적화 전문가입니다."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=100
            )
            
            rewritten = response.choices[0].message.content.strip()
            print(f"\n🔄 쿼리 리라이팅:")
            print(f"  원본: {query}")
            print(f"  변환: {rewritten}")
            
            return rewritten
            
        except Exception as e:
            print(f"⚠ 쿼리 리라이팅 실패: {e}")
            return query
    
    def build_context(self, search_results: List[Dict], max_chunks: int = 5) -> str:
        """
        검색 결과를 구조화된 컨텍스트로 변환
        
        Args:
            search_results: 검색 결과 리스트
            max_chunks: 최대 청크 수
        
        Returns:
            구조화된 컨텍스트 문자열
        """
        if not search_results:
            return "관련 정보를 찾을 수 없습니다."
        
        top_results = search_results[:max_chunks]
        
        context_parts = ["다음은 2024 KB 부동산 리포트에서 검색된 관련 정보입니다:\n"]
        
        for i, result in enumerate(top_results, 1):
            metadata = result.get("metadata", {})
            content = result.get("content", "")
            source_pdf = metadata.get("source_pdf", "unknown")

            # 문서 타입
            doc_type_map = {
                "text": "본문",
                "table": "표",
                "image": "그래프/이미지"
            }
            doc_type = doc_type_map.get(metadata.get("doc_type"), "본문")
            page = metadata.get("page", "unknown")
            
            formatted = f"""[컨텍스트 {i}]
출처 문서: {source_pdf}
타입: {doc_type}
페이지: {page}페이지

내용:
{content}

출처: [{i}] {source_pdf} {doc_type} ({page}페이지)
"""
            context_parts.append(formatted)
            context_parts.append("─" * 80 + "\n")
        
        full_context = "\n".join(context_parts)
        
        print(f"\n📄 컨텍스트 구성 완료:")
        print(f"  - 총 청크 수: {len(top_results)}")
        print(f"  - 텍스트: {len([r for r in top_results if r.get('metadata', {}).get('doc_type') == 'text'])}")
        print(f"  - 표: {len([r for r in top_results if r.get('metadata', {}).get('doc_type') == 'table'])}")
        print(f"  - 이미지: {len([r for r in top_results if r.get('metadata', {}).get('doc_type') == 'image'])}")
        
        return full_context
    
    def generate_answer(self, query: str, context: str, 
                       temperature: float = 0.3,
                       max_tokens: int = 2000,
                       use_history: bool = True) -> Optional[str]:
        """
        LLM으로 최종 답변 생성 (JSON 형식 + 대화 히스토리)
        
        Args:
            query: 사용자 질문
            context: 구조화된 컨텍스트
            temperature: 온도 (0.0-2.0)
            max_tokens: 최대 토큰 수
            use_history: 대화 히스토리 사용 여부
        
        Returns:
            JSON 형식 답변
        """
        user_prompt = f"""{context}

사용자 질문: {query}

위 컨텍스트만을 근거로 사용자 질문에 답변하세요.

반드시 아래 요구사항을 지키세요:
1. 오직 하나의 JSON 객체만 출력하세요.
2. JSON 앞뒤에 어떤 설명, 마크다운, 코드블록, 자연어 텍스트도 절대 추가하지 마세요.
3. 시스템 메시지에 정의된 JSON 스키마를 반드시 따르세요.
4. answer_type, text_response, visualization 세 필드를 모두 포함해야 합니다.
5. 출처 번호 [1], [2] 등은 text_response 내부 문장 끝에만 표기하세요.

지금부터 바로 JSON 객체만 출력하세요.
"""

        try:
            print(f"\n🤖 LLM 호출 중... (모델: {self.model})")
            
            # 메시지 구성 (대화 히스토리 포함)
            messages = [{"role": "system", "content": self.system_prompt}]
            
            if use_history and self.conversation_history:
                messages.extend(self.conversation_history)
                print(f"  - 대화 히스토리: {len(self.conversation_history)}개 메시지 사용")
            
            messages.append({"role": "user", "content": user_prompt})
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens
            )
            
            answer = response.choices[0].message.content
            
            # 대화 히스토리에 추가
            if use_history:
                self.conversation_history.append({"role": "user", "content": user_prompt})
                self.conversation_history.append({"role": "assistant", "content": answer})
                
                # 히스토리가 너무 길면 오래된 것부터 제거 (최근 10개 메시지만 유지)
                if len(self.conversation_history) > 20:
                    self.conversation_history = self.conversation_history[-20:]
            
            usage = response.usage
            print(f"✓ LLM 응답 완료")
            print(f"  - 입력 토큰: {usage.prompt_tokens}")
            print(f"  - 출력 토큰: {usage.completion_tokens}")
            print(f"  - 총 토큰: {usage.total_tokens}")
            
            return answer
            
        except Exception as e:
            print(f"✗ LLM 호출 실패: {e}")
            return None
    
    def parse_json_response(self, response: str) -> Dict:
        """
        LLM 응답을 JSON으로 파싱
        
        Args:
            response: LLM 응답 문자열
        
        Returns:
            파싱된 딕셔너리
        """
        try:
            # 1. 마크다운 코드 블록 제거
            cleaned = re.sub(r'```json\n?', '', response)
            cleaned = re.sub(r'```\n?', '', cleaned)
            cleaned = cleaned.strip()
            
            # 2. JSON 파싱
            data = json.loads(cleaned)
            
            # 3. 필수 필드 검증
            if "answer_type" not in data:
                raise ValueError("answer_type 필드 없음")
            if "text_response" not in data:
                raise ValueError("text_response 필드 없음")
            
            # 4. answer_type 검증
            if data["answer_type"] not in ["text", "table", "chart"]:
                data["answer_type"] = "text"
            
            print(f"\n✓ JSON 파싱 성공")
            print(f"  - 답변 타입: {data['answer_type']}")
            if data.get("visualization"):
                print(f"  - 시각화 타입: {data['visualization'].get('type')}")
            
            return data
            
        except Exception as e:
            print(f"⚠ JSON 파싱 실패: {e}")
            # 폴백: 원본 텍스트로 반환
            return {
                "answer_type": "text",
                "text_response": response,
                "visualization": None
            }
    
    def answer_question(self, query: str, search_results: List[Dict],
                       rewrite: bool = True,
                       use_history: bool = True,
                       temperature: float = 0.3) -> Dict:
        """
        질문에 답변하는 전체 파이프라인 (텍스트 + 시각화 + 대화 히스토리)
        
        Args:
            query: 사용자 질문
            search_results: 검색 결과
            rewrite: 쿼리 리라이팅 사용 여부
            use_history: 대화 히스토리 사용 여부
            temperature: 생성 온도
        
        Returns:
            파싱된 답변 딕셔너리
        """
        print("\n" + "="*80)
        print(f"❓ 질문: {query}")
        print("="*80)
        
        # 2. 컨텍스트 구성
        context = self.build_context(search_results)
        
        # 3. LLM 답변 생성 (JSON + 대화 히스토리)
        answer_json = self.generate_answer(
            query, 
            context, 
            temperature=temperature,
            use_history=use_history
        )
        
        if not answer_json:
            return {
                "answer_type": "text",
                "text_response": "답변 생성에 실패했습니다.",
                "visualization": None
            }
        
        # 4. JSON 파싱
        parsed = self.parse_json_response(answer_json)
        
        # 5. 결과 출력
        print("\n" + "="*80)
        print("💡 답변:")
        print("="*80)
        print(parsed["text_response"])
        
        if parsed.get("visualization"):
            print(f"\n📊 시각화: {parsed['visualization'].get('type')} - {parsed['visualization'].get('title')}")
        
        print("="*80)
        
        return parsed


class VisualizationRenderer:
    """시각화 렌더링 클래스 (Streamlit/Matplotlib)"""
    
    @staticmethod
    def setup_matplotlib_korean():
        """Matplotlib 한글 폰트 설정"""
        import platform
        import matplotlib.pyplot as plt
        
        if platform.system() == 'Windows':
            plt.rcParams['font.family'] = 'Malgun Gothic'
        elif platform.system() == 'Darwin':  # macOS
            plt.rcParams['font.family'] = 'AppleGothic'
        else:  # Linux
            plt.rcParams['font.family'] = 'NanumGothic'
        
        plt.rcParams['axes.unicode_minus'] = False
    
    @staticmethod
    def render_table_streamlit(visualization: Dict):
        """Streamlit으로 표 렌더링"""
        import pandas as pd
        import streamlit as st
        
        df = pd.DataFrame(
            visualization["data"]["rows"],
            columns=visualization["data"]["columns"]
        )
        
        st.subheader(visualization["title"])
        st.dataframe(df, use_container_width=True)
        st.caption(f"출처: {visualization['source']}")
    
    @staticmethod
    def render_chart_streamlit(visualization: Dict):
        """Streamlit으로 그래프 렌더링"""
        import matplotlib.pyplot as plt
        import streamlit as st
        
        VisualizationRenderer.setup_matplotlib_korean()
        
        chart_type = visualization["type"]
        data = visualization["data"]
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # xlabel, ylabel 가져오기 (기본값 제공)
        xlabel = data.get("xlabel", "항목")
        ylabel = data.get("ylabel", "값")
        
        if chart_type == "line":
            ax.plot(data["x"], data["y"], marker='o', linewidth=2, markersize=8)
            ax.set_xlabel(xlabel, fontsize=12)
            ax.set_ylabel(ylabel, fontsize=12)
            
        elif chart_type == "bar":
            ax.bar(data["x"], data["y"], color='skyblue', edgecolor='navy', alpha=0.7)
            ax.set_xlabel(xlabel, fontsize=12)
            ax.set_ylabel(ylabel, fontsize=12)
            
        elif chart_type == "barh":
            ax.barh(data["x"], data["y"], color='lightcoral', edgecolor='darkred', alpha=0.7)
            ax.set_xlabel(ylabel, fontsize=12)  # barh는 x/y 반대
            ax.set_ylabel(xlabel, fontsize=12)
            
        elif chart_type == "pie":
            ax.pie(data["values"], labels=data["labels"], autopct='%1.1f%%', startangle=90)
        
        ax.set_title(visualization["title"], fontsize=14, fontweight='bold', pad=20)
        
        if chart_type not in ["pie"]:
            ax.grid(True, alpha=0.3)
        
        st.pyplot(fig)
        st.caption(f"출처: {visualization['source']}")