"""
qa_system.py
[6단계 통합] LLM 통합 - 대화 히스토리 지원 버전

검색 결과를 LLM에 전달하여 자연스러운 답변 생성
- 쿼리 리라이팅
- 컨텍스트 구성
- 프롬프트 관리
- LLM 호출 (텍스트 답변만)
- 대화 히스토리 관리 (연속적인 질의응답)
"""

from openai import OpenAI
from typing import List, Dict, Optional


class QASystem:
    """Q&A 시스템 통합 클래스 (대화 히스토리 지원)"""
    
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
        self.conversation_history = []  # 대화 히스토리 저장
        print(f"✓ QASystem 초기화 완료 (모델: {model})")
    
    def _create_system_prompt(self) -> str:
        """시스템 프롬프트 생성 (대화 전용)"""
        return """당신은 부동산 시장 전문가입니다.
    부동산 리포트를 기반으로 실무자들에게 명확하고 유용한 정보를 제공합니다.

    답변 스타일:
    1. 친근하지만 전문적인 톤을 유지하세요
    2. 불필요한 격식은 생략하고 핵심만 전달하세요
    3. 질문 의도를 파악해서 정확히 답변하세요
    4. 이전 대화를 자연스럽게 이어가세요

    답변 규칙:
    1. 리포트에 있는 내용만 답변하세요
    2. 수치는 정확하게 인용하세요
    3. 중요한 정보 뒤에는 [1], [2] 형태로 출처를 표기하세요
    4. 모르는 내용은 솔직하게 "리포트에 해당 정보가 없습니다"라고 하세요
    5. 간결하고 명확하게 답변하세요

    출처 표기:
    - 답변 끝에 간단히 출처 목록 작성

    답변 예시:
    2024년 서울 아파트 매매가격은 2.0% 올랐습니다. [1] 특히 강남구는 전고점을 넘어섰네요. [2]

    출처:
    [1] KB 리포트, 표Ⅰ-2. 지역별 주택 매매가격 변동률 (12페이지)
    [2] KB 리포트, 본문 (25페이지)

    추천 질문에 답할 때:
    - "질문 추천해드릴까요?" 같은 간단한 제안
    - 2-3개 핵심 질문만 추천
    - 페이지 번호는 필요할 때만 언급
    """

    def add_to_history(self, role: str, content: str):
        """
        대화 히스토리에 메시지 추가
        
        Args:
            role: 'user' 또는 'assistant'
            content: 메시지 내용
        """
        self.conversation_history.append({
            "role": role,
            "content": content
        })
    
    def get_conversation_history(self) -> List[Dict]:
        """대화 히스토리 반환"""
        return self.conversation_history
    
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
            
            # 기관 정보
            institution = metadata.get("institution", "unknown")
            institution_map = {
                "hd": "HD 현대 리포트",
                "kb": "KB 부동산 리포트",
                "khi": "KHI 주택금융 리포트"
            }
            source_name = institution_map.get(institution, f"{institution} 리포트")
            
            # 문서 타입
            doc_type_map = {
                "text": "본문",
                "table": "표",
                "image": "그래프/이미지"
            }
            doc_type = doc_type_map.get(metadata.get("doc_type"), "본문")
            page = metadata.get("page", "unknown")
            
            # 추가 정보 (있는 경우)
            extra_info = ""
            if metadata.get("table_id"):
                extra_info = f"\n표 ID: {metadata.get('table_id')}"
            elif metadata.get("image_path"):
                image_path = metadata.get('image_path')
                image_filename = image_path.split('\\')[-1] if '\\' in image_path else image_path.split('/')[-1]
                extra_info = f"\n이미지: {image_filename}"
            
            formatted = f"""[컨텍스트 {i}]
출처 기관: {source_name}
타입: {doc_type}
페이지: {page}페이지{extra_info}

내용:
{content}

출처: [{i}] {source_name} {doc_type} ({page}페이지)
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
        LLM으로 최종 답변 생성 (대화 히스토리 지원)
        
        Args:
            query: 사용자 질문
            context: 구조화된 컨텍스트
            temperature: 온도 (0.0-2.0)
            max_tokens: 최대 토큰 수
            use_history: 대화 히스토리 사용 여부
        
        Returns:
            텍스트 답변
        """
        user_prompt = f"""{context}

사용자 질문: {query}

위 컨텍스트를 기반으로 사용자 질문에 답변해주세요.
출처 번호 [1], [2] 등을 명시하세요."""

        try:
            print(f"\n🤖 LLM 호출 중... (모델: {self.model}, 히스토리: {use_history})")
            
            # 메시지 구성
            messages = [{"role": "system", "content": self.system_prompt}]
            
            # 대화 히스토리 추가 (use_history가 True일 때만)
            if use_history and self.conversation_history:
                # 최근 4개의 대화만 포함 (너무 길어지는 것 방지)
                recent_history = self.conversation_history[-8:]  # user + assistant 쌍 4개
                messages.extend(recent_history)
                print(f"  - 대화 히스토리 {len(recent_history)}개 메시지 포함")
            
            # 현재 질문 추가
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
                self.add_to_history("user", query)
                self.add_to_history("assistant", answer)
            
            usage = response.usage
            print(f"✓ LLM 응답 완료")
            print(f"  - 입력 토큰: {usage.prompt_tokens}")
            print(f"  - 출력 토큰: {usage.completion_tokens}")
            print(f"  - 총 토큰: {usage.total_tokens}")
            print(f"  - 현재 대화 턴 수: {len(self.conversation_history) // 2}")
            
            return answer
            
        except Exception as e:
            print(f"✗ LLM 호출 실패: {e}")
            return None
    
    def answer_question(self, query: str, search_results: List[Dict],
                       rewrite: bool = True,
                       use_history: bool = True,
                       temperature: float = 0.3) -> str:
        """
        질문에 답변하는 전체 파이프라인 (대화 히스토리 지원)
        
        Args:
            query: 사용자 질문
            search_results: 검색 결과
            rewrite: 쿼리 리라이팅 사용 여부
            use_history: 대화 히스토리 사용 여부
            temperature: 생성 온도
        
        Returns:
            텍스트 답변
        """
        print("\n" + "="*80)
        print(f"❓ 질문: {query}")
        print("="*80)
        
        # 1. 쿼리 리라이팅 (선택)
        search_query = query
        if rewrite:
            search_query = self.rewrite_query(query)
        
        # 2. 컨텍스트 구성
        context = self.build_context(search_results)
        
        # 3. LLM 답변 생성 (대화 히스토리 포함)
        answer = self.generate_answer(
            search_query, 
            context, 
            use_history=use_history,
            temperature=temperature
        )
        
        if not answer:
            return "답변 생성에 실패했습니다."
        
        # 4. 결과 출력
        print("\n" + "="*80)
        print("💡 답변:")
        print("="*80)
        print(answer)
        print("="*80)
        
        return answer