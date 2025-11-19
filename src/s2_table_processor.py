"""
table_processor.py
[2단계] 표 데이터 처리

표를 자연어로 변환하여 RAG 시스템에 활용
"""

import pandas as pd
from typing import Dict
import os
from openai import OpenAI
import json


class TableProcessor:
    """표 데이터를 자연어로 변환하는 클래스"""
    
    def __init__(self, cache_path: str = "data/cache/table_descriptions.json"):
        """
        TableProcessor 초기화
        
        Args:
            cache_path: 캐시 파일 경로
        """
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.cache_path = cache_path
        self.cache = self.load_cache()
        
        # 캐시 디렉토리 생성
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    
    def load_cache(self) -> Dict:
        """
        캐시 파일 로드
        
        Returns:
            캐시 딕셔너리
        """
        if os.path.exists(self.cache_path):
            try:
                with open(self.cache_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except:
                return {}
        return {}
    
    def save_cache(self):
        """캐시 파일 저장"""
        with open(self.cache_path, 'w', encoding='utf-8') as f:
            json.dump(self.cache, f, ensure_ascii=False, indent=2)
    
    def clean_table_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        표 데이터 정제 (LLM에 넣기 전 기본 정제)
        
        Args:
            df: 정제할 DataFrame
        
        Returns:
            정제된 DataFrame
        """
        df_clean = df.copy()
        
        # 1. 빈 행/열 제거
        df_clean = df_clean.dropna(how='all')
        df_clean = df_clean.dropna(axis=1, how='all')
        
        # 2. 헤더 정리
        df_clean.columns = [str(col).strip() for col in df_clean.columns]
        
        # 3. 문자열 공백 제거
        for col in df_clean.select_dtypes(include=['object']).columns:
            df_clean[col] = df_clean[col].astype(str).str.strip()
        
        return df_clean
    
    def convert_to_natural_language(self, df: pd.DataFrame, table_id: str = "", caption: str = "") -> str:
        """
        LLM을 사용하여 표를 자연어로 변환 (캐싱 적용)
        
        Args:
            df: 변환할 DataFrame
            table_id: 표 ID (캐싱용)
            caption: 표 제목 (있으면 컨텍스트로 활용)
        
        Returns:
            자연어로 변환된 표 설명
        """
        if df.empty:
            return "빈 표입니다."
        
        # 캐시 확인
        cache_key = f"{table_id}_{caption}" if caption else f"{table_id}"
        if cache_key in self.cache:
            print(f"  ✓ 캐시에서 로드: {table_id}")
            return self.cache[cache_key]
        
        # DataFrame을 문자열로 변환
        table_str = df.to_string()
        
        prompt = f"""
다음 표를 자연스러운 한국어 문장으로 변환해주세요.

{f'표 제목: {caption}' if caption else ''}

표 데이터:
{table_str}

요구사항:
1. 표의 구조를 이해하고 의미 있는 정보를 문장으로 작성
2. 병합된 셀이나 계층적 구조가 있다면 그것을 고려
3. 불필요한 정보는 제외하고 핵심만 간결하게
4. 각 항목을 명확하게 설명
5. 한국어로 자연스럽게 작성
6. 나중에 이 텍스트로 질문-답변을 할 수 있도록 충분한 정보 포함
{f'7. 표 제목({caption})의 맥락을 고려하여 설명' if caption else ''}

출력 형식:
- 문단 형태로 작성
- 각 주요 정보는 명확하게 구분
"""
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "당신은 표 데이터를 자연스러운 한국어로 변환하는 전문가입니다. 변환된 텍스트는 RAG 시스템에서 검색 및 질의응답에 활용됩니다."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=1000
            )
            
            result = response.choices[0].message.content.strip()
            
            # 캐시에 저장
            self.cache[cache_key] = result
            self.save_cache()
            
            return result
            
        except Exception as e:
            print(f"⚠ LLM 변환 실패: {e}")
            return f"표 변환에 실패했습니다. 원본 데이터:\n{table_str}"

    def process_table(self, df: pd.DataFrame, table_id: str, page_num: int, caption: str = "") -> Dict:
        """
        표 전체 처리 (정제 + 자연어 변환)
        
        Args:
            df: 처리할 DataFrame
            table_id: 표 ID
            page_num: 페이지 번호
            caption: 표 제목
        
        Returns:
            표 처리 결과 (RAG에 필요한 정보만)
        """
        # 데이터 정제
        df_clean = self.clean_table_data(df)
        
        # 자연어 변환
        natural_language = self.convert_to_natural_language(df_clean, table_id, caption)
        
        if caption:
            content = f"[{caption}]\n\n{natural_language}"
        else:
            content = natural_language

        result = {
            "table_id": table_id,
            "page_num": page_num,
            "caption": caption,  # ← caption 필드 추가
            "content": content,  # 벡터 DB에 저장할 핵심 내용 (제목 포함)
            "content_type": "table"  # 검색시 필터링용
        }
        
        return result