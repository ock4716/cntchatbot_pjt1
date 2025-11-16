"""
search_engine.py
[5단계] 검색 엔진

하이브리드 검색(벡터 + BM25)을 수행하는 모듈입니다.
- 벡터 검색 (FAISS)
- 키워드 검색 (BM25)
- 하이브리드 검색 (Reciprocal Rank Fusion)
"""

import numpy as np
import faiss
from typing import List, Dict
from rank_bm25 import BM25Okapi
import re


class SearchEngine:
    """하이브리드 검색 엔진 클래스 (벡터 + 키워드 + RRF)"""
    
    def __init__(self, 
                 faiss_index: faiss.Index,
                 metadata: List[Dict],
                 chunks: List[Dict],
                 embedding_manager=None):
        """
        SearchEngine 초기화
        
        Args:
            faiss_index: FAISS 인덱스
            metadata: 메타데이터 리스트
            chunks: 청크 리스트 (BM25용)
            embedding_manager: EmbeddingManager 인스턴스 (쿼리 임베딩용)
        """
        self.faiss_index = faiss_index
        self.metadata = metadata
        self.chunks = chunks
        self.embedding_manager = embedding_manager
        
        # BM25 인덱스 생성
        print("🔧 BM25 인덱스 생성 중...")
        self._build_bm25_index()
        
        print("✓ SearchEngine 초기화 완료")
        print(f"  - FAISS 벡터 수: {faiss_index.ntotal}")
        print(f"  - BM25 문서 수: {len(self.bm25_corpus)}")
    
    def _tokenize_korean(self, text: str) -> List[str]:
        """
        한글 텍스트 토큰화 (간단한 방법)
        
        Args:
            text: 토큰화할 텍스트
        
        Returns:
            토큰 리스트
        """
        # 공백과 특수문자 기준으로 분리
        tokens = re.findall(r'\w+', text.lower())
        return tokens
    
    def _build_bm25_index(self):
        """BM25 인덱스 구축"""
        # 각 청크의 content를 토큰화
        self.bm25_corpus = []
        for chunk in self.chunks:
            content = chunk.get('content', '')
            tokens = self._tokenize_korean(content)
            self.bm25_corpus.append(tokens)
        
        # BM25 인덱스 생성
        self.bm25 = BM25Okapi(self.bm25_corpus)
        print(f"✓ BM25 인덱스 생성 완료: {len(self.bm25_corpus)}개 문서")
    
    def vector_search(self, 
                     query: str,
                     top_k: int = 10) -> List[Dict]:
        """
        벡터 검색 (FAISS)
        
        Args:
            query: 검색 쿼리
            top_k: 반환할 결과 수
        
        Returns:
            검색 결과 리스트
        """
        if not self.embedding_manager:
            raise ValueError("EmbeddingManager가 필요합니다.")
        
        # 쿼리 임베딩
        query_embedding = self.embedding_manager.embed_text(query)
        query_embedding = query_embedding.reshape(1, -1).astype('float32')
        
        # FAISS 검색
        distances, indices = self.faiss_index.search(query_embedding, top_k)
        
        # 결과 구성
        results = []
        for i, (idx, distance) in enumerate(zip(indices[0], distances[0])):
            if idx < len(self.metadata):
                result = {
                    "rank": i + 1,
                    "chunk_id": self.metadata[idx]["chunk_id"],
                    "content": self.metadata[idx]["content"],
                    "metadata": self.metadata[idx]["metadata"],
                    "score": float(1 / (1 + distance)),  # 거리를 점수로 변환
                    "search_type": "vector"
                }
                results.append(result)
        
        return results
    
    def keyword_search(self,
                      query: str,
                      top_k: int = 10) -> List[Dict]:
        """
        키워드 검색 (BM25)
        
        Args:
            query: 검색 쿼리
            top_k: 반환할 결과 수
        
        Returns:
            검색 결과 리스트
        """
        # 쿼리 토큰화
        query_tokens = self._tokenize_korean(query)
        
        # BM25 스코어 계산
        scores = self.bm25.get_scores(query_tokens)
        
        # 상위 top_k개 선택
        top_indices = np.argsort(scores)[::-1][:top_k]
        
        # 결과 구성
        results = []
        for i, idx in enumerate(top_indices):
            if scores[idx] > 0:  # 스코어가 0보다 큰 것만
                chunk = self.chunks[idx]
                result = {
                    "rank": i + 1,
                    "chunk_id": chunk["chunk_id"],
                    "content": chunk["content"],
                    "metadata": chunk["metadata"],
                    "score": float(scores[idx]),
                    "search_type": "keyword"
                }
                results.append(result)
        
        return results
    
    def reciprocal_rank_fusion(self,
                               vector_results: List[Dict],
                               keyword_results: List[Dict],
                               k: int = 60) -> List[Dict]:
        """
        Reciprocal Rank Fusion (RRF) 알고리즘
        
        두 검색 결과를 최적으로 융합합니다.
        RRF 공식: score = 1/(k + rank_vector) + 1/(k + rank_keyword)
        
        Args:
            vector_results: 벡터 검색 결과
            keyword_results: 키워드 검색 결과
            k: RRF 상수 (기본값 60, 낮을수록 상위 랭크에 가중치)
        
        Returns:
            융합된 검색 결과
        """
        # chunk_id별로 점수 계산
        chunk_scores = {}
        chunk_data = {}
        
        # 벡터 검색 결과 처리
        for result in vector_results:
            chunk_id = result["chunk_id"]
            rank = result["rank"]
            rrf_score = 1 / (k + rank)
            
            chunk_scores[chunk_id] = chunk_scores.get(chunk_id, 0) + rrf_score
            chunk_data[chunk_id] = result
        
        # 키워드 검색 결과 처리
        for result in keyword_results:
            chunk_id = result["chunk_id"]
            rank = result["rank"]
            rrf_score = 1 / (k + rank)
            
            chunk_scores[chunk_id] = chunk_scores.get(chunk_id, 0) + rrf_score
            if chunk_id not in chunk_data:
                chunk_data[chunk_id] = result
        
        # 점수 기준으로 정렬
        sorted_chunks = sorted(chunk_scores.items(), key=lambda x: x[1], reverse=True)
        
        # 결과 구성
        results = []
        for i, (chunk_id, score) in enumerate(sorted_chunks):
            result = chunk_data[chunk_id].copy()
            result["rank"] = i + 1
            result["rrf_score"] = float(score)
            result["search_type"] = "hybrid"
            results.append(result)
        
        return results
    
    def hybrid_search(self,
                     query: str,
                     top_k: int = 10) -> List[Dict]:
        """
        하이브리드 검색 (벡터 + 키워드 + RRF)
        
        Args:
            query: 검색 쿼리
            top_k: 반환할 결과 수
        
        Returns:
            최종 검색 결과
        """
        # 1. 벡터 검색 (의미적 유사도)
        vector_results = self.vector_search(query, top_k=top_k*2)
        
        # 2. 키워드 검색 (정확한 매칭)
        keyword_results = self.keyword_search(query, top_k=top_k*2)
        
        # 3. RRF로 융합
        hybrid_results = self.reciprocal_rank_fusion(vector_results, keyword_results)
        
        # 4. 상위 top_k개만 반환
        return hybrid_results[:top_k]