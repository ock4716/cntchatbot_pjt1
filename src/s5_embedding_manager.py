"""
embedding_manager.py
[5단계] 임베딩 및 FAISS

임베딩 생성과 FAISS 인덱스 관리를 담당합니다.
- OpenAI 임베딩 생성 (text-embedding-3-large)
- FAISS 인덱스 생성/로드
- 메타데이터 관리
- 임베딩 캐싱
"""

import os
import json
import pickle
import hashlib
from typing import List, Dict, Optional, Tuple
import numpy as np
from openai import OpenAI
import faiss


class EmbeddingManager:
    """임베딩 생성 및 FAISS 인덱스 관리 클래스"""
    
    def __init__(self, 
                 openai_api_key: str,
                 institution: str = "unknown",  # ← 추가
                 model: str = "text-embedding-3-large",
                 cache_path: str = None,  # ← None으로 변경
                 dimension: int = 3072):
        """
        EmbeddingManager 초기화
        
        Args:
            openai_api_key: OpenAI API 키
            model: 임베딩 모델명
            cache_path: 임베딩 캐시 파일 경로
            dimension: 임베딩 차원 (text-embedding-3-large = 3072)
        """
        self.client = OpenAI(api_key=openai_api_key)
        self.model = model
        self.institution = institution  # ← 추가
    
        # 캐시 경로 자동 생성 (기관별)
        if cache_path is None:
            cache_path = f"data/cache/embeddings_{institution}.pkl"

        self.cache_path = cache_path
        self.dimension = dimension
        self.embedding_cache = self.load_embedding_cache()
        
        # 캐시 디렉토리 생성
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        
        print(f"✓ EmbeddingManager 초기화 완료")
        print(f"  - 모델: {model}")
        print(f"  - 차원: {dimension}")
        print(f"  - 캐시: {len(self.embedding_cache)}개 임베딩")
    
    def load_embedding_cache(self) -> Dict[str, np.ndarray]:
        """
        임베딩 캐시 로드
        
        Returns:
            캐시 딕셔너리 {text_hash: embedding_vector}
        """
        if os.path.exists(self.cache_path):
            try:
                with open(self.cache_path, 'rb') as f:
                    cache = pickle.load(f)
                print(f"✓ 캐시 로드: {len(cache)}개 임베딩")
                return cache
            except Exception as e:
                print(f"⚠ 캐시 로드 실패 ({e}), 새로 시작합니다.")
                return {}
        return {}
    
    def save_embedding_cache(self):
        """임베딩 캐시 저장"""
        try:
            with open(self.cache_path, 'wb') as f:
                pickle.dump(self.embedding_cache, f)
            print(f"✓ 캐시 저장: {len(self.embedding_cache)}개 임베딩")
        except Exception as e:
            print(f"⚠ 캐시 저장 실패: {e}")
    
    def get_text_hash(self, text: str) -> str:
        """
        텍스트의 MD5 해시 계산 (캐시 키로 사용)
        
        Args:
            text: 해시를 계산할 텍스트
        
        Returns:
            MD5 해시 문자열
        """
        return hashlib.md5(text.encode('utf-8')).hexdigest()
    
    def embed_text(self, text: str) -> np.ndarray:
        """
        텍스트를 임베딩 벡터로 변환
        
        Args:
            text: 임베딩할 텍스트
        
        Returns:
            임베딩 벡터 (numpy array)
        """
        # 캐시 확인
        text_hash = self.get_text_hash(text)
        if text_hash in self.embedding_cache:
            return self.embedding_cache[text_hash]
        
        # OpenAI API 호출
        try:
            response = self.client.embeddings.create(
                input=text,
                model=self.model
            )
            
            embedding = np.array(response.data[0].embedding, dtype='float32')
            
            # 캐시에 저장
            self.embedding_cache[text_hash] = embedding
            
            return embedding
            
        except Exception as e:
            print(f"⚠ 임베딩 생성 실패: {e}")
            # 실패 시 제로 벡터 반환
            return np.zeros(self.dimension, dtype='float32')
    
    def embed_chunks(self, chunks: List[Dict], batch_size: int = 100) -> Tuple[List[np.ndarray], List[str]]:
        """
        여러 청크를 배치로 임베딩
        
        Args:
            chunks: 청크 리스트
            batch_size: 배치 크기 (OpenAI API는 최대 2048개까지 지원)
        
        Returns:
            (임베딩 벡터 리스트, 청크 ID 리스트)
        """
        embeddings = []
        chunk_ids = []
        
        print(f"\n📊 임베딩 생성 시작...")
        print(f"  - 총 청크 수: {len(chunks)}")
        print(f"  - 배치 크기: {batch_size}")
        
        # 캐시 히트/미스 카운트
        cache_hits = 0
        cache_misses = 0
        
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i+batch_size]
            batch_texts = [chunk['content'] for chunk in batch]
            batch_chunk_ids = [chunk['chunk_id'] for chunk in batch]
            
            print(f"\n  배치 {i//batch_size + 1}/{(len(chunks)-1)//batch_size + 1} 처리 중...")
            
            # 배치 내에서 캐시 확인
            batch_embeddings = []
            texts_to_embed = []
            text_indices = []
            
            for j, text in enumerate(batch_texts):
                text_hash = self.get_text_hash(text)
                if text_hash in self.embedding_cache:
                    batch_embeddings.append(self.embedding_cache[text_hash])
                    cache_hits += 1
                else:
                    batch_embeddings.append(None)  # 나중에 채울 자리
                    texts_to_embed.append(text)
                    text_indices.append(j)
                    cache_misses += 1
            
            # 캐시에 없는 것만 API 호출
            if texts_to_embed:
                try:
                    response = self.client.embeddings.create(
                        input=texts_to_embed,
                        model=self.model
                    )
                    
                    # 결과를 해당 위치에 채우기
                    for idx, data in enumerate(response.data):
                        embedding = np.array(data.embedding, dtype='float32')
                        original_idx = text_indices[idx]
                        batch_embeddings[original_idx] = embedding
                        
                        # 캐시에 저장
                        text_hash = self.get_text_hash(texts_to_embed[idx])
                        self.embedding_cache[text_hash] = embedding
                    
                    print(f"    ✓ {len(texts_to_embed)}개 새로 임베딩 생성")
                    
                except Exception as e:
                    print(f"    ✗ 배치 임베딩 실패: {e}")
                    # 실패한 경우 제로 벡터로 채우기
                    for idx in text_indices:
                        if batch_embeddings[idx] is None:
                            batch_embeddings[idx] = np.zeros(self.dimension, dtype='float32')
            
            embeddings.extend(batch_embeddings)
            chunk_ids.extend(batch_chunk_ids)
            
            # 진행률 출력
            progress = min((i + batch_size) / len(chunks) * 100, 100)
            print(f"    진행률: {progress:.1f}%")
        
        print(f"\n✓ 임베딩 생성 완료!")
        print(f"  - 캐시 히트: {cache_hits}개")
        print(f"  - 새로 생성: {cache_misses}개")
        print(f"  - 총 임베딩: {len(embeddings)}개")
        
        # 캐시 저장
        if cache_misses > 0:
            self.save_embedding_cache()
        
        return embeddings, chunk_ids
    
    def create_faiss_index(self, embeddings: List[np.ndarray]) -> faiss.Index:
        """
        FAISS 인덱스 생성
        
        Args:
            embeddings: 임베딩 벡터 리스트
        
        Returns:
            FAISS 인덱스
        """
        print(f"\n🔧 FAISS 인덱스 생성 중...")
        
        # Flat 인덱스 생성 (정확한 최근접 이웃 검색)
        index = faiss.IndexFlatL2(self.dimension)
        
        # 임베딩을 numpy 배열로 변환
        embeddings_array = np.array(embeddings).astype('float32')
        
        # 인덱스에 추가
        index.add(embeddings_array)
        
        print(f"✓ FAISS 인덱스 생성 완료")
        print(f"  - 인덱스 타입: Flat (L2 distance)")
        print(f"  - 벡터 수: {index.ntotal}")
        print(f"  - 차원: {self.dimension}")
        
        return index
    
    def save_index(self, index: faiss.Index, index_path: str):
        """
        FAISS 인덱스 저장
        
        Args:
            index: 저장할 FAISS 인덱스
            index_path: 저장 경로
        """
        os.makedirs(os.path.dirname(index_path), exist_ok=True)
        
        try:
            faiss.write_index(index, index_path)
            print(f"✓ 인덱스 저장: {index_path}")
        except Exception as e:
            print(f"✗ 인덱스 저장 실패: {e}")
    
    def load_index(self, index_path: str) -> Optional[faiss.Index]:
        """
        FAISS 인덱스 로드
        
        Args:
            index_path: 인덱스 파일 경로
        
        Returns:
            FAISS 인덱스 또는 None
        """
        if not os.path.exists(index_path):
            print(f"⚠ 인덱스 파일이 없습니다: {index_path}")
            return None
        
        try:
            index = faiss.read_index(index_path)
            print(f"✓ 인덱스 로드: {index_path}")
            print(f"  - 벡터 수: {index.ntotal}")
            return index
        except Exception as e:
            print(f"✗ 인덱스 로드 실패: {e}")
            return None
    
    def save_metadata(self, chunks: List[Dict], chunk_ids: List[str], metadata_path: str):
        """
        메타데이터 저장
        
        Args:
            chunks: 청크 리스트
            chunk_ids: 청크 ID 리스트 (인덱스 순서와 동일)
            metadata_path: 저장 경로
        """
        os.makedirs(os.path.dirname(metadata_path), exist_ok=True)
        
        # chunk_id를 키로 하는 딕셔너리 생성
        chunk_dict = {chunk['chunk_id']: chunk for chunk in chunks}
        
        # 인덱스 순서대로 메타데이터 배열 생성
        metadata = []
        for i, chunk_id in enumerate(chunk_ids):
            chunk = chunk_dict.get(chunk_id, {})
            metadata.append({
                "index": i,
                "chunk_id": chunk_id,
                "content": chunk.get("content", ""),
                "metadata": chunk.get("metadata", {})
            })
        
        try:
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, ensure_ascii=False, indent=2)
            print(f"✓ 메타데이터 저장: {metadata_path}")
            print(f"  - 항목 수: {len(metadata)}")
        except Exception as e:
            print(f"✗ 메타데이터 저장 실패: {e}")
    
    def load_metadata(self, metadata_path: str) -> Optional[List[Dict]]:
        """
        메타데이터 로드
        
        Args:
            metadata_path: 메타데이터 파일 경로
        
        Returns:
            메타데이터 리스트 또는 None
        """
        if not os.path.exists(metadata_path):
            print(f"⚠ 메타데이터 파일이 없습니다: {metadata_path}")
            return None
        
        try:
            with open(metadata_path, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
            print(f"✓ 메타데이터 로드: {metadata_path}")
            print(f"  - 항목 수: {len(metadata)}")
            return metadata
        except Exception as e:
            print(f"✗ 메타데이터 로드 실패: {e}")
            return None
    
    def search(self, query: str, index: faiss.Index, metadata: List[Dict], 
               top_k: int = 10) -> List[Dict]:
        """
        벡터 검색 수행
        
        Args:
            query: 검색 쿼리
            index: FAISS 인덱스
            metadata: 메타데이터 리스트
            top_k: 반환할 결과 수
        
        Returns:
            검색 결과 리스트
        """
        # 쿼리 임베딩
        query_embedding = self.embed_text(query)
        query_embedding = query_embedding.reshape(1, -1).astype('float32')
        
        # FAISS 검색
        distances, indices = index.search(query_embedding, top_k)
        
        # 결과 구성
        results = []
        for i, (idx, distance) in enumerate(zip(indices[0], distances[0])):
            if idx < len(metadata):
                result = {
                    "rank": i + 1,
                    "chunk_id": metadata[idx]["chunk_id"],
                    "content": metadata[idx]["content"],
                    "metadata": metadata[idx]["metadata"],
                    "distance": float(distance),
                    "similarity": float(1 / (1 + distance))  # 거리를 유사도로 변환
                }
                results.append(result)
        
        return results
    
    def build_index_from_chunks(self, chunks_path: str, 
                                output_dir: str = None) -> Tuple[faiss.Index, List[Dict]]:

        """
        청크 파일에서 인덱스 구축 (전체 파이프라인)
        
        Args:
            chunks_path: 청크 JSON 파일 경로
            output_dir: 출력 디렉토리
        
        Returns:
            (FAISS 인덱스, 메타데이터 리스트)
        """
        if output_dir is None:
            output_dir = f"data/vector_store/{self.institution}"
    
        print("\n" + "="*80)
        print("🚀 FAISS 인덱스 구축 시작")
        print("="*80)
        
        # 1. 청크 로드
        print("\n1️⃣ 청크 로드 중...")
        with open(chunks_path, 'r', encoding='utf-8') as f:
            chunks = json.load(f)
        print(f"✓ {len(chunks)}개 청크 로드 완료")
        
        # 2. 임베딩 생성
        print("\n2️⃣ 임베딩 생성 중...")
        embeddings, chunk_ids = self.embed_chunks(chunks, batch_size=100)
        
        # 3. FAISS 인덱스 생성
        print("\n3️⃣ FAISS 인덱스 생성 중...")
        index = self.create_faiss_index(embeddings)
        
        # 4. 인덱스 저장
        print("\n4️⃣ 인덱스 저장 중...")
        index_path = os.path.join(output_dir, "faiss_index.bin")
        self.save_index(index, index_path)
        
        # 5. 메타데이터 저장
        print("\n5️⃣ 메타데이터 저장 중...")
        metadata_path = os.path.join(output_dir, "metadata.json")
        self.save_metadata(chunks, chunk_ids, metadata_path)
        
        # 6. 메타데이터 로드 (검증)
        metadata = self.load_metadata(metadata_path)
        
        print("\n" + "="*80)
        print("✅ FAISS 인덱스 구축 완료!")
        print("="*80)
        print(f"📁 출력 디렉토리: {output_dir}")
        print(f"  - faiss_index.bin")
        print(f"  - metadata.json")
        print("="*80 + "\n")
        
        return index, metadata