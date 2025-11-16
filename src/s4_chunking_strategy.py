"""
chunking_strategy.py
[4단계] 청킹 전략

문서를 토큰 기반으로 분할하고 출처(기관) 정보를 추가합니다.
- 토큰 기반 청킹 (섹션 구분 없음)
- 기관별 출처 구분 (HD, KB, KHI)
- 표/이미지와 텍스트 연결
- 청크 오버랩 처리
"""

import tiktoken
from typing import List, Dict
import json
import os


class ChunkingStrategy:
    """문서를 청킹하는 클래스"""
    
    def __init__(self, chunk_size: int = 1000, overlap: int = 300, 
                 model: str = "gpt-4"):
        """
        ChunkingStrategy 초기화
        
        Args:
            chunk_size: 청크 크기 (토큰 수)
            overlap: 오버랩 크기 (토큰 수)
            model: 토큰 계산에 사용할 모델명
        """
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.encoder = tiktoken.encoding_for_model(model)
    
    def count_tokens(self, text: str) -> int:
        """
        텍스트의 토큰 수 계산
        
        Args:
            text: 토큰을 계산할 텍스트
        
        Returns:
            토큰 수
        """
        return len(self.encoder.encode(text))
    
    def chunk_text_by_tokens(self, text: str, max_tokens: int) -> List[str]:
        """
        텍스트를 토큰 수 기준으로 분할
        
        Args:
            text: 분할할 텍스트
            max_tokens: 최대 토큰 수
        
        Returns:
            분할된 텍스트 리스트
        """
        # 문장 단위로 먼저 분할
        sentences = text.replace('\n', ' ').split('. ')
        
        chunks = []
        current_chunk = "" # 현재 청크 텍스트
        current_tokens = 0 # 현재 청크의 토큰 수
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            
            sentence = sentence + '. '
            sentence_tokens = self.count_tokens(sentence)
            
            # 현재 청크에 추가해도 max_tokens를 넘지 않으면 추가
            if current_tokens + sentence_tokens <= max_tokens:
                current_chunk += sentence # 현재 청크에 문장 추가
                current_tokens += sentence_tokens # 토큰 수 갱신
            else:
                # 현재 청크 저장하고 새 청크 시작
                if current_chunk:
                    chunks.append(current_chunk.strip()) # 현재 청크를 청크 리스트에 추가
                current_chunk = sentence # 새 청크 시작
                current_tokens = sentence_tokens # 토큰 수 갱신
        
        # 마지막 청크 추가
        if current_chunk:
            chunks.append(current_chunk.strip()) # 마지막 청크를 청크 리스트에 추가
        
        return chunks
    
    def chunk_text(self, text_blocks: List[Dict], institution: str) -> List[Dict]:
        """
        텍스트를 토큰 기반으로 청킹
        
        Args:
            text_blocks: 텍스트 블록 리스트
            institution: 기관 코드 (hd, kb, khi)
        
        Returns:
            청크 리스트
        """
        
        # 페이지별로 그룹핑
        pages_dict = {}
        for block in text_blocks:
            page_num = block.get("page_num", 0)
            if page_num not in pages_dict:
                pages_dict[page_num] = []
            pages_dict[page_num].append(block["text"])
        
        # 페이지별로 청킹
        all_chunks = []
        chunk_counter = 1
        
        for page_num in sorted(pages_dict.keys()):
            # 해당 페이지의 텍스트 합치기
            page_text = "\n".join(pages_dict[page_num])
            
            # 페이지별 청킹
            text_chunks = self.chunk_text_by_tokens(page_text, self.chunk_size)
            
            for chunk_text in text_chunks:
                all_chunks.append({
                    "chunk_id": f"chunk_{chunk_counter:04d}",
                    "content": chunk_text,
                    "metadata": {
                        "institution": institution,
                        "doc_type": "text",
                        "page": page_num,  
                        "chunk_tokens": self.count_tokens(chunk_text)
                    }
                })
                chunk_counter += 1
        
        return all_chunks
    
    def make_table_to_chunk(self, table_data: Dict) -> Dict:
        """
        하나의 표를 하나의 청크로 생성
        
        Args:
            table_data: 표 데이터 (processed_contents.json에서 로드)
        
        Returns:
            표 청크
        """
        # 표의 자연어 변환 결과 사용 (content 필드가 있으면)
        table_content = table_data.get("content", "")
        
        return {
            "chunk_id": f"chunk_table_{table_data['table_id']}",
            "content": table_content,
            "metadata": {
                "institution": table_data.get("institution", "unknown"),
                "doc_type": "table",
                "table_id": table_data["table_id"],
                "page": table_data.get("page_num", 0),
                "chunk_tokens": self.count_tokens(table_content)
            }
        }
    
    def make_image_to_chunk(self, image_data: Dict) -> Dict:
        """
        하나의 이미지를 하나의 청크로 생성
        
        Args:
            image_data: 이미지 데이터 (processed_contents.json에서 로드)
        
        Returns:
            이미지 청크
        """
        # GPT-4V 분석 결과 사용 (description 필드가 있으면)
        image_description = image_data.get("description", "")
        
        chunk_content = image_description
        
        # 이미지 경로에서 식별자 추출
        image_path = image_data.get("image_path", "")
        image_filename = image_data.get("image_filename", "")
        image_id = image_filename.replace(".", "_") if image_filename else "unknown"
        
        return {
            "chunk_id": f"chunk_image_{image_id}",
            "content": chunk_content,
            "metadata": {
                "institution": image_data.get("institution", "unknown"),
                "doc_type": "image",
                "image_path": image_path,
                "page": image_data.get("page_num", 0),
                "chunk_tokens": self.count_tokens(chunk_content)
            }
        }
    
    def apply_overlap(self, chunks: List[Dict]) -> List[Dict]:
        """
        청크 간 오버랩 적용
        
        Args:
            chunks: 청크 리스트
        
        Returns:
            오버랩이 적용된 청크 리스트
        """
        if not chunks or self.overlap == 0:
            return chunks
        
        overlapped_chunks = []
        
        for i, chunk in enumerate(chunks):
            # 표나 이미지 청크는 오버랩 적용하지 않음
            if chunk["metadata"]["doc_type"] != "text":
                overlapped_chunks.append(chunk)
                continue
            
            content = chunk["content"]
            
            # 다음 청크가 있고, 같은 타입이면 오버랩 추가
            if i < len(chunks) - 1:
                next_chunk = chunks[i + 1]
                if next_chunk["metadata"]["doc_type"] == "text":
                    # 다음 청크의 시작 부분을 현재 청크의 끝에 추가
                    next_content = next_chunk["content"]
                    
                    # 오버랩할 텍스트 추출 (토큰 기준)
                    tokens = self.encoder.encode(next_content)
                    if len(tokens) > self.overlap:
                        overlap_tokens = tokens[:self.overlap]
                        overlap_text = self.encoder.decode(overlap_tokens)
                        content += f"\n\n[다음 내용 미리보기]\n{overlap_text}..."
            
            overlapped_chunks.append({
                **chunk,
                "content": content,
                "metadata": {
                    **chunk["metadata"],
                    "has_overlap": True,
                    "chunk_tokens": self.count_tokens(content)
                }
            })
        
        return overlapped_chunks
    
    def process_from_json(self, json_path: str = "data/processed/processed_contents.json") -> List[Dict]:
        """
        JSON 파일에서 데이터를 로드하여 청킹 수행
        
        Args:
            json_path: processed_contents.json 파일 경로
        
        Returns:
            청크 리스트
        """
        # JSON 파일 로드
        if not os.path.exists(json_path):
            raise FileNotFoundError(f"파일을 찾을 수 없습니다: {json_path}")
        
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # 기관 정보 추출 (첫 번째 페이지에서)
        institution = "unknown"
        if data.get("pages") and len(data["pages"]) > 0:
            institution = data["pages"][0].get("institution", "unknown")
        elif "institution" in data:
            institution = data["institution"]
        
        print(f"✓ JSON 로드 완료: {json_path}")
        print(f"✓ 기관: {institution.upper()}")
        
        # 텍스트 블록 수집
        text_blocks = []
        for page_data in data.get("pages", []):
            text_blocks.extend(page_data.get("text_blocks", []))
        
        # 표 데이터 수집
        tables = []
        for page_data in data.get("pages", []):
            tables.extend(page_data.get("tables", []))
        
        # 이미지 데이터 수집
        images = []
        for page_data in data.get("pages", []):
            images.extend(page_data.get("images", []))
        
        print(f"  - 텍스트 블록: {len(text_blocks)}개")
        print(f"  - 표: {len(tables)}개")
        print(f"  - 이미지: {len(images)}개")
        
        print(f"\n1️⃣ 텍스트 청킹 중...")
        text_chunks = self.chunk_text(text_blocks, institution)
        print(f"  ✓ {len(text_chunks)}개 텍스트 청크 생성")
        
        print(f"\n2️⃣ 표 청킹 중...")
        table_chunks = []
        for table_data in tables:
            table_chunk = self.make_table_to_chunk(table_data)
            table_chunks.append(table_chunk)
        print(f"  ✓ {len(table_chunks)}개 표 청크 생성")
        
        print(f"\n3️⃣ 이미지 청킹 중...")
        image_chunks = []
        for image_data in images:
            image_chunk = self.make_image_to_chunk(image_data)
            image_chunks.append(image_chunk)
        print(f"  ✓ {len(image_chunks)}개 이미지 청크 생성")
        
        # 모든 청크 결합
        all_chunks = text_chunks + table_chunks + image_chunks
        
        print(f"\n4️⃣ 오버랩 적용 중...")
        final_chunks = self.apply_overlap(all_chunks)
        print(f"  ✓ 최종 {len(final_chunks)}개 청크 생성")
        
        return final_chunks
    
    def save_chunks(self, chunks: List[Dict], output_path: str):
        """
        청크를 JSON 파일로 저장
        
        Args:
            chunks: 저장할 청크 리스트
            output_path: 저장 경로
        """
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(chunks, f, ensure_ascii=False, indent=2)
        
        print(f"\n✓ 청크 저장 완료: {output_path}")