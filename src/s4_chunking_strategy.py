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
        # tiktoken: OpenAI의 토큰 계산 라이브러리
        # 모델별로 다른 토크나이저를 사용 (gpt-4, gpt-3.5-turbo 등)
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
    
    def split_text_by_tokens(self, text: str, max_tokens: int) -> List[str]:
        """
        텍스트를 토큰 수 기준으로 분할
        
        핵심 로직:
        1. 문장 단위로 먼저 분할 ('. ' 기준)
        2. 각 문장을 순회하며 토큰 수 계산
        3. max_tokens를 넘지 않는 선에서 문장들을 하나의 청크로 묶음
        4. 넘으면 새로운 청크 시작
        
        Args:
            text: 분할할 텍스트
            max_tokens: 최대 토큰 수
        
        Returns:
            분할된 텍스트 리스트
        """
        # 문장 단위로 먼저 분할 (줄바꿈을 공백으로 변환 후 '. '로 분할)
        sentences = text.replace('\n', ' ').split('. ')
        
        chunks = []
        current_chunk = ""  # 현재 청크에 누적되는 텍스트
        current_tokens = 0  # 현재 청크의 누적 토큰 수
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            
            # 문장 끝에 마침표와 공백 추가
            sentence = sentence + '. '
            sentence_tokens = self.count_tokens(sentence)
            
            # 현재 청크에 추가해도 max_tokens를 넘지 않으면 추가
            if current_tokens + sentence_tokens <= max_tokens:
                current_chunk += sentence
                current_tokens += sentence_tokens
            else:
                # max_tokens를 넘으면: 현재 청크를 저장하고 새 청크 시작
                if current_chunk:
                    chunks.append(current_chunk.strip())
                # 새 청크는 현재 문장으로 시작
                current_chunk = sentence
                current_tokens = sentence_tokens
        
        # 마지막 청크 추가 (루프가 끝난 후 남은 청크)
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        return chunks
    
    def chunk_pages(self, text_blocks: List[Dict], institution: str) -> List[Dict]:
        """
        페이지를 토큰 기반으로 청킹
        
        핵심 로직:
        1. 페이지별로 텍스트 블록을 그룹핑
        2. 각 페이지의 텍스트를 합쳐서 하나의 문자열로 만듦
        3. split_text_by_tokens()로 토큰 기반 분할
        4. 각 청크에 메타데이터(기관, 페이지 번호 등) 추가
        
        Args:
            text_blocks: 텍스트 블록 리스트 (각 블록은 {"text": "...", "page_num": 1} 형태)
            institution: 기관 코드 (hd, kb, khi)
        
        Returns:
            청크 리스트 (각 청크는 chunk_id, content, metadata 포함)
        """
        
        # 1단계: 페이지별로 텍스트 블록 그룹핑
        # pages_dict = {1: ["text1", "text2"], 2: ["text3"], ...}
        pages_dict = {}
        for block in text_blocks:
            page_num = block.get("page_num", 0)
            if page_num not in pages_dict:
                pages_dict[page_num] = []
            pages_dict[page_num].append(block["text"])
        
        # 2단계: 페이지별로 청킹 수행
        all_chunks = []
        chunk_counter = 1  # 전체 청크에 대한 일련번호
        
        for page_num in sorted(pages_dict.keys()):
            # 해당 페이지의 모든 텍스트 블록을 하나로 합침
            page_text = "\n".join(pages_dict[page_num])
            
            # 토큰 기반으로 분할 (self.chunk_size 기준)
            text_chunks = self.split_text_by_tokens(page_text, self.chunk_size)
            
            # 각 청크에 메타데이터 추가
            for chunk_text in text_chunks:
                all_chunks.append({
                    "chunk_id": f"chunk_{chunk_counter:04d}",  # chunk_0001, chunk_0002, ...
                    "content": chunk_text,
                    "metadata": {
                        "institution": institution,  # 출처 기관
                        "doc_type": "text",  # 텍스트 청크임을 표시
                        "page": page_num,  # 원본 페이지 번호
                        "chunk_tokens": self.count_tokens(chunk_text)  # 청크의 토큰 수
                    }
                })
                chunk_counter += 1
        
        return all_chunks
    
    def make_table_to_chunk(self, table_data: Dict) -> Dict:
        """
        하나의 표를 하나의 청크로 생성
        
        핵심 개념:
        - 표는 분할하지 않고 전체를 하나의 청크로 유지
        - 표의 자연어 변환 결과(content)를 사용
        
        Args:
            table_data: 표 데이터 ({"content": "...", "table_id": "...", "page_num": 1, ...})
        
        Returns:
            표 청크
        """
        # 표의 자연어 변환 결과 사용
        table_content = table_data.get("content", "")
        caption = table_data.get("caption", "")
        
        # 표 청크 생성 (doc_type="table"로 구분)
        return {
            "chunk_id": f"chunk_table_{table_data['table_id']}",
            "content": table_content,
            "metadata": {
                "institution": table_data.get("institution", "unknown"),
                "doc_type": "table",  # 표 청크임을 표시
                "table_id": table_data["table_id"],

                "page": table_data.get("page_num", 0),
                "chunk_tokens": self.count_tokens(table_content)
            }
        }
    
    def make_image_to_chunk(self, image_data: Dict) -> Dict:
        """
        하나의 이미지를 하나의 청크로 생성
        
        핵심 개념:
        - 이미지는 GPT-4V 분석 결과(description)를 텍스트로 변환하여 저장
        - 이미지 자체가 아닌 이미지 설명을 청크로 만듦
        
        Args:
            image_data: 이미지 데이터 ({"description": "...", "image_path": "...", ...})
        
        Returns:
            이미지 청크
        """
        # GPT-4V 분석 결과(이미지 설명) 사용
        image_description = image_data.get("description", "")
        caption = image_data.get("caption", "")
        
        if caption:
            chunk_content = f"[{caption}]\n\n{image_description}"
        else:
            chunk_content = image_description
    
        # 이미지 파일명에서 식별자 추출 (예: image_01.png -> image_01_png)
        image_path = image_data.get("image_path", "")
        image_filename = image_data.get("image_filename", "")
        image_id = image_filename.replace(".", "_") if image_filename else "unknown"
        
        # 이미지 청크 생성 (doc_type="image"로 구분)
        return {
            "chunk_id": f"chunk_image_{image_id}",
            "content": chunk_content,
            "metadata": {
                "institution": image_data.get("institution", "unknown"),
                "doc_type": "image",  # 이미지 청크임을 표시
                "image_path": image_path,  # 원본 이미지 경로 보존
                "caption": caption,
                "page": image_data.get("page_num", 0),
                "chunk_tokens": self.count_tokens(chunk_content)
            }
        }
    
    def apply_overlap(self, chunks: List[Dict]) -> List[Dict]:
        """
        청크 간 오버랩 적용
        
        핵심 개념:
        - 연속된 텍스트 청크 간에 문맥을 유지하기 위해 오버랩 추가
        - 현재 청크의 끝에 다음 청크의 시작 부분을 미리보기로 추가
        - 표/이미지 청크는 오버랩 적용 안 함 (독립적인 정보 단위)
        
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
            # (각각 독립적인 정보 단위이므로)
            if chunk["metadata"]["doc_type"] != "text":
                overlapped_chunks.append(chunk)
                continue
            
            content = chunk["content"]
            
            # 다음 청크가 있고, 다음 청크도 텍스트인 경우에만 오버랩 추가
            if i < len(chunks) - 1:
                next_chunk = chunks[i + 1]
                if next_chunk["metadata"]["doc_type"] == "text":
                    next_content = next_chunk["content"]
                    
                    # 다음 청크의 시작 부분을 self.overlap 토큰만큼 추출
                    tokens = self.encoder.encode(next_content)
                    if len(tokens) > self.overlap:
                        overlap_tokens = tokens[:self.overlap]
                        overlap_text = self.encoder.decode(overlap_tokens)
                        # 현재 청크 끝에 "[다음 내용 미리보기]" 형태로 추가
                        content += f"\n\n[다음 내용 미리보기]\n{overlap_text}..."
            
            # 오버랩이 적용된 청크 생성
            overlapped_chunks.append({
                **chunk,
                "content": content,
                "metadata": {
                    **chunk["metadata"],
                    "has_overlap": True,  # 오버랩 적용 여부 표시
                    "chunk_tokens": self.count_tokens(content)  # 오버랩 포함한 토큰 수로 재계산
                }
            })
        
        return overlapped_chunks
    
    def process_from_json(self, json_path: str) -> List[Dict]:
        """
        JSON 파일에서 데이터를 로드하여 청킹 수행
        
        전체 워크플로우:
        1. JSON 파일 로드 (기관별 processed.json)
        2. 텍스트/표/이미지 데이터 분리
        3. 각 타입별로 청킹 수행
        4. 모든 청크 결합
        5. 오버랩 적용
        
        Args:
            json_path: processed.json 파일 경로
                      (예: data/processed/hd/hd_report_processed.json)
        
        Returns:
            청크 리스트
        """
        # JSON 파일 로드
        if not os.path.exists(json_path):
            raise FileNotFoundError(f"파일을 찾을 수 없습니다: {json_path}")
        
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # 기관 정보 추출 (JSON 구조에 따라 다른 위치에서 추출 시도)
        institution = data["institution"]
        
        print(f"✓ JSON 로드 완료: {json_path}")
        print(f"✓ 기관: {institution.upper()}")
        
        # 텍스트 블록 수집 (모든 페이지의 text_blocks 수집)
        text_blocks = []
        for page_data in data.get("texts", []):
            text_blocks.extend(page_data.get("text", []))
        
        # 표 데이터 수집 (모든 페이지의 tables 수집)
        tables = []
        for page_data in data.get("tables", []):
            tables.extend(page_data.get("content", []))
        
        # 이미지 데이터 수집 (모든 페이지의 images 수집)
        images = []
        for page_data in data.get("images", []):
            images.extend(page_data.get("description", []))
        
        print(f"  - 텍스트 블록: {len(text_blocks)}개")
        print(f"  - 표: {len(tables)}개")
        print(f"  - 이미지: {len(images)}개")
        
        # 1단계: 텍스트 청킹
        print(f"\n1️⃣ 텍스트 청킹 중...")
        text_chunks = self.chunk_pages(text_blocks, institution)
        print(f"  ✓ {len(text_chunks)}개 텍스트 청크 생성")
        
        # 2단계: 표 청킹 (각 표를 하나의 청크로)
        print(f"\n2️⃣ 표 청킹 중...")
        table_chunks = []
        for table_data in tables:
            table_chunk = self.make_table_to_chunk(table_data)
            table_chunks.append(table_chunk)
        print(f"  ✓ {len(table_chunks)}개 표 청크 생성")
        
        # 3단계: 이미지 청킹 (각 이미지를 하나의 청크로)
        print(f"\n3️⃣ 이미지 청킹 중...")
        image_chunks = []
        for image_data in images:
            image_chunk = self.make_image_to_chunk(image_data)
            image_chunks.append(image_chunk)
        print(f"  ✓ {len(image_chunks)}개 이미지 청크 생성")
        
        # 4단계: 모든 청크 결합
        # 순서: 텍스트 -> 표 -> 이미지
        all_chunks = text_chunks + table_chunks + image_chunks
        
        # 5단계: 오버랩 적용 (텍스트 청크 간에만)
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
        # 출력 디렉토리가 없으면 생성
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # JSON 파일로 저장 (한글 깨짐 방지: ensure_ascii=False, 들여쓰기: indent=2)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(chunks, f, ensure_ascii=False, indent=2)
        
        print(f"\n✓ 청크 저장 완료: {output_path}")