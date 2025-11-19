"""
document_processor.py
[1단계] PDF 파싱 및 전처리

이 모듈은 PDF 문서에서 텍스트, 이미지, 표를 추출합니다.
- PDF 텍스트/이미지/표 추출 (pymupdf)
- 기관별 구분 (HD, KB, KHI)
- KHI는 이미지 추출 제외
- 캡션/제목 추출 없음
"""

import fitz  # pymupdf
import camelot
import os
from typing import List, Dict
from pathlib import Path
import json


class PDFProcessor:
    """PDF 문서 파싱 및 전처리 클래스"""
    
    def __init__(self, pdf_path: str, output_dir: str = None):
        """
        PDFProcessor 초기화
        
        Args:
            pdf_path: 처리할 PDF 파일 경로
            output_dir: 처리된 데이터를 저장할 디렉토리
        """
        self.pdf_path = pdf_path
        
        # PDF 파일명 추출 (확장자 제외)
        self.pdf_name = Path(pdf_path).stem
        
        # 기관명 자동 감지
        self.institution = self._detect_institution()
        print(f"✓ 감지된 기관: {self.institution.upper()}")
        
        # 프로젝트 루트 디렉토리 찾기
        if output_dir is None:
            project_root = Path(__file__).parent.parent
            output_dir = project_root / "data" / "processed"
        
        self.output_dir = Path(output_dir)
        
        # PDF 문서 열기
        try:
            self.doc = fitz.open(pdf_path)
            self.total_pages = len(self.doc)
            print(f"✓ PDF 로드 완료: {self.total_pages}페이지")
        except Exception as e:
            raise Exception(f"PDF 파일을 열 수 없습니다: {e}")
        
        # 출력 디렉토리 생성
        self.images_dir = self.output_dir / "images"
        self.tables_dir = self.output_dir / "tables"
        self.images_dir.mkdir(parents=True, exist_ok=True)
        self.tables_dir.mkdir(parents=True, exist_ok=True)
    
    def _detect_institution(self) -> str:
        """
        PDF 파일명에서 기관 자동 감지
        
        Returns:
            기관 코드 ("hd", "kb", "khi", "unknown")
        """
        filename = self.pdf_name.lower()
        
        # 파일명 패턴 매칭
        if "hd" in filename or "hyundai" in filename or "현대" in filename:
            return "hd"
        elif "khi" in filename or "housing" in filename or "주택금융" in filename:
            return "khi"
        elif "kb" in filename:
            return "kb"
        
        print(f"⚠ 기관을 감지할 수 없습니다. 기본값(unknown) 사용")
        return "unknown"
    
    def _find_caption(self, page, bbox: tuple, element_type: str = "image", 
                  search_distance: float = 100) -> str:
        """
        이미지나 표 근처에서 제목(캡션) 찾기
        
        Args:
            page: PyMuPDF 페이지 객체
            bbox: 요소의 경계 좌표 (x0, y0, x1, y1)
            element_type: "image" 또는 "table"
            search_distance: 제목을 찾을 최대 거리 (픽셀)
        
        Returns:
            찾은 제목 문자열 (없으면 빈 문자열)
        """
        # 검색할 키워드 패턴
        if element_type == "image":
            patterns = ["그림", "Figure", "Fig", "차트", "Chart", "그래프", "Graph"]
        else:  # table
            patterns = ["표", "Table", "Tab"]
        
        # 요소의 위쪽과 아래쪽 검색 영역 정의
        x0, y0, x1, y1 = bbox
        
        # 위쪽 검색 영역 (요소 위 100px)
        search_area_top = (x0 - 50, y0 - search_distance, x1 + 50, y0)
        
        candidate_captions = []
        
        # 텍스트 블록에서 제목 후보 찾기
        text_dict = page.get_text("dict")
        
        for block in text_dict.get("blocks", []):
            if block.get("type") != 0:  # 텍스트 블록만
                continue
            
            for line in block.get("lines", []):
                line_bbox = line.get("bbox", (0, 0, 0, 0))
                line_y = line_bbox[1]  # y 좌표
                
                # 라인의 모든 span 텍스트 합치기
                line_text = ""
                for span in line.get("spans", []):
                    line_text += span.get("text", "")
                
                line_text = line_text.strip()
                
                # 패턴 매칭
                has_pattern = any(pattern in line_text for pattern in patterns)
                
                if not has_pattern:
                    continue
                
                # 위쪽 또는 아래쪽 검색 영역에 있는지 확인
                in_top_area = (search_area_top[1] <= line_y <= search_area_top[3])
                
                if in_top_area:
                    # 요소와의 거리 계산
                    if in_top_area:
                        distance = y0 - line_y
                    
                    candidate_captions.append({
                        "text": line_text,
                        "distance": abs(distance),
                        "position": "top"
                    })
        
        # 가장 가까운 제목 선택
        if candidate_captions:
            # 거리순 정렬
            candidate_captions.sort(key=lambda x: x["distance"])
            return candidate_captions[0]["text"]
        
        return ""

    def extract_text_blocks(self, page_num: int) -> List[Dict]:
        """
        특정 페이지의 텍스트 블록 추출 (섹션 구분 없이 전체 텍스트)
        
        Args:
            page_num: 페이지 번호 (0부터 시작)
        
        Returns:
            텍스트 블록 정보 리스트
        """
        if page_num >= self.total_pages:
            raise ValueError(f"페이지 번호가 범위를 벗어났습니다: {page_num}")
        
        page = self.doc[page_num]
        blocks = []
        
        # 딕셔너리 형태로 텍스트 추출
        text_dict = page.get_text("dict")
        
        for block in text_dict.get("blocks", []):
            # 이미지 블록은 건너뛰기
            if block.get("type") != 0:
                continue
            
            for line in block.get("lines", []):
                for span in line.get("spans", []):
                    text = span.get("text", "").strip()
                    if not text:
                        continue
                    
                    bbox = span.get("bbox", (0, 0, 0, 0))
                    font_size = span.get("size", 0)
                    font_name = span.get("font", "")
                    
                    blocks.append({
                        "text": text,
                        "bbox": bbox,
                        "font_size": font_size,
                        "font_name": font_name,
                        "page_num": page_num,
                        "institution": self.institution
                    })
        
        return blocks
    
    def extract_images(self, page_num: int) -> List[Dict]:
        """
        특정 페이지의 이미지 추출 및 저장
        KHI는 이미지 추출 제외
        
        Args:
            page_num: 페이지 번호 (0부터 시작)
        
        Returns:
            이미지 정보 리스트
        """
        # KHI는 이미지 추출 안 함
        if self.institution == "khi":
            return []
        
        if page_num >= self.total_pages:
            raise ValueError(f"페이지 번호가 범위를 벗어났습니다: {page_num}")
        
        page = self.doc[page_num]
        images_info = []
        
        # 페이지의 모든 이미지 가져오기
        image_list = page.get_images(full=True)
        
        for img_index, img in enumerate(image_list, start=1):
            xref = img[0]
            
            try:
                base_image = self.doc.extract_image(xref)
                image_bytes = base_image["image"]
                image_ext = base_image["ext"]
                
                # 이미지 파일명 생성 (PDF명_page_XX_img_YY.ext)
                image_filename = f"{self.pdf_name}_page_{page_num+1:02d}_img_{img_index:02d}.{image_ext}"
                image_path = self.images_dir / image_filename
                
                # 이미지 저장
                with open(image_path, "wb") as img_file:
                    img_file.write(image_bytes)
                
                # 이미지 위치 정보
                image_rects = page.get_image_rects(xref)
                bbox = image_rects[0] if image_rects else (0, 0, 0, 0)
                
                caption = ""
                if self.institution == "kb":
                    caption = self._find_caption(page, bbox, element_type="image")
                if caption:
                    print(f"    ✓ 캡션 발견: {caption}")

                images_info.append({
                    "image_path": str(image_path),
                    "image_filename": image_filename,
                    "bbox": tuple(bbox),
                    "xref": xref,
                    "page_num": page_num,
                    "institution": self.institution,
                    "caption": caption
                })
                
            except Exception as e:
                print(f"⚠ 이미지 추출 실패 (페이지 {page_num+1}, xref {xref}): {e}")
                continue
        
        return images_info
    
    def extract_tables(self, page_num: int) -> List[Dict]:
        """
        특정 페이지의 표 추출
        기관별로 약간의 파라미터 차이 적용
        
        Args:
            page_num: 페이지 번호 (0부터 시작)
        
        Returns:
            표 정보 리스트
        """
        tables_info = []
        page_str = str(page_num + 1)

        page = self.doc[page_num]
        
        # 기관별 파라미터 설정
        if self.institution == "khi":
            line_scale = 40
            accuracy_threshold = 40
        else:  # hd, kb, unknown
            line_scale = 40
            accuracy_threshold = 50
        
        try:
            tables = camelot.read_pdf(
                self.pdf_path,
                pages=page_str,
                flavor='lattice',
                line_scale=line_scale
            )
            
            for idx, table in enumerate(tables, start=1):
                # 정확도 필터링
                if table.accuracy < accuracy_threshold:
                    continue
                
                # 표 ID 생성 (PDF명_T페이지_번호_lattice)
                table_id = f"{self.pdf_name}_T{page_num+1:02d}_{idx:02d}_lattice"
                csv_path = self.tables_dir / f"{table_id}.csv"
                table.to_csv(str(csv_path))

                caption = ""
                if self.institution == "kb":
                    # Camelot 표의 bbox 정보 (x1, y1, x2, y2)
                    # PyMuPDF는 (x0, y0, x1, y1) 형식이므로 변환 필요
                    table_bbox = table._bbox
                    # Camelot bbox를 PyMuPDF 형식으로 변환
                    # Camelot: (x1, y1, x2, y2) - 좌하단, 우상단
                    # PyMuPDF: (x0, y0, x1, y1) - 좌상단, 우하단
                    # PDF 좌표계는 아래가 원점이므로 y 좌표 변환 필요
                    page_height = page.rect.height
                    pymupdf_bbox = (
                        table_bbox[0],  # x0
                        page_height - table_bbox[3],  # y0 (상단)
                        table_bbox[2],  # x1
                        page_height - table_bbox[1]   # y1 (하단)
                    )
                    
                    caption = self._find_caption(page, pymupdf_bbox, element_type="table")
                    if caption:
                        print(f"    ✓ 표 캡션 발견: {caption}")
                    
                tables_info.append({
                    "table_id": table_id,
                    "dataframe": table.df,
                    "accuracy": table.accuracy,
                    "method": "lattice",
                    "page_num": page_num,
                    "csv_path": str(csv_path),
                    "institution": self.institution,
                    "caption": caption 
                })
                
        except Exception as e:
            print(f"⚠ 표 추출 실패 (페이지 {page_num+1}): {e}")
        
        return tables_info
    
    def analyze_layout(self, page_num: int) -> Dict:
        """
        페이지의 전체 레이아웃 분석
        
        Args:
            page_num: 페이지 번호 (0부터 시작)
        
        Returns:
            페이지 레이아웃 정보
        """
        text_blocks = self.extract_text_blocks(page_num)
        images = self.extract_images(page_num)
        tables = self.extract_tables(page_num)
        
        # 모든 요소를 y 좌표 순서대로 정렬
        elements = []
        
        # 텍스트 블록 추가
        for block in text_blocks:
            elements.append({
                "type": "text",
                "content": block["text"],
                "bbox": block["bbox"],
                "y_position": block["bbox"][1],
                "font_size": block["font_size"]
            })
        
        # 이미지 추가
        for img in images:
            elements.append({
                "type": "image",
                "content": img["image_filename"],
                "bbox": img["bbox"],
                "y_position": img["bbox"][1],
                "image_path": img["image_path"]
            })
        
        # 표 추가
        for table in tables:
            elements.append({
                "type": "table",
                "content": table["table_id"],
                "bbox": (0, 0, 0, 0),
                "y_position": 0,
                "table_id": table["table_id"],
                "accuracy": table["accuracy"]
            })
        
        # y 좌표로 정렬
        elements.sort(key=lambda x: x["y_position"])
        
        layout = {
            "page": page_num,
            "institution": self.institution,
            "total_elements": len(elements),
            "text_blocks": len([e for e in elements if e["type"] == "text"]),
            "images": len([e for e in elements if e["type"] == "image"]),
            "tables": len([e for e in elements if e["type"] == "table"]),
            "elements": elements
        }
        
        return layout
    
    def process_entire_document(self) -> Dict:
        """
        전체 문서 처리
        
        Returns:
            전체 문서 처리 결과
        """
        all_pages = []
        
        print(f"\n{'='*60}")
        print(f"PDF 문서 전체 처리 시작: {self.total_pages}페이지")
        print(f"기관: {self.institution.upper()}")
        print(f"{'='*60}\n")
        
        for page_num in range(self.total_pages):
            print(f"[페이지 {page_num+1}/{self.total_pages}] 처리 중...")
            
            try:
                text_blocks = self.extract_text_blocks(page_num)
                images = self.extract_images(page_num)
                tables = self.extract_tables(page_num)
                layout = self.analyze_layout(page_num)
                
                page_data = {
                    "page_num": page_num,
                    "institution": self.institution,
                    "text_blocks": text_blocks,
                    "images": images,
                    "tables": tables,
                    "layout": layout
                }
                
                all_pages.append(page_data)
                
                print(f"  ✓ 텍스트 블록: {len(text_blocks)}개")
                print(f"  ✓ 이미지: {len(images)}개")
                print(f"  ✓ 표: {len(tables)}개")
                
            except Exception as e:
                print(f"  ✗ 오류 발생: {e}")
                continue
        
        result = {
            "pdf_path": self.pdf_path,
            "pdf_name": self.pdf_name,
            "institution": self.institution,
            "total_pages": self.total_pages,
            "pages": all_pages
        }
        
        # 결과를 JSON으로 저장
        result_path = self.output_dir / "parsed_document.json"
        self._save_result(result, str(result_path))
        
        print(f"\n{'='*60}")
        print(f"✓ 전체 문서 처리 완료!")
        print(f"✓ 결과 저장: {result_path}")
        print(f"{'='*60}\n")
        
        return result
    
    def _save_result(self, result: Dict, output_path: str):
        """
        처리 결과를 JSON 파일로 저장 (DataFrame 제외)
        
        Args:
            result: 저장할 결과 딕셔너리
            output_path: 저장 경로
        """
        # DataFrame은 JSON으로 저장할 수 없으므로 제외
        result_copy = result.copy()
        for page in result_copy.get("pages", []):
            for table in page.get("tables", []):
                if "dataframe" in table:
                    # DataFrame을 리스트로 변환하여 저장
                    table["data"] = table["dataframe"].values.tolist()
                    table["columns"] = table["dataframe"].columns.tolist()
                    del table["dataframe"]
        
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(result_copy, f, ensure_ascii=False, indent=2)
    
    def close(self):
        """PDF 문서 닫기"""
        if self.doc:
            self.doc.close()
            print("✓ PDF 문서 닫기 완료")