"""
image_analyzer.py
[3단계] 그래프/이미지 분석

이 모듈은 추출된 이미지를 GPT-4V로 분석합니다.
- GPT-4V로 이미지 설명 생성
- 그래프 트렌드 분석
- 분석 결과 캐싱
"""

import os
import json
from typing import Dict
from PIL import Image
import base64
from openai import OpenAI


class ImageAnalyzer:
    """이미지와 그래프를 GPT-4V로 분석하는 클래스"""
    
    def __init__(self, openai_api_key: str, cache_path: str = "data/cache/image_descriptions.json"):
        """
        ImageAnalyzer 초기화
        
        Args:
            openai_api_key: OpenAI API 키
            cache_path: 캐시 파일 경로
        """
        self.client = OpenAI(api_key=openai_api_key)
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
    
    def resize_image(self, image_path: str, max_size: int = 1024) -> str:
        """
        이미지 리사이징 (API 비용 절감)
        
        Args:
            image_path: 이미지 경로
            max_size: 최대 크기 (픽셀)
        
        Returns:
            리사이징된 이미지 경로
        """
        img = Image.open(image_path)
        
        # 이미 작으면 그대로 반환
        if max(img.size) <= max_size:
            return image_path
        
        # 비율 유지하며 리사이징
        ratio = max_size / max(img.size)
        new_size = tuple(int(dim * ratio) for dim in img.size)
        img_resized = img.resize(new_size, Image.LANCZOS)
        
        # 임시 파일로 저장
        temp_path = image_path.replace('.', '_resized.')
        img_resized.save(temp_path)
        
        return temp_path
    
    def encode_image_to_base64(self, image_path: str) -> str:
        """
        이미지를 base64로 인코딩
        
        Args:
            image_path: 이미지 경로
        
        Returns:
            base64 인코딩된 이미지 문자열
        """
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    
    def analyze_image(self, image_path: str, caption: str = "") -> str:
        """
        이미지를 GPT-4V로 분석
        
        Args:
            image_path: 분석할 이미지 경로
            caption: 이미지 제목 (있으면 분석에 활용)
        
        Returns:
            이미지 분석 설명
        """
        # 캐시 확인
        cache_key = f"{image_path}_{caption}"
        if cache_key in self.cache:
            print(f"  ✓ 캐시에서 로드: {image_path}")
            return self.cache[cache_key]
        
        # 이미지 리사이징
        resized_path = self.resize_image(image_path)
        
        # base64 인코딩
        base64_image = self.encode_image_to_base64(resized_path)
        
        # GPT-4V API 호출
        try:
            prompt = f"""이 그래프/차트를 분석해주세요.

{f'제목: {caption}' if caption else ''}

다음 내용을 포함해주세요:
1. 그래프가 무엇을 보여주는지 (제목, 축, 범례)
2. 주요 트렌드와 패턴
3. 특이사항이나 주목할 만한 점
4. 수치 데이터 (가능한 경우)

명확하고 간결하게 한국어로 설명해주세요."""

            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": prompt
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{base64_image}"
                                }
                            }
                        ]
                    }
                ],
                max_tokens=500
            )
            
            description = response.choices[0].message.content
            
            # 캐시에 저장
            self.cache[cache_key] = description
            self.save_cache()
            
            # 리사이징된 임시 파일 삭제
            if resized_path != image_path and os.path.exists(resized_path):
                os.remove(resized_path)
            
            return description
            
        except Exception as e:
            print(f"  ✗ 이미지 분석 실패: {e}")
            return f"이미지 분석 실패: {str(e)}"
    
    def generate_graph_description(self, image_path: str, 
                                   page_num: int = 0,
                               caption: str = "") -> Dict:
        """
        그래프 설명 생성 (구조화된 형태)
        
        Args:
            image_path: 이미지 경로
            page_num: 페이지 번호
            caption: 이미지 제목 (있으면 분석에 활용)
        
        Returns:
            그래프 설명 딕셔너리
        """
        # 이미지 분석
        description = self.analyze_image(image_path, caption=caption)
        
        result = {
            "image_path": image_path,
            "page": page_num,
            "caption": caption, 
            "description": description,
            "analysis_date": None  # 실제로는 datetime 사용
        }
        
        return result
    
    def analyze_multiple_images(self, image_infos: list) -> list:
        """
        여러 이미지를 한 번에 분석
        
        Args:
            image_infos: 이미지 정보 리스트
                [{"image_path": "...", "page_num": ...}, ...]
        
        Returns:
            분석 결과 리스트
        """
        results = []
        
        for i, img_info in enumerate(image_infos, 1):
            print(f"\n[{i}/{len(image_infos)}] 이미지 분석 중: {img_info.get('image_path')}")
            
            # caption 정보가 있으면 출력
            caption = img_info.get('caption', '')
            if caption:
                print(f"  제목: {caption}")

            result = self.generate_graph_description(
                image_path=img_info.get('image_path'),
                page_num=img_info.get('page_num', 0),
                caption=caption 
            )
            
            results.append(result)
        
        return results