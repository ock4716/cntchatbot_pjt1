# 🏛️ 부동산 리포트 RAG 챗봇

부동산 보고서를 기반으로 건설사 실무진에게 정확한 정보를 제공하는 AI 챗봇입니다.

![부동산 보고서 AI 챗봇 시연 영상](https://github.com/user-attachments/assets/9398295f-2a55-460c-9699-9a8f6a73ff7b)

## ✨ 주요 기능

- **하이브리드 검색**: FAISS 벡터 검색 + BM25 키워드 검색으로 정확한 정보 검색
- **멀티모달 처리**: 텍스트, 표, 그래프/이미지 통합 분석
- **시각화 지원**: 표와 차트로 데이터 시각화 (막대, 선, 파이 그래프)
- **대화 컨텍스트**: 이전 대화 내용을 기억하는 자연스러운 대화
- **출처 추적**: 모든 답변에 명확한 출처 표기

## 🚀 빠른 시작

### 1. 환경 설정

```bash
# 가상환경 생성 및 활성화
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 패키지 설치
pip install -r requirements.txt
```

### 2. API 키 설정

`.env` 파일 생성:
```
OPENAI_API_KEY=your_api_key_here
```

### 3. 챗봇 실행

```bash
streamlit run app.py
```

## 📁 프로젝트 구조

```
rag_project/
├── src/
│   ├── s1_document_processor.py    # PDF 파싱 및 추출
│   ├── s2_table_processor.py       # 표 → 자연어 변환
│   ├── s3_image_analyzer.py        # GPT-4V 이미지 분석
│   ├── s4_chunking_strategy.py     # 문서 청킹
│   ├── s5_embedding_manager.py     # 임베딩 및 FAISS
│   ├── s6_search_engine.py         # 하이브리드 검색
│   └── s8_qa_system_integrated.py  # Q&A + 시각화
├── data/
│   ├── raw/              # 원본 PDF
│   ├── processed/        # 처리된 데이터
│   ├── cache/            # 임베딩/분석 캐시
│   └── vector_store/     # FAISS 인덱스
├── app.py               # Streamlit 앱
└── README.md
```

## 🔧 RAG 파이프라인

1. **PDF 파싱** → PyMuPDF + Camelot으로 텍스트/표/이미지 추출
2. **표 처리** → GPT-4로 표를 자연어로 변환
3. **이미지 분석** → GPT-4V로 그래프 설명 생성
4. **청킹** → 토큰 기반 문서 분할 (1000토큰, 300 오버랩)
5. **임베딩** → OpenAI text-embedding-3-large
6. **검색** → FAISS + BM25 하이브리드 (RRF 융합)
7. **생성** → GPT-4o로 답변 생성 (텍스트 + JSON 시각화)

## 💡 사용 예시

**질문 예시:**
- "2024년 서울 아파트 가격 변동률은?"
- "지역별 주택 가격을 표로 정리해줘"
- "서울과 지방의 가격 추이를 그래프로 보여줘"

**시각화 요청:**
- `표로 정리해줘` → 테이블 생성
- `막대그래프로` → 막대 차트
- `추이를 선그래프로` → 선 그래프
- `비율을 원그래프로` → 파이 차트

## 🛠️ 주요 기술 스택

- **LLM**: OpenAI GPT-4o, GPT-4V
- **임베딩**: text-embedding-3-large
- **벡터 DB**: FAISS
- **키워드 검색**: BM25 (rank-bm25)
- **PDF 처리**: PyMuPDF, Camelot
- **프론트엔드**: Streamlit
- **시각화**: Matplotlib

## ⚙️ 설정

**Streamlit 사이드바에서 조정 가능:**
- **검색 민감도** (Temperature): 0.0~1.0
- **참고할 페이지 수** (Top-k): 1~10
- **대화 컨텍스트 사용**: ON/OFF

## 📝 라이선스

이 프로젝트는 교육용으로 제작되었습니다.
