import streamlit as st
import os
from dotenv import load_dotenv

# .env 파일에서 환경 변수 로드
load_dotenv()

# 페이지 설정
st.set_page_config(
    page_title="부동산 리포트 Q&A AI",
    page_icon="🏛️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 커스텀 CSS
st.markdown("""
<style>
    /* 기본 배경색 (전체 앱) */
    .stApp {
        background-color: white;
        color: #000000;
    }
    
    /* 사이드바 스타일 (진한 청록색) */
    [data-testid="stSidebar"] {
        background-color: #0e7490;
    }
    
    /* 사이드바 내부 버튼 스타일 */
    [data-testid="stSidebar"] .stButton button {
        background-color: #164e63;
        color: white;
        border: none;
        width: 100%;
        text-align: left;
        padding: 12px;
        margin: 5px 0;
        border-radius: 5px;
        box-shadow: none; 
    }
    
    [data-testid="stSidebar"] .stButton button:hover {
        background-color: #0e7490; 
    }
    
    /* 챗봇 메시지 스타일 */
    .bot-message {
        background-color: #e0f7fa;
        color: #000000 !important;
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
        border-left: 5px solid #0e7490; 
    }
    
    /* 사용자 메시지 스타일 */
    .user-message {
        background-color: #f0f0f0;
        color: #000000 !important;
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
        text-align: right;
    }
    
    /* 참고자료 박스 */
    .reference-box {
        background-color: #fefce8; 
        color: #000000 !important;
        padding: 15px;
        border-radius: 8px;
        margin: 10px 0;
        border-left: 4px solid #eab308;
    }
    
    /* 사이드바 헤더 스타일 */
    .header-icon {
        display: flex;
        align-items: center;
        gap: 10px;
        color: white;
        font-size: 1.5em;
        font-weight: bold;
        padding: 10px 15px; 
    }
    
    /* 설정 슬라이더 캡션 */
    [data-testid="stSidebar"] div.stCaption {
        color: #e0f7fa !important; 
    }
    
    /* 일반 텍스트 (사이드바) */
    [data-testid="stSidebar"] h3, [data-testid="stSidebar"] label {
        color: white !important;
    }

    /* 메인 영역 제목/부제목 */
    h1, h2, h3, h4, h5, h6 {
        color: #000000 !important;
    }
    
    /* 채팅 입력창 */
    [data-testid="stChatInput"] input,
    [data-testid="stChatInput"] textarea {
        color: #000000 !important;
        background-color: white !important;
    }
    
    /* info 박스 */
    .stAlert.info {
        background-color: #f0f0f0; 
        border-left-color: #0e7490;
        color: #000000 !important;
    }
    
    /* 탭 메뉴 스타일 */
    .stTabs [data-testid="stTab"] {
        color: #000000 !important;
        background-color: transparent !important;
    }

    /* 선택된 탭의 밑줄 색상 (빨간색) */
    .stTabs [data-testid="stTab"][aria-selected="true"] {
        border-bottom: 2px solid red !important;
        color: red !important;
    }

    .stTabs [data-testid="stTab"][aria-selected="false"] {
        border-bottom: 2px solid transparent !important;
    }

    /* 기타 캡션/작은 글씨 */
    .stCaption, .stMarkdown small, .stMarkdown p {
        color: #000000 !important;
    }
    
</style>
""", unsafe_allow_html=True)

# 세션 상태 초기화
if 'messages' not in st.session_state:
    st.session_state.messages = []

if 'current_section' not in st.session_state:
    st.session_state.current_section = "서울 아파트 주간 시황"

if 'references' not in st.session_state:
    st.session_state.references = []

if 'qa_system' not in st.session_state:
    st.session_state.qa_system = None

if 'search_engine' not in st.session_state:
    st.session_state.search_engine = None

if 'user_questions' not in st.session_state:
    st.session_state.user_questions = []

if 'current_visualization' not in st.session_state:
    st.session_state.current_visualization = None

# .env에서 OpenAI API 키 로드 및 QA 시스템 자동 초기화
if st.session_state.qa_system is None:
    api_key = os.getenv("OPENAI_API_KEY")
    if api_key:
        try:
            from src.s8_qa_system_integrated import QASystem
            st.session_state.qa_system = QASystem(openai_api_key=api_key, model="gpt-4o")
            print("✅ QA 시스템이 .env의 API 키로 자동 초기화되었습니다.")
        except ImportError:
            st.error("⚠️ qa_system_integrated.py 파일을 찾을 수 없습니다.")
        except Exception as e:
            st.error(f"❌ QA 시스템 초기화 실패: {e}")
    else:
        st.warning("⚠️ .env 파일에 OPENAI_API_KEY가 설정되지 않았습니다.")

# Search Engine 초기화
if st.session_state.search_engine is None:
    import json
    import faiss
    from pathlib import Path
    
    try:
        # 경로 설정
        vector_store_path = Path("data/vector_store/kb")
        processed_path = Path("data/processed/kb")
        
        faiss_index_path = vector_store_path / "faiss_index.bin"
        metadata_path = vector_store_path / "metadata.json"
        chunks_path = processed_path / "kb_report_chunks.json"
        
        # 파일 로드
        faiss_index = faiss.read_index(str(faiss_index_path))
        
        with open(metadata_path, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
        
        with open(chunks_path, 'r', encoding='utf-8') as f:
            chunks = json.load(f)
        
        # EmbeddingManager 초기화
        from src.s5_embedding_manager import EmbeddingManager
        from src.s6_search_engine import SearchEngine
        
        api_key = os.getenv("OPENAI_API_KEY")
        embedding_manager = EmbeddingManager(
            openai_api_key=api_key,
            institution="kb"  
        )
        # SearchEngine 초기화
        st.session_state.search_engine = SearchEngine(
            faiss_index=faiss_index,
            metadata=metadata,
            chunks=chunks,
            embedding_manager=embedding_manager
        )
        print("✅ Search Engine이 자동으로 초기화되었습니다.")
        
    except Exception as e:
        print(f"❌ Search Engine 초기화 실패: {e}")
        st.session_state.search_engine = None

# RAG 응답 생성 함수
def generate_response(query: str, temperature: float, top_k: int, use_conversation: bool = True) -> tuple:
    """
    RAG 파이프라인 + 시각화 지원
    
    Returns:
        (응답 딕셔너리, 참고 문서 리스트)
    """
    if st.session_state.qa_system is None:
        response = {"answer_type": "text", "text_response": "⚠️ QA 시스템이 초기화되지 않았습니다.", "visualization": None}
        return response, []
    
    if st.session_state.search_engine is None:
        response = {"answer_type": "text", "text_response": "⚠️ 벡터 DB가 로드되지 않았습니다.", "visualization": None}
        return response, []
    
    try:
        # 1. 검색 수행
        search_results = st.session_state.search_engine.hybrid_search(query, top_k=top_k)
        
        # 2. QASystem으로 답변 생성 (시각화 포함)
        qa_system = st.session_state.qa_system
        
        result_dict = qa_system.answer_question(
            query=query,
            search_results=search_results,
            rewrite=False,
            use_history=use_conversation,
            temperature=temperature
        )
        
        # 3. 참고 문서 정리
        references = []
        for i, result in enumerate(search_results[:top_k], 1):
            metadata = result.get("metadata", {})
            content = result.get("content", "")
            
            institution = metadata.get("institution", "unknown")
            institution_map = {
                "hd": "HD 현대",
                "kb": "KB금융",
                "khi": "KHI 주택금융"
            }
            source_name = institution_map.get(institution, institution)
            
            doc_type_map = {
                "text": "본문",
                "table": "표",
                "image": "그래프"
            }
            doc_type = doc_type_map.get(metadata.get("doc_type"), "본문")
            
            references.append({
                "page": metadata.get("page", "N/A"),
                "text": content[:300],
                "source": f"{source_name} - {doc_type}",
                "institution": source_name
            })
        
        return result_dict, references
        
    except Exception as e:
        print(f"Error in generate_response: {e}")
        error_response = {
            "answer_type": "text",
            "text_response": f"오류가 발생했습니다: {str(e)}",
            "visualization": None
        }
        return error_response, []

# 사이드바
with st.sidebar:
    st.markdown('<div class="header-icon">🏛️ 부동산 리포트 Q&A AI</div>', unsafe_allow_html=True)
    
    # 시스템 상태 표시
    st.markdown("### 🔧 시스템 상태")
    if st.session_state.qa_system:
        st.success("✅ QA 시스템 연결됨")
    else:
        st.error("❌ QA 시스템 미연결")
    
    if st.session_state.search_engine:
        st.success("✅ 벡터 DB 연결됨")
    else:
        st.warning("⚠️ 벡터 DB 미연결")
    
    # 대화 히스토리 리셋 버튼
    if st.button("🔄 대화 초기화", use_container_width=True):
        st.session_state.messages = []
        if st.session_state.qa_system:
            st.session_state.qa_system.clear_history()
        st.session_state.current_visualization = None
        st.rerun()
    
    st.markdown("---")
    
    # 최근 물어본 질문
    st.markdown("### 💬 최근 물어본 질문")
    
    recent_questions = st.session_state.user_questions[-4:][::-1]
    
    if recent_questions:
        for idx, question in enumerate(recent_questions):
            display_question = question if len(question) <= 30 else question[:27] + "..."
            if st.button(display_question, key=f"recent_q_{idx}", use_container_width=True):
                st.session_state.selected_question = question
    else:
        st.caption("아직 질문 히스토리가 없습니다.")
    
    st.markdown("---")
    st.markdown("### ⚙️ 설정")
    
    temperature = st.slider(
        "검색 민감도 (Temperature)",
        min_value=0.0,
        max_value=1.0,
        value=0.3,
        step=0.1,
        help="낮을수록 정확하고 일관된 답변"
    )
    
    top_k = st.slider(
        "참고할 페이지 수 (Top-k)",
        min_value=1,
        max_value=10,
        value=5,
        step=1,
        help="검색할 문서 청크 수"
    )
    
    use_conversation = st.checkbox(
        "대화 컨텍스트 사용",
        value=True,
        help="이전 대화 내용을 참고하여 답변"
    )

# 메인 영역 레이아웃
col1, col2 = st.columns([2, 1])

with col1:
    st.title("💬 부동산 인사이트봇")
    
    # 초기 안내 메시지
    if len(st.session_state.messages) == 0:
        st.markdown("""
        <div class="bot-message">
            <strong>🏛️ 부동산 인사이트봇</strong><br><br>
            안녕하세요! 부동산 인사이트봇입니다. 지역(예: 서울 강남구), 거래종류
            (매매/전세), 기간(예: 최근 3개월) 등을 입력하시면 최신 동향 요약을 제
            공합니다. <br><br>
            "지역별 가격을 표로 보여줘", "그래프로 보여줘" 같은 요청도 가능합니다!
        </div>
        """, unsafe_allow_html=True)
    
    # 채팅 히스토리 표시
    for message in st.session_state.messages:
        if message["role"] == "user":
            st.markdown(f'<div class="user-message">{message["content"]}</div>', 
                       unsafe_allow_html=True)
        else:
            # 텍스트 응답
            st.markdown(f'<div class="bot-message"><strong>🏛️ 부동산 인사이트봇</strong><br><br>{message["content"]}</div>', 
                       unsafe_allow_html=True)
            
            # 참고자료 표시
            if "references" in message and message["references"]:
                with st.expander("🔍 근거 자료 및 데이터 확인"):
                    for ref in message["references"]:
                        st.markdown(f"""
                        <div class="reference-box">
                            <strong>REFERENCE TEXT (PAGE {ref['page']})</strong><br>
                            <small>출처: {ref.get('source', 'N/A')}</small><br><br>
                            "{ref['text']}"
                        </div>
                        """, unsafe_allow_html=True)
    
    # 최근 질문 버튼 클릭 처리
    if 'selected_question' in st.session_state:
        user_input = st.session_state.selected_question
        del st.session_state.selected_question
    else:
        user_input = st.chat_input("질문을 입력하세요. 예) 2024년 1분기 서울 지역별 주택 가격 변동률은?")
    
    if user_input:
        # 사용자 질문 히스토리에 추가
        st.session_state.user_questions.append(user_input)
        
        # 사용자 메시지 추가
        st.session_state.messages.append({
            "role": "user",
            "content": user_input
        })
        
        # AI 응답 생성 (시각화 포함)
        result_dict, references = generate_response(
            user_input, 
            temperature, 
            top_k,
            use_conversation
        )
        
        # 응답 저장
        st.session_state.messages.append({
            "role": "assistant",
            "content": result_dict["text_response"],
            "references": references,
            "visualization": result_dict.get("visualization")
        })
        
        # 시각화가 있으면 세션에 저장
        if result_dict.get("visualization"):
            st.session_state.current_visualization = result_dict["visualization"]
        
        st.rerun()

with col2:
    st.markdown("### 시각화 미리보기")
    
    # 탭 생성
    tab1, tab2 = st.tabs(["📊 표 보기", "📈 차트 보기"])
    
    # 최신 메시지에서 시각화 데이터 가져오기
    visualization_data = None
    if st.session_state.messages:
        last_assistant_messages = [msg for msg in st.session_state.messages if msg["role"] == "assistant"]
        if last_assistant_messages and "visualization" in last_assistant_messages[-1]:
            visualization_data = last_assistant_messages[-1]["visualization"]
    
    with tab1:
        if visualization_data and visualization_data.get("type") == "table":
            from src.s8_qa_system_integrated import VisualizationRenderer
            VisualizationRenderer.render_table_streamlit(visualization_data)
        else:
            st.info("표 데이터가 없습니다. '표로 보여줘' 같은 요청을 해보세요!")
    
    with tab2:
        if visualization_data and visualization_data.get("type") in ["bar", "barh", "line", "pie"]:
            from src.s8_qa_system_integrated import VisualizationRenderer
            VisualizationRenderer.render_chart_streamlit(visualization_data)
        else:
            st.info("차트 데이터가 없습니다. '그래프로 보여줘' 같은 요청을 해보세요!")
    
    st.markdown("---")
    st.markdown("### 출처 / 레퍼런스")
    st.caption("검색 결과에서 구성된 컨텍스트와 출처 리스트")
    
    # 최신 메시지의 레퍼런스 표시
    if st.session_state.messages:
        last_messages = [msg for msg in st.session_state.messages if msg["role"] == "assistant"]
        if last_messages and "references" in last_messages[-1]:
            references = last_messages[-1]["references"]
            if references:
                for idx, ref in enumerate(references, 1):
                    source = ref.get("source", "N/A")
                    page = ref.get("page", "N/A")
                    st.markdown(f"**[{idx}]** {source} ({page}페이지)")
            else:
                st.caption("아직 검색 결과가 없습니다.")
        else:
            st.caption("아직 검색 결과가 없습니다.")
    else:
        st.caption("아직 검색 결과가 없습니다.")

if __name__ == "__main__":
    pass