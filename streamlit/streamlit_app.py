# streamlit_app_v0_4_6.py

import streamlit as st
import requests
from bs4 import BeautifulSoup
import re
import pandas as pd
import io
import nltk
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt
from nltk.tokenize import sent_tokenize
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet

# nltk.download('punkt')
# ✅ punkt 경로 수동 지정
nltk.data.path.append("./.nltk_data")

# 텍스트 전처리
def clean_text_from_url(url):
    try:
        res = requests.get(url)
        soup = BeautifulSoup(res.text, 'html.parser')
        for tag in soup(['script', 'style', 'noscript']):
            tag.decompose()
        text = soup.get_text(separator=' ', strip=True)
        return re.sub(r'\s+', ' ', text)
    except Exception as e:
        st.error(f"❌ Failed to fetch URL:\n{e}")
        return ""

def chunk_text_by_sentence(text, chunk_size=500):
    sentences = sent_tokenize(text, language='english')
    chunks, current_chunk = [], ""
    for sentence in sentences:
        if len(current_chunk) + len(sentence) < chunk_size:
            current_chunk += " " + sentence
        else:
            chunks.append(current_chunk.strip())
            current_chunk = sentence
    if current_chunk:
        chunks.append(current_chunk.strip())
    return chunks

# ---------- 시각화 (다크모드용 개선 버전) ----------
def draw_chart(df, threshold):
    fig, ax = plt.subplots(figsize=(10, min(0.6 * len(df), 10)))
    fig.patch.set_facecolor('#0e1117')
    ax.set_facecolor('#0e1117')

    fig.suptitle(
        f"Similarity Score by Question ({df[df['status']=='Covered'].shape[0]}/{len(df)})",
        fontsize=14,
        color='white'
    )

    colors = ['#00e676' if s == 'Covered' else '#757575' for s in df['status']]
    bars = ax.barh(df["question"], df["similarity"], color=colors)

    for i, bar in enumerate(bars):
        score = df["similarity"].iloc[i]
        ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                f"{score:.2f}", va='center', fontsize=9, color='white')

    ax.axvline(threshold, color='red', linestyle='--', label=f'Threshold = {threshold:.2f}')
    ax.set_xlabel("Similarity Score", color='white')
    ax.set_ylabel("Question", color='white')
    ax.tick_params(axis='x', colors='white')
    ax.tick_params(axis='y', colors='white')
    ax.invert_yaxis()
    ax.legend(facecolor='#0e1117', edgecolor='white', labelcolor='white')
    st.pyplot(fig)

def draw_pie_chart(df):
    import matplotlib.pyplot as plt

    covered = df[df['status'] == 'Covered'].shape[0]
    not_covered = df[df['status'] == 'Not Covered'].shape[0]
    total = covered + not_covered

    # 색상 및 레이블
    labels = ['Covered', 'Not Covered']
    sizes = [covered, not_covered]
    colors = ['#00e676', '#757575']

    fig, ax = plt.subplots(figsize=(1.6, 1.6))
    fig.patch.set_facecolor('#0e1117')  # 다크모드 배경
    ax.set_facecolor('#0e1117')

    wedges, texts = ax.pie(
        sizes,
        labels=labels,
        colors=colors,
        startangle=90,
        wedgeprops=dict(width=0.45, edgecolor='white'),
        textprops=dict(color='white', fontsize=4)
    )

    # 중앙 텍스트 추가
    ax.text(0, 0, f"{covered}/{total}\nCovered", color='white',
            ha='center', va='center', fontsize=4, fontweight='bold')

    ax.axis('equal')  # 원형 유지
    st.pyplot(fig)


# ---------- PDF ----------
def generate_pdf(df, model_name, threshold):
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4)
    styles = getSampleStyleSheet()
    elements = []

    elements.append(Paragraph("Embedding Relevance Score Report", styles['Title']))
    elements.append(Spacer(1, 12))
    summary = f"Model: {model_name} | Threshold: {threshold:.2f} | Coverage: {df[df['status'] == 'Covered'].shape[0]}/{df.shape[0]}"
    elements.append(Paragraph(summary, styles['Normal']))
    elements.append(Spacer(1, 12))

    data = [["#", "Question", "Similarity", "Status", "Top Matched Chunk"]]
    for i, row in df.iterrows():
        data.append([
            f"Q{i+1}",
            row["question"],
            f"{row['similarity']:.4f}",
            row["status"],
            row["matched_chunk"][:100] + "..."
        ])

    table = Table(data, colWidths=[30, 150, 60, 60, 200])
    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#4F81BD")),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 10),
        ('FONTSIZE', (0, 1), (-1, -1), 8),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('GRID', (0, 0), (-1, -1), 0.25, colors.gray),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.whitesmoke, colors.lightgrey])
    ]))

    elements.append(table)
    doc.build(elements)
    buffer.seek(0)
    return buffer

# ---------- 하이라이팅 ----------
def highlight_chunk(chunk, question):
    words = [w.strip('.,!?') for w in question.lower().split()]
    for w in sorted(set(words), key=len, reverse=True):
        if len(w) > 2:
            chunk = re.sub(
                r"\b({})\b".format(re.escape(w)),
                r"<mark>\1</mark>",
                chunk,
                flags=re.IGNORECASE
            )
    return chunk

# ---------- Streamlit UI ----------
st.set_page_config(page_title="Embedding Relevance Score Analyzer", layout="wide")
st.title("🔍 Embedding Relevance Score Analyzer v0.5")

# ✅ 다크모드 대응 하이라이팅 스타일
st.markdown("""
<style>
html, body, mark {
    background-color: #FFD54F !important;
    color: black !important;
    padding: 0 2px;
    border-radius: 2px;
}
[data-testid="stMarkdownContainer"] mark {
    background-color: #FFD54F !important;
    color: black !important;
}
</style>
""", unsafe_allow_html=True)

model_options = {
    "all-MiniLM-L6-v2": "Fast & lightweight",
    "all-mpnet-base-v2": "High accuracy",
    "BAAI/bge-base-en-v1.5": "Strong semantic understanding",
    "paraphrase-MiniLM-L6-v2": "Optimized for paraphrase",
    "all-distilroberta-v1": "RoBERTa-based"
}

# Sidebar
with st.sidebar:
    model_name = st.selectbox(
    label="🔬 Select Model",
    options=list(model_options.keys()),
    help="ℹ️ 모델별 속도, 정확도, 특성은 아래 '📘 모델 설명 보기'에서 확인할 수 있습니다."
    )

    with st.expander("📘 모델 설명 보기"):
        st.markdown("""
    <div style='font-size:10.5px'>
        
    | 모델명 | 특징 | 속도 | 정확도 | 적합한 용도 |
    |--------|------|------|--------|--------------|
    | **all-MiniLM-L6-v2** | (Default) 빠르고 가볍고 무난 | 🟢 매우 빠름 | ⚪ 중간 | 실시간 분석 |
    | **all-mpnet-base-v2** | 높은 정확도 | 🔶 느림 | 🟢 매우 높음 | 품질 위주 분석 |
    | **BAAI/bge-base-en-v1.5** | GenAI 검색 최적화 | 🟡 보통 | 🟢 높음 | GPT/RAG 검색 |
    | **paraphrase-MiniLM-L6-v2** | 패러프레이즈 특화 | 🟢 빠름 | ⚪ 중간 | 의미 유사성 파악 |
    | **all-distilroberta-v1** | RoBERTa 기반 | 🟡 보통 | ⚪ 중간 | RoBERTa 결과 비교 |

    </div>
    """, unsafe_allow_html=True)


    threshold = st.slider("📊 Similarity Threshold", 0.6, 0.9, 0.75, step=0.01)
    chunk_size = st.slider("🧩 Chunk Size", 300, 1000, 500, step=100)
    top_n = st.slider("🔝 Top N Chunks", 1, 5, 3)

# Main inputs
st.header("Step 1: Enter Questions")
questions_input = st.text_area("One question per line", height=200)

st.header("Step 2: Enter Target URL")
url = st.text_input("Enter target URL")

if st.button("🚀 Run Analysis"):
    if not questions_input or not url:
        st.warning("Please provide both questions and a URL.")
    else:
        with st.spinner("Running analysis..."):
            questions = [q.strip() for q in questions_input.strip().split('\n') if q.strip()]
            model = SentenceTransformer(model_name)
            full_text = clean_text_from_url(url)
            chunks = chunk_text_by_sentence(full_text, chunk_size=chunk_size)

            chunk_vectors = model.encode(chunks)
            question_vectors = model.encode(questions)
            similarity_matrix = cosine_similarity(question_vectors, chunk_vectors)

            results = []
            for i, q in enumerate(questions):
                top_indices = similarity_matrix[i].argsort()[::-1][:top_n]
                top_scores = similarity_matrix[i][top_indices]
                top_chunks = [chunks[idx] for idx in top_indices]

                top_match_text = ""
                for rank, (score, chunk) in enumerate(zip(top_scores, top_chunks), 1):
                    top_match_text += f"🔹 Top {rank} (score={score:.4f})\n{chunk[:300]}...\n\n"

                status = 'Covered' if top_scores[0] >= threshold else 'Not Covered'
                results.append({
                    "question": q,
                    "similarity": round(float(top_scores[0]), 4),
                    "status": status,
                    "matched_chunk": top_match_text.strip()
                })

            df_result = pd.DataFrame(results)

        st.success("✅ Analysis Complete")

        # 표
        st.subheader("🔍 Top Matched Chunks")
        st.dataframe(df_result[["question", "similarity", "status", "matched_chunk"]], use_container_width=True)

        # 시각화
        st.subheader("📊 Coverage Visualization")
        draw_chart(df_result, threshold)
        draw_pie_chart(df_result)

                # 시각화: 막대 + 파이차트 + PCA
        def draw_chart(df, threshold):
            fig, ax = plt.subplots(figsize=(10, min(0.6 * len(df), 10)))
            fig.patch.set_facecolor('#0e1117')
            ax.set_facecolor('#0e1117')
            fig.suptitle(f"Similarity Score by Question ({df[df['status']=='Covered'].shape[0]}/{len(df)})", fontsize=14, color='white')
            colors = ['#00e676' if s == 'Covered' else '#757575' for s in df['status']]
            bars = ax.barh(df["question"], df["similarity"], color=colors)
            for i, bar in enumerate(bars):
                score = df["similarity"].iloc[i]
                ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2, f"{score:.2f}", va='center', fontsize=9, color='white')
            ax.axvline(threshold, color='red', linestyle='--', label=f'Threshold = {threshold:.2f}')
            ax.set_xlabel("Similarity Score", color='white')
            ax.set_ylabel("Question", color='white')
            ax.tick_params(axis='x', colors='white')
            ax.tick_params(axis='y', colors='white')
            ax.invert_yaxis()
            ax.legend(facecolor='#0e1117', edgecolor='white', labelcolor='white')
            st.pyplot(fig)

        def draw_pie_chart(df):
            covered = df[df['status'] == 'Covered'].shape[0]
            not_covered = df[df['status'] == 'Not Covered'].shape[0]
            fig, ax = plt.subplots(figsize=(1.6, 1.6))
            fig.patch.set_facecolor('#0e1117')
            ax.set_facecolor('#0e1117')
            wedges, texts = ax.pie([covered, not_covered], labels=['Covered', 'Not Covered'],
                                colors=['#00e676', '#757575'], startangle=90,
                                wedgeprops=dict(width=0.45, edgecolor='white'),
                                textprops=dict(color='white', fontsize=4))
            ax.text(0, 0, f"{covered}/{covered + not_covered}\nCovered", color='white', ha='center', va='center', fontsize=4, fontweight='bold')
            ax.axis('equal')
            st.pyplot(fig)

        def draw_pca_plot(question_vectors, matched_chunk_vectors):
            pca = PCA(n_components=2)
            all_vectors = np.vstack([question_vectors, matched_chunk_vectors])
            reduced = pca.fit_transform(all_vectors)
            question_points = reduced[:len(question_vectors)]
            chunk_points = reduced[len(question_vectors):]
            fig, ax = plt.subplots(figsize=(10, 6))
            fig.patch.set_facecolor('#0e1117')
            ax.set_facecolor('#0e1117')
            ax.grid(True, color='#444444', linestyle='--', linewidth=0.5)  # 👈 회색 점선으로 그리드 추가
            ax.set_title("Question & Answer Embeddings in 2D Space (PCA Reduced)", color='white')
            ax.scatter(question_points[:, 0], question_points[:, 1], color='deepskyblue', label='Question')
            ax.scatter(chunk_points[:, 0], chunk_points[:, 1], color='limegreen', label='Answer Chunk')
            for i in range(len(question_points)):
                ax.text(question_points[i, 0], question_points[i, 1], f"Q{i+1}", color='white', fontsize=9)
                ax.text(chunk_points[i, 0], chunk_points[i, 1], f"A{i+1}", color='white', fontsize=9)
            ax.set_xlabel("PCA Component 1", color='white')
            ax.set_ylabel("PCA Component 2", color='white')
            ax.tick_params(colors='white')
            ax.legend(facecolor='#0e1117', edgecolor='white', labelcolor='white')
            st.pyplot(fig)

        # 🔍 Top 1 청크만 추출 (정규식 기반)
        top1_chunks = []

        for row in df_result.itertuples():
            match = re.search(r"🔹 Top 1 \(score=.*?\)\n(.+?)(?=\n🔹 Top|\Z)", row.matched_chunk, re.DOTALL)
            if match:
                top1_chunks.append(match.group(1).strip())
            else:
                top1_chunks.append("")

        # ✅ Top 1 청크 임베딩
        matched_chunk_vectors = model.encode(top1_chunks)

        # 🎯 PCA 시각화
        draw_pca_plot(question_vectors, matched_chunk_vectors)

        # 📋 차트 하단 Q/A 요약 테이블
        pca_table_data = []
        for i, row in df_result.iterrows():
            pca_table_data.append({
                "Q Label": f"Q{i+1}",
                "Q Text": row["question"],
                "A Label": f"A{i+1}",
                "A Text": top1_chunks[i]
            })

        df_pca_table = pd.DataFrame(pca_table_data)
        st.subheader("📋 Label Reference Table")
        st.dataframe(df_pca_table, use_container_width=True)


        # PCA 라벨-내용 매핑 테이블 표시
        pca_mapping = pd.DataFrame({
            "Label": [f"Q{i+1}" for i in range(len(questions))] + [f"A{i+1}" for i in range(len(questions))],
            "Type": ["Question"] * len(questions) + ["Answer Chunk"] * len(questions),
            "Text": questions + [row.matched_chunk[:100] + "..." for row in df_result.itertuples()]
        })

        # Highlighted Chunk Preview
        st.subheader("💬 Highlighted Chunk Preview")
        for i, row in df_result.iterrows():
            st.markdown(f"**Q{i+1}: {row['question']}**", unsafe_allow_html=True)

            # 점수 강조: score=0.78xx 부분만 색상 처리
            # row["matched_chunk"]는 multiline string이므로 정규식으로 치환
            highlighted_text = highlight_chunk(row["matched_chunk"], row["question"])
            highlighted_text = re.sub(
                r"(score=)(\d\.\d+)",
                r"<span style='color:#FF4B4B; font-weight:bold;'>\1\2</span>",
                highlighted_text
            )

            st.markdown(
                f"<div style='background:#0e1117; padding:10px; border-radius:5px; font-size:14px'>{highlighted_text}</div>",
                unsafe_allow_html=True
            )
            st.markdown("---")

        # Export Results 섹션 내 버튼 스타일 커스터마이징 추가
        st.subheader("📥 Export Results")

        # CSV와 PDF 버퍼 생성
        csv_buffer = io.StringIO()
        df_result.to_csv(csv_buffer, index=False, encoding='utf-8-sig')
        pdf_buffer = generate_pdf(df_result, model_name, threshold)

        # ✅ CSS 추가: 버튼을 좌측 정렬로 나란히, 스타일 통일
        st.markdown("""
            <style>
            .button-row {
                display: flex;
                justify-content: flex-start;
                gap: 10px;
                margin-bottom: 1rem;
            }
            .button-row .stButton > button {
                background-color: #1f77b4;
                color: white;
                padding: 0.4em 1.2em;
                border-radius: 6px;
                border: none;
                font-size: 0.9em;
            }
            </style>
        """, unsafe_allow_html=True)

             # ✅ 버튼 나란히 배치
        st.markdown('<div class="button-row">', unsafe_allow_html=True)
        col1, col2 = st.columns([1, 1])
        with col1:
            st.download_button("Download CSV", csv_buffer.getvalue(), "embedding_analysis_results.csv", "text/csv", key="csv_dl")
        with col2:
            st.download_button("Download PDF", pdf_buffer, "embedding_analysis_results.pdf", "application/pdf", key="pdf_dl")
        st.markdown('</div>', unsafe_allow_html=True)
