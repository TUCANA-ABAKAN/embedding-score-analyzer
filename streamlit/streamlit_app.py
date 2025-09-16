import streamlit as st
import requests
from bs4 import BeautifulSoup
import re
import pandas as pd
import io
import nltk
import textwrap
from textwrap import shorten
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
import plotly.graph_objects as go

# nltk punkt 경로 수동 지정
# nltk.download('punkt')
nltk.data.path.append("./.nltk_data")

# ---------- 텍스트 전처리 ----------
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
    import os, pickle
    from nltk.tokenize.punkt import PunktSentenceTokenizer

    punkt_path = os.path.join('streamlit', '.nltk_data', 'tokenizers', 'punkt', 'english.pickle')
    try:
        with open(punkt_path, 'rb') as f:
            tokenizer = pickle.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"❌ Couldn't find punkt tokenizer at: {punkt_path}")

    sentences = tokenizer.tokenize(text)
    chunks, current = [], ""
    for s in sentences:
        if len(current) + len(s) < chunk_size:
            current += " " + s
        else:
            if current.strip():
                chunks.append(current.strip())
            current = s
    if current.strip():
        chunks.append(current.strip())
    return chunks

# ---------- 시각화 함수들 ----------
def draw_chart(df, threshold):
    fig, ax = plt.subplots(figsize=(10, min(0.6 * len(df), 10)))
    fig.patch.set_facecolor('#0e1117'); ax.set_facecolor('#0e1117')
    fig.suptitle(
        f"Similarity Score by Question ({df[df['status']=='Covered'].shape[0]}/{len(df)})",
        fontsize=14, color='white'
    )
    colors_bar = ['#00e676' if s=='Covered' else '#757575' for s in df['status']]
    bars = ax.barh(df["question"], df["similarity"], color=colors_bar)
    for i, bar in enumerate(bars):
        score = df["similarity"].iloc[i]
        ax.text(bar.get_width()+0.01, bar.get_y()+bar.get_height()/2,
                f"{score:.2f}", va='center', fontsize=9, color='white')
    ax.axvline(threshold, color='red', linestyle='--', label=f'Threshold = {threshold:.2f}')
    ax.set_xlabel("Similarity Score", color='white')
    ax.set_ylabel("Question", color='white')
    ax.tick_params(colors='white')
    ax.invert_yaxis()
    ax.legend(facecolor='#0e1117', edgecolor='white', labelcolor='white')
    st.pyplot(fig)

def draw_pie_chart(df):
    covered = df[df['status']=='Covered'].shape[0]
    not_covered = df[df['status']=='Not Covered'].shape[0]
    fig, ax = plt.subplots(figsize=(1.6,1.6))
    fig.patch.set_facecolor('#0e1117'); ax.set_facecolor('#0e1117')
    wedges, texts = ax.pie(
        [covered, not_covered],
        labels=['Covered','Not Covered'],
        colors=['#00e676','#757575'],
        startangle=90,
        wedgeprops=dict(width=0.45, edgecolor='white'),
        textprops=dict(color='white', fontsize=4)
    )
    ax.text(0,0,f"{covered}/{covered+not_covered}\nCovered",
            color='white', ha='center', va='center', fontsize=4, fontweight='bold')
    ax.axis('equal')
    st.pyplot(fig)

def generate_pdf(df, model_name, threshold):
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4)
    styles = getSampleStyleSheet()
    elements = []
    elements.append(Paragraph("Embedding Relevance Score Report", styles['Title']))
    elements.append(Spacer(1,12))
    summary = (
        f"Model: {model_name} | "
        f"Threshold: {threshold:.2f} | "
        f"Coverage: {df[df['status']=='Covered'].shape[0]}/{df.shape[0]}"
    )
    elements.append(Paragraph(summary, styles['Normal']))
    elements.append(Spacer(1,12))

    data = [["#", "Question", "Similarity", "Status", "Top Matched Chunk"]]
    for i, row in df.iterrows():
        data.append([
            f"Q{i+1}",
            row["question"],
            f"{row['similarity']:.4f}",
            row["status"],
            row["matched_chunk"][:100] + "..."
        ])
    table = Table(data, colWidths=[30,150,60,60,200])
    table.setStyle(TableStyle([
        ('BACKGROUND',(0,0),(-1,0),colors.HexColor("#4F81BD")),
        ('TEXTCOLOR',(0,0),(-1,0),colors.white),
        ('FONTNAME',(0,0),(-1,0),'Helvetica-Bold'),
        ('FONTSIZE',(0,0),(-1,0),10),
        ('FONTSIZE',(0,1),(-1,-1),8),
        ('ALIGN',(0,0),(-1,-1),'LEFT'),
        ('VALIGN',(0,0),(-1,-1),'TOP'),
        ('GRID',(0,0),(-1,-1),0.25,colors.gray),
        ('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.whitesmoke,colors.lightgrey])
    ]))
    elements.append(table)
    doc.build(elements)
    buffer.seek(0)
    return buffer

def highlight_chunk(chunk, question):
    words = [w.strip('.,!?') for w in question.lower().split()]
    for w in sorted(set(words), key=len, reverse=True):
        if len(w)>2:
            chunk = re.sub(rf"\b({re.escape(w)})\b", r"<mark>\1</mark>", chunk, flags=re.IGNORECASE)
    return chunk

def _wrap_hovertext(s: str, width: int | None = 120) -> str:
    if not s:
        return ""
    # width가 None이면 줄바꿈하지 않고 원문 그대로(단, 줄바꿈은 <br>로)
    if width is None:
        return s.replace("\n", "<br>")
    # 지정한 너비 기준으로 단어를 잘라서 <br> 삽입 (긴 단어는 중간에서 안 자름)
    return "<br>".join(textwrap.wrap(s, width=width, break_long_words=False, replace_whitespace=False))


def draw_pca_plot_plotly(question_vectors, matched_chunk_vectors, questions, top1_chunks, wrap_width=120, show_full_hover=False):
    # PCA 축소
    pca = PCA(n_components=2)
    all_vectors = np.vstack([question_vectors, matched_chunk_vectors])
    reduced = pca.fit_transform(all_vectors)
    question_points = reduced[: len(question_vectors)]
    chunk_points = reduced[len(question_vectors):]

    # 호버텍스트 처리: show_full_hover=True면 줄바꿈 없이 전체, 아니면 wrap_width 기준으로
    q_hover = [
        _wrap_hovertext(q, width=None if show_full_hover else wrap_width) for q in questions
    ]
    a_hover = [
        _wrap_hovertext(a, width=None if show_full_hover else wrap_width) for a in top1_chunks
    ]

    fig = go.Figure()

    # 질문 점 (파란)
    fig.add_trace(
        go.Scatter(
            x=question_points[:, 0],
            y=question_points[:, 1],
            mode="markers+text",
            marker=dict(size=12, color="#29b6f6"),
            text=[f"Q{i+1}" for i in range(len(questions))],
            textposition="top center",
            hovertemplate="%{hovertext}<extra></extra>",
            hovertext=q_hover,
            name="Questions",
            hoverlabel=dict(align="left", font=dict(size=12)),
        )
    )

    # 답변 점 (연두)
    fig.add_trace(
        go.Scatter(
            x=chunk_points[:, 0],
            y=chunk_points[:, 1],
            mode="markers+text",
            marker=dict(size=12, color="#76ff03"),
            text=[f"A{i+1}" for i in range(len(top1_chunks))],
            textposition="bottom center",
            hovertemplate="%{hovertext}<extra></extra>",
            hovertext=a_hover,
            name="Answer Chunks",
            hoverlabel=dict(align="left", font=dict(size=12)),
        )
    )

    # Q-A 연결선 (Top-1)
    for i in range(len(questions)):
        fig.add_trace(
            go.Scatter(
                x=[question_points[i, 0], chunk_points[i, 0]],
                y=[question_points[i, 1], chunk_points[i, 1]],
                mode="lines",
                line=dict(color="yellow", dash="dash"),
                hoverinfo="none",
                showlegend=False,
            )
        )

    fig.update_layout(
        title="🧭 PCA: Hover to Highlight Q & A",
        template="plotly_dark",
        hovermode="closest",
        xaxis_title="PCA 1",
        yaxis_title="PCA 2",
        height=600,
        legend=dict(bgcolor="rgba(0,0,0,0)"),
    )

    st.plotly_chart(fig, use_container_width=True)

# ---------- Streamlit UI ----------
st.set_page_config(page_title="Embedding Relevance Score Analyzer", layout="wide")
st.title("🔍 Embedding Relevance Score Analyzer v0.5")

# 다크모드 하이라이팅 CSS
st.markdown("""
<style>
html, body, mark { background-color:#FFD54F!important; color:black!important; padding:0 2px; border-radius:2px; }
[data-testid="stMarkdownContainer"] mark { background-color:#FFD54F!important; color:black!important; }
</style>
""", unsafe_allow_html=True)

model_options = {
    "all-MiniLM-L6-v2":"Fast & lightweight",
    "all-mpnet-base-v2":"High accuracy",
    "BAAI/bge-base-en-v1.5":"Strong semantic understanding",
    "paraphrase-MiniLM-L6-v2":"Optimized for paraphrase",
    "all-distilroberta-v1":"RoBERTa-based"
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
        <style>
        .model-desc-table {font-size:10px; width:100%; border-collapse:collapse; margin-top:4px;}
        .model-desc-table th, .model-desc-table td {padding:6px 8px; border:1px solid #444;}
        .model-desc-table th {background:#1f2a44; color:#fff; font-weight:600; text-align:left;}
        .model-desc-table tr:nth-child(even){background:#1e2330;}
        .model-desc-table tr:hover {background:rgba(255,255,255,0.04);}
        </style>
        <table class="model-desc-table">
            <thead>
                <tr>
                    <th>모델명</th>
                    <th>특징</th>
                    <th>속도</th>
                    <th>정확도</th>
                    <th>적합한 용도</th>
                </tr>
            </thead>
            <tbody>
                <tr>
                    <td><strong>all-MiniLM-L6-v2</strong></td>
                    <td>빠르고 가볍고 무난</td>
                    <td>🟢 매우 빠름</td>
                    <td>⚪ 중간</td>
                    <td>실시간 분석</td>
                </tr>
                <tr>
                    <td><strong>all-mpnet-base-v2</strong></td>
                    <td>높은 정확도</td>
                    <td>🔶 느림</td>
                    <td>🟢 매우 높음</td>
                    <td>품질 위주 분석</td>
                </tr>
                <tr>
                    <td><strong>BAAI/bge-base-en-v1.5</strong></td>
                    <td>GenAI 검색 최적화</td>
                    <td>🟡 보통</td>
                    <td>🟢 높음</td>
                    <td>GPT/RAG 검색</td>
                </tr>
                <tr>
                    <td><strong>paraphrase-MiniLM-L6-v2</strong></td>
                    <td>패러프레이즈 특화</td>
                    <td>🟢 빠름</td>
                    <td>⚪ 중간</td>
                    <td>의미 유사성 파악</td>
                </tr>
                <tr>
                    <td><strong>all-distilroberta-v1</strong></td>
                    <td>RoBERTa 기반</td>
                    <td>🟡 보통</td>
                    <td>⚪ 중간</td>
                    <td>RoBERTa 결과 비교</td>
                </tr>
            </tbody>
        </table>
        """, unsafe_allow_html=True)

    threshold = st.slider("📊 Similarity Threshold", 0.6, 0.9, 0.75, step=0.01)
    chunk_size = st.slider("🧩 Chunk Size", 300, 1000, 500, step=100)
    top_n = st.slider("🔝 Top N Chunks", 1, 5, 3)

# 1) 질문 입력
st.header("Step 1: Enter Questions")
questions_input = st.text_area("One question per line", height=200)

# 2) URL 입력
st.header("Step 2: Enter Target URL")
url = st.text_input("Enter target URL")

if st.button("🚀 Run Analysis"):
    if not questions_input or not url:
        st.warning("Please provide both questions and a URL.")
    else:
        with st.spinner("Running analysis..."):
            # 1) 사용자 입력 파싱 및 텍스트 가져오기
            questions = [q.strip() for q in questions_input.split('\n') if q.strip()]
            model = SentenceTransformer(model_name)
            full_text = clean_text_from_url(url)
            chunks = chunk_text_by_sentence(full_text, chunk_size)

            # 2) 임베딩 및 유사도 계산
            chunk_vectors = model.encode(chunks)
            question_vectors = model.encode(questions)
            sim_matrix = cosine_similarity(question_vectors, chunk_vectors)

            # 3) 결과 정리
            results = []
            for i, q in enumerate(questions):
                idxs = sim_matrix[i].argsort()[::-1][:top_n]
                scores = sim_matrix[i][idxs]
                top_chunks = [chunks[j] for j in idxs]
                match_text = ""
                for rank, (s, c) in enumerate(zip(scores, top_chunks), 1):
                    match_text += f"🔹 Top {rank} (score={s:.4f})\n{c[:300]}...\n\n"
                status = 'Covered' if scores[0] >= threshold else 'Not Covered'
                results.append({
                    "question": q,
                    "similarity": round(float(scores[0]), 4),
                    "status": status,
                    "matched_chunk": match_text.strip()
                })
            df_result = pd.DataFrame(results)

        # ✅ 분석 완료 메시지 (Run Analysis 바로 아래)
        st.success("✅ Analysis Complete")
    
        # ── ChatGPT 붙여넣기용 요약 (키워드, URL 포함) ──
        def build_chatgpt_summary(df_result, questions, top1_chunks, model_name, threshold, url, keywords=None):
            total = len(df_result)
            covered = df_result[df_result['status']=='Covered'].shape[0]
            lines = []
            # 메타 정보
            lines.append(f"URL: {url}")
            if keywords:
                lines.append(f"Keywords: {', '.join(keywords)}")
            lines.append(f"전체 커버리지: {covered}/{total} ({covered/total*100:.1f}%)")
            lines.append(f"모델: {model_name} | Threshold: {threshold:.2f}")
            lines.append("\n=== 질문별 디테일 ===")
            for i, row in enumerate(df_result.itertuples()):
                q = row.question
                status = row.status
                sim = row.similarity
                chunk_summary = top1_chunks[i].replace("\n", " ")
                if len(chunk_summary) > 200:
                    chunk_summary = chunk_summary[:197] + "..."
                lines.append(f"{i+1}. 질문: \"{q}\"")
                lines.append(f"   상태: {status}")
                lines.append(f"   similarity: {sim:.4f}")
                lines.append(f"   Top-1 청크 요약: \"{chunk_summary}\"")
                if status != 'Covered':
                    lines.append(f"   개선 제안: 콘텐츠 추가 필요\n")
            not_covered_idxs = [i+1 for i, r in enumerate(df_result.itertuples()) if r.status!='Covered']
            if not_covered_idxs:
                lines.append("\n=== 커버리지 갭 질문 번호 ===")
                lines.append(f"Not Covered 질문: {', '.join(map(str, not_covered_idxs))}")
            # 최종 출력
            return "\n".join(lines)

        # ─── expander 전 공백 ───
        st.write("")
        st.write("")

        # ─── 청크 분할 결과 접기/펼치기 ───
        st.subheader("🧩 Text Chunks Preview")
        with st.expander("🔍 View Text Chunks", expanded=False):
            for i, chunk in enumerate(chunks):
                st.markdown(f"<b>Chunk {i+1}</b>", unsafe_allow_html=True)
                st.markdown(
                    f"<div style='background:#1e1e1e; padding:10px; border-radius:5px; color:white; font-size:13px'>{chunk}</div>",
                    unsafe_allow_html=True
                )
                st.markdown("---")

        # ─── 전체 텍스트 보기 ───
        st.subheader("📄 Full Text from URL")
        with st.expander("🔍 View full raw text", expanded=False):
            st.markdown(
                f"<div style='background:#0e1117; padding:10px; color:white'>{full_text}</div>",
                unsafe_allow_html=True
            )
            
        # === Top Matched Chunks (가독성 향상된 스타일 적용) ===
        st.subheader("🔍 Top Matched Chunks")

        df_display = df_result[["question", "similarity", "status", "matched_chunk"]].reset_index(drop=True)

        def highlight_border(row):
            # 질문 열 왼쪽에 색 바만 넣음 (전체 행 배경은 안 깔음)
            if row["status"] == "Covered":
                border = "4px solid rgba(0, 230, 118, 0.8)"  # 연두색
            else:
                border = "4px solid rgba(255, 107, 107, 0.8)"  # 주황/빨강
            styles = []
            for col in row.index:
                if col == "question":
                    styles.append(f"border-left: {border}; padding-left:6px;")
                else:
                    styles.append("")  # 나머지는 기본
            return styles

        def style_status_cell(val):
            if val == "Covered":
                return "background-color: #00e676; color: black; font-weight: bold; border-radius: 4px; padding: 4px 8px;"
            else:
                return "background-color: #ff6b6b; color: white; font-weight: bold; border-radius: 4px; padding: 4px 8px;"

        styled = (
            df_display.style
                .apply(highlight_border, axis=1)
                .map(style_status_cell, subset=["status"])
                .format({"similarity": "{:.4f}"})
                .set_properties(**{"color": "#f0f0f0"})  # 기본 글자색: 밝은 회색/흰색
        )

        # 인덱스 숨기기 (버전 따라 hide_index 없을 수 있어서 예외 처리)
        try:
            styled = styled.hide_index()
        except AttributeError:
            styled = styled.set_table_styles([
                {"selector": "th.row_heading, td.row_heading", "props": [("display", "none")]}
            ], overwrite=False)

        st.dataframe(styled, use_container_width=True)

        # 커버리지 시각화
        st.subheader("📊 Coverage Visualization")
        draw_chart(df_result, threshold)
        draw_pie_chart(df_result)

        # Top-1 청크만 추출
        top1_chunks = []
        for row in df_result.itertuples():
            m = re.search(r"🔹 Top 1 \(score=.*?\)\n(.+?)(?=\n🔹 Top|\Z)", row.matched_chunk, re.DOTALL)
            top1_chunks.append(m.group(1).strip() if m else "")

        matched_chunk_vectors = model.encode(top1_chunks)

        # ▶️ Plotly hover PCA 시각화
        st.subheader("🧭 PCA: Hover to Highlight Q & A")
        draw_pca_plot_plotly(question_vectors, matched_chunk_vectors, questions, top1_chunks)

        # === Q/A 요약 테이블 (Label Reference Table) ===
        pca_table = pd.DataFrame([
            {"Q Label": f"Q{i+1}", "Q Text": row.question,
            "A Label": f"A{i+1}", "A Text": top1_chunks[i]}
            for i, row in enumerate(df_result.itertuples())
        ])

        st.subheader("📋 Label Reference Table")

        def highlight_qa_labels(row):
            styles = []
            for col in row.index:
                if col == "Q Label":
                    styles.append("background-color: rgba(3,169,244,0.2); color: white; font-weight: bold; padding:4px; border-radius:3px;")
                elif col == "A Label":
                    styles.append("background-color: rgba(0,230,118,0.2); color: black; font-weight: bold; padding:4px; border-radius:3px;")
                else:
                    styles.append("")  # 나머지 컬럼은 기본
            return styles

        styled_pca_table = (
            pca_table.style
                .apply(highlight_qa_labels, axis=1)
                .format({"Q Label": "{}", "A Label": "{}", "Q Text": "{}", "A Text": "{}"})
        )

        # 인덱스 숨기기 (버전 따라 hide_index가 없을 수 있으니 안전하게 처리)
        try:
            styled_pca_table = styled_pca_table.hide_index()
        except AttributeError:
            styled_pca_table = styled_pca_table.set_table_styles([
                {"selector": "th.row_heading, td.row_heading", "props": [("display", "none")]}
            ], overwrite=False)

        st.dataframe(styled_pca_table, use_container_width=True)

        # Highlighted Chunk Preview
        st.subheader("💬 Highlighted Chunk Preview")
        for i, row in df_result.iterrows():
            st.markdown(f"**Q{i+1}: {row.question}**", unsafe_allow_html=True)
            hl = highlight_chunk(row.matched_chunk, row.question)
            hl = re.sub(r"(score=)(\d\.\d+)",
                        r"<span style='color:#FF4B4B;font-weight:bold;'>\1\2</span>", hl)
            st.markdown(f"<div style='background:#0e1117; padding:10px; color:white'>{hl}</div>",
                        unsafe_allow_html=True)
            st.markdown("---")

        # Export Results
        #st.subheader("📥 Export Results")
        #csv_buf = io.StringIO(); df_result.to_csv(csv_buf, index=False, encoding='utf-8-sig')
        #pdf_buf = generate_pdf(df_result, model_name, threshold)
        #st.markdown("""
        #<style>
        #.button-row{display:flex;gap:10px;margin-bottom:1rem;}
        #.button-row .stButton>button{background:#1f77b4;color:white;padding:0.4em 1.2em;border-radius:6px;border:none;}
        #</style>
        #""", unsafe_allow_html=True)
        #st.markdown('<div class="button-row">', unsafe_allow_html=True)
        #c1, c2 = st.columns([1,1])
        #with c1:
            #st.download_button("Download CSV", csv_buf.getvalue(), "embedding_analysis_results.csv", "text/csv")
        #with c2:
            #st.download_button("Download PDF", pdf_buf, "embedding_analysis_results.pdf", "application/pdf")
        #st.markdown('</div>', unsafe_allow_html=True)

        # 사용 예시
        summary_text = build_chatgpt_summary(
            df_result, questions, top1_chunks,
            model_name, threshold,
            url=url,
            keywords=[kw.strip() for kw in questions_input.split('\n')]
        )
        st.subheader("📥 ChatGPT 붙여넣기용 요약")
        st.code(summary_text, language="text")
