import nltk
nltk.data.path.append(r"C:\tools\nltk_data")
nltk.download('punkt', download_dir=r"C:\tools\nltk_data", quiet=True)
from nltk.tokenize import sent_tokenize

import tkinter as tk
from tkinter import ttk, messagebox, filedialog, scrolledtext
import requests
from bs4 import BeautifulSoup
import pandas as pd
import re
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib

matplotlib.rcParams['font.family'] = 'Malgun Gothic'
matplotlib.rcParams['axes.unicode_minus'] = False

df_result = None

model_options = {
    "all-MiniLM-L6-v2": "(Default) Fast & lightweight. Good balance of speed and accuracy.",
    "all-mpnet-base-v2": "High accuracy, but slower. Great for quality-focused tasks.",
    "BAAI/bge-base-en-v1.5": "Latest open-source model with strong semantic alignment.",
    "paraphrase-MiniLM-L6-v2": "Trained for paraphrase tasks. Lightweight & fast.",
    "all-distilroberta-v1": "RoBERTa-based. Balanced performance."
}

def clean_text_from_url(url):
    try:
        res = requests.get(url)
        soup = BeautifulSoup(res.text, 'html.parser')
        for tag in soup(['script', 'style', 'noscript']):
            tag.decompose()
        text = soup.get_text(separator=' ', strip=True)
        return re.sub(r'\s+', ' ', text)
    except Exception as e:
        messagebox.showerror("Error", f"Failed to fetch URL:\n{e}")
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

def draw_chart(df, threshold):
    for widget in chart_frame.winfo_children():
        widget.destroy()

    total = df.shape[0]
    covered = df[df['status'] == 'Covered'].shape[0]
    coverage_text = f"Similarity Score by Question ({covered}/{total})"

    fig, ax = plt.subplots(figsize=(12, min(0.6 * total + 1, 10)))  # 질문 수에 따라 높이 조정
    fig.subplots_adjust(left=0.35, right=0.98, top=0.88)  # 왼쪽 margin 확보

    fig.suptitle(coverage_text, fontsize=14, fontweight='bold')

    colors = ['green' if s == 'Covered' else 'gray' for s in df['status']]
    ax.barh(df["question"], df["similarity"], color=colors)
    ax.axvline(threshold, color='red', linestyle='--', label=f'Threshold = {threshold:.2f}')
    ax.set_xlabel("Similarity Score")
    ax.invert_yaxis()
    ax.legend()
    ax.tick_params(axis='y', labelsize=8)

    canvas = FigureCanvasTkAgg(fig, master=chart_frame)
    canvas_widget = canvas.get_tk_widget()
    canvas_widget.pack(side='top', fill='both', expand=True)


    covered = df[df['status'] == 'Covered'].shape[0]
    total = df.shape[0]
    rate_label = ttk.Label(
        chart_frame,
        text=f"📊 Coverage Rate : {covered}/{total} ({covered/total:.0%})",
        font=('Segoe UI', 11, 'bold'),
        foreground='white',
        background='darkgreen',
        padding=6
    )
    rate_label.pack(side='bottom', pady=(5, 5))

def analyze():
    global df_result
    questions_raw = question_input.get("1.0", tk.END).strip().split('\n')
    url = url_entry.get().strip()
    threshold = threshold_slider.get() / 100
    selected_model = model_selector.get()

    if not questions_raw or not url:
        messagebox.showwarning("Input Missing", "Please enter both questions and a URL.")
        return

    model = SentenceTransformer(selected_model)
    full_text = clean_text_from_url(url)
    if not full_text:
        return

    chunks = chunk_text_by_sentence(full_text)
    chunk_vectors = model.encode(chunks)
    question_vectors = model.encode(questions_raw)
    similarity_matrix = cosine_similarity(question_vectors, chunk_vectors)

    results = []
    for i, q in enumerate(questions_raw):
        top_idx = similarity_matrix[i].argmax()
        top_score = similarity_matrix[i][top_idx]
        status = 'Covered' if top_score >= threshold else 'Not Covered'
        results.append({
            "question": q,
            "similarity": round(float(top_score), 4),
            "status": status,
            "matched_chunk": chunks[top_idx]
        })

    df_result = pd.DataFrame(results)
    draw_chart(df_result, threshold)

    output_textbox.config(state='normal')
    output_textbox.delete("1.0", tk.END)
    for i, row in df_result.iterrows():
        output_textbox.insert(tk.END, f"Q{i+1}: {row['question']}\n")
        output_textbox.insert(tk.END, f"→ Similarity: {row['similarity']} / Status: {row['status']}\n")
        output_textbox.insert(tk.END, f"→ Top Matching Chunk:\n{row['matched_chunk'][:300]}...\n\n")
    output_textbox.config(state='disabled')

def save_results():
    if df_result is None or df_result.empty:
        messagebox.showinfo("No Result", "No results to save yet. Run the analysis first.")
        return
    file_path = filedialog.asksaveasfilename(defaultextension=".csv",
                                             filetypes=[("CSV files", "*.csv")])
    if file_path:
        df_result.to_csv(file_path, index=False, encoding="utf-8-sig")
        messagebox.showinfo("Saved", f"Results saved to:\n{file_path}")

def update_model_desc(event):
    selected = model_selector.get()
    model_description.set(model_options.get(selected, "No description available."))

# ---------- Tkinter UI ----------
root = tk.Tk()
root.title("Embedding Relevance Score Analyzer v0.3.6")
root.geometry("1920x1080")

ttk.Label(root, text="Model: Select Sentence Embedding Model", font=('Segoe UI', 11, 'bold')).pack(fill='x', padx=10, pady=(10, 0))
model_selector = ttk.Combobox(root, values=list(model_options.keys()), state="readonly")
model_selector.set("all-MiniLM-L6-v2")
model_selector.pack(fill='x', padx=10, pady=5)

model_description = tk.StringVar()
desc_label = ttk.Label(root, textvariable=model_description, foreground='gray')
model_description.set(model_options[model_selector.get()])
desc_label.pack(fill='x', padx=10, pady=(0, 10))
model_selector.bind("<<ComboboxSelected>>", update_model_desc)

ttk.Label(root, text="Step 1: Enter Questions (one per line)", font=('Segoe UI', 12, 'bold')).pack(fill='x', padx=10, pady=(10, 0))
question_input = scrolledtext.ScrolledText(root, height=10)
question_input.pack(fill='x', padx=10, pady=(0, 10))

ttk.Label(root, text="Step 2: Enter Target URL", font=('Segoe UI', 11, 'bold')).pack(fill='x', padx=10)
url_entry = ttk.Entry(root, font=("Segoe UI", 10))
url_entry.pack(fill='x', padx=10, pady=(0, 10), ipady=4)

slider_frame = ttk.Frame(root)
slider_frame.pack(fill='x', padx=10)
ttk.Label(slider_frame, text="Step 3: Set Similarity Threshold", font=('Segoe UI', 11, 'bold')).pack(side='left')
threshold_slider = ttk.Scale(slider_frame, from_=60, to=90, orient='horizontal')
threshold_slider.set(75)
threshold_slider.pack(side='left', fill='x', expand=True, padx=10)
threshold_label = ttk.Label(slider_frame, text="0.75")
threshold_label.pack(side='left')
threshold_slider.config(command=lambda val: threshold_label.config(text=f"{float(val)/100:.2f}"))

button_frame = ttk.Frame(root)
button_frame.pack(pady=(10, 5))
tk.Button(button_frame, text="Run Analysis", command=analyze).pack(side='left', padx=5)
tk.Button(button_frame, text="Save Results to CSV", command=save_results).pack(side='left', padx=5)

split_frame = ttk.Frame(root)
split_frame.pack(fill='both', expand=True, padx=10, pady=10)

left_frame = ttk.Frame(split_frame, width=600)
left_frame.pack(side='left', fill='both', expand=True)
ttk.Label(left_frame, text="Top Matched Chunks", font=('Segoe UI', 11, 'bold')).pack(anchor='w')
output_textbox = scrolledtext.ScrolledText(left_frame, height=30, font=("Segoe UI", 10))
output_textbox.pack(fill='both', expand=True)
output_textbox.config(state='disabled')

chart_frame = ttk.Frame(split_frame)
chart_frame.pack(side='left', fill='both', expand=True)

root.mainloop()
