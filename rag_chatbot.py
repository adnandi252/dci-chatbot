#!/usr/bin/env python3
"""
=============================================================================
 RAG Chatbot — Retrieval-Augmented Generation dengan LangChain
=============================================================================
Pipeline:
  1. Baca PDF dari folder data/  →  Document Loader (PyPDF)
  2. Chunking teks              →  RecursiveCharacterTextSplitter
  3. Embedding                  →  HuggingFace (all-MiniLM-L6-v2) — GRATIS
  4. Vector Store               →  FAISS (penyimpanan lokal)
  5. Retrieval + LLM            →  Groq (Llama 3) dengan custom prompt

Fitur:
  Strict Grounding — "Saya tidak tahu" jika konteks tidak relevan
  Source Citation — tampilkan sumber jawaban (halaman & file)
  Loop Percakapan — tanya berkali-kali, ketik 'keluar' untuk berhenti
  Persistent Index — FAISS disimpan, tidak perlu re-embedding tiap run
=============================================================================
"""

import os
import sys
import warnings
from pathlib import Path

from dotenv import load_dotenv

# ─── LangChain ──────────────────────────────────────────────────────────────
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
# LangChain 0.3.x: text_splitter dipisah ke package langchain_text_splitters
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
# LangChain 0.3.x: prompt templates pindah ke langchain_core
from langchain_core.prompts import ChatPromptTemplate

# Abaikan warning minor dari library (opsional, biar output bersih)
warnings.filterwarnings("ignore", category=UserWarning, module="langchain")
warnings.filterwarnings("ignore", category=FutureWarning)

# ─── Konfigurasi Awal ──────────────────────────────────────────────────────
load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    print("[ERROR] GROQ_API_KEY tidak ditemukan di file .env!")
    print("  1. Copy .env.example menjadi .env")
    print("  2. Isi GROQ_API_KEY dengan API key dari https://console.groq.com/keys")
    sys.exit(1)

# Path ke folder-folder proyek
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
FAISS_DIR = BASE_DIR / "faiss_index"
FAISS_INDEX_PATH = FAISS_DIR / "index.faiss"

# ─── 1. Konfigurasi Model & Embedding ──────────────────────────────────────

# --- LLM: Groq (Llama 3) ---
# Groq menawarkan inferensi super cepat untuk model Llama 3 secara gratis!
# Daftar model: https://console.groq.com/docs/models
LLM = ChatGroq(
    model="llama-3.3-70b-versatile",   # Model terbaru & paling capable
    temperature=0.1,                     # Suhu rendah → jawaban lebih faktual
    max_tokens=1024,                     # Batasi panjang jawaban
    groq_api_key=GROQ_API_KEY,
)

# --- Embedding: HuggingFace (all-MiniLM-L6-v2) ---
# Model embedding gratis & ringan dari sentence-transformers.
# Ukuran: ~80MB — sekali download, bisa dipakai offline selamanya.
# Menghasilkan vektor 384 dimensi — cukup untuk RAG skala kecil-menengah.
# normalize_embeddings=True penting agar cosine similarity akurat.
EMBEDDING_MODEL = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"},                   # Pakai CPU (default)
    encode_kwargs={"normalize_embeddings": True},     # Normalisasi vektor
)

# ─── 2. Chunking (Pemotongan Teks) ─────────────────────────────────────────
"""
Mengapa chunk_size=1000 dan chunk_overlap=200?

  chunk_size=1000:
    • Model all-MiniLM-L6-v2 punya max token limit ~512 token per kalimat.
    • 1000 karakter ≈ 200-250 token (untuk bahasa Indonesia/Inggris).
    • Cukup panjang untuk menangkap satu paragraf utuh / satu ide.

  chunk_overlap=200:
    • 20% overlap memastikan tidak ada konteks yang terpotong di
      sambungan antar-chunk.
    • Kalimat yang terbelah di akhir chunk akan muncul kembali di awal
      chunk berikutnya, jadi informasinya tidak hilang.

  RecursiveCharacterTextSplitter:
    • Memisahkan teks secara hierarkis: paragraf → kalimat → kata.
    • Jauh lebih baik daripada splitter naif yang cuma potong per N karakter.
"""
TEXT_SPLITTER = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    length_function=len,
    separators=["\n\n", "\n", ".", " ", ""],
)


# ─── 3. Fungsi: Load & Index PDF ──────────────────────────────────────────

def load_and_index_pdfs() -> FAISS:
    """
    Membaca semua PDF dari folder data/, melakukan chunking & embedding,
    lalu menyimpannya ke FAISS vector store.

    Returns:
        FAISS vector store yang siap digunakan untuk retrieval.
    """
    # Cari semua file PDF di folder data/
    pdf_files = list(DATA_DIR.glob("*.pdf"))
    if not pdf_files:
        print(f"[ERROR] Tidak ada file PDF di folder: {DATA_DIR}")
        print("  Letakkan minimal 1 file PDF di folder 'data/' lalu jalankan ulang.")
        sys.exit(1)

    print(f"Ditemukan {len(pdf_files)} file PDF:")
    for pdf in pdf_files:
        print(f"   • {pdf.name}")

    # Load semua PDF
    all_documents = []
    for pdf_path in pdf_files:
        print(f"\nMembaca: {pdf_path.name}...")
        loader = PyPDFLoader(str(pdf_path))
        documents = loader.load()

        # Tambahkan metadata: sumber file
        for doc in documents:
            doc.metadata["source"] = pdf_path.name

        all_documents.extend(documents)
        print(f"   OK {len(documents)} halaman dimuat.")

    # Chunking
    print(f"\nMelakukan chunking...")
    chunks = TEXT_SPLITTER.split_documents(all_documents)
    print(f"   OK {len(all_documents)} halaman -> {len(chunks)} chunk (chunk_size=1000, chunk_overlap=200)")

    # Buat FAISS vector store dari chunks
    print(f"\nMembuat embedding dan menyimpan ke FAISS...")
    print(f"   Model: all-MiniLM-L6-v2 (384 dimensi)")
    vector_store = FAISS.from_documents(chunks, EMBEDDING_MODEL)

    # Simpan ke disk agar tidak perlu re-index tiap kali jalan
    FAISS_DIR.mkdir(parents=True, exist_ok=True)
    vector_store.save_local(str(FAISS_DIR))
    print(f"   OK Index tersimpan di: {FAISS_DIR}/")

    return vector_store


def load_existing_index() -> FAISS:
    """
    Memuat FAISS index yang sudah ada dari disk.

    Returns:
        FAISS vector store.
    """
    print(f"Memuat FAISS index dari: {FAISS_DIR}/")
    vector_store = FAISS.load_local(
        str(FAISS_DIR),
        EMBEDDING_MODEL,
        allow_dangerous_deserialization=True,
    )
    print(f"   OK Index berhasil dimuat!")
    return vector_store


# ─── 4. Setup Retriever & Prompt Template ──────────────────────────────────

def setup_rag_chain(vector_store: FAISS):
    """
    Membuat RAG chain: retriever → prompt → LLM.

    Args:
        vector_store: FAISS vector store yang sudah siap.

    Returns:
        Sebuah dictionary berisi chain, retriever, dan komponen lainnya.
    """

    # --- Retriever ---
    # k=3: Ambil 3 chunk teratas berdasarkan similarity.
    # TANPA score_threshold — karena:
    #   1. all-MiniLM-L6-v2 dengan normalisasi menghasilkan skor
    #      yang bervariasi (0.3-0.9), tidak ada threshold universal.
    #   2. Prompt template SUDAH menginstruksikan LLM untuk menjawab
    #      "Saya tidak tahu" jika konteks tidak relevan.
    #   3. LLM lebih baik dalam menilai relevansi daripada threshold
    #      angka mati.
    #
    # Strict grounding tetap terjamin oleh prompt, bukan oleh filter
    # threshold yang sering salah positif/negatif.
    retriever = vector_store.as_retriever(
        search_kwargs={"k": 3},
    )

    # --- Custom Prompt Template ---
    # Instruksi STRICT: LLM HANYA boleh pakai konteks dari dokumen.
    # Jika tidak ada jawaban di konteks, WAJIB bilang "Saya tidak tahu."
    prompt_template = ChatPromptTemplate.from_messages([
        (
            "system",
            """Anda adalah asisten AI yang membantu menjawab pertanyaan.
            Anda HANYA boleh menjawab berdasarkan konteks dokumen yang diberikan.
            JANGAN menggunakan pengetahuan umum Anda sendiri.

            ATURAN KETAT:
            1. Jawablah pertanyaan dengan bahasa Indonesia yang baik dan benar.
            2. Jawablah HANYA berdasarkan konteks di bawah ini.
            3. Jika jawaban tidak dapat ditemukan dalam konteks, katakan dengan
               tegas: "Saya tidak tahu. Informasi tersebut tidak tersedia dalam
               dokumen yang diberikan."
            4. JANGAN menggunakan pengetahuan Anda di luar konteks.

            Konteks:
            {context}

            Pertanyaan: {question}
            """,
        ),
    ])

    # --- Chain: Prompt → LLM (tanpa retriever) ---
    # Retriever dipanggil terpisah di loop percakapan agar hanya SEKALI
    # per pertanyaan. Lihat fungsi run_chat_loop() untuk detailnya.
    chain = prompt_template | LLM

    def format_docs(docs):
        """Format dokumen untuk dimasukkan ke prompt."""
        return "\n\n---\n\n".join(doc.page_content for doc in docs)

    return {
        "chain": chain,
        "retriever": retriever,
        "format_docs": format_docs,
    }


# ─── 5. Fungsi: Tampilkan Sumber Jawaban ──────────────────────────────────

def format_sources(retrieved_docs) -> str:
    """
    Memformat sumber jawaban dari dokumen yang diretrieve.

    Args:
        retrieved_docs: List of Document objects hasil retrieval.

    Returns:
        String yang berisi informasi sumber untuk ditampilkan ke user.
    """
    if not retrieved_docs:
        return ""

    sources = []
    for i, doc in enumerate(retrieved_docs, 1):
        metadata = doc.metadata
        source_file = metadata.get("source", "Tidak diketahui")
        page = metadata.get("page", "?")
        # Ambil cuplikan teks (100 karakter pertama) sebagai bukti
        snippet = doc.page_content[:150].replace("\n", " ").strip()
        sources.append(
            f"  [{i}] {source_file} | Halaman {page + 1}\n"
            f"      Kutipan: \"{snippet}...\""
        )

    return "\n" + "\n\n".join(sources)


# ─── 6. Loop Percakapan ────────────────────────────────────────────────────

def run_chat_loop(chain, retriever, format_docs):
    """
    Loop percakapan interaktif.
    User bisa bertanya berkali-kali tanpa restart.
    Ketik 'keluar' untuk berhenti.
    """
    print("\n" + "=" * 60)
    print("RAG CHATBOT -- SIAP DIGUNAKAN!")
    print("=" * 60)
    print("Ketik pertanyaanmu tentang dokumen PDF.")
    print("Ketik 'keluar' untuk menghentikan program.")
    print("Ketik 'reset' untuk me-reload index dari folder data/")
    print("-" * 60)

    while True:
        try:
            # Input dari user
            question = input("\nAnda: ").strip()

            # Perintah keluar
            if question.lower() in ("keluar", "quit", "exit", "q"):
                print("Chatbot: Sampai jumpa!\n")
                break

            # Perintah reset
            if question.lower() == "reset":
                print("Me-reload index...")
                return "reset"  # Signal untuk restart

            # Skip jika kosong
            if not question:
                continue

            print("Chatbot: ", end="", flush=True)

            # --- Retrieval + Generate ---
            # Ambil dokumen relevan dari FAISS (selalu ada, k=3)
            # Strict grounding ditangani oleh prompt LLM, bukan threshold
            retrieved_docs = retriever.invoke(question)

            # Format konteks dari dokumen yang diretrieve, lalu kirim ke LLM
            context = format_docs(retrieved_docs)
            response = chain.invoke({"context": context, "question": question})
            answer = response.content if hasattr(response, "content") else str(response)
            print(answer)

            # --- Source Citation ---
            # Tampilkan sumber jawaban agar transparan
            sources_text = format_sources(retrieved_docs)
            if sources_text:
                print(f"\nSUMBER:{sources_text}")

        except KeyboardInterrupt:
            print("\n\nChatbot: Sampai jumpa!\n")
            break

        except Exception as e:
            print(f"\n[ERROR] {e}")
            print("   Silakan coba lagi.")


# ─── 7. Main Entry Point ───────────────────────────────────────────────────

def main():
    """Entry point utama aplikasi."""
    print("=" * 60)
    print("  RAG CHATBOT - Retrieval-Augmented Generation")
    print("  LLM: Groq (Llama 3)  |  Embedding: all-MiniLM-L6-v2")
    print("  Vector Store: FAISS")
    print("=" * 60)

    # Cek apakah FAISS index sudah ada
    if FAISS_INDEX_PATH.exists():
        print("\nIndex FAISS ditemukan. Memuat index yang sudah ada...")
        print("    (Hapus folder faiss_index/ jika ingin re-index ulang)")
        vector_store = load_existing_index()
    else:
        print("\nIndex FAISS belum ada. Membuat index baru dari PDF...")
        vector_store = load_and_index_pdfs()

    # Setup RAG chain
    rag = setup_rag_chain(vector_store)

    # Jalankan loop percakapan
    result = run_chat_loop(rag["chain"], rag["retriever"], rag["format_docs"])

    # Jika user ketik 'reset', restart dari awal (pakai loop, bukan rekursi)
    while result == "reset":
        print("\nMenghapus index lama dan membuat ulang...")
        if FAISS_DIR.exists():
            import shutil
            shutil.rmtree(FAISS_DIR)
        vector_store = load_and_index_pdfs()
        rag = setup_rag_chain(vector_store)
        result = run_chat_loop(rag["chain"], rag["retriever"], rag["format_docs"])


if __name__ == "__main__":
    main()
