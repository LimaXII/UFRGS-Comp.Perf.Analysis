import faiss
import numpy as np
import ollama

from sentence_transformers import SentenceTransformer
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from pypdf import PdfReader

pdf_path: str = "base_docs/test_doc.pdf"

text_blocks: list[str] = [
    "O TESTE DO LUCCAS é um jogo de aventura e desafios onde o protagonista "
    "precisa superar uma série de provas para alcançar seu objetivo.",
    "O herói da história se chama LUCCAS LIMA. Ele é um personagem corajoso "
    "que enfrenta enigmas, obstáculos e inimigos ao longo da jornada.",
    "No jogo O TESTE DO LUCCAS, LUCCAS LIMA deve usar sua inteligência e "
    "habilidades para desvendar segredos e avançar de fase. Cada nível traz "
    "novos desafios e recompensas.",
]

doc = SimpleDocTemplate(
    pdf_path,
    pagesize=A4,
    leftMargin=2 * cm,
    rightMargin=2 * cm,
    topMargin=2 * cm,
    bottomMargin=2 * cm,
)
styles = getSampleStyleSheet()
body_style = ParagraphStyle(
    "Body",
    parent=styles["Normal"],
    fontSize=12,
    leading=18,
    spaceAfter=14,
)

story: list = []
for block in text_blocks:
    story.append(Paragraph(block, body_style))
    story.append(Spacer(1, 0.4 * cm))

doc.build(story)

print("PDF created")

reader = PdfReader(pdf_path)

document_text = ""
for page in reader.pages:
    document_text += page.extract_text()

print("\nExtracted text:")
print(document_text)

model = SentenceTransformer("intfloat/multilingual-e5-large")

doc_embedding = model.encode([document_text])

dimension = doc_embedding.shape[1]

print("\nEmbedding dimension:", dimension)

index = faiss.IndexFlatL2(dimension)

index.add(np.array(doc_embedding))

documents = [document_text]

print("FAISS index created")

def retrieve(query):

    query_embedding = model.encode([query])

    distances, indices = index.search(np.array(query_embedding), k=1)

    return documents[indices[0][0]]

def rag_query(question):

    context = retrieve(question)

    prompt: str = f"""
You are an assistant that answers questions using the provided context.

Context:
{context}

Question:
{question}

Answer:
"""

    response: dict = ollama.chat(
        model="qwen2.5:7b",
        messages=[{"role": "user", "content": prompt}]
    )

    return response["message"]["content"]

question = "Quem é o protagonista do jogo O TESTE DO LUCCAS?"

answer: str = rag_query(question)

print("\nQuestion:", question)
print("\nAnswer:")
print(answer)