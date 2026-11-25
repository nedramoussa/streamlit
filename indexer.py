import os
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
load_dotenv() # charge GOOGLE_API_KEY depuis le fichier .env
FICHIER_PDF = os.getenv("FICHIER_PDF", "politique_boutique.pdf")
EMBEDDINGS = os.getenv("EMBEDDINGS", "gemini-embedding-001")
CHROMA_DIR = os.getenv("CHROMA_DIR", "./chroma_db")
COLLECTION = os.getenv("COLLECTION", "politique_boutique")

def connecter_chroma():
    """Se connecte à la base vectorielle Chroma (sans indexer : utilisée aussi par l'agent)."""
    embeddings = GoogleGenerativeAIEmbeddings(model=EMBEDDINGS)
    return Chroma(
    collection_name=COLLECTION,
    embedding_function=embeddings,
    persist_directory=CHROMA_DIR,
    )
# Ce bloc ne s'exécute qu'avec « uv run indexer.py », pas quand l'agent importe connecter_chroma
if __name__ == "__main__":
    # 1. CHARGEMENT DU PDF
    docs = PyPDFLoader(FICHIER_PDF).load()
    print("Pages chargées :", len(docs))
    # 2. DÉCOUPAGE (CHUNKING)
    splitter = RecursiveCharacterTextSplitter(chunk_size=500,
chunk_overlap=100)
    chunks = splitter.split_documents(docs)
    print("Chunks :", len(chunks))
    # 3 + 4. EMBEDDINGS GEMINI + STOCKAGE CHROMADB
    db = connecter_chroma()
    db.reset_collection() # vide la collection : relancer le script ne crée pas de doublons
    db.add_documents(chunks)
    print(f"Index ChromaDB créé : {len(chunks)} chunks dans '{COLLECTION}'({CHROMA_DIR})")
    # 5. TEST RAPIDE DU RETRIEVER
    question = "Puis-je retourner un produit après 20 jours ?"
    print(f"\nTest : {question}")
    for doc in db.similarity_search(question, k=2):
        print(" •", doc.page_content.replace("\n", " ")[:120], "...")