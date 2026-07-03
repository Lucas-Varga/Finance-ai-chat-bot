from pathlib import Path
from langchain_unstructured import UnstructuredLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores.utils import filter_complex_metadata
from langchain_chroma import Chroma 
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

def criar_db():
    documentos = carregar_documentos("base")
    chunks = dividir_chunks(documentos)
    chunksFiltrados = filter_complex_metadata(chunks)
    vetorizar_chunks(chunksFiltrados)


def carregar_documentos(caminho_pasta: str = "base") -> list:

    pasta = Path(r"/Users/vargalb/Documents/Automações/AgentRag/base")
    if not pasta.exists():
        raise FileNotFoundError(f"A pasta '{caminho_pasta}' não foi encontrada.")
    if not pasta.is_dir():
        raise NotADirectoryError(f"O caminho '{caminho_pasta}' não é uma pasta.")

    arquivos = [
        str(caminho)
        for caminho in pasta.rglob("*")
        if caminho.is_file() and not caminho.name.startswith(".")
    ]

    if not arquivos:
        print("Nenhum arquivo válido encontrado na pasta.")
        return []

    loader = UnstructuredLoader(file_path=arquivos, languages=["por"])
    return loader.load()


def dividir_chunks(documentos):
    separador_documentos = RecursiveCharacterTextSplitter(
         chunk_size=1000,
         chunk_overlap=500,
         length_function=len,
         add_start_index=True
    )
    chunks = separador_documentos.split_documents(documentos)
    print("tamanho dos chunks", len(chunks))
    return chunks


def vetorizar_chunks(chunks):
    db = Chroma.from_documents(chunks, OpenAIEmbeddings(), persist_directory="db")
    print("db criado")

criar_db()