from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

CAMINHO_DB = "db"

prompt_template = """
Responda a pergunta do usuário:
{pergunta}
com base nessas informações a baixo, passando para o usuário apenas a resposta precisa
de acordo com o conhecimento que você tiver disponivel :

{base_conhecimento}

caso não souber, diga que não sabe.
"""


def perguntar():
    pergunta = input("Escreva sua pergunta: ")

    funcao_embedding = OpenAIEmbeddings()

    db = Chroma(persist_directory=CAMINHO_DB, embedding_function=funcao_embedding)

    resultados = db.similarity_search_with_relevance_scores(pergunta, k=3)
    if len(resultados) == 0 or resultados[0][1] < 0.7:
        print("Não conseguiu encontrar uma informação relevante na base.")
        return
    textos_resultado = []
    for resultado in resultados:
        texto = resultado[0].page_content
        textos_resultado.append(texto)

    base_conhecimento = "\n\n----\n\n".join(textos_resultado)
    prompt = ChatPromptTemplate.from_template(prompt_template)
    prompt = prompt.invoke({"pergunta": pergunta, "base_conhecimento": base_conhecimento})

    modelo = ChatOpenAI()
    texto_resposta = modelo.invoke(prompt).content
    print("Chat GPT: ", texto_resposta)

perguntar()