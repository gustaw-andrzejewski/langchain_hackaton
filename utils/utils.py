import os
import shutil

import tiktoken
from git import Repo
from langchain.callbacks import get_openai_callback
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders.generic import GenericLoader
from langchain.document_loaders.parsers import LanguageParser
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.memory import ConversationSummaryMemory
from langchain.text_splitter import Language, RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.vectorstores.base import VectorStoreRetriever

from .ConversionAbortedException import ConversionAbortedException

EMBEDDINGS_PRICE = 0.0001 / 1000


def num_tokens_from_string(string: str) -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.get_encoding("cl100k_base")
    num_tokens = len(encoding.encode(string))
    return num_tokens


def get_repo_converstion_price(texts: list) -> float:
    """Returns estimated cost of embedding the github repository"""
    total_tokens = 0
    for text in texts:
        num_tokens = num_tokens_from_string(text.page_content)
        total_tokens += num_tokens
    price = total_tokens * EMBEDDINGS_PRICE
    return price


def set_openai_key(openai_key: str) -> None:
    """Sets up the openai key"""
    os.environ["OPENAI_API_KEY"] = openai_key


def download_repo(repo_url: str, repo_path: str) -> None:
    """Downloads github repository to a given path"""
    if os.path.exists(repo_path):
        shutil.rmtree(repo_path)
    Repo.clone_from(repo_url, to_path=repo_path)


def get_language_enum(lang_str: str) -> Language:
    """Returns language enum based on a provided language name"""
    lang_str_upper = lang_str.upper()

    if hasattr(Language, lang_str_upper):
        return getattr(Language, lang_str_upper)
    else:
        raise ValueError(f"'{lang_str}' is not a recognized language.")


def get_language_extension(lang_enum: Language) -> str:
    """Returns language extenstion from the provided language"""
    extension_map = {
        Language.CPP: ".cpp",
        Language.GO: ".go",
        Language.JAVA: ".java",
        Language.JS: ".js",
        Language.PHP: ".php",
        Language.PROTO: ".proto",
        Language.PYTHON: ".py",
        Language.RST: ".rst",
        Language.RUBY: ".rb",
        Language.RUST: ".rs",
        Language.SCALA: ".scala",
        Language.SWIFT: ".swift",
        Language.MARKDOWN: ".md",
        Language.LATEX: ".tex",
        Language.HTML: ".html",
        Language.SOL: ".sol",
        Language.CSHARP: ".cs",
    }

    return extension_map.get(lang_enum, "Unknown extension")


def load_repo(repo_path: str, main_language: str) -> list:
    """Loads local repository to a set of documents"""
    language = get_language_enum(main_language)
    language_extenstion = get_language_extension(language)
    loader = GenericLoader.from_filesystem(
        repo_path,
        glob="**/*",
        suffixes=[language_extenstion],
        parser=LanguageParser(language=language, parser_threshold=500),
    )
    documents = loader.load()

    return documents


def split_documents(documents: list, main_language: str) -> list:
    """Splits the documents"""
    Language = get_language_enum(main_language)
    python_splitter = RecursiveCharacterTextSplitter.from_language(
        language=Language, chunk_size=2000, chunk_overlap=200
    )
    texts = python_splitter.split_documents(documents)
    return texts


def create_retriever(texts: list, show_cost: bool = True) -> VectorStoreRetriever:
    """Creates a VectorStoreRetriever for querying the repository"""

    price = get_repo_converstion_price(texts)

    if show_cost:
        proceed = input(
            f"The estimated conversion to vector store is ${price:.5f}. Do you want to proceed? (yes/no): "
        ).lower()
        if proceed != "yes":
            raise ConversionAbortedException("Conversion aborted by user.")

    db = Chroma.from_documents(texts, OpenAIEmbeddings(disallowed_special=()))
    retriever = db.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 5},
    )
    return retriever


def get_openai_qa_chain(
    model_name: str, retriever: VectorStoreRetriever
) -> ConversationalRetrievalChain:
    """Creates ConversationalRetrievalChain for querying the repository with openai model"""
    llm = ChatOpenAI(model_name=model_name)
    memory = ConversationSummaryMemory(
        llm=llm, memory_key="chat_history", return_messages=True
    )
    qa = ConversationalRetrievalChain.from_llm(llm, retriever=retriever, memory=memory)
    return qa


def query_repo(
    question: str, qa: ConversationalRetrievalChain, show_cost: bool = True
) -> str:
    """Asks a question about a repository and returns the answer"""
    print("Processing the question, this might take a while...")
    if show_cost:
        with get_openai_callback() as cb:
            answer = qa(question)
            print(cb)
    else:
        answer = qa(question)
    return answer


def get_colored_text(text: str, color: str, bold: bool = False) -> str:
    color_mapping = {
        "blue": "\033[34m",
        "red": "\033[31m",
        "yellow": "\033[33m",
        "green": "\033[32m",
        "purple": "\033[95m",
    }
    color_code = color_mapping.get(color.lower(), "")
    reset_code = "\033[0m"
    colored_text = f"{color_code}{text}{reset_code}"
    if bold:
        return f"\033[1m{colored_text}{reset_code}"
    return colored_text


def print_qa_message(question: str, answer: str) -> None:
    print(
        f'{get_colored_text("Question:", color = "blue", bold = True)} {question.strip()}'
    )
    print(
        f'{get_colored_text("Answer:", color = "blue", bold = True)} {answer.strip()}'
    )
