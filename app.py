from langchain.text_splitter import Language

from utils import (
    create_retriever,
    download_repo,
    get_openai_qa_chain,
    load_repo,
    print_qa_message,
    query_repo,
    set_openai_key,
    split_documents,
)
from utils import ConversionAbortedException


def main():
    openai_key = input("Enter OpenAI Key: ")
    set_openai_key(openai_key)

    show_cost_response = input(
        "Do you want to know how much you spend? (yes/no): "
    ).lower()
    show_cost = show_cost_response == "yes"

    repo_url = input("Enter Public GitHub Repository URL: ")

    print("\nSelect the main language of the repository:")
    for lang in Language:
        print(f"{lang.name}")
    main_language = input().upper()

    if not hasattr(Language, main_language):
        print(f"'{main_language}' is not a recognized language.")
        return

    print("\nDownloading and processing the repository. This might take a while...")
    repo_path = "temp_repo"
    download_repo(repo_url, repo_path)
    documents = load_repo(repo_path, main_language)
    texts = split_documents(documents, main_language)
    try:
        retriever = create_retriever(texts, show_cost=show_cost)
    except ConversionAbortedException as e:
        print(e)

    print("\nSelect an OpenAI model:")
    print("1. gpt-3.5-turbo")
    print("2. gpt-4")
    model_choice = input()
    model_name = "gpt-3.5-turbo" if model_choice == "1" else "gpt-4"

    qa_chain = get_openai_qa_chain(model_name, retriever)
    print("Repository converted to vector store. You can now ask questions.")

    while True:
        question = input("\nEnter your question (or type 'exit' to quit): ")
        if question.lower() == "exit":
            break
        answer = query_repo(question, qa_chain, show_cost=show_cost)
        print_qa_message(question, answer["answer"])


if __name__ == "__main__":
    main()
