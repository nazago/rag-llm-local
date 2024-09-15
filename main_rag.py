from __future__ import annotations  # noqa: D100

import logging
from pathlib import Path
from typing import TYPE_CHECKING

from langchain.text_splitter import MarkdownHeaderTextSplitter
from langchain_chroma import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import OllamaLLM

import config


if TYPE_CHECKING:
    from xml.dom.minidom import Document


# Mini Logger Setup
logging.basicConfig(
    filename="build/log/output.log",  # Log file
    filemode="w",  # write mode
    format="%(asctime)s - %(levelname)s - %(message)s",  # Log format
    datefmt="%Y-%m-%d %H:%M:%S",  # Date format
    level=logging.INFO,  # Logging level
)

class DocumentManager:
    """Manages the loading and splitting of markdown documents from a specified directory.

    This class is responsible for loading markdown files from a specified directory,
    and splitting the content of those files into sections based on headers.

    Attributes:
        directory_path (str): Path to the directory containing markdown files.
        glob_pattern (str): The pattern used to search for markdown files. Defaults to "./*.md".
        documents (list): List of loaded markdown document contents.
        all_sections (list): List of split sections from the markdown files.
    """

    def __init__(self, directory_path: str, glob_pattern: str = "./*.md") -> None:
        """Initializes the DocumentManager with a directory path and an optional glob pattern.

        Args:
            directory_path (str): Path to the directory containing markdown files.
            glob_pattern (str, optional): The pattern used to search for markdown files. Defaults to "./*.md".
        """
        self.directory_path = directory_path
        self.glob_pattern = glob_pattern
        self.documents = []
        self.all_sections = []

    def load_markdown_files(self) -> None:
        """Loads markdown files from the specified directory and stores their content in the documents list.

        This method recursively searches for `.md` files within the directory and its subdirectories.
        """
        for filepath in Path(self.directory_path).rglob("*.md"):
            self.documents.append(filepath.read_text(encoding="utf-8"))

    def split_documents(self) -> None:
        """Splits the loaded markdown documents into sections based on header levels.

        The documents are split using headers such as `#`, `##`, `###`, and `####`.
        The resulting sections are stored in the all_sections list.
        """
        headers_to_split_on = [("#", "Header 1"), ("##", "Header 2"), ("###", "Header 3"), ("####", "Header 4")]
        text_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on, strip_headers=False)
        for doc in self.documents:
            sections = text_splitter.split_text(doc)
            self.all_sections.extend(sections)

class EmbeddingManager:
    """Manages the creation, persistence, and retrieval of document embeddings using a vector database.

    This class is responsible for creating embeddings for a collection of document sections,
    persisting them in a vector database, and retrieving the stored vector database.

    Attributes:
        all_sections (list[Document]): List of document sections to be embedded.
        persist_directory (str): Path to the directory where the embeddings are stored.
        vectordb: The vector database used to store and retrieve embeddings.
        embedding: The embedding model used for creating document embeddings.
    """

    def __init__(self, all_sections: list[Document], persist_directory: str = config.DATABASE_DIRECTORY) -> None:
        """Initializes the EmbeddingManager with document sections and a persistence directory.

        Args:
            all_sections (list[Document]): List of document sections to be embedded.
            persist_directory (str, optional): Path to the directory where the embeddings are stored. Defaults to "db".
        """
        self.all_sections = all_sections
        self.persist_directory = persist_directory
        self.vectordb = None
        self.embedding = OllamaEmbeddings(model=config.EMBEDDING_MODEL)

    def create_and_persist_embeddings(self) -> None:
        """Creates embeddings for the document sections and persists them in the vector database.

        This method uses the embedding model to generate embeddings for each document section,
        and stores them in the vector database located in the specified persistence directory.
        """
        self.vectordb = Chroma.from_documents(documents=self.all_sections, embedding=self.embedding, persist_directory=self.persist_directory)

    def retrieve_vector_database(self) -> None:
        """Retrieves the persisted vector database.

        This method loads the vector database from the specified persistence directory,
        allowing access to the previously stored embeddings.
        """
        self.vectordb = Chroma(persist_directory=self.persist_directory, embedding_function=self.embedding)

def format_docs(docs: list[Document]) -> None:  # noqa: D103
    return "\n\n".join(doc.page_content for doc in docs)

if __name__ == "__main__":
    documents_loader_and_split = DocumentManager(config.DOCS_DIRECTORY)
    embed_manager = EmbeddingManager(documents_loader_and_split.all_sections)

    if config.DATABASE_CREATION:
        documents_loader_and_split.load_markdown_files()
        documents_loader_and_split.split_documents()
        docs          = documents_loader_and_split.documents
        splitted_docs = documents_loader_and_split.all_sections
        logging.info("DOCUMENTS:\n%s", docs)
        logging.info("SPLITTED DOCS:\n%s",splitted_docs)
        logging.info("Number of sections: %s",len(splitted_docs))
        embed_manager.create_and_persist_embeddings()
    else:
        embed_manager.retrieve_vector_database()
        logging.info(f"Number of embedded vectors: {embed_manager.vectordb._collection.count()}")  # noqa: G004, SLF001

    # retriver out: list of Document objects from documents_loader_and_split.all_sections
    retriever = embed_manager.vectordb.as_retriever(search_type="mmr",
                                                    search_kwargs={"k": 4}) # gives closest k chunks

    llm = OllamaLLM(model= config.LLM_MODEL, temperature=0.4)

    RAG_TEMPLATE = """
    You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question.
    These notes come from documentation and a laboratory diary.
    If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.

    <context>
    {context}
    </context>

    Answer the following question:

    {question}"""

    rag_prompt = ChatPromptTemplate.from_template(RAG_TEMPLATE)

    rag_chain = rag_prompt | llm | StrOutputParser()

    print("Welcome to information retrieval LLM! To exit, type `exit`")
    while True:
        print("Ask something")
        user_input = input()
        if user_input == "exit":
            print("Goodbye!")
            break
        context = retriever.invoke(user_input)
        print(f"Question: {user_input}\nContext:\n{format_docs(context)}")
        print("##################################################\nElaborating your prompt...")
        result = rag_chain.invoke({"context": format_docs(context), "question": user_input})
        print("##################################################")
        print(f"AI BOT reply:\n {result}")
