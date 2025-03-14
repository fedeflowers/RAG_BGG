from marker.converters.pdf import PdfConverter
from marker.models import create_model_dict
from marker.output import text_from_rendered
from langchain.text_splitter import MarkdownHeaderTextSplitter
import fitz  # PyMuPDF
from utils.utils_funcs import read_token_from_file
from qdrant_client import QdrantClient
from langchain_qdrant import QdrantVectorStore
from langchain.embeddings import OpenAIEmbeddings  # Correct import path for OpenAI embeddings
import os
from pymongo import MongoClient, errors


class IngestionManager:
    def __init__(self, path_qdrant_key, openai_key, path_qdrant_cloud, collection_mongo, local = True ):
        self.headers_to_split =  [
            ("#", "Header 1"),
            ("##", "Header 2"),
            ("###", "Header 3"),
            ("####", "Header 4"),
            ("#####", "Header 5"),
            ("######", "Header 6"),
        ]

        self.text_splitter = MarkdownHeaderTextSplitter(
            self.headers_to_split,
            strip_headers=False
        )
        self.collection_mongo = collection_mongo
        os.environ["OPENAI_API_KEY"] = openai_key
        QDRANT_HOST = os.getenv("QDRANT_HOST", "http://qdrant:6333")
        # self.collection_qdrant = collection_qdrant
        self.docs_processed = None
        if local:
            self.qdrant_connection = self.URL = QDRANT_HOST
            self.qdrant_client = QdrantClient(url=self.qdrant_connection)
        else:
            self.qdrant_connection = read_token_from_file(path_qdrant_key)
            self.URL = read_token_from_file(path_qdrant_cloud)
            self.qdrant_client = QdrantClient(
                url=self.URL,
                api_key=self.qdrant_connection,
            )
        self.converter = PdfConverter(
            artifact_dict=create_model_dict(),
        )

    def find_headers_in_pdf(self, pdf_path, headers):
        """
        Finds specified headers in a PDF and returns their page numbers.

        :param pdf_path: Path to the PDF file
        :param headers: Dictionary of headers to search for
        :return: Dictionary mapping headers to lists of page numbers
        """
        # Open the PDF file
        document = fitz.open(pdf_path)
        header_pages = {header: [] for header in headers.values()}
        
        # Iterate through each page
        for page_num in range(len(document)):
            page = document.load_page(page_num)
            text = page.get_text("text")  # Extract raw text
            
            # Split text into lines for line-by-line matching
            lines = text.split("\n")
            
            for line in lines:
                for _, header_value in headers.items():
                    # Match exact line content with header
                    if line.strip() == header_value.strip()[2:-2]:
                        header_pages[header_value].append(page_num + 1)  # 1-indexed
        
        document.close()
        return header_pages

    def ingest_pdfs(self, pdf_file):
        """
        Ingests a single PDF file, processes it, and extracts metadata.

        This function reads the provided PDF file object, converts it into a text representation,
        and splits the text into chunks based on predefined headers. It then finds the headers in the PDF
        and adds metadata to each chunk, which includes the game name and page numbers where the headers are located.

        :param pdf_file: The PDF file object to be ingested.
        :type pdf_file: _io.BytesIO (or a similar file-like object)

        :return: List of processed chunks with metadata.
        """
        
        import tempfile
        import os

        # Save the uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(pdf_file.read())
            temp_pdf_path = tmp_file.name

        to_ingest = []
        file_name = pdf_file.name
        game_name = file_name.rsplit(".", 1)[0]
        if self.check_game_name(game_name):
            raise Exception("Game name already exists in the database")


        # Convert the temporary PDF file into text
        rendered = self.converter(temp_pdf_path)
        text, _, _ = text_from_rendered(rendered)
        chunks = self.text_splitter.split_text(text)

        # Add metadata for pages
        result = []
        for chunk in chunks:
            result.append(self.find_headers_in_pdf(temp_pdf_path, chunk.metadata))

        pages = []  # Pages for each chunk
        for el in result:
            values = list(el.values())  # Convert values to a list

            # Check if values has content
            if values:
                # Ensure the first element is not empty
                if values[0]:
                    if len(values) >= 2 and values[0] != values[-1] and values[-1]:
                        pages.append(f"{values[0][0]} - {values[-1][0]}")
                    else:
                        pages.append(f"{values[0][0]}")
                else:
                    pages.append("Not found in Rulebook")
            else:
                pages.append("Not found in Rulebook")

        # Extract game metadata from the file name
        
        txt_dict = {
            "game_name": game_name,
        }

        # Add metadata to each chunk
        for i, chunk in enumerate(chunks):
            chunk.metadata['game_name'] = txt_dict['game_name']
            chunk.metadata['pages'] = pages[i]
            to_ingest.append(chunk)
        #add game to mongo
        self.add_game_name_to_mongo(game_name)

        return to_ingest



    def add_embeddings_to_qdrant(self, docs_processed, collection_name="automatic_ingestion", embedder_openai="text-embedding-ada-002"):
        """Create embeddings for documents and add them to Qdrant."""


        if os.environ["OPENAI_API_KEY"]:
            embedding_model = OpenAIEmbeddings(
                model=embedder_openai,  # Specify the desired OpenAI embedding model
            )

        vector_store = QdrantVectorStore.from_documents(
            docs_processed,
            embedding_model,
            url = self.URL,
            #prefer_grpc=True,
            api_key=self.qdrant_connection,
            collection_name=collection_name,
            force_recreate = False
        )
        return vector_store


    def add_game_name_to_mongo(self, game_name):
        try:
            # Insert the new game name
            self.collection_mongo.insert_one({"game_name": game_name})
            return True  # Successfully added
        except errors.PyMongoError as e:
            raise Exception(f"An error occurred while adding the game name to MongoDB: {e}")
        
    def check_game_name(self, game_name):
        # Check if the game name already exists
        if self.collection_mongo.find_one({"game_name": game_name}):
            return True  # Game name already exists
        return False

