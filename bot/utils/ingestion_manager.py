# Sostituito Marker con OpenAI extractor (molto più leggero)
# from marker.converters.pdf import PdfConverter
# from marker.models import create_model_dict
# from marker.output import text_from_rendered
from utils.openai_extractor import OpenAILightExtractor
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
        # OpenAI extractor invece di Marker (zero modelli locali)
        self.converter = OpenAILightExtractor(api_key=openai_key)

    def create_page_mapping(self, text):
        """Crea una mappa delle posizioni dei riferimenti pagina nel testo"""
        import re
        
        page_mapping = []
        lines = text.split('\n')
        
        current_page = 1
        for i, line in enumerate(lines):
            # Trova i marcatori di pagina
            page_match = re.search(r'--- Page (\d+) ---', line)
            if page_match:
                current_page = int(page_match.group(1))
            
            # Associa ogni riga al numero di pagina corrente
            page_mapping.append({
                'line_index': i,
                'line_content': line,
                'page_number': current_page
            })
        
        return page_mapping
    
    def find_page_for_chunk(self, chunk_content, page_mapping):
        """Trova la pagina di appartenenza di un chunk basandosi sulla mappa"""
        # Prendi le prime righe del chunk per identificare la pagina
        chunk_lines = chunk_content.split('\n')[:3]  # Prime 3 righe dovrebbero bastare
        
        # Cerca queste righe nella mappa
        for chunk_line in chunk_lines:
            if chunk_line.strip():  # Ignora righe vuote
                for mapping in page_mapping:
                    if mapping['line_content'].strip() == chunk_line.strip():
                        return f"Page {mapping['page_number']}"
        
        # Fallback: cerca qualsiasi riferimento pagina diretto nel chunk
        import re
        page_match = re.search(r'--- Page (\d+) ---', chunk_content)
        if page_match:
            return f"Page {page_match.group(1)}"
        
        return "Not found in Rulebook"

    def extract_page_references_from_markdown(self, markdown_text):
        """Estrae i riferimenti alle pagine dal markdown chunk"""
        import re
        
        # Cerca pattern "--- Page X ---" nel markdown
        page_matches = re.findall(r'--- Page (\d+) ---', markdown_text)
        
        if page_matches:
            # Converte in numeri e trova range
            page_numbers = [int(p) for p in page_matches]
            min_page = min(page_numbers)
            max_page = max(page_numbers)
            
            if min_page == max_page:
                return str(min_page)
            else:
                return f"{min_page} - {max_page}"
        else:
            return "Not found in Rulebook"

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


        # Converti PDF in markdown usando OpenAI API (molto più leggero di Marker)
        text = self.converter.extract_from_pdf(temp_pdf_path)
        
        # Prima di dividere in chunks, crea una mappa dei riferimenti pagina
        page_mapping = self.create_page_mapping(text)
        
        chunks = self.text_splitter.split_text(text)

        # Associa ogni chunk alle pagine corrette usando la mappa
        pages = []  # Pages for each chunk
        for chunk in chunks:
            page_ref = self.find_page_for_chunk(chunk.page_content, page_mapping)
            pages.append(page_ref)

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

