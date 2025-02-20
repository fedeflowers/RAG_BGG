from qdrant_client import QdrantClient
from qdrant_client.http import models

import requests
from bs4 import BeautifulSoup

def read_token_from_file(file_path="token.txt"):
    """
    Read a token from a file located at the specified file_path.
    
    Parameters:
        file_path (str): The path to the file containing the token.
        
    Returns:
        str: The token read from the file with leading and trailing whitespaces removed.
    """
    with open(file_path, "r") as file:
        return file.read().strip()

def process_pdfs(path, tokenizer):
    # Load dataset from Hugging Face
    RAW_KNOWLEDGE_BASE = []

    # List the PDF files in the folder
    pdf_files = [f for f in os.listdir(path) if f.endswith(".pdf")]
    pdf_texts = []

    for pdf_file in tqdm(pdf_files):
        # Extract text from PDF using pdfplumber
        with pdfplumber.open(os.path.join(folder_path, pdf_file)) as pdf:
            text = ""
            for page in pdf.pages:
                text += page.extract_text()
        pdf_texts.append(text)

    pdf_dict = [{"game_name" : pdf_file.removesuffix(".pdf").split("_")[0], "game_id" : pdf_file.removesuffix(".pdf").split("_")[1]} for i, pdf_file in enumerate(pdf_files)]

    RAW_KNOWLEDGE_BASE = tokenizer.create_documents(
        pdf_texts,
        pdf_dict
    )

    docs_processed = tokenizer.split_documents(RAW_KNOWLEDGE_BASE)
    return docs_processed

def add_embeddings_to_qdrant(docs_processed, URL, API_KEY, collection_name="test_2"):
    """Create embeddings for documents and add them to Qdrant."""
    
    URL=URL 
    API_KEY=API_KEY

    # Initialize the embedding model
    embedding_model = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        multi_process=True,  # Enable multiprocessing
        model_kwargs={"device": "cuda"},
        encode_kwargs={"normalize_embeddings": True},  # Set `True` for cosine similarity
    )


    vector_store = QdrantVectorStore.from_documents(
        docs_processed,
        embedding_model,
        url = URL,
        prefer_grpc=True,
        api_key=API_KEY,
        collection_name=collection_name,
        force_recreate = True
    )



def connect_Qdrant(URL, API_KEY):
    # Initialize Qdrant client
    qdrant_client = QdrantClient(
        url=URL, 
        api_key=API_KEY,
    )
    return qdrant_client




# def retrieve_query(query,  embedding_model, qdrant_client, vector_store, metadata_filter=None, k=1,):
#     '''
#     Retrieve query from Qdrant with metadata filtering. 
#     k: Number of documents to retrieve.
#     metadata_filter: Dictionary specifying filter conditions (e.g., { "key": "game_name", "value": "Unlock" })
#     '''
    
#     if metadata_filter:
#         # Create a filter based on the provided metadata
#         filter_conditions = models.Filter(
#             must=[
#                 models.FieldCondition(
#                     key=metadata_filter["key"],  # Metadata key, e.g., 'game_name'
#                     match=models.MatchValue(
#                         value=metadata_filter["value"],  # Metadata value, e.g., 'Unlock'
#                     ),
#                 ),
#             ]
#         )
        
#         # Debug: Print filter conditions to verify
#         # print(f"Filter conditions: {filter_conditions}")

#         # Perform similarity search with metadata filtering
#         result = vector_store.similarity_search(
#             query=query,
#             k=k,
#             filter=filter_conditions  # Applying the metadata filter
#         )
#     else:
#         # Perform regular similarity search without filtering
#         result = vector_store.similarity_search(
#             query=query,
#             k=k,
#         )
    
#     # Debug: Print retrieved results
#     # for doc in result:
#         # print(f"Retrieved result: {doc}")
    
#     # Extract content from the result
#     game_id = result[-1].metadata["game_id"]
#     context = [doc.page_content for doc in result]

#     image_metadata = {}
#     for doc in result:
#         for key, value in doc.metadata.items():
#             if key.startswith("image"):
#                 image_metadata[key] = value
    
    
#     return context, game_id, image_metadata
def retrieve_query(query, k, embedding_model, qdrant_client, vector_store, metadata_filter, similarity_threshold):
    '''
    Retrieve query from Qdrant with metadata filtering. 
    k: Number of documents to retrieve.
    metadata_filter: Dictionary specifying filter conditions (e.g., { "key": "game_name", "value": "Unlock" })
    '''
    metadata = context = []
    if metadata_filter:
        # Create a filter based on the provided metadata
        filter_conditions = models.Filter(
            must=[
                models.FieldCondition(
                    key=metadata_filter["key"],  # Metadata key, e.g., 'game_name'
                    match=models.MatchValue(
                        value=metadata_filter["value"],  # Metadata value, e.g., 'Unlock'
                    ),
                ),
            ]
        )
        
        # Debug: Print filter conditions to verify
        # print(f"Filter conditions: {filter_conditions}")

        # Perform similarity search with metadata filtering
        # result = vector_store.similarity_search(
        #     query=query,
        #     k=k,
        #     filter=filter_conditions  # Applying the metadata filter
        # )
        result = vector_store.search(
            query=query,  # Replace with your query vector
            limit=k,
            filter=filter_conditions,  # Metadata filter
            score_threshold=similarity_threshold,  # Set your threshold here, e.g., 0.75
            search_type="similarity_score_threshold"
        )
    else:
        # Perform regular similarity search without filtering
        # result = vector_store.similarity_search(
        #     query=query,
        #     k=k,
        # )
        result = vector_store.search(
            query=query,  # Replace with your query vector
            limit=k,
            filter=filter_conditions,  # Metadata filter
            score_threshold=similarity_threshold,  # Set your threshold here, e.g., 0.75
            search_type="similarity_score_threshold"
        )
    # Debug: Print retrieved results
    # for doc in result:
        # print(f"Retrieved result: {doc}")
    
    # Extract content from the result
    if result:
        # game_id = result[-1].metadata["game_id"]
        metadata = [doc.metadata for doc in result]
        context = [doc.page_content for doc in result]

    # Loop through metadata and find elements that start with "image"
    # for doc in result:
    #     for key, value in doc.metadata.items():
    #         if key.startswith("image"):
    #             image_metadata[key] = value
    
    return metadata, context




def get_game_details(game_id):
    # Step 3: Use the ID to retrieve game details from the API
    api_url = f"https://boardgamegeek.com/xmlapi/boardgame/{game_id}"
    response = requests.get(api_url)
    
    if response.status_code == 200:
        soup = BeautifulSoup(response.content, 'xml')
        # Extracting the description of the game
        description = soup.find('description').text
        return description
    else:
        print("Failed to retrieve game details.")
        return None



def get_templated_prompt():
        template_string = """
        System: You are Boardy, an expert assistant specializing in board games. Your role is to provide authoritative, precise, and practical guidance on game rules, mechanics, strategies, and scenarios. 
        You respond as the ultimate reference for the games discussed, ensuring clarity and correctness. Your answers should feel as though they’re guiding the player through a live game session. 
        Avoid general advice or unrelated topics. Instead, focus entirely on providing rule explanations, strategic insights, and in-game examples based on the player's current scenario.

        The game you're explaining today is: **{name}**

        ---
        **Previous Conversation**:
        This is the previous 5 exchanges between the player and Boardy. It can help you understand the context of the current question.
        The messages are given in chronological order, from the most recent to the oldest, QUESTION_1 and ANSWER_1 are the most recent messages up to QUESTION_5 and ANSWER_5 which are the oldest messages:  
        _{history}_

        **Current Situation**:  
        This is the specific context that can help you answer the question, Usually it should give you the game's rules, mechanics, and scenarios only if presented in context:  
        _{context}_

        **References for the question**:
        Here you can find metadata about the documents of the context you retrieved, they are in this format [('Header', 'page number'), ...], use them to guide the user if he/she is looking for a specific rule or mechanic,
        if there are no references, you can say that no references were found in the rulebook:
        _{reference}_
        ---
        **Player's Question**:  
        _{question}_

        ---
        **Boardy's Response**:  
        Provide your answer in an instructive and conversational tone as if you’re explaining the rules at the table. clarify mechanics, provide examples only if retrieved from the context:

        - **Game Rule Explanation**: Offer precise details on the relevant game rules present in player's question, mechanics, or actions related to the question.
        """
        
        return template_string


def extract_first_header(metadata):
    """
    Extracts the first available header (based on priority) and associated pages from metadata.
    
    Args:
        metadata (dict): Dictionary containing metadata with headers and pages.
    
    Returns:
        tuple: A tuple containing the header and pages (if available), or None if no header is found.
    """
    # Define the header priorities in order
    header_priority = ["Header 1", "Header 2", "Header 3", "Header 4", "Header 5", "Header 6"]
    
    # Find the first available header based on priority
    for header in header_priority:
        if header in metadata:
            selected_header = metadata[header]
            break
    else:
        # No header found
        return None
    
    # Extract pages if available
    pages = metadata.get("pages", None)
    if pages and pages == "Not found in Rulebook":
        pages = None

    
    return selected_header, pages

def retrieve_games_list(games_collection):
        """Retrieve all unique game names."""
        return games_collection.distinct("game_name")

