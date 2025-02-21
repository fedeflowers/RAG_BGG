def chatbot_page():
    import streamlit as st
    from langchain.chat_models import ChatOpenAI
    import pandas as pd
    from langchain_qdrant import QdrantVectorStore
    from qdrant_client.http.models import  VectorParams
    # from langchain_ollama.llms import OllamaLLM
    from langchain.prompts import PromptTemplate
    from langchain_core.output_parsers import StrOutputParser
    import asyncio
    # from langchain.embeddings import OpenAIEmbeddings
    from langchain_community.embeddings import OpenAIEmbeddings
    from pymongo import MongoClient
    from utils.ingestion_manager import IngestionManager
    from openai_api_key_verifier import list_models
    # from utils.avatar_manager import AvatarManager
    from utils.utils_funcs import retrieve_games_list
    from utils.utils_funcs import (
        retrieve_query, 
        get_templated_prompt,
        extract_first_header
    )
    import base64
    import re
    from io import BytesIO
    from PIL import Image
    from utils.login_page import UserAuthApp


    # Constants
    NUM_DOCS_RETRIEVED = 5
    SIMILARITY_THRESHOLD = 0.9
    DECAY = 0.05
    MIN_DOCUMENTS = 5
    USER_ICON = "icons\\user_icon.png"  # Replace with your user icon
    BOT_ICON = "icons\\bot_icon.png"  # Replace with your bot icon 
    COLLECTION_NAME = "automatic_ingestion_v3"
    EMBEDDING_SIZE = 1536
    MONGO_URI = "mongodb://mongodb:27017"
    

    # Initialize session state keys for models and game options    
    if "initialized" not in st.session_state:
        st.session_state.initialized = False

    if "openai_model" not in st.session_state:
        st.session_state['openai_model'] = "gpt-4o-mini"

    if "messages" not in st.session_state:
        st.session_state.messages = []

    # if "selected_game" not in st.session_state:
    #     st.session_state.selected_game = ""

    if "mongo_client" not in st.session_state:
        client = MongoClient(MONGO_URI)
        st.session_state.mongo_client = client
    if "user_authenticator" not in st.session_state:
        st.session_state.user_authenticator = UserAuthApp()
    if "mongo_collection_chats" not in st.session_state:
        st.session_state.mongo_collection_chats = st.session_state.mongo_client["RAG_DB"]["chat_messages"]
    if "mongo_collection_games" not in st.session_state:
        st.session_state.mongo_collection_games = st.session_state.mongo_client["RAG_DB"]["games"]
    if "collection_qdrant" not in st.session_state:
        st.session_state.collection_qdrant = COLLECTION_NAME

    
    
    

# Helper Functions
    def save_message_to_mongo(role, content, game_name, user_id, chat_messages_collection):
        """Save a message to the MongoDB collection."""
        chat_messages_collection.insert_one({
            "user_id": user_id,  # Add user identification
            "role": role,
            "content": content,
            "game_name": game_name,
            "timestamp": pd.Timestamp.now()
        })

    def retrieve_previous_conversations_by_game(user_id, chat_messages_collection):
        """Retrieve all unique game names with conversations for the specified user."""
        return chat_messages_collection.distinct("game_name", {"user_id": user_id})

    def retrieve_conversations_for_game(user_id, game_name, chat_messages_collection):
        """Retrieve conversations for a specific game and user."""
        return list(chat_messages_collection.find({"user_id": user_id, "game_name": game_name}).sort("timestamp", 1))

    

    # Only run this block if not initialized
    if not st.session_state.initialized:
        # Initial setup
        # openai.api_key = st.secrets["openai"]

        # Display a loading message
        placeholder = st.empty()
        placeholder.write("Initializing app... Please wait")

        #st.session_state.models_available = list_models(st.session_state.openai_key)
        st.session_state.models_available = ["gpt-3.5-turbo","gpt-3.5-turbo-instruct", "gpt-4o-mini", "gpt-4o", "gpt-4"]

        # Initialize embedding model
        st.session_state.embedding_model = OpenAIEmbeddings(
            openai_api_key=st.session_state.openai_key,
            model="text-embedding-ada-002",  # Specify the desired OpenAI embedding model
        )

        # Connect to Qdrant
        ingestion_manager = IngestionManager(path_qdrant_key="",
                                            openai_key= st.session_state.openai_key,
                                            path_qdrant_cloud="",
                                            collection_mongo= st.session_state.mongo_collection_games)
        st.session_state.ingestion_manager = ingestion_manager
        st.session_state.qdrant_client = ingestion_manager.qdrant_client
        #parser 
        st.session_state.parser = StrOutputParser()
        # Initialize vector store
        collections = st.session_state.qdrant_client.get_collections()
        # collections_list = [desc.name for desc in collections[0][1]]
        coll_names = [c.name for c in collections.collections]
        if COLLECTION_NAME not in coll_names:
            # Define parameters for the collection
            vector_params = VectorParams(
                size=EMBEDDING_SIZE,
                distance="Cosine"  # Use the appropriate distance metric
            )
            # Create the collection
            st.session_state.qdrant_client.create_collection(
                collection_name=COLLECTION_NAME,
                vectors_config=vector_params
            )
        else:
            st.session_state.vector_store = QdrantVectorStore(
                client=st.session_state.qdrant_client,
                collection_name=COLLECTION_NAME,
                embedding=st.session_state.embedding_model,
            )

        

        # Set initialization flag to True
        st.session_state.initialized = True
        placeholder.empty()

        game_options = retrieve_games_list(st.session_state.mongo_collection_games)
        #no games found
        if len(game_options) != 0:
            st.session_state.selected_game = game_options[0]
                
    #LLM SETTINGS
    with st.sidebar.expander("🔧 OpenAI Settings", expanded=True):
        if "temperature" not in st.session_state:
            st.session_state.temperature = 0.6
        temperature = st.slider("Temperature", 0.0, 1.0,st.session_state.temperature,  0.05)
        if "current_model" not in st.session_state:
            st.session_state.current_model = "gpt-4o-mini"  # default selection
        default_index = st.session_state.models_available.index(st.session_state.current_model)
        # Display the select box with the current model as default
        selected_model = st.selectbox("Select a Model", st.session_state.models_available, index=default_index)

    if "llm" not in st.session_state or (
        st.session_state.temperature != temperature or
        st.session_state.current_model != selected_model
    ):
        st.session_state.temperature = temperature
        st.session_state.current_model = selected_model
        # Initialize the LLM
        st.session_state.llm = ChatOpenAI(openai_api_key=st.session_state.openai_key,model=selected_model, temperature=temperature)
        print("LLM instance updated successfully, model: ", selected_model, ", temperature: ", temperature, ".")


    # SIDEBAR PREVIOUS CONVERSATIONS
    game_options = retrieve_games_list(st.session_state.mongo_collection_games)
    # print("GAME OPTIONS", game_options)
    # Store the selected game in session state
    if len(game_options) != 0 :
        st.sidebar.title("Previous Conversations")
    else:
        st.sidebar.markdown("No games found, <br> ingest a PDF to start", unsafe_allow_html=True)

    # Game options for dropdown -> GENERATE THIS DINAMICALLY! (find another way to select boardgame)
    if "selected_game" in st.session_state:
        selected_game = st.sidebar.selectbox("Select a game:", game_options, index=game_options.index(st.session_state.selected_game))

    # Update selected game only when the selection changes
    if "selected_game" in st.session_state:
        if selected_game != st.session_state.selected_game:
            st.session_state.selected_game = selected_game
            st.rerun()
    
        
    

    # Retrieve all unique games for the logged-in user
    st.session_state.games = retrieve_previous_conversations_by_game(st.session_state.user, st.session_state.mongo_collection_chats)

    for game_name in st.session_state.games:
        with st.sidebar.container():
            # Create a horizontal layout with columns
            cols = st.columns([2, 1])  # Adjust column proportions
            if cols[0].button(f"🎮 {game_name}", key=f"select_{game_name}"):
                st.session_state.selected_game = game_name
                st.rerun()
            if cols[1].button("❌", key=f"delete_{game_name}"):
                st.session_state.games.remove(game_name)
                #remove from mongo
                st.session_state.mongo_collection_chats.delete_many({"game_name": game_name})
                st.rerun()

    # Display conversations for the selected game
    if "selected_game" in st.session_state:
        st.title(f"Conversations for {st.session_state.selected_game}")
        conversations = retrieve_conversations_for_game(st.session_state.user, st.session_state.selected_game, st.session_state.mongo_collection_chats)
        # print(conversations)
        history = []
        if conversations:

            conversations_copy = conversations.copy()
            conversations_copy.sort(key=lambda x: x["timestamp"], reverse=True)
            last_5 = conversations_copy[:5]
            for i, message in enumerate(last_5):
                occurrence = {}
                if message["role"] == "assistant":
                    occurrence["ANSWER_" + str(i)] = message["content"]
                else:
                    occurrence["QUESTION_" + str(i)] = message["content"]
                history.append(occurrence)


            for message in conversations:
                if message["role"] == "assistant":
                    with st.chat_message(message["role"], avatar=BOT_ICON):
                        st.markdown(message["content"])
                else:
                    with st.chat_message(message["role"], avatar=USER_ICON):
                        st.markdown(message["content"])

            # print(history)
        else:
            with st.chat_message("assistant", avatar=BOT_ICON):
                st.markdown("I am Boardy, your personal board game geek, Ask anything about this game, I am happy to answer any questions")
    else:
        st.write("Select a game from the sidebar to view conversations.")

    # Display chat messages from history on app rerun
    if "messages" in st.session_state:
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

    st.sidebar.divider()


    def go_to_ingestion():
        st.session_state.page = "ingestion"
    #button for pdf ingestion
    st.sidebar.button("Ingest PDF", on_click=go_to_ingestion)
    st.sidebar.button("Logout", on_click=st.session_state.user_authenticator.logout)




    def stream_response(prompt, context, name, llm, parser, template, input_variables, history, avatar, reference):
        with st.chat_message("assistant", avatar=avatar):
            message_placeholder = st.empty()
            full_response = ""

            # Create a Langchain chain
            try:
                chain = PromptTemplate(template=template, input_variables=input_variables) | llm | parser

                # Stream the model's output chunk by chunk
                for chunk in chain.stream({"context": context, "question": prompt, "name": name, "history": history, "reference": reference}):
                    full_response += chunk
                    message_placeholder.markdown(full_response)

                #add references to response
                if reference:
                    unique_headers = set()
                    formatted_references = []

                    # Loop through the references to filter and format
                    for entry in reference:
                        if entry and isinstance(entry, tuple):  # Ignore None and invalid entries
                            header, page = entry
                            if header and header not in unique_headers:
                                unique_headers.add(header)
                                # Format the page, handling ranges or None
                                if page:
                                    formatted_page = f"Page {page}"
                                else:
                                    formatted_page = "No page specified"
                                formatted_references.append(f"- {header} ({formatted_page})")

                    # Combine the references into a markdown section
                    if formatted_references:
                        references_section = "\n\n### References:\n" + "\n".join(formatted_references)
                        full_response += references_section
                        message_placeholder.markdown(full_response)

                # Save the final message to MongoDB
                save_message_to_mongo("assistant", full_response, st.session_state.selected_game, st.session_state.user, st.session_state.mongo_collection_chats)
                #add reference to pages and headers

            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
        st.rerun()



    def batch_response(prompt, context, description, name, llm, parser, template, input_variables):
        # message_placeholder = st.empty()
        full_response = ""
        # Create a Langchain chain
        try:
            chain = PromptTemplate(template=template, input_variables=input_variables) | llm | parser
            # message_placeholder.markdown(full_response)
            # Stream the model's output chunk by chunk
            full_response = chain.invoke({"context": context, "question": prompt, "description": description, "name": name})
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")

        return full_response



    prompt = st.chat_input(" ", max_chars=1000)
    
    if prompt:
        with st.chat_message("user", avatar=USER_ICON):
            st.markdown(prompt)
            # Add user message to chat history
            # Save user message to MongoDB
        save_message_to_mongo("user", prompt, st.session_state.selected_game, st.session_state.user, st.session_state.mongo_collection_chats)
        # Dynamically set metadata filter based on selected game
        metadata_filter = {'key': 'metadata.game_name', 'value': st.session_state.selected_game}
        context = []
        metadata = []
        while SIMILARITY_THRESHOLD > DECAY and len(context) < MIN_DOCUMENTS:
            try:
                metadata, context = retrieve_query(
                                        prompt,
                                        NUM_DOCS_RETRIEVED,
                                        st.session_state.embedding_model,
                                        st.session_state.qdrant_client, 
                                        st.session_state.vector_store,
                                        metadata_filter = metadata_filter,
                                        similarity_threshold=SIMILARITY_THRESHOLD
                                        )
                SIMILARITY_THRESHOLD -= DECAY
                print("setting similarity threshold to", SIMILARITY_THRESHOLD)
                if len(context) >= MIN_DOCUMENTS:
                    break
            except Exception as e:
                print(f"Error: {e}")
                SIMILARITY_THRESHOLD -= DECAY
                print("setting similarity threshold to", SIMILARITY_THRESHOLD)
        # description = get_game_details(game_id)
        name = st.session_state.selected_game
        #get references from metadata
        reference = []
        if metadata:
            for doc in metadata:
                reference.append(extract_first_header(doc))
        # print("reference", reference)
        # print("metadata", metadata)

        input_variables = ["context", "question", "name", "history", "reference"]
        try:
            template_string = get_templated_prompt()
            # print(context)
            stream_response(prompt, context, name, st.session_state.llm, st.session_state.parser, template_string, input_variables, history,  BOT_ICON, reference)
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")





