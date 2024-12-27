def chatbot_page():
    import streamlit as st
    import openai
    from langchain.chat_models import ChatOpenAI
    import pandas as pd
    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain_qdrant import QdrantVectorStore
    # from langchain_ollama.llms import OllamaLLM
    from langchain.prompts import PromptTemplate
    from langchain_core.output_parsers import StrOutputParser
    import asyncio
    # from langchain.embeddings import OpenAIEmbeddings
    from langchain_community.embeddings import OpenAIEmbeddings
    from qdrant_client.http import models
    from pymongo import MongoClient
    # from utils.avatar_manager import AvatarManager
    from utils.utils_funcs import (
        read_token_from_file,
        retrieve_query, 
        get_game_details,
        connect_Qdrant
    )
    import base64
    import re
    from io import BytesIO
    from PIL import Image
    from login_page import UserAuthApp


    # Constants
    # MAX_TOKENS = 1200
    TEMPERATURE=0.2
    NUM_DOCS_RETRIEVED = 10
    SIMILARITY_THRESHOLD = 0.9
    DECAY = 0.05
    MIN_DOCUMENTS = 10
    USER_ICON = "icons\\user_icon.png"  # Replace with your user icon
    BOT_ICON = "icons\\bot_icon.png"  # Replace with your bot icon    
    COLLECTION_NAME = "automatic_ingestion"
    if COLLECTION_NAME == "transformer_sentece_splitter_2":
        EMBEDDING_MODEL_NAME = "thenlper/gte-small"
    elif COLLECTION_NAME == "automatic_ingestion":
        EMBEDDING_MODEL_NAME = "text-embedding-ada-002"

    # Initialize session state keys for models and game options    
    if "initialized" not in st.session_state:
        st.session_state.initialized = False

    if "openai_model" not in st.session_state:
        st.session_state['openai_model'] = "gpt-4o-mini"

    if "messages" not in st.session_state:
        st.session_state.messages = []

    if "selected_game" not in st.session_state:
        st.session_state.selected_game = "Unlock Secret Adventures"

    if "mongo_client" not in st.session_state:
        client = MongoClient("mongodb://localhost:27017/")
        st.session_state.mongo_client = client
    if "user_authenticator" not in st.session_state:
        st.session_state.user_authenticator = UserAuthApp()

    chat_messages_collection = st.session_state.mongo_client["RAG_DB"]["chat_messages"]

# Helper Functions
    def save_message_to_mongo(role, content, game_name, user_id):
        """Save a message to the MongoDB collection."""
        chat_messages_collection.insert_one({
            "user_id": user_id,  # Add user identification
            "role": role,
            "content": content,
            "game_name": game_name,
            "timestamp": pd.Timestamp.now()
        })

    def retrieve_previous_conversations_by_game(user_id):
        """Retrieve all unique game names with conversations for the specified user."""
        return chat_messages_collection.distinct("game_name", {"user_id": user_id})

    def retrieve_conversations_for_game(user_id, game_name):
        """Retrieve conversations for a specific game and user."""
        return list(chat_messages_collection.find({"user_id": user_id, "game_name": game_name}).sort("timestamp", 1))

    # Only run this block if not initialized
    if not st.session_state.initialized:
        # Initial setup
        URL = st.secrets["qdrant_URL"]
        API_KEY = st.secrets["qdrant_API_KEY"]
        openai.api_key = st.secrets["openai"]

        # Display a loading message
        placeholder = st.empty()
        placeholder.write("Initializing app... Please wait")
        
        # Initialize embedding model
        st.session_state.embedding_model = OpenAIEmbeddings(
            openai_api_key=openai.api_key,
            model="text-embedding-ada-002",  # Specify the desired OpenAI embedding model
        )
        # else:
        #     st.session_state.embedding_model = HuggingFaceEmbeddings(
        #         model_name=EMBEDDING_MODEL_NAME,
        #         multi_process=True,
        #         model_kwargs={"device": "cuda"},
        #         encode_kwargs={"normalize_embeddings": True},
        #     )
        # text-embedding-ada-002

        # Connect to Qdrant
        st.session_state.qdrant_client = connect_Qdrant(URL, API_KEY)
        #parser 
        st.session_state.parser = StrOutputParser()
        # Initialize vector store
        st.session_state.vector_store = QdrantVectorStore(
            client=st.session_state.qdrant_client,
            collection_name=COLLECTION_NAME,
            embedding=st.session_state.embedding_model,
        )

        # Initialize the LLM
        st.session_state.llm = ChatOpenAI(openai_api_key=openai.api_key,model=st.session_state['openai_model'], temperature=TEMPERATURE)

        # Set initialization flag to True
        st.session_state.initialized = True
        placeholder.empty()
                         

    # Game options for dropdown -> GENERATE THIS DINAMICALLY! (find another way to select boardgame)
    game_options = ["Unlock Secret Adventures", "The Mind Extreme", "SpellBook", "Chimera Station"]
    selected_game = st.sidebar.selectbox("Select a game:", game_options, index=game_options.index(st.session_state.selected_game))

    # Update selected game only when the selection changes
    if selected_game != st.session_state.selected_game:
        st.session_state.selected_game = selected_game
        st.rerun()

        
    # SIDEBAR PREVIOUS CONVERSATIONS
    # Store the selected game in session state

    st.sidebar.title("Previous Conversations")

    # Retrieve all unique games for the logged-in user
    games = retrieve_previous_conversations_by_game(st.session_state.user)

    # Sidebar with clickable boxes for each game
    if games:
        for game_name in games:
            if st.sidebar.button(f"{game_name}"):
                st.session_state.selected_game = game_name
                st.rerun()

    # Display conversations for the selected game
    if "selected_game" in st.session_state:
        st.title(f"Conversations for {st.session_state.selected_game}")
        conversations = retrieve_conversations_for_game(st.session_state.user, st.session_state.selected_game)
        if conversations:
            for message in conversations:
                if message["role"] == "assistant":
                    with st.chat_message(message["role"], avatar=BOT_ICON):
                        st.markdown(message["content"])
                else:
                    with st.chat_message(message["role"], avatar=USER_ICON):
                        st.markdown(message["content"])
        else:
            st.write("No conversations available for this game.")
    else:
        st.write("Select a game from the sidebar to view conversations.")

    # Display chat messages from history on app rerun
    if "messages" in st.session_state:
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

    st.sidebar.divider()
    st.sidebar.button("Logout", on_click=st.session_state.user_authenticator.logout)


    def stream_response(prompt, context, description, name, llm, parser, template, input_variables, avatar):
        with st.chat_message("assistant", avatar=avatar):
            message_placeholder = st.empty()
            full_response = ""

            # Create a Langchain chain
            try:
                chain = PromptTemplate(template=template, input_variables=input_variables) | llm | parser

                # Stream the model's output chunk by chunk
                for chunk in chain.stream({"context": context, "question": prompt, "description": description, "name": name}):
                    full_response += chunk
                    message_placeholder.markdown(full_response)

                # Save the final message to MongoDB
                save_message_to_mongo("assistant", full_response, st.session_state.selected_game, st.session_state.user)
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")



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

    template_string = """
        System: You are Boardy, an expert assistant specializing in board games. Your role is to provide authoritative, precise, and practical guidance on game rules, mechanics, strategies, and scenarios. 
        You respond as the ultimate reference for the games discussed, ensuring clarity and correctness. Your answers should feel as though they’re guiding the player through a live game session. 
        Avoid general advice or unrelated topics. Instead, focus entirely on providing rule explanations, strategic insights, and in-game examples based on the player's current scenario.

        The game you're explaining today is: **{name}**

        ---
        **Game Overview**:  
        Here’s a description of the game to give you more context about its theme, goals, and mechanics:  
        _{description}_

        ---
        **Current Situation**:  
        This is the specific context that can help you answer the question, Usually it should give you the game's rules, mechanics, and scenarios only if presented in context:  
        _{context}_

        ---
        **Player's Question**:  
        _{question}_

        ---
        **Boardy's Response**:  
        Provide your answer in an instructive and conversational tone as if you’re explaining the rules at the table. clarify mechanics, provide examples only if retrieved from the context:

        - **Game Rule Explanation**: Offer precise details on the relevant game rules present in player's question, mechanics, or actions related to the question.
        """


    prompt = st.chat_input(" ", max_chars=1000)
    if prompt:
        
        # Dynamically set metadata filter based on selected game
        metadata_filter = {'key': 'metadata.game_name', 'value': st.session_state.selected_game}
        context = []
        while SIMILARITY_THRESHOLD > DECAY and len(context) < MIN_DOCUMENTS:
            try:
                context, game_id = retrieve_query(
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
        name, description = get_game_details(game_id)

        # # # Call the async function to stream the response
        # intermediate_response = batch_response(prompt, context, description, name, st.session_state.llm, st.session_state.parser, template_string, ["context", "question", "description", "name"])
        input_variables = ["context", "question", "description", "name"]
        # if intermediate_response:
        #     print("intermediate_response TRUE")
        # Second part of the model, actual streaming with images
        # buffer = ""
        # full_response = ""
        # message_placeholder = st.empty()  # Placeholder for displaying the response
        try:
            # chain = PromptTemplate(template=template_string, input_variables=input_variables) | st.session_state.llm | st.session_state.parser
            
            # Display user message in chat container
            with st.chat_message("user", avatar=USER_ICON):
                st.markdown(prompt)
            # Add user message to chat history
            # st.session_state.messages.append({"role": "user", "content": prompt})
            # Save user message to MongoDB
            save_message_to_mongo("user", prompt, st.session_state.selected_game, st.session_state.user)
            # Stream the model's output chunk by chunk
            stream_response(prompt, context, description, name, st.session_state.llm, st.session_state.parser, template_string, input_variables, BOT_ICON)

        except Exception as e:
            st.error(f"An error occurred: {str(e)}")





