if __name__ == '__main__':
    import streamlit as st
    import openai
    from langchain.chat_models import ChatOpenAI
    import pandas as pd
    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain_qdrant import QdrantVectorStore
    from langchain_ollama.llms import OllamaLLM
    from langchain.prompts import PromptTemplate
    from langchain_core.output_parsers import StrOutputParser
    import asyncio
    from langchain.embeddings import OpenAIEmbeddings
    from qdrant_client.http import models
    from utils.utils_funcs import (
        read_token_from_file,
        retrieve_query, 
        get_game_details,
        connect_Qdrant
    )
    MAX_TOKENS = 400
    COLLECTION_NAME = "openai"
    if COLLECTION_NAME == "transformer_sentece_splitter_2":
        EMBEDDING_MODEL_NAME = "thenlper/gte-small"
    elif COLLECTION_NAME == "openai":
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

    # Only run this block if not initialized
    if not st.session_state.initialized:
        # Initial setup
        URL = st.secrets["qdrant_URL"]
        API_KEY = st.secrets["qdrant_API_KEY"]
        openai.api_key = st.secrets["openai"]

        # Display a loading message
        placeholder = st.empty()
        placeholder.write("Initializing app... please wait")
        
        # Initialize embedding model
        if COLLECTION_NAME == "openai":
            st.session_state.embedding_model = OpenAIEmbeddings(
                openai_api_key=openai.api_key,
                model="text-embedding-ada-002",  # Specify the desired OpenAI embedding model
            )
        else:
            st.session_state.embedding_model = HuggingFaceEmbeddings(
                model_name=EMBEDDING_MODEL_NAME,
                multi_process=True,
                model_kwargs={"device": "cuda"},
                encode_kwargs={"normalize_embeddings": True},
            )
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
        st.session_state.llm = ChatOpenAI(openai_api_key=openai.api_key,model=st.session_state['openai_model'], temperature=1, max_tokens=MAX_TOKENS)

        # Set initialization flag to True
        st.session_state.initialized = True
        placeholder.empty()
        st.success("App ready!")

    # Game options for dropdown
    game_options = ["Unlock Secret Adventures", "The Mind Extreme", "SpellBook", "Chimera Station"]
    selected_game = st.selectbox("Select a game:", game_options, index=game_options.index(st.session_state.selected_game))

    # Store the selected game in session state
    st.session_state.selected_game = selected_game

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Capture the user input for the question

    def stream_response(prompt, context, description, name, llm, parser):
        message_placeholder = st.empty()
        full_response = ""

        # Create a Langchain chain
        try:
            chain = PromptTemplate(template=template_string, input_variables=["context", "question", "description", "name", "MAX_TOKENS"]) | llm | parser

            # Stream the model's output chunk by chunk
            for chunk in chain.stream({"context": context, "question": prompt, "description": description, "name": name, "MAX_TOKENS": MAX_TOKENS}):
                full_response += chunk
                message_placeholder.markdown(full_response)

        except Exception as e:
            st.error(f"An error occurred: {str(e)}")

        # Add the assistant's complete response to the chat history
        if "messages" not in st.session_state:
            st.session_state.messages = []
        st.session_state.messages.append({"role": "assistant", "content": full_response, name:"Boardy"})

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
        This is the specific context or scenario the player is in, which might affect your answer:  
        _{context}_

        ---
        **Player's Question**:  
        _{question}_

        ---
        **Boardy's Response**:  
        Provide your answer in an instructive and conversational tone as if you’re explaining the rules and strategies at the table. Include relevant examples, clarify mechanics:

        - **Game Rule Explanation**: Offer precise details on the relevant game rules, mechanics, or actions related to the question.
        - Boardy's Response must me MAXIMUM {MAX_TOKENS} words long, so adapt the response accodingly.
        """

    prompt = st.chat_input("How can I play this game?")
    if prompt:
        # Display user message in chat container
        with st.chat_message("user"):
            st.markdown(prompt)
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})

        # Dynamically set metadata filter based on selected game
        metadata_filter = {'key': 'metadata.game_name', 'value': st.session_state.selected_game}

        # Retrieve query and game details
        context, game_id = retrieve_query(query = prompt, embedding_model=st.session_state.embedding_model, qdrant_client=st.session_state.qdrant_client, vector_store=st.session_state.vector_store, metadata_filter=metadata_filter, k=5)
        name, description = get_game_details(game_id)

        # Call the async function to stream the response
        stream_response(prompt, context, description, name, st.session_state.llm, st.session_state.parser)

        # # Create a response placeholder in the chat interface
        # with st.chat_message(name="Boardy", avatar='😊'):
        #     message_placeholder = st.empty()
        #     full_response = ""  # Variable to accumulate the chunks of the response

        #     for response in openai.ChatCompletion.create(
        #         model=st.session_state['openai_model'],
        #         messages=[{"role": "user", "content": context}],
        #         temperature=0.7,
        #         max_tokens=1000,
        #         messages=[{"role": m["role"], "content": m["content"]} for m in st.session_state.messages],
        #         stream=True
        #     )
        #     chain = PromptTemplate(template=template_string, input_variables=["context", "question"]) | llm | parser

        #     # Stream the model's output chunk by chunk
        #     for chunk in chain.stream({"context": context, "question": prompt, "description": description, "name": name}):
        #         full_response += chunk  # Append each chunk to the full response
        #         message_placeholder.markdown(full_response)  # Update the placeholder with the partial response

        #     # Add the assistant's complete response to the chat history
        #     st.session_state.messages.append({"role": "assistant", "content": full_response})
