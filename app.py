if __name__ == '__main__':
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
    from langchain.embeddings import OpenAIEmbeddings
    from qdrant_client.http import models
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

    # MAX_TOKENS = 1200
    TEMPERATURE=0.2
    NUM_DOCS_RETRIEVED = 10
    SIMILARITY_THRESHOLD = 0.9
    DECAY = 0.05
    MIN_DOCUMENTS = 15
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
        st.success("App ready!")

    # Game options for dropdown -> GENERATE THIS DINAMICALLY! (find another way to select boardgame)
    game_options = ["Unlock Secret Adventures", "The Mind Extreme", "SpellBook", "Chimera Station"]
    selected_game = st.selectbox("Select a game:", game_options, index=game_options.index(st.session_state.selected_game))

    # Store the selected game in session state
    st.session_state.selected_game = selected_game

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Capture the user input for the question

    def stream_response(prompt, context, description, name, llm, parser, template, input_variables):
        message_placeholder = st.empty()
        full_response = ""

        # Create a Langchain chain
        try:
            chain = PromptTemplate(template=template, input_variables=input_variables) | llm | parser

            # Stream the model's output chunk by chunk
            for chunk in chain.stream({"context": context, "question": prompt, "description": description, "name": name}):
                full_response += chunk
                message_placeholder.markdown(full_response)

        except Exception as e:
            st.error(f"An error occurred: {str(e)}")

        # Add the assistant's complete response to the chat history
        if "messages" not in st.session_state:
            st.session_state.messages = []
        st.session_state.messages.append({"role": "assistant", "content": full_response, name:"Boardy"})

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
    #magari cambiare sto prompt con un altro che oltre ad inserire le immagini le riesca anche ad allinear con il testo un attimino meglio
    # template_string_final_substitution = """system: Given the context: {context}, and the following list of image IDs, that also describe shortly the image itself: {list_images}, enhance the context by incorporating relevant image IDs as visual recommendations for the final user. 
    # Ensure that the original content remains unchanged, except for the addition of image references. Aim for a natural integration of the images into the text.
    # """
    template_string_final_substitution = """system: Given the context: {context}, and the following list of image IDs {list_images},
    Enhance the context by naturally incorporating relevant image IDs
    as visual recommendations for the user, the added images should be explained by the context, to not add images that are not mentioned in the context. Ensure that the original content remains unchanged except for the addition of image references. 
    Use `{list_images}` to refer to images where they are specifically relevant, integrating them into the text without separating by commas or any punctuation. To reference an image, use the format `![<list_images_key>](list_images_key)` to embed it directly.
    Avoid unnecessary references, and exclude a final period if the sentence ends with an image. Only add images that enhance understanding and align perfectly with the context.
    If you are terminating the paragraph or sentence with an image, do not add a final period like DO this way instead `bla bla bal <image>`
    ***VERY IMPORTANT*** DO NOT ADD IMAGES THAT DO NOT REFER TO {context}.
    ***VERY IMPORTANT*** DO NOT USE ALL IMAGES. ONLY ADD IMAGES THAT ARE RELEVANT.
    """



    # template_string_final_substitution = """system: Given the context: {context}, and the following list of image IDs, that also describe shortly the image itself: {list_images}, enhance the context by incorporating relevant image IDs as visual recommendations for the final user. 
    # Ensure that the original contet remains unchanged, except for the addition of image references. Aim for a natural integration of the images into the text.
    # try to seamlessly incorporate the images into the context, do not separate images by commas, if you end a paragraph or sentece with an image to NOT add the final point,
    # for example, if you end a sentece "bla bla bal <image>" then do not add the '.' at the end. They are preferred at the end of a sentence or of a paragraph. Do not add not needed images,
    # Add only images that enhance the understanding of the context and that the {list_images} values reflect that context perfectly.
    # """

    prompt = st.chat_input(" ")
    if prompt:
        # Display user message in chat container
        with st.chat_message("user"):
            st.markdown(prompt)
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})

        # Dynamically set metadata filter based on selected game
        metadata_filter = {'key': 'metadata.game_name', 'value': st.session_state.selected_game}
        context = []
        while SIMILARITY_THRESHOLD > DECAY and len(context) < MIN_DOCUMENTS:
            try:
                context, game_id, image_metadata = retrieve_query(prompt, NUM_DOCS_RETRIEVED, st.session_state.embedding_model,
                                                                st.session_state.qdrant_client, st.session_state.vector_store,
                                                                metadata_filter = metadata_filter,
                                                                similarity_threshold=SIMILARITY_THRESHOLD)
                SIMILARITY_THRESHOLD -= DECAY
                if len(context) >= MIN_DOCUMENTS:
                    break
            except Exception as e:
                print(f"Error: {e}")
                SIMILARITY_THRESHOLD -= DECAY
                print("setting similarity threshold to", SIMILARITY_THRESHOLD)
        name, description = get_game_details(game_id)

        # # # Call the async function to stream the response
        intermediate_response = batch_response(prompt, context, description, name, st.session_state.llm, st.session_state.parser, template_string, ["context", "question", "description", "name"])
        input_variables = ["context", "list_images"]

        # Second part of the model, actual streaming with images
        buffer = ""
        full_response = ""
        message_placeholder = st.empty()  # Placeholder for displaying the response
        try:
            chain = PromptTemplate(template=template_string_final_substitution, input_variables=input_variables) | st.session_state.llm | st.session_state.parser

            # Initialize an empty buffer and a full response
            buffer = ""
            full_response = ""

            # Stream the model's output chunk by chunk
            for chunk in chain.stream({"context": intermediate_response, "list_images": list(image_metadata.keys())}):
                buffer += chunk  # Accumulate chunk into the buffer

                # Check if buffer contains any complete image placeholders
                while re.search(r"!\[.*?\]\(.*?\)", buffer):
                    match = re.search(r"!\[.*?\]\((.*?)\)", buffer)  # Look for the next placeholder
                    placeholder_key = match.group(1)

                    if placeholder_key in image_metadata:
                        split_text = buffer.split(f"![{placeholder_key}]({placeholder_key})", 1)
                        # Display only the new text since last update
                        new_text = split_text[0]
                        if new_text:
                            st.markdown(new_text)
                            full_response += new_text

                        # Decode and display the image
                        image_data = base64.b64decode(image_metadata[placeholder_key])
                        image = Image.open(BytesIO(image_data))
                        st.image(image)  # Display the decoded image

                        buffer = split_text[1]  # Update buffer with remaining text
                    else:
                        break  # Stop if placeholder key not found in metadata

        except Exception as e:
            st.error(f"An error occurred: {str(e)}")

        # Add the assistant's complete response to the chat history
        if "messages" not in st.session_state:
            st.session_state.messages = []
        st.session_state.messages.append({"role": "assistant", "content": full_response, "name": "Boardy"})





