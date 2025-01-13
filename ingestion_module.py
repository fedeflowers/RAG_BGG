
def ingestion_page():
    import streamlit as st
    from utils.ingestion_manager import IngestionManager

    st.title("PDF Upload Page")

    # Select collection name
    st.write("Select a collection name for the PDF ingestion:")
    # collection_name = st.text_input("Collection Name", value="automatic_ingestion")

    # File uploader for PDF
    uploaded_file = st.file_uploader("Drag and drop a PDF file here", type="pdf")

    if uploaded_file is not None:
        st.success("File uploaded successfully!")
        if st.button("Start Ingestion"):
            with st.spinner("Ingestion in progress..."):
                try:
                    QDRANT_KEY = "keys\qdrant.txt"
                    OPENAI_API_KEY = "keys\OpenAI.txt"
                    QDRANT_CLOUD =  "keys\qdrant_URL.txt"
                    COLLECTION_MONGO = st.session_state.mongo_collection_games
                    COLLECTION_QDRANT = st.session_state.collection_qdrant
                    ingestion_manager = IngestionManager(path_qdrant_key=QDRANT_KEY,
                                                        path_openai_key= OPENAI_API_KEY,
                                                        path_qdrant_cloud=QDRANT_CLOUD,
                                                        collection_mongo= COLLECTION_MONGO)
                    print("porca madonna")

                    to_ingest = ingestion_manager.ingest_pdfs(uploaded_file)

                    ingestion_manager.add_embeddings_to_qdrant(to_ingest, COLLECTION_QDRANT)

                except Exception as e:
                    st.error(f"An error occurred while processing the PDF: {e}")

                st.success("Ingestion complete!")

       

    # Button to go back to chatbot page
    if st.button("Go to Chatbot Page"):
        st.session_state.page = 'chatbot'
        st.rerun()
