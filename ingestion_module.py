
def ingestion_page():
    import streamlit as st
    from utils.ingestion_manager import IngestionManager

    st.title("PDF Upload Page")

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
                    COLLECTION_NAME = "automatic_ingestion"
                    ingestion_manager = IngestionManager(QDRANT_KEY, OPENAI_API_KEY, QDRANT_CLOUD, COLLECTION_NAME)

                    to_ingest = ingestion_manager.ingest_pdfs(uploaded_file)

                    ingestion_manager.add_embeddings_to_qdrant(to_ingest, COLLECTION_NAME)
                    #deve aggiungere anche il nome su mongo così poi lo recupero per la lista dei giochi

                except Exception as e:
                    st.error(f"An error occurred while processing the PDF: {e}")

                st.success("Ingestion complete!")
        #add code here to process the PDF file if needed
       

    # Button to go back to chatbot page
    if st.button("Go to Chatbot Page"):
        st.session_state.page = 'chatbot'
        st.rerun()
