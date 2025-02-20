
def ingestion_page():
    import streamlit as st
    from utils.ingestion_manager import IngestionManager
    from utils.utils_funcs import retrieve_games_list

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
                    to_ingest = st.session_state.ingestion_manager.ingest_pdfs(uploaded_file)

                    st.session_state.ingestion_manager.add_embeddings_to_qdrant(to_ingest, st.session_state.collection_qdrant)

                except Exception as e:
                    st.error(f"An error occurred while processing the PDF: {e}")

                st.success("Ingestion complete!")
                st.session_state.selected_game = retrieve_games_list(st.session_state.mongo_collection_games)[-1]

       

    # Button to go back to chatbot page
    if st.button("Go to Chatbot Page"):
        st.session_state.page = 'chatbot'
        st.rerun()
