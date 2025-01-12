import streamlit as st

def ingestion_page():
    st.title("PDF Upload Page")

    # File uploader for PDF
    uploaded_file = st.file_uploader("Drag and drop a PDF file here", type="pdf")

    if uploaded_file is not None:
        st.success("File uploaded successfully!")
        #add code here to process the PDF file if needed

    # Button to go back to chatbot page
    if st.button("Go to Chatbot Page"):
        st.session_state.page = 'chatbot'
        st.rerun()
