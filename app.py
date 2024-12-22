if __name__ == '__main__':
    import streamlit as st
    from login_page import *
    from main_page import chatbot
    # Main Flow
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False

    if st.session_state.authenticated:
        chatbot()
    else:
        option = st.sidebar.selectbox("Choose", ["Login", "Sign Up"])
        if option == "Login":
            login_page()
        elif option == "Sign Up":
            signup_page()