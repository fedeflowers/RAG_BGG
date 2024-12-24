if __name__ == '__main__':
    import streamlit as st
    from login_page import *
    from css import css
    from main_page import chatbot
    # Main Flow
    #markdown RULES CUSTOM
    st.markdown(
        css,
        unsafe_allow_html=True,
    )
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