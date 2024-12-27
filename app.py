if __name__ == '__main__':
    import streamlit as st
    from login_page import *
    from css import css
    from main_page import chatbot_page
    from streamlit_cookies_manager import EncryptedCookieManager
    from utils.utils_funcs import read_token_from_file
    from cookie_manager import CookieManager

    # Main Flow
    #markdown RULES CUSTOM
    st.markdown(
        css,
        unsafe_allow_html=True,
    )
    user_auth = UserAuthApp()
    
    user_auth.check_cookie_login()
    
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False
        st.session_state.page = 'login'

    if st.session_state.authenticated:
        st.session_state.page = 'chatbot'
        chatbot_page()
    else:
        option = st.sidebar.selectbox("Choose", ["Login", "Sign Up"])
        if option == "Login":
            user_auth.login_page()
        elif option == "Sign Up":
            user_auth.signup_page()