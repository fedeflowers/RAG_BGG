import streamlit as st
from pymongo import MongoClient
from streamlit_cookies_manager import EncryptedCookieManager
from utils.utils_funcs import read_token_from_file
# import bcrypt
from utils.cookie_manager import CookieManager
from openai_api_key_verifier import verify_api_key



class UserAuthApp:
    def __init__(self):
        # MongoDB setup
        self.client = MongoClient("mongodb://localhost:27017/")
        self.db = self.client["user_database"]
        self.users_collection = self.db["users"]

        # Initialize session state
        if 'authenticated' not in st.session_state:
            st.session_state.authenticated = False
        if 'user' not in st.session_state:
            st.session_state.user = None
        if 'page' not in st.session_state:
            st.session_state.page = 'login'

    # def hash_password(self, password):
    #     return bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())

    # def verify_password(self, password, hashed):
    #     return bcrypt.checkpw(password.encode('utf-8'), hashed)

    def add_user(self, username, password, openai_key):
        if self.users_collection.find_one({"username": username}):
            return False, "User already exists."
        # hashed = self.hash_password(password)
        self.users_collection.insert_one({"username": username, "password": password, "openai_key": openai_key})
        return True, "User created successfully."

    def authenticate_user(self, username, password):
        user = self.users_collection.find_one({"username": username})
        if user and password == user["password"]:
            CookieManager().set_cookie("username", username)
            st.session_state.openai_key = user["openai_key"]
            return True, user
        return False, None

    def login_page(self):
        st.title("Login")
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")

        if st.button("Login"):
            success, user = self.authenticate_user(username, password)
            if success:
                st.session_state.authenticated = True
                st.session_state.user = user
                st.session_state.page = 'chatbot'


                st.rerun()
            else:
                st.error("Invalid username or password.")

    def signup_page(self):
        st.title("Sign Up")
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        confirm_password = st.text_input("Confirm Password", type="password")
        openai_key = st.text_input("OpenAI API Key", type="password")

        if st.button("Sign Up"):
            psw_check = True
            user_check = True
            api_check = verify_api_key(openai_key)
            if not api_check:
                st.error("Invalid OpenAI API Key.")
            if password != confirm_password:
                st.error("Passwords do not match.")
                psw_check = False
            if self.users_collection.find_one({"username": username}):
                st.error("Username already exists.")
                user_check = False

            # Verify if the API key is valid
            if psw_check and user_check and api_check:
                success, message = self.add_user(username, password, openai_key)
                if success:
                    st.success(message)
                    st.session_state.page = 'login'
                else:
                    st.error(message)

    def main_app(self):
        st.title("Main Application")
        st.write(f"Welcome, {st.session_state.user['username']}!")
        if st.button("Logout"):
            self.logout()

    def logout(self):
        CookieManager().remove_cookie("username")
        st.session_state.authenticated = False
        st.session_state.user = None
        st.session_state.page = 'login'

    def check_cookie_login(self):
        # Check if a user is logged in via cookies
        username = CookieManager.get_cookie("username")
        if username and not st.session_state.authenticated:
            user = self.users_collection.find_one({"username": username})
            if user:
                st.session_state.authenticated = True
                st.session_state.user = user
                st.session_state.page = 'chatbot'

# # Run the app
# if __name__ == "__main__":
#     app = UserAuthApp()
#     app.run()
