import streamlit as st
from pymongo import MongoClient
from streamlit_cookies_manager import EncryptedCookieManager
from utils.utils_funcs import read_token_from_file
import bcrypt
from cookie_manager import CookieManager


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

    def hash_password(self, password):
        return bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())

    def verify_password(self, password, hashed):
        return bcrypt.checkpw(password.encode('utf-8'), hashed)

    def add_user(self, username, password):
        if self.users_collection.find_one({"username": username}):
            return False, "User already exists."
        hashed = self.hash_password(password)
        self.users_collection.insert_one({"username": username, "password": hashed})
        return True, "User created successfully."

    def authenticate_user(self, username, password):
        user = self.users_collection.find_one({"username": username})
        if user and self.verify_password(password, user["password"]):
            CookieManager().set_cookie("username", username)
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

        if st.button("Sign Up"):
            if password != confirm_password:
                st.error("Passwords do not match.")
            else:
                success, message = self.add_user(username, password)
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
