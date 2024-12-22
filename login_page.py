import streamlit as st
from pymongo import MongoClient
import bcrypt

# MongoDB setup
client = MongoClient("mongodb://localhost:27017/")
db = client["user_database"]
users_collection = db["users"]

# Authentication Functions
def hash_password(password):
    return bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())

def verify_password(password, hashed):
    return bcrypt.checkpw(password.encode('utf-8'), hashed)

def add_user(username, password):
    if users_collection.find_one({"username": username}):
        return False, "User already exists."
    hashed = hash_password(password)
    users_collection.insert_one({"username": username, "password": hashed})
    return True, "User created successfully."

def authenticate_user(username, password):
    user = users_collection.find_one({"username": username})
    if user and verify_password(password, user["password"]):
        return True, user
    return False, None

# Your Existing Application Logic
def main_app():
    st.title("Main Application")
    st.write("This is your existing app logic.")
    # Add your app's features and components here.
    if st.button("Logout"):
        st.session_state.authenticated = False
        st.session_state.page = 'refresh'

# Login and Signup Logic
def login_page():
    st.title("Login")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    if st.button("Login"):
        success, user = authenticate_user(username, password)
        if success:
            st.session_state.authenticated = True
            st.session_state.user = user
            st.session_state.page = 'refresh'
        else:
            st.error("Invalid username or password.")

def signup_page():
    st.title("Sign Up")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    confirm_password = st.text_input("Confirm Password", type="password")
    if st.button("Sign Up"):
        if password != confirm_password:
            st.error("Passwords do not match.")
        else:
            success, message = add_user(username, password)
            if success:
                st.success(message)
                st.session_state.page = 'refresh'
            else:
                st.error(message)

