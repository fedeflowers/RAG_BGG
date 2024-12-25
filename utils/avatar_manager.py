# import streamlit as st
# from streamlit_cookies_manager import EncryptedCookieManager
# from utils.utils_funcs import read_token_from_file

# class AvatarManager:
#     def __init__(self):
#         # Initialize the cookie manager
#         self.cookies = EncryptedCookieManager(
#             prefix="avatar_manager",
#             password=read_token_from_file("keys/psw_cookies_manager.txt"),  # Use a strong password or environment variable
#         )
#         if not self.cookies.ready():
#             st.stop()

#         # Define users and their avatars
#         self.users = {
#             "User1": "👩‍💻",
#             "User2": "👨‍💻",
#             "User3": "🧑‍🎨",
#             "User4": "👩‍🔬",
#         }
#         self.load_user_data()

#     @st.cache_data
#     def load_user_data(_self):
#         """Load user data from cookies."""
#         selected_user = _self.cookies.get("selected_user", "User1")
#         if "selected_user" not in st.session_state:
#             st.session_state.selected_user = selected_user

#     @st.cache_resource
#     def save_user_data(_self, selected_user):
#         """Save user data to cookies."""
#         _self.cookies["selected_user"] = selected_user
#         _self.cookies.save()

#     def display_avatar_icon(self):
#         """Display the main avatar icon at the top-left corner."""
#         def display_overlay():
#             st.session_state.show_overlay = True
#         # current_user = st.session_state.selected_user
#         current_user = self.cookies.get("selected_user", "User1")
#         avatar = self.users[current_user]

#         # Create a top-left button to show the current avatar
#         st.markdown(
#             f"""
#             <div style="position: fixed; top: 10px; left: 10px; z-index: 1000;">
#                 <button style="border: none; background: transparent; cursor: pointer;"
#                         onclick=display_overlay()">
#                     <span style="font-size: 30px;">{avatar}</span>
#                 </button>
#             </div>
#             """,
#             unsafe_allow_html=True,
#         )

#     def display_avatar_selection_page(self):
#         """Display a specialized page to select an avatar."""
#         if "show_overlay" not in st.session_state:
#             st.session_state.show_overlay = True  # Control the visibility of the overlay

#         # Render overlay if active
#         if st.session_state.show_overlay:
#             st.markdown(
#                 """
#                 <div id="avatar-container" style="position: fixed; top: 0; left: 0; width: 100%; height: 100%; 
#                     background-color: rgba(0, 0, 0, 0.8); color: white; display: block; z-index: 1000;">
#                     <div style="margin: 10% auto; padding: 20px; background: #d9d9d9; color: #333; 
#                                 width: 50%; border-radius: 10px; text-align: center;">
#                         <h2 style="color: #000;">Select an Avatar</h2>
#                         <div id="avatar-options" style="display: flex; flex-wrap: wrap; justify-content: center;">
#                         </div>
#                         <button onclick="document.getElementById('avatar-container').style.display='none';"
#                                 style="margin-top: 20px; padding: 10px 20px; font-size: 16px; 
#                                     background-color: #007bff; color: white; border: none; 
#                                     border-radius: 5px; cursor: pointer;">
#                             Close
#                         </button>
#                     </div>
#                 </div>
#                 """,
#                 unsafe_allow_html=True,
#             )


# # def toggle_overlay():
# #             st.session_state.show_overlay = not st.session_state.show_overlay

# #         st.button("Apri Sovraimpressione", on_click=toggle_overlay)

# #         # Simulazione della sovraimpressione
# #         if st.session_state.show_overlay:
# #             with st.container():
# #                 st.markdown(
# #                     """
# #                     <div style="
# #                         position: fixed;
# #                         top: 20%;
# #                         left: 25%;
# #                         width: 50%;
# #                         height: 50%;
# #                         background-color: white;
# #                         border: 2px solid black;
# #                         border-radius: 10px;
# #                         box-shadow: 0px 4px 8px rgba(0, 0, 0, 0.2);
# #                         z-index: 10;
# #                         padding: 20px;
# #                         text-align: center;
# #                     ">
# #                         <h2>Pagina Sovraimpressione</h2>
# #                         <p>Questo è un contenuto in sovraimpressione.</p>
# #                     </div>
# #                     """,
# #                     unsafe_allow_html=True,
# #                 )
# #                 st.button("Chiudi", on_click=toggle_overlay)