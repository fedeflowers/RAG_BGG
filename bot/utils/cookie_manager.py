from streamlit_cookies_controller import CookieController

class CookieManager:
    # Static variable to store the CookieController
    controller = CookieController()

    @staticmethod
    def get_cookie(cookie_name):
        # Get the cookie using the static controller
        cookie = CookieManager.controller.get(cookie_name)
        return cookie
        # print(f"Cookie retrieved: {cookie_name} = {cookie}")
    @staticmethod
    def set_cookie(cookie_name, value):
        # Get the cookie using the static controller
        CookieManager.controller.set(cookie_name, value)
        # print(f"Cookie retrieved: {cookie_name} = {cookie}")
    @staticmethod
    def remove_cookie(cookie_name):
        # Remove the cookie using the static controller
        CookieManager.controller.remove(cookie_name)
        # print(f"Cookie removed: {cookie_name}")
