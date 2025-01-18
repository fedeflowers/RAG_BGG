css = """
    <style>
        /* Default button style */
        div.stButton > button {
            padding: 0.5rem 1rem; /* Consistent padding */
            border: 2px solid transparent; /* Default transparent border */
            font-size: 16px; /* Font size */
            background-color: gray; /* Default background */
            color: white; /* Default text color */
            transition: background-color 0.3s ease, color 0.3s ease, border 0.3s ease; /* Smooth transitions */
        }

        /* Hover style for button */
        div.stButton > button:hover {
            background-color: green !important; /* Green background on hover */
            color: white !important; /* White text on hover */
            border: 2px solid green !important; /* Green border on hover */
        }

        /* Focus (clicked) style for button */
        div.stButton > button:focus {
            background-color: green !important; /* Green background when clicked */
            color: white !important; /* White text */
            border: 2px solid green !important; /* Green border */
            outline: none !important; /* Remove default outline */
            box-shadow: none !important; /* Remove any shadow effect */
        }

        /* Targeting textarea inside stTextArea specifically */
        div.stTextArea > textarea {
            border: 2px solid gray !important; /* Default border color */
            border-radius: 5px !important; /* Rounded corners */
            padding: 0.5rem !important; /* Consistent padding */
            font-size: 16px !important; /* Text size */
            background-color: #101010 !important; /* Dark background */
            color: white !important; /* Default text color */
            transition: border 0.3s ease, background-color 0.3s ease !important; /* Smooth transitions */
            resize: none; /* Prevent resizing manually */
            overflow-y: hidden; /* Hide vertical scroll bar */
            min-height: 50px; /* Set minimum height for better UX */
        }

        /* Hover style for input fields */
        div.stTextArea > textarea:hover {
            border: 2px solid green !important; /* Green border on hover */
            background-color: #202020 !important; /* Slightly lighter background */
        }

        /* Focus style for input fields (when clicked or typing) */
        div.stTextArea > textarea:focus {
            border: 2px solid green !important; /* Green border on focus */
            background-color: #202020 !important; /* Keep the background */
            color: white !important; /* Ensure text remains visible */
            outline: none !important; /* Remove default outline */
            box-shadow: none !important; /* Remove any shadow effect */
        }

        /* Ensure the green border during invalid state */
        div.stTextArea > textarea:invalid {
            border: 2px solid green !important; /* Force green border when invalid */
        }

        /* Fixes for specific error states when clicked (active state) */
        div.stTextArea > textarea:focus:invalid {
            border: 2px solid green !important; /* Green border when focused and invalid */
        }

        /* Specifically for selected state (when clicked but not necessarily focused) */
        div.stTextArea > textarea:active {
            border: 2px solid green !important; /* Ensure green border when selected */
            background-color: #202020 !important; /* Keep the background color */
        }
    </style>

    <script>
        /* Adjust the height of the textarea based on the content */
        const textarea = document.querySelector('textarea');
        textarea.addEventListener('input', function () {
            this.style.height = 'auto';  /* Reset height to auto */
            this.style.height = (this.scrollHeight) + 'px';  /* Set height based on scrollHeight */
        });
    </script>

    """