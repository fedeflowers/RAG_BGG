# RAG for board games

## Requirements
Openai API key is needed, and docker to run the app

## Overview
This application is designed to help users interact with board game rulebooks efficiently. By uploading a PDF file of a game's rules, users can ask questions, retrieve saved chats, and reference game details with ease.

## Features
- **User Authentication:** Secure sign-up and login process to manage user accounts.
- **PDF Upload:** Ingest board game rulebooks in PDF format.
- **Interactive Q&A:** Start with initial questions to guide user interaction.
- **Saved Chats:** View, retrieve, and delete past chat sessions based on ingested files.
- **Reference Management:** Easily reference specific details from uploaded games.
- **No Library Handling:** Clearly indicates when no games have been ingested.

## Getting Started

### Sign Up
Users need to create an account before accessing the app.

![Sign Up](readme_images/sign_up.png)

### Login
Log in with your credentials to start using the app.

![Login](readme_images/login.png)

### Upload PDF
After logging in, upload a PDF file containing the board game rules.

![Upload PDF](readme_images/upload_pdf.png)

### Start Questions
Begin your interaction with the app by answering initial questions.

![Start Questions](readme_images/start_questions.png)

### Adjust Yuor Bot
Switch your openai bot and temperature

![Openai Bot](readme_images/openai_settings.png)


### Reference Feature
Access detailed references from the ingested game rules.

![Reference](readme_images/reference.png)

### Saved Chats
View your saved chats, retrieve them based on file names, or delete them if no longer needed.

![Saved Chats](readme_images/saved_chats.png)

### No Libraries
If no games are ingested, the app will notify you accordingly.

![No Libraries](readme_images/no_libraries.png)

## How to Start the Project with Docker

1. **Build the Docker Image:**
   ```bash
   docker build -t app-name .
