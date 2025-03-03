# RAG for board games

## Requirements
Openai API key is needed, docker facilitates to run the app, otherwise install mongoDB and Qdrant locally and change settings to get access to these two locally

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

## How to Start the Project with Docker

1. **Create external network:**
   ```bash
   docker network create rag_bgg_rag_network
   
2. **Build and run image:**
   ```bash
   docker-compose up --build

3. **If image is already built run**
   ```bash
   docker-compose up

4. **Access streamlit at**
   ```bash
   http://localhost:8501/

#### Required at least 8Gb of RAM given to docker


# APP functionalities

### Sign Up
Users need to create an account before accessing the app.

![Sign Up](bot/readme_images/sign_up.png)

### Login
Log in with your credentials to start using the app.

![Login](bot/readme_images/login.png)

### Upload PDF
After logging in, upload a PDF file containing the board game rules.

![Upload PDF](bot/readme_images/upload_pdf.png)

### Start Questions
Begin your interaction with the app by answering initial questions.

![Start Questions](bot/readme_images/start_questions.png)

### Adjust Yuor Bot
Switch your openai bot and temperature

![Openai Bot](bot/readme_images/openai_settings.png)


### Reference Feature
Access detailed references from the ingested game rules.

![Reference](bot/readme_images/reference.png)

### Saved Chats
View your saved chats, retrieve them based on file names, or delete them if no longer needed.

![Saved Chats](bot/readme_images/saved_chats.png)

### No Libraries
If no games are ingested, the app will notify you accordingly.

![No Libraries](bot/readme_images/no_libraries.png)
