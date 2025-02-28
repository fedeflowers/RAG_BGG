// init-mongo.js
db = db.getSiblingDB("RAG_DB");
db.createCollection("chat_messages");
db.createCollection("games");
db.createCollection("users");

db = db.getSiblingDB("local");
db.createCollection("chat_messages");
db.createCollection("startup_log");

db = db.getSiblingDB("user_database");
db.createCollection("users");
