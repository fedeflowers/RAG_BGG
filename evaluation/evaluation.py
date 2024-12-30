from qdrant_client import models
from tqdm import tqdm
import pickle
import re
from templates import TemplatesCatalog, Template
from vector_db import VectorDB

import logging

logger = logging.getLogger("Dataset Generation")
logger.setLevel(logging.DEBUG)

# Create a console handler
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)  # Set the level for the handler

# Create a formatter and set it for the handler
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)

# Add the handler to the logger
logger.addHandler(console_handler)

def format_docs(docs):
    return "\n\n".join(doc for doc in docs)



def generate_eval_dataset(model,emb,db:VectorDB,collection_name,title:str="Skull King"):

    if title:
        filter = models.Filter(
            must=[
                models.FieldCondition(
                    key='metadata.title',
                    match=models.MatchValue(value=title),
                )
            ]
        )
    else:
        filter = None

    chunks = db.scroll(collection_name,filter)
    eval_template = Template(TemplatesCatalog.EVAL_ANSWER.value,model,["page_content","metadata"])
    chatbot_template = Template(TemplatesCatalog.BG_CHATBOT.value,model,["title","context","question"])
    
    logger.info("Generating queries....")
    queries = []
    for chunk in tqdm(chunks):
        res = chunk
        resp = eval_template.invoke({"page_content":chunk["page_content"],"metadata":str(chunk["metadata"])})
        res["queries"] = resp
        queries.append(res)
    logger.info("Queries generated.")

    retriever = db.get_retriever(collection_name,emb)

    logger.info("Generating answers...")
    answers = []
    for query in tqdm(queries):
        questions_string = query["queries"]
        questions = questions_string.split("\n")
        for question in questions:
            if question == "No questions can be generated.":
                continue
            question = re.sub(r"([0-9]\. )","",question)
            contexts = [doc.page_content for doc in retriever.invoke(question)]
            answer = chatbot_template.invoke({"title":query["metadata"]["title"], "context": format_docs(contexts), "question": question})
            answers.append({"contexts": contexts, "question": question,"answer":answer})
    logger.info("Answers generated.")

    #TODO: magari salviamo tutto su un db
    file_name_answer = "_".join(el.lower() for el in title.split(" "))+"_answer.pkl"

    with open(file_name_answer,"wb") as f:
        pickle.dump(queries,f)

    logger.info("Asnwers Saved!")

    