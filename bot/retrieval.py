from qdrant_client import QdrantClient
from langchain_qdrant import QdrantVectorStore
from langchain_huggingface import HuggingFaceEmbeddings

from transformers import pipeline
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

READER_MODEL_NAME = "HuggingFaceH4/zephyr-7b-beta"
EMBEDDING_MODEL_NAME = "thenlper/gte-small"

def read_token_from_file(file_path="token.txt"):
    with open(file_path, "r") as file:
        return file.read().strip()

def retrieve_query(query, k = 1):
    '''
    retrieve query from Qdrant, k = number of docs to look for
    '''
    URL=read_token_from_file("keys/qdrant_URL.txt") 
    API_KEY=read_token_from_file("keys/qdrant.txt")

    # Initialize Qdrant client
    qdrant_client = QdrantClient(
        url=URL, 
        api_key=API_KEY,
    )

    embedding_model = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        multi_process=True,  # Enable multiprocessing
        model_kwargs={"device": "cuda"},
        encode_kwargs={"normalize_embeddings": True},  # Set `True` for cosine similarity
    )


    vector_store = QdrantVectorStore(
    client=qdrant_client,
    collection_name="test_1",
    embedding=embedding_model,
    )

    results = vector_store.similarity_search(query=query,k=k)
    for doc in results:
        print(f"* {doc.page_content} [{doc.metadata}]")




bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)
model = AutoModelForCausalLM.from_pretrained(READER_MODEL_NAME, quantization_config=bnb_config)
tokenizer = AutoTokenizer.from_pretrained(READER_MODEL_NAME)

READER_LLM = pipeline(
    model=model,
    tokenizer=tokenizer,
    task="text-generation",
    do_sample=True,
    temperature=0.2,
    repetition_penalty=1.1,
    return_full_text=False,
    max_new_tokens=500,
)

prompt_in_chat_format = [
    {
        "role": "system",
        "content": """Using the information contained in the context,
give a comprehensive answer to the question.
Respond only to the question asked, response should be concise and relevant to the question.
Provide the number of the source document when relevant.
If the answer cannot be deduced from the context, do not give an answer.""",
    },
    {
        "role": "user",
        "content": """Context:
{context}
---
Now here is the question you need to answer.

Question: {question}""",
    },
]
RAG_PROMPT_TEMPLATE = tokenizer.apply_chat_template(
    prompt_in_chat_format, tokenize=False, add_generation_prompt=True
)
# print(RAG_PROMPT_TEMPLATE)

if __name__ == '__main__':
    context = "4+4 = 8 "
    final_prompt = RAG_PROMPT_TEMPLATE.format(question="What is 4+4", context=context)

    # Redact an answer
    answer = READER_LLM(final_prompt)[0]["generated_text"]
    print(answer)