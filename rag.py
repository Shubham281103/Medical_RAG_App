from langchain_core.prompts import PromptTemplate
from langchain_community.llms import CTransformers
from langchain.chains import LLMChain
from langchain_huggingface import HuggingFaceEmbeddings
from fastapi import FastAPI, Request, Form, Response
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.encoders import jsonable_encoder
from langchain_community.vectorstores import Chroma
import os
import json

app = FastAPI()
templates = Jinja2Templates(directory="templates")

# Mount static if it exists
if os.path.exists("static"):
    app.mount("/static", StaticFiles(directory="static"), name="static")

# LLM Configuration
local_llm = "models/meditron-7b.Q4_K_M.gguf"

config = {
    'max_new_tokens': 256,
    'context_length': 512,
    'repetition_penalty': 1.1,
    'temperature': 0.1,
    'top_k': 50,
    'top_p': 0.9,
    'stream': True,
    'threads': int(os.cpu_count() / 2)
}

llm = CTransformers(
    model=local_llm,
    model_type="llama",
    lib="avx2",
    **config
)

print("LLM Initialized...")

# Prompt template
prompt_template = """Use the following pieces of information to answer the user's question.
If you don't know the answer, just say that you don't know, don't try to make up an answer.

Context: {context}
Question: {question}

Only return the helpful answer below and nothing else.
Helpful answer:
"""

prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

# Embeddings and Chroma setup
embeddings = HuggingFaceEmbeddings(model_name="NeuML/pubmedbert-base-embeddings")
db = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)
retriever = db.as_retriever(search_kwargs={"k": 2})

# Token approximation
def approximate_token_count(text: str) -> int:
    return len(text.split())

# Truncate context until token count fits
def truncate_context(context: str, question: str, token_limit: int = 512, buffer: int = 100) -> str:
    context_lines = context.splitlines()
    truncated_context = ""
    for line in context_lines:
        temp_context = f"{truncated_context}\n{line}".strip()
        input_text = prompt_template.format(context=temp_context, question=question)
        if approximate_token_count(input_text) > token_limit - buffer:
            break
        truncated_context = temp_context
    return truncated_context.strip()

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/get_response")
async def get_response(query: str = Form(...)):
    # Retrieve docs manually
    docs = retriever.invoke(query)
    combined_context = "\n\n".join([doc.page_content for doc in docs])
    truncated_context = truncate_context(combined_context, query)

    # Build final chain
    chain = LLMChain(llm=llm, prompt=prompt, verbose=True)
    print(f"\n>>> Final Prompt Token Count: {approximate_token_count(prompt_template.format(context=truncated_context, question=query))}")
    print(f"\n>>> Prompt Preview:\n{truncated_context[:300]}...\n")

    # Run model with truncated context
    response = chain.invoke({"context": truncated_context, "question": query})
    answer = response["text"] if isinstance(response, dict) else response

    # Get source doc info manually
    source_document = docs[0].page_content if docs else "No source document found."
    doc = docs[0].metadata.get('source', 'N/A') if docs else "N/A"

    # Package response
    response_data = jsonable_encoder(json.dumps({
        "answer": answer,
        "source_document": source_document,
        "doc": doc
    }))
    
    return Response(response_data)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)