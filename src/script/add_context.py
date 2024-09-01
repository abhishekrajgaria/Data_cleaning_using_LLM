import os
import json
import argparse
from tqdm import tqdm
from constants import *

from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size=786, chunk_overlap=50, length_function=len, add_start_index=True
)

modelPath = "sentence-transformers/sentence-t5-large"
custom_cache_dir = "/scratch/general/vast/u1471428/dcp/models"
chroma_persist_directory = "/scratch/general/vast/u1471428/dcp/chroma_db2"

embeddings_model = HuggingFaceEmbeddings(
    model_name=modelPath, cache_folder=custom_cache_dir
)


def get_context(query):
    results = db.similarity_search_with_score(query, k=5)
    contexts = ""
    flag = False
    for result in results:
        context = result[0].page_content
        score = result[1]
        if score > 0.1:
            flag = True
            contexts += context + " "
    if flag:
        return context
    return None


def custom_relevance_score_fn(similarity_score: float) -> float:
    # Example calculation (customize as needed)
    relevance_score = 1 / (1 + similarity_score)
    return relevance_score


db = Chroma(
    persist_directory=chroma_persist_directory,
    embedding_function=embeddings_model,
    collection_name="hosp_data_try4",
)

retriever = db.as_retriever(search_type="mmr", search_kwargs={"k": 3, "fetch_k": 5})


def add_context(json_array):
    json_context_data = []
    none_count = 0
    for json_obj in json_array:
        query = json_obj["question"]

        context = get_context(query)
        if not context:
            none_count += 1
            continue

        query_with_context = f"question: {query} context: {context}"
        obj = {}
        obj["question"] = query
        obj["answer"] = json_obj["answer"]
        json_context_data.append(obj)
    print("none count", none_count)
    with open("qa_with_context_data.json", "w") as file:
        json.dump(json_context_data, file)


if __name__ == "__main__":
    qa_json_filepath = "qa_data.json"

    json_array = None

    with open(qa_json_filepath, "r") as file:
        json_array = json.load(file)

    add_context(json_array)
