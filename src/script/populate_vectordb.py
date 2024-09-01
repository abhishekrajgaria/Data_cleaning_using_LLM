import os
import argparse
import multiprocessing
from tqdm import tqdm
from constants import *

# import torch.multiprocessing as mp

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
db = Chroma(
    persist_directory=chroma_persist_directory,
    embedding_function=embeddings_model,
    collection_name="hosp_data_try4",
)


def add_to_db(doc):
    db.add_documents(doc)


def add_file_data_to_db():
    txt_files = []
    for dirpath, _, filenames in os.walk(file_data_folder_path):
        for filename in filenames:
            if filename.endswith(".txt"):
                txt_files.append(os.path.join(dirpath, filename))
    # print(len(txt_files))
    # print(txt_files)
    tasks = []
    for file_path in tqdm(txt_files, desc="Processing files", unit="file"):
        with open(file_path, "r") as file:
            for line in tqdm(
                file,
                desc=f"Reading lines in {os.path.basename(file_path)}",
                unit="line",
                leave=False,
            ):
                chunk = [line]
                docs = splitter.create_documents(chunk)
                # print(docs)
                for doc in docs:
                    # print(doc)
                    # tasks.append([doc])
                    db.add_documents([doc])
    # n_workers = 5
    # if multiprocessing.get_start_method(allow_none=True) is None:
    #     multiprocessing.set_start_method("spawn")

    # print(f"# of tasks {len(tasks)}")

    # with multiprocessing.Pool(n_workers) as pool:
    #     pool.map(add_to_db, tasks)


if __name__ == "__main__":
    # multiprocessing.set_start_method("spawn", force=True)

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--create_db", action="store_true", help="Flag to create something"
    )

    args = parser.parse_args()

    # if args.create:

    add_file_data_to_db()
