import pandas as pd
import time
import json
from prompts import get_hosp_qa_prompt
from constants import *
from utils import get_gemini
import multiprocessing
from secret_keys import GEMINI_API_KEY
from find_fds import find_functional_dependencies


model = None

fields_to_avoid = ["measurename", "phonenumber", "providernumber"]

import re


def extract_question(text):
    match = re.search(r"Question:\s*(.*)", text)
    if match:
        return match.group(1)
    return None


def send_request(prompt):
    global model
    retry_count = 0
    prompt_parts = [prompt]

    while retry_count < MAX_RETRY:
        try:
            response = model.generate_content(prompt_parts)
            # print(response.text)
            return response.text.replace("\n", "").strip()
        except Exception as e:
            print(f"Error: {e}")
            time.sleep(RETRY_DELAY)
            retry_count += 1

    return None


def gen_qa_template(fd_name, target_name):
    pass


def gen_qa(task):

    fd_name, fd_value, target_name, target_value = task

    # print(fd_name, fd_value, target_name)

    # prompt = get_hosp_qa_prompt(fd_name, fd_value, target_name)

    # response = extract_question(send_request(prompt))
    question = f"What is {target_name} associated with {fd_name} {fd_value} ?"

    return (question, target_value)


def compute_tasks(df: pd.DataFrame, fds: list[tuple]):
    tasks = []
    for ind, row in df.iterrows():
        for fd in fds:
            if fd[0] in fields_to_avoid or fd[1] in fields_to_avoid:
                continue
            tasks.append((fd[0], row[fd[0]], fd[1], row[fd[1]]))

    # tasks = list(set(tasks))

    n_workers = 5

    print(f"# of tasks {len(tasks)}")

    with multiprocessing.Pool(n_workers) as pool:
        qa_and_s = pool.map(gen_qa, tasks)

    return qa_and_s


def gen_and_save_json(qas):
    json_array = []
    for qa in qas:
        obj = {}
        obj["question"] = qa[0]
        obj["answer"] = qa[1]
        json_array.append(obj)

    with open("qa_data.json", "w") as file:
        json.dump(json_array, file, indent=4)


if __name__ == "__main__":
    data_file_path = f"{table_data_folder_path}{hosp_20k_path}"
    df = pd.read_csv(data_file_path)

    fds = find_functional_dependencies(df)

    model = get_gemini(GEMINI_API_KEY)
    QAs = compute_tasks(df, fds)

    gen_and_save_json(QAs)
