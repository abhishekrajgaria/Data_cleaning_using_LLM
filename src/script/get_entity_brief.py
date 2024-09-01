import os
import json
from datetime import datetime, timedelta
import time
import tempfile
import argparse
import pandas as pd

from utils import *
from constants import *
from secret_keys import *
from prompts import *

from openai import OpenAI

client = OpenAI(api_key=OPENAI_API_KEY2)


def check_batch_status(batch_id):
    start_time = datetime.now()
    timeout = timedelta(hours=24)

    while datetime.now() - start_time < timeout:
        batch = client.batches.retrieve(batch_id)
        if batch.status == "completed":
            return batch
        elif batch.status in ["failed", "cancelled"]:
            raise Exception(f"Batch failed or was cancelled. Status: {batch.status}")
        time.sleep(30)  # Check every 30s

    raise TimeoutError("Batch processing timed out after 24 hours")


def retrieve(batch_job, file_path):
    try:
        batch = check_batch_status(batch_job.id)
    except TimeoutError as e:
        print(f"Error: {e}")
        print("Attempting to cancel the batch...")
        client.batches.cancel(batch_job.id)
        exit(1)
    except Exception as e:
        print(f"Error: {e}")
        exit(1)

    print(batch)
    result_file_id = batch.output_file_id
    result_content = client.files.content(result_file_id).content

    with open(file_path, "wb") as file:
        file.write(result_content)

    print(f"Results written to {file_path}")


def send_gpt_request(batch_file_path):
    batch_input_file = client.files.create(
        file=open(batch_file_path, "rb"), purpose="batch"
    )
    batch_input_file_id = batch_input_file.id

    req = client.batches.create(
        input_file_id=batch_input_file_id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
        metadata={"description": "nightly eval job"},
    )
    return req


def get_gpt_request(df: pd.DataFrame) -> list[dict]:
    json_request = []
    print(df.shape)
    for ind, row in df.iterrows():
        hospitalname = row["hospitalname"]
        city = row["city"]
        state = row["state"]

        prompt = get_hosp_brief_prompt(hospitalname, city, state)

        request = create_gpt_request(prompt, ind + 1)

        json_request.append(request)

    return json_request


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--data", type=str, required=True, help="Data name")

    args = parser.parse_args()

    data_name = args.data

    data_filename = table_data_folder_path

    if data_name == "hosp_20":
        data_filename += hosp_20k_path

    elif data_name == "hosp_100":
        data_filename += hosp_100k_path

    df = pd.read_csv(data_filename)

    filter_df = df[["hospitalname", "city", "state"]].drop_duplicates()
    filter_df = filter_df.reset_index(drop=True)

    print(filter_df.shape)
    # print(filter_df.head())

    gpt_requests = get_gpt_request(filter_df)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".jsonl") as temp_file:
        for item in gpt_requests:
            temp_file.write((json.dumps(item) + "\n").encode("utf-8"))

    batch_job = send_gpt_request(temp_file.name)

    res_file = "gpt_output.jsonl"

    retrieve(batch_job, res_file)
