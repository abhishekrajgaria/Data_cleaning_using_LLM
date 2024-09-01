import json


def gen_file(gpt_output_file, txt_file_path):
    with open(gpt_output_file, "r") as jsonl_file, open(txt_file_path, "w") as txt_file:
        for line in jsonl_file:
            # print(line)
            data = json.loads(line.strip())
            brief = data["response"]["body"]["choices"][0]["message"]["content"]
            brief = " ".join(brief.split("\n"))
            brief = " ".join(brief.split(" "))
            txt_file.write(f"{brief}\n")

    print(f"Data stored in line-based TXT file: {txt_file_path}")


if __name__ == "__main__":
    gpt_output_file = "./gpt_output.jsonl"
    result_txt_file = "../data/files/hospital_brief.txt"
    gen_file(gpt_output_file, result_txt_file)
