# %%


import time
from tqdm import tqdm
import json
import random
from openai import OpenAI
import os
from datetime import datetime

pid = os.getpid()
seed = int(pid)+int(datetime.now().timestamp())
print("seed: ", seed)
random.seed(seed)


# %%


# %%

# %%
out_dir = "data"
os.system(f"mkdir -p {out_dir}")

current_time_no_symbols = datetime.now().strftime(
    "%Y-%m-%d %H:%M:%S").replace("-", "").replace(":", "").replace(" ", "")


# Modify OpenAI's API key and API base to use vLLM's API server.
# openai_api_base = "http://localhost:8080/v1"
openai_api_base = "http://0.0.0.0:8000/v1"
client = OpenAI(
    base_url=openai_api_base,
)

model_name = "AXCXEPT/EZO-Common-9B-gemma-2-it"
model_name = "weblab-GENIAC/Tanuki-8x8B-dpo-v1.0"
model_name = "cyberagent/calm3-22b-chat"


# %%

# 元の文章のparse
neko_path = "source/neko.txt"
with open(neko_path, "r") as f:
    text = f.read()

# clean
lines = text.split("\n")
new_lines = []
for line in lines:
    line = line.strip()
    if len(line) < 10:
        continue
    new_lines.append(line)

len(new_lines)

# %%


genres = [
    "心温まる日常会話",
    "楽しい会話",
    "ユーモラスな会話",
    "日常会話",
    "傾聴の会話",
    "相手の興味をそそる会話",
]


def clean_output(text):
    if text.startswith("「"):
        text = text[1:]
    if text.endswith("」"):
        text = text[:-1]
    return text


def ask_model(messages, client):
    completion = client.chat.completions.create(model=model_name,
                                                messages=messages,
                                                temperature=0.7,
                                                max_tokens=1024,
                                                # skip_special_tokens=False,
                                                stop="### 指示:",
                                                )
    text = completion.choices[0].message.content.strip()
    text = clean_output(text)
    return text


def gen_init_text(line, genre):
    system_list = [
        f"次の文章をもとに､{genre}のはじめのセリフを出力しなさい｡対話はタメ口で行われ､敬語は使いません｡セリフのみを出力し、それ以外は何も出力しないでください｡",
        f"次の文章をもとに､{genre}のはじめのセリフを出力しなさい｡セリフのみを出力し、それ以外は何も出力しないでください｡",
    ]
    system_message = random.choice(system_list)

    prompt = f"""
    #参考にする文章: {line}
    """

    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": prompt}
    ]
    text = ask_model(messages, client)
    return text


# %%
n_replies = 4


def gen_text(new_lines, genres, client,
             n_replies=8):
    line = random.choice(new_lines)
    genre = random.choice(genres)

    text = gen_init_text(line, genre)

    system_list = [
        f"{genre}の返答文を生成しなさい｡対話はタメ口で行われ､敬語は使いません｡セリフのみを出力し、それ以外は何も出力しないでください｡",
        f"{genre}の返答文を生成しなさい｡セリフのみを出力し、それ以外は何も出力しないでください｡",
    ]
    system_message = random.choice(system_list)
    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": text}
    ]

    for j in range(n_replies):

        res = ask_model(messages, client)
        if j % 2 == 1:
            messages.append({"role": "user", "content": res})
        else:
            messages.append({"role": "assistant", "content": res})

    return messages


# %%
# messages = gen_text(new_lines, genres, client, n_replies=7)
# messages

# %%

out_file_path = f"data/neko_gen_mult_{current_time_no_symbols}.jsonl"

with open(out_file_path, "a") as f:
    for i in tqdm(range(10**4)):
        try:
            messages = gen_text(new_lines, genres, client, n_replies=7)
        except Exception as e:
            print(e)
            time.sleep(5)
            continue
        f.write(json.dumps({"messages": messages}, ensure_ascii=False)+"\n")
        time.sleep(1)

# %%
