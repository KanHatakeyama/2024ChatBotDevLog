# %%
# 吾輩は猫であるをもとにtextを生成する

"""
vllm


python -m vllm.entrypoints.openai.api_server --model cyberagent/calm3-22b-chat

export LIBRARY_PATH="/usr/local/cuda-12.3/lib64/stubs:$LIBRARY_PATH"
python -m vllm.entrypoints.openai.api_server --model weblab-GENIAC/Tanuki-8x8B-dpo-v1.0     --max-model-len 4096 --port 8000 --gpu-memory-utilization 0.9     --trust-remote-code --tensor-parallel-size 2

"""

# %%
# 元の文章のparse
import time
from tqdm import tqdm
import json
import random
from openai import OpenAI
neko_path = "data/neko.txt"
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


# Modify OpenAI's API key and API base to use vLLM's API server.
# openai_api_base = "http://localhost:8080/v1"
openai_api_base = "http://0.0.0.0:8000/v1"
client = OpenAI(
    base_url=openai_api_base,
)

model_name = "AXCXEPT/EZO-Common-9B-gemma-2-it"
model_name = "weblab-GENIAC/Tanuki-8x8B-dpo-v1.0"


# %%


genres = [
    "心温まる日常会話",
    "楽しい会話",
    "ユーモラスな会話",
    "日常会話",
    "傾聴の会話",
    "相手の興味をそそる会話",
]


def gen_text():
    line = random.choice(new_lines)
    genre = random.choice(genres)

    system_message = f"""以下は、タスクを説明する指示です。要求を適切に満たす応答を書きなさい。
    次の文章をもとに､userとassistantの{genre}を作成してください｡対話はタメ口で行われ､敬語は使いません｡
    """

    prompt = f"""
    #参考にする文章: {line}
    """

    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": prompt}
    ]
    completion = client.chat.completions.create(model=model_name,
                                                messages=messages,
                                                temperature=0.7,
                                                max_tokens=1024,
                                                # skip_special_tokens=False,
                                                stop="### 指示:",
                                                )
    text = completion.choices[0].message.content.strip()
    return text


# %%
out_file_path = "data/neko_gen.jsonl"

with open(out_file_path, "a") as f:
    for i in tqdm(range(10**5)):
        text = gen_text()
        f.write(json.dumps({"text": text}, ensure_ascii=False)+"\n")
        time.sleep(1)

# %%
"""
upload
huggingface-cli upload multiturn-conv-from-wagahai-neko data/neko_gen.jsonl data.jsonl --repo-type dataset

"""

# %%
