# %%
import json
from tqdm import tqdm
# %%
import time
from tqdm import tqdm
import copy
import transformers
from vllm import LLM, SamplingParams
import json
import os
from datetime import datetime
from datasets import load_dataset
import random


# %%

pid = os.getpid()
seed = int(pid)+int(datetime.now().timestamp())
print("seed: ", seed)
random.seed(seed)


out_dir = "data"
os.system(f"mkdir -p {out_dir}")

current_time_no_symbols = datetime.now().strftime(
    "%Y-%m-%d %H:%M:%S").replace("-", "").replace(":", "").replace(" ", "")

# %%
target_jsonl = "source/split_twitter_archive_cleaned_aa.jsonl"
tweet_list = []
with open(target_jsonl, "r") as f:
    for line in tqdm(f):
        tweet_list.append(json.loads(line))

# %%
tweet_list[-11]

# %%

model_name = "cyberagent/calm3-22b-chat"
tensor_parallel_size = 1
# トークナイザーとモデルの準備
model = LLM(
    model=model_name,
    trust_remote_code=True,
    max_model_len=2000,
    tensor_parallel_size=tensor_parallel_size,
)


# %%
tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)

# %%


# %%
"""
export CUDA_VISIBLE_DEVICES=0
export CUDA_VISIBLE_DEVICES=1
"""


def extract_random_text(ds):
    seed = int(pid)+int(datetime.now().timestamp())
    random.seed(seed)
    record = ds[random.randint(0, len(ds))]
    extract_length = random.randint(100, 300)
    try:
        extract_pos = random.randint(0, len(record["text"])-extract_length)
        extracted_text = record["text"][extract_pos:extract_pos +
                                        extract_length].strip()
    except:
        extracted_text = record["text"][0:100].strip()

    return extracted_text


# %%
# auto reload modules


# %%
max_tokens = 1024


def llm_gen(model, prompt_list,
            temperature=0.7, top_k=50,
            max_tokens=max_tokens,
            ):
    outputs = model.generate(
        prompt_list,
        sampling_params=SamplingParams(
            temperature=temperature,
            max_tokens=max_tokens,
            top_k=top_k,
            stop="\n\n",  # 2回改行が出たら終了｡会話の二周目以降なので｡
        )
    )
    return [i.outputs[0].text.strip() for i in outputs]


def clean_output(text):
    if text.startswith("「") and text.endswith("」"):
        text = text[1:-1]
    return text


# %%
def ask_model(messages, model):
    messages = copy.deepcopy(messages)

    if messages[-1]["role"] == "user":
        messages.append({"role": "assistant", "content": ""})
    else:
        messages.append({"role": "user", "content": ""})

    prompt = tokenizer.apply_chat_template(messages, tokenize=False,)
    prompt = prompt.replace("<|im_end|>\n", "")
    text = llm_gen(model, [prompt], temperature=0.7,
                   top_k=50, max_tokens=1024)[0]
    return text


genres = [
    "心温まる日常会話",
    "楽しい会話",
    "ユーモラスな会話",
    "日常会話",
    "傾聴の会話",
    "相手の興味をそそる会話",
    "長めの会話",
    # "指示を聞いて答えるタイプの会話",
]


def gen_reply(tweet, genre):
    system_list = [
        f"あなたはSNSで{genre}をしています｡口調やノリを相手に合わせながら､返事をしてください｡返答のみを出力し、それ以外は何も出力しないでください｡",
    ]
    system_message = random.choice(system_list)

    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": tweet},
    ]
    text = ask_model(messages, model)
    text = clean_output(text)
    return text


# %%

def gen_prompt(tweet_list, genres):
    genre = random.choice(genres)
    tweet = random.choice(tweet_list)["text"]
    system_list = [
        f"あなたはSNSで{genre}をしています｡口調やノリを相手に合わせながら､返事をしてください｡返答のみを出力し、それ以外は何も出力しないでください｡",
    ]
    system_message = random.choice(system_list)

    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": tweet},
        {"role": "assistant", "content": ""},
    ]

    prompt = tokenizer.apply_chat_template(messages, tokenize=False,)
    prompt = prompt.replace("<|im_end|>\n", "")
    return prompt, tweet


# %%
batch_size = 300


out_file_path = f"data/reply_{current_time_no_symbols}.jsonl"

with open(out_file_path, "a") as f:
    for i in tqdm(range(5*10**4)):
        try:
            prompts, tweets = zip(*[gen_prompt(tweet_list, genres)
                                  for i in range(batch_size)])
            replies = llm_gen(model, prompts, temperature=0.7,
                              top_k=50, max_tokens=256)
            replies = [clean_output(i) for i in replies]
            for prompt, tweet, reply in zip(prompts, tweets, replies):
                messages = [
                    # {"role": "system", "content": prompt},
                    {"role": "user", "content": tweet},
                    {"role": "assistant", "content": reply},
                ]
                f.write(json.dumps({"messages": messages},
                        ensure_ascii=False)+"\n")
        except Exception as e:
            print(e)
            time.sleep(5)
            continue
        time.sleep(3)

# %%
