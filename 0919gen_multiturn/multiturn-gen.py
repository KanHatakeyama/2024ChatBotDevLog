"""
export CUDA_VISIBLE_DEVICES=0
export CUDA_VISIBLE_DEVICES=1
"""


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

model_name = "cyberagent/calm3-22b-chat"

# %%
ds = load_dataset("globis-university/aozorabunko-clean", split="train")

# %%


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


pid = os.getpid()
seed = int(pid)+int(datetime.now().timestamp())
print("seed: ", seed)
random.seed(seed)


out_dir = "data"
os.system(f"mkdir -p {out_dir}")

current_time_no_symbols = datetime.now().strftime(
    "%Y-%m-%d %H:%M:%S").replace("-", "").replace(":", "").replace(" ", "")


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
        )
    )
    return [i.outputs[0].text.strip() for i in outputs]


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


def clean_output(text):
    if text.startswith("「"):
        text = text[1:]
    if text.endswith("」"):
        text = text[:-1]
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


def gen_init_text(line, genre):
    system_list = [
        f"次の文章を部分的に引用しながら､{genre}のはじめのセリフを出力しなさい｡対話はタメ口で行われ､敬語は使いません｡セリフのみを出力し、それ以外は何も出力しないでください｡",
        f"次の文章を部分的に引用しながら､{genre}のはじめのセリフを出力しなさい｡対話は敬語で行われます｡セリフのみを出力し、それ以外は何も出力しないでください｡",
    ]
    system_message = random.choice(system_list)

    prompt = f"""
    #参考にする文章: {line}
    """

    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": prompt},
    ]
    text = ask_model(messages, model)
    text = clean_output(text)
    return text


# %%
def gen_text(genres, client,
             n_replies=8):
    genre = random.choice(genres)

    seed_text = extract_random_text(ds)
    text = gen_init_text(seed_text, genre)

    system_list = [
        f"{genre}の返答文を生成しなさい｡セリフのみを出力し、それ以外は何も出力しないでください｡",
    ]
    system_message = random.choice(system_list)
    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": text}
    ]

    for j in range(n_replies):
        res = ask_model(messages, client)
        res = clean_output(res)
        if j % 2 == 1:
            messages.append({"role": "user", "content": res})
        else:
            messages.append({"role": "assistant", "content": res})

    return messages


# %%
# seed_text=extract_random_text(ds)
# prompt=gen_init_text(seed_text,random.choice(genres))
# print(prompt)


out_file_path = f"data/gen_mult_{current_time_no_symbols}.jsonl"

with open(out_file_path, "a") as f:
    for i in tqdm(range(5*10**4)):
        try:
            messages = gen_text(genres, model, n_replies=11)
        except Exception as e:
            print(e)
            time.sleep(5)
            continue
        f.write(json.dumps({"messages": messages}, ensure_ascii=False)+"\n")
        time.sleep(1)

# %%
