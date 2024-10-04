
"""
tanuki
python -m vllm.entrypoints.openai.api_server \
    --model weblab-GENIAC/Tanuki-8B-dpo-v1.0 \
    --enable-lora \
    --max-lora-rank 128\
    --lora-modules lora=/data/2024_llm/sftlab/out_data/sftlab-experiments/0923twitter/1-tanuki8b_lora_twitter-zero1/checkpoint-7851

gemma
python -m vllm.entrypoints.openai.api_server \
    --model AXCXEPT/EZO-Common-9B-gemma-2-it \
    --enable-lora \
    --max-lora-rank 256\
    --lora-modules lora=/data/2024_llm/sftlab/out_data/sftlab-experiments/0923twitter/1-gemma_lora_twitter_lora-zero1/checkpoint-6870


"""

from openai import OpenAI
stop = "### 指示:"
stop = "<end_of_turn>"


# Modify OpenAI's API key and API base to use vLLM's API server.
# openai_api_base = "http://localhost:8080/v1"
openai_api_base = "http://0.0.0.0:8000/v1"
client = OpenAI(
    # api_key=openai_api_key,
    base_url=openai_api_base,
)

# model_name="AXCXEPT/EZO-Common-9B-gemma-2-it"
model_name = "weblab-GENIAC/Tanuki-8B-dpo-v1.0"
model_name = "lora"

messages = [
]

while True:
    q = input("user:")

    if q == "reset":
        print("reset conversation----")
        messages = []
        continue

    # print("user: ", q)
    messages.append({"role": "user", "content": q})
    completion = client.chat.completions.create(model=model_name,
                                                messages=messages,
                                                temperature=0.3,
                                                max_tokens=1024,
                                                # skip_special_tokens=False,
                                                # repetition_penalty=1.1,
                                                frequency_penalty=1.1,
                                                stop=stop,
                                                )
    out = completion.choices[0].message.content.strip()
    print("assistant: ", out)
    messages.append({"role": "assistant", "content": out})
