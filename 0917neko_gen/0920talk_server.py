
"""

python -m vllm.entrypoints.openai.api_server --model /data/2024_llm/sftlab/out_data/sftlab-experiments/test/1-gemma_full-zero3
"""

from openai import OpenAI

# Modify OpenAI's API key and API base to use vLLM's API server.
# openai_api_base = "http://localhost:8080/v1"
openai_api_base = "http://0.0.0.0:8000/v1"
client = OpenAI(
    # api_key=openai_api_key,
    base_url=openai_api_base,
)

# model_name="AXCXEPT/EZO-Common-9B-gemma-2-it"
model_name = "/data/2024_llm/sftlab/out_data/sftlab-experiments/test/1-gemma_full-zero3"

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
                                                temperature=0.7,
                                                max_tokens=1024,
                                                # skip_special_tokens=False,
                                                stop="<end_of_turn>",
                                                )
    out = completion.choices[0].message.content.strip()
    print("assistant: ", out)
    messages.append({"role": "assistant", "content": out})
