#
# Copyright © 2024 Intel Corporation
# SPDX-License-Identifier: Apache 2.0
#

from transformers import AutoTokenizer, TextStreamer
from intel_npu_acceleration_library import NPUModelForCausalLM, int8
import time

model_id = "meta-llama/Meta-Llama-3-8B-Instruct"

model = NPUModelForCausalLM.from_pretrained(model_id, dtype=int8, use_cache=True).eval()
tokenizer = AutoTokenizer.from_pretrained(model_id)
streamer = TextStreamer(tokenizer, skip_special_tokens=True, skip_prompt=True)

print("Run inference with Llama3 on NPU\n")


query = input(">")


messages = [
    {
        "role": "system",
        "content": "You are an helpful chatbot that can provide information about the Intel NPU",
    },
    {"role": "user", "content": query},
]

input_ids = tokenizer.apply_chat_template(
    messages, add_generation_prompt=True, return_tensors="pt"
).to(model.device)

terminators = [tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|eot_id|>")]

# Measure the start time
start_time = time.time()

outputs = model.generate(
    input_ids,
    max_new_tokens=256,
    eos_token_id=terminators,
    do_sample=True,
    streamer=streamer,
)


# Measure the end time
end_time = time.time()

# Calulate the number of tokens generated
num_tokens_generated = outputs.shape[1]

# Calculate the tokens per second
time_taken = end_time - start_time
tokens_per_second = num_tokens_generated / time_taken

# Print the tokens per second
print(f"Tokens per second: {tokens_per_second:.2f}")
