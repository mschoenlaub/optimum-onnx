"""Simple example: Export Devstral Small 2 to ONNX and generate text.

Usage:
    uv pip install onnxruntime
    uv run examples/devstral-small-2.py
"""

from transformers import AutoTokenizer

from optimum.onnxruntime import ORTModelForCausalLM


model_id = "mistralai/Devstral-Small-2-24B-Instruct-2512"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = ORTModelForCausalLM.from_pretrained(model_id, export=True)

# Chat with instruction-tuned model
conversation = [{"role": "user", "content": "Hello! How are you?"}]
prompt = tokenizer.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)
inputs = tokenizer(prompt, return_tensors="pt")

outputs = model.generate(**inputs, max_new_tokens=50, pad_token_id=tokenizer.eos_token_id)
response = tokenizer.decode(outputs[0], skip_special_tokens=True)

print(response)