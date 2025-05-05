from transformers import AutoTokenizer, AutoModelForCausalLM

# 1. Load your fine-tuned model and tokenizer
model_path = "./fine_tuned_model"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path)

pad_token_id = 50257



if tokenizer.pad_token is None:
	tokenizer.add_special_tokens({'pad_token': '<pad>'})
	pad_token_id = tokenizer.convert_tokens_to_ids('<pad>')

# 2. Prepare an input prompt

input_prompt = input("Please enter your prompt: ")
input_ids = tokenizer.encode(input_prompt, return_tensors="pt")


attention_mask = input_ids.ne(tokenizer.pad_token_id).long()

# 3. Generate a response
output = model.generate(
    input_ids,
    attention_mask=attention_mask,
    max_length=100,  # Set a maximum length for the generated response
    num_return_sequences=1,  # Number of responses to generate
    temperature=1,  # Controls the randomness of the output
    top_p=0.9,  # Nucleus sampling
    top_k=50,  # Top-k sampling
    do_sample=True,  # Enable sampling to introduce variability
pad_token_id=tokenizer.pad_token_id,
)

# 4. Decode and print the generated response
response = tokenizer.decode(output[0], skip_special_tokens=True)
print("Generated Response:", response)
