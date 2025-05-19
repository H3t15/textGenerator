import streamlit as st
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

# ✅ Set page config (must be the first Streamlit command)
st.set_page_config(page_title="GPT-2 Text Generator", layout="centered")

# ✅ Load model and tokenizer (cached after first run)
@st.cache_resource
def load_model():
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")  # You can try "gpt2-medium" for better results
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    model.eval()
    return tokenizer, model

tokenizer, model = load_model()

# ✅ App UI
st.title("🧠 GPT-2 Text Generator")
st.markdown("Generate creative text using the GPT-2 language model by entering a custom prompt below.")

# Prompt input
prompt = st.text_area("Enter your prompt:", "What if dreams were just windows to alternate realities where we live different lives?", height=150)

# Generation controls
max_length = st.slider("Maximum length of generated text (in tokens)", min_value=50, max_value=500, value=150)
temperature = st.slider("Creativity (temperature)", min_value=0.7, max_value=1.5, value=1.0, step=0.1)

# Generate button
if st.button("Generate"):
    if not prompt.strip():
        st.warning("Please enter a prompt to generate text.")
    else:
        with st.spinner("Generating text..."):
            # Tokenize and generate
            input_ids = tokenizer.encode(prompt, return_tensors="pt")
            prompt_len = input_ids.shape[1]
            total_len = min(prompt_len + max_length, 1024)  # GPT-2 token limit

            output = model.generate(
                input_ids,
                max_length=total_len,
                temperature=temperature,
                do_sample=True,
                top_k=40,
                top_p=0.90,
                no_repeat_ngram_size=2,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id
            )

            # Decode and clean result
            generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
            result = generated_text[len(prompt):].strip()

            # Output display
            st.success("Generated Text:")
            st.text_area("Output:", result, height=300)
