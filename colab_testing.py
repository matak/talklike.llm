# Funkce pro generování odpovědí
def generate_babis_response(prompt, max_length=100, temperature=0.7):
    """Vygeneruje odpověď ve stylu Babiše"""
    
    # Tokenizace promptu
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=config.max_seq_length)
    
    # Generování
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=max_length,
            num_return_sequences=1,
            temperature=temperature,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
    
    # Dekódování
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Odstranění původního promptu
    response = response.replace(prompt, "").strip()
    
    return response

# Testovací prompty
test_prompts = [
    "Uživatel: Jaký je váš názor na inflaci?\nAndrej Babiš: ",
    "Uživatel: Co si myslíte o Bruselu?\nAndrej Babiš: ",
    "Uživatel: Jak hodnotíte opozici?\nAndrej Babiš: ",
    "Uživatel: Jaké máte plány?\nAndrej Babiš: "
]

print("🧪 Testování fine-tuned modelu:")
print("=" * 50)

for prompt in test_prompts:
    response = generate_babis_response(prompt)
    print(f"\nPrompt: {prompt.strip()}")
    print(f"Odpověď: {response}")
    print("-" * 30) 