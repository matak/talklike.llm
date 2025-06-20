# Vytvoření Gradio interface
import gradio as gr

def babis_chat(message, history):
    """Chat funkce pro Gradio"""
    prompt = f"Uživatel: {message}\nAndrej Babiš: "
    response = generate_babis_response(prompt, max_length=150, temperature=0.8)
    return response

# Vytvoření interface
iface = gr.ChatInterface(
    fn=babis_chat,
    title="🤖 Babiš Chat Bot",
    description="Fine-tuned model ve stylu Andreje Babiše",
    examples=[
        ["Jaký je váš názor na inflaci?"],
        ["Co si myslíte o Bruselu?"],
        ["Jak hodnotíte opozici?"],
        ["Jaké máte plány do budoucna?"],
        ["Jak trávíte čas s rodinou?"],
    ]
)

# Spuštění interface
iface.launch(share=True, debug=True)

print("🌐 Gradio interface spuštěn!")
print("Sdílejte odkaz s ostatními pro testování.") 