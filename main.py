from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import os

# Configuration des chemins
current_dir = os.path.dirname(__file__)
model_path = os.path.join(current_dir, "model_finetuned")

# Chargement du modèle
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path)
tokenizer.pad_token = tokenizer.eos_token  # Assurer la cohérence

print("🤖 Chatbot (modèle fine-tuné) démarré. Tape 'exit' pour quitter.")

while True:
    user_input = input("👤 Toi: ")
    if user_input.lower() in ["exit", "quit"]:
        break
    
    # Formatage du prompt
    prompt = f"Utilisateur: {user_input} Bot:"
    inputs = tokenizer.encode(prompt, return_tensors="pt")
    
    # Génération avec contraintes
    outputs = model.generate(
        inputs,
        max_length=inputs.shape[1] + 50,  # Limite de longueur
        pad_token_id=tokenizer.eos_token_id,
        do_sample=True,
        top_k=40,
        top_p=0.90,
        temperature=0.7,
        num_return_sequences=1,
        eos_token_id=tokenizer.eos_token_id  # Pour arrêter la génération
    )
    
    # Décodage et nettoyage
    full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    bot_response = full_response.split("Bot:")[-1].strip()
    
    print(f"🤖 Bot: {bot_response}")