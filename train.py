import sqlite3
import random
import os
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    TextDataset, 
    DataCollatorForLanguageModeling
)
import torch

print(torch.cuda.get_device_name(0))

# Chemins absolus
current_dir = os.path.dirname(__file__)
db_path = os.path.abspath(os.path.join(current_dir, "data/database.sqlite"))
output_path = os.path.abspath(os.path.join(current_dir, "train.txt"))
model_output_dir = os.path.abspath(os.path.join(current_dir, "model_finetuned"))

def generer_train_file():
    
    # Templates améliorés
    question_templates = [
    "Dans quel club jouait {player} en {year} ?",
    "Où évoluait {player} lors de la saison {year} ?",
    "Pour quelle équipe jouait {player} en {year} ?",
        "Quel était le club de {player} en {year} ?"
    ]
    answer_templates = [
        "{player} jouait pour {club} durant la saison {season}.",
        "En {season}, {player} était au club {club}.",
        "{player} évoluait à {club} pendant la saison {season}."
    ]

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    dataset = []

    # Génère beaucoup d'exemples variés
    cursor.execute("""
        SELECT p.player_name, t.team_long_name, m.season
        FROM Player p
        JOIN Match m ON m.home_player_1 = p.player_api_id
        JOIN Team t ON m.home_team_api_id = t.team_api_id
        WHERE p.player_name IS NOT NULL AND t.team_long_name IS NOT NULL
        LIMIT 2000
    """)
    for nom, club, saison in cursor.fetchall():
        year = saison[:4]
        for _ in range(2):  # 2 formulations différentes par exemple
            question = random.choice(question_templates).format(player=nom, year=year)
            reponse = random.choice(answer_templates).format(player=nom, club=club, season=saison)
            dataset.append((f"Utilisateur: {question}", f"Bot: {reponse}"))

    # Exemples négatifs
    negative_questions = [
        "Dans quel club jouait Lionel Messi en 2050 ?",
        "Qui a gagné la Coupe du Monde 2022 ?",
        "Quel est le meilleur joueur de tous les temps ?"
    ]
    for nq in negative_questions:
        dataset.append((f"Utilisateur: {nq}", "Bot: Je ne sais pas répondre à cette question."))

    random.shuffle(dataset)
    with open(output_path, "w", encoding="utf-8") as f:
        for q, r in dataset:
            f.write(f"\n{q} {r}")

    print(f"✅ {len(dataset)} paires générées dans {output_path}")

# Génération du fichier d'entraînement
generer_train_file()

# Chargement du modèle
model_name = "bigscience/bloomz-560m"
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token  # Correction du token de padding
model = AutoModelForCausalLM.from_pretrained(model_name)

# Dataset adapté au dialogue
def load_dataset(file_path, tokenizer, block_size=128):
    return TextDataset(
        tokenizer=tokenizer,
        file_path=file_path,
        block_size=block_size
    )

train_dataset = load_dataset(output_path, tokenizer)

# Configuration d'entraînement
training_args = TrainingArguments(
    output_dir=model_output_dir,
    overwrite_output_dir=True,
    num_train_epochs=3,  # Augmentation des epochs
    per_device_train_batch_size=4,
    gradient_accumulation_steps=2,
    save_strategy="epoch",
    learning_rate=2e-5,
    logging_dir="./logs",
    fp16=torch.cuda.is_available()
)

# Data collator
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, 
    mlm=False
)

# Lancement de l'entraînement
trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=train_dataset
)

print("🚀 Début de l'entraînement...")
trainer.train()
print("✅ Entraînement terminé!")

# Sauvegarde finale
trainer.save_model()
tokenizer.save_pretrained(model_output_dir)
