# chatbot

## Description : 
Ce Chatbot est basé sur un modèle de NLP fine-tuné à partir d'une base de données sur les football européen.

## Prérequis
Pour tester, vous aurez besoin de : 
- La base de données sqlite à télécharger ici : [European Soccer Database (Kaggle)](https://www.kaggle.com/code/alaasedeeq/european-soccer-database-with-sqlite3/input)
- Installer le contenue du fichier requirements.txt dans votre environnement d'exécution *`pip install -r requirements.txt`*
- Générer le texte d'entraînement et entraîner le modèle **`python train.py`**
- Démarrer le chatbot **`python main.py`**

## Prompt de test
Le bot pourra répondre aux question comme : 
- Dans quel club jouait Kevin De Bruyne en 2013 ?
- Où évoluait Eden Hazard lors de la saison 2011 ?