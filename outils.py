import re
import pandas as pd
from pypinyin import lazy_pinyin
import pickle as pk
import tensorflow as tf

# -- Partie Prétraitement --
def clean_content(text):
    if not isinstance(text, str):
        return ""
    
    # Garder les caractères chinois et ponctuation chinoise
    text = re.sub(r"[^\u4e00-\u9fff\u3000-\u303F\uff00-\uffef]", "", text)
    
    # Normaliser les espaces (rare, mais au cas où)
    text = text.replace(" ", "").strip()

    return text

# convert the content column to pinyin
t9_map = {
    "a": "2", "b": "2", "c": "2",
    "d": "3", "e": "3", "f": "3",
    "g": "4", "h": "4", "i": "4",
    "j": "5", "k": "5", "l": "5",
    "m": "6", "n": "6", "o": "6",
    "p": "7", "q": "7", "r": "7", "s": "7",
    "t": "8", "u": "8", "v": "8",
    "w": "9", "x": "9", "y": "9", "z": "9",
    "1": "1", "2": "2", "3": "3", "4": "4",
    "5": "5", "6": "6", "7": "7", "8": "8",
    "9": "9", "0": "0",
    "。":"。", "，":"，", "？":"？", "！":"！",
}

# Fonction pour convertir une chaîne de caractères en code T9
def pinyin_to_t9(text):
    t9_code = ""
    if pd.isna(text):
        return ""
    for char in text.lower():
        t9_code += t9_map.get(char, char)  # Conserver les caractères non mappés
    return t9_code

def validate_t9(t9_code):
    # Vérifie que le code T9 est numérique (ou vide pour ponctuation)
    return bool(re.match(r'^[0-9]+$', t9_code)) or t9_code in {"。", "，", "？", "！"}

def generer_sequence_contextuelle(row):
    tokens = row["tokens"]
    sequence = []
    for token in tokens:
        if not isinstance(token, str) or not re.search(r'[\u4e00-\u9fff]', token):
            continue
        for char, py in zip(token, lazy_pinyin(token)):
            t9 = pinyin_to_t9(py)
            if validate_t9(t9):  # Vérifier que le T9 est valide
                sequence.append(f"{char}|{py}|{t9}")
    return ' '.join(sequence)



# -- Partie Modèles --

def load_dataset(path):
    return tf.data.Dataset.load(path)

def load_vectorizer(path):
    with open(path, "rb") as f:
        return pk.load(f)
    
def load_params(path):
    with open(path, "rb") as f:
        return pk.load(f)
    
def plot_history(history, model):
    import matplotlib.pyplot as plt

    # Plot training & validation accuracy values
    plt.plot(history.history['sparse_categorical_accuracy'])
    plt.plot(history.history['val_sparse_categorical_accuracy'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.savefig(f'{model}_accuracy.png')
    plt.show()

    # Plot training & validation loss values
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.savefig(f'{model}_loss.png')
    plt.show()