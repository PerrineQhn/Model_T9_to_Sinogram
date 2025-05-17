import pickle as pk
import re

import pandas as pd
import numpy as np
import tensorflow as tf
from pypinyin import lazy_pinyin

from sklearn.metrics import precision_score, recall_score, f1_score
from typing import List, Tuple, Dict


# -- Partie Prétraitement --
def clean_content(text):
    if not isinstance(text, str):
        return ""

    # Garder les caractères chinois et ponctuation chinoise
    text = re.sub(r"[^\u4e00-\u9fff\u3000-\u303F\uff00-\uffef]", "", text)

    # Normaliser les espaces (rare, mais au cas où)
    text = text.replace(" ", "").strip()

    return text


# Map les lettres latines et chiffres au format T9
t9_map = {
            "a":"2","b":"2","c":"2","d":"3","e":"3","f":"3",
            "g":"4","h":"4","i":"4","j":"5","k":"5","l":"5",
            "m":"6","n":"6","o":"6","p":"7","q":"7","r":"7","s":"7",
            "t":"8","u":"8","v":"8","w":"9","x":"9","y":"9","z":"9",
            "1":"1","2":"2","3":"3","4":"4","5":"5","6":"6","7":"7","8":"8","9":"9","0":"0"
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
    return bool(re.match(r"^[0-9]+$", t9_code)) or t9_code in {"。", "，", "？", "！"}


def generer_sequence_contextuelle(row):
    tokens = row["tokens"]
    sequence = []
    for token in tokens:
        if not isinstance(token, str) or not re.search(r"[\u4e00-\u9fff]", token):
            continue
        for char, py in zip(token, lazy_pinyin(token)):
            t9 = pinyin_to_t9(py)
            if validate_t9(t9):  # Vérifier que le T9 est valide
                sequence.append(f"{char}|{py}|{t9}")
    return " ".join(sequence)



def generer_sequence_contextuelle_new(text, pinyin):
    def pinyin_to_t9(py):
        return "".join(t9_map.get(ch, ch) for ch in py)

    # 1) On garde strictement les sinogrammes
    clean_text = "".join(c for c in text if re.match(r"[\u4e00-\u9fff]", c))

    # 2) On ne retire que la ponctuation
    pinyin_cleaned = re.sub(r'[，。？！：；‘’“”＂（）《》【】、．——－]', ' ', pinyin)

    def split_all_pinyin(text):
        return [syll for part in text.split() for syll in re.findall(r"[a-zü]+[1-5]", part)]

    pinyin_list = split_all_pinyin(pinyin_cleaned)

    sequence = []
    for char, py in zip(clean_text, pinyin_list):
        py = re.sub(r"[1-5]", "", py)
        t9 = pinyin_to_t9(py)
        # Si tu as une fonction validate_t9(t9), tu peux la garder
        sequence.append(f"{char}|{py}|{t9}")

    return " ".join(sequence)


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

    # Graphique de l'historique d'entraînement
    plt.plot(history.history["sparse_categorical_accuracy"])
    plt.plot(history.history["val_sparse_categorical_accuracy"])
    plt.title("Model accuracy")
    plt.ylabel("Accuracy")
    plt.xlabel("Epoch")
    plt.legend(["Train", "Validation"], loc="upper left")
    plt.savefig(f"{model}_accuracy.png")
    plt.show()

    # Graphique de l'historique de perte
    plt.plot(history.history["loss"])
    plt.plot(history.history["val_loss"])
    plt.title("Model loss")
    plt.ylabel("Loss")
    plt.xlabel("Epoch")
    plt.legend(["Train", "Validation"], loc="upper left")
    plt.savefig(f"{model}_loss.png")
    plt.show()


# -- Partie Evaluation Sklearn --
def extract_triplets_from_row(t9_line: str, window_size: int = 6) -> List[Tuple[str, str, str]]:
    """
    Extrait une liste de triplets (context, t9, target_char) à partir d'une ligne de type :
    "云|yun|986 在|zai|924 天|tian|8426 ..."
    """
    # Séparer chaque triplet sinogramme/pinyin/T9
    items = t9_line.strip().split()
    parsed = [item.split('|') for item in items if len(item.split('|')) == 3]

    triplets = []
    for i in range(window_size, len(parsed)):
        # Contexte = les `window_size` sinogrammes précédents
        context = ''.join([p[0] for p in parsed[i - window_size:i]])
        # Prochaine entrée : code T9
        t9_next = parsed[i][2]
        # Prochaine sortie : sinogramme à prédire
        char_next = parsed[i][0]
        triplets.append((context, t9_next, char_next))

    return triplets

def build_evaluation_corpus(df: pd.DataFrame, window_size: int = 6) -> pd.DataFrame:
    """Construit un DataFrame avec colonnes ['context', 't9', 'target_char']."""
    all_triplets = []
    for seq in df['char_pinyin_t9_sequence']:
        all_triplets.extend(extract_triplets_from_row(seq, window_size))
    return pd.DataFrame(all_triplets, columns=["context", "t9", "target_char"])


def generate_text(model, input_t9_sequence, context_chars, input_tv, target_tv, max_length=100, context_size=5):
    """
    Génère une séquence de caractères chinois à partir d'une séquence T9 et d'un contexte en sinogrammes.
    
    Args:
        model: Modèle Keras entraîné.
        input_t9_sequence: Chaîne de séquences T9 séparées par des espaces (ex. "94664 486").
        context_chars: Liste de caractères chinois pour le contexte initial (ex. ["经", "央"]).
        input_tv: Couche TextVectorization pour les entrées T9.
        target_tv: Couche TextVectorization pour les caractères cibles.
        max_length: Longueur maximale de la séquence générée.
        context_size: Taille du contexte (nombre de caractères précédents utilisés).
    
    Returns:
        Chaîne de caractères chinois générée.
    """
    # Préparer l'entrée T9
    t9_tokens = input_t9_sequence.strip().split(" ")
    t9_tokens = t9_tokens[:max_length]  # Limiter à max_length
    if not t9_tokens:
        return ""
    
    # Vectoriser les tokens T9
    t9_vectorized = input_tv(t9_tokens).to_tensor(default_value=0, shape=(len(t9_tokens), 6))
    t9_vectorized = tf.expand_dims(t9_vectorized, axis=0)  # Shape: (1, seq_len, 6)
    
    # Initialiser le contexte avec les caractères fournis
    context = []
    if context_chars:
        # Vectoriser les caractères de contexte
        context_ids = target_tv(context_chars).to_tensor(default_value=0).numpy().flatten().tolist()
        context.extend(context_ids)
    # Remplir avec des zéros si le contexte est trop court
    while len(context) < context_size:
        context.insert(0, 0)  # Padding avec 0 (sera masqué)
    # Tronquer si le contexte est trop long
    context = context[-context_size:]
    
    # Initialiser la séquence générée
    generated_chars = []
    
    # Générer caractère par caractère
    for i in range(len(t9_tokens)):
        # Préparer le contexte
        context_tensor = tf.constant([context[-context_size:]], dtype=tf.int64)
        context_tensor = tf.expand_dims(context_tensor, axis=1)  # Shape: (1, 1, context_size)
        context_tensor = tf.repeat(context_tensor, repeats=tf.shape(t9_vectorized)[1], axis=1)
        
        # Prédire le caractère suivant
        inputs = {
            "t9_input": t9_vectorized,
            "context_input": context_tensor
        }
        predictions = model.predict(inputs, verbose=0)  # Shape: (1, seq_len, vocab_size)
        
        # Obtenir la prédiction pour la position actuelle
        pred_char_idx = np.argmax(predictions[0, i], axis=-1)
        pred_char = target_tv.get_vocabulary()[pred_char_idx]
        
        # Ajouter le caractère généré (sauf si c'est un token spécial)
        if pred_char not in ['', '[UNK]']:
            generated_chars.append(pred_char)
        
        # Mettre à jour le contexte
        context.append(int(pred_char_idx))
        if len(context) > context_size:
            context.pop(0)
    
    # Convertir la liste de caractères en chaîne
    return ''.join(generated_chars)


def evaluate_char_predictions(references: List[str], predictions: List[str]) -> Dict[str, float]:
    """
    Calcule la précision, le rappel et la F1-score pour une liste de chaînes de caractères sinogrammes.
    Chaque paire (ref, pred) est comparée caractère par caractère avec gestion des longueurs différentes.

    Args:
        references (List[str]): Réponses de référence (ground truth)
        predictions (List[str]): Réponses générées par le modèle

    Returns:
        Dict[str, float]: Dictionnaire avec 'precision', 'recall', 'f1'
    """
    y_true, y_pred = [], []
    for ref, pred in zip(references, predictions):
        min_len = min(len(ref), len(pred))
        y_true.extend(ref[:min_len])
        y_pred.extend(pred[:min_len])

        # FP pour surplus prédiction
        if len(pred) > min_len:
            y_pred.extend(pred[min_len:])
            y_true.extend(['[PAD]'] * (len(pred) - min_len))
        # FN pour surplus référence
        if len(ref) > min_len:
            y_true.extend(ref[min_len:])
            y_pred.extend(['[PAD]'] * (len(ref) - min_len))

    mask = [(t != '[PAD]' and p != '[PAD]') for t, p in zip(y_true, y_pred)]
    y_true_f = [t for t, m in zip(y_true, mask) if m]
    y_pred_f = [p for p, m in zip(y_pred, mask) if m]

    return {
        "precision": precision_score(y_true_f, y_pred_f, average="micro", zero_division=0),
        "recall":    recall_score(y_true_f, y_pred_f, average="micro", zero_division=0),
        "f1":        f1_score(y_true_f, y_pred_f, average="micro", zero_division=0)
    }