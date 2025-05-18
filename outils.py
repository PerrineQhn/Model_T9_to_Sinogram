import os
import pickle as pk
import re
import csv
from typing import Dict, List, Tuple, Union

import numpy as np
import pandas as pd
import tensorflow as tf
from lxml import etree
from pypinyin import lazy_pinyin
from sklearn.metrics import classification_report

# -- Partie Constantes --
MAX_SEQUENCE_LENGTH = 20

# -- Partie Prétraitement --
def clean_content(text):
    if not isinstance(text, str):
        return ""

    # Garder les caractères chinois et ponctuation chinoise
    text = re.sub(r"[^\u4e00-\u9fff\u3000-\u303F\uff00-\uffef]", "", text)

    # Normaliser les espaces (rare, mais au cas où)
    text = text.replace(" ", "").strip()

    return text


def get_text_from_xml(xml_file):
    """
    Extract text from XML file, handling different tags and ensuring correct punctuation.
    Args:
        xml_file (str): Path to the XML file.

    Returns:
        sentences (list): List of extracted sentences with correct punctuation.
    """
    sentences = []
    try:
        tree = etree.parse(xml_file)
        root = tree.getroot()

        for text_elem in root.iter("text"):
            for paragraph in text_elem.iter("p"):
                for sentence_elem in paragraph.iter("s"):
                    sentence_text = ""
                    for child in sentence_elem.iter("w", "c"):
                        if child.text:
                            sentence_text += child.text.strip()
                    if sentence_text:
                        # Remove unnecessary spaces
                        sentence_text = "".join(sentence_text.split())
                        sentences.append(sentence_text)
    except etree.XMLSyntaxError as e:
        print(f"Error parsing XML {xml_file}: {e}")
        return []
    return sentences


def process_xml_directories(pinyin_directory, chinese_directory):
    """
    Process XML files in pinyin and Chinese directories, extracting and concatenating text.
    Args:
        pinyin_directory (str): Path to directory containing pinyin XML files.
        chinese_directory (str): Path to directory containing Chinese character XML files.

    Returns:
        filename_pinyin (dict): Dictionary mapping pinyin XML filenames to concatenated text.
        filename_char (dict): Dictionary mapping Chinese XML filenames to concatenated text.
    """
    filename_pinyin = {}
    filename_char = {}

    # Process pinyin XML files
    for filename in os.listdir(pinyin_directory):
        if filename.lower().endswith(".xml"):
            file_path = os.path.join(pinyin_directory, filename)
            sentences = get_text_from_xml(file_path)
            if sentences:
                # Concatenate sentences with spaces (or no spaces, depending on preference)
                concatenated_text = "".join(sentences)  # No spaces for pinyin
                filename_pinyin[filename.lower()] = concatenated_text
            else:
                print(f"No text extracted from {filename}")

    # Process Chinese character XML files
    for filename in os.listdir(chinese_directory):
        if filename.lower().endswith(".xml"):
            file_path = os.path.join(chinese_directory, filename)
            sentences = get_text_from_xml(file_path)
            if sentences:
                # Concatenate sentences with spaces (or no spaces, depending on preference)
                concatenated_text = "".join(sentences)  # No spaces for Chinese
                filename_char[filename.lower()] = concatenated_text
            else:
                print(f"No text extracted from {filename}")

    return filename_pinyin, filename_char


def create_csv(filename_char, filename_pinyin, output_file):
    """
    Create a CSV file from Chinese character and pinyin XML files.
    The CSV contains columns for Content (Chinese text) and Pinyin (pinyin text).
    Args:
        filename_char (dict): Dictionary mapping filenames to concatenated Chinese text.
        filename_pinyin (dict): Dictionary mapping filenames to concatenated pinyin text.
        output_file (str): Path to the output CSV file (default: 'pinyin_char.csv').

    Returns:
        None
    """
    with open(output_file, "w", encoding="utf-8", newline="") as f:
        # Initialiser l'écrivain CSV
        writer = csv.writer(f, lineterminator="\n")
        # Écrire l'en-tête
        writer.writerow(["content", "pinyin"])

        # Itérer sur les fichiers dans filename_char
        for key in filename_char.keys():
            if key not in filename_pinyin:
                print(f"Warning: {key} not found in filename_pinyin")
                continue

            # Obtenir les textes chinois et pinyin
            char_text = filename_char[key]
            pinyin_text = filename_pinyin[key]

            # Diviser les textes en phrases
            char_sentences = [
                s.strip() for s in re.split(r"[。；]", char_text) if s.strip()
            ]
            pinyin_sentences = [
                s.strip() for s in re.split(r"[。；]", pinyin_text) if s.strip()
            ]

            # Vérifier la correspondance du nombre de phrases
            if len(char_sentences) != len(pinyin_sentences):
                print(
                    f"Warning: Mismatch in sentence count for {key}: "
                    f"{len(char_sentences)} Chinese vs {len(pinyin_sentences)} pinyin"
                )
                min_sentences = min(len(char_sentences), len(pinyin_sentences))
                char_sentences = char_sentences[:min_sentences]
                pinyin_sentences = pinyin_sentences[:min_sentences]

            # Écrire chaque paire de phrases dans le CSV
            for char_sent, pinyin_sent in zip(char_sentences, pinyin_sentences):
                # Nettoyer le texte pour éviter les problèmes de format CSV
                char_sent = char_sent.replace("\n", " ").replace("\r", " ")
                pinyin_sent = pinyin_sent.replace("\n", " ").replace("\r", " ")
                writer.writerow([char_sent, pinyin_sent])


# Map les lettres latines et chiffres au format T9
t9_map = {
    "a": "2",
    "b": "2",
    "c": "2",
    "d": "3",
    "e": "3",
    "f": "3",
    "g": "4",
    "h": "4",
    "i": "4",
    "j": "5",
    "k": "5",
    "l": "5",
    "m": "6",
    "n": "6",
    "o": "6",
    "p": "7",
    "q": "7",
    "r": "7",
    "s": "7",
    "t": "8",
    "u": "8",
    "v": "8",
    "w": "9",
    "x": "9",
    "y": "9",
    "z": "9",
    "1": "1",
    "2": "2",
    "3": "3",
    "4": "4",
    "5": "5",
    "6": "6",
    "7": "7",
    "8": "8",
    "9": "9",
    "0": "0",
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
    pinyin_cleaned = re.sub(r"[，。？！：；‘’“”＂（）《》【】、．——－]", " ", pinyin)

    def split_all_pinyin(text):
        return [
            syll for part in text.split() for syll in re.findall(r"[a-zü]+[1-5]", part)
        ]

    pinyin_list = split_all_pinyin(pinyin_cleaned)

    sequence = []
    for char, py in zip(clean_text, pinyin_list):
        py = re.sub(r"[1-5]", "", py)
        t9 = pinyin_to_t9(py)
        # Si tu as une fonction validate_t9(t9), tu peux la garder
        sequence.append(f"{char}|{py}|{t9}")

    return " ".join(sequence)


# -- Partie Création du Dataset pour l'entraînement --
def pad_t9_input(inputs, labels, t9_dim=6):
    """
    Pad the T9 input to a fixed size. If the input is shorter, it will be padded with zeros.
    If the input is longer, it will be truncated.

    Args:
        inputs (dict): Dictionary containing the input tensors.
        labels (tf.Tensor): Tensor containing the labels.
        t9_dim (int): Fixed size for T9 input.
    Returns:
        tuple: Tuple containing the padded inputs and labels.
    """
    t9_input = inputs["t9_input"]
    t9_input = t9_input[:, :t9_dim]  # Truncate second dimension
    seq_len = tf.shape(t9_input)[0]
    current_dim = tf.shape(t9_input)[1]
    paddings = [[0, 0], [0, tf.maximum(0, t9_dim - current_dim)]]
    t9_input_padded = tf.pad(t9_input, paddings, constant_values=0)
    return {
        "t9_input": t9_input_padded,
        "context_input": inputs["context_input"],
    }, labels


# Truncate sequence length
def truncate_sequence(inputs, labels, max_length=MAX_SEQUENCE_LENGTH):
    """
    Truncate the sequence to a maximum length. If the sequence is shorter, it will be padded.
    Args:
        inputs (dict): Dictionary containing the input tensors.
        labels (tf.Tensor): Tensor containing the labels.
        max_length (int): Maximum length for the sequence.
    Returns:
        tuple: Tuple containing the truncated inputs and labels.
    """
    return {
        "t9_input": inputs["t9_input"][:max_length],
        "context_input": inputs["context_input"][:max_length],
    }, labels[:max_length]


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


# -- Partie Evaluation Models--
def evaluate_models(model_simple, model_with_1_mask, model_transformer, ds_test):
    print("Évaluation du modèle simple...")
    metrics_simple = model_simple.evaluate(ds_test, verbose=1, return_dict=True)
    # print("Metrics returned for model_simple:", metrics_simple)

    print("\nÉvaluation du modèle avec 1 masqué...")
    metrics_1_mask = model_with_1_mask.evaluate(ds_test, verbose=1, return_dict=True)
    # print("Metrics returned for model_with_1_mask:", metrics_1_mask)

    print("\nÉvaluation du modèle Transformer...")
    metrics_transformer = model_transformer.evaluate(
        ds_test, verbose=1, return_dict=True
    )
    # print("Metrics returned for model_transformer:", metrics_transformer)

    # Extract metrics
    loss_simple = metrics_simple["loss"]
    perplexity_simple = metrics_simple.get(
        "perplexity", 0.0
    )  # Use .get() to handle missing metrics
    acc_simple = metrics_simple.get("sparse_categorical_accuracy", 0.0)
    weighted_acc_simple = metrics_simple.get(
        "weighted_sparse_categorical_accuracy", 0.0
    )

    loss_1_mask = metrics_1_mask["loss"]
    perplexity_1_mask = metrics_1_mask.get("perplexity", 0.0)
    acc_1_mask = metrics_1_mask.get("sparse_categorical_accuracy", 0.0)
    weighted_acc_1_mask = metrics_1_mask.get(
        "weighted_sparse_categorical_accuracy", 0.0
    )

    loss_transformer = metrics_transformer["loss"]
    perplexity_transformer = metrics_transformer.get("perplexity", 0.0)
    acc_transformer = metrics_transformer.get("sparse_categorical_accuracy", 0.0)
    weighted_acc_transformer = metrics_transformer.get(
        "weighted_sparse_categorical_accuracy", 0.0
    )

    return {
        "simple": {
            "loss": loss_simple,
            "accuracy": acc_simple,
            "perplexity": perplexity_simple,
            "weighted_accuracy": weighted_acc_simple,
        },
        "with_1_mask": {
            "loss": loss_1_mask,
            "accuracy": acc_1_mask,
            "perplexity": perplexity_1_mask,
            "weighted_accuracy": weighted_acc_1_mask,
        },
        "transformer": {
            "loss": loss_transformer,
            "accuracy": acc_transformer,
            "perplexity": perplexity_transformer,
            "weighted_accuracy": weighted_acc_transformer,
        },
    }


# -- Partie Evaluation Sklearn --
def extract_triplets_from_row(
    t9_line: str, window_size: int = 6
) -> List[Tuple[str, str, str]]:
    """
    Extrait une liste de triplets (context, t9, target_char) à partir d'une ligne de type :
    "云|yun|986 在|zai|924 天|tian|8426 ..."
    """
    # Séparer chaque triplet sinogramme/pinyin/T9
    items = t9_line.strip().split()
    parsed = [item.split("|") for item in items if len(item.split("|")) == 3]

    triplets = []
    for i in range(window_size, len(parsed)):
        # Contexte = les `window_size` sinogrammes précédents
        context = "".join([p[0] for p in parsed[i - window_size : i]])
        # Prochaine entrée : code T9
        t9_next = parsed[i][2]
        # Prochaine sortie : sinogramme à prédire
        char_next = parsed[i][0]
        triplets.append((context, t9_next, char_next))

    return triplets


def build_evaluation_corpus(df: pd.DataFrame, window_size: int = 6) -> pd.DataFrame:
    """Construit un DataFrame avec colonnes ['context', 't9', 'target_char']."""
    all_triplets = []
    for seq in df["char_pinyin_t9_sequence"]:
        all_triplets.extend(extract_triplets_from_row(seq, window_size))
    return pd.DataFrame(all_triplets, columns=["context", "t9", "target_char"])


def generate_text(
    model,
    input_t9_sequence,
    context_chars,
    input_tv,
    target_tv,
    max_length=100,
    context_size=5,
):
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
    t9_vectorized = input_tv(t9_tokens).to_tensor(
        default_value=0, shape=(len(t9_tokens), 6)
    )
    t9_vectorized = tf.expand_dims(t9_vectorized, axis=0)  # Shape: (1, seq_len, 6)

    # Initialiser le contexte avec les caractères fournis
    context = []
    if context_chars:
        # 1) On wrappe en tf.constant pour forcer un RaggedTensor
        chars_tensor = tf.constant(context_chars, dtype=tf.string)
        vect = target_tv(chars_tensor)  # RaggedTensor si ragged=True

        # 2) Si c'est un RaggedTensor, on densifie ; sinon on garde tel quel
        if isinstance(vect, tf.RaggedTensor):
            dense_ids = vect.to_tensor(default_value=0)
        else:
            dense_ids = vect

        # 3) On a désormais un Tensor dense : on récupère les IDs
        context_ids = dense_ids.numpy().flatten().tolist()
        context.extend(context_ids)

    # Remplir avec des zéros si le contexte est trop court
    while len(context) < context_size:
        context.insert(0, 0)
    # Tronquer si nécessaire
    context = context[-context_size:]

    # Initialiser la séquence générée
    generated_chars = []

    # Générer caractère par caractère
    for i in range(len(t9_tokens)):
        # Préparer le contexte
        context_tensor = tf.constant([context[-context_size:]], dtype=tf.int64)
        context_tensor = tf.expand_dims(
            context_tensor, axis=1
        )  # Shape: (1, 1, context_size)
        context_tensor = tf.repeat(
            context_tensor, repeats=tf.shape(t9_vectorized)[1], axis=1
        )

        # Prédire le caractère suivant
        inputs = {"t9_input": t9_vectorized, "context_input": context_tensor}
        predictions = model.predict(
            inputs, verbose=0
        )  # Shape: (1, seq_len, vocab_size)

        # Obtenir la prédiction pour la position actuelle
        pred_char_idx = np.argmax(predictions[0, i], axis=-1)
        pred_char = target_tv.get_vocabulary()[pred_char_idx]

        # Ajouter le caractère généré (sauf si c'est un token spécial)
        if pred_char not in ["", "[UNK]"]:
            generated_chars.append(pred_char)

        # Mettre à jour le contexte
        context.append(int(pred_char_idx))
        if len(context) > context_size:
            context.pop(0)

    # Convertir la liste de caractères en chaîne
    return "".join(generated_chars)


def evaluate_char_predictions(
    references: List[str], predictions: List[str], output_dict: bool = False
) -> Union[str, Dict]:
    """
    Calcule la précision, le rappel et la F1-score pour une liste de chaînes de caractères sinogrammes.
    Chaque paire (ref, pred) est comparée caractère par caractère avec gestion des longueurs différentes.
    Utilise classification_report pour un tableau détaillé par classe.

    Args:
        references (List[str]): Réponses de référence (ground truth)
        predictions (List[str]): Réponses générées par le modèle
        output_dict (bool): Si True, retourne un dictionnaire; sinon, une chaîne formatée

    Returns:
        Union[str, Dict]: Rapport de classification (chaîne ou dictionnaire selon output_dict)
    """
    y_true, y_pred = [], []
    for ref, pred in zip(references, predictions):
        min_len = min(len(ref), len(pred))
        y_true.extend(ref[:min_len])
        y_pred.extend(pred[:min_len])

        # FP pour surplus prédiction
        if len(pred) > min_len:
            y_pred.extend(pred[min_len:])
            y_true.extend(["[PAD]"] * (len(pred) - min_len))
        # FN pour surplus référence
        if len(ref) > min_len:
            y_true.extend(ref[min_len:])
            y_pred.extend(["[PAD]"] * (len(ref) - min_len))

    # Filtrer les paires avec [PAD]
    mask = [(t != "[PAD]" and p != "[PAD]") for t, p in zip(y_true, y_pred)]
    y_true_f = [t for t, m in zip(y_true, mask) if m]
    y_pred_f = [p for p, m in zip(y_pred, mask) if m]

    # Générer le rapport
    return classification_report(
        y_true_f, y_pred_f, zero_division=0, output_dict=output_dict
    )


def evaluation_tradi(subset_df, model_name, model, input_tv, target_tv):
    """
    Évalue les prédictions de caractères sur un sous-ensemble de données.
    Args:
        subset_df (pd.DataFrame): Sous-ensemble de données à évaluer.
        model_name (str): Nom du modèle.
        model: Modèle Keras chargé.
        input_tv: Couche TextVectorization pour les entrées T9.
        target_tv: Couche TextVectorization pour les caractères cibles.
    """
    refs, preds = [], []
    for _, row in subset_df.iterrows():
        refs.append(row["target_char"])
        pred = generate_text(model, row["t9"], row["context"], input_tv, target_tv)
        preds.append(pred)
    print(f"Résultats pour {model_name} :")
    print(classification_report(refs, preds, output_dict=False, zero_division=0))
