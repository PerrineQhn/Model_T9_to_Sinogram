Système T9 chinois, inspiré du clavier prédictif de Sogou. L’idée serait de partir d’un corpus de textes en chinois, de générer les séquences pinyin et T9 correspondantes, et de développer un modèle capable de prédire les caractères chinois à partir des touches saisies en prenant en compte le contexte gauche.

### 1. Préparation des Données

L’objectif de cette étape est de transformer un corpus de textes en chinois, tel que le dataset "chinese-official-daily-news-since-2016", en un format exploitable pour entraîner un modèle seq2seq inspiré du système T9 de Sogou. Le modèle devra prendre en entrée des séquences numériques T9 dérivées du pinyin des caractères chinois et produire en sortie les caractères chinois correspondants, tout en tenant compte du contexte gauche pour résoudre les ambiguïtés. Voici les étapes prévues pour préparer les données :

Tout d’abord, le corpus sera chargé à partir d’un fichier CSV contenant des articles de presse en chinois. L’accent sera mis sur la colonne contenant le texte brut, qui fournit le matériel principal pour l’entraînement. Les informations des autres colonnes, comme les dates ou les titres, seront ignorées, car elles ne sont pas directement pertinentes pour la tâche.

Ensuite, un nettoyage des textes sera effectué pour garantir leur qualité. Seuls les caractères chinois et la ponctuation chinoise pertinente seront conservés, en éliminant les chiffres, les lettres latines, les symboles non désiré. Les données non valides, telles que les valeurs manquantes, seront remplacées par des chaînes vides pour éviter les erreurs ultérieures.

Pour faciliter la conversion des textes en pinyin et T9, les textes nettoyés seront segmentés en mots. 
Chaque mot segmenté sera ensuite converti en sa transcription pinyin sans tons, pour simplifier le processus. À partir du pinyin, une séquence T9 sera générée en associant chaque lettre latine à une touche numérique selon le standard T9 (par exemple, "a, b, c" correspondent à la touche 2, "d, e, f" à la touche 3, et ainsi de suite). Les caractères de ponctuation chinoise, comme les virgules ou les points, seront conservés tels quels pour préserver le contexte. Pour chaque caractère chinois, une séquence au format "caractère|pinyin|t9" sera créée, permettant de lier chaque caractère à sa représentation numérique tout en maintenant l’ordre contextuel. Ces séquences seront enregistrées dans un fichier pour référence future.

À partir de ces séquences, des paires d’entrée-sortie seront construites pour le modèle seq2seq. Les entrées consisteront en séquences de codes T9 séparés par des espaces, représentant la suite des touches numériques saisies. Les sorties seront des séquences de caractères chinois concaténés sans espaces, correspondant au texte attendu. Ces paires seront organisées dans une structure de données adaptée pour les étapes suivantes.

Pour préparer les données à l’entraînement, les séquences T9 et les séquences de caractères seront converties en indices numériques. Les séquences T9 seront tokenisées en unités de codes numériques, avec un vocabulaire limité aux combinaisons T9 les plus fréquentes. Les séquences de caractères chinois seront tokenisées au niveau des caractères, chaque caractère étant traité comme une unité distincte.

Enfin, les données seront divisées en trois ensembles : entraînement, validation et test. Une répartition standard, par exemple 80 % pour l’entraînement, 10 % pour la validation et 10 % pour le test, sera utilisée pour permettre l’apprentissage, l’ajustement des hyperparamètres et l’évaluation des performances du modèle. Cette division garantira que le modèle est testé sur des données non vues, tout en disposant d’un ensemble suffisant pour l’entraînement.

Ces étapes permettront de transformer le corpus brut en un format structuré, prêt pour l’entraînement d’un modèle capable de prédire des caractères chinois à partir de séquences T9, en exploitant le contexte gauche pour des prédictions précises et contextuellement appropriées.

### 2. Modèles utilisés et à comparer

### 3. Evaluation des modèles
