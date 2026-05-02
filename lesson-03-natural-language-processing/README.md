# NLP examples

Small scripts for natural language processing workflows in this folder.

## Environment

Create and use a virtual environment (recommended), then install dependencies from this directory:

```bash
python -m venv .venv
```

**Windows (PowerShell):**

```powershell
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

**macOS / Linux:**

```bash
source .venv/bin/activate
pip install -r requirements.txt
```

## spaCy English pipeline (required for `4_spacy_ner.py`)

spaCy does **not** ship the pretrained English model inside the `spacy` package. You must download the pipeline once per environment:

```bash
python -m spacy download en_core_web_sm
```

If you skip this step, `spacy.load("en_core_web_sm")` fails with **`OSError: [E050] Can't find model 'en_core_web_sm'`**. Run the command above with the **same** Python interpreter (or activated venv) you use to run the script.

## NER example (`4_spacy_ner.py`)

[Named Entity Recognition](https://spacy.io/usage/linguistic-features#named-entities) with spaCy: pretrained `en_core_web_sm` on sample text.

```bash
python 4_spacy_ner.py
```

## Stemming and lemmatization (`3_stemming_lemmatization_nltk.py`)

NLTK demo comparing **stemming** (Porter stemmer) and **lemmatization** (WordNet). The script downloads required NLTK data (`wordnet`, `omw-1.4`) if missing.

```bash
python 3_stemming_lemmatization_nltk.py
```
