# Homework Exercises

This folder contains the home practice exercises for lesson 04.

## Exercise 01: Document Classification and Summarization

The goal of this exercise is to build a Python program that reads text documents from a folder, sends each document to an LLM, classifies the document into one of three topics, summarizes it, and saves the summary in a folder named after the detected class.

Allowed classes:

- cars
- sport
- music

The program uses one LLM call per document.

The model must return JSON in this format:

```json
{
  "class_name": "cars",
  "summary": "short summary of the document"
}
```

## Project Structure

```text
exercise01-document-classification/
  documents/
    car_1.txt
    car_2.txt
    sport_1.txt
    music_1.txt
    music_2.txt
  summaries/
  document_classifier.py
```

## Requirements

Install dependencies using:

```bash
pip install -r requirements.txt
```

requirements.txt:

```text
anthropic
python-dotenv
```

## Environment Variables

Create a `.env` file in the root lesson folder:

```env
ANTHROPIC_API_KEY=your_api_key_here
```

## Running the Program

From the `lesson-04-llm-apis` folder run:

```bash
python homework/exercise01-document-classification/document_classifier.py
```

## Expected Output

The program will:

1. Read all `.txt` files from the `documents` folder
2. Send each document to the LLM
3. Receive a JSON response with:
   - class_name
   - summary
4