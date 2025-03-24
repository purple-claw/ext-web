from flask import Flask, request, jsonify
import spacy
from collections import defaultdict

app = Flask(__name__)
nlp = spacy.load("en_core_web_sm")

def extract_relations(text):
    doc = nlp(text)
    relations = defaultdict(list)
    
    for token in doc:
        if token.dep_ in ("nsubj", "dobj", "pobj", "attr"):
            head = token.head.text
            relations[head].append(token.text)
    
    return relations

@app.route("/process", methods=["POST"])
def process_text():
    data = request.json
    text = data.get("text", "")
    relations = extract_relations(text)
    return jsonify({"relations": relations})

if __name__ == "__main__":
    app.run(debug=True)
