from flask import Flask, request, jsonify
from transformers import AutoModelForTokenClassification, AutoTokenizer, pipeline

MODEL_NAME = "nitinsri/mira-v1"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForTokenClassification.from_pretrained(MODEL_NAME)

ner_pipeline = pipeline("ner", model=model, tokenizer=tokenizer)

app = Flask(__name__)

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.json
        text = data.get("text", "")

        if not text:
            return jsonify({"error": "No text provided"}), 400

        # Run model inference
        ner_results = ner_pipeline(text)
        
        # Process relationships into JSON format
        relationships = [{"word": ent["word"], "entity": ent["entity"]} for ent in ner_results]

        return jsonify({"relationships": relationships})
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
