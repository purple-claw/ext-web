from flask import Flask, request, jsonify
import os
from transformers import AutoModelForTokenClassification, AutoTokenizer, pipeline

MODEL_NAME = "nitinsri/mira-v1"

# Set Hugging Face token
HF_TOKEN = "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"  # Load from environment variable

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_auth_token=HF_TOKEN)
model = AutoModelForTokenClassification.from_pretrained(MODEL_NAME, use_auth_token=HF_TOKEN)

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
