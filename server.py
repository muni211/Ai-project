from flask import Flask, request, jsonify
from core.ai_engine import AIEngine  # ייבוא מנוע AI
from nlp.nlp_module import NLPModule  # ייבוא מודול NLP

app = Flask(__name__)  # יצירת שרת Flask

# יצירת אובייקטים מהמנועים
ai_engine = AIEngine()
nlp_module = NLPModule()

@app.route("/process", methods=["POST"])
def process():
    data = request.json.get("text", "")
    nlp_result = nlp_module.process_text(data)
    ai_response = ai_engine.analyze(nlp_result)
    return jsonify({"response": ai_response})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)