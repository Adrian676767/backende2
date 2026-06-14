from flask import Flask, render_template, jsonify, request
import requests
import os
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)

# HuggingFace API token z environment premennej
HUGGINGFACE_TOKEN = os.getenv('HUGGINGFACE_TOKEN')
API_URL = "https://api-inference.huggingface.co/models/Qwen/Qwen2.5-7B-Instruct"

students_db = [
    {"id": 1, "name": "Michal Horváth", "age": 16, "vyska": 175,},
    {"id": 2, "name": "Sofia Kováčová", "age": 15, "vyska": 162},
    {"id": 3, "name": "Jakub Malý", "age": 17, "vyska": 182},
    {"id": 4, "name": "Ema Srncová", "age": 16, "vyska": 168},
    {"id": 5, "name": "Matúš Veľký", "age": 15, "vyska": 170},
    {"id": 6, "name": "Lucia Malá", "age": 17, "vyska": 160},
    {"id": 7, "name": "David Novák", "age": 16, "vyska": 178},
    {"id": 8, "name": "Peter Tóth", "age": 15, "vyska": 165},
    {"id": 9, "name": "Nela Szabóová", "age": 17, "vyska": 172},
    {"id": 10, "name": "Adam Molnár", "age": 16, "vyska": 180},
    {"id": 11, "name": "Nina Balážová", "age": 15, "vyska": 158}
]

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/students')
def get_students():
    sort_by = request.args.get('sort', 'name')
    if sort_by == 'name':
        sorted_students = sorted(students_db, key=lambda x: x['name'])
    elif sort_by == 'age':
        sorted_students = sorted(students_db, key=lambda x: x['age'])
    elif sort_by == 'vyska':
        sorted_students = sorted(students_db, key=lambda x: x['vyska'])
    else:
        sorted_students = students_db
    return jsonify(sorted_students)

@app.route('/chat', methods=['POST'])
def chat_ai():
    data = request.json
    user_message = data.get('message', '')
    student_info = data.get('student_info', {})
    
    student_name = student_info.get('name', 'žiak')
    student_age = student_info.get('age', '16')
    student_vyska = student_info.get('vyska', '170')
    
    print(f"[DEBUG] Message: {user_message}")
    print(f"[DEBUG] Student: {student_name}")
    
    # Špeciálna inštrukcia – AI hrá rolu daného žiaka
    system_instruction = (
        f"Teraz si študent strednej školy. Voláš sa {student_name}, máš {student_age} rokov a meriaš {student_vyska} cm. "
        f"Odpovedaj v prvej osobe ako 'ja'. Hovor uvoľnene, priateľsky, ako mladý študent. Používaj emojis. "
        f"Komunikuj iba po slovensky. Nepíš dlhé texty - stručne a k veci!"
    )
    
    try:
        GROQ_API_KEY = os.getenv('GROQ_API_KEY')
        
        if not GROQ_API_KEY:
            print("[ERROR] GROQ_API_KEY je prázdny!")
            return jsonify({
                "response": "⚠️ API key nie je nastavený! 😅"
            }), 500
        
        import requests
        
        headers = {
            "Authorization": f"Bearer {GROQ_API_KEY}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": "mixtral-8x7b-32768",  # Rýchly a silný model
            "messages": [
                {"role": "system", "content": system_instruction},
                {"role": "user", "content": user_message}
            ],
            "max_tokens": 150,
            "temperature": 0.8
        }
        
        print("[INFO] Posielam na Groq API...")
        response = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            json=payload,
            headers=headers,
            timeout=30
        )
        
        print(f"[INFO] Groq status: {response.status_code}")
        
        if response.status_code == 401:
            print("[ERROR] Groq token je neplatný!")
            return jsonify({
                "response": "❌ API key je neplatný! 😅"
            }), 401
        
        if response.status_code != 200:
            print(f"[ERROR] Groq chyba: {response.status_code}")
            print(f"[ERROR] Response: {response.text}")
            return jsonify({
                "response": "⚠️ Chyba pri komuniácií s AI. Skús to neskôr! 🤖"
            })
        
        result = response.json()
        ai_response = result['choices'][0]['message']['content'].strip()
        
        if not ai_response:
            ai_response = "Hmm, neviem si spomeniť čo si vravel... 🤔"
        
        print(f"[INFO] Response: {ai_response}")
        
    except requests.Timeout:
        print("[ERROR] Request timeout")
        ai_response = "Server je pomalý, skús to neskôr! ⏳"
    except Exception as e:
        print(f"[ERROR] Exception: {str(e)}")
        ai_response = f"Chyba: {str(e)[:30]} 😅"
    
    return jsonify({"response": ai_response})

if __name__ == '__main__':
    # Pre Render a ostatné produkčné prostredia
    port = int(os.getenv('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
