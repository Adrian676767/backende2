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
    {"id": 1, "name": "Michal Horváth", "age": 16, "vyska": 175},
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
    
    # Špeciálna inštrukcia – AI hrá rolu daného žiaka
    system_instruction = (
        f"Teraz hráš rolu človeka. Si žiak strednej školy, voláš sa {student_name}, "
        f"máš {student_age} rokov a meriaš {student_vyska} cm. "
        f"Odpovedaj priamo v prvej osobe (ako 'ja'), hovor uvoľnene, ako mladý študent, používaj emojis. "
        f"Komunikuj výhradne po slovensky a primerane k svojmu veku. Nepíš dlhé texty, odpovedaj stručne a k veci."
    )
    
    prompt = f"<|im_start|>system\n{system_instruction}<|im_end|>\n<|im_start|>user\n{user_message}<|im_end|>\n<|im_start|>assistant\n"
    
    try:
        if not HUGGINGFACE_TOKEN:
            return jsonify({
                "response": "⚠️ HuggingFace token nie je nastavený! Skontroluj environment premenné v Render.",
                "error": "HUGGINGFACE_TOKEN missing"
            }), 500
        
        headers = {
            "Authorization": f"Bearer {HUGGINGFACE_TOKEN}"
        }
        
        payload = {
            "inputs": prompt,
            "parameters": {
                "max_new_tokens": 150,
                "temperature": 0.8,
                "return_full_text": False
            }
        }
        
        response = requests.post(API_URL, json=payload, headers=headers, timeout=15)
        
        # Overenie chýb
        if response.status_code != 200:
            print(f"HuggingFace API chyba: {response.status_code} - {response.text}")
            return jsonify({
                "response": "Prepáč, model je teraz zaneprázdnený. Skús to znova za chvíľu! 😅"
            })
        
        result = response.json()
        
        if isinstance(result, list) and len(result) > 0:
            ai_response = result[0].get('generated_text', '').strip()
        else:
            ai_response = "Hmm, neviem si spomeniť čo si vravel... 🤔"
            
    except requests.Timeout:
        ai_response = "Užas, server je pomalý. Skús to znova! ⏳"
    except Exception as e:
        print(f"Chyba: {e}")
        ai_response = "Uups, niečo sa pokazilo! Skús to znova prosím. 😅"
    
    return jsonify({"response": ai_response})

if __name__ == '__main__':
    # Pre Render a ostatné produkčné prostredia
    port = int(os.getenv('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
