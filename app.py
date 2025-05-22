from flask import Flask, render_template, request, redirect, url_for, session, jsonify
from werkzeug.utils import secure_filename
from plant_diagnosis_model import predict_image_with_gradcam
from openai import OpenAI
from dotenv import load_dotenv
import os
import secrets

load_dotenv(dotenv_path="key.env") 


app = Flask(__name__)
app.secret_key = secrets.token_hex(16)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}


client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


USERS = {
    "admin@example.com": "password123"
}


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS



@app.route('/')
def index():
    if 'user' not in session:
        return redirect('/login')
    return render_template('index.html')

@app.route('/', methods=['POST'])
def diagnose_text():
    if 'user' not in session:
        return redirect('/login')

    height = float(request.form['height'])
    name = request.form['name']
    disease = request.form['disease']

    prompt = (
        f"Diagnose the potential plant disease based on the following information:\n"
        f"Plant Name: {name}\n"
        f"Height: {height} feet\n"
        f"Symptoms: {disease}"
    )

    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful assistant named Farmer from Farm Sense. "
                               "You predict plant diseases based on the information provided, give detailed tips "
                               "on treatment, and provide no unnecessary fluff. Your output includes a numeric "
                               "disease likelihood and actionable steps."
                },
                {"role": "user", "content": prompt}
            ],
            max_tokens=1500,
            temperature=0.5
        )
        diagnosis = response.choices[0].message.content.strip()
    except Exception as e:
        diagnosis = f"Error: {str(e)}"

    return render_template('result.html', diagnosis=diagnosis)

@app.route('/upload', methods=['POST'])
def upload_image():
    if 'user' not in session:
        return redirect('/login')

    if 'file' not in request.files or request.files['file'].filename == '':
        return "No file selected"

    file = request.files['file']
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        save_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(save_path)

        try:
            diagnosis, gradcam_path = predict_image_with_gradcam(save_path)
        except Exception as e:
            return f"Model error: {str(e)}"

        os.remove(save_path)

        return render_template('result.html', diagnosis=diagnosis, image_url='/' + gradcam_path)

    return "Invalid file type"

@app.route('/ask', methods=['POST'])
def ask():
    if 'user' not in session:
        return jsonify({'error': 'Not authenticated'}), 401

    question = request.json.get('question')

    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful assistant specializing in farming and plant care. "
                               "You answer agricultural queries concisely and accurately."
                },
                {"role": "user", "content": question}
            ],
            max_tokens=1000,
            temperature=0.6
        )
        answer = response.choices[0].message.content.strip()
        return jsonify({'answer': answer})
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        if USERS.get(email) == password:
            session['user'] = email
            return redirect('/')
        else:
            return render_template('login.html', error="Invalid email or password")
    return render_template('login.html')

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        if email in USERS:
            return render_template('signup.html', error="User already exists")
        USERS[email] = password
        session['user'] = email
        return redirect('/')
    return render_template('signup.html')

@app.route('/logout')
def logout():
    session.clear()
    return redirect('/login')

if __name__ == '__main__':
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    app.run(debug=True, use_reloader=False)
