#!/usr/bin/env python3
import os
import uuid
import subprocess
import glob 
from flask import Flask, request, render_template, send_from_directory, redirect, url_for, flash, jsonify
from werkzeug.utils import secure_filename

BASE_DIR = os.path.abspath(os.path.dirname(__file__))

DATA_DIR = os.path.join(BASE_DIR, 'data')
CONTENT_DIR = os.path.join(DATA_DIR, 'content')
STYLE_DIR = os.path.join(DATA_DIR, 'style')
STYLIZED_DIR = os.path.join(DATA_DIR, 'stylized')
OUTPUT_JSON = os.path.join(DATA_DIR, 'cycle_scores.json')
ALLOWED_EXT = {'png', 'jpg', 'jpeg'}

def ensure_directories_exist():
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(CONTENT_DIR, exist_ok=True)
    os.makedirs(STYLE_DIR, exist_ok=True)
    os.makedirs(STYLIZED_DIR, exist_ok=True)
    print(f"Ensured directories exist: {DATA_DIR}")

# 用于启动时清理目录的函数
def cleanup_directory(directory_path):
    files_to_delete = glob.glob(os.path.join(directory_path, '*'))
    count = 0
    for f in files_to_delete:
        try:
            if os.path.isfile(f):
                os.remove(f)
                count += 1
        except Exception as e:
            print(f"Error deleting file {f} from {directory_path}: {e}")
    return count
# -----------------

app = Flask(__name__)
app.secret_key = 'replace-this-with-a-secure-key'
app.config['MAX_CONTENT_LENGTH'] = 200 * 1024 * 1024

# --- 辅助函数 ---
def allowed_filename(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXT

# -----------------

@app.route('/')
def index():
    ensure_directories_exist() 
    
    styles = [f for f in os.listdir(STYLE_DIR) if allowed_filename(f)]
    styles.insert(0, '__all__') 
    
    stylized = [f for f in os.listdir(STYLIZED_DIR) if allowed_filename(f)]
    return render_template('index.html', styles=styles, stylized=stylized)

@app.route('/upload', methods=['POST'])
def upload():
    ensure_directories_exist() 
    
    files = request.files.getlist('content_files')
    if not files:
        flash('No content images uploaded')
        return redirect(url_for('index'))

    saved = []
    for f in files:
        if f and allowed_filename(f.filename):
            filename = secure_filename(f.filename)
            unique_name = f"{uuid.uuid4().hex}_{filename}"
            dest = os.path.join(CONTENT_DIR, unique_name) 
            f.save(dest)
            saved.append(unique_name)
    flash(f'Saved {len(saved)} content images')
    return redirect(url_for('index'))

@app.route('/run', methods=['POST'])
def run_script():
    ensure_directories_exist()
    
    deleted_s_count = cleanup_directory(STYLIZED_DIR)
    style_filename = request.form.get('style_file')
    alphas = request.form.getlist('alphas') or ['0.3','0.5','0.7','0.9','1.0']
    max_samples = request.form.get('max_samples', '2000')

    if not style_filename:
        flash('Please select a style image')
        return redirect(url_for('index'))

    selected_style_path = ""
    if style_filename != '__all__':
        selected_style_path = os.path.join(STYLE_DIR, style_filename)
        if not os.path.exists(selected_style_path):
            flash('Selected style image does not exist')
            return redirect(url_for('index'))
        print("Preparing to run script with single style:", selected_style_path)
    else:
        print("Preparing to run script with ALL styles from:", STYLE_DIR)


    CODE_PATH = os.path.join('generate_cycle_scores.py')
    PROJECT_ROOT_DIR = os.path.join(BASE_DIR, os.pardir)
    
    cmd = [
        'python', CODE_PATH,
        '--content_dir', CONTENT_DIR,
        '--stylized_dir', STYLIZED_DIR,
        '--output_json', OUTPUT_JSON,
        '--alphas', *alphas,
        '--max_samples', str(max_samples),
    ]

    if style_filename == '__all__':
        cmd.extend(['--style_dir', STYLE_DIR])
    else:
        cmd.extend(['--style_file', selected_style_path])
        cmd.extend(['--style_dir', STYLE_DIR])

        with open(os.path.join(DATA_DIR, 'last_selected_style.txt'), 'w') as fh:
            print("Writing last selected style to:", DATA_DIR + 'last_selected_style.txt')
            fh.write(style_filename)


    try:
        proc = subprocess.run(cmd, cwd=PROJECT_ROOT_DIR, capture_output=True, text=True, check=True)
        flash('Generation complete')
        
    except subprocess.CalledProcessError as e:
        flash('Script failed to run: Check server logs')
        print(f"Error output:\n{e.stderr}")
        print(f"Stdout output:\n{e.stdout}")

    return redirect(url_for('index'))

@app.route('/data/<path:filename>')
def serve_data(filename):
    return send_from_directory(DATA_DIR, filename)

@app.route('/stylized/<path:filename>')
def serve_stylized(filename):
    return send_from_directory(STYLIZED_DIR, filename)

@app.route('/status')
def status():
    ensure_directories_exist()

    styles = [f for f in os.listdir(STYLE_DIR) if allowed_filename(f)]
    stylized = [f for f in os.listdir(STYLIZED_DIR) if allowed_filename(f)]
    return jsonify({
        'styles': styles,
        'stylized': stylized,
        'output_json': os.path.exists(OUTPUT_JSON)
    })

if __name__ == '__main__':
    
    ensure_directories_exist() 
    
    print("--- Cleaning up previous session data ---")
    deleted_c_count = cleanup_directory(CONTENT_DIR)
    deleted_s_count = cleanup_directory(STYLIZED_DIR)
    print(f"Cleared {deleted_c_count} files from 'content/' and {deleted_s_count} files from 'stylized/'.")
    print("-----------------------------------------")
    
    app.run(host='0.0.0.0', port=8080, debug=True)