from flask import Flask, render_template, request, send_file, jsonify
import os
from dotenv import load_dotenv
from image_generator import generate_survey_image
from text_generation import generate_email

app = Flask(__name__)

load_dotenv()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/generate', methods=['POST'])
def generate():
    # Get all form data
    survey_name = request.form.get('survey_name')
    medical_specialty = request.form.get('medical_specialty', 'general')
    tone = request.form.get('tone', 'professional')
    image_style = request.form.get('image_style', 'professional')
    compensation = request.form.get('compensation')
    loi = request.form.get('loi')
    educational_info = request.form.get('educational_info')
    include_text = request.form.get('include_text') == 'on'

    # Validate inputs
    errors = []
    if not survey_name or len(survey_name.strip()) < 3:
        errors.append("Survey name is required and must be at least 3 characters.")
    if not compensation or not compensation.isdigit() or int(compensation) < 0:
        errors.append("Compensation must be a non-negative number.")
    if not loi or not loi.isdigit() or int(loi) < 1:
        errors.append("Length of interview must be a positive number.")
    if tone.lower() == "educational" and not educational_info:
        errors.append("Educational tone requires educational information.")

    if errors:
        return jsonify({"success": False, "errors": errors})

    # Generate email content
    email_content, email_error = generate_email(survey_name, compensation, loi, tone, educational_info)
    if email_error:
        return jsonify({"success": False, "errors": [email_error]})

    # Generate image - now using only 2 parameters
    pil_image, image_filename, image_base64, image_message = generate_survey_image(
        survey_name=survey_name,
        medical_specialty=medical_specialty
    )

    if not image_filename:
        return jsonify({"success": False, "errors": [image_message]})

    return jsonify({
        "success": True,
        "email_content": email_content,
        "image_base64": image_base64,
        "image_filename": image_filename,
        "image_description": image_message,
        "download_url": f"/download/{image_filename}"
    })

@app.route('/download/<path:filename>')
def download(filename):
    return send_file(filename, as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True)