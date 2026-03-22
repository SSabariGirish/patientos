import os
import io
import json
import google.generativeai as genai
from flask import Flask, render_template, request, jsonify
from dotenv import load_dotenv
import PIL.Image

# Load API Key
load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# Use the current stable Flash model for speed and multimodal capabilities
model = genai.GenerativeModel('gemini-3.1-flash-lite-preview')
app = Flask(__name__)

# --- CORE LOGIC FUNCTIONS ---

def analyze_combination_gemini(current_meds, planned_med, reason, current_images=[], planned_images=[]):
    """Analyzes the safety using text and multiple categories of images."""
    
    content_payload = [
        f"User's manual text for CURRENT Medications: {current_meds if current_meds.strip() else 'None provided.'}\n",
        f"User's manual text for PLANNED Medication: {planned_med if planned_med.strip() else 'None provided.'}\n",
        f"Reason for taking planned medication: {reason}\n\n"
    ]
    
    if current_images:
        content_payload.append("IMAGES OF CURRENT MEDICATIONS:")
        content_payload.extend(current_images)
        
    if planned_images:
        content_payload.append("IMAGES OF PLANNED MEDICATION:")
        content_payload.extend(planned_images)

    prompt_text = """
    CRITICAL INSTRUCTION: You MUST read the attached images. 
    Treat the 'Current Medication' images + text as what the user is already taking.
    Treat the 'Planned Medication' images + text as what the user wants to add.

    Analyze the combination of ALL current medications with the planned medication for:
    1. Duplicate active ingredients (Overdose risk).
    2. Known severe drug-drug interactions.

    Return ONLY a raw JSON object (no markdown, no backticks) with:
    - "alert_level": "success", "warning", or "danger".
    - "title": A short, punchy title.
    - "explanation": Explicitly state what you identified from their images AND text for both current and planned meds.
    - "the_clash": Detailed explanation. Include standard adult dosages and safe alternative suggestions if applicable. Use HTML <br> and <strong> tags.
    - "action": Direct, actionable advice.
    """
    content_payload.append(prompt_text)

    try:
        response = model.generate_content(content_payload)
        
        raw_text = response.text.strip()
        if raw_text.startswith("```json"):
            raw_text = raw_text[7:-3]
        elif raw_text.startswith("```"):
            raw_text = raw_text[3:-3]
            
        return json.loads(raw_text)

    except Exception as e:
        print(f"Gemini API Error: {e}")
        return {
            "alert_level": "warning",
            "title": "⚠️ System Error",
            "explanation": "We could not process the images or text at this time.",
            "the_clash": "Please try typing your medications manually.",
            "action": "If you are unsure, consult a pharmacist or GP."
        }

def quick_extract_text(images):
    """Lightweight prompt just to read the image text for the UI auto-fill."""
    prompt = "Read the attached image(s). Return ONLY a comma-separated list of the medication names (and dosages if visible). Do not include any conversational text, markdown, or explanation. Just the list."
    try:
        response = model.generate_content([prompt] + images)
        return response.text.strip()
    except Exception as e:
        error_msg = str(e)
        print(e)
        if "429" in error_msg or "quota" in error_msg.lower():
            print("❌ Quota Exceeded! The API is temporarily locked.")
            return "API Busy - Please type manually"
        print(f"❌ Extraction Error: {e}")
        return ""

# --- FLASK ROUTES ---

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        current_meds = request.form.get('current_meds', '')
        planned_med = request.form.get('planned_med', '')
        reason = request.form.get('reason', '')
        
        # Grab BOTH sets of images for the main analysis
        raw_current_files = request.files.getlist('prescription_images')
        raw_planned_files = request.files.getlist('planned_prescription_images')
        
        current_images = []
        planned_images = []
        
        for file in raw_current_files:
            if file and file.filename != '':
                img = PIL.Image.open(io.BytesIO(file.read())).convert('RGB')
                current_images.append(img)
                
        for file in raw_planned_files:
            if file and file.filename != '':
                img = PIL.Image.open(io.BytesIO(file.read())).convert('RGB')
                planned_images.append(img)
        
        result = analyze_combination_gemini(current_meds, planned_med, reason, current_images, planned_images)
        return render_template('results.html', result=result, current=current_meds, planned=planned_med)
        
    return render_template('index.html')

@app.route('/extract_meds', methods=['POST'])
def extract_meds():
    """Universal extraction endpoint with heavy debugging for AJAX calls."""
    print("\n--- 🔍 BACKGROUND EXTRACTION STARTED ---")
    images = []
    
    try:
        # Explicitly look for the 'extraction_files' key we defined in JS
        files = request.files.getlist('extraction_files')
        print(f"Files received by backend: {len(files)}")
        
        for file in files:
            if file and file.filename != '':
                print(f"Processing image: {file.filename}")
                file_bytes = file.read()
                img = PIL.Image.open(io.BytesIO(file_bytes)).convert('RGB')
                images.append(img)
                
        if not images:
            print("❌ No valid images were processed into RAM.")
            return jsonify({"extracted_text": ""})
            
        print("Sending images to Gemini for quick read...")
        extracted_text = quick_extract_text(images)
        print(f"✅ Text returned from AI: '{extracted_text}'")
        print("----------------------------------------\n")
        
        return jsonify({"extracted_text": extracted_text})

    except Exception as e:
        print(f"❌ FATAL SERVER ERROR in /extract_meds: {e}")
        return jsonify({"extracted_text": ""}), 500

@app.route('/panic')
def panic():
    """Renders the emergency instructions page."""
    return render_template('panic.html')

if __name__ == '__main__':
    app.run(debug=True)