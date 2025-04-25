import os
import numpy as np
from flask import Flask, request, jsonify, render_template, redirect, url_for, send_from_directory
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from flask_pymongo import PyMongo
from twilio.rest import Client  # Twilio SMS Integration

app = Flask(__name__, static_folder="static")

# MongoDB Configuration
app.config["MONGO_URI"] = "mongodb://localhost:27017/contactDB"
mongo = PyMongo(app)

# Twilio Configuration
TWILIO_SID = "ACd4701a9a1f7f1b71b5a4b10bfd4389fb"
TWILIO_AUTH_TOKEN = "4236b47f84521b9bf8e5e484a97f97d1"
TWILIO_PHONE_NUMBER = "+19036368066"
YOUR_PHONE_NUMBER = "+919817245943"

client = Client(TWILIO_SID, TWILIO_AUTH_TOKEN)

def send_sms(message_body):
    """Sends an SMS using Twilio API"""
    message = client.messages.create(
        body=message_body,
        from_=TWILIO_PHONE_NUMBER,
        to=YOUR_PHONE_NUMBER
    )
    return message.sid

# Create upload directory
UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Load trained model
model = load_model("model_checkpoint.h5")

# Disease Labels
class_labels = {
    0: 'Acne', 1: 'Actinic Keratosis', 2: 'Benign Tumors', 3: 'Bullous',
    4: 'Candidiasis', 5: 'Drug Eruption', 6: 'Eczema', 7: 'Hives (Urticaria)',
    8: 'Infestations Bites', 9: 'Lichen', 10: 'Lupus', 11: 'Melanoma',
    12: 'Moles', 13: 'Psoriasis', 14: 'Rosacea', 15: 'Seborrh Keratoses',
    16: 'Skin Cancer', 17: 'Sunlight Damage', 18: 'Tinea',
    19: 'Unknown or No Disease', 20: 'Vascular Tumors', 21: 'Vasculitis',
    22: 'Vitiligo', 23: 'Warts'
}

# Disease Information Dictionary
disease_info = {
    "Acne": {
        "description": "A common skin condition that occurs when hair follicles become clogged with oil and dead skin cells, leading to pimples, blackheads, and whiteheads.",
        "cause": "Hormonal changes, excessive oil production, bacteria, and inflammation.",
        "treatment": "Use topical treatments (benzoyl peroxide, salicylic acid), oral medications, and maintain proper skincare."
    },
    "Actinic Keratosis": {
        "description": "A rough, scaly patch on the skin caused by years of sun exposure, which may develop into skin cancer if untreated.",
        "cause": "Long-term exposure to ultraviolet (UV) light from the sun or tanning beds.",
        "treatment": "Cryotherapy, laser therapy, chemical peels, and topical medications."
    },
    "Benign Tumors": {
        "description": "Non-cancerous growths on or under the skin, usually harmless but sometimes requiring removal.",
        "cause": "Genetic factors, infections, or environmental exposure.",
        "treatment": "Monitoring, surgical removal, or laser therapy if necessary."
    },
    "Bullous": {
        "description": "A skin condition that causes large, fluid-filled blisters, often due to immune system disorders.",
        "cause": "Autoimmune diseases, infections, or allergic reactions.",
        "treatment": "Corticosteroids, immunosuppressants, and proper wound care."
    },
    "Candidiasis": {
        "description": "A fungal infection caused by Candida yeast, often affecting warm, moist areas of the body.",
        "cause": "Weakened immune system, diabetes, prolonged antibiotic use, and poor hygiene.",
        "treatment": "Antifungal creams, oral antifungal medications, and maintaining proper hygiene."
    },
    "Drug Eruption": {
        "description": "An adverse skin reaction caused by medications, leading to rashes, redness, or blisters.",
        "cause": "Allergic reaction or sensitivity to certain drugs.",
        "treatment": "Stopping the medication, antihistamines, corticosteroids, and hydration."
    },
    "Eczema": {
        "description": "A condition that makes the skin red, inflamed, and itchy.",
        "cause": "Genetics, allergens, irritants, and environmental triggers.",
        "treatment": "Moisturizers, corticosteroids, and avoiding known triggers."
    },
    "Hives (Urticaria)": {
        "description": "A skin reaction that causes itchy welts due to an allergic reaction or unknown triggers.",
        "cause": "Allergens, stress, infections, or medications.",
        "treatment": "Antihistamines, corticosteroids, and avoiding allergens."
    },
    "Infestations Bites": {
        "description": "Skin irritation and rashes caused by insect bites or parasitic infections such as scabies or lice.",
        "cause": "Bites from mosquitoes, fleas, ticks, mites, or lice infestations.",
        "treatment": "Topical creams, antihistamines, and proper hygiene to prevent further infestations."
    },
    "Lichen": {
        "description": "A skin condition characterized by thick, scaly patches that may be itchy or painful.",
        "cause": "Autoimmune reactions, chronic inflammation, or unknown triggers.",
        "treatment": "Corticosteroid creams, antihistamines, and phototherapy."
    },
    "Lupus": {
        "description": "An autoimmune disease that affects the skin, causing rashes and sensitivity to sunlight.",
        "cause": "Genetic predisposition, environmental factors, and immune system dysfunction.",
        "treatment": "Anti-inflammatory drugs, immunosuppressants, and avoiding sunlight exposure."
    },
    "Melanoma": {
        "description": "A serious and aggressive form of skin cancer that develops from melanocytes.",
        "cause": "Excessive UV exposure, genetic mutations, and fair skin type.",
        "treatment": "Surgical removal, chemotherapy, immunotherapy, and radiation therapy."
    },
    "Moles": {
        "description": "Clusters of pigmented skin cells that appear as small, dark brown spots.",
        "cause": "Genetic factors, sun exposure, and hormonal changes.",
        "treatment": "Usually harmless, but removal is recommended if a mole changes in size, shape, or color."
    },
    "Psoriasis": {
        "description": "A chronic autoimmune skin disease that speeds up the life cycle of skin cells, causing scaly patches.",
        "cause": "Immune system dysfunction, genetic factors, and environmental triggers.",
        "treatment": "Topical treatments, phototherapy, and systemic medications."
    },
    "Rosacea": {
        "description": "A chronic skin condition causing redness, visible blood vessels, and bumps on the face.",
        "cause": "Unknown, but triggers include sun exposure, stress, spicy foods, and alcohol.",
        "treatment": "Topical treatments, antibiotics, laser therapy, and avoiding triggers."
    },
    "Seborrh Keratoses": {
        "description": "Non-cancerous, wart-like growths that appear on the skin, often with age.",
        "cause": "Genetics and aging.",
        "treatment": "Cryotherapy, laser removal, or electrosurgery if necessary."
    },
    "Skin Cancer": {
        "description": "Uncontrolled growth of abnormal skin cells, often due to sun exposure.",
        "cause": "UV radiation, genetics, and weakened immune system.",
        "treatment": "Surgery, chemotherapy, radiation therapy, and immunotherapy."
    },
    "Sunlight Damage": {
        "description": "Skin damage caused by prolonged exposure to UV rays, leading to premature aging and increased cancer risk.",
        "cause": "Excessive sun exposure, tanning beds, and lack of sunscreen use.",
        "treatment": "Sunscreen, antioxidants, retinoids, and skin-repairing treatments."
    },
    "Tinea": {
        "description": "A fungal infection affecting the skin, scalp, or nails, also known as ringworm.",
        "cause": "Fungal overgrowth due to moisture, poor hygiene, or direct contact.",
        "treatment": "Antifungal creams, oral antifungal medications, and maintaining dry skin."
    },
    "Unknown or No Disease": {
        "description": "No recognizable skin disease detected.",
        "cause": "N/A",
        "treatment": "Consult a dermatologist if symptoms persist."
    },
    "Vascular Tumors": {
        "description": "Abnormal growth of blood vessels in the skin, which may be benign or malignant.",
        "cause": "Genetic mutations, environmental triggers, or unknown causes.",
        "treatment": "Monitoring, laser therapy, or surgical removal if necessary."
    },
    "Vasculitis": {
        "description": "Inflammation of blood vessels, leading to skin rashes, ulcers, or organ damage.",
        "cause": "Autoimmune conditions, infections, or allergic reactions.",
        "treatment": "Corticosteroids, immunosuppressants, and managing underlying conditions."
    },
    "Vitiligo": {
        "description": "A condition where the skin loses its pigment cells, causing white patches.",
        "cause": "Autoimmune disorder, genetic factors, and unknown triggers.",
        "treatment": "Topical corticosteroids, light therapy, and skin grafting."
    },
    "Warts": {
        "description": "Small, rough skin growths caused by the human papillomavirus (HPV).",
        "cause": "Direct contact with HPV, weakened immune system.",
        "treatment": "Cryotherapy, salicylic acid, and laser removal."
    }
}

# Function to preprocess image
def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# Home route
@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

# Contact page route
@app.route("/contact", methods=["GET", "POST"])
def contact():
    if request.method == "GET":
        return send_from_directory("contact-us", "public/index.html")
    
    elif request.method == "POST":
        name = request.form.get("name")
        email = request.form.get("email")
        message = request.form.get("message")

        if not name or not email or not message:
            return jsonify({"error": "All fields are required"}), 400

        # Insert into MongoDB
        mongo.db.contacts.insert_one({"name": name, "email": email, "message": message})

        return jsonify({"message": "Message sent successfully!"}), 201

# Fetch messages from MongoDB
@app.route("/messages", methods=["GET"])
def get_messages():
    contacts = list(mongo.db.contacts.find({}, {"_id": 0}))
    return jsonify(contacts), 200

# Location page route
@app.route("/location", methods=["GET"])
def location():
    return render_template("location.html")


# Predict route
@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded!"})
    
    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No selected file!"})
    
    if file:
        filepath = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
        file.save(filepath)

        # Preprocess and predict
        img_array = preprocess_image(filepath)
        predictions = model.predict(img_array)
        class_index = np.argmax(predictions)
        confidence = float(np.max(predictions)) * 100

        # Get Disease Name
        predicted_disease = class_labels[class_index]

        # Get Disease Details (if available)
        disease_details = disease_info.get(predicted_disease, {
            "description": "No information available.",
            "cause": "Unknown.",
            "treatment": "Consult a doctor for further evaluation."
        })

        result = {
            "disease": predicted_disease,
            "confidence": confidence,
            "description": disease_details["description"],
            "cause": disease_details["cause"],
            "treatment": disease_details["treatment"],
            "image_path": filepath
        }

        # Send SMS with Disease Prediction + Treatment
        message_body = (
            f"Prediction: {predicted_disease}\n"
            f"Treatment: {disease_details['treatment']}"
            f"Description: {disease_details['description']}"
            f"Cause: {disease_details['cause']}"
        )
        sms_sid = send_sms(message_body)
        result["sms_status"] = sms_sid

        return jsonify(result)  # Return JSON response

if __name__ == "__main__":
    app.run(debug=True, port=5000)
