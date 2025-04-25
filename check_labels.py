import tensorflow as tf
import numpy as np

# Load the trained model
MODEL_PATH = "model_checkpoint.h5"
model = tf.keras.models.load_model(MODEL_PATH)

# Manually defined class names (Ensure it matches your dataset)
CLASS_NAMES = [
    "Acne", "Actinic Keratosis", "Benign Tumors", "Bullous", "Candidiasis", "Drug Eruption", "Eczema",
    "Infestations Bites", "Lichen", "Lupus", "Moles", "Psoriasis", "Rosacea", "Seborrh Keratoses",
    "Skin Cancer", "Sunlight Damage", "Tinea", "Unknown or No Disease", "Vascular Tumors", "Vasculitis",
    "Vitiligo", "Warts", "Melanoma", "Hives (Urticaria)"
]

# Ensure the model's last layer has the correct number of classes
num_classes_in_model = model.layers[-1].output_shape[-1]
num_classes_in_list = len(CLASS_NAMES)

if num_classes_in_model == num_classes_in_list:
    print(f"✅ The number of classes in the model ({num_classes_in_model}) matches CLASS_NAMES ({num_classes_in_list})!\n")
    
    # Print index mapping
    print("✅ Label-to-Disease Mapping from the Model:")
    for i, label in enumerate(CLASS_NAMES):
        print(f"Index {i}: {label}")

else:
    print(f"❌ Mismatch! Model has {num_classes_in_model} classes, but CLASS_NAMES has {num_classes_in_list} classes.")
