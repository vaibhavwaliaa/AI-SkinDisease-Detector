import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2

# Define paths to the train and test directories
train_dir = 'dataset/Skin_Disease_Dataset/train'
test_dir = 'dataset/Skin_Disease_Dataset/test'

# ImageDataGenerator for preprocessing and data augmentation
train_datagen = ImageDataGenerator(
    rescale=1./255,         # Rescale pixel values to [0, 1]
    rotation_range=30,     # Random rotation
    width_shift_range=0.2, # Random width shift
    height_shift_range=0.2, # Random height shift
    shear_range=0.2,       # Random shear
    zoom_range=0.2,        # Random zoom
    horizontal_flip=True,  # Horizontal flip
    fill_mode='nearest'
)

test_datagen = ImageDataGenerator(rescale=1./255)  # Just rescale for test data

# Flow data from directories
train_data = train_datagen.flow_from_directory(
    train_dir,
    target_size=(224, 224),  # Resize images to 224x224 (VGG16 input size)
    batch_size=32,
    class_mode='categorical',  # Categorical labels (multi-class classification)
)

test_data = test_datagen.flow_from_directory(
    test_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',  # Categorical labels
)

print("Training and test data are ready!")





# Load VGG16 pre-trained model without the top (classification) layer
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Freeze the layers of VGG16 to retain the pre-trained features
base_model.trainable = False

# Build the full model
model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(1024, activation='relu'),
    layers.Dense(train_data.num_classes, activation='softmax')  # Number of classes = number of diseases
])

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])





# Automatically detect and use GPU if available, otherwise use CPU
strategy = tf.distribute.MirroredStrategy()  # Works for multiple GPUs or CPU fallback
print(f"✅ Using {strategy.num_replicas_in_sync} GPU(s)" if tf.config.list_physical_devices('GPU') else "✅ Using CPU")

# Create and compile the model within the strategy scope
with strategy.scope():
    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dropout(0.3),  # Added dropout for better generalization
        layers.Dense(1024, activation='relu'),
        layers.Dense(train_data.num_classes, activation='softmax')
    ])
    
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), 
                  loss='categorical_crossentropy', 
                  metrics=['accuracy'])

model.summary()

# Get class names
class_labels = train_data.class_indices
class_labels = {v: k for k, v in class_labels.items()}  # Reverse mapping

print("Class Labels Mapping:", class_labels)
