import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing import image_dataset_from_directory

# Load the dataset (Ensure you have 'train' and 'test' folders structured properly)
dataset_path = "dataset"  # Update this with your dataset path
train_dataset = image_dataset_from_directory(
    dataset_path + "/train",
    image_size=(150, 150),
    batch_size=32,
    shuffle=True
)
test_dataset = image_dataset_from_directory(
    dataset_path + "/test",
    image_size=(150, 150),
    batch_size=32
)

# Get class names (Desert, Forest, Mountain, Plains)
class_names = train_dataset.class_names
print(f"Classes found: {class_names}")

# Build the CNN model
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(256, activation='relu'),
    layers.Dense(len(class_names), activation='softmax')  # 4 classes
])

# Compile the model
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Train the model
model.fit(train_dataset, validation_data=test_dataset, epochs=10)

# Save the trained model
model.save("terrain_classifier.h5")
print("Model successfully saved as 'terrain_classifier.h5'")
