from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Training data augmentation and rescaling
train_datagen = ImageDataGenerator(
    rescale=1./255,          # Normalize images to [0, 1]
    rotation_range=20,       # Random rotation
    width_shift_range=0.2,   # Random horizontal shift
    height_shift_range=0.2,  # Random vertical shift
    shear_range=0.2,         # Shear transformation
    zoom_range=0.2,          # Random zoom
    horizontal_flip=True,    # Flip horizontally
    fill_mode='nearest'      # Fill missing pixels after transformations
)

# Test/validation data rescaling only (no augmentation)
test_datagen = ImageDataGenerator(rescale=1./255)

# Set up data generators for training and validation
train_generator = train_datagen.flow_from_directory(
    'data/train',            # Training data directory
    target_size=(150, 150),  # Resize images
    batch_size=32,           # Batch size
    class_mode='sparse'      # For integer labels
)

validation_generator = test_datagen.flow_from_directory(
    'data/test',             # Validation data directory
    target_size=(150, 150),  # Resize images
    batch_size=32,           # Batch size
    class_mode='sparse'      # For integer labels
)

# Optionally, print a few sample batches to check the images
# Example of checking the first batch:
# X_batch, y_batch = train_generator.next()
# print("Batch shape:", X_batch.shape)
# print("Batch labels:", y_batch)
