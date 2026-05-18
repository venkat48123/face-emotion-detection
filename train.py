import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import os
from model import create_emotion_model

# Hyperparameters and Configuration
IMG_HEIGHT = 48
IMG_WIDTH = 48
BATCH_SIZE = 64
EPOCHS = 5
NUM_CLASSES = 4
DATA_DIR = 'dataset' # Should contain 'train' and 'test' subdirectories

def main():
    if not os.path.exists(DATA_DIR):
        print(f"Error: Dataset directory '{DATA_DIR}' not found.")
        print("Please create it and structure it like:")
        print("dataset/")
        print("├── train/")
        print("│   ├── angry/")
        print("│   ├── happy/")
        print("│   ├── neutral/")
        print("│   └── sad/")
        print("└── test/")
        print("    ├── ...")
        return

    # Data Augmentation to improve model performance
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=15,
        width_shift_range=0.15,
        height_shift_range=0.15,
        shear_range=0.15,
        zoom_range=0.15,
        horizontal_flip=True,
        fill_mode='nearest'
    )

    test_datagen = ImageDataGenerator(rescale=1./255)

    print("Loading training data...")
    train_generator = train_datagen.flow_from_directory(
        os.path.join(DATA_DIR, 'train'),
        target_size=(IMG_HEIGHT, IMG_WIDTH),
        color_mode="grayscale",
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=True
    )

    print("Loading validation data...")
    validation_generator = test_datagen.flow_from_directory(
        os.path.join(DATA_DIR, 'test'),
        target_size=(IMG_HEIGHT, IMG_WIDTH),
        color_mode="grayscale",
        batch_size=BATCH_SIZE,
        class_mode='categorical'
    )

    # Initialize model
    model = create_emotion_model(input_shape=(IMG_HEIGHT, IMG_WIDTH, 1), num_classes=NUM_CLASSES)

    # Callbacks for hyperparameter tuning and saving the best model
    checkpoint = ModelCheckpoint('emotion_model.h5', monitor='val_accuracy', save_best_only=True, mode='max', verbose=1)
    early_stopping = EarlyStopping(monitor='val_loss', patience=12, restore_best_weights=True, verbose=1)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.0001, verbose=1)

    print("Starting training...")
    # Train model
    history = model.fit(
        train_generator,
        steps_per_epoch=train_generator.n // train_generator.batch_size,
        epochs=EPOCHS,
        validation_data=validation_generator,
        validation_steps=validation_generator.n // validation_generator.batch_size,
        callbacks=[checkpoint, early_stopping, reduce_lr]
    )
    
    print("Training complete. Best model saved as 'emotion_model.h5'.")

if __name__ == "__main__":
    main()
