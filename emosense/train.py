"""Train a CNN for facial emotion recognition on FER-2013.

Expected folder layout (the Kaggle FER-2013 image version):
    fer2013/train/<emotion>/*.png
    fer2013/test/<emotion>/*.png
"""
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks

DATA_DIR = "fer2013"
IMG_SIZE = (48, 48)
BATCH = 64

train_ds = tf.keras.utils.image_dataset_from_directory(
    f"{DATA_DIR}/train", color_mode="grayscale", image_size=IMG_SIZE,
    batch_size=BATCH, label_mode="categorical", seed=42)
test_ds = tf.keras.utils.image_dataset_from_directory(
    f"{DATA_DIR}/test", color_mode="grayscale", image_size=IMG_SIZE,
    batch_size=BATCH, label_mode="categorical", shuffle=False)

class_names = train_ds.class_names
print("Classes:", class_names)
with open("labels.txt", "w") as f:
    f.write("\n".join(class_names))

augment = models.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.05),
    layers.RandomZoom(0.1),
])

model = models.Sequential([layers.Input((48, 48, 1)), augment, layers.Rescaling(1 / 255)])
for filters in (32, 64, 128, 256):
    model.add(layers.Conv2D(filters, 3, padding="same", activation="relu"))
    model.add(layers.BatchNormalization())
    model.add(layers.Conv2D(filters, 3, padding="same", activation="relu"))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D())
    model.add(layers.Dropout(0.25))
model.add(layers.GlobalAveragePooling2D())
model.add(layers.Dense(256, activation="relu"))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(len(class_names), activation="softmax"))

model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
              loss="categorical_crossentropy", metrics=["accuracy"])
model.summary()

model.fit(
    train_ds, validation_data=test_ds, epochs=50,
    callbacks=[
        callbacks.ModelCheckpoint("emotion_model.keras", save_best_only=True,
                                  monitor="val_accuracy"),
        callbacks.ReduceLROnPlateau(patience=4, factor=0.5, verbose=1),
        callbacks.EarlyStopping(patience=10, restore_best_weights=True),
    ],
)

loss, acc = model.evaluate(test_ds)
print(f"Test accuracy: {acc:.3f}")
