import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# --- STEP 1: LOAD THE DATA ---
print("Downloading MNIST dataset...")
mnist = tf.keras.datasets.mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# --- STEP 2: PREPROCESS THE DATA ---
# Normalize pixel values to be between 0 and 1
train_images = train_images / 255.0
test_images = test_images / 255.0

# --- STEP 3: BUILD THE CNN ARCHITECTURE ---
model = tf.keras.models.Sequential([
    tf.keras.layers.Reshape((28, 28, 1), input_shape=(28, 28)),
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax') 
])

# --- STEP 4: COMPILE THE MODEL ---
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# --- STEP 5: TRAIN THE MODEL ---
print("\nStarting training phase (this will take a minute)...")
model.fit(train_images, train_labels, epochs=5)

# --- STEP 6: EVALUATE PERFORMANCE ---
print("\nTesting the AI on unseen images...")
test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
print(f'\nFinal Test Accuracy: {test_acc * 100:.2f}%')

# --- STEP 7: VISUALLY TEST A SPECIFIC IMAGE ---
print("\nTesting the AI with a specific image from the dataset...")
# Change this number (between 0 and 9999) to test different handwritten digits!
image_index = 0 

test_image = test_images[image_index]
true_label = test_labels[image_index]

# The AI expects a batch of images, so we wrap our single image in an array
image_batch = np.expand_dims(test_image, axis=0)

# Get the AI's prediction
predictions = model.predict(image_batch)
predicted_label = np.argmax(predictions[0])

print(f"AI Predicts: {predicted_label} | Actual Label: {true_label}")

# Create the visual result
plt.imshow(test_image, cmap='gray')
plt.title(f"AI Predicted: {predicted_label} | Actual: {true_label}")
plt.axis('off')

# Save the result as an image file in your folder instead of relying on a pop-up
plt.savefig("prediction_result.png")
print("\nSuccess! Open the 'prediction_result.png' file in VS Code to see the image.")