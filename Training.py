import os
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from tensorflow.keras.preprocessing import image_dataset_from_directory
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve, auc, confusion_matrix, ConfusionMatrixDisplay

# ----------------------------------------------------------
# 1. Configuration initiale
# ----------------------------------------------------------
print("Using TensorFlow version:", tf.__version__)
tf.random.set_seed(42)  # Pour la reproductibilité

# Chemins vers les dossiers "good" et "bad"
GOOD_DIR = r"C:\Users\Abdelhamid\Desktop\Code-vis-apprentissage\soumaya777777-Solution\Code\Screw-main\screw_dataset-images\screw_dataset-images\screw_dataset\good"
BAD_DIR = r"C:\Users\Abdelhamid\Desktop\Code-vis-apprentissage\soumaya777777-Solution\Code\Screw-main\screw_dataset-images\screw_dataset-images\screw_dataset\bad"

# Vérification de l'existence des dossiers
if not os.path.exists(GOOD_DIR) or not os.path.exists(BAD_DIR):
    raise FileNotFoundError("Les dossiers 'good' ou 'bad' n'existent pas.")

# Paramètres
BATCH_SIZE = 32
IMG_SIZE = (224, 224)
VAL_SPLIT = 0.2  # 20% des données pour la validation
EPOCHS = 100

# ----------------------------------------------------------
# 2. Chargement et préparation du dataset
# ----------------------------------------------------------
# Chargement des datasets
train_dataset = image_dataset_from_directory(
    os.path.dirname(GOOD_DIR),  # Utilisation du dossier parent
    validation_split=VAL_SPLIT,
    subset="training",
    label_mode="categorical",
    batch_size=BATCH_SIZE,
    image_size=IMG_SIZE,
    shuffle=True,
    seed=42
)

val_dataset = image_dataset_from_directory(
    os.path.dirname(GOOD_DIR),  # Utilisation du dossier parent
    validation_split=VAL_SPLIT,
    subset="validation",
    label_mode="categorical",
    batch_size=BATCH_SIZE,
    image_size=IMG_SIZE,
    shuffle=True,
    seed=42
)

# Vérification des classes
class_names = train_dataset.class_names
print("\nClasses détectées :", class_names)

# Vérification de la répartition des classes
good_count = len(os.listdir(GOOD_DIR))
bad_count = len(os.listdir(BAD_DIR))

print(f"\nNombre d'images par classe :")
print(f"- Good (conforme) : {good_count}")
print(f"- Bad (non conforme) : {bad_count}")

if good_count == 0 or bad_count == 0:
    raise ValueError("Une des classes est vide. Vérifiez le dataset.")

# ----------------------------------------------------------
# 2bis. Data Augmentation (Nouvelle section)
# ----------------------------------------------------------
# Définition de la data augmentation
data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
    layers.RandomContrast(0.2),
])

# Fonction de prétraitement modifiée
def preprocess_train(image, label):
    # Appliquer l'augmentation seulement pendant l'entraînement
    image = data_augmentation(image)
    image = tf.cast(image, tf.float32) / 255.0  # Normalisation
    return image, label

def preprocess_val(image, label):
    # Pour le dataset de validation, on applique seulement la normalisation
    image = tf.cast(image, tf.float32) / 255.0
    return image, label

# Préprocessing et optimisation
AUTOTUNE = tf.data.AUTOTUNE

# Appliquer l'augmentation seulement au dataset d'entraînement
train_dataset = train_dataset.map(preprocess_train, num_parallel_calls=AUTOTUNE).cache().prefetch(AUTOTUNE)
val_dataset = val_dataset.map(preprocess_val, num_parallel_calls=AUTOTUNE).cache().prefetch(AUTOTUNE)

# ----------------------------------------------------------
# 3. Définition du modèle amélioré
# ----------------------------------------------------------
num_classes = len(class_names)

model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3)),
    layers.MaxPooling2D(2, 2),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D(2, 2),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D(2, 2),
    layers.Conv2D(256, (3, 3), activation='relu'),
    layers.MaxPooling2D(2, 2),
    layers.Flatten(),
    layers.Dense(512, activation='relu'),
    layers.Dropout(0.5),  # Réduit le surapprentissage
    layers.Dense(num_classes, activation='softmax')
])

# Compilation avec les paramètres corrects
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),  # Taux d'apprentissage réduit
    loss='categorical_crossentropy',
    metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
)

model.summary()

# ----------------------------------------------------------
# 4. Entraînement avec gestion du surapprentissage
# ----------------------------------------------------------
early_stop = callbacks.EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True
)

reduce_lr = callbacks.ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.2,
    patience=5,
    min_lr=0.00001
)

# Entraînement
history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=EPOCHS,
    callbacks=[early_stop, reduce_lr],
    verbose=1
)

# ----------------------------------------------------------
# 5. Visualisation des résultats
# ----------------------------------------------------------
# Courbes d'apprentissage
plt.figure(figsize=(12, 5))

# Courbe de précision
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.title('Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

# Courbe de perte
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.show()

# ----------------------------------------------------------
# 6. Évaluation finale
# ----------------------------------------------------------
print("\nÉvaluation sur le dataset de validation :")
val_loss, val_acc, val_precision, val_recall = model.evaluate(val_dataset)
print(f"Validation Accuracy: {val_acc:.2%}")
print(f"Validation Precision: {val_precision:.2%}")
print(f"Validation Recall: {val_recall:.2%}")

# Sauvegarde du modèle
model.save("mon_cnn_amelioré.h5")
print("\nModèle sauvegardé sous 'mon_cnn_ameliorévrai.h5'")

# ----------------------------------------------------------
# 7. Visualisations supplémentaires
# ----------------------------------------------------------
# Récupération des prédictions et vraies étiquettes
y_true = []
y_pred_probs = []

for images, labels in val_dataset:
    y_true.extend(labels.numpy())
    y_pred_probs.extend(model.predict(images, verbose=0))

y_true = np.argmax(y_true, axis=1)
y_pred_probs = np.array(y_pred_probs)[:, 1]  # Probabilités pour la classe positive (bad)

# Courbe ROC
fpr, tpr, thresholds = roc_curve(y_true, y_pred_probs)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(10, 5))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'Courbe ROC (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('Taux de Faux Positifs')
plt.ylabel('Taux de Vrais Positifs')
plt.title('Courbe ROC')
plt.legend(loc="lower right")
plt.show()

# Matrice de confusion
y_pred = np.argmax(y_pred_probs.reshape(-1, 1), axis=1)  # Conversion des probabilités en classes

cm = confusion_matrix(y_true, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
disp.plot(cmap=plt.cm.Blues)
plt.title('Matrice de Confusion')
plt.show()
