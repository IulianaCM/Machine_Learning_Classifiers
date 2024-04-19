import cv2
import os
import numpy as np
from sklearn import svm
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt

# Directorul cu imaginile
albine_folder_path = "Dataset/Albine"
fluturi_folder_path = "Dataset/Fluturi"
test_folder_path = "Test"

# Funcție pentru citirea imaginilor și extragerea caracteristicilor
def extract_features(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    image = cv2.resize(image, (50, 50))
    
    # Aplicăm Gaussian Blur pentru reducerea zgomotului
    blurred_image = cv2.GaussianBlur(image, (5, 5), 0)
    
    # Aplicăm edge detection cu Canny
    edges = cv2.Canny(blurred_image, 10, 100)
    
    # Calculăm moments pentru caracteristicile conturului
    moments = cv2.moments(edges)

    # Returnăm atât caracteristicile, cât și imaginea prelucrată
    return [moments['m00'], moments['m10'], moments['m01'], moments['m20'], moments['m11'], moments['m02']], edges

# Încărcarea datelor și etichetarea lor
data = []
labels = []

# Procesăm imaginile cu albine
for filename in os.listdir(albine_folder_path):
    file_path = os.path.join(albine_folder_path, filename)
    if os.path.isfile(file_path) and (filename.endswith(".png") or filename.endswith(".jpg")):
        features, edges1 = extract_features(file_path)

        if features is not None:
            data.append(features)
            labels.append(0)  # 0 pentru albine

# Procesăm imaginile cu fluturi
for filename in os.listdir(fluturi_folder_path):
    file_path = os.path.join(fluturi_folder_path, filename)
    if os.path.isfile(file_path) and (filename.endswith(".png") or filename.endswith(".jpg")):
        features, edges2 = extract_features(file_path)

        if features is not None:
            data.append(features)
            labels.append(1)  # 1 pentru fluturi

# Transformarea listelor în numpy arrays
X = np.array(data)
y = np.array(labels)

# Împărțirea datelor în set de antrenare și set de testare
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Antrenarea clasificatorului SVM
clf = svm.SVC(kernel='rbf')
clf.fit(X_train, y_train)

# Predictie pe setul de test
y_pred = clf.predict(X_test)

# Calcul matrice de confuzie
cm = confusion_matrix(y_test, y_pred)

# Vizualizare matrice de confuzie utilizând seaborn
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, 
            xticklabels=clf.classes_, yticklabels=clf.classes_)

plt.title('Matrice de confuzie SVM')
plt.xlabel('Valori reale')
plt.ylabel('Valori prezise')
plt.show()

# Calculul acurateții
accuracy = accuracy_score(y_test, y_pred)
print(f'Acuratețea clasificatorului SVM: {accuracy * 100:.2f}%')

report = classification_report(y_test, y_pred)
print("Raport de clasificare SVM:\n", report)

fpr, tpr, thresholds = roc_curve(y_test, y_pred)
roc_auc = auc(fpr, tpr)

# Curba ROC
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'Curba ROC (AUC = {roc_auc:.2f})')
#plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('Specificitatea')
plt.ylabel('Sensibilitatea')
plt.title('Curba ROC (Receiver Operating Characteristic)')
plt.legend(loc="lower right")
plt.show()

# Testare și vizualizare pentru SVM
test_image_paths = [os.path.join(test_folder_path, filename) for filename in os.listdir(test_folder_path) if os.path.isfile(os.path.join(test_folder_path, filename))]
for i, test_image_path in enumerate(test_image_paths):
    test_features, test_edges = extract_features(test_image_path)
    test_features = np.array(test_features).reshape(1, -1)  # Reshape for a single sample
    test_prediction = clf.predict(test_features)  # Predict classes
    
    # Vizualizare imagine originală de test cu clasa prezisă
    plt.figure(figsize=(10, 4))
    
    plt.subplot(1, 3, 1)
    test_image = cv2.imread(test_image_path)
    plt.imshow(cv2.cvtColor(test_image, cv2.COLOR_BGR2RGB))
    if test_prediction[0] == 0:
        plt.title(f"Test Image {i + 1} - ==ALBINE==")
    else:
        plt.title(f"Test Image {i + 1} - ==FLUTURI==")
    plt.axis('off')

    # Vizualizare imagine prelucrată cu Canny Edge Detection
    plt.subplot(1, 3, 2)
    plt.imshow(test_edges, cmap='gray')
    plt.title("Canny Edge Detection")
    plt.axis('off')
    
    plt.show()

    # Afișare rezultat clasificare
    if test_prediction[0] == 0:
        print(f"Test Image {i + 1} - ==ALBINE==")
    else:
        print(f"Test Image {i + 1} - ==FLUTURI==")