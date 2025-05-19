# ====== 1. Import Required Libraries ======
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier

from imblearn.over_sampling import SMOTE

# ====== 2. Define File Paths ======
original_data_path = 'Data/original_data/network_traffic_data.csv'
preprocessed_dir = 'Data/preprocessed_data'
results_dir = 'Data/Results'
visualizations_dir = 'Data/visualizations'

# Create directories if they do not exist
os.makedirs(preprocessed_dir, exist_ok=True)
os.makedirs(results_dir, exist_ok=True)
os.makedirs(visualizations_dir, exist_ok=True)

# ====== 3. Load and Shuffle Original Data ======
data = pd.read_csv(original_data_path)
data = data.sample(frac=1, random_state=42).reset_index(drop=True)  # Shuffle data

# ====== 4. Exploratory Data Analysis (EDA) - Visualizations ======

# 4.1 Class Distribution
plt.figure(figsize=(6, 4))
sns.countplot(x='Label', data=data)
plt.title('Class Distribution')
plt.savefig(os.path.join(visualizations_dir, 'class_distribution.png'))
plt.close()

# 4.2 Correlation Heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(data.select_dtypes(include='number').corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.savefig(os.path.join(visualizations_dir, 'correlation_heatmap.png'))
plt.close()

# 4.3 Boxplot of Numeric Features
plt.figure(figsize=(10, 6))
sns.boxplot(data=data[['Duration', 'PacketCount', 'ByteCount']])
plt.title('Boxplot of Numeric Features')
plt.savefig(os.path.join(visualizations_dir, 'boxplot_numeric_features.png'))
plt.close()

# 4.4 Pairplot for Selected Features
sns.pairplot(data[['Duration', 'PacketCount', 'ByteCount', 'Label']], hue='Label')
plt.savefig(os.path.join(visualizations_dir, 'pairplot.png'))
plt.close()

# ====== 5. Feature and Target Separation ======
X = data.drop('Label', axis=1)
y = data['Label']

# ====== 6. Encoding Categorical Features ======
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

categorical_cols = X.select_dtypes(include=['object']).columns
X_encoded = pd.get_dummies(X, columns=categorical_cols)

# ====== 7. Split Dataset into Training and Testing Sets ======
X_train, X_test, y_train, y_test = train_test_split(
    X_encoded, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

# Save preprocessed data
pd.DataFrame(X_train).to_csv(os.path.join(preprocessed_dir, 'X_train.csv'), index=False)
pd.DataFrame(X_test).to_csv(os.path.join(preprocessed_dir, 'X_test.csv'), index=False)
pd.DataFrame(y_train, columns=['Label']).to_csv(os.path.join(preprocessed_dir, 'y_train.csv'), index=False)
pd.DataFrame(y_test, columns=['Label']).to_csv(os.path.join(preprocessed_dir, 'y_test.csv'), index=False)

# ====== 8. Apply SMOTE for Class Balancing ======
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# ====== 9. Feature Scaling ======
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_resampled)
X_test_scaled = scaler.transform(X_test)

# ====== 10. Define Machine Learning Models ======
models = {
    'SVM': SVC(),
    'Naive_Bayes': GaussianNB(),
    'KNN': KNeighborsClassifier(),
    'Random_Forest': RandomForestClassifier(random_state=42),
    'Decision_Tree': DecisionTreeClassifier(random_state=42),
    'ANN': MLPClassifier(random_state=42, max_iter=500)
}

# ====== 11. Train and Evaluate Each Model ======
results_summary = []

for name, model in models.items():
    print(f"\nTraining {name}...")
    model.fit(X_train_scaled, y_train_resampled)
    preds = model.predict(X_test_scaled)

    # Save predictions
    pred_df = pd.DataFrame({'Actual': y_test, 'Predicted': preds})
    filename = f'predictions_{name.replace(" ", "_").lower()}_model.csv'
    pred_df.to_csv(os.path.join(results_dir, filename), index=False)

    # Calculate accuracy
    acc = accuracy_score(y_test, preds)
    results_summary.append({'Model': name, 'Accuracy': acc})

    # Print classification report
    print(f"Accuracy for {name}: {acc:.4f}")
    print(classification_report(y_test, preds))

    # Save confusion matrix
    cm = confusion_matrix(y_test, preds)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f"{name} - Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, f'{name.replace(" ", "_").lower()}_confusion_matrix.png'))
    plt.close()

# ====== 12. Save Model Accuracy Summary ======
results_df = pd.DataFrame(results_summary)
results_df.to_csv(os.path.join(results_dir, 'model_performance_summary.csv'), index=False)

# ====== 13. Visualize Model Accuracy Comparison ======
plt.figure(figsize=(10, 6))
sns.barplot(x='Model', y='Accuracy', data=results_df, palette='viridis')
plt.title('Model Accuracy Comparison')
plt.ylim(0, 1)
plt.xticks(rotation=45)
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(results_dir, 'model_accuracy_comparison.png'))
plt.show()

