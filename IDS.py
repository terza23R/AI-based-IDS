import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from scipy.io import arff
from tabulate import tabulate

# Funkcija za popravku ARFF datoteka
def fix_arff_format_v2(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    corrected_lines = [line.replace(" 'icmp'", 'icmp').replace(" 'tcp'", 'tcp').replace(" 'udp'", 'udp') 
                       for line in lines]

    corrected_file_path = file_path.replace('.arff', '_corrected_v2.arff')
    with open(corrected_file_path, 'w') as corrected_file:
        corrected_file.writelines(corrected_lines)
        
    return corrected_file_path

# Putanje do ARFF fajlova
train_file_path = 'KDDTrain+.arff'
test_file_path = 'KDDTest+.arff'

# Popravljanje ARFF formata
corrected_train_file_path_v2 = fix_arff_format_v2(train_file_path)
corrected_test_file_path_v2 = fix_arff_format_v2(test_file_path)

# Učitavanje ARFF fajlova
train_data_corrected_v2, _ = arff.loadarff(corrected_train_file_path_v2)
test_data_corrected_v2, _ = arff.loadarff(corrected_test_file_path_v2)

# Pretvaranje u pandas dataframe
df_train_corrected_v2 = pd.DataFrame(train_data_corrected_v2)
df_test_corrected_v2 = pd.DataFrame(test_data_corrected_v2)

# Konvertovanje kategorijskih podataka u numeričke vrijednosti
label_encoder = LabelEncoder()

for column in df_train_corrected_v2.columns:
    if df_train_corrected_v2[column].dtype == object:
        df_train_corrected_v2[column] = label_encoder.fit_transform(df_train_corrected_v2[column])

for column in df_test_corrected_v2.columns:
    if df_test_corrected_v2[column].dtype == object:
        df_test_corrected_v2[column] = label_encoder.fit_transform(df_test_corrected_v2[column])

# Odvajanje karakteristika i oznaka
X_train = df_train_corrected_v2.drop(columns=['class']).values
y_train = df_train_corrected_v2['class'].values

X_test = df_test_corrected_v2.drop(columns=['class']).values
y_test = df_test_corrected_v2['class'].values

# Definisanje modela
models = {
    'Decision Tree': DecisionTreeClassifier(),
    'KNN': KNeighborsClassifier(),
    'Random Forest': RandomForestClassifier(),
    'Logistic Regression': LogisticRegression(max_iter=1000)
}

# Inicijalizacija rezultata
results = []
confusion_matrices = {}

# Treniranje i evaluacija svakog modela
for model_name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    
    # Čuvanje tačnosti, f1-score, preciznosti i odziva (recall) za svaki model
    avg_f1_score = report['weighted avg']['f1-score']
    avg_precision = report['weighted avg']['precision']
    avg_recall = report['weighted avg']['recall']
    
    # Čuvanje rezultata
    results.append([model_name, accuracy * 100, avg_precision * 100, avg_recall * 100, avg_f1_score * 100])
    
    # Čuvanje matrice konfuzije
    confusion_matrices[model_name] = confusion_matrix(y_test, y_pred)

# Kreiranje tabele sa rezultatima
table = tabulate(results, headers=["Algoritam", "Tačnost (%)", "Preciznost (%)", "Odziv (%)", "F1-score (%)"], tablefmt="grid")
print("\nUporedna tabela metrika modela:")
print(table)

# Pretvaranje rezultata u dataframe za lakšu manipulaciju
results_df = pd.DataFrame(results, columns=["Algoritam", "Tačnost (%)", "Preciznost (%)", "Odziv (%)", "F1-score (%)"])



# Grouped Bar Plot za poređenje Preciznosti, Odziva i F1-score-a
metrics = ['Preciznost (%)', 'Odziv (%)', 'F1-score (%)']

# Plotovanje grouped bar plot sa prilagođenom y-osom (65-95)
ax = results_df.set_index('Algoritam')[metrics].plot(kind='bar', figsize=(10, 6))
plt.title("Poređenje metrika (Preciznost, Odziv, F1-score) za različite algoritme")
plt.xlabel("Algoritam")
plt.ylabel("Vrijednost (%)")
plt.legend(title="Metričke vrijednosti")
plt.ylim(65,90)  # Ograničavanje y-osi na interval od 65 do 95%
plt.grid(True)
plt.xticks(rotation=45)
plt.show()


# Horizontalni Bar Plot za tačnost
plt.figure(figsize=(10, 6))
sns.barplot(x='Tačnost (%)', y='Algoritam', data=results_df, palette='viridis')
plt.title("Poređenje tačnosti između algoritama")
plt.xlabel("Tačnost (%)")
plt.ylabel("Algoritam")
plt.xlim(65,90)
plt.grid(True)
plt.show()

# Boxplot za raspodjelu Preciznosti, Odziva i F1-score-a
plt.figure(figsize=(10, 6))
sns.boxplot(data=results_df[metrics])
plt.title("Boxplot raspodjele preciznosti, odziva i F1-score-a")
plt.ylabel("Vrijednost (%)")
plt.grid(True)
plt.show()

# Plot matrica konfuzije
fig, axs = plt.subplots(2, 2, figsize=(12, 12))
axs = axs.flatten()

for i, (model_name, cm) in enumerate(confusion_matrices.items()):
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=axs[i])
    axs[i].set_title(f"Matrica konfuzije - {model_name}")
    axs[i].set_xlabel("Predviđene klase")
    axs[i].set_ylabel("Prave klase")

plt.tight_layout()
plt.show()
