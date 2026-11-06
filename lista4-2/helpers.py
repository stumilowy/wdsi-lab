import pandas as pd
import seaborn as sns
import numpy as np
from mlxtend.frequent_patterns import apriori, association_rules
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
import matplotlib.pyplot as plt


def load_trams():
    return pd.read_csv("Symulowane_tramwaje.csv")


def prepare_trams_exp(trams_df):
    # Cechy (X) - wszystko oprócz predykowanej klasy
    X = trams_df.drop('Opóźniony', axis=1)
    # Cel (y) - tylko predykowana klasa
    y = trams_df['Opóźniony']
    # 'sparse_output=False' ułatwi nam odczytanie nazw kolumn
    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    X_transformed = encoder.fit_transform(X)
    # Nazwy cech do drzewa
    feature_names = encoder.get_feature_names_out(X.columns)
    return (X_transformed, feature_names), y


def preprocess_trams(trams):
    dummy_line = pd.get_dummies(trams["Linia"])
    dummy_line.columns = [f"Linia {c}" for c in dummy_line.columns]
    # Szczyt poranny i popołudniowy łączymy w jedną kategorię "Szczyt"
    pora = trams["Pora_Dnia"].replace({"Szczyt_Poranny": "Szczyt", "Szczyt_Popołudniowy": "Szczyt"})
    dummy_daytime = pd.get_dummies(pora)
    dummy_rest = trams.drop(columns=["Linia", "Pora_Dnia"])
    dummy_rest = dummy_rest == "Tak"
    return pd.concat([dummy_line, dummy_daytime, dummy_rest], axis=1)


def plot_tree_wrapper(model, feature_names, class_names):
    """
    Funkcja wizualizująca drzewo decyzyjne.
    """
    plt.figure(figsize=(20, 12))
    plot_tree(model, 
                feature_names=feature_names, 
                class_names=class_names,  
                filled=True, 
                rounded=True,
                fontsize=10)
    plt.title("Drzewo Decyzyjne", fontsize=16)
    plt.show()

    print("Tekstowa reprezentacja drzewa decyzyjnego (Dostępna)")
    tree_ = model.tree_
    
    # Zmieniliśmy parametr path na condition_from_parent, co lepiej oddaje logikę
    def recurse(node, depth, condition_from_parent="[Start] Korzeń drzewa"):
        indent = "    " * depth
        
        # Obliczanie statystyk
        gini = tree_.impurity[node]
        samples = tree_.n_node_samples[node]
        raw_value = tree_.value[node][0]
        pred_class = class_names[np.argmax(raw_value)]
        
        total_value = np.sum(raw_value)
        if total_value > 0:
            expected_counts = (raw_value / total_value) * samples
            counts = np.floor(expected_counts).astype(int)
            remainder = int(samples - np.sum(counts))
            
            if remainder > 0:
                fractional_parts = expected_counts - counts
                indices = np.argsort(fractional_parts)[::-1] 
                for i in range(remainder):
                    counts[indices[i]] += 1
            formatted_value = counts.tolist()
        else:
            formatted_value = [0] * len(raw_value)
        
        info = f"Gini: {gini:.3f}, Próbek: {samples}, Rozkład: {formatted_value}, Klasa: {pred_class}"
        
        # Przechodzenie i wypisywanie struktury
        if tree_.children_left[node] == tree_.children_right[node]:
            # Węzeł jest liściem
            print(f"{indent}- [LIŚĆ] {condition_from_parent}")
            print(f"{indent}  Statystyki: {info}")
        else:
            # Węzeł wewnętrzny
            feature_name = feature_names[tree_.feature[node]]
            threshold = tree_.threshold[node]
            
            # Najpierw wypisujemy informacje o aktualnym węźle
            print(f"{indent}- [WĘZEŁ] {condition_from_parent}")
            print(f"{indent}  Statystyki: {info}")
            
            # Od razu wchodzimy w lewą gałąź (warunek spełniony)
            left_condition = f"Jeśli {feature_name} <= {threshold:.3f}"
            recurse(tree_.children_left[node], depth + 1, left_condition)
            
            # Następnie wchodzimy w prawą gałąź (warunek niespełniony)
            right_condition = f"Jeśli {feature_name} > {threshold:.3f}"
            recurse(tree_.children_right[node], depth + 1, right_condition)

    # Uruchamiamy od węzła zerowego (korzenia)
    recurse(0, 0)


def load_uci():
    # Wczytujemy dane
    return pd.read_csv("student-mat.csv", sep=";")


def process_uci(df):
    # Dyskretyzacja
    # Oceny są w skali 0-20. Podzielimy je na trzy kategorie:
    # 0-9: Fail (Niezaliczone), 10-15: Pass (Zaliczone), 16-20: Excellent (Świetnie)
    bins = [-1, 9, 15, 20]
    labels = ['Fail', 'Pass', 'Excellent']

    for col in ['G1', 'G2', 'G3']:
        df[col] = pd.cut(df[col], bins=bins, labels=labels)

    # Inne zmienne numeryczne (np. wiek, nieobecności, oceny jakościowe 1-5) 
    # rzutujemy na tekst. Dzięki temu np. wiek 15 stanie się kategorią "15".
    # (Możesz też stworzyć koszyki dla nieobecności, jeśli zależy Ci na lepszych regułach).
    df = df.astype(str)

    # One-Hot Encoding (Kodowanie 0/1)
    # zmiana kolumny np. 'sex' z wartościami 'F'/'M' na dwie kolumny: 'sex_F' i 'sex_M' z wartościami True/False.
    df_encoded = pd.get_dummies(df)

    # mlxtend wymaga wartości logicznych (boolean)
    df_encoded = df_encoded.astype(bool)
    return df_encoded


def prepare_exp_uci(df):
    """
    Funkcja dyskretyzuje oceny i przygotowuje zbiór UCI Student Performance.
    Zwraca przetworzone cechy (X), zmienną celu (y), nazwy cech oraz nazwy klas,
    gotowe do użycia w scikit-learn i funkcjach wizualizujących.
    """
    bins = [-1, 9, 15, 20]
    labels = ['Fail', 'Pass', 'Excellent']

    for col in ['G1', 'G2', 'G3']:
        df[col] = pd.cut(df[col], bins=bins, labels=labels)

    y = df['G3']
    X = df.drop(columns=['G3'])

    # Kodowanie zmiennych kategorialnych
    X_encoded = pd.get_dummies(X, drop_first=True)

    # Wyciągnięcie nazw cech w formie listy stringów
    feature_names = X_encoded.columns.tolist()
    
    # Nazwy klas to nasze zdefiniowane etykiety
    class_names = labels
    return X_encoded, y, feature_names, class_names