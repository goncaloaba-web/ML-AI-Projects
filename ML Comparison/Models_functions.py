from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier 
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
import random
import numpy as np
import time
import matplotlib.pyplot as plt
import seaborn as sns
from memory_profiler import memory_usage
import gc
from sklearn.metrics import accuracy_score

def SVM_model(train_X, train_y, test_X, test_y):

    scaler = StandardScaler()
    train_X_scaled = scaler.fit_transform(train_X)
    test_X_scaled = scaler.transform(test_X)

    SVC_model = SVC(kernel='rbf', gamma='scale', C=1.0)

    def train_model():
        SVC_model.fit(train_X_scaled, train_y)
        return SVC_model

    gc.collect()
    mem_usage = memory_usage((train_model,), interval=0.05)
    memory_used = max(mem_usage) - min(mem_usage)

    start_train = time.time()
    SVC_model.fit(train_X_scaled, train_y)
    train_time = time.time() - start_train

    SVC_predictions = SVC_model.predict(test_X_scaled)

    return SVC_predictions, train_time, memory_used


def RF_model(train_X, train_y, test_X, test_y):

    RF_clf = RandomForestClassifier(n_estimators=100, random_state=42)

    def train_model():
        RF_clf.fit(train_X, train_y)
        return RF_clf

    gc.collect()
    mem_usage = memory_usage((train_model,), interval=0.05)
    memory_used = max(mem_usage) - min(mem_usage)

    start_train = time.time()
    RF_clf.fit(train_X, train_y)
    train_time = time.time() - start_train

    RF_predictions = RF_clf.predict(test_X)

    return RF_predictions, train_time, memory_used


def XGB_model(train_X, train_y, test_X, test_y):

    XGB_clf = XGBClassifier(n_estimators=100, random_state=42, eval_metric='mlogloss')

    def train_model():
        XGB_clf.fit(train_X, train_y)
        return XGB_clf

    gc.collect()
    mem_usage = memory_usage((train_model,), interval=0.05)
    memory_used = max(mem_usage) - min(mem_usage)

    start_train = time.time()
    XGB_clf.fit(train_X, train_y)
    train_time = time.time() - start_train

    XGB_predictions = XGB_clf.predict(test_X)

    return XGB_predictions, train_time, memory_used


def DL_model(train_X, train_y, test_X, test_y):

    scaler = StandardScaler()
    train_X_scaled = scaler.fit_transform(train_X)
    test_X_scaled = scaler.transform(test_X)

    random.seed(42)
    np.random.seed(42)
    tf.random.set_seed(42)

    model = Sequential([
        Dense(128, activation='relu', input_shape=(train_X.shape[1],)),
        Dropout(0.3, seed=42),
        Dense(64, activation='relu'),
        Dropout(0.3, seed=42),
        Dense(32, activation='relu'),
        Dense(7, activation='softmax')
    ])

    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    def train_model():
        model.fit(
            train_X_scaled, train_y,
            epochs=50,
            batch_size=32,
            verbose=0
        )
        return model

    gc.collect()
    mem_usage = memory_usage((train_model,), interval=0.05)
    memory_used = max(mem_usage) - min(mem_usage)

    start_train = time.time()
    model.fit(
        train_X_scaled, train_y,
        epochs=50,
        batch_size=32,
        verbose=0
    )
    train_time = time.time() - start_train

    DL_predictions = model.predict(test_X_scaled).argmax(axis=1)

    return DL_predictions, train_time, memory_used





#                                   Mostrar Mtrizes etc...


def mostrar_relatorio_completo(nome_dataset, y_true, modelos):
    
    classes = [
    'Insufficient_Weight', 'Normal_Weight', 'Overweight_Level_I',
    'Overweight_Level_II', 'Obesity_Type_I', 'Obesity_Type_II', 'Obesity_Type_III'
    ]

    for nome, y_pred, _, _ in modelos:
        matriz = confusion_matrix(y_true, y_pred)
        report = classification_report(y_true, y_pred, output_dict=True)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        sns.heatmap(matriz, annot=True, fmt='d', cmap='Blues',
                    xticklabels=classes, yticklabels=classes, ax=ax1)
        ax1.set_title('Matriz de Confusão')
        ax1.set_xlabel('Previsto')
        ax1.set_ylabel('Real')
        plt.setp(ax1.get_xticklabels(), rotation=45, ha='right')

        metricas = [
            ['Accuracy',     f"{report['accuracy']:.2%}",                        '-',                                           '-',                                         '-'],
            ['Macro avg',    f"{report['macro avg']['precision']:.2%}",          f"{report['macro avg']['recall']:.2%}",        f"{report['macro avg']['f1-score']:.2%}",    '-'],
            ['Weighted avg', f"{report['weighted avg']['precision']:.2%}",       f"{report['weighted avg']['recall']:.2%}",     f"{report['weighted avg']['f1-score']:.2%}", f"{int(report['weighted avg']['support'])}"],
        ]

        ax2.axis('off')
        tabela = ax2.table(
            cellText=metricas,
            colLabels=['', 'Precision', 'Recall', 'F1-Score', 'Support'],
            loc='center',
            cellLoc='center'
        )
        tabela.scale(1, 2)
        ax2.set_title('Métricas Globais')

        fig.suptitle(
            f'{nome} — {nome_dataset} | Accuracy: {report["accuracy"]:.2%}',
            fontsize=14
        )
        plt.tight_layout()
        plt.show()


#                                   Devolver Accuracy
def return_accuracy (preds, trues):
    accuracy = []

    for i in range(5):
        accuracy.append(accuracy_score(trues[i], preds[i]))
    
    return accuracy

#                                   Mostrar Accuracy Gráfico

def mostrar_accuracy_graficos(modelos, accuracy):

    X = [20, 40, 60, 80, 100]
    Y = accuracy      

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(X, Y, marker='o', linewidth=2, markersize=7)

    for x, y in zip(X, Y):
        ax.annotate(f'{y:.3f}',
                    (x, y),
                    textcoords="offset points",
                    xytext=(0, 10),
                    ha='center')

    ax.set_xlabel('Tamanho do Dataset (%)')
    ax.set_ylabel('Accuracy')
    ax.set_title(f'Accuracy vs Tamanho do Dataset — {modelos}')
    ax.set_xticks(X)
    ax.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    plt.show()



#                                   Mostrar Complexidade

def mostrar_complexidade_graficos(model, memory20, memory40, memory60, memory80, memory100, time20, time40, time60, time80, time100):

    X = [20, 40, 60, 80, 100]
    Y_memory = [memory20, memory40, memory60, memory80, memory100]
    Y_time = [time20, time40, time60, time80, time100]

    # Gráfico de Memória
    fig1, ax1 = plt.subplots(figsize=(7, 5))
    ax1.plot(X, Y_memory, marker='o', linewidth=2, color='steelblue', markersize=7)
    for x, y in zip(X, Y_memory):
        ax1.annotate(f'{y:.2f}', (x, y), textcoords="offset points", xytext=(0, 10), ha='center', fontsize=9)
    ax1.set_title(f'Memória Utilizada no Treino vs Tamanho do Dataset — {model}', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Tamanho do Dataset (%)')
    ax1.set_ylabel('Memória (MB)')
    ax1.set_xticks(X)
    ax1.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()

    # Gráfico de Tempo
    fig2, ax2 = plt.subplots(figsize=(7, 5))
    ax2.plot(X, Y_time, marker='o', linewidth=2, color='darkorange', markersize=7)
    for x, y in zip(X, Y_time):
        ax2.annotate(f'{y:.2f}', (x, y), textcoords="offset points", xytext=(0, 10), ha='center', fontsize=9)
    ax2.set_title(f'Tempo Gasto no Treino vs Tamanho do Dataset — {model}', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Tamanho do Dataset (%)')
    ax2.set_ylabel('Tempo (s)')
    ax2.set_xticks(X)
    ax2.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()