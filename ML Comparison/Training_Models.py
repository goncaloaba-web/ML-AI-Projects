import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier 
import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout
import matplotlib.pyplot as plt
import seaborn as sns
import random
import numpy as np
from Models_functions import SVM_model, RF_model, XGB_model, DL_model, mostrar_relatorio_completo, mostrar_complexidade_graficos, mostrar_accuracy_graficos, return_accuracy

df_100 = pd.read_csv(r"C:\Users\gonca\Desktop\df_100.csv")

X = df_100.drop('Obesity', axis=1)
y = df_100['Obesity']

if __name__ == '__main__':
    #                                                                                   DATASET COM 100% DOS DADOS

    train_X_100, test_X_100, train_y_100, test_y_100 = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

    pred100_SVM, train100_time_SVM, memory100_SVM = SVM_model(train_X_100, train_y_100, test_X_100, test_y_100) 
    pred100_RF, train100_time_RF, memory100_RF = RF_model(train_X_100, train_y_100, test_X_100, test_y_100)
    pred100_XGB, train100_time_XGB, memory100_XGB = XGB_model(train_X_100, train_y_100, test_X_100, test_y_100)
    pred100_DL, train100_time_DL, memory100_DL = DL_model(train_X_100, train_y_100, test_X_100, test_y_100)

    modelos_100 = [
        ("SVM",          pred100_SVM, train100_time_SVM, memory100_SVM),
        ("Random Forest",pred100_RF,  train100_time_RF, memory100_RF),
        ("XGBoost",      pred100_XGB, train100_time_XGB, memory100_XGB),
        ("Deep Learning",pred100_DL,  train100_time_DL, memory100_DL),
    ]

    mostrar_relatorio_completo("100% dos Dados", test_y_100, modelos_100)


    #                                                                                   DATASET COM 80 % DOS DADOS

    X_80, _, y_80, _ = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)

    train_X_80, test_X_80, train_y_80, test_y_80 = train_test_split(X_80, y_80, test_size=0.3, random_state=42, stratify=y_80) 

    pred80_SVM, train80_time_SVM, memory80_SVM = SVM_model(train_X_80, train_y_80, test_X_80, test_y_80) 
    pred80_RF, train80_time_RF, memory80_RF = RF_model(train_X_80, train_y_80, test_X_80, test_y_80)
    pred80_XGB, train80_time_XGB, memory80_XGB = XGB_model(train_X_80, train_y_80, test_X_80, test_y_80)
    pred80_DL, train80_time_DL, memory80_DL = DL_model(train_X_80, train_y_80, test_X_80, test_y_80)

    modelos_80 = [
        ("SVM",           pred80_SVM, train80_time_SVM, memory80_SVM),
        ("Random Forest", pred80_RF,  train80_time_RF, memory80_RF),
        ("XGBoost",       pred80_XGB, train80_time_XGB, memory80_XGB),
        ("Deep Learning", pred80_DL,  train80_time_DL, memory80_DL),
    ]

    mostrar_relatorio_completo("80% dos Dados", test_y_80, modelos_80)

    #                                                                                   DATASET COM 60 % DOS DADOS

    X_60, _, y_60, _ = train_test_split(X, y, test_size=0.40, random_state=42, stratify=y)

    train_X_60, test_X_60, train_y_60, test_y_60 = train_test_split(X_60, y_60, test_size=0.3, random_state=42, stratify=y_60) 

    pred60_SVM, train60_time_SVM, memory60_SVM = SVM_model(train_X_60, train_y_60, test_X_60, test_y_60) 
    pred60_RF, train60_time_RF, memory60_RF = RF_model(train_X_60, train_y_60, test_X_60, test_y_60)
    pred60_XGB, train60_time_XGB, memory60_XGB = XGB_model(train_X_60, train_y_60, test_X_60, test_y_60)
    pred60_DL, train60_time_DL, memory60_DL = DL_model(train_X_60, train_y_60, test_X_60, test_y_60)

    modelos_60 = [
        ("SVM",           pred60_SVM, train60_time_SVM, memory60_SVM),
        ("Random Forest", pred60_RF,  train60_time_RF, memory60_RF),
        ("XGBoost",       pred60_XGB, train60_time_XGB, memory60_XGB),
        ("Deep Learning", pred60_DL,  train60_time_DL, memory60_DL),
    ]

    mostrar_relatorio_completo("60% dos Dados", test_y_60, modelos_60)

    #                                                                                   DATASET COM 40 % DOS DADOS

    X_40, _, y_40, _ = train_test_split(X, y, test_size=0.60, random_state=42, stratify=y)

    train_X_40, test_X_40, train_y_40, test_y_40 = train_test_split(X_40, y_40, test_size=0.3, random_state=42, stratify=y_40)

    pred40_SVM, train40_time_SVM, memory40_SVM = SVM_model(train_X_40, train_y_40, test_X_40, test_y_40) 
    pred40_RF, train40_time_RF, memory40_RF = RF_model(train_X_40, train_y_40, test_X_40, test_y_40)
    pred40_XGB, train40_time_XGB, memory40_XGB = XGB_model(train_X_40, train_y_40, test_X_40, test_y_40)
    pred40_DL, train40_time_DL, memory40_DL = DL_model(train_X_40, train_y_40, test_X_40, test_y_40)

    modelos_40 = [
        ("SVM",           pred40_SVM, train40_time_SVM, memory40_SVM),
        ("Random Forest", pred40_RF,  train40_time_RF, memory40_RF),
        ("XGBoost",       pred40_XGB, train40_time_XGB, memory40_XGB),
        ("Deep Learning", pred40_DL,  train40_time_DL, memory40_DL),
    ]

    mostrar_relatorio_completo("40% dos Dados", test_y_40, modelos_40)

    #                                                                                   DATASET COM 20 % DOS DADOS

    X_20, _, y_20, _ = train_test_split(X, y, test_size=0.80, random_state=42, stratify=y)

    train_X_20, test_X_20, train_y_20, test_y_20 = train_test_split(X_20, y_20, test_size=0.3, random_state=42, stratify=y_20)

    pred20_SVM, train20_time_SVM, memory20_SVM = SVM_model(train_X_20, train_y_20, test_X_20, test_y_20) 
    pred20_RF, train20_time_RF, memory20_RF = RF_model(train_X_20, train_y_20, test_X_20, test_y_20)
    pred20_XGB, train20_time_XGB, memory20_XGB = XGB_model(train_X_20, train_y_20, test_X_20, test_y_20)
    pred20_DL, train20_time_DL, memory20_DL = DL_model(train_X_20, train_y_20, test_X_20, test_y_20)

    modelos_20 = [
        ("SVM",           pred20_SVM, train20_time_SVM, memory20_SVM),
        ("Random Forest", pred20_RF,  train20_time_RF, memory20_RF),
        ("XGBoost",       pred20_XGB, train20_time_XGB, memory20_XGB),
        ("Deep Learning", pred20_DL,  train20_time_DL, memory20_DL),
    ]

    mostrar_relatorio_completo("20% dos Dados", test_y_20, modelos_20)

    #Armazenar Predições e Verdadeiros:
    Y_trues = [test_y_20, test_y_40, test_y_60, test_y_80, test_y_100]
    SVM_preds = [pred20_SVM, pred40_SVM, pred60_SVM, pred80_SVM, pred100_SVM]
    RF_preds  = [pred20_RF,  pred40_RF,  pred60_RF,  pred80_RF,  pred100_RF]
    XGB_preds = [pred20_XGB, pred40_XGB, pred60_XGB, pred80_XGB, pred100_XGB]
    DL_preds  = [pred20_DL,  pred40_DL,  pred60_DL,  pred80_DL,  pred100_DL]

    SVM_accuracy = return_accuracy(SVM_preds, Y_trues)
    RF_accuracy = return_accuracy(RF_preds, Y_trues)
    XGB_accuracy = return_accuracy(XGB_preds, Y_trues)
    DL_accuracy = return_accuracy(DL_preds, Y_trues)

    # SVM
    mostrar_complexidade_graficos('Support Vector Machine', memory20_SVM, memory40_SVM, memory60_SVM, memory80_SVM, memory100_SVM, train20_time_SVM, train40_time_SVM, train60_time_SVM, train80_time_SVM, train100_time_SVM)
    mostrar_accuracy_graficos('Support Vector Machine', SVM_accuracy)

    # RF
    mostrar_complexidade_graficos('Random Forest', memory20_RF, memory40_RF, memory60_RF, memory80_RF, memory100_RF, train20_time_RF, train40_time_RF, train60_time_RF, train80_time_RF, train100_time_RF)
    mostrar_accuracy_graficos('Random Forest', RF_accuracy)

    # XGB
    mostrar_complexidade_graficos('XGBoost', memory20_XGB, memory40_XGB, memory60_XGB, memory80_XGB, memory100_XGB, train20_time_XGB, train40_time_XGB, train60_time_XGB, train80_time_XGB, train100_time_XGB)
    mostrar_accuracy_graficos('XGBoost', XGB_accuracy)

    # DL
    mostrar_complexidade_graficos('Deep Learning', memory20_DL, memory40_DL, memory60_DL, memory80_DL, memory100_DL, train20_time_DL, train40_time_DL, train60_time_DL, train80_time_DL, train100_time_DL)
    mostrar_accuracy_graficos('Deep Learning', DL_accuracy)