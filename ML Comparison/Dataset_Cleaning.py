#Tratar do DATASET
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder

#Tratar do DATASET
df = pd.read_csv(r"ObesityDataSet_raw_and_data_sinthetic.csv")

X = df.drop('NObeyesdad', axis=1)
y = df['NObeyesdad']

#Codificar a Classe alvo
ordem_classe = [
    'Insufficient_Weight',
    'Normal_Weight',
    'Overweight_Level_I',
    'Overweight_Level_II',
    'Obesity_Type_I',
    'Obesity_Type_II',
    'Obesity_Type_III'
]
mapeamento = {classe: idx 
              for idx, classe in enumerate(ordem_classe)}
y = df['NObeyesdad'].map(mapeamento).values

#Codificar Atributos Categóricos
colunas_categoricas = X.select_dtypes(include=['object']).columns

#Binários 
ordem_genero = {"Female": 0, "Male": 1}
X['Gender'] = X['Gender'].map(ordem_genero)

resto_binarias = [col for col in colunas_categoricas if X[col].nunique() == 2]
resto_binarias.remove('Gender')
for col in resto_binarias:
    X[col] = X[col].map({'yes': True, 'no': False})

#Não Binários 
colunas_ordinal = [col for col in colunas_categoricas if col not in resto_binarias and col not in ['CAEC', 'CALC']]

#Codifiquei a CAEC e CALC como ordinais, pois tem uma ordem lógica de frequência (Para evitar a perda da ordem,que seria o caso se usássemos One-Hot Encoding ou o Ordinal Encoding sem considerar a ordem dos valores))

ordem_CAEC =    {
    'no': 0,
    'Sometimes': 1,
    'Frequently': 2,
    'Always': 3
}
ordem_CALC =    {
    'no': 0,
    'Sometimes': 1,
    'Frequently': 2,
    'Always': 3
}
X['CAEC'] = X['CAEC'].map(ordem_CAEC)
X['CALC'] = X['CALC'].map(ordem_CALC)

oe = OrdinalEncoder()
X[colunas_ordinal] = oe.fit_transform(X[colunas_ordinal])

#Arredondar os valores numéricos (Se calhar foram erros no dataset ??)
X['Age'] = X['Age'].round()
X['FCVC'] = X['FCVC'].round()
X['NCP'] = X['NCP'].round()
X['CH2O'] = X['CH2O'].round()
X["FAF"] = X["FAF"].round()
X["TUE"] = X["TUE"].round()

#Remover as variaveis peso e altura, pois estão diretamente relacionadas com a obesidade e podem causar data leakage
X = X.drop(['Height', 'Weight'], axis=1)

#Criar o dataset final completo (100%)
df_100 = X.copy()
df_100['Obesity'] = y
df_100.to_csv(r"C:\Users\gonca\Desktop\Projeto BBAS\df_100.csv", index=False)


