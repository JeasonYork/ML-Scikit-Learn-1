------------Lecture des donneées---------------

import pandas as pd

#charger les données dans un df
df = pd.read_csv("/kaggle/input/heart-disease/heart_disease.csv", sep=',', index_col = 'ID')
# afficher les données
df.head()
print("Les données de df: \n", df.head())

# afficher une premiere description des variables
print("\n description des variables:\n")
df.describe()

--------------Afficher les informations des patient ayants 37 et aussi le plus agé------
# afficher les informations des patients ayant 37 ans
df_37 = df[df['age'] == 37]
df_37.head() # la variable target est égale à 1 pour les patients donc ils sont malade
print("\n ***les informations des patients ayant 37 ans:\n",df_37)
print("\n ***la variable target est égale à 1 pour les patients agés de 37 ans donc ils sont malade\n")

# afficher les informations des patients les plus agés
df_old_patient = df.sort_values(by='age', ascending=False).iloc[0]
print("***Les informations du client le plus agée:\n", df_old_patient)
print("\n ***Comme la variable target =0 donc le patient n'est pas malade")

---------------Comparer les proportions de malade et non malades entre les patients de sexe masculin et feminin------
#*********** méthode 1 ********************
print ("************Avec la méthode value_counts***************\n")
# aperçu des valeur pour la comparaison du resultat
print("value_counts:\n",df['target'].value_counts())

# nombre de total patients
nbr_patients = len(df)
print("\nNombre de patients total:",nbr_patients)

# nombre de patient malades avec la methode value_counts
nbr_pat_malades = df['target'].value_counts().loc[1]

# nombre de patient sains avec la methode value_counts
nbr_pat_sains = df['target'].value_counts().loc[0]

print("\nLe pourcentage de patients malades est =", nbr_pat_malades*100/nbr_patients,"%")
print("\nLe pourcentage de patients sains est =", nbr_pat_sains*100/nbr_patients,"%")


#********** méthode 2 ***********************
print ("\n************Sans la méthode value_counts***************\n")
# nombre de total patients
nbr_patients = len(df)
print("nombre de patients total",nbr_patients)

# nombre de patient malades
nbr_pat_malades = len(df[df['target'] == 1])

# nombre de patient sains
nbr_pat_sains  = len(df[df['target'] == 0])

print("Le pourcentage de patients malades est =", nbr_pat_malades*100/nbr_patients,"%")
print("Le pourcentage de patients sains est =", nbr_pat_sains*100/nbr_patients,"%")

------------Comparer les moyennes d'ages entres les individus malades et non malde------------
# La moyenne d'age des patients malades
df_pat_malade = df[df['target'] == 1]
moy_pat_malade = df_pat_malade['age'].mean()
print("La moyenne d'age des patients malades est:",moy_pat_malade)

# La moyenne d'age des patient sains
df_pat_sain = df[df['target'] == 0]
moy_pat_sain = df_pat_sain['age'].mean()
print("La moyenne d'age des patients sains est:",moy_pat_sain)

------------Remplacer les modalités 'Male' et 'Female'------------
# remplacer les données
df = df.replace(to_replace = ['Male','Female'], value = ['0','1'])
df.head()

------------Remplacer les valeurs abérantes------------
# les valeurs abérantes
df_thalach_abr = df[(df['thalach'] < 50) | (df['thalach'] > 250)]
df_thalach_abr.head() # il n'y a pas de valeur abérante

print("nombre de ligne avec (thalach<50 ou thalach>250):",len(df_thalach_abr))
print("comme il n'a pas de valeurs aberrantes, on n'a rien à remplacer, le code suivant aurait été utile pour remplacer les valeurs abérantes")

# le code suivant aurait permet de remplacer les valeurs abérantes s'il y avait par le mode (la valeur la plus répondue)
df['thalach'][(df['thalach'] < 50) | (df['thalach'] > 250)] = df['thalach'].mode()[0]

df[(df['thalach'] < 50) | (df['thalach'] > 250)].head()

-------------Afficher le nombre de valeurs manquantes pour chaque colonne de df ---------------
#Afficher le nombre de valeurs manquantes pour chaque colonne de df.
df.isna().sum(axis = 0)

-------------Supprimer les lignes de df qui ne sont pas labélisées (target est absente) ---------------
#supprimer les lignes de df qui ne sont pas labélisées (target est absente)

df = df.dropna(axis = 0, how = 'any', subset = ['target']) 

# vérifier si toutes les lignes dont la target est absente sont bien supprimées
df_target_NaN = df[df['target'].isna()]
df_target_NaN.head()

-------------Remplacer les valeurs manqauntes des colonnes 'ca' et 'exang' ---------------

# remplacer les valeurs manqauntes de la colonne ca
  #Le mode de 'ca'
mode_ca = df['ca'].mode()[0]
 # Affectation du mode aux lignes avec des valeurs manquantes
#df[df['ca'].isna()] = mode_ca

  # avec fillna()
df['ca'] = df['ca'].fillna(mode_ca)

# remplacer les valeurs manqauntes de la colonne exang
  #Le mode de exang
mode_exang = df['exang'].mode()[0]
 # Affectation du mode aux lignes avec des valeurs manquantes
#df[df['exang'].isna()] = mode_exang

  # avec fillna()
df['exang'] = df['exang'].fillna(mode_exang)


# vérification
df[['ca','exang']].isna().sum()

-------------Remplacer les valeurs manqauntes des colonnes 'trestbps' et 'chol' et 'thalach' ---------------

# Remplacer les valeurs manquantes de la colonne trestbps par sa mediane
df['trestbps'] = df['trestbps'].fillna(df['trestbps'].median())

# Remplacer les valeurs manquantes de la colonne chol par sa mediane
df['chol'] = df['chol'].fillna(df['chol'].median())

# Remplacer les valeurs manquantes de la colonne thalach par sa mediane
df['thalach'] = df['thalach'].fillna(df['thalach'].median())

#vérification d'absence de valeurs NaN pour ces colonnes
df[['trestbps','chol','thalach']].isna().sum()

-------------Séparer les variables explicatives de df dans un Dataframe ---------------
# Les variables explicatives
X = df.drop(['target'], axis=1)

# Les variables cible
y = df['target']

-------------Appliquer cette transformation à chaque colonne de X et stocker le résultat dans un nouveau DataFrame nomé X_norm ---------------
# La normalisation des valeurs 
X_norm = ( 2 * ( X - X.min(axis = 0) ) / ( X.max(axis = 0) - X.min(axis = 0) ) ) - 1

# Vérification des 10 premieres valeurs
X_norm.head(10)
# Vérification des 10 dérnieres valeurs
#X_norm.tail(10)

-------------Apprentissage ---------------
-------------Division et entrainement du modèle de classification ---------------
# Importer la focntion train_test_split 
from sklearn.model_selection import train_test_split

# séparation des données
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 2)

# nous allons utiliser un modele de classification linéare (Régression logistique)

# Importation de la classe LogisticRegression 
from sklearn.linear_model import LogisticRegression

# Instancier un modèle LogisticRegression 
log_reg = LogisticRegression()

# Entrainer le modèle sur le jeu de données avec la méthode fit
log_reg.fit(X_train, y_train)

# Prediction sur les données de test
y_pred_test_log_reg = log_reg.predict(X_test)

print(y_pred_test_log_reg[:10])

-------------Evaluation du modele ---------------
------------- Afficher le taux de bonnes prédictions-------------
### importer confusion_matrix
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score


# Calculer la matrice de confusion des prédictions produites par le modèle log_reg
confusion_matrix = confusion_matrix(y_test, y_pred_test_log_reg)
(VN, FP), (FN, VP) = confusion_matrix
print("matrice_confusion=\n",confusion_matrix)

print("\nD'apres la matrice de confusion, il y'a à peu près autant de vrais positifs:",VP, " et de vrais négatifs: ",VN,",\ndonc on peut u'iliser l'accuracy pour évaluer la pérformance")

# Calcul de l'accuracy
accuracy = accuracy_score(y_test,y_pred_test_log_reg)
print("\nLe taux de bonnes prédictions=",accuracy,'%')

-------------Sur l'echantillon de test, combien de patients malades n ont pas été détéctés par le modele -------------
#Nombre de patients malades qui n'ont pas été détécté par le modèle, cela correspond au nombre de (FN)
print("Nombre de patients malades qui n'ont pas été détécté par le modèle, cela correspond au nombre de (FN)",FN)

