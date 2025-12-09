import pandas as pd
import numpy as np
import re
from collections import Counter

# =============================================================================
# ÉTAPE 3 : NETTOYAGE ET PRÉPARATION DES DONNÉES
# =============================================================================

print("="*80)
print("NETTOYAGE ET PRÉPARATION DES DONNÉES")
print("="*80)

# Chargement
df = pd.read_csv("data/cosmetics.csv")
print(f"\n✓ Dataset chargé : {df.shape[0]} lignes, {df.shape[1]} colonnes")

# -----------------------------------------------------------------------------
# 3.1 TRAITEMENT DES VALEURS MANQUANTES
# -----------------------------------------------------------------------------
print("\n--- Traitement des valeurs manquantes ---")

# Afficher les valeurs manquantes
missing_before = df.isnull().sum()
print("\nValeurs manquantes par colonne (avant traitement):")
print(missing_before[missing_before > 0])

# Stratégie de traitement
# Price : remplir par la médiane du type de produit
if 'Price' in df.columns and 'Label' in df.columns:
    df['Price'] = df.groupby('Label')['Price'].transform(
        lambda x: x.fillna(x.median())
    )

# Rank : remplir par la moyenne
if 'Rank' in df.columns:
    df['Rank'].fillna(df['Rank'].mean(), inplace=True)

# Ingredients : supprimer les lignes sans ingrédients (critique pour recommandations)
if 'Ingredients' in df.columns:
    df = df[df['Ingredients'].notna()]

# Types de peau : remplir par 0 (produit non adapté)
skin_cols = ['Combination', 'Dry', 'Normal', 'Oily', 'Sensitive']
for col in skin_cols:
    if col in df.columns:
        df[col].fillna(0, inplace=True)

print(f"\n✓ Lignes après nettoyage : {df.shape[0]}")

# -----------------------------------------------------------------------------
# 3.2 NETTOYAGE DE LA COLONNE INGREDIENTS
# -----------------------------------------------------------------------------
print("\n--- Nettoyage des ingrédients ---")

def clean_ingredients(ingredient_string):
    """
    Nettoie et standardise les ingrédients
    """
    if pd.isna(ingredient_string) or not isinstance(ingredient_string, str):
        return []
    
    # Lowercase et séparation
    ingredients = ingredient_string.lower().split(',')
    
    # Nettoyage
    cleaned = []
    for ing in ingredients:
        # Supprimer espaces, caractères spéciaux
        ing = ing.strip()
        ing = re.sub(r'[^\w\s-]', '', ing)  # Garder lettres, chiffres, tirets
        ing = re.sub(r'\s+', ' ', ing)  # Normaliser espaces
        
        if ing and len(ing) > 2:  # Ignorer ingrédients trop courts
            cleaned.append(ing)
    
    return cleaned

# Appliquer le nettoyage
df['Ingredients_List'] = df['Ingredients'].apply(clean_ingredients)
df['Ingredients_Count'] = df['Ingredients_List'].apply(len)

print(f"✓ Ingrédients nettoyés et transformés en listes")
print(f"  Nombre moyen d'ingrédients par produit : {df['Ingredients_Count'].mean():.1f}")

# -----------------------------------------------------------------------------
# 3.3 CRÉATION D'UNE COLONNE SKIN_TYPE UNIQUE
# -----------------------------------------------------------------------------
print("\n--- Création de la colonne skin_type ---")

def get_skin_types(row):
    """
    Retourne une liste des types de peau compatibles
    """
    skin_types = []
    for col in ['Combination', 'Dry', 'Normal', 'Oily', 'Sensitive']:
        if col in row.index and row[col] == 1:
            skin_types.append(col)
    
    return skin_types if skin_types else ['Unknown']

df['skin_types'] = df.apply(get_skin_types, axis=1)

# Afficher la distribution
print("\nDistribution des compatibilités types de peau :")
all_types = [t for types in df['skin_types'] for t in types]
type_counter = Counter(all_types)
for skin_type, count in type_counter.most_common():
    print(f"  {skin_type}: {count} produits")

# -----------------------------------------------------------------------------
# 3.4 CATÉGORISATION DES PRIX (BUDGET)
# -----------------------------------------------------------------------------
print("\n--- Catégorisation des prix ---")

if 'Price' in df.columns:
    # Définir les tranches de prix
    df['price_category'] = pd.cut(
        df['Price'],
        bins=[0, 20, 50, 100, 500],
        labels=['Budget', 'Medium', 'Premium', 'Luxury']
    )
    
    print("\nDistribution par catégorie de prix :")
    print(df['price_category'].value_counts().sort_index())
    
    print(f"\nStatistiques de prix par catégorie :")
    print(df.groupby('price_category')['Price'].agg(['min', 'max', 'mean', 'count']))

# -----------------------------------------------------------------------------
# 3.5 NORMALISATION DES NOMS DE MARQUES
# -----------------------------------------------------------------------------
print("\n--- Normalisation des marques ---")

if 'Brand' in df.columns:
    # Nettoyer les noms de marques
    df['Brand_Clean'] = df['Brand'].str.strip().str.title()
    
    print(f"Nombre de marques uniques : {df['Brand_Clean'].nunique()}")
    print(f"\nTop 10 marques :")
    print(df['Brand_Clean'].value_counts().head(10))

# -----------------------------------------------------------------------------
# 3.6 CRÉATION D'UN SCORE DE QUALITÉ
# -----------------------------------------------------------------------------
print("\n--- Création du score de qualité ---")

if 'Rank' in df.columns:
    # Normaliser le Rank (1 = meilleur, donc inverser)
    df['Rank_normalized'] = 1 - (df['Rank'] - df['Rank'].min()) / (df['Rank'].max() - df['Rank'].min())
    
    print(f"✓ Score de qualité créé (0-1, 1 = meilleur)")

# -----------------------------------------------------------------------------
# 3.7 EXTRACTION DES INGRÉDIENTS LES PLUS FRÉQUENTS
# -----------------------------------------------------------------------------
print("\n--- Analyse des ingrédients fréquents ---")

# Compter tous les ingrédients
all_ingredients = []
for ing_list in df['Ingredients_List']:
    all_ingredients.extend(ing_list)

ingredient_counter = Counter(all_ingredients)

print(f"\nTotal d'ingrédients uniques : {len(ingredient_counter)}")
print(f"\nTop 30 ingrédients les plus fréquents :")
for ing, count in ingredient_counter.most_common(30):
    pct = 100 * count / len(df)
    print(f"  {ing}: {count} produits ({pct:.1f}%)")

# Sauvegarder la liste des ingrédients communs pour feature engineering
common_ingredients = [ing for ing, count in ingredient_counter.most_common(100)]
df['has_common_ingredients'] = df['Ingredients_List'].apply(
    lambda x: sum(1 for ing in x if ing in common_ingredients)
)

# -----------------------------------------------------------------------------
# 3.8 SAUVEGARDE DU DATASET NETTOYÉ
# -----------------------------------------------------------------------------
print("\n--- Sauvegarde du dataset nettoyé ---")

# Colonnes à garder
columns_to_keep = [
    'Label', 'Brand', 'Brand_Clean', 'Name', 'Price', 'price_category',
    'Rank', 'Rank_normalized', 'Ingredients', 'Ingredients_List', 
    'Ingredients_Count', 'skin_types', 'has_common_ingredients',
    'Combination', 'Dry', 'Normal', 'Oily', 'Sensitive'
]

# Filtrer les colonnes existantes
columns_to_keep = [col for col in columns_to_keep if col in df.columns]

df_clean = df[columns_to_keep].copy()

# Sauvegarder
df_clean.to_csv("data/cosmetics_cleaned.csv", index=False)
print(f"\n✓ Dataset nettoyé sauvegardé : data/cosmetics_cleaned.csv")
print(f"  Lignes : {df_clean.shape[0]}")
print(f"  Colonnes : {df_clean.shape[1]}")

# -----------------------------------------------------------------------------
# 3.9 RÉSUMÉ DU NETTOYAGE
# -----------------------------------------------------------------------------
print("\n" + "="*80)
print("RÉSUMÉ DU NETTOYAGE")
print("="*80)

print(f"\n✓ Lignes conservées : {df_clean.shape[0]}")
print(f"✓ Colonnes créées : skin_types, Ingredients_List, price_category, Rank_normalized")
print(f"✓ Valeurs manquantes restantes : {df_clean.isnull().sum().sum()}")
print(f"✓ Produits par type de peau :")
for skin_type, count in type_counter.most_common():
    if skin_type != 'Unknown':
        print(f"    {skin_type}: {count}")

print("\n✓ Données prêtes pour le feature engineering et la modélisation !")