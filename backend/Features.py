import pandas as pd
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle

# =============================================================================
# ÉTAPE 4 : FEATURE ENGINEERING
# =============================================================================

print("="*80)
print("FEATURE ENGINEERING POUR LE SYSTÈME DE RECOMMANDATION")
print("="*80)

# Chargement des données nettoyées
df = pd.read_csv("data/cosmetics_cleaned.csv")
print(f"\n✓ Dataset chargé : {df.shape[0]} produits")

# Convertir Ingredients_List (stocké comme string) en liste
import ast
df['Ingredients_List'] = df['Ingredients_List'].apply(
    lambda x: ast.literal_eval(x) if isinstance(x, str) else []
)

# -----------------------------------------------------------------------------
# 4.1 CRÉATION DE LA BASE D'INGRÉDIENTS IRRITANTS
# -----------------------------------------------------------------------------
print("\n--- Création de la base d'ingrédients irritants ---")

# Liste des ingrédients potentiellement irritants (commune en cosmétique)
IRRITANT_INGREDIENTS = {
    # Alcools desséchants
    'alcohol', 'alcohol denat', 'ethanol', 'sd alcohol', 'isopropyl alcohol',
    
    # Parfums et fragrances
    'fragrance', 'parfum', 'perfume', 'aroma',
    
    # Acides forts (concentration élevée)
    'glycolic acid', 'salicylic acid', 'lactic acid', 'citric acid',
    
    # Sulfates
    'sodium lauryl sulfate', 'sls', 'sodium laureth sulfate', 'sles',
    
    # Conservateurs controversés
    'methylisothiazolinone', 'methylchloroisothiazolinone', 'formaldehyde',
    'dmdm hydantoin', 'imidazolidinyl urea', 'diazolidinyl urea',
    
    # Huiles essentielles (peuvent être irritantes)
    'essential oil', 'peppermint oil', 'eucalyptus oil', 'tea tree oil',
    'lavender oil', 'lemon oil', 'orange oil',
    
    # Autres
    'menthol', 'camphor', 'witch hazel', 'benzoyl peroxide',
    'retinol', 'retinoid', 'tretinoin'
}

# Fonction pour détecter les ingrédients irritants
def contains_irritants(ingredient_list, irritant_set):
    """
    Vérifie si un produit contient des ingrédients irritants
    Retourne: (bool, list) - (contient_irritants, liste_des_irritants_trouvés)
    """
    found_irritants = []
    
    for ingredient in ingredient_list:
        # Vérifier si l'ingrédient contient un mot-clé irritant
        for irritant in irritant_set:
            if irritant in ingredient.lower():
                found_irritants.append(ingredient)
                break
    
    return len(found_irritants) > 0, found_irritants

# Appliquer la détection
df['contains_irritants'], df['irritant_list'] = zip(
    *df['Ingredients_List'].apply(lambda x: contains_irritants(x, IRRITANT_INGREDIENTS))
)

print(f"✓ Produits avec ingrédients potentiellement irritants : {df['contains_irritants'].sum()}")
print(f"  ({100 * df['contains_irritants'].sum() / len(df):.1f}% du dataset)")

# Sauvegarder la base d'ingrédients irritants
with open('data/irritant_ingredients.pkl', 'wb') as f:
    pickle.dump(IRRITANT_INGREDIENTS, f)
print("\n✓ Base d'ingrédients irritants sauvegardée")

# -----------------------------------------------------------------------------
# 4.2 ENCODAGE DES TYPES DE PEAU (MULTI-LABEL)
# -----------------------------------------------------------------------------
print("\n--- Encodage des types de peau ---")

# Créer des features binaires pour chaque type de peau
skin_types = ['Combination', 'Dry', 'Normal', 'Oily', 'Sensitive']
for skin in skin_types:
    if skin not in df.columns:
        df[skin] = 0

print(f"✓ Features de type de peau créées : {skin_types}")

# -----------------------------------------------------------------------------
# 4.3 VECTORISATION DES INGRÉDIENTS (TF-IDF)
# -----------------------------------------------------------------------------
print("\n--- Vectorisation TF-IDF des ingrédients ---")

# Préparer les ingrédients comme texte
df['Ingredients_Text'] = df['Ingredients_List'].apply(lambda x: ' '.join(x))

# Créer le vectorizer TF-IDF
tfidf = TfidfVectorizer(
    max_features=200,  # Garder les 200 ingrédients les plus importants
    min_df=5,          # Ignorer les ingrédients présents dans moins de 5 produits
    ngram_range=(1, 2) # Considérer les ingrédients simples et composés
)

# Fit et transform
ingredient_tfidf = tfidf.fit_transform(df['Ingredients_Text'])

print(f"✓ Vectorisation TF-IDF complétée")
print(f"  Shape de la matrice : {ingredient_tfidf.shape}")
print(f"  Nombre de features d'ingrédients : {len(tfidf.get_feature_names_out())}")

# Sauvegarder le vectorizer
with open('models/tfidf_vectorizer.pkl', 'wb') as f:
    pickle.dump(tfidf, f)
print("✓ Vectorizer TF-IDF sauvegardé")

# -----------------------------------------------------------------------------
# 4.4 NORMALISATION DES FEATURES NUMÉRIQUES
# -----------------------------------------------------------------------------
print("\n--- Normalisation des features numériques ---")

# Prix normalisé (0-1)
if 'Price' in df.columns:
    df['Price_normalized'] = (df['Price'] - df['Price'].min()) / (df['Price'].max() - df['Price'].min())

# Rank déjà normalisé dans l'étape précédente
# Créer un score composite de qualité
if 'Rank_normalized' in df.columns:
    df['Quality_Score'] = df['Rank_normalized']
else:
    df['Quality_Score'] = 0.5  # Score par défaut

print("✓ Features numériques normalisées")

# -----------------------------------------------------------------------------
# 4.5 ENCODAGE DES CATÉGORIES DE PRODUITS
# -----------------------------------------------------------------------------
print("\n--- Encodage des catégories de produits ---")

if 'Label' in df.columns:
    # One-hot encoding des labels
    label_dummies = pd.get_dummies(df['Label'], prefix='product_type')
    
    print(f"✓ {len(label_dummies.columns)} catégories de produits encodées")
    print(f"  Exemples : {list(label_dummies.columns)[:5]}")

# -----------------------------------------------------------------------------
# 4.6 FEATURES ADDITIONNELLES
# -----------------------------------------------------------------------------
print("\n--- Création de features additionnelles ---")

# Nombre d'ingrédients (indicateur de complexité)
df['ingredient_complexity'] = df['Ingredients_Count']

# Score de popularité de la marque
if 'Brand_Clean' in df.columns:
    brand_counts = df['Brand_Clean'].value_counts()
    df['brand_popularity'] = df['Brand_Clean'].map(brand_counts)
    df['brand_popularity_norm'] = (df['brand_popularity'] - df['brand_popularity'].min()) / \
                                   (df['brand_popularity'].max() - df['brand_popularity'].min())

# Prix relatif dans sa catégorie
if 'Label' in df.columns and 'Price' in df.columns:
    df['price_vs_category_avg'] = df.groupby('Label')['Price'].transform(
        lambda x: (x - x.mean()) / x.std() if x.std() > 0 else 0
    )

print("✓ Features additionnelles créées :")
print("  - ingredient_complexity")
print("  - brand_popularity_norm")
print("  - price_vs_category_avg")

# -----------------------------------------------------------------------------
# 4.7 CRÉATION DE LA MATRICE DE FEATURES FINALE
# -----------------------------------------------------------------------------
print("\n--- Construction de la matrice de features finale ---")

# Colonnes de features à inclure
feature_columns = [
    'Price_normalized', 'Quality_Score', 'ingredient_complexity',
    'brand_popularity_norm', 'price_vs_category_avg', 'contains_irritants',
    'Combination', 'Dry', 'Normal', 'Oily', 'Sensitive'
]

# Vérifier que toutes les colonnes existent
feature_columns = [col for col in feature_columns if col in df.columns]

# Créer la matrice de features
X_basic = df[feature_columns].fillna(0)

print(f"✓ Matrice de features de base : {X_basic.shape}")
print(f"  Features incluses : {feature_columns}")

# -----------------------------------------------------------------------------
# 4.8 SAUVEGARDE DES FEATURES ET MÉTADONNÉES
# -----------------------------------------------------------------------------
print("\n--- Sauvegarde des données enrichies ---")

# Colonnes à sauvegarder pour le système de recommandation
output_columns = [
    'Label', 'Brand_Clean', 'Name', 'Price', 'price_category',
    'Rank', 'Quality_Score', 'Ingredients_List', 'Ingredients_Text',
    'contains_irritants', 'irritant_list',
    'Combination', 'Dry', 'Normal', 'Oily', 'Sensitive',
    'Price_normalized', 'ingredient_complexity', 'brand_popularity_norm'
]

output_columns = [col for col in output_columns if col in df.columns]
df_features = df[output_columns].copy()

# Sauvegarder
df_features.to_csv("data/cosmetics_with_features.csv", index=False)
print(f"\n✓ Dataset avec features sauvegardé : data/cosmetics_with_features.csv")
print(f"  Lignes : {df_features.shape[0]}")
print(f"  Colonnes : {df_features.shape[1]}")

# Sauvegarder également la matrice TF-IDF
from scipy.sparse import save_npz
save_npz('data/ingredient_tfidf_matrix.npz', ingredient_tfidf)
print("✓ Matrice TF-IDF sauvegardée")

# -----------------------------------------------------------------------------
# 4.9 STATISTIQUES FINALES
# -----------------------------------------------------------------------------
print("\n" + "="*80)
print("RÉSUMÉ DU FEATURE ENGINEERING")
print("="*80)

print(f"\n✓ Dataset enrichi : {df_features.shape[0]} produits, {df_features.shape[1]} features")
print(f"\n✓ Features créées :")
print(f"  - {ingredient_tfidf.shape[1]} features TF-IDF d'ingrédients")
print(f"  - {len(feature_columns)} features structurelles")
print(f"  - Base de {len(IRRITANT_INGREDIENTS)} ingrédients irritants")

print(f"\n✓ Distribution des produits :")
print(f"  - Avec ingrédients irritants : {df['contains_irritants'].sum()}")
print(f"  - Sans ingrédients irritants : {(~df['contains_irritants']).sum()}")

print(f"\n✓ Fichiers créés :")
print(f"  - data/cosmetics_with_features.csv")
print(f"  - data/ingredient_tfidf_matrix.npz")
print(f"  - models/tfidf_vectorizer.pkl")
print(f"  - data/irritant_ingredients.pkl")

print("\n✓ Prêt pour la construction du système de recommandation !")