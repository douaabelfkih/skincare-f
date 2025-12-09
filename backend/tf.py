import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
import pickle
import ast
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# ÉTAPE 5 : MODÈLE TF-IDF + COSINE SIMILARITY
# =============================================================================

print("="*80)
print("SYSTÈME DE RECOMMANDATION : TF-IDF + COSINE SIMILARITY")
print("="*80)

# -----------------------------------------------------------------------------
# 5.1 CHARGEMENT DES DONNÉES
# -----------------------------------------------------------------------------
print("\n--- Chargement des données ---")

df = pd.read_csv("data/cosmetics_with_features.csv")

# Convertir les listes
df['Ingredients_List'] = df['Ingredients_List'].apply(
    lambda x: ast.literal_eval(x) if isinstance(x, str) else []
)

print(f"✓ Dataset chargé : {df.shape[0]} produits")

# -----------------------------------------------------------------------------
# 5.2 CRÉATION DU DOCUMENT TEXTUEL PAR PRODUIT
# -----------------------------------------------------------------------------
print("\n--- Création des documents textuels ---")

def create_product_document(row):
    """
    Crée un document textuel représentant le produit
    Combine : type de produit, type de peau, ingrédients
    """
    doc_parts = []
    
    # Type de produit (Label) - répété pour plus de poids
    if pd.notna(row['Label']):
        doc_parts.append(row['Label'].lower())
        doc_parts.append(row['Label'].lower())  # Répéter pour importance
    
    # Types de peau compatibles
    skin_types = ['Combination', 'Dry', 'Normal', 'Oily', 'Sensitive']
    for skin in skin_types:
        if skin in row and row[skin] == 1:
            doc_parts.append(f"skin_{skin.lower()}")
            doc_parts.append(f"skin_{skin.lower()}")  # Répéter
    
    # Catégorie de prix
    if pd.notna(row.get('price_category')):
        doc_parts.append(f"budget_{row['price_category'].lower()}")
    
    # Ingrédients
    if isinstance(row['Ingredients_List'], list):
        doc_parts.extend(row['Ingredients_List'])
    
    # Sans irritants
    if row.get('contains_irritants') == False:
        doc_parts.append('no_irritants')
        doc_parts.append('gentle')
        doc_parts.append('safe')
    
    return ' '.join(doc_parts)

# Appliquer à tous les produits
df['product_document'] = df.apply(create_product_document, axis=1)

print(f"✓ Documents créés pour {len(df)} produits")
print(f"\nExemple de document :")
print(f"  {df['product_document'].iloc[0][:200]}...")

# -----------------------------------------------------------------------------
# 5.3 VECTORISATION TF-IDF
# -----------------------------------------------------------------------------
print("\n--- Vectorisation TF-IDF ---")

tfidf_vectorizer = TfidfVectorizer(
    max_features=500,      # Nombre max de features
    min_df=2,              # Ignorer les termes trop rares
    max_df=0.95,           # Ignorer les termes trop fréquents
    ngram_range=(1, 2),    # Unigrammes et bigrammes
    stop_words=None        # Garder tous les mots (importants pour ingrédients)
)

# Créer la matrice TF-IDF
tfidf_matrix = tfidf_vectorizer.fit_transform(df['product_document'])

print(f"✓ Matrice TF-IDF créée : {tfidf_matrix.shape}")
print(f"  Vocabulaire : {len(tfidf_vectorizer.vocabulary_)} termes")

# -----------------------------------------------------------------------------
# 5.4 FONCTION DE CRÉATION DU PROFIL UTILISATEUR
# -----------------------------------------------------------------------------
print("\n--- Création de la fonction de profil utilisateur ---")

def create_user_profile(skin_type, budget, product_type, 
                        avoid_irritants=True, ingredients_to_avoid=None):
    """
    Crée un document textuel représentant le profil utilisateur
    
    Args:
        skin_type: str - 'Combination', 'Dry', 'Normal', 'Oily', 'Sensitive'
        budget: str - 'Budget', 'Medium', 'Premium', 'Luxury'
        product_type: str - Type de produit souhaité
        avoid_irritants: bool - Éviter les irritants
        ingredients_to_avoid: list - Ingrédients spécifiques à éviter
    
    Returns:
        str - Document textuel du profil
    """
    profile_parts = []
    
    # Type de produit (très important)
    profile_parts.append(product_type.lower())
    profile_parts.append(product_type.lower())
    profile_parts.append(product_type.lower())
    
    # Type de peau
    profile_parts.append(f"skin_{skin_type.lower()}")
    profile_parts.append(f"skin_{skin_type.lower()}")
    
    # Budget
    profile_parts.append(f"budget_{budget.lower()}")
    
    # Sans irritants
    if avoid_irritants:
        profile_parts.append('no_irritants')
        profile_parts.append('gentle')
        profile_parts.append('safe')
    
    return ' '.join(profile_parts)

# -----------------------------------------------------------------------------
# 5.5 FONCTION DE RECOMMANDATION
# -----------------------------------------------------------------------------
print("\n--- Création de la fonction de recommandation ---")

def recommend_products(skin_type, budget, product_type,
                       avoid_irritants=True, ingredients_to_avoid=None,
                       n_recommendations=10):
    """
    Recommande des produits basés sur le profil utilisateur
    
    Returns:
        DataFrame avec les produits recommandés et leurs scores
    """
    
    # Créer le profil utilisateur
    user_profile = create_user_profile(
        skin_type, budget, product_type, 
        avoid_irritants, ingredients_to_avoid
    )
    
    # Vectoriser le profil
    user_vector = tfidf_vectorizer.transform([user_profile])
    
    # Calculer la similarité cosinus avec tous les produits
    similarities = cosine_similarity(user_vector, tfidf_matrix).flatten()
    
    # Ajouter les scores au DataFrame
    df_results = df.copy()
    df_results['similarity_score'] = similarities
    
    # ----- FILTRAGE -----
    
    # Filtrer par type de produit
    df_results = df_results[df_results['Label'].str.lower() == product_type.lower()]
    
    # Filtrer par type de peau
    if skin_type in ['Combination', 'Dry', 'Normal', 'Oily', 'Sensitive']:
        df_results = df_results[df_results[skin_type] == 1]
    
    # Filtrer par budget (inclure les catégories inférieures ou égales)
    budget_order = ['Budget', 'Medium', 'Premium', 'Luxury']
    if budget in budget_order:
        max_idx = budget_order.index(budget)
        allowed_budgets = budget_order[:max_idx + 1]
        df_results = df_results[df_results['price_category'].isin(allowed_budgets)]
    
    # Filtrer les irritants
    if avoid_irritants:
        df_results = df_results[df_results['contains_irritants'] == False]
    
    # Filtrer les ingrédients spécifiques à éviter
    if ingredients_to_avoid:
        def has_avoided(ing_list):
            for ing in ing_list:
                for avoid in ingredients_to_avoid:
                    if avoid.lower() in ing.lower():
                        return True
            return False
        df_results = df_results[~df_results['Ingredients_List'].apply(has_avoided)]
    
    # Trier par score de similarité
    df_results = df_results.sort_values('similarity_score', ascending=False)
    
    # Sélectionner les colonnes à retourner
    output_cols = ['Name', 'Brand_Clean', 'Label', 'Price', 'price_category',
                   'similarity_score', 'Quality_Score']
    
    return df_results[output_cols].head(n_recommendations)

# -----------------------------------------------------------------------------
# 5.6 SPLIT TRAIN/TEST ET ÉVALUATION
# -----------------------------------------------------------------------------
print("\n" + "="*80)
print("ÉVALUATION DU MODÈLE")
print("="*80)

# Créer des profils de test synthétiques
print("\n--- Création des profils de test ---")

# Générer des combinaisons de profils
skin_types = ['Combination', 'Dry', 'Normal', 'Oily', 'Sensitive']
budgets = ['Budget', 'Medium', 'Premium', 'Luxury']
product_types = df['Label'].unique()

# Créer 100 profils de test aléatoires
np.random.seed(42)
n_test_profiles = 100

test_profiles = []
for _ in range(n_test_profiles):
    profile = {
        'skin_type': np.random.choice(skin_types),
        'budget': np.random.choice(budgets),
        'product_type': np.random.choice(product_types),
        'avoid_irritants': np.random.choice([True, False], p=[0.7, 0.3])
    }
    test_profiles.append(profile)
# Transformer la liste en DataFrame
test_profiles_df = pd.DataFrame(test_profiles)

# Sauvegarder dans un fichier CSV
test_profiles_df.to_csv("test_profiles.csv", index=False)

print(f"✓ Les profils de test ont été sauvegardés dans 'test_profiles.csv'")
print(f"✓ {n_test_profiles} profils de test créés")

# Évaluer : vérifier si les recommandations respectent les critères
print("\n--- Évaluation de la conformité ---")

def evaluate_recommendation(recommendations, profile):
    """
    Évalue si les recommandations respectent le profil
    
    Returns:
        dict avec les scores de conformité
    """
    if len(recommendations) == 0:
        return {'valid': False, 'scores': {}}
    
    scores = {
        'product_type_match': 0,
        'skin_type_match': 0,
        'budget_match': 0,
        'no_irritants_match': 0
    }
    
    n_recs = len(recommendations)
    
    for _, rec in recommendations.iterrows():
        # Type de produit
        if rec['Label'].lower() == profile['product_type'].lower():
            scores['product_type_match'] += 1
        
        # Budget respecté
        budget_order = ['Budget', 'Medium', 'Premium', 'Luxury']
        if rec['price_category'] in budget_order[:budget_order.index(profile['budget']) + 1]:
            scores['budget_match'] += 1
    
    # Normaliser
    for key in scores:
        scores[key] = scores[key] / n_recs if n_recs > 0 else 0
    
    return {'valid': True, 'scores': scores}

# Exécuter l'évaluation
results = []
for profile in test_profiles:
    try:
        recs = recommend_products(
            skin_type=profile['skin_type'],
            budget=profile['budget'],
            product_type=profile['product_type'],
            avoid_irritants=profile['avoid_irritants'],
            n_recommendations=10
        )
        
        eval_result = evaluate_recommendation(recs, profile)
        eval_result['n_recommendations'] = len(recs)
        results.append(eval_result)
    except Exception as e:
        results.append({'valid': False, 'scores': {}, 'n_recommendations': 0})

# Calculer les métriques globales
valid_results = [r for r in results if r['valid'] and r['n_recommendations'] > 0]

print(f"\n✓ Profils avec recommandations : {len(valid_results)}/{n_test_profiles}")

if valid_results:
    avg_scores = {
        'product_type_match': np.mean([r['scores']['product_type_match'] for r in valid_results]),
        'budget_match': np.mean([r['scores']['budget_match'] for r in valid_results])
    }
    
    # Score global = moyenne des conformités
    global_accuracy = np.mean(list(avg_scores.values())) * 100

# -----------------------------------------------------------------------------
# 5.7 AFFICHAGE DES RÉSULTATS
# -----------------------------------------------------------------------------
print("\n" + "="*80)
print("RÉSULTATS DE L'ÉVALUATION")
print("="*80)

print(f"\n📊 Métriques de conformité :")
print(f"   • Type de produit respecté  : {avg_scores['product_type_match']*100:.2f}%")
print(f"   • Budget respecté           : {avg_scores['budget_match']*100:.2f}%")
print(f"\n🎯 ACCURACY GLOBALE : {global_accuracy:.2f}%")

# Interprétation
print("\n📈 Interprétation :")
if global_accuracy >= 90:
    print("   ✅ Excellent ! Le modèle respecte très bien les critères utilisateur.")
elif global_accuracy >= 70:
    print("   ✅ Bon. Le modèle fonctionne correctement.")
elif global_accuracy >= 50:
    print("   ⚠️ Acceptable. Des améliorations sont possibles.")
else:
    print("   ❌ Faible. Le modèle nécessite des ajustements.")

# -----------------------------------------------------------------------------
# 5.8 TEST AVEC UN PROFIL RÉEL
# -----------------------------------------------------------------------------
print("\n" + "="*80)
print("TEST AVEC UN PROFIL RÉEL")
print("="*80)

print("\n👤 Profil test :")
print("   • Type de peau    : Sensitive")
print("   • Budget          : Medium")
print("   • Type de produit : Moisturizer")
print("   • Éviter irritants: Oui")

test_recs = recommend_products(
    skin_type='Sensitive',
    budget='Medium',
    product_type='Moisturizer',
    avoid_irritants=True,
    n_recommendations=5
)

print(f"\n🎯 Top 5 Recommandations :")
print("-" * 80)

if len(test_recs) > 0:
    for i, (_, row) in enumerate(test_recs.iterrows(), 1):
        print(f"\n  {i}. {row['Name']}")
        print(f"     Marque : {row['Brand_Clean']}")
        print(f"     Prix   : ${row['Price']:.2f} ({row['price_category']})")
        print(f"     Score  : {row['similarity_score']*100:.1f}%")
else:
    print("  Aucun produit trouvé pour ce profil.")

# -----------------------------------------------------------------------------
# 5.9 SAUVEGARDE DU MODÈLE
# -----------------------------------------------------------------------------
print("\n--- Sauvegarde du modèle ---")

model_data = {
    'tfidf_vectorizer': tfidf_vectorizer,
    'tfidf_matrix': tfidf_matrix,
    'df': df,
    'evaluation_metrics': {
        'accuracy': global_accuracy,
        'product_type_match': avg_scores['product_type_match'],
        'budget_match': avg_scores['budget_match']
    }
}

with open('models/tfidf_cosine_model.pkl', 'wb') as f:
    pickle.dump(model_data, f)

print(f"\n✓ Modèle sauvegardé : models/tfidf_cosine_model.pkl")

# -----------------------------------------------------------------------------
# 5.10 RÉSUMÉ FINAL
# -----------------------------------------------------------------------------
print("\n" + "="*80)
print("RÉSUMÉ FINAL")
print("="*80)

print(f"""
📊 CONFIGURATION DU MODÈLE
   • Méthode           : TF-IDF + Cosine Similarity
   • Features TF-IDF   : {tfidf_matrix.shape[1]}
   • Produits indexés  : {tfidf_matrix.shape[0]}

📈 ÉVALUATION
   • Profils testés    : {n_test_profiles}
   • Accuracy globale  : {global_accuracy:.2f}%

✅ Le modèle est prêt pour l'intégration !
""")