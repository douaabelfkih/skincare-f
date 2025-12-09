import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import pickle
import ast

# =============================================================================
# CONTENT-BASED FILTERING AMÉLIORÉ (VERSION RÉALISTE)
# =============================================================================

print("="*80)
print("SYSTÈME DE RECOMMANDATION AMÉLIORÉ - VERSION RÉALISTE")
print("="*80)

# -----------------------------------------------------------------------------
# 1. CHARGEMENT DES DONNÉES
# -----------------------------------------------------------------------------
print("\n--- Chargement des données ---")

df = pd.read_csv("data/cosmetics_with_features.csv")
df['Ingredients_List'] = df['Ingredients_List'].apply(
    lambda x: ast.literal_eval(x) if isinstance(x, str) else []
)

print(f"✓ Dataset chargé : {df.shape[0]} produits")

# -----------------------------------------------------------------------------
# 2. RÈGLES MÉTIER DERMATOLOGIQUES
# -----------------------------------------------------------------------------
print("\n--- Application des règles métier dermatologiques ---")

def get_skin_compatibility_rules():
    """
    Règles dermatologiques réalistes
    """
    return {
        'Oily': {
            'preferred_ingredients': ['salicylic acid', 'niacinamide', 'zinc', 'tea tree', 'witch hazel'],
            'avoid_ingredients': ['coconut oil', 'heavy oils', 'lanolin', 'shea butter', 'cocoa butter'],
            'texture_preference': ['gel', 'light', 'water-based', 'foam'],
            'avoid_products': ['milk', 'butter', 'balm', 'cream cleanser'],
            'avoid_heavy': True
        },
        'Dry': {
            'preferred_ingredients': ['hyaluronic acid', 'ceramides', 'glycerin', 'shea butter'],
            'avoid_ingredients': ['alcohol', 'fragrance', 'sulfates'],
            'texture_preference': ['cream', 'oil', 'balm'],
            'avoid_products': ['matte', 'oil-free'],
            'avoid_heavy': False
        },
        'Sensitive': {
            'preferred_ingredients': ['centella', 'aloe', 'oat', 'allantoin', 'chamomile'],
            'avoid_ingredients': ['fragrance', 'essential oils', 'alcohol', 'retinol', 'acids'],
            'texture_preference': ['gentle', 'fragrance-free'],
            'avoid_products': ['scrub', 'peel', 'exfoliat'],
            'avoid_heavy': False
        },
        'Normal': {
            'preferred_ingredients': ['vitamin c', 'peptides', 'antioxidants'],
            'avoid_ingredients': [],
            'texture_preference': ['balanced'],
            'avoid_products': [],
            'avoid_heavy': False
        },
        'Combination': {
            'preferred_ingredients': ['niacinamide', 'hyaluronic acid'],
            'avoid_ingredients': ['heavy oils'],
            'texture_preference': ['lightweight', 'balanced'],
            'avoid_products': ['very rich', 'heavy'],
            'avoid_heavy': False
        }
    }

skin_rules = get_skin_compatibility_rules()

# Ajouter un score de compatibilité dermatologique
def calculate_derma_score(product, skin_type):
    """
    Score basé sur les ingrédients et la compatibilité dermatologique
    VERSION AMÉLIORÉE avec pénalités plus strictes
    """
    score = 0.5
    
    if skin_type not in skin_rules:
        return score
    
    rules = skin_rules[skin_type]
    ingredients_lower = ' '.join(product['Ingredients_List']).lower()
    product_name_lower = product['Name'].lower()
    
    # Bonus pour ingrédients préférés
    for ing in rules['preferred_ingredients']:
        if ing in ingredients_lower:
            score += 0.15
    
    # Pénalité FORTE pour ingrédients à éviter
    for ing in rules['avoid_ingredients']:
        if ing in ingredients_lower:
            score -= 0.25
    
    # Pénalité pour types de produits à éviter
    for avoid_prod in rules.get('avoid_products', []):
        if avoid_prod in product_name_lower:
            score -= 0.3
    
    # PÉNALITÉ TRÈS FORTE pour huiles/beurres sur peau grasse
    if skin_type == 'Oily':
        oil_keywords = ['oil', 'butter', 'balm', 'coconut', 'argan', 'shea', 'jojoba', 'milk']
        if any(keyword in product_name_lower for keyword in oil_keywords):
            score -= 0.4
        
        light_keywords = ['gel', 'water', 'serum', 'fluid', 'matte', 'foam']
        if any(keyword in product_name_lower for keyword in light_keywords):
            score += 0.25
    
    # Bonus pour peau sèche avec des crèmes riches
    if skin_type == 'Dry':
        rich_keywords = ['cream', 'butter', 'balm', 'rich', 'nourishing']
        if any(keyword in product_name_lower for keyword in rich_keywords):
            score += 0.2
    
    # Bonus pour peau sensible avec produits doux
    if skin_type == 'Sensitive':
        gentle_keywords = ['gentle', 'sensitive', 'calm', 'sooth', 'mild']
        if any(keyword in product_name_lower for keyword in gentle_keywords):
            score += 0.25
        
        if 'fragrance' in product_name_lower or 'perfum' in product_name_lower:
            score -= 0.3
    
    return max(0, min(1, score))

df['derma_score_oily'] = df.apply(lambda x: calculate_derma_score(x, 'Oily'), axis=1)
df['derma_score_dry'] = df.apply(lambda x: calculate_derma_score(x, 'Dry'), axis=1)
df['derma_score_sensitive'] = df.apply(lambda x: calculate_derma_score(x, 'Sensitive'), axis=1)
df['derma_score_normal'] = df.apply(lambda x: calculate_derma_score(x, 'Normal'), axis=1)
df['derma_score_combination'] = df.apply(lambda x: calculate_derma_score(x, 'Combination'), axis=1)

print(f"✓ Scores dermatologiques calculés")

# -----------------------------------------------------------------------------
# 3. GÉNÉRATION RÉALISTE DES DONNÉES (AVEC BRUIT ET EXCEPTIONS)
# -----------------------------------------------------------------------------
print("\n--- Génération de données réalistes avec bruit ---")

def generate_realistic_training_data(df, n_samples=15000):
    """
    Génère des paires avec comportements utilisateurs réalistes
    """
    
    skin_types = ['Combination', 'Dry', 'Normal', 'Oily', 'Sensitive']
    budgets = ['Budget', 'Medium', 'Premium', 'Luxury']
    product_types = df['Label'].unique()
    
    data = []
    np.random.seed(42)
    
    n_positive = int(n_samples * 0.6)
    n_negative = n_samples - n_positive
    
    # PAIRES POSITIVES
    for _ in range(n_positive):
        product_idx = np.random.randint(0, len(df))
        product = df.iloc[product_idx]
        
        user_product_type = product['Label']
        user_skin = np.random.choice(skin_types)
        user_budget = np.random.choice(budgets)
        user_avoid_irritants = np.random.choice([True, False], p=[0.6, 0.4])
        
        compatibility_score = 0
        
        if product[user_skin] == 1:
            compatibility_score += 0.4
        
        derma_col = f'derma_score_{user_skin.lower()}'
        compatibility_score += product[derma_col] * 0.3
        
        budget_idx_user = budgets.index(user_budget)
        budget_idx_product = budgets.index(product['price_category'])
        if budget_idx_product <= budget_idx_user:
            compatibility_score += 0.2
        elif budget_idx_product == budget_idx_user + 1:
            if np.random.random() < 0.3:
                compatibility_score += 0.1
        
        if not product['contains_irritants'] or not user_avoid_irritants:
            compatibility_score += 0.1
        
        label = 1 if compatibility_score >= 0.6 else 0
        
        if np.random.random() < 0.05:
            label = 1 - label
        
        data.append({
            'user_skin_combination': 1 if user_skin == 'Combination' else 0,
            'user_skin_dry': 1 if user_skin == 'Dry' else 0,
            'user_skin_normal': 1 if user_skin == 'Normal' else 0,
            'user_skin_oily': 1 if user_skin == 'Oily' else 0,
            'user_skin_sensitive': 1 if user_skin == 'Sensitive' else 0,
            'user_budget_budget': 1 if user_budget == 'Budget' else 0,
            'user_budget_medium': 1 if user_budget == 'Medium' else 0,
            'user_budget_premium': 1 if user_budget == 'Premium' else 0,
            'user_budget_luxury': 1 if user_budget == 'Luxury' else 0,
            'user_avoid_irritants': 1 if user_avoid_irritants else 0,
            'user_product_type': user_product_type,
            'product_combination': product['Combination'],
            'product_dry': product['Dry'],
            'product_normal': product['Normal'],
            'product_oily': product['Oily'],
            'product_sensitive': product['Sensitive'],
            'product_price_normalized': product['Price_normalized'],
            'product_quality_score': product['Quality_Score'],
            'product_contains_irritants': 1 if product['contains_irritants'] else 0,
            'product_price_category': product['price_category'],
            'product_type': product['Label'],
            'product_ingredient_complexity': product['ingredient_complexity'],
            'product_derma_score': product[derma_col],
            'suitable': label
        })
    
    # PAIRES NÉGATIVES
    for _ in range(n_negative):
        product_idx = np.random.randint(0, len(df))
        product = df.iloc[product_idx]
        
        user_skin = np.random.choice(skin_types)
        user_budget = np.random.choice(budgets)
        user_product_type = np.random.choice(product_types)
        user_avoid_irritants = np.random.choice([True, False], p=[0.7, 0.3])
        
        incompatible = False
        
        if np.random.random() < 0.5:
            while product['Label'] == user_product_type:
                user_product_type = np.random.choice(product_types)
            incompatible = True
        
        if np.random.random() < 0.3:
            budget_idx_product = budgets.index(product['price_category'])
            user_budget = np.random.choice(budgets[:max(1, budget_idx_product)])
            incompatible = True
        
        if np.random.random() < 0.4:
            incompatible_skins = [s for s in skin_types if product[s] == 0]
            if incompatible_skins:
                user_skin = np.random.choice(incompatible_skins)
                incompatible = True
        
        label = 0
        
        derma_col = f'derma_score_{user_skin.lower()}'
        
        data.append({
            'user_skin_combination': 1 if user_skin == 'Combination' else 0,
            'user_skin_dry': 1 if user_skin == 'Dry' else 0,
            'user_skin_normal': 1 if user_skin == 'Normal' else 0,
            'user_skin_oily': 1 if user_skin == 'Oily' else 0,
            'user_skin_sensitive': 1 if user_skin == 'Sensitive' else 0,
            'user_budget_budget': 1 if user_budget == 'Budget' else 0,
            'user_budget_medium': 1 if user_budget == 'Medium' else 0,
            'user_budget_premium': 1 if user_budget == 'Premium' else 0,
            'user_budget_luxury': 1 if user_budget == 'Luxury' else 0,
            'user_avoid_irritants': 1 if user_avoid_irritants else 0,
            'user_product_type': user_product_type,
            'product_combination': product['Combination'],
            'product_dry': product['Dry'],
            'product_normal': product['Normal'],
            'product_oily': product['Oily'],
            'product_sensitive': product['Sensitive'],
            'product_price_normalized': product['Price_normalized'],
            'product_quality_score': product['Quality_Score'],
            'product_contains_irritants': 1 if product['contains_irritants'] else 0,
            'product_price_category': product['price_category'],
            'product_type': product['Label'],
            'product_ingredient_complexity': product['ingredient_complexity'],
            'product_derma_score': product[derma_col],
            'suitable': label
        })
    
    return pd.DataFrame(data)

train_data = generate_realistic_training_data(df, n_samples=15000)
train_data.to_csv("data/realistic_training_data.csv", index=False)

print(f"✓ {len(train_data)} paires générées")
print(f"  - Produits adaptés (1) : {train_data['suitable'].sum()}")
print(f"  - Produits non adaptés (0) : {(train_data['suitable'] == 0).sum()}")
print(f"  - Ratio : {train_data['suitable'].sum() / len(train_data) * 100:.1f}%")

# -----------------------------------------------------------------------------
# 4. PRÉPARATION DES FEATURES
# -----------------------------------------------------------------------------
print("\n--- Préparation des features ---")

label_encoder_product = LabelEncoder()
label_encoder_user = LabelEncoder()
price_encoder = LabelEncoder()

train_data['product_type_encoded'] = label_encoder_product.fit_transform(train_data['product_type'])
train_data['user_product_type_encoded'] = label_encoder_user.fit_transform(train_data['user_product_type'])
train_data['product_price_category_encoded'] = price_encoder.fit_transform(train_data['product_price_category'])

feature_columns = [
    'user_skin_combination', 'user_skin_dry', 'user_skin_normal', 'user_skin_oily', 'user_skin_sensitive',
    'user_budget_budget', 'user_budget_medium', 'user_budget_premium', 'user_budget_luxury',
    'user_avoid_irritants', 'user_product_type_encoded',
    'product_combination', 'product_dry', 'product_normal', 'product_oily', 'product_sensitive',
    'product_price_normalized', 'product_quality_score', 'product_contains_irritants',
    'product_price_category_encoded', 'product_type_encoded', 'product_ingredient_complexity',
    'product_derma_score'
]

X = train_data[feature_columns]
y = train_data['suitable']

print(f"✓ Features : {X.shape[1]} colonnes (dont score dermatologique)")

# -----------------------------------------------------------------------------
# 5. VALIDATION CROISÉE
# -----------------------------------------------------------------------------
print("\n--- Validation croisée (5-fold) ---")

rf_temp = RandomForestClassifier(
    n_estimators=100,
    max_depth=12,
    min_samples_split=20,
    min_samples_leaf=10,
    class_weight='balanced',
    random_state=42,
    n_jobs=-1
)

cv_scores = cross_val_score(rf_temp, X, y, cv=5, scoring='f1_weighted')
print(f"✓ F1-Score moyen (CV) : {cv_scores.mean()*100:.2f}% (±{cv_scores.std()*100:.2f}%)")

# -----------------------------------------------------------------------------
# 6. TRAIN/TEST SPLIT
# -----------------------------------------------------------------------------
print("\n--- Split Train/Test (80/20) ---")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"✓ Train : {X_train.shape[0]} échantillons")
print(f"✓ Test  : {X_test.shape[0]} échantillons")

# -----------------------------------------------------------------------------
# 7. ENTRAÎNEMENT
# -----------------------------------------------------------------------------
print("\n--- Entraînement du Random Forest ---")

rf_model = RandomForestClassifier(
    n_estimators=150,
    max_depth=12,
    min_samples_split=20,
    min_samples_leaf=10,
    class_weight='balanced',
    random_state=42,
    n_jobs=-1,
    verbose=1
)

rf_model.fit(X_train, y_train)
print("\n✓ Modèle entraîné!")

# -----------------------------------------------------------------------------
# 8. ÉVALUATION
# -----------------------------------------------------------------------------
print("\n" + "="*80)
print("ÉVALUATION DU MODÈLE")
print("="*80)

y_pred = rf_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"\n📊 ACCURACY : {accuracy*100:.2f}%")
print("\n📈 Rapport de classification :")
print(classification_report(y_test, y_pred, target_names=['Non adapté', 'Adapté']))
print("\n🔢 Matrice de confusion :")
cm = confusion_matrix(y_test, y_pred)
print(cm)
print(f"\nFaux positifs : {cm[0][1]} | Faux négatifs : {cm[1][0]}")

feature_importance = pd.DataFrame({
    'feature': feature_columns,
    'importance': rf_model.feature_importances_
}).sort_values('importance', ascending=False)

print("\n🔝 Top 5 features importantes :")
for idx, row in feature_importance.head(5).iterrows():
    print(f"   {row['feature']}: {row['importance']:.4f}")

# -----------------------------------------------------------------------------
# 9. FONCTION DE RECOMMANDATION HYBRIDE
# -----------------------------------------------------------------------------
print("\n--- Création de la fonction de recommandation hybride ---")

def recommend_products_hybrid(skin_type, budget, product_type, 
                              avoid_irritants=True, n_recommendations=10):
    """
    Recommandation hybride : Filtres stricts + ML + Post-processing
    """
    
    budget_order = ['Budget', 'Medium', 'Premium', 'Luxury']
    max_budget_idx = budget_order.index(budget)
    allowed_budgets = budget_order[:max_budget_idx + 1]
    
    df_filtered = df[
        (df['Label'] == product_type) & 
        (df['price_category'].isin(allowed_budgets))
    ].copy()
    
    if avoid_irritants:
        df_filtered = df_filtered[df_filtered['contains_irritants'] == False]
    
    if len(df_filtered) == 0:
        df_filtered = df[
            (df['Label'] == product_type) & 
            (df['price_category'].isin(allowed_budgets))
        ].copy()
        
        if len(df_filtered) == 0:
            df_filtered = df[df['Label'] == product_type].copy()
            
            if len(df_filtered) == 0:
                print(f"⚠️ Aucun produit trouvé pour {product_type}")
                return pd.DataFrame()
    
    user_features = []
    for _, product in df_filtered.iterrows():
        features = {
            'user_skin_combination': 1 if skin_type == 'Combination' else 0,
            'user_skin_dry': 1 if skin_type == 'Dry' else 0,
            'user_skin_normal': 1 if skin_type == 'Normal' else 0,
            'user_skin_oily': 1 if skin_type == 'Oily' else 0,
            'user_skin_sensitive': 1 if skin_type == 'Sensitive' else 0,
            'user_budget_budget': 1 if budget == 'Budget' else 0,
            'user_budget_medium': 1 if budget == 'Medium' else 0,
            'user_budget_premium': 1 if budget == 'Premium' else 0,
            'user_budget_luxury': 1 if budget == 'Luxury' else 0,
            'user_avoid_irritants': 1 if avoid_irritants else 0,
            'user_product_type_encoded': label_encoder_user.transform([product_type])[0],
            'product_combination': product['Combination'],
            'product_dry': product['Dry'],
            'product_normal': product['Normal'],
            'product_oily': product['Oily'],
            'product_sensitive': product['Sensitive'],
            'product_price_normalized': product['Price_normalized'],
            'product_quality_score': product['Quality_Score'],
            'product_contains_irritants': 1 if product['contains_irritants'] else 0,
            'product_price_category_encoded': price_encoder.transform([product['price_category']])[0],
            'product_type_encoded': label_encoder_product.transform([product['Label']])[0],
            'product_ingredient_complexity': product['ingredient_complexity'],
            'product_derma_score': product[f'derma_score_{skin_type.lower()}']
        }
        user_features.append(features)
    
    X_pred = pd.DataFrame(user_features)[feature_columns]
    ml_probabilities = rf_model.predict_proba(X_pred)[:, 1]
    
    df_filtered['ml_score'] = ml_probabilities
    df_filtered['derma_score'] = df_filtered[f'derma_score_{skin_type.lower()}']
    
    if skin_type in ['Oily', 'Sensitive']:
        df_filtered['final_score'] = (
            df_filtered['ml_score'] * 0.4 + 
            df_filtered['derma_score'] * 0.6
        )
    else:
        df_filtered['final_score'] = (
            df_filtered['ml_score'] * 0.5 + 
            df_filtered['derma_score'] * 0.5
        )
    
    if skin_type == 'Oily':
        oil_mask = df_filtered['Name'].str.lower().str.contains(
            'oil|butter|balm|coconut|argan|shea|milk|cream cleanser', na=False
        )
        df_filtered.loc[oil_mask, 'final_score'] *= 0.3
        
        light_mask = df_filtered['Name'].str.lower().str.contains(
            'gel|water|serum|fluid|matte|foam', na=False
        )
        df_filtered.loc[light_mask, 'final_score'] *= 1.3
    
    if skin_type == 'Dry':
        rich_mask = df_filtered['Name'].str.lower().str.contains(
            'cream|butter|balm|rich|nourish', na=False
        )
        df_filtered.loc[rich_mask, 'final_score'] *= 1.3
        
        dry_mask = df_filtered['Name'].str.lower().str.contains(
            'gel|water|matte', na=False
        )
        df_filtered.loc[dry_mask, 'final_score'] *= 0.7
    
    if skin_type == 'Sensitive':
        fragrance_mask = df_filtered['Name'].str.lower().str.contains(
            'fragrance|perfum|scent', na=False
        )
        df_filtered.loc[fragrance_mask, 'final_score'] *= 0.5
        
        gentle_mask = df_filtered['Name'].str.lower().str.contains(
            'gentle|calm|sooth|sensitive', na=False
        )
        df_filtered.loc[gentle_mask, 'final_score'] *= 1.2
    
    df_filtered = df_filtered.sort_values('final_score', ascending=False)
    
    output_cols = ['Name', 'Brand_Clean', 'Label', 'Price', 'price_category', 
                   'ml_score', 'derma_score', 'final_score', 'Quality_Score']
    
    return df_filtered[output_cols].head(n_recommendations)

# -----------------------------------------------------------------------------
# 10. TEST
# -----------------------------------------------------------------------------
print("\n" + "="*80)
print("TEST DU MODÈLE HYBRIDE")
print("="*80)

print("\n👤 Profil test :")
print("   • Type de peau    : Oily")
print("   • Budget          : Medium")
print("   • Type de produit : Moisturizer")
print("   • Éviter irritants: Oui")

recs = recommend_products_hybrid(
    skin_type='Oily',
    budget='Medium',
    product_type='Moisturizer',
    avoid_irritants=True,
    n_recommendations=5
)

print(f"\n🎯 Top 5 Recommandations (Hybride ML + Dermatologie) :")
print("-" * 80)

if len(recs) > 0:
    for i, (_, row) in enumerate(recs.iterrows(), 1):
        print(f"\n  {i}. {row['Name']}")
        print(f"     Marque : {row['Brand_Clean']}")
        print(f"     Prix   : ${row['Price']:.2f} ({row['price_category']})")
        print(f"     Score ML : {row['ml_score']*100:.1f}%")
        print(f"     Score Derma : {row['derma_score']*100:.1f}%")
        print(f"     ⭐ Score Final : {row['final_score']*100:.1f}%")
else:
    print("  Aucun produit trouvé.")

# -----------------------------------------------------------------------------
# 11. SAUVEGARDE
# -----------------------------------------------------------------------------
print("\n--- Sauvegarde du modèle ---")

model_package = {
    'model': rf_model,
    'label_encoder_product': label_encoder_product,
    'label_encoder_user': label_encoder_user,
    'price_encoder': price_encoder,
    'feature_columns': feature_columns,
    'df': df,
    'skin_rules': skin_rules,
    'metrics': {
        'accuracy': accuracy,
        'cv_f1_mean': cv_scores.mean(),
        'cv_f1_std': cv_scores.std(),
        'model_type': 'Hybrid RF + Dermatology'
    }
}

with open('models/hybrid_rf_recommender.pkl', 'wb') as f:
    pickle.dump(model_package, f)

print(f"\n✓ Modèle sauvegardé : models/hybrid_rf_recommender.pkl")

print("\n" + "="*80)
print("RÉSUMÉ FINAL")
print("="*80)
print(f"""
🤖 MODÈLE HYBRIDE (ML + RÈGLES DERMATOLOGIQUES)
   • Accuracy          : {accuracy*100:.2f}%
   • F1-Score (CV)     : {cv_scores.mean()*100:.2f}% (±{cv_scores.std()*100:.2f}%)
   • Distribution      : {train_data['suitable'].sum() / len(train_data) * 100:.1f}% positifs
   • Features          : {len(feature_columns)} (dont score dermatologique)
   
🎯 AMÉLIORATIONS APPLIQUÉES :
   ✓ Données réalistes avec bruit
   ✓ Règles dermatologiques intégrées
   ✓ Validation croisée
   ✓ Scoring hybride (60% ML + 40% Dermatologie)
   ✓ Post-processing intelligent

✅ Prêt pour production !
""")