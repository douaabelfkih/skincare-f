# app.py - API Flask pour le système de recommandation

from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
import pickle

app = Flask(__name__)
CORS(app)  # Permet les requêtes cross-origin (React → Flask)

# =============================================================================
# CHARGEMENT DU MODÈLE HYBRIDE
# =============================================================================

print("🔄 Chargement du modèle hybride...")

# Charger le modèle Random Forest hybride
with open('models/hybrid_rf_recommender.pkl', 'rb') as f:
    model_package = pickle.load(f)

rf_model = model_package['model']
label_encoder_product = model_package['label_encoder_product']
label_encoder_user = model_package['label_encoder_user']
price_encoder = model_package['price_encoder']
feature_columns = model_package['feature_columns']
df = model_package['df']
skin_rules = model_package['skin_rules']

print(f"✅ Modèle hybride chargé : {len(df)} produits")
print(f"   Accuracy : {model_package['metrics']['accuracy']*100:.2f}%")

# =============================================================================
# MAPPING DES LABELS (Français → Anglais)
# =============================================================================

PRODUCT_TYPE_MAP = {
    'nettoyant': 'Cleanser',
    'tonique': 'Toner',
    'sérum': 'Treatment',
    'hydratant': 'Moisturizer',
    'crème solaire': 'Sun protect',
    'huile visage': 'Face Oil',
    'masque': 'Face Mask',
    'exfoliant': 'Treatment',
    'contour des yeux': 'Eye cream',
    'brume': 'Mist',
    'moisturizer': 'Moisturizer',
    'cleanser': 'Cleanser',
    'treatment': 'Treatment'
}

SKIN_TYPE_MAP = {
    'combination': 'Combination',
    'mixte': 'Combination',
    'dry': 'Dry',
    'sèche': 'Dry',
    'normal': 'Normal',
    'normale': 'Normal',
    'oily': 'Oily',
    'grasse': 'Oily',
    'sensitive': 'Sensitive',
    'sensible': 'Sensitive'
}

BUDGET_MAP = {
    'budget': 'Budget',
    'économique': 'Budget',
    'medium': 'Medium',
    'moyen': 'Medium',
    'premium': 'Premium',
    'luxury': 'Luxury',
    'luxe': 'Luxury'
}

# =============================================================================
# FONCTION DE RECOMMANDATION HYBRIDE
# =============================================================================

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
        # Fallback : essayer sans le filtre irritants
        df_filtered = df[
            (df['Label'] == product_type) & 
            (df['price_category'].isin(allowed_budgets))
        ].copy()
        
        if len(df_filtered) == 0:
            # Fallback : élargir le budget
            df_filtered = df[df['Label'] == product_type].copy()
            
            if len(df_filtered) == 0:
                return pd.DataFrame()
    
    # Créer les features pour chaque produit
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
    
    # Scoring hybride adaptatif
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
    
    # Post-processing par type de peau
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

# =============================================================================
# ROUTES API
# =============================================================================

@app.route('/api/health', methods=['GET'])
def health_check():
    """Vérifie que l'API fonctionne"""
    return jsonify({
        'status': 'ok',
        'model_loaded': True,
        'model_type': 'Hybrid Random Forest + Dermatology',
        'products_count': len(df),
        'accuracy': round(model_package['metrics']['accuracy'] * 100, 2)
    })


@app.route('/api/options', methods=['GET'])
def get_options():
    """Retourne les options disponibles pour le formulaire"""
    return jsonify({
        'skin_types': ['oily', 'dry', 'normal', 'combination', 'sensitive'],
        'budgets': ['budget', 'medium', 'premium', 'luxury'],
        'product_types': list(set(PRODUCT_TYPE_MAP.values()))
    })


@app.route('/api/recommend', methods=['POST'])
def recommend():
    """
    Endpoint principal de recommandation
    
    Body JSON attendu:
    {
        "skinType": "oily",
        "budget": "medium",
        "productType": "Moisturizer",
        "avoidIrritants": true
    }
    """
    try:
        data = request.get_json()
        
        # Extraire les paramètres
        skin_type_raw = data.get('skinType', 'normal').lower()
        budget_raw = data.get('budget', 'medium').lower()
        product_type_raw = data.get('productType', 'Moisturizer').lower()
        avoid_irritants = data.get('avoidIrritants', True)
        
        # Mapper vers les valeurs anglaises
        skin_type = SKIN_TYPE_MAP.get(skin_type_raw, 'Normal')
        budget = BUDGET_MAP.get(budget_raw, 'Medium')
        product_type = PRODUCT_TYPE_MAP.get(product_type_raw, product_type_raw.title())
        
        # Validation
        if skin_type not in ['Combination', 'Dry', 'Normal', 'Oily', 'Sensitive']:
            return jsonify({
                'success': False,
                'error': f'Type de peau invalide: {skin_type_raw}'
            }), 400
        
        if budget not in ['Budget', 'Medium', 'Premium', 'Luxury']:
            return jsonify({
                'success': False,
                'error': f'Budget invalide: {budget_raw}'
            }), 400
        
        # Obtenir les recommandations avec le modèle hybride
        recs_df = recommend_products_hybrid(
            skin_type=skin_type,
            budget=budget,
            product_type=product_type,
            avoid_irritants=avoid_irritants,
            n_recommendations=10
        )
        
        # Formater les résultats
        recommendations = []
        for _, row in recs_df.iterrows():
            recommendations.append({
                'name': row['Name'],
                'brand': row['Brand_Clean'],
                'product_type': row['Label'],
                'price': float(row['Price']),
                'price_category': row['price_category'],
                'ml_score': round(float(row['ml_score']) * 100, 1),
                'derma_score': round(float(row['derma_score']) * 100, 1),
                'final_score': round(float(row['final_score']) * 100, 1),
                'quality_score': round(float(row['Quality_Score']) * 100, 1)
            })
        
        return jsonify({
            'success': True,
            'profile': {
                'skin_type': skin_type,
                'budget': budget,
                'product_type': product_type,
                'avoid_irritants': avoid_irritants
            },
            'count': len(recommendations),
            'recommendations': recommendations
        })
    
    except Exception as e:
        import traceback
        print(f"❌ Erreur: {str(e)}")
        print(traceback.format_exc())
        
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/stats', methods=['GET'])
def get_stats():
    """Statistiques du dataset"""
    try:
        stats = {
            'total_products': len(df),
            'by_category': df['Label'].value_counts().to_dict(),
            'by_price_category': df['price_category'].value_counts().to_dict(),
            'by_skin_type': {
                'Oily': int(df['Oily'].sum()),
                'Dry': int(df['Dry'].sum()),
                'Normal': int(df['Normal'].sum()),
                'Combination': int(df['Combination'].sum()),
                'Sensitive': int(df['Sensitive'].sum())
            },
            'price_range': {
                'min': float(df['Price'].min()),
                'max': float(df['Price'].max()),
                'avg': float(df['Price'].mean())
            }
        }
        
        return jsonify({
            'success': True,
            'stats': stats
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


# =============================================================================
# LANCEMENT
# =============================================================================

if __name__ == '__main__':
    print("\n" + "="*80)
    print("🚀 API SYSTÈME DE RECOMMANDATION HYBRIDE")
    print("="*80)
    print(f"\n✅ Modèle chargé : Hybrid RF + Dermatology")
    print(f"✅ Produits disponibles : {len(df)}")
    print(f"✅ Accuracy : {model_package['metrics']['accuracy']*100:.2f}%")
    print(f"\n📍 URL : http://localhost:5000")
    print("\n📋 Endpoints disponibles:")
    print("   - GET  /api/health      → Vérifier le statut")
    print("   - GET  /api/options     → Options disponibles")
    print("   - GET  /api/stats       → Statistiques du dataset")
    print("   - POST /api/recommend   → Obtenir des recommandations")
    print("\n" + "="*80 + "\n")
    
    app.run(debug=True, port=5000, host='0.0.0.0')
    import os
print("Current working directory:", os.getcwd())
print("Files:", os.listdir())
print("Models folder:", os.listdir("models"))
