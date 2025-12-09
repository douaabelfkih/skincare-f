import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os

# =============================================================================
# CONFIGURATION DE LA PAGE
# =============================================================================
st.set_page_config(
    page_title="SkinMatch - Recommandation Beauté",
    page_icon="💄",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# CSS personnalisé
st.markdown("""
<style>
    .main {
        background: linear-gradient(135deg, #fce7f3 0%, #fbcfe8 50%, #f3e8ff 100%);
    }
    .stButton>button {
        width: 100%;
        background: linear-gradient(to right, #ec4899, #f43f5e, #a855f7);
        color: white;
        font-weight: bold;
        border: none;
        border-radius: 12px;
        padding: 12px;
        font-size: 16px;
        transition: all 0.3s;
    }
    .stButton>button:hover {
        transform: scale(1.02);
        box-shadow: 0 10px 25px rgba(236, 72, 153, 0.3);
    }
    .product-card {
        background: white;
        padding: 20px;
        border-radius: 16px;
        border: 2px solid #fce7f3;
        margin: 10px 0;
        transition: all 0.3s;
    }
    .product-card:hover {
        border-color: #ec4899;
        box-shadow: 0 10px 25px rgba(236, 72, 153, 0.15);
        transform: translateY(-2px);
    }
    .score-badge {
        background: linear-gradient(to right, #ec4899, #a855f7);
        color: white;
        padding: 8px 16px;
        border-radius: 20px;
        font-weight: bold;
        display: inline-block;
    }
    h1, h2, h3 {
        color: #831843;
    }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# CHARGEMENT DU MODÈLE
# =============================================================================
@st.cache_resource
def load_model():
    """Charge le modèle hybride une seule fois"""
    try:
        with open('backend/models/hybrid_rf_recommender.pkl', 'rb') as f:
            model_package = pickle.load(f)
        return model_package
    except FileNotFoundError:
        st.error("❌ Fichier modèle introuvable. Assurez-vous que 'models/hybrid_rf_recommender.pkl' existe.")
        st.stop()
    except Exception as e:
        st.error(f"❌ Erreur lors du chargement du modèle: {str(e)}")
        st.stop()

model_package = load_model()
rf_model = model_package['model']
label_encoder_product = model_package['label_encoder_product']
label_encoder_user = model_package['label_encoder_user']
price_encoder = model_package['price_encoder']
feature_columns = model_package['feature_columns']
df = model_package['df']
skin_rules = model_package['skin_rules']

# =============================================================================
# MAPPINGS
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
}

SKIN_TYPE_MAP = {
    'Mixte': 'Combination',
    'Sèche': 'Dry',
    'Normale': 'Normal',
    'Grasse': 'Oily',
    'Sensible': 'Sensitive'
}

BUDGET_MAP = {
    'Budget (< 15€)': 'Budget',
    'Medium (15€ - 30€)': 'Medium',
    'Premium (30€ - 60€)': 'Premium',
    'Luxe (> 60€)': 'Luxury'
}

# =============================================================================
# FONCTION DE RECOMMANDATION
# =============================================================================
def recommend_products_hybrid(skin_type, budget, product_type, 
                              avoid_irritants=True, n_recommendations=10):
    """Recommandation hybride : Filtres stricts + ML + Post-processing"""
    
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

# =============================================================================
# INTERFACE STREAMLIT
# =============================================================================

# Initialisation de session_state
if 'step' not in st.session_state:
    st.session_state.step = 1
if 'skin_type' not in st.session_state:
    st.session_state.skin_type = None
if 'budget' not in st.session_state:
    st.session_state.budget = None
if 'product_type' not in st.session_state:
    st.session_state.product_type = None
if 'avoid_irritants' not in st.session_state:
    st.session_state.avoid_irritants = True
if 'recommendations' not in st.session_state:
    st.session_state.recommendations = None

# Header
st.markdown("<h1 style='text-align: center;'>💄 SkinMatch - Votre Coach Beauté</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #9333ea; font-size: 18px;'>Trouvez les produits parfaits pour votre routine</p>", unsafe_allow_html=True)
st.markdown("---")

# Progress bar
progress = st.session_state.step / 4
st.progress(progress)
st.markdown(f"<p style='text-align: center; color: #831843;'>Étape {st.session_state.step} sur 4</p>", unsafe_allow_html=True)
st.markdown("<br>", unsafe_allow_html=True)

# Affichage des recommandations
if st.session_state.recommendations is not None:
    recs_df = st.session_state.recommendations
    
    st.markdown("### ✨ Vos Recommandations Personnalisées")
    
    # Profil utilisateur
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.info(f"**Type de peau:** {st.session_state.skin_type}")
    with col2:
        st.info(f"**Budget:** {st.session_state.budget}")
    with col3:
        st.info(f"**Produit:** {st.session_state.product_type}")
    with col4:
        st.info(f"**Sans irritants:** {'Oui' if st.session_state.avoid_irritants else 'Non'}")
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    if len(recs_df) == 0:
        st.warning("Aucun produit trouvé avec ces critères. Essayez d'élargir votre budget.")
    else:
        # Affichage des produits
        for idx, row in recs_df.iterrows():
            st.markdown(f"""
            <div class="product-card">
                <div style="display: flex; justify-content: space-between; align-items: start;">
                    <div style="flex: 1;">
                        <span class="score-badge">#{idx + 1}</span>
                        <span style="color: #9333ea; font-size: 12px; margin-left: 10px;">{row['Brand_Clean']}</span>
                        <h3 style="margin: 10px 0;">{row['Name']}</h3>
                        <p style="color: #666; font-size: 14px;">
                            {row['Label']} • {row['price_category']}
                        </p>
                    </div>
                    <div style="text-align: right;">
                        <div style="background: linear-gradient(to right, #ec4899, #f43f5e); color: white; padding: 12px 20px; border-radius: 12px; font-size: 20px; font-weight: bold; margin-bottom: 10px;">
                            ${row['Price']:.2f}
                        </div>
                        <div style="background: #f3e8ff; padding: 8px 16px; border-radius: 8px;">
                            <div style="font-size: 12px; color: #9333ea;">Score de match</div>
                            <div style="font-size: 24px; font-weight: bold; background: linear-gradient(to right, #9333ea, #ec4899); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">
                                {row['final_score']*100:.0f}%
                            </div>
                        </div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("🔄 Nouvelle recherche"):
        st.session_state.step = 1
        st.session_state.recommendations = None
        st.rerun()

# Étape 1: Type de peau
elif st.session_state.step == 1:
    st.markdown("### 🌟 Quel est votre type de peau ?")
    
    cols = st.columns(5)
    skin_types = [
        ('Mixte', '🌓', 'Zone T grasse, joues sèches'),
        ('Sèche', '🏜️', 'Tiraillements, desquamation'),
        ('Normale', '😊', 'Équilibrée, peu de problèmes'),
        ('Grasse', '💧', 'Brillance, pores dilatés'),
        ('Sensible', '🌸', 'Rougeurs, réactions fréquentes')
    ]
    
    for col, (skin_type, emoji, desc) in zip(cols, skin_types):
        with col:
            if st.button(f"{emoji}\n\n**{skin_type}**\n\n{desc}", key=f"skin_{skin_type}"):
                st.session_state.skin_type = skin_type
                st.session_state.step = 2
                st.rerun()

# Étape 2: Budget
elif st.session_state.step == 2:
    st.markdown("### 💰 Quel est votre budget ?")
    
    cols = st.columns(4)
    budgets = [
        ('Budget (< 15€)', '💵'),
        ('Medium (15€ - 30€)', '💳'),
        ('Premium (30€ - 60€)', '💎'),
        ('Luxe (> 60€)', '👑')
    ]
    
    for col, (budget, emoji) in zip(cols, budgets):
        with col:
            if st.button(f"{emoji}\n\n**{budget}**", key=f"budget_{budget}"):
                st.session_state.budget = budget
                st.session_state.step = 3
                st.rerun()
    
    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("← Retour"):
        st.session_state.step = 1
        st.rerun()

# Étape 3: Type de produit
elif st.session_state.step == 3:
    st.markdown("### 🧴 Quel produit recherchez-vous ?")
    
    search = st.text_input("🔍 Rechercher un type de produit...")
    
    products = [
        ('Nettoyant', '🧼'),
        ('Tonique', '💧'),
        ('Sérum', '✨'),
        ('Hydratant', '🌊'),
        ('Crème solaire', '☀️'),
        ('Huile visage', '🌿'),
        ('Masque', '🎭'),
        ('Exfoliant', '🌟'),
        ('Contour des yeux', '👁️'),
        ('Brume', '🌫️')
    ]
    
    if search:
        products = [(p, e) for p, e in products if search.lower() in p.lower()]
    
    cols = st.columns(5)
    for i, (product, emoji) in enumerate(products):
        with cols[i % 5]:
            if st.button(f"{emoji}\n\n**{product}**", key=f"product_{product}"):
                st.session_state.product_type = product
                st.session_state.step = 4
                st.rerun()
    
    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("← Retour"):
        st.session_state.step = 2
        st.rerun()

# Étape 4: Préférences
elif st.session_state.step == 4:
    st.markdown("### ⚙️ Préférences supplémentaires")
    
    avoid = st.checkbox("🌿 Éviter les irritants (parfums, alcool, colorants)", value=True)
    st.session_state.avoid_irritants = avoid
    
    st.markdown("---")
    st.markdown("### 📋 Récapitulatif")
    st.success(f"✓ Peau **{st.session_state.skin_type.lower()}**")
    st.success(f"✓ Budget **{st.session_state.budget}**")
    st.success(f"✓ Produit **{st.session_state.product_type}**")
    
    col1, col2 = st.columns([1, 3])
    with col1:
        if st.button("← Retour"):
            st.session_state.step = 3
            st.rerun()
    with col2:
        if st.button("✨ Voir mes recommandations"):
            with st.spinner("🔍 Analyse de votre profil..."):
                # Mapper les valeurs
                skin_type_en = SKIN_TYPE_MAP[st.session_state.skin_type]
                budget_en = BUDGET_MAP[st.session_state.budget]
                product_type_en = PRODUCT_TYPE_MAP.get(st.session_state.product_type.lower(), st.session_state.product_type)
                
                # Obtenir les recommandations
                recs_df = recommend_products_hybrid(
                    skin_type=skin_type_en,
                    budget=budget_en,
                    product_type=product_type_en,
                    avoid_irritants=st.session_state.avoid_irritants,
                    n_recommendations=10
                )
                
                st.session_state.recommendations = recs_df
                st.rerun()

# Footer
st.markdown("---")
st.markdown(f"""
<div style='text-align: center; color: #9333ea; padding: 20px;'>
    <p>✨ Propulsé par IA • {len(df)} produits analysés • Précision: {model_package['metrics']['accuracy']*100:.1f}%</p>
</div>
""", unsafe_allow_html=True)
