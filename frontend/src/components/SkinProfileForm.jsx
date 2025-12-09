import { useState } from 'react';

const skinTypes = [
  { id: 'combination', label: 'Mixte', image: '/mixte.png', desc: 'Zone T grasse, joues sèches' },
  { id: 'dry', label: 'Sèche', image: '/dry.png', desc: 'Tiraillements, desquamation' },
  { id: 'normal', label: 'Normale', image: '/normal.png', desc: 'Équilibrée, peu de problèmes' },
  { id: 'oily', label: 'Grasse', image: '/oily.png', desc: 'Brillance, pores dilatés' },
  { id: 'sensitive', label: 'Sensible', image: '/sensitive.png', desc: 'Rougeurs, réactions fréquentes' }
];

const budgets = [
  { id: 'budget', label: 'Budget', range: '< 15€', color: 'bg-emerald-50 border-emerald-400 text-emerald-700' },
  { id: 'medium', label: 'Medium', range: '15€ - 30€', color: 'bg-blue-50 border-blue-400 text-blue-700' },
  { id: 'premium', label: 'Premium', range: '30€ - 60€', color: 'bg-purple-50 border-purple-400 text-purple-700' },
  { id: 'luxury', label: 'Luxe', range: '> 60€', color: 'bg-amber-50 border-amber-400 text-amber-700' }
];

const productTypes = [
  { name: 'Nettoyant', icon: '🧼' },
  { name: 'Tonique', icon: '💧' },
  { name: 'Sérum', icon: '✨' },
  { name: 'Hydratant', icon: '🌊' },
  { name: 'Crème solaire', icon: '☀️' },
  { name: 'Huile visage', icon: '🌿' },
  { name: 'Masque', icon: '🎭' },
  { name: 'Exfoliant', icon: '🌟' },
  { name: 'Contour des yeux', icon: '👁️' },
  { name: 'Brume', icon: '🌫️' }
];

const API_URL = 'http://localhost:5000/api';

export default function SkinProfileForm() {
  const [step, setStep] = useState(1);
  const [form, setForm] = useState({
    skinType: '',
    budget: '',
    productType: '',
    avoidIrritants: false,
    irritantsToAvoid: ''
  });
  const [loading, setLoading] = useState(false);
  const [recommendations, setRecommendations] = useState([]);
  const [error, setError] = useState(null);
  const [searchProduct, setSearchProduct] = useState('');

  const filteredProducts = productTypes.filter(p => 
    p.name.toLowerCase().includes(searchProduct.toLowerCase())
  );

  const fetchRecommendations = async () => {
    setLoading(true);
    setError(null);
    
    try {
      const response = await fetch(`${API_URL}/recommend`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          skinType: form.skinType,
          budget: form.budget,
          productType: form.productType,
          avoidIrritants: form.avoidIrritants,
          irritantsToAvoid: form.irritantsToAvoid
        })
      });
      
      const data = await response.json();
      
      if (data.success) {
        setRecommendations(data.recommendations);
      } else {
        setError(data.error || 'Erreur lors de la récupération des recommandations');
      }
    } catch (err) {
      setError('Impossible de se connecter au serveur. Vérifiez que l\'API Flask est lancée.');
    } finally {
      setLoading(false);
    }
  };

  const handleSubmit = () => {
    fetchRecommendations();
  };

  const resetForm = () => {
    setRecommendations([]);
    setStep(1);
    setForm({
      skinType: '',
      budget: '',
      productType: '',
      avoidIrritants: false,
      irritantsToAvoid: ''
    });
    setError(null);
  };

  const canProceed = () => {
    if (step === 1) return form.skinType !== '';
    if (step === 2) return form.budget !== '';
    if (step === 3) return form.productType !== '';
    return true;
  };

  const ProgressBar = () => (
    <div className="mb-10">
      <div className="flex justify-between items-center mb-4">
        {[1, 2, 3, 4].map((s, idx) => (
          <div key={s} className="flex items-center flex-1">
            <div className={`relative flex items-center justify-center w-10 h-10 rounded-full text-sm font-bold transition-all duration-300 ${
              s < step ? 'bg-gradient-to-br from-pink-500 to-rose-500 text-white shadow-lg scale-100' : 
              s === step ? 'bg-gradient-to-br from-pink-500 to-rose-500 text-white ring-4 ring-pink-200 shadow-xl scale-110' : 
              'bg-gray-100 text-gray-400 scale-90'
            }`}>
              {s < step ? '✓' : s}
            </div>
            {idx < 3 && (
              <div className="flex-1 h-1 mx-2">
                <div className="h-full bg-gray-200 rounded-full overflow-hidden">
                  <div 
                    className={`h-full transition-all duration-500 ${
                      s < step ? 'bg-gradient-to-r from-pink-500 to-rose-500 w-full' : 'w-0'
                    }`}
                  />
                </div>
              </div>
            )}
          </div>
        ))}
      </div>
      <div className="flex justify-between text-xs text-gray-500 px-1">
        <span className={step === 1 ? 'font-semibold text-pink-600' : ''}>Type de peau</span>
        <span className={step === 2 ? 'font-semibold text-pink-600' : ''}>Budget</span>
        <span className={step === 3 ? 'font-semibold text-pink-600' : ''}>Produit</span>
        <span className={step === 4 ? 'font-semibold text-pink-600' : ''}>Préférences</span>
      </div>
    </div>
  );

  if (recommendations.length > 0) {
    return (
      <div className="min-h-screen relative p-4 py-8 overflow-hidden">
        {/* Background with image and overlay */}
        <div className="fixed inset-0 z-0">
          <img 
            src="/background.jpg"
            alt="Beauty background"
            className="absolute inset-0 w-full h-full object-cover"
          />
          <div className="absolute inset-0 bg-gradient-to-br from-rose-100/90 via-pink-100/85 to-purple-100/90 backdrop-blur-sm" />
        </div>
        
        <div className="max-w-3xl mx-auto relative z-10">
          <div className="bg-white/80 backdrop-blur-sm rounded-3xl shadow-2xl p-8 border border-pink-100">
            <div className="text-center mb-8">
              <div className="inline-block p-4 bg-gradient-to-br from-pink-100 to-purple-100 rounded-full mb-4">
                <div className="text-6xl">✨</div>
              </div>
              <h2 className="text-3xl font-bold bg-gradient-to-r from-pink-600 to-purple-600 bg-clip-text text-transparent mb-2">
                Vos Recommandations Personnalisées
              </h2>
              <p className="text-gray-600">
                {recommendations.length} produits soigneusement sélectionnés pour vous
              </p>
            </div>
            
            <div className="bg-gradient-to-r from-pink-50 to-purple-50 rounded-2xl p-5 mb-8 border border-pink-200">
              <p className="text-sm font-semibold text-pink-700 mb-3 flex items-center gap-2">
                <span className="text-lg">👤</span> Votre profil beauté
              </p>
              <div className="flex flex-wrap gap-2">
                <span className="px-4 py-2 bg-white rounded-full text-sm font-medium shadow-sm flex items-center gap-2">
                  <img 
                    src={skinTypes.find(s => s.id === form.skinType)?.image} 
                    alt={skinTypes.find(s => s.id === form.skinType)?.label}
                    className="w-5 h-5 object-contain"
                  />
                  {skinTypes.find(s => s.id === form.skinType)?.label}
                </span>
                <span className="px-4 py-2 bg-white rounded-full text-sm font-medium shadow-sm">
                  💰 {budgets.find(b => b.id === form.budget)?.label}
                </span>
                <span className="px-4 py-2 bg-white rounded-full text-sm font-medium shadow-sm">
                  {productTypes.find(p => p.name === form.productType)?.icon} {form.productType}
                </span>
                {form.avoidIrritants && (
                  <span className="px-4 py-2 bg-white rounded-full text-sm font-medium shadow-sm">
                    🌿 Sans irritants
                  </span>
                )}
              </div>
            </div>

            <div className="space-y-4 max-h-[500px] overflow-y-auto pr-2 custom-scrollbar">
              {recommendations.map((rec, index) => (
                <div 
                  key={index}
                  className="group bg-gradient-to-br from-white to-pink-50/30 rounded-2xl p-5 border-2 border-gray-100 hover:border-pink-300 hover:shadow-lg transition-all duration-300"
                >
                  <div className="flex justify-between items-start gap-4">
                    <div className="flex-1">
                      <div className="flex items-center gap-2 mb-2">
                        <span className="bg-gradient-to-r from-pink-500 to-rose-500 text-white text-xs font-bold px-3 py-1 rounded-full shadow-sm">
                          #{index + 1}
                        </span>
                        <span className="text-xs text-gray-500 font-medium">{rec.brand}</span>
                      </div>
                      <h3 className="font-bold text-gray-800 mb-3 text-lg group-hover:text-pink-600 transition-colors">
                        {rec.name}
                      </h3>
                      <div className="flex flex-wrap gap-2">
                        <span className="text-xs bg-white border border-gray-200 px-3 py-1 rounded-full font-medium">
                          {rec.product_type}
                        </span>
                        <span className="text-xs bg-emerald-100 text-emerald-700 px-3 py-1 rounded-full font-medium">
                          {rec.price_category}
                        </span>
                      </div>
                    </div>
                    <div className="text-right flex flex-col items-end">
                      <div className="bg-gradient-to-br from-pink-500 to-rose-500 text-white px-4 py-2 rounded-xl font-bold text-xl shadow-md mb-3">
                        ${rec.price}
                      </div>
                      <div className="bg-purple-50 px-3 py-2 rounded-lg">
                        <div className="text-xs text-purple-600 font-medium">Score de match</div>
                        <div className="text-2xl font-bold bg-gradient-to-r from-purple-600 to-pink-600 bg-clip-text text-transparent">
                          {rec.final_score}%
                        </div>
                      </div>
                    </div>
                  </div>
                </div>
              ))}
            </div>

            <button 
              onClick={resetForm}
              className="w-full mt-8 py-4 rounded-xl border-2 border-pink-300 text-pink-600 font-semibold hover:bg-pink-50 hover:border-pink-400 transition-all duration-300 shadow-sm"
            >
              ← Nouvelle recherche
            </button>
          </div>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="min-h-screen relative p-6 flex items-center justify-center overflow-hidden">
        {/* Background with image and overlay */}
        <div className="fixed inset-0 z-0">
          <img 
            src="/background.jpg"
            alt="Beauty background"
            className="absolute inset-0 w-full h-full object-cover"
          />
          <div className="absolute inset-0 bg-gradient-to-br from-rose-100/90 via-pink-100/85 to-purple-100/90 backdrop-blur-sm" />
        </div>
        
        <div className="bg-white rounded-3xl shadow-2xl p-10 max-w-md w-full text-center border border-red-100 relative z-10">
          <div className="inline-block p-4 bg-red-50 rounded-full mb-4">
            <div className="text-7xl">😕</div>
          </div>
          <h2 className="text-2xl font-bold text-gray-800 mb-3">Oups !</h2>
          <p className="text-gray-600 mb-8 leading-relaxed">{error}</p>
          <button 
            onClick={resetForm}
            className="px-8 py-4 bg-gradient-to-r from-pink-500 to-rose-500 text-white rounded-xl font-semibold hover:shadow-xl transition-all duration-300"
          >
            Réessayer
          </button>
        </div>
      </div>
    );
  }

  if (loading) {
    return (
      <div className="min-h-screen relative p-6 flex items-center justify-center overflow-hidden">
        {/* Background with image and overlay */}
        <div className="fixed inset-0 z-0">
          <div 
            className="absolute inset-0 bg-cover bg-center"
            style={{
              backgroundImage: 'url(https://images.unsplash.com/photo-1596462502278-27bfdc403348?w=1920&q=80)',
            }}
          />
          <div className="absolute inset-0 bg-gradient-to-br from-rose-100/90 via-pink-100/85 to-purple-100/90 backdrop-blur-sm" />
        </div>
        
        <div className="text-center relative z-10">
          <div className="relative w-24 h-24 mx-auto mb-6">
            <div className="absolute inset-0 border-4 border-pink-200 rounded-full"></div>
            <div className="absolute inset-0 border-4 border-transparent border-t-pink-500 rounded-full animate-spin"></div>
            <div className="absolute inset-2 border-4 border-transparent border-t-purple-500 rounded-full animate-spin" style={{ animationDirection: 'reverse', animationDuration: '1s' }}></div>
          </div>
          <p className="text-gray-700 font-semibold text-lg mb-2">Analyse de votre profil...</p>
          <p className="text-gray-500">Recherche des meilleurs produits ✨</p>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen relative p-4 py-8 overflow-hidden">
      {/* Background with image and overlay */}
      <div className="fixed inset-0 z-0">
        <img 
          src="/background.jpg"
          alt="Beauty background"
          className="absolute inset-0 w-full h-full object-cover"
        />
        <div className="absolute inset-0 bg-gradient-to-br from-rose-100/80 via-pink-100/75 to-purple-100/80" />
      </div>
      
      <div className="max-w-2xl mx-auto relative z-10">
        <div className="text-center mb-8">
          <div className="inline-block p-3 bg-gradient-to-br from-pink-100 to-purple-100 rounded-full mb-4">
            <span className="text-5xl">💄</span>
          </div>
          <h1 className="text-4xl font-bold bg-gradient-to-r from-pink-600 to-purple-600 bg-clip-text text-transparent mb-2">
            Mon Profil Beauté
          </h1>
          <p className="text-gray-600">Trouvez les produits parfaits pour votre routine</p>
        </div>

        <div className="bg-white/80 backdrop-blur-sm rounded-3xl shadow-2xl p-8 border border-pink-100">
          <ProgressBar />

          <div className="min-h-[380px]">
            {step === 1 && (
              <div className="space-y-3 animate-fadeIn">
                <h2 className="text-xl font-bold text-gray-800 mb-6 flex items-center gap-2">
                  <span className="text-2xl">🌟</span> Quel est votre type de peau ?
                </h2>
                <div className="space-y-3">
                  {skinTypes.map(type => (
                    <button
                      key={type.id}
                      onClick={() => setForm({ ...form, skinType: type.id })}
                      className={`w-full p-5 rounded-2xl border-2 transition-all duration-300 text-left flex items-center gap-4 group ${
                        form.skinType === type.id 
                          ? 'border-pink-500 bg-gradient-to-r from-pink-50 to-purple-50 shadow-md scale-[1.02]' 
                          : 'border-gray-200 hover:border-pink-300 hover:shadow-sm'
                      }`}
                    >
                      <div className={`w-12 h-12 rounded-full bg-white shadow-sm flex items-center justify-center transition-transform group-hover:scale-110 ${form.skinType === type.id ? 'scale-110 ring-2 ring-pink-300' : ''}`}>
                        <img 
                          src={type.image} 
                          alt={type.label}
                          className="w-8 h-8 object-contain"
                        />
                      </div>
                      <div className="flex-1">
                        <p className="font-bold text-gray-800 mb-1">{type.label}</p>
                        <p className="text-sm text-gray-600">{type.desc}</p>
                      </div>
                      {form.skinType === type.id && (
                        <div className="text-pink-500 text-2xl">✓</div>
                      )}
                    </button>
                  ))}
                </div>
              </div>
            )}

            {step === 2 && (
              <div className="space-y-6 animate-fadeIn">
                <h2 className="text-xl font-bold text-gray-800 mb-6 flex items-center gap-2">
                  <span className="text-2xl">💰</span> Quel est votre budget ?
                </h2>
                
                {/* Budget Scale */}
                <div className="relative py-8">
                  <div className="absolute top-12 left-0 right-0 h-2 bg-gradient-to-r from-emerald-400 via-blue-400 via-purple-400 to-amber-400 rounded-full" />
                  
                  <div className="relative flex justify-between items-center">
                    {budgets.map((b, index) => (
                      <button
                        key={b.id}
                        onClick={() => setForm({ ...form, budget: b.id })}
                        className="relative flex flex-col items-center group"
                      >
                        <div className={`w-16 h-16 rounded-full border-4 transition-all duration-300 flex items-center justify-center font-bold text-xl ${
                          form.budget === b.id 
                            ? `${b.color} border-4 shadow-xl scale-125 z-10` 
                            : 'bg-white border-gray-300 hover:border-gray-400 hover:scale-110'
                        }`}>
                          {index + 1}
                        </div>
                        <div className={`mt-4 text-center transition-all duration-300 ${
                          form.budget === b.id ? 'scale-105' : ''
                        }`}>
                          <p className={`font-bold text-sm ${form.budget === b.id ? b.color.split(' ')[2] : 'text-gray-700'}`}>
                            {b.label}
                          </p>
                          <p className={`text-xs mt-1 ${form.budget === b.id ? 'font-semibold text-gray-700' : 'text-gray-500'}`}>
                            {b.range}
                          </p>
                        </div>
                      </button>
                    ))}
                  </div>
                </div>

                <div className="bg-gradient-to-r from-pink-50 to-purple-50 rounded-xl p-4 mt-6 border border-pink-200">
                  <p className="text-sm text-gray-600 text-center">
                    <span className="font-semibold">Conseil :</span> Sélectionnez votre budget pour voir les produits adaptés
                  </p>
                </div>
              </div>
            )}

            {step === 3 && (
              <div className="space-y-6 animate-fadeIn">
                <h2 className="text-xl font-bold text-gray-800 mb-6 flex items-center gap-2">
                  <span className="text-2xl">🧴</span> Quel produit recherchez-vous ?
                </h2>
                
                <div className="relative">
                  <input
                    type="text"
                    placeholder="Rechercher un type de produit..."
                    value={searchProduct}
                    onChange={(e) => setSearchProduct(e.target.value)}
                    className="w-full p-4 pl-12 border-2 border-gray-200 rounded-2xl focus:border-pink-400 focus:outline-none transition-all shadow-sm text-gray-700 placeholder-gray-400"
                  />
                  <span className="absolute left-4 top-1/2 -translate-y-1/2 text-xl">🔍</span>
                </div>

                <div className="grid grid-cols-2 gap-3 max-h-80 overflow-y-auto custom-scrollbar pr-2">
                  {filteredProducts.map(product => (
                    <button
                      key={product.name}
                      onClick={() => setForm({ ...form, productType: product.name })}
                      className={`relative p-5 rounded-2xl border-2 transition-all duration-300 text-center group overflow-hidden transform hover:-translate-y-1 ${
                        form.productType === product.name 
                          ? 'border-pink-500 bg-gradient-to-br from-pink-50 to-purple-50 shadow-lg scale-[1.02]' 
                          : 'border-gray-200 hover:border-pink-400 hover:shadow-xl bg-white'
                      }`}
                    >
                      {/* Animated gradient overlay on hover */}
                      <div className="absolute inset-0 bg-gradient-to-br from-pink-400/0 via-purple-400/0 to-pink-400/0 group-hover:from-pink-400/10 group-hover:via-purple-400/10 group-hover:to-pink-400/10 transition-all duration-500 rounded-2xl" />
                      
                      {/* Shine effect on hover */}
                      <div className="absolute inset-0 opacity-0 group-hover:opacity-100 transition-opacity duration-500">
                        <div className="absolute inset-0 bg-gradient-to-r from-transparent via-white/30 to-transparent -translate-x-full group-hover:translate-x-full transition-transform duration-700" />
                      </div>

                      {/* Icon container with gradient background */}
                      <div className={`relative w-16 h-16 mx-auto mb-3 rounded-2xl flex items-center justify-center transition-all duration-300 ${
                        form.productType === product.name 
                          ? 'bg-gradient-to-br from-pink-400 to-purple-400 scale-110 shadow-md' 
                          : 'bg-gradient-to-br from-gray-100 to-gray-200 group-hover:from-pink-200 group-hover:to-purple-200 group-hover:shadow-lg'
                      }`}>
                        <span className={`text-3xl transition-all duration-300 group-hover:scale-125 group-hover:rotate-12 ${
                          form.productType === product.name ? 'scale-110' : ''
                        }`}>
                          {product.icon}
                        </span>
                      </div>
                      
                      {/* Product name */}
                      <div className={`relative font-bold text-sm transition-all duration-300 ${
                        form.productType === product.name ? 'text-pink-700' : 'text-gray-800 group-hover:text-pink-600 group-hover:scale-105'
                      }`}>
                        {product.name}
                      </div>

                      {/* Selected indicator */}
                      {form.productType === product.name && (
                        <div className="absolute top-2 right-2 w-6 h-6 bg-pink-500 rounded-full flex items-center justify-center text-white text-xs font-bold shadow-lg animate-bounce">
                          ✓
                        </div>
                      )}
                      
                      {/* Pulse ring on hover */}
                      <div className="absolute inset-0 rounded-2xl border-2 border-pink-400 opacity-0 group-hover:opacity-100 group-hover:scale-105 transition-all duration-300" />
                    </button>
                  ))}
                </div>

                {filteredProducts.length === 0 && (
                  <div className="text-center py-8 text-gray-500">
                    <div className="text-4xl mb-2">🔍</div>
                    <p>Aucun produit trouvé</p>
                  </div>
                )}

                {form.productType && (
                  <div className="bg-gradient-to-r from-pink-50 to-purple-50 rounded-xl p-4 border border-pink-200 text-center">
                    <p className="text-sm text-gray-700">
                      <span className="font-semibold">Sélectionné :</span> {form.productType}
                    </p>
                  </div>
                )}
              </div>
            )}

            {step === 4 && (
              <div className="space-y-5 animate-fadeIn">
                <h2 className="text-xl font-bold text-gray-800 mb-6 flex items-center gap-2">
                  <span className="text-2xl">⚙️</span> Préférences supplémentaires
                </h2>
                <button
                  onClick={() => setForm({ ...form, avoidIrritants: !form.avoidIrritants })}
                  className={`w-full p-5 rounded-2xl border-2 transition-all duration-300 text-left flex items-center gap-4 group ${
                    form.avoidIrritants 
                      ? 'border-pink-500 bg-gradient-to-r from-pink-50 to-purple-50 shadow-md' 
                      : 'border-gray-200 hover:border-pink-300 hover:shadow-sm bg-white'
                  }`}
                >
                  <span className={`w-7 h-7 rounded-lg border-2 flex items-center justify-center transition-all font-bold ${
                    form.avoidIrritants ? 'bg-gradient-to-br from-pink-500 to-rose-500 border-pink-500 text-white shadow-md' : 'border-gray-300 bg-white'
                  }`}>
                    {form.avoidIrritants && '✓'}
                  </span>
                  <div className="flex-1">
                    <p className="font-bold text-gray-800 mb-1">Éviter les irritants</p>
                    <p className="text-sm text-gray-600">Parfums, alcool, colorants, etc.</p>
                  </div>
                  <span className="text-2xl group-hover:scale-110 transition-transform">🌿</span>
                </button>

                {form.avoidIrritants && (
                  <div className="animate-fadeIn">
                    <label className="block text-sm font-semibold text-gray-700 mb-3">
                      Ingrédients spécifiques à éviter (optionnel)
                    </label>
                    <textarea
                      value={form.irritantsToAvoid}
                      onChange={(e) => setForm({ ...form, irritantsToAvoid: e.target.value })}
                      placeholder="Ex: parabens, sulfates, silicones..."
                      className="w-full p-4 border-2 border-gray-200 rounded-2xl focus:border-pink-400 focus:outline-none resize-none transition-all shadow-sm"
                      rows={3}
                    />
                  </div>
                )}

                <div className="bg-gradient-to-r from-pink-50 to-purple-50 rounded-2xl p-5 mt-6 border border-pink-200">
                  <p className="text-sm font-semibold text-pink-700 mb-3 flex items-center gap-2">
                    <span>📋</span> Récapitulatif
                  </p>
                  <div className="space-y-2 text-sm">
                    <p className="flex items-center gap-2">
                      <span className="text-pink-500">✓</span>
                      <span className="font-medium">Peau {skinTypes.find(s => s.id === form.skinType)?.label.toLowerCase()}</span>
                    </p>
                    <p className="flex items-center gap-2">
                      <span className="text-pink-500">✓</span>
                      <span className="font-medium">Budget {budgets.find(b => b.id === form.budget)?.label.toLowerCase()}</span>
                    </p>
                    <p className="flex items-center gap-2">
                      <span className="text-pink-500">✓</span>
                      <span className="font-medium">{form.productType}</span>
                    </p>
                  </div>
                </div>
              </div>
            )}
          </div>

          <div className="flex gap-3 mt-8">
            {step > 1 && (
              <button
                onClick={() => setStep(step - 1)}
                className="px-6 py-4 rounded-xl border-2 border-gray-300 text-gray-700 font-semibold hover:bg-gray-50 hover:border-gray-400 transition-all duration-300 shadow-sm"
              >
                ← Retour
              </button>
            )}
            <button
              onClick={() => step < 4 ? setStep(step + 1) : handleSubmit()}
              disabled={!canProceed()}
              className={`flex-1 py-4 rounded-xl font-bold transition-all duration-300 shadow-lg ${
                canProceed()
                  ? 'bg-gradient-to-r from-pink-500 via-rose-500 to-purple-500 text-white hover:shadow-xl hover:scale-[1.02]'
                  : 'bg-gray-200 text-gray-400 cursor-not-allowed shadow-none'
              }`}
            >
              {step < 4 ? 'Continuer →' : '✨ Voir mes recommandations'}
            </button>
          </div>
        </div>
      </div>

      <style jsx>{`
        @keyframes fadeIn {
          from { opacity: 0; transform: translateY(10px); }
          to { opacity: 1; transform: translateY(0); }
        }
        .animate-fadeIn {
          animation: fadeIn 0.4s ease-out;
        }
        .custom-scrollbar::-webkit-scrollbar {
          width: 8px;
        }
        .custom-scrollbar::-webkit-scrollbar-track {
          background: #f1f1f1;
          border-radius: 10px;
        }
        .custom-scrollbar::-webkit-scrollbar-thumb {
          background: linear-gradient(to bottom, #ec4899, #a855f7);
          border-radius: 10px;
        }
        .custom-scrollbar::-webkit-scrollbar-thumb:hover {
          background: linear-gradient(to bottom, #db2777, #9333ea);
        }
      `}</style>
    </div>
  );
}