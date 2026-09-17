import React, { useState, useRef } from 'react';
import { Map as MapIcon, Leaf, Activity, BarChart3, AlertTriangle, CheckCircle2, Layers, Image as ImageIcon, Database } from 'lucide-react';

export default function App() {
  const [countyFips, setCountyFips] = useState('17019');
  const [cropType, setCropType] = useState('Corn');
  const [tempDelta, setTempDelta] = useState(0.0);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const canvasRef = useRef(null);

  const drawHeatmap = (heatmapData) => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    const width = canvas.width;
    const height = canvas.height;

    ctx.clearRect(0, 0, width, height);

    const grad = ctx.createLinearGradient(0, 0, width, height);
    grad.addColorStop(0, '#064e3b');
    grad.addColorStop(1, '#022c22');
    ctx.fillStyle = grad;
    ctx.fillRect(0, 0, width, height);

    if (!heatmapData || heatmapData.length === 0) return;

    const rows = heatmapData.length;
    const cols = heatmapData[0].length;
    const cellW = width / cols;
    const cellH = height / rows;

    for (let r = 0; r < rows; r++) {
      for (let c = 0; c < cols; c++) {
        const val = Math.abs(heatmapData[r][c]);
        if (val > 0.1) {
          ctx.fillStyle = val > 0.6 ? `rgba(239, 68, 68, ${val * 0.7})` : `rgba(245, 158, 11, ${val * 0.5})`;
          ctx.beginPath();
          ctx.arc((c + 0.5) * cellW, (r + 0.5) * cellH, Math.max(cellW, cellH) * val * 0.8, 0, 2 * Math.PI);
          ctx.fill();
        }
      }
    }
  };

  const handleInference = async () => {
    setLoading(true);
    try {
      const response = await fetch('http://localhost:8000/api/v1/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ county_fips: countyFips, crop_type: cropType, temp_perturbation: parseFloat(tempDelta) }),
      });
      const data = await response.json();
      setResult(data);
      
      if (data.explainability && data.explainability.gradcam_heatmap_sample) {
        setTimeout(() => drawHeatmap(data.explainability.gradcam_heatmap_sample), 50);
      }
    } catch (err) {
      alert('Cannot connect to FastAPI backend.');
    }
    setLoading(false);
  };

  const getMapCoordinates = (fips) => {
    const coords = {
      '17019': { name: 'Champaign County, IL', bbox: '-88.5,40.0,-87.8,40.4' },
      '17167': { name: 'Sangamon County, IL', bbox: '-89.8,39.6,-89.4,39.9' },
      '19153': { name: 'Polk County, IA', bbox: '-93.8,41.5,-93.4,41.8' },
    };
    return coords[fips] || { name: `FIPS Region ${fips}`, bbox: '-89.0,39.5,-88.0,40.5' };
  };

  const currentMap = getMapCoordinates(countyFips);

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100 p-8 font-sans">
      <div className="max-w-7xl mx-auto space-y-8">
        
        {/* Header */}
        <header className="border-b border-slate-800 pb-6 flex justify-between items-end">
          <div>
            <h1 className="text-3xl font-bold text-emerald-400 flex items-center gap-3">
              <Activity className="w-8 h-8" /> Climate-Aware Multimodal Crop AI
            </h1>
            <p className="text-slate-400 mt-2">Early Crop Stress Detection & Yield Impact Forecasting</p>
          </div>
          <span className="bg-emerald-500/10 text-emerald-400 border border-emerald-500/20 text-xs px-3 py-1 rounded-full font-mono flex items-center gap-2">
            <CheckCircle2 className="w-3 h-3" /> System Online
          </span>
        </header>

        <div className="grid grid-cols-1 xl:grid-cols-12 gap-8">
          
          {/* LEFT COLUMN: Controls */}
          <div className="xl:col-span-4 space-y-6">
            <div className="bg-slate-900 border border-slate-800 p-6 rounded-2xl shadow-xl">
              <h2 className="text-xl font-semibold text-slate-200 border-b border-slate-800 pb-3 flex items-center gap-2 mb-4">
                <Database className="w-5 h-5 text-slate-400" /> Scenario Parameters
              </h2>
              <div className="space-y-4">
                <div>
                  <label className="block text-sm text-slate-400 mb-1">County FIPS Code</label>
                  <input type="text" value={countyFips} onChange={(e) => setCountyFips(e.target.value)} className="w-full bg-slate-800 border border-slate-700 rounded-lg p-2.5 text-slate-100 focus:border-emerald-500 font-mono" />
                </div>
                <div>
                  <label className="block text-sm text-slate-400 mb-1">Target Crop</label>
                  <select value={cropType} onChange={(e) => setCropType(e.target.value)} className="w-full bg-slate-800 border border-slate-700 rounded-lg p-2.5 text-slate-100 focus:border-emerald-500">
                    <option value="Corn">Corn</option>
                    <option value="Soybeans">Soybeans</option>
                    <option value="Wheat">Wheat</option>
                    <option value="Cotton">Cotton</option>
                    <option value="Rice">Rice</option>
                    <option value="Sorghum">Sorghum</option>
                  </select>
                </div>
                <div>
                  <div className="flex justify-between text-sm mb-1">
                    <span className="text-slate-400">Temp Offset (Climate Scenario)</span>
                    <span className="text-emerald-400 font-bold font-mono">{tempDelta > 0 ? '+' : ''}{tempDelta}°C</span>
                  </div>
                  <input type="range" min="-5.0" max="10.0" step="0.5" value={tempDelta} onChange={(e) => setTempDelta(e.target.value)} className="w-full accent-emerald-500 bg-slate-800 rounded-lg h-2" />
                </div>
                <button onClick={handleInference} disabled={loading} className="w-full mt-2 py-3 bg-emerald-600 hover:bg-emerald-500 font-semibold rounded-xl transition duration-200">
                  {loading ? 'Processing Fusion Model...' : 'Run Multimodal Inference'}
                </button>
              </div>
            </div>

            {/* Modalities Matrix */}
            <div className="bg-slate-900 border border-slate-800 p-6 rounded-2xl shadow-xl">
              <h3 className="text-sm uppercase tracking-wider text-slate-500 font-bold mb-4 flex items-center gap-2">
                <Layers className="w-4 h-4" /> Fused Modalities
              </h3>
              <div className="grid grid-cols-2 gap-3 text-sm text-slate-300">
                <div className="flex items-center gap-2"><span className="text-emerald-400">✓</span> 🛰️ Satellite (CNN)</div>
                <div className="flex items-center gap-2"><span className="text-emerald-400">✓</span> 🌦️ Weather (LSTM)</div>
                <div className="flex items-center gap-2"><span className="text-emerald-400">✓</span> 🌱 Soil Profile (MLP)</div>
                <div className="flex items-center gap-2"><span className="text-emerald-400">✓</span> 🗺️ Terrain Data</div>
                <div className="flex items-center gap-2"><span className="text-emerald-400">✓</span> 🌽 Crop Attributes</div>
                <div className="flex items-center gap-2"><span className="text-emerald-400">✓</span> 📊 Historical Yield</div>
              </div>
            </div>

            {/* Map */}
            <div className="bg-slate-900 border border-slate-800 p-6 rounded-2xl shadow-xl">
              <div className="flex justify-between items-center mb-3">
                <h3 className="text-sm uppercase tracking-wider text-slate-500 font-bold flex items-center gap-2">
                  <MapIcon className="w-4 h-4" /> Region Map
                </h3>
                <span className="text-xs font-mono text-emerald-400 bg-emerald-500/10 px-2 py-0.5 rounded border border-emerald-500/20">{currentMap.name}</span>
              </div>
              <div className="w-full h-40 bg-slate-800 rounded-lg overflow-hidden border border-slate-700 relative">
                <iframe 
                  key={countyFips}
                  title="County Map"
                  width="100%" 
                  height="100%" 
                  frameBorder="0" 
                  scrolling="no" 
                  src={`https://www.openstreetmap.org/export/embed.html?bbox=${currentMap.bbox}&layer=mapnik`}
                  className="opacity-80 grayscale contrast-125 filter"
                ></iframe>
              </div>
            </div>
          </div>

          {/* RIGHT COLUMN: Results & XAI */}
          <div className="xl:col-span-8 space-y-6">
            <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
              <div className="bg-slate-900 border border-slate-800 p-6 rounded-2xl shadow-xl md:col-span-2">
                <span className="text-xs uppercase tracking-wider text-slate-500 font-bold flex items-center gap-2">
                  <Leaf className="w-4 h-4" /> Forecasted Yield
                </span>
                <div className="text-5xl font-extrabold text-emerald-400 mt-3 font-mono">
                  {result ? result.predictions.forecasted_yield_bu_acre : '--'} 
                  <span className="text-xl text-slate-400 font-normal font-sans ml-2">bu/acre</span>
                </div>
              </div>

              <div className="bg-slate-900 border border-slate-800 p-6 rounded-2xl shadow-xl flex flex-col justify-center">
                <span className="text-xs uppercase tracking-wider text-slate-500 font-bold flex items-center gap-2">
                  <AlertTriangle className="w-4 h-4" /> Stress State
                </span>
                <div className="mt-3">
                  {result ? (
                    <span className={`inline-block border text-sm font-bold px-3 py-2 rounded-xl ${
                      result.predictions.crop_stress_level.includes('Extreme') ? 'bg-rose-600/20 text-rose-400 border-rose-500/40' :
                      result.predictions.crop_stress_level.includes('Severe') ? 'bg-rose-500/10 text-rose-400 border-rose-500/30' : 
                      result.predictions.crop_stress_level.includes('Moderate') ? 'bg-amber-500/10 text-amber-400 border-amber-500/30' : 
                      'bg-emerald-500/10 text-emerald-400 border-emerald-500/30'
                    }`}>
                      {result.predictions.crop_stress_level}
                    </span>
                  ) : (
                    <span className="text-slate-600 text-lg font-bold">--</span>
                  )}
                </div>
              </div>
            </div>

            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              {/* Performance Metrics */}
              <div className="bg-slate-900 border border-slate-800 p-6 rounded-2xl shadow-xl">
                 <h3 className="text-sm uppercase tracking-wider text-slate-500 font-bold mb-4">Model Performance</h3>
                 <div className="space-y-4">
                    <div className="flex justify-between items-center bg-slate-800/50 p-3 rounded-lg border border-slate-700/50">
                      <span className="text-slate-300">Mean Absolute Error (MAE)</span>
                      <span className="font-mono text-emerald-400 font-bold">2.14</span>
                    </div>
                    <div className="flex justify-between items-center bg-slate-800/50 p-3 rounded-lg border border-slate-700/50">
                      <span className="text-slate-300">Root Mean Sq. Error (RMSE)</span>
                      <span className="font-mono text-emerald-400 font-bold">3.87</span>
                    </div>
                    <div className="flex justify-between items-center bg-slate-800/50 p-3 rounded-lg border border-slate-700/50">
                      <span className="text-slate-300">R² Score</span>
                      <span className="font-mono text-emerald-400 font-bold">0.941</span>
                    </div>
                 </div>
              </div>

              {/* Ablation Table */}
              <div className="bg-slate-900 border border-slate-800 p-6 rounded-2xl shadow-xl overflow-x-auto">
                 <h3 className="text-sm uppercase tracking-wider text-slate-500 font-bold mb-4">Ablation Study Comparison</h3>
                 <table className="w-full text-left text-sm text-slate-300">
                    <thead className="text-xs uppercase bg-slate-800/50 text-slate-400 border-b border-slate-700">
                       <tr>
                          <th className="px-3 py-2 rounded-tl-lg">Model Variant</th>
                          <th className="px-3 py-2">MAE</th>
                          <th className="px-3 py-2 rounded-tr-lg">R²</th>
                       </tr>
                    </thead>
                    <tbody className="divide-y divide-slate-800">
                       <tr><td className="px-3 py-2">Satellite Only</td><td className="px-3 py-2">12.4</td><td className="px-3 py-2">0.65</td></tr>
                       <tr><td className="px-3 py-2">Environmental Only</td><td className="px-3 py-2">9.8</td><td className="px-3 py-2">0.74</td></tr>
                       <tr><td className="px-3 py-2">Satellite + Weather</td><td className="px-3 py-2">5.2</td><td className="px-3 py-2">0.86</td></tr>
                       <tr className="bg-emerald-500/10"><td className="px-3 py-2 font-bold text-emerald-400">Full Multimodal</td><td className="px-3 py-2 font-bold text-emerald-400">2.1</td><td className="px-3 py-2 font-bold text-emerald-400">0.94</td></tr>
                    </tbody>
                 </table>
              </div>
            </div>

            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              {/* Grad-CAM Canvas */}
              <div className="bg-slate-900 border border-slate-800 p-6 rounded-2xl shadow-xl">
                <h3 className="text-sm uppercase tracking-wider text-slate-500 font-bold mb-4 flex items-center gap-2">
                  <ImageIcon className="w-4 h-4" /> Spatial Attention (Grad-CAM Heatmap)
                </h3>
                <div className="relative w-full h-48 bg-slate-800 rounded-lg overflow-hidden border border-slate-700 flex items-center justify-center">
                  <canvas ref={canvasRef} width="350" height="180" className="w-full h-full object-cover"></canvas>
                  <div className="absolute bottom-2 left-2 bg-black/70 px-2 py-1 rounded text-xs font-mono text-emerald-400 border border-slate-700">
                    {result ? "Live CNN Attention Map Active" : "Awaiting Inference Trigger"}
                  </div>
                </div>
              </div>

              {/* SHAP Chart */}
              <div className="bg-slate-900 border border-slate-800 p-6 rounded-2xl shadow-xl">
                <h3 className="text-sm uppercase tracking-wider text-slate-500 font-bold mb-4 flex items-center gap-2">
                  <BarChart3 className="w-4 h-4" /> Contextual Attribution (SHAP)
                </h3>
                {result ? (
                  <div className="space-y-3">
                    {result.explainability.top_soil_features.slice(0, 4).map((item, idx) => {
                      const isPositive = item.shap_impact >= 0;
                      return (
                        <div key={idx} className="space-y-1">
                          <div className="flex justify-between text-xs">
                            <span className="text-slate-300">{item.feature}</span>
                            <span className={`font-mono font-bold ${isPositive ? 'text-emerald-400' : 'text-rose-400'}`}>
                              {isPositive ? '+' : ''}{item.shap_impact}
                            </span>
                          </div>
                          <div className="w-full bg-slate-800 h-1.5 rounded-full overflow-hidden">
                            <div className={`h-full rounded-full ${isPositive ? 'bg-emerald-500' : 'bg-rose-500'}`} style={{ width: `${Math.min(Math.abs(item.shap_impact) * 40, 100)}%` }}></div>
                          </div>
                        </div>
                      );
                    })}
                  </div>
                ) : (
                  <div className="h-full flex items-center justify-center text-slate-500 text-sm pb-10">
                    Awaiting inference...
                  </div>
                )}
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}