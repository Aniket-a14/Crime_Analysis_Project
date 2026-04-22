"use client";

import { useState } from 'react';
import { 
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip as RechartsTooltip, ResponsiveContainer, Cell,
  AreaChart, Area
} from 'recharts';
import { 
  TrendingUp, ShieldAlert, Crosshair, BarChart3, AlertTriangle, Zap, Activity, ChevronRight, Hash,
  Database, Cpu, Globe, Info, Terminal, Search
} from 'lucide-react';
import { cn } from '@/lib/utils';
import Image from 'next/image';

interface PythonDataProps {
  initialData: any;
}

export default function Dashboard({ initialData }: PythonDataProps) {
  const [activeTab, setActiveTab] = useState<number>(2);

  const tabs = [
    { id: 2, num: "02", name: "Trend Analysis", icon: TrendingUp, desc: "Linear Regression Engine" },
    { id: 3, num: "03", name: "Severity Engine", icon: ShieldAlert, desc: "Decision Tree Logic" },
    { id: 4, num: "04", name: "Hotspot Predictor", icon: Crosshair, desc: "Logit Classification" },
    { id: 5, num: "05", name: "Crime Forecaster", icon: BarChart3, desc: "Polynomial OLS" },
    { id: 6, num: "06", name: "Risk Matrix", icon: AlertTriangle, desc: "Global Risk Scoring" }
  ];

  const renderTabContent = () => {
    const commonSizes = "(max-width: 1024px) 100vw, (max-width: 1600px) calc(100vw - 320px), 1280px";
    
    switch(activeTab) {
      case 2:
        return (
          <div className="space-y-6 animate-in fade-in slide-in-from-bottom-4 duration-500">
            {/* MAIN PLOT HIGHLIGHT */}
            <div className="stealth-card p-6 border-b-[3px] border-b-indigo-500/50">
              <div className="flex items-center gap-2 mb-4">
                <Database className="w-4 h-4 text-emerald-400" />
                <span className="data-label text-emerald-400/80">PRIMARY_VISUAL_ARTIFACT // PLOT_02.PNG</span>
              </div>
              <div className="relative w-full h-[600px] rounded-lg border border-white/5 bg-black/40 overflow-hidden group">
                <Image 
                  src="/outputs/plot_02.png" 
                  alt="Trend Plot" 
                  fill 
                  sizes={commonSizes} 
                  priority 
                  loading="eager"
                  className="object-contain p-2 hover:scale-[1.02] transition-transform duration-1000" 
                />
                <div className="absolute top-4 right-4 flex gap-2">
                   <div className="px-3 py-1 bg-indigo-500 text-[10px] font-mono text-white font-bold opacity-80 backdrop-blur-md rounded">SOURCE: MATPLOTLIB_RENDERER</div>
                </div>
              </div>
            </div>

            <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
              <div className="stealth-card p-0 col-span-1 lg:col-span-12 overflow-hidden flex flex-col min-h-[400px]">
                <div className="p-6 border-b border-white/5 flex items-center justify-between bg-slate-900/20">
                  <div className="flex items-center gap-3">
                    <div className="w-10 h-10 rounded-lg bg-indigo-500/10 flex items-center justify-center border border-indigo-500/20">
                      <TrendingUp className="w-5 h-5 text-indigo-400" />
                    </div>
                    <div>
                      <h3 className="font-sans font-bold text-lg text-slate-100 tracking-tight leading-none">Numerical Breakdown</h3>
                      <p className="data-label mt-1">Secondary Analytics: 02_RISING_TRENDS</p>
                    </div>
                  </div>
                </div>
                
                <div className="flex-1 p-6">
                  <ResponsiveContainer width="100%" height={300}>
                    <BarChart data={initialData.trends?.trends?.slice(0,10) || []} layout="vertical" margin={{ top: 0, right: 30, left: 20, bottom: 0 }}>
                      <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" horizontal={false} vertical={true} />
                      <XAxis type="number" hide />
                      <YAxis dataKey="Crime_Type" type="category" width={180} stroke="#475569" tick={{fill: '#94a3b8', fontSize: 10, fontFamily: 'var(--font-jetbrains-mono)', fontWeight: 500}} axisLine={false} tickLine={false} />
                      <Bar dataKey="Slope" fill="#6366f1" radius={[0, 2, 2, 0]} barSize={16}>
                        {(initialData.trends?.trends?.slice(0,10) || []).map((_: any, index: number) => (
                            <Cell key={`cell-${index}`} fill={index < 3 ? '#f43f5e' : '#6366f1'} />
                        ))}
                      </Bar>
                    </BarChart>
                  </ResponsiveContainer>
                </div>
              </div>
            </div>
          </div>
        );
      case 3:
        return (
          <div className="space-y-6 animate-in fade-in slide-in-from-bottom-4 duration-500">
            {/* MAIN PLOT HIGHLIGHT */}
            <div className="stealth-card p-6 border-b-[3px] border-b-violet-500/50">
              <div className="flex items-center gap-3 mb-4">
                <Globe className="w-5 h-5 text-violet-400" />
                <h3 className="font-sans font-bold text-slate-100 uppercase tracking-wide">Severity Engine Analysis Matrix // PLOT_03.PNG</h3>
              </div>
              <div className="relative w-full h-[600px] rounded-lg border border-white/5 bg-slate-950 overflow-hidden">
                <Image 
                  src="/outputs/plot_03.png" 
                  alt="Severity Confusion Matrix" 
                  fill 
                  sizes={commonSizes} 
                  priority 
                  loading="eager"
                  className="object-contain p-4 mix-blend-screen opacity-90 hover:opacity-100 transition-opacity duration-1000" 
                />
              </div>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
              {[
                { label: "DT_ACCURACY", value: (initialData.severity?.accuracy * 100).toFixed(1) + "%", color: "text-emerald-500" },
                { label: "MACRO_PRECISION", value: (initialData.severity?.precision * 100).toFixed(1) + "%", color: "text-indigo-500" },
                { label: "MACRO_RECALL", value: (initialData.severity?.recall * 100).toFixed(1) + "%", color: "text-violet-500" },
                { label: "WEIGHTED_F1", value: (initialData.severity?.f1 * 100).toFixed(1) + "%", color: "text-slate-200" },
              ].map((m, i) => (
                <div key={i} className="stealth-card p-4 flex flex-col items-center justify-center text-center">
                  <span className="data-label mb-2">{m.label}</span>
                  <span className={cn("text-2xl font-black font-sans tracking-tighter", m.color)}>{m.value}</span>
                </div>
              ))}
            </div>
            
            <div className="stealth-card p-6 flex flex-col">
              <div className="flex items-center gap-3 mb-6">
                <Terminal className="w-5 h-5 text-indigo-400" />
                <h3 className="font-sans font-bold text-slate-100">Entropy Decision Weights</h3>
              </div>
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
                {initialData.severity?.top_features?.map((f: any, i: number) => (
                  <div key={i} className="group p-3 bg-slate-950/50 rounded-lg border border-white/5">
                    <div className="flex justify-between items-center mb-1.5">
                      <span className="text-[10px] font-mono font-bold text-slate-400 group-hover:text-slate-100 transition-colors">{f.feature.toUpperCase()}</span>
                      <span className="text-[9px] font-mono text-indigo-400">{(f.importance).toFixed(4)}</span>
                    </div>
                    <div className="w-full bg-slate-900 h-1 rounded-full overflow-hidden">
                      <div className="bg-indigo-500 h-full group-hover:bg-indigo-400 transition-all duration-500" style={{width: `${f.importance * 300}%`}}></div>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>
        );
      case 4:
        return (
          <div className="space-y-6 animate-in fade-in slide-in-from-bottom-4 duration-500">
            {/* MAIN PLOT HIGHLIGHT */}
            <div className="stealth-card p-6 border-b-[3px] border-b-rose-500/50">
              <div className="flex items-center gap-3 mb-4">
                <Activity className="w-5 h-5 text-rose-500" />
                <h3 className="font-sans font-bold text-slate-100 uppercase tracking-wide">Predictive Accuracy Curve (ROC) // PLOT_04.PNG</h3>
              </div>
              <div className="relative w-full h-[600px] rounded-lg border border-white/5 bg-slate-950 overflow-hidden">
                <Image 
                  src="/outputs/plot_04.png" 
                  alt="Hotspot ROC Plot" 
                  fill 
                  sizes={commonSizes} 
                  priority 
                  loading="eager"
                  className="object-contain p-4 hover:scale-[1.01] transition-transform duration-1000" 
                />
              </div>
            </div>

             <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              <div className="stealth-card p-6 flex items-center justify-between">
                <div>
                  <span className="data-label mb-1 block">LOGIT_ENGINE</span>
                  <span className="text-3xl font-black text-indigo-500 tracking-tighter">{(initialData.hotspots?.accuracy_lr * 100).toFixed(1)}%</span>
                </div>
                <div className="h-8 w-[1px] bg-white/10"></div>
                <div>
                  <span className="data-label mb-1 block">NAIVE_BAYES</span>
                  <span className="text-3xl font-black text-violet-500 tracking-tighter">{(initialData.hotspots?.accuracy_nb * 100).toFixed(1)}%</span>
                </div>
              </div>
              <div className="stealth-card p-6 flex flex-col justify-center">
                 <h4 className="data-label text-white/40 mb-2">MODEL_STATUS</h4>
                 <div className="flex items-center gap-2">
                    <div className="w-2 h-2 rounded-full bg-emerald-500"></div>
                    <span className="text-[11px] font-mono text-emerald-500 font-bold tracking-widest">CONVERGENCE_ACHIEVED</span>
                 </div>
              </div>
            </div>
            
            <div className="stealth-card p-6">
               <h3 className="font-sans font-bold text-slate-100 uppercase tracking-wide text-sm mb-6">Hotspot Index</h3>
               <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-3">
                {initialData.hotspots?.hotspots?.map((h: any, i: number) => (
                  <div key={i} className="flex justify-between items-center p-3 bg-slate-900/40 border border-white/5 rounded">
                    <span className="font-sans font-bold text-[11px] text-slate-300 truncate w-48" title={h.CRIME_TYPE}>{h.CRIME_TYPE}</span>
                    <span className="font-mono text-[11px] font-bold text-rose-500">{(h.Probability * 100).toFixed(1)}%</span>
                  </div>
                ))}
               </div>
            </div>
          </div>
        );
      case 5:
        return (
          <div className="space-y-6 animate-in fade-in slide-in-from-bottom-4 duration-500">
            {/* MAIN PLOT HIGHLIGHT */}
            <div className="stealth-card p-0 flex flex-col h-[700px] overflow-hidden border-b-[3px] border-b-indigo-500/50">
              <div className="p-6 border-b border-white/5 bg-slate-900/20 flex items-center justify-between">
                <div className="flex items-center gap-3">
                  <BarChart3 className="w-5 h-5 text-indigo-400" />
                  <h3 className="font-sans font-bold text-slate-100 uppercase">Polynomial Regression Forecast // PLOT_05.PNG</h3>
                </div>
              </div>
              <div className="relative w-full flex-1 bg-black/20">
                <Image 
                  src="/outputs/plot_05.png" 
                  alt="Forecasting Convergence" 
                  fill 
                  sizes={commonSizes} 
                  priority 
                  loading="eager"
                  className="object-contain p-6 hover:scale-[1.01] transition-transform duration-1000" 
                />
              </div>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
              <div className="stealth-card p-6 flex flex-col justify-center">
                <span className="data-label mb-2">MODEL_R2</span>
                <span className="text-3xl font-black text-slate-100">{(initialData.forecasts?.r2 * 100).toFixed(1)}%</span>
              </div>
              <div className="stealth-card p-6 flex flex-col justify-center">
                <span className="data-label mb-2">RMSE_ERROR</span>
                <span className="text-3xl font-black text-slate-100">{initialData.forecasts?.rmse.toFixed(2)}</span>
              </div>
              <div className="stealth-card p-6 flex flex-col justify-center bg-indigo-500 border-none">
                <span className="data-label mb-2 text-indigo-900">PREDICTED_LOAD</span>
                <span className="text-4xl font-black text-white">{initialData.forecasts?.total_predicted_load.toLocaleString(undefined, {maximumFractionDigits:0})}</span>
              </div>
            </div>
          </div>
        );
      case 6:
        return (
          <div className="space-y-6 animate-in fade-in slide-in-from-bottom-4 duration-500 min-h-screen">
             {/* MAIN PLOT HIGHLIGHT */}
             <div className="stealth-card p-0 overflow-hidden flex flex-col min-h-[700px] relative border-b-[3px] border-b-rose-500/50">
                <div className="p-6 border-b border-white/5 bg-slate-900/20 flex items-center gap-3">
                   <AlertTriangle className="w-4 h-4 text-rose-500" />
                   <h3 className="font-sans font-bold text-slate-100 uppercase tracking-wide text-sm">Risk Scatter Synthesis Matrix // PLOT_06.PNG</h3>
                </div>
                <div className="relative w-full flex-1 bg-white min-h-[600px]">
                  <Image 
                    src="/outputs/plot_06.png" 
                    alt="Multi-dimensional Risk Scatter" 
                    fill 
                    sizes={commonSizes} 
                    priority 
                    loading="eager"
                    className="object-contain p-6 mix-blend-multiply transition-all hover:scale-[1.01] duration-1000" 
                  />
                </div>
             </div>

             <div className="stealth-card p-8 bg-slate-900/20">
              <div className="grid grid-cols-1 md:grid-cols-3 gap-12 text-center">
                <div>
                  <div className="text-5xl font-black text-rose-500 mb-2">{initialData.riskMatrix?.risk_distribution?.High || 0}</div>
                  <div className="data-label">CRITICAL_VECTORS</div>
                </div>
                <div>
                  <div className="text-5xl font-black text-indigo-500 mb-2">{initialData.riskMatrix?.risk_distribution?.Medium || 0}</div>
                  <div className="data-label">MODERATED_NODES</div>
                </div>
                <div>
                  <div className="text-5xl font-black text-slate-400 mb-2">{initialData.riskMatrix?.risk_distribution?.Low || 0}</div>
                  <div className="data-label">STABILIZED_SYSTEMS</div>
                </div>
              </div>
            </div>
          </div>
        );
      default:
        return null;
    }
  };

  return (
    <div className="max-w-[1600px] mx-auto p-6 md:p-12 w-full flex flex-col lg:flex-row gap-12 items-start">
      
      {/* COMMAND DOCK - MINIMALIST SIDEBAR */}
      <nav className="w-full lg:w-72 shrink-0 space-y-8 lg:sticky lg:top-12">
        <div className="space-y-1 px-4 border-b border-white/5 pb-8 mb-4">
          <h1 className="text-2xl font-sans font-black tracking-tighter text-white leading-none">CRIME_WATCH</h1>
          <p className="data-label text-indigo-500/80">INTEL_PORTAL_V4.2</p>
        </div>

        <div className="space-y-1">
          <span className="data-label px-4 mb-3 block text-white/20">PIPELINES</span>
          {tabs.map((tab) => {
            const isActive = activeTab === tab.id;
            return (
              <button
                key={tab.id}
                onClick={() => setActiveTab(tab.id)}
                className={cn(
                  "w-full group flex flex-col items-start px-4 py-4 rounded-lg transition-all duration-300 border border-transparent",
                  isActive 
                    ? "bg-slate-900/60 border-white/5 shadow-lg shadow-black/20" 
                    : "hover:bg-white/5"
                )}
              >
                <div className="flex items-center justify-between w-full mb-1">
                  <div className="flex items-center gap-3">
                    <tab.icon className={cn("w-4 h-4 transition-colors", isActive ? "text-indigo-400" : "text-slate-600")} />
                    <span className={cn("font-sans font-bold text-sm tracking-tight transition-colors", isActive ? "text-slate-100" : "text-slate-500 group-hover:text-slate-300")}>{tab.name}</span>
                  </div>
                  {isActive && <div className="w-1.5 h-1.5 rounded-full bg-indigo-500 animate-pulse ring-4 ring-indigo-500/20"></div>}
                </div>
                <span className="text-[10px] font-mono text-slate-600 pl-7 group-hover:text-slate-400 transition-colors uppercase tracking-wider">{tab.desc}</span>
              </button>
            )
          })}
        </div>

        <div className="pt-8 px-4 mt-8 border-t border-white/5">
          <div className="flex items-center gap-2 mb-4">
            <div className="w-2 h-2 rounded-full bg-emerald-500 animate-ping"></div>
            <span className="data-label text-emerald-500">SYSTEM_NOMINAL</span>
          </div>
          <p className="text-[11px] font-mono text-slate-600 leading-relaxed italic">"Encrypted secure uplink active for analytical node [77-CX-9]. External CSV injection successful."</p>
        </div>
      </nav>

      {/* CORE CONTENT REGION */}
      <div className="flex-1 min-w-0 w-full space-y-8">
        <header className="flex justify-between items-end pb-4">
           <div>
             <span className="data-label">VIEWPORT_ACTIVE</span>
             <h2 className="text-3xl font-sans font-black text-slate-100 tracking-tighter mt-1">
               {tabs.find(t => t.id === activeTab)?.name.toUpperCase()}
             </h2>
           </div>
           <div className="hidden md:flex gap-4 items-center">
             <div className="h-1 w-24 bg-slate-900 overflow-hidden rounded-full">
               <div className="bg-indigo-500/30 h-full w-[45%]"></div>
             </div>
             <span className="data-label">CACHE_UTIL: 45%</span>
           </div>
        </header>

        {renderTabContent()}
      </div>

    </div>
  );
}
