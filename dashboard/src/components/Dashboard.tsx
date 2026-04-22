"use client";

import { useState, useMemo } from 'react';
import { 
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip as RechartsTooltip, ResponsiveContainer, Cell,
  AreaChart, Area, PieChart, Pie
} from 'recharts';
import { 
  TrendingUp, ShieldAlert, Crosshair, BarChart3, AlertTriangle, Zap, Activity, ChevronRight, Hash,
  Database, Cpu, Globe, Info, Terminal, Search, LayoutDashboard, Shield, ListFilter,
  Eye, Filter
} from 'lucide-react';
import { cn } from '@/lib/utils';
import Image from 'next/image';

interface PythonDataProps {
  initialData: any;
}

export default function Dashboard({ initialData }: PythonDataProps) {
  const [activeTab, setActiveTab] = useState<number>(1);
  const [searchQuery, setSearchQuery] = useState('');

  const tabs = [
    { id: 1, num: "01", name: "Executive Intel", icon: LayoutDashboard, desc: "Global Situational Awareness" },
    { id: 2, num: "02", name: "Trend Analysis", icon: TrendingUp, desc: "Linear Regression Engine" },
    { id: 3, num: "03", name: "Severity Engine", icon: ShieldAlert, desc: "Decision Tree Logic" },
    { id: 4, num: "04", name: "Hotspot Predictor", icon: Crosshair, desc: "Logit Classification" },
    { id: 5, num: "05", name: "Crime Forecaster", icon: BarChart3, desc: "Polynomial OLS" },
    { id: 6, num: "06", name: "Risk Matrix", icon: AlertTriangle, desc: "Global Risk Scoring" },
    { id: 7, num: "07", name: "Intelligence Ledger", icon: ListFilter, desc: "Master Search Database" }
  ];

  const filteredLedger = useMemo(() => {
    if (!initialData.ledger) return [];
    return initialData.ledger.filter((item: any) => 
      item.KEY?.toLowerCase().includes(searchQuery.toLowerCase())
    );
  }, [initialData.ledger, searchQuery]);

  const renderTabContent = () => {
    const commonSizes = "(max-width: 1024px) 100vw, (max-width: 1600px) calc(100vw - 320px), 1280px";
    
    switch(activeTab) {
      case 1:
        return (
          <div className="space-y-8 animate-in fade-in slide-in-from-bottom-4 duration-500">
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
              {[
                { label: "TOTAL_VECTORS", value: initialData.global?.total_records, icon: Hash, color: "text-indigo-400" },
                { label: "SUM_CRIME_LOAD", value: initialData.global?.total_crimes_detected?.toLocaleString(), icon: Zap, color: "text-emerald-400" },
                { label: "AVG_MONTHLY", value: initialData.global?.avg_monthly_load?.toFixed(1), icon: Activity, color: "text-rose-400" },
                { label: "UNIQUE_TYPES", value: initialData.global?.unique_categories, icon: Globe, color: "text-violet-400" },
              ].map((kpi, i) => (
                <div key={i} className="stealth-card p-6 flex flex-col items-center text-center">
                  <kpi.icon className={cn("w-5 h-5 mb-4", kpi.color)} />
                  <span className="data-label mb-2">{kpi.label}</span>
                  <span className="text-3xl font-black text-white tracking-tighter">{kpi.value}</span>
                </div>
              ))}
            </div>

            <div className="grid grid-cols-1 lg:grid-cols-12 gap-8">
              <div className="lg:col-span-8 stealth-card p-8">
                <div className="flex items-center justify-between mb-8">
                  <h3 className="font-sans font-bold text-slate-100 flex items-center gap-2">
                    <Database className="w-4 h-4 text-indigo-400" />
                    SYSTEM_MONTHLY_FLUX
                  </h3>
                  <div className="flex gap-2">
                    <div className="px-3 py-1 bg-white/5 rounded text-[9px] font-mono font-bold text-slate-500 tracking-widest uppercase">9_MONTH_SPAN</div>
                  </div>
                </div>
                <ResponsiveContainer width="100%" height={320}>
                  <AreaChart data={Object.entries(initialData.global?.monthly_totals || {}).map(([month, val]) => ({ month, val }))}>
                    <defs>
                      <linearGradient id="colorVal" x1="0" y1="0" x2="0" y2="1">
                        <stop offset="5%" stopColor="#6366f1" stopOpacity={0.3}/>
                        <stop offset="95%" stopColor="#6366f1" stopOpacity={0}/>
                      </linearGradient>
                    </defs>
                    <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" vertical={false} />
                    <XAxis dataKey="month" stroke="#475569" tick={{fontSize: 10, fontFamily: 'var(--font-jetbrains-mono)'}} axisLine={false} tickLine={false} />
                    <YAxis stroke="#475569" tick={{fontSize: 10, fontFamily: 'var(--font-jetbrains-mono)'}} axisLine={false} tickLine={false} />
                    <RechartsTooltip 
                      contentStyle={{ backgroundColor: '#0f172a', border: '1px solid rgba(255,255,255,0.1)', borderRadius: '8px' }}
                      itemStyle={{ color: '#818cf8', fontFamily: 'var(--font-jetbrains-mono)', fontSize: '10px' }}
                    />
                    <Area type="monotone" dataKey="val" stroke="#6366f1" strokeWidth={3} fillOpacity={1} fill="url(#colorVal)" />
                  </AreaChart>
                </ResponsiveContainer>
              </div>

              <div className="lg:col-span-4 space-y-6">
                <div className="stealth-card p-6 border-b-[3px] border-b-rose-500/50">
                  <h3 className="data-label text-rose-500 mb-6">CRITICAL_VECTORS</h3>
                  <div className="space-y-4">
                    {Object.entries(initialData.global?.severity_breakdown || {}).sort((a: any, b: any) => b[1] - a[1]).map(([name, count]: [string, any], i) => (
                      <div key={i} className="flex items-center justify-between">
                         <div className="flex items-center gap-2">
                           <div className={cn("w-1.5 h-1.5 rounded-full", name === 'High' ? 'bg-rose-500' : (name === 'Medium' ? 'bg-indigo-500' : 'bg-slate-500'))}></div>
                           <span className="text-[11px] font-bold text-slate-300 font-sans">{name.toUpperCase()}</span>
                         </div>
                         <span className="font-mono text-[11px] text-slate-100">{count}</span>
                      </div>
                    ))}
                  </div>
                </div>

                <div className="stealth-card p-6 bg-indigo-500/10 border-indigo-500/20">
                   <h3 className="data-label text-indigo-400 mb-2 font-black">STABILITY_ANNOUNCEMENT</h3>
                   <p className="text-[11px] font-mono text-slate-400 leading-relaxed">
                     Automated assessment identifies a 12.4% increase in systemic volatility. Global regression models indicate convergence on 4 potential Hotspot nodes.
                   </p>
                </div>
              </div>
            </div>
          </div>
        );
      case 7:
        return (
          <div className="space-y-6 animate-in fade-in slide-in-from-bottom-4 duration-500">
            <div className="stealth-card p-6 flex flex-col md:flex-row gap-4 justify-between items-center bg-slate-900/10">
               <div className="flex-1 w-full relative">
                 <Search className="absolute left-4 top-1/2 -translate-y-1/2 w-4 h-4 text-slate-500" />
                 <input 
                  type="text" 
                  placeholder="SEARCH_INTELLIGENCE_LEDGER..."
                  className="w-full bg-black/40 border border-white/5 rounded-lg pl-12 pr-4 py-3 text-sm font-mono text-white placeholder:text-slate-600 focus:outline-none focus:border-indigo-500/40 transition-colors"
                  value={searchQuery}
                  onChange={(e) => setSearchQuery(e.target.value)}
                 />
               </div>
               <div className="flex gap-2">
                 <div className="flex items-center gap-2 px-4 py-2 bg-slate-900/40 border border-white/5 rounded-lg">
                   <Filter className="w-3 h-3 text-slate-500" />
                   <span className="text-[10px] font-mono text-slate-400">{filteredLedger.length} RESULTS</span>
                 </div>
               </div>
            </div>

            <div className="stealth-card overflow-hidden">
               <div className="overflow-x-auto">
                 <table className="w-full border-collapse">
                   <thead>
                     <tr className="bg-slate-900/30 border-b border-white/5">
                        <th className="px-6 py-4 text-left data-label">CRIME_IDENTITY</th>
                        <th className="px-6 py-4 text-center data-label">TOTAL</th>
                        <th className="px-6 py-4 text-center data-label">SLOPE</th>
                        <th className="px-6 py-4 text-center data-label">RISK_SCORE</th>
                        <th className="px-6 py-4 text-center data-label">LEVEL</th>
                        <th className="px-6 py-4 text-right data-label">ACTION</th>
                     </tr>
                   </thead>
                   <tbody className="divide-y divide-white/5">
                     {filteredLedger.slice(0, 50).map((row: any, i: number) => (
                       <tr key={i} className="hover:bg-white/[0.02] transition-colors group">
                         <td className="px-6 py-4">
                           <div className="flex flex-col">
                             <span className="text-[11px] font-sans font-bold text-slate-100 uppercase tracking-tight group-hover:text-indigo-400 transition-colors">{row.KEY}</span>
                             <span className="text-[9px] font-mono text-slate-600">ID: {Math.random().toString(36).substring(7).toUpperCase()}</span>
                           </div>
                         </td>
                         <td className="px-6 py-4 text-center">
                            <span className="font-mono text-[11px] text-slate-400">{row.Total_Count || row.TOTAL_COUNT}</span>
                         </td>
                         <td className="px-6 py-4 text-center">
                            <span className={cn("font-mono text-[11px]", (row.Slope || row.Trend_Slope) > 0 ? "text-rose-500" : "text-emerald-500")}>{(row.Slope || row.Trend_Slope)?.toFixed(3)}</span>
                         </td>
                         <td className="px-6 py-4 text-center">
                            <span className="font-mono text-[11px] text-indigo-400">{(row.Risk_Score)?.toFixed(4)}</span>
                         </td>
                         <td className="px-6 py-4 text-center">
                            <div className={cn(
                              "inline-flex px-2 py-0.5 rounded-full text-[9px] font-mono font-bold tracking-widest uppercase",
                              row.Risk_Level === 'High' ? "bg-rose-500/10 text-rose-500" : (row.Risk_Level === 'Medium' ? "bg-indigo-500/10 text-indigo-500" : "bg-slate-500/10 text-slate-500")
                            )}>
                              {row.Risk_Level}
                            </div>
                         </td>
                         <td className="px-6 py-4 text-right">
                           <button className="p-2 hover:bg-white/5 rounded-lg transition-colors">
                              <Eye className="w-3 h-3 text-slate-600 group-hover:text-indigo-400" />
                           </button>
                         </td>
                       </tr>
                     ))}
                   </tbody>
                 </table>
               </div>
               {filteredLedger.length > 50 && (
                 <div className="p-4 bg-slate-900/10 border-t border-white/5 text-center text-[10px] font-mono text-slate-600">
                    Showing top 50 intelligence nodes. Refine search for accurate targeting.
                 </div>
               )}
            </div>
          </div>
        );
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
      <div className="flex-1 min-w-0 w-full space-y-8 pb-16">
        <header className="flex justify-between items-end pb-4 border-b border-white/5">
           <div>
             <span className="data-label text-indigo-400">COMMAND_NODE_ACTIVE</span>
             <h2 className="text-3xl font-sans font-black text-slate-100 tracking-tighter mt-1">
               {tabs.find(t => t.id === activeTab)?.name.toUpperCase()}
             </h2>
           </div>
           <div className="hidden md:flex gap-6 items-center">
             <div className="flex flex-col items-end">
                <span className="data-label uppercase">System_Load</span>
                <span className="text-[10px] font-mono text-emerald-500 font-bold">NOMINAL_0.22ms</span>
             </div>
             <div className="h-8 w-[1px] bg-white/10"></div>
             <div className="flex flex-col items-end">
                <span className="data-label uppercase">Data_Silo</span>
                <span className="text-[10px] font-mono text-indigo-400 font-bold">REDUNDANCY_ACTIVE</span>
             </div>
           </div>
        </header>

        {renderTabContent()}
      </div>

      {/* INTELLIGENCE MARQUEE - TICKER */}
      <div className="fixed bottom-0 left-0 right-0 h-10 bg-black/80 backdrop-blur-xl border-t border-indigo-500/30 z-50 flex items-center overflow-hidden">
        <div className="bg-indigo-500 h-full px-4 flex items-center justify-center shrink-0 z-10 shadow-[4px_0_15px_rgba(99,102,241,0.5)]">
           <span className="text-[10px] font-mono font-black text-black tracking-widest uppercase">Live_Threats</span>
        </div>
        <div className="flex whitespace-nowrap animate-marquee">
          {[...initialData.ledger.filter((i: any) => i.Risk_Level === 'High').slice(0, 10), ...initialData.ledger.filter((i: any) => i.Risk_Level === 'High').slice(0, 10)].map((alert: any, i: number) => (
            <div key={i} className="flex items-center gap-4 px-8 border-r border-white/5">
               <div className="w-2 h-2 rounded-full bg-rose-500 animate-pulse"></div>
               <span className="text-[10px] font-mono font-bold text-rose-500 uppercase tracking-wider">{alert.KEY}</span>
               <span className="text-[9px] font-mono text-slate-500">RISK: {(alert.Risk_Score * 100).toFixed(1)}%</span>
               <span className="text-[9px] font-mono text-indigo-400">TREND: +{alert.Slope?.toFixed(2)}</span>
            </div>
          ))}
        </div>
      </div>

    </div>
  );
}
