import { getPythonOutputs } from '@/lib/data';
import Dashboard from '@/components/Dashboard';

// Trigger HMR compilation
export default async function Home() {
  const data = await getPythonOutputs();
  
  return (
    <main className="min-h-screen p-4 md:p-8">
      <div className="max-w-7xl mx-auto space-y-8">
        <header className="space-y-3 pb-8 border-b border-cyan-900/30">
          <h1 className="text-4xl md:text-6xl font-black uppercase tracking-tighter text-transparent bg-clip-text bg-gradient-to-r from-cyan-400 to-indigo-400 neon-text">
            CRIME<span className="font-light opacity-50">WATCH</span>
          </h1>
          <p className="text-cyan-400/70 font-mono text-sm tracking-widest uppercase flex items-center gap-2">
            <span className="h-2 w-2 rounded-full bg-red-500 animate-pulse"></span>
            Global Safety Risk Matrix // Online
          </p>
        </header>

        {data.error ? (
          <div className="p-8 border border-red-500/50 bg-red-950/20 text-red-400 font-mono rounded glass-panel">
            [SYSTEM FAILURE]: {data.error}
          </div>
        ) : (
          <Dashboard initialData={data} />
        )}
      </div>
    </main>
  );
}
