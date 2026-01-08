"use client";

import { useState, useEffect } from "react";
import { api, Item } from "@/lib/api";
import { Search, User, Sparkles, ArrowRight, BarChart3 } from "lucide-react";

export default function Home() {
  const [users, setUsers] = useState<{ user_id: number }[]>([]);
  const [selectedUser, setSelectedUser] = useState<number | null>(null);
  
  const [baselineItems, setBaselineItems] = useState<Item[]>([]);
  const [hybridItems, setHybridItems] = useState<Item[]>([]);
  
  const [query, setQuery] = useState("");
  const [alpha, setAlpha] = useState(0.5);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    api.getUsers().then(setUsers);
  }, []);

  useEffect(() => {
    if (selectedUser !== null) {
      setLoading(true);
      api.getRecommendations(selectedUser)
        .then((data) => {
          setBaselineItems(data.items);
          setHybridItems([]);
          setLoading(false);
        });
    }
  }, [selectedUser]);

  const handleRerank = async () => {
    if (selectedUser === null || !query) return;
    setLoading(true);
    try {
      const data = await api.getHybridRecommendations(selectedUser, query, alpha);
      setHybridItems(data.items);
    } catch (e) {
      console.error(e);
    }
    setLoading(false);
  };

  return (
    <div className="min-h-screen bg-slate-50 text-slate-900 font-sans p-8">
      <div className="max-w-6xl mx-auto space-y-8">
        
        <header className="flex items-center justify-between border-b pb-6 border-slate-200">
          <div>
            <h1 className="text-3xl font-bold tracking-tight text-slate-900">
              Graph + NLP Recommender
            </h1>
            <p className="text-slate-500 mt-2">
              LightGCN (Collaborative Filtering) + SBERT (Semantic Reranking)
            </p>
          </div>
          <div className="flex items-center gap-2 bg-white px-4 py-2 rounded-lg border shadow-sm">
            <User size={20} className="text-slate-400" />
            <select 
              className="bg-transparent outline-none font-medium"
              onChange={(e) => setSelectedUser(Number(e.target.value))}
              value={selectedUser ?? ""}
            >
              <option value="" disabled>Select a User ID</option>
              {users.map(u => (
                <option key={u.user_id} value={u.user_id}>User {u.user_id}</option>
              ))}
            </select>
          </div>
        </header>

        <div className="bg-white p-6 rounded-xl shadow-sm border border-slate-200 space-y-4">
          <div className="flex gap-4 items-end">
            <div className="flex-1 space-y-2">
              <label className="text-sm font-semibold text-slate-700 flex items-center gap-2">
                <Search size={16} /> Natural Language Query
              </label>
              <input 
                type="text" 
                placeholder="e.g. '90s sci-fi movies about space'"
                className="w-full px-4 py-2 rounded-lg border border-slate-300 focus:ring-2 focus:ring-blue-500 outline-none transition"
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                onKeyDown={(e) => e.key === 'Enter' && handleRerank()}
              />
            </div>
            
            <div className="w-48 space-y-2">
              <label className="text-sm font-semibold text-slate-700 flex justify-between">
                <span>Hybrid Weight (α)</span>
                <span className="text-blue-600">{alpha}</span>
              </label>
              <input 
                type="range" min="0" max="1" step="0.1"
                value={alpha}
                onChange={(e) => setAlpha(parseFloat(e.target.value))}
                className="w-full accent-blue-600"
              />
              <div className="flex justify-between text-xs text-slate-400 px-1">
                <span>NLP</span>
                <span>Graph</span>
              </div>
            </div>

            <button 
              onClick={handleRerank}
              disabled={loading || !selectedUser}
              className="px-6 py-2.5 bg-blue-600 hover:bg-blue-700 text-white font-medium rounded-lg flex items-center gap-2 transition disabled:opacity-50"
            >
              <Sparkles size={18} /> Rerank
            </button>
          </div>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
          
          <section>
            <h2 className="text-xl font-bold flex items-center gap-2 mb-4 text-slate-700">
              <BarChart3 size={20} /> Baseline (Graph Only)
            </h2>
            <div className="space-y-3">
              {baselineItems.length === 0 ? (
                <div className="text-slate-400 italic p-4 text-center border-2 border-dashed rounded-lg">
                  Select a user to see recommendations
                </div>
              ) : (
                baselineItems.map((item) => (
                  <Card key={item.id} item={item} type="baseline" />
                ))
              )}
            </div>
          </section>

          <section>
            <h2 className="text-xl font-bold flex items-center gap-2 mb-4 text-slate-700">
              <Sparkles size={20} className="text-amber-500" /> Hybrid Reranked
            </h2>
            <div className="space-y-3">
              {hybridItems.length === 0 ? (
                <div className="text-slate-400 italic p-4 text-center border-2 border-dashed rounded-lg">
                  Enter a query to rerank results
                </div>
              ) : (
                hybridItems.map((item) => (
                  <Card key={item.id} item={item} type="hybrid" />
                ))
              )}
            </div>
          </section>

        </div>
      </div>
    </div>
  );
}

function Card({ item, type }: { item: Item, type: 'baseline' | 'hybrid' }) {
  return (
    <div className={`p-4 rounded-lg border shadow-sm transition hover:shadow-md ${
      type === 'hybrid' ? 'bg-white border-blue-100' : 'bg-slate-50 border-slate-200'
    }`}>
      <div className="flex justify-between items-start">
        <div>
          <h3 className="font-bold text-slate-900">{item.title}</h3>
          <p className="text-sm text-slate-500 mt-0.5">{item.genres.split('|').join(', ')}</p>
        </div>
        <span className={`text-xs font-mono px-2 py-1 rounded ${
          type === 'hybrid' ? 'bg-blue-50 text-blue-700' : 'bg-slate-200 text-slate-600'
        }`}>
          {item.score.toFixed(3)}
        </span>
      </div>
      {item.explanation && (
        <div className="mt-3 text-xs bg-slate-100 p-2 rounded text-slate-600 flex items-center gap-2">
          <ArrowRight size={12} /> {item.explanation}
        </div>
      )}
    </div>
  );
}