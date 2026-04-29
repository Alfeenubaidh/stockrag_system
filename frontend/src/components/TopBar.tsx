import React, { Fragment, useCallback, useEffect, useState } from 'react';
import { Code, Database, LayoutGrid, Loader2, ChevronDown, User, X } from 'lucide-react';

interface KbDocument {
  ticker: string;
  doc_count: number;
  sections: string[];
  last_updated: string;
}

function KnowledgeBaseModal({ onClose }: { onClose: () => void }) {
  const [docs, setDocs] = useState<KbDocument[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const fetchDocs = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const res = await fetch(`${import.meta.env.VITE_API_URL}/documents`);
      if (res.status === 503) throw new Error('Vector store offline — start Qdrant and retry');
      if (!res.ok) throw new Error(`Server error (HTTP ${res.status})`);
      setDocs(await res.json());
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Unknown error');
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => { fetchDocs(); }, [fetchDocs]);

  useEffect(() => {
    const handler = (e: KeyboardEvent) => { if (e.key === 'Escape') onClose(); };
    document.addEventListener('keydown', handler);
    return () => document.removeEventListener('keydown', handler);
  }, [onClose]);

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/40 backdrop-blur-sm"
      onClick={(e) => { if (e.target === e.currentTarget) onClose(); }}
    >
      <div className="bg-white rounded-2xl shadow-2xl w-full max-w-3xl mx-4 flex flex-col max-h-[80vh]">

        {/* Header */}
        <div className="flex items-center justify-between px-6 py-4 border-b border-slate-100">
          <div className="flex items-center gap-2">
            <Database className="w-5 h-5 text-orange-600" />
            <h2 className="text-base font-semibold text-slate-900">Knowledge Base</h2>
          </div>
          <button
            onClick={onClose}
            className="p-1.5 rounded-lg text-slate-400 hover:text-slate-700 hover:bg-slate-100 transition-colors"
            aria-label="Close"
          >
            <X className="w-4 h-4" />
          </button>
        </div>

        {/* Body */}
        <div className="overflow-auto flex-1 px-6 py-4">
          {loading && (
            <div className="flex items-center justify-center py-16 gap-3 text-slate-500">
              <Loader2 className="w-5 h-5 animate-spin" />
              <span className="text-sm">Loading documents…</span>
            </div>
          )}

          {!loading && error && (
            <div className="flex flex-col items-center py-16 gap-2">
              <p className="text-sm font-medium text-red-500">Failed to load: {error}</p>
              <button onClick={fetchDocs} className="text-xs text-orange-600 hover:underline mt-1">
                Retry
              </button>
            </div>
          )}

          {!loading && !error && docs.length === 0 && (
            <div className="flex items-center justify-center py-16">
              <p className="text-sm text-slate-400">No documents ingested yet.</p>
            </div>
          )}

          {!loading && !error && docs.length > 0 && (
            <table className="w-full text-sm border-collapse">
              <thead>
                <tr className="text-left text-xs font-semibold text-slate-500 uppercase tracking-wide">
                  <th className="pb-3 pr-6 w-20">Ticker</th>
                  <th className="pb-3 pr-6 w-28">Chunks</th>
                  <th className="pb-3 pr-6">Sections</th>
                  <th className="pb-3 w-32 text-right">Last Updated</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-100">
                {docs.map((doc) => (
                  <tr key={doc.ticker} className="hover:bg-slate-50 transition-colors">
                    <td className="py-3 pr-6 font-semibold text-slate-900">{doc.ticker}</td>
                    <td className="py-3 pr-6 text-slate-600 tabular-nums">
                      {doc.doc_count.toLocaleString()}
                    </td>
                    <td className="py-3 pr-6 text-slate-500 max-w-xs truncate">
                      {doc.sections.length === 0
                        ? <span className="text-slate-300 italic">—</span>
                        : doc.sections.slice(0, 4).join(', ') +
                          (doc.sections.length > 4 ? ` +${doc.sections.length - 4} more` : '')}
                    </td>
                    <td className="py-3 text-slate-500 tabular-nums text-right whitespace-nowrap">
                      {doc.last_updated}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
        </div>

        {/* Footer */}
        {!loading && !error && docs.length > 0 && (
          <div className="px-6 py-3 border-t border-slate-100 text-xs text-slate-400">
            {docs.length} ticker{docs.length !== 1 ? 's' : ''} ·{' '}
            {docs.reduce((s, d) => s + d.doc_count, 0).toLocaleString()} chunks total
          </div>
        )}

      </div>
    </div>
  );
}

export const TopBar: React.FC = () => {
  const [kbOpen, setKbOpen] = useState(false);

  return (
    <Fragment>
      <header className="h-14 border-b border-slate-200 bg-white/80 backdrop-blur-md sticky top-0 z-10 flex items-center justify-between px-6">
        {/* Tabs */}
        <nav className="flex items-center h-full gap-8">
          <button className="flex items-center gap-2 text-sm font-medium text-slate-900 border-b-2 border-orange-600 h-full px-1">
            <LayoutGrid className="w-4 h-4" />
            Models
          </button>

          <button
            className="flex items-center gap-2 text-sm font-medium text-slate-500 hover:text-slate-700 h-full px-1 transition-colors"
            onClick={() => setKbOpen(true)}
          >
            <Database className="w-4 h-4" />
            Knowledge Base
          </button>

          <button
            className="flex items-center gap-2 text-sm font-medium text-slate-500 hover:text-slate-700 h-full px-1 transition-colors"
            onClick={() => window.open(`${import.meta.env.VITE_API_URL}/docs`, '_blank')}
          >
            <Code className="w-4 h-4" />
            API
          </button>
        </nav>

        {/* Right Side */}
        <div className="flex items-center gap-4">
          <div className="flex items-center gap-2 hover:bg-slate-50 p-1.5 rounded-lg cursor-pointer transition-colors">
            <div className="w-8 h-8 rounded-full bg-orange-100 flex items-center justify-center text-orange-700 overflow-hidden">
              <User className="w-5 h-5" />
            </div>
            <ChevronDown className="w-4 h-4 text-slate-400" />
          </div>
        </div>
      </header>

      {kbOpen && <KnowledgeBaseModal onClose={() => setKbOpen(false)} />}
    </Fragment>
  );
};
