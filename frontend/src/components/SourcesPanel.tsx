import React from 'react';
import { ChevronDown, ChevronRight, FileText, ExternalLink } from 'lucide-react';
import { Citation } from '../store/chatStore';

interface SourcesPanelProps {
  citations: Citation[];
}

export const SourcesPanel: React.FC<SourcesPanelProps> = ({ citations }) => {
  const [isExpanded, setIsExpanded] = React.useState(false);

  return (
    <div className="mt-4 pt-4 border-t border-slate-100">
      <button
        onClick={() => setIsExpanded(!isExpanded)}
        className="flex items-center gap-2 text-xs font-medium text-slate-500 hover:text-orange-600 transition-colors group"
      >
        <span className="bg-slate-100 p-1 rounded group-hover:bg-orange-100 transition-colors">
          {isExpanded ? <ChevronDown className="w-3 h-3" /> : <ChevronRight className="w-3 h-3" />}
        </span>
        View {citations.length} Citations
      </button>

      {isExpanded && (
        <div className="mt-3 grid grid-cols-1 gap-3 animate-in fade-in slide-in-from-top-2 duration-300">
          {citations.map((citation, idx) => (
            <div key={idx} className="bg-slate-50 p-3 rounded-lg border border-slate-100">
              <div className="flex items-center justify-between mb-2">
                <div className="flex items-center gap-2 text-xs font-semibold text-slate-700">
                  <FileText className="w-3.5 h-3.5 text-blue-500" />
                  {citation.source}
                </div>
                <div className="text-[10px] font-bold text-green-600 bg-green-50 px-1.5 rounded border border-green-100">
                  {Math.round(citation.relevance * 100)}% Relevant
                </div>
              </div>
              <p className="text-xs text-slate-500 italic leading-relaxed">
                "{citation.chunk}"
              </p>
              <button className="mt-2 flex items-center gap-1 text-[10px] text-blue-600 hover:underline">
                Open document <ExternalLink className="w-2.5 h-2.5" />
              </button>
            </div>
          ))}
        </div>
      )}
    </div>
  );
};
