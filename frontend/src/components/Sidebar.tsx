import React from 'react';
import { useChatStore } from '../store/chatStore';
import { MessageSquare, Plus, Settings, HelpCircle, Hash } from 'lucide-react';
import { clsx, type ClassValue } from 'clsx';
import { twMerge } from 'tailwind-merge';

function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}

export const Sidebar: React.FC = () => {
  const { conversations, activeChatId, setActiveChat, createNewChat } = useChatStore();

  return (
    <aside className="w-[260px] bg-[#f1f5f9] border-r border-slate-200 flex flex-col h-screen fixed left-0 top-0 z-20">
      {/* Title */}
      <div className="p-4 mb-2">
        <h1 className="text-xl font-semibold text-slate-800 flex items-center gap-2">
          <Hash className="w-5 h-5 text-orange-600" />
          RAG Interface
        </h1>
      </div>

      {/* New Chat Button */}
      <div className="px-3 mb-6">
        <button
          onClick={createNewChat}
          className="w-full flex items-center gap-2 px-3 py-2 bg-white border border-slate-200 rounded-lg hover:border-slate-300 transition-colors text-sm font-medium text-slate-700 shadow-sm"
        >
          <Plus className="w-4 h-4 text-orange-600" />
          New Chat
        </button>
      </div>

      {/* Recent Chats */}
      <div className="flex-1 overflow-y-auto px-3">
        <div className="text-xs font-semibold text-slate-400 uppercase tracking-wider mb-2 px-2">
          Recent Chats
        </div>
        <div className="space-y-1">
          {conversations.map((chat) => (
            <button
              key={chat.id}
              onClick={() => setActiveChat(chat.id)}
              className={cn(
                "w-full text-left px-3 py-2 rounded-lg text-sm transition-colors flex flex-col gap-0.5",
                activeChatId === chat.id 
                  ? "bg-white text-slate-900 shadow-sm border border-slate-200" 
                  : "text-slate-600 hover:bg-slate-200/50"
              )}
            >
              <span className="font-medium truncate">{chat.title}</span>
              <span className="text-xs text-slate-400 truncate">{chat.preview}</span>
            </button>
          ))}
          {conversations.length === 0 && (
             <div className="text-xs text-slate-400 px-3 py-2 italic">
               No chats yet.
             </div>
          )}
        </div>
      </div>

      {/* Bottom Actions */}
      <div className="p-3 border-t border-slate-200 flex flex-col gap-1">
        <button className="flex items-center gap-3 px-3 py-2 rounded-lg hover:bg-slate-200/50 transition-colors text-sm text-slate-600">
          <Settings className="w-4 h-4" />
          Settings
        </button>
        <button className="flex items-center gap-3 px-3 py-2 rounded-lg hover:bg-slate-200/50 transition-colors text-sm text-slate-600">
          <HelpCircle className="w-4 h-4" />
          Help
        </button>
      </div>
    </aside>
  );
};
