import React from 'react';
import { useChatStore } from '../store/chatStore';
import { MessageBubble } from './MessageBubble';
import { InputBox } from './InputBox';
import { Loader2 } from 'lucide-react';

export const ChatWindow: React.FC = () => {
  const { messages, isLoading, sendMessage } = useChatStore();
  const scrollRef = React.useRef<HTMLDivElement>(null);

  React.useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTo({
        top: scrollRef.current.scrollHeight,
        behavior: 'smooth',
      });
    }
  }, [messages]);

  const handleSuggestionClick = (suggestion: string) => {
    sendMessage(suggestion);
  };

  return (
    <main className="flex-1 flex flex-col h-screen ml-[260px] relative overflow-hidden bg-white">
      <div 
        ref={scrollRef}
        className="flex-1 overflow-y-auto px-4 pb-4 pt-6"
      >
        <div className="max-w-[800px] mx-auto w-full">
          {messages.length === 0 ? (
            <div className="h-full flex flex-col items-center justify-center pt-24 text-center">
              <div className="w-16 h-16 bg-orange-100 text-orange-600 rounded-2xl flex items-center justify-center mb-6 shadow-sm">
                <img src="https://upload.wikimedia.org/wikipedia/commons/e/e0/Google_GenAI_logo.svg" className="w-10 h-10" alt="Logo" />
              </div>
              <h2 className="text-3xl font-bold text-slate-800 mb-4 tracking-tight">
                How can I help you today?
              </h2>
              <p className="text-slate-500 max-w-md mx-auto leading-relaxed">
                Upload your documents and start a conversation. I can analyze text, answer questions, and provide citations.
              </p>
              
              <div className="grid grid-cols-2 gap-4 mt-12 w-full max-w-2xl px-4">
                <button 
                  onClick={() => handleSuggestionClick("What are the key findings in the latest whitepaper?")}
                  className="p-4 border border-slate-100 rounded-xl bg-slate-50/50 hover:bg-white hover:border-orange-200 transition-all text-left group shadow-sm"
                >
                  <p className="text-sm font-medium text-slate-700 mb-1 group-hover:text-orange-700">Explore Data</p>
                  <p className="text-xs text-slate-500 leading-relaxed">"What are the key findings in the latest whitepaper?"</p>
                </button>
                <button 
                  onClick={() => handleSuggestionClick("Summarize the legal requirements for small firms.")}
                  className="p-4 border border-slate-100 rounded-xl bg-slate-50/50 hover:bg-white hover:border-orange-200 transition-all text-left group shadow-sm"
                >
                  <p className="text-sm font-medium text-slate-700 mb-1 group-hover:text-orange-700">Summarize Docs</p>
                  <p className="text-xs text-slate-500 leading-relaxed">"Summarize the legal requirements for small firms."</p>
                </button>
              </div>
            </div>
          ) : (
            <>
              {messages.map((msg) => (
                <MessageBubble 
                  key={msg.id} 
                  message={msg} 
                  onSuggestionClick={handleSuggestionClick} 
                />
              ))}
              {isLoading && (
                <div className="flex gap-4 items-start mb-8">
                  <div className="w-8 h-8 rounded-full bg-slate-50 border border-slate-100 flex items-center justify-center">
                    <Loader2 className="w-4 h-4 text-slate-400 animate-spin" />
                  </div>
                  <div className="flex gap-1.5 pt-3">
                    <div className="w-1.5 h-1.5 bg-slate-300 rounded-full animate-bounce [animation-delay:-0.3s]"></div>
                    <div className="w-1.5 h-1.5 bg-slate-300 rounded-full animate-bounce [animation-delay:-0.15s]"></div>
                    <div className="w-1.5 h-1.5 bg-slate-300 rounded-full animate-bounce"></div>
                  </div>
                </div>
              )}
            </>
          )}
        </div>
      </div>
      
      <InputBox />
    </main>
  );
};
