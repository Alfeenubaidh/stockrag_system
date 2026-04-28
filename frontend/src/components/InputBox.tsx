import React from 'react';
import TextareaAutosize from 'react-textarea-autosize';
import { Send, Paperclip, Loader2 } from 'lucide-react';
import { useChatStore } from '../store/chatStore';

export const InputBox: React.FC = () => {
  const [input, setInput] = React.useState('');
  const { sendMessage, isLoading } = useChatStore();

  const handleSend = async () => {
    if (!input.trim() || isLoading) return;
    const msg = input;
    setInput('');
    await sendMessage(msg);
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  return (
    <div className="sticky bottom-0 w-full pt-2 pb-8 bg-gradient-to-t from-white via-white to-transparent">
      <div className="max-w-[800px] mx-auto px-4">
        <div className="relative group">
          <div className="absolute inset-0 bg-[#c2410c]/5 rounded-2xl blur-xl opacity-0 group-focus-within:opacity-100 transition-opacity pointer-events-none" />
          
          <div className="relative bg-white border border-slate-200 rounded-2xl shadow-xl shadow-slate-100/50 p-2 flex flex-col gap-2 transition-all group-focus-within:border-orange-200 group-focus-within:ring-4 group-focus-within:ring-orange-50">
            <div className="flex items-end gap-2 p-1">
              <button className="p-2 text-slate-400 hover:text-slate-600 hover:bg-slate-50 rounded-xl transition-all">
                <Paperclip className="w-5 h-5" />
              </button>
              
              <TextareaAutosize
                value={input}
                onChange={(e) => setInput(e.target.value)}
                onKeyDown={handleKeyDown}
                placeholder="Ask anything about your documents..."
                minRows={1}
                maxRows={8}
                className="flex-1 resize-none bg-transparent border-none focus:ring-0 p-2 text-slate-800 placeholder:text-slate-400 text-sm leading-relaxed"
              />

              <button
                onClick={handleSend}
                disabled={!input.trim() || isLoading}
                className={`p-2.5 rounded-xl transition-all shadow-lg ${
                  input.trim() && !isLoading 
                    ? 'bg-orange-600 text-white shadow-orange-200 hover:bg-orange-700' 
                    : 'bg-slate-100 text-slate-300 shadow-none cursor-not-allowed'
                }`}
              >
                {isLoading ? (
                  <Loader2 className="w-5 h-5 animate-spin" />
                ) : (
                  <Send className="w-5 h-5" />
                )}
              </button>
            </div>
            
            <div className="flex items-center justify-between px-3 py-1.5 border-t border-slate-50">
                <div className="text-[10px] text-slate-400">
                  <span className="font-semibold">Shift + Enter</span> for new line
                </div>
                <div className="text-[10px] text-slate-400">
                  Gemini 1.5 Flash
                </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};
