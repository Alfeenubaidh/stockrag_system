import React from 'react';
import ReactMarkdown from 'react-markdown';
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter';
import { vscDarkPlus } from 'react-syntax-highlighter/dist/esm/styles/prism';
import { Copy, Check } from 'lucide-react';
import { Message } from '../store/chatStore';
import { SourcesPanel } from './SourcesPanel';
import { motion } from 'motion/react';

interface MessageBubbleProps {
  message: Message;
  onSuggestionClick: (suggestion: string) => void;
}

export const MessageBubble: React.FC<MessageBubbleProps> = ({ message, onSuggestionClick }) => {
  const [copied, setCopied] = React.useState(false);

  const handleCopy = (code: string) => {
    navigator.clipboard.writeText(code);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  const isAssistant = message.role === 'assistant';

  return (
    <div className={`flex flex-col w-full mb-8 ${isAssistant ? '' : 'items-end'}`}>
      <div className={isAssistant ? 'max-w-[800px] w-full mx-auto break-words overflow-wrap-anywhere' : 'max-w-[90%] md:max-w-[85%] bg-slate-100 p-4 rounded-2xl shadow-sm text-slate-800'}>
        <div className="markdown-body prose prose-slate max-w-none break-words">
          <ReactMarkdown
            components={{
              code({ node, inline, className, children, ...props }: any) {
                const match = /language-(\w+)/.exec(className || '');
                return !inline && match ? (
                  <div className="relative group my-4 rounded-xl overflow-hidden border border-slate-700">
                    <div className="absolute right-3 top-3 z-10">
                      <button
                        onClick={() => handleCopy(String(children).replace(/\n$/, ''))}
                        className="p-1.5 rounded-md bg-slate-800/50 hover:bg-slate-800 border border-slate-600 text-slate-400 hover:text-white transition-all opacity-0 group-hover:opacity-100"
                      >
                        {copied ? <Check className="w-4 h-4 text-green-500" /> : <Copy className="w-4 h-4" />}
                      </button>
                    </div>
                    <SyntaxHighlighter
                      style={vscDarkPlus}
                      language={match[1]}
                      PreTag="div"
                      customStyle={{
                        margin: 0,
                        padding: '1.25rem',
                        fontSize: '0.9rem',
                        background: '#1e293b',
                      }}
                      {...props}
                    >
                      {String(children).replace(/\n$/, '')}
                    </SyntaxHighlighter>
                  </div>
                ) : (
                  <code className={className} {...props}>
                    {children}
                  </code>
                );
              },
            }}
          >
            {message.content}
          </ReactMarkdown>
        </div>

        {/* Citations Panel */}
        {isAssistant && message.citations && message.citations.length > 0 && (
          <SourcesPanel citations={message.citations} />
        )}
      </div>

      {/* Suggestions */}
      {isAssistant && message.suggestions && message.suggestions.length > 0 && (
        <div className="flex flex-wrap gap-2 mt-4">
          {message.suggestions.map((suggestion, idx) => (
            <motion.button
              key={idx}
              whileHover={{ scale: 1.02 }}
              whileTap={{ scale: 0.98 }}
              onClick={() => onSuggestionClick(suggestion)}
              className="text-xs px-3 py-1.5 rounded-full border border-slate-200 bg-white hover:border-orange-200 hover:bg-orange-50 text-slate-600 hover:text-orange-700 transition-all shadow-sm"
            >
              {suggestion}
            </motion.button>
          ))}
        </div>
      )}
    </div>
  );
};
