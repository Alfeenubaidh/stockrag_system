import { create } from 'zustand';

export interface Citation {
  chunk: string;
  source: string;
  relevance: number;
}

export interface Message {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  citations?: Citation[];
  suggestions?: string[];
}

export interface Conversation {
  id: string;
  title: string;
  preview: string;
  createdAt: string;
}

interface ChatState {
  conversations: Conversation[];
  activeChatId: string | null;
  messages: Message[];
  isLoading: boolean;

  fetchConversations: () => Promise<void>;
  setActiveChat: (id: string) => void;
  sendMessage: (message: string) => Promise<void>;
  createNewChat: () => void;
}

export const useChatStore = create<ChatState>((set, get) => ({
  conversations: [],
  activeChatId: null,
  messages: [],
  isLoading: false,

  fetchConversations: async () => {
    // No backend endpoint for conversation history
  },

  setActiveChat: (id: string) => {
    set({ activeChatId: id, messages: [] });
  },

  createNewChat: () => {
    set({ activeChatId: Date.now().toString(), messages: [] });
  },

  sendMessage: async (content: string) => {
    const { activeChatId, messages } = get();
    const chatId = activeChatId || Date.now().toString();

    const userMsg: Message = { id: Date.now().toString(), role: 'user', content };
    const assistantId = (Date.now() + 1).toString();
    const assistantMsg: Message = { id: assistantId, role: 'assistant', content: '' };

    set({
      messages: [...messages, userMsg, assistantMsg],
      isLoading: true,
      activeChatId: chatId,
    });

    try {
      const res = await fetch('/api/query/stream', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ question: content, ticker: null, top_k: 5 }),
      });

      if (!res.ok || !res.body) {
        throw new Error(`HTTP ${res.status}`);
      }

      const reader = res.body.getReader();
      const decoder = new TextDecoder();
      let buffer = '';

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split('\n');
        buffer = lines.pop() ?? '';

        for (const line of lines) {
          if (!line.startsWith('data:')) continue;
          const token = line.replace(/^data: /, '').replace(/\\n/g, '\n');
          if (token === '[DONE]' || token === '[ERROR]') {
            set((state) => ({ messages: state.messages, isLoading: false }));
            return;
          }
          set((state) => ({
            messages: state.messages.map((m) =>
              m.id === assistantId ? { ...m, content: m.content + token } : m
            ),
          }));
        }
      }
    } catch (err) {
      set((state) => ({
        messages: state.messages.map((m) =>
          m.id === assistantId ? { ...m, content: 'Backend unavailable' } : m
        ),
      }));
    } finally {
      set({ isLoading: false });
    }
  },
}));
