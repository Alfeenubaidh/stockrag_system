// Matches the interfaces used in chatStore.ts and services/api.ts

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

// API request / response shapes (mirrors backend QueryRequest / QueryResponse)

export interface QueryRequest {
  question: string;
  ticker?: string | null;
  top_k?: number;
}

export interface QueryResponse {
  answer: string;
  citations: string[];
  latency_ms: number;
}
