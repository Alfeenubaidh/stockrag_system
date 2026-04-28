import React from 'react';
import { Sidebar } from './components/Sidebar';
import { TopBar } from './components/TopBar';
import { ChatWindow } from './components/ChatWindow';
import { useChatStore } from './store/chatStore';

export default function App() {
  const { fetchConversations } = useChatStore();

  React.useEffect(() => {
    fetchConversations();
  }, [fetchConversations]);

  return (
    <div className="flex min-h-screen bg-[#f8fafc]">
      <Sidebar />
      <div className="flex-1 flex flex-col overflow-hidden">
        <div className="ml-[260px]">
           <TopBar />
        </div>
        <ChatWindow />
      </div>
    </div>
  );
}
