import React, { useState, useCallback } from 'react';
import Chat from './components/Chat';
import KnowledgeGraph from './components/KnowledgeGraph';
import StateHistory from './components/StateHistory';
import './App.css';

export interface Message {
  role: 'user' | 'assistant';
  content: string;
  timestamp: Date;
}

const App: React.FC = () => {
  const [activeTab, setActiveTab] = useState<'chat' | 'kg' | 'history'>('chat');
  const [messages, setMessages] = useState<Message[]>([]);
  const [kgRefresh, setKgRefresh] = useState(0);

  // Callback to trigger KG refresh
  const triggerKgRefresh = useCallback(() => setKgRefresh(r => r + 1), []);

  return (
    <div className="App">
      <header className="App-header">
        <div className="container">
          <h1>
            <span className="text-accent">AI</span> Personal Trainer
          </h1>
        </div>
      </header>
      <div className="centered-content">
        <div className="main-layout">
          <aside className="sidebar-tabs">
            <button
              className={activeTab === 'chat' ? 'side-tab active' : 'side-tab'}
              onClick={() => setActiveTab('chat')}
              aria-label="Agent Chat"
            >
              <span role="img" aria-label="Chat">💬</span> Agent Chat
            </button>
            <button
              className={activeTab === 'kg' ? 'side-tab active' : 'side-tab'}
              onClick={() => setActiveTab('kg')}
              aria-label="Knowledge Graph"
            >
              <span role="img" aria-label="Graph">🧠</span> Knowledge Graph
            </button>
            <button
              className={activeTab === 'history' ? 'side-tab active' : 'side-tab'}
              onClick={() => setActiveTab('history')}
              aria-label="State History"
            >
              <span role="img" aria-label="History">📜</span> State History
            </button>
          </aside>
          <main className="App-main">
            {activeTab === 'chat' ? (
              <Chat messages={messages} setMessages={setMessages} onPreferenceAdded={triggerKgRefresh} />
            ) : activeTab === 'kg' ? (
              <div className="kg-graph-area">
                <KnowledgeGraph refresh={kgRefresh} />
              </div>
            ) : (
              <div className="state-history-area">
                <StateHistory />
              </div>
            )}
          </main>
        </div>
      </div>
    </div>
  );
};

export default App; 