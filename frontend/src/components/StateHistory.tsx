import React, { useEffect, useState } from 'react';
// If using MUI:
// import Accordion from '@mui/material/Accordion';
// import AccordionSummary from '@mui/material/AccordionSummary';
// import AccordionDetails from '@mui/material/AccordionDetails';
// import ExpandMoreIcon from '@mui/icons-material/ExpandMore';

const StateHistory: React.FC = () => {
  const [history, setHistory] = useState<any[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [expanded, setExpanded] = useState<number | null>(null);
  const [refreshKey, setRefreshKey] = useState(0);

  const fetchHistory = async () => {
    setLoading(true);
    try {
      const res = await fetch('http://localhost:8000/api/state-history');
      if (!res.ok) throw new Error('Failed to fetch state history');
      const data = await res.json();
      setHistory(data.history || []);
      setError(null);
    } catch (err: any) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const clearHistory = async () => {
    try {
      const res = await fetch('http://localhost:8000/api/state-history/clear', {
        method: 'POST'
      });
      if (!res.ok) throw new Error('Failed to clear state history');
      setHistory([]);
      setError(null);
    } catch (err: any) {
      setError(err.message);
    }
  };

  useEffect(() => {
    fetchHistory();
  }, [refreshKey]);

  if (loading) return <div style={{ color: '#888', textAlign: 'center', width: '100%' }}>Loading state history...</div>;
  if (error) return <div style={{ color: 'red', textAlign: 'center', width: '100%' }}>Error: {error}</div>;

  return (
    <div style={{ width: '100%', maxWidth: '800px', margin: '0 auto' }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1rem' }}>
        <h3 style={{ margin: 0, color: '#333' }}>Agent State History</h3>
        <div style={{ display: 'flex', gap: '0.5rem' }}>
          <button 
            onClick={() => setRefreshKey(k => k + 1)}
            style={{ 
              padding: '0.5rem 1rem', 
              background: '#007bff', 
              color: 'white', 
              border: 'none', 
              borderRadius: '4px', 
              cursor: 'pointer' 
            }}
          >
            Refresh
          </button>
          <button 
            onClick={clearHistory}
            style={{ 
              padding: '0.5rem 1rem', 
              background: '#dc3545', 
              color: 'white', 
              border: 'none', 
              borderRadius: '4px', 
              cursor: 'pointer' 
            }}
          >
            Clear
          </button>
        </div>
      </div>

      {!history.length ? (
        <div style={{ color: '#888', textAlign: 'center', width: '100%' }}>No state history yet. Try chatting with the agent!</div>
      ) : (
        <div style={{ maxHeight: 500, overflowY: 'auto', width: '100%' }}>
          {history.map((state, idx) => {
            const summary = `Status: ${state.status || 'unknown'} | Messages: ${state.messages ? state.messages.length : 0}`;
            return (
              <div key={idx} style={{ border: '1px solid #ddd', borderRadius: 6, margin: '8px 0', background: '#fafbfc' }}>
                <div
                  style={{ cursor: 'pointer', padding: '8px 12px', fontWeight: 500, background: '#f0f2f5' }}
                  onClick={() => setExpanded(expanded === idx ? null : idx)}
                >
                  {summary} {expanded === idx ? '▲' : '▼'}
                </div>
                {expanded === idx && (
                  <pre style={{ margin: 0, padding: 12, fontSize: 13, background: '#fff', borderTop: '1px solid #eee', overflowX: 'auto' }}>
                    {JSON.stringify(state, null, 2)}
                  </pre>
                )}
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
};

export default StateHistory; 