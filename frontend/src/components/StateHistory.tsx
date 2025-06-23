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

  useEffect(() => {
    setLoading(true);
    fetch('http://localhost:8000/api/state-history')
      .then(res => {
        if (!res.ok) throw new Error('Failed to fetch state history');
        return res.json();
      })
      .then(data => {
        setHistory(data.history || []);
        setLoading(false);
      })
      .catch(err => {
        setError(err.message);
        setLoading(false);
      });
  }, []);

  if (loading) return <div style={{ color: '#888', textAlign: 'center', width: '100%' }}>Loading state history...</div>;
  if (error) return <div style={{ color: 'red', textAlign: 'center', width: '100%' }}>Error: {error}</div>;

  if (!history.length) return <div style={{ color: '#888', textAlign: 'center', width: '100%' }}>No state history yet.</div>;

  // If MUI is available, use Accordion. Otherwise, fallback to simple HTML.
  return (
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
  );
};

export default StateHistory; 