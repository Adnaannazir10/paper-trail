import React, { useState, useEffect } from 'react';
import ReactMarkdown from 'react-markdown';
import './App.css';

interface JournalListItem {
  journal_name: string;
}

interface JournalListResponse {
  journals: JournalListItem[];
  total_count: number;
}

function App() {
  const [query, setQuery] = useState('');
  const [journal, setJournal] = useState('');
  const [response, setResponse] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [journals, setJournals] = useState<JournalListItem[]>([]);
  const [journalsLoading, setJournalsLoading] = useState(true);

  // Fetch journals on component mount
  useEffect(() => {
    const fetchJournals = async () => {
      try {
        const response = await fetch('http://localhost:8000/api/journals/list');
        if (!response.ok) {
          throw new Error(`HTTP error! status: ${response.status}`);
        }
        const data: JournalListResponse = await response.json();
        setJournals(data.journals);
      } catch (err) {
        console.error('Error fetching journals:', err);
        setError('Failed to load journals');
      } finally {
        setJournalsLoading(false);
      }
    };

    fetchJournals();
  }, []);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!query.trim()) return;

    setLoading(true);
    setError('');
    setResponse('');

    try {
      const requestBody: any = {
        query: query.trim()
      };

      if (journal.trim()) {
        requestBody.journal = journal.trim();
      }

      const response = await fetch('http://localhost:8000/api/ask_llm', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(requestBody),
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const data = await response.json();
      setResponse(data.llm_response);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'An error occurred');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="App">
      <header className="App-header">
        <h1>AI Research Assistant</h1>
        <p>Ask intelligent questions about your journal documents</p>
      </header>

      <main className="App-main">
        <form onSubmit={handleSubmit} className="ask-form">
          <div className="form-group">
            <label htmlFor="query">What would you like to know?</label>
            <textarea
              id="query"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              placeholder="Ask any question about your research documents, findings, or specific topics..."
              rows={4}
              required
            />
          </div>

          <div className="form-group">
            <label htmlFor="journal">Filter by Journal (Optional)</label>
            <select
              id="journal"
              value={journal}
              onChange={(e) => setJournal(e.target.value)}
              disabled={journalsLoading}
            >
              <option value="">All Journals</option>
              {journals.map((journalItem, index) => (
                <option key={index} value={journalItem.journal_name}>
                  {journalItem.journal_name}
                </option>
              ))}
            </select>
            {journalsLoading && <small>Loading journals...</small>}
          </div>

          <button 
            type="submit" 
            disabled={loading || !query.trim()}
            className={loading ? 'loading' : ''}
          >
            {loading ? 'Processing...' : 'Get Answer'}
          </button>
        </form>

        {error && (
          <div className="error">
            <h3>⚠️ Something went wrong</h3>
            <p>{error}</p>
          </div>
        )}

        {response && (
          <div className="response">
            <h3>AI Response</h3>
            <div className="response-content">
              <ReactMarkdown>{response}</ReactMarkdown>
            </div>
          </div>
        )}
      </main>
    </div>
  );
}

export default App;
