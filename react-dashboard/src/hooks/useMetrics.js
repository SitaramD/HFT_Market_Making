import { useState, useEffect } from 'react';
import axios from 'axios';

const API = process.env.REACT_APP_TSDB_API_URL || '';

export function useMetrics(interval = 2000) {
  const [metrics, setMetrics] = useState(null);
  const [error,   setError]   = useState(null);

  useEffect(() => {
    const fetch = async () => {
      try {
        const { data } = await axios.get(`${API}/metrics`);
        setMetrics(data);
      } catch (e) {
        setError(e.message);
      }
    };
    fetch();
    const id = setInterval(fetch, interval);
    return () => clearInterval(id);
  }, [interval]);

  return { metrics, error };
}
