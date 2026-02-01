// Copyright (c) 2026 Abdulrahman Asiri.
// Engineered via Vibe Coding.
// Licensed under the MIT License.

import React from 'react';
import ReactDOM from 'react-dom/client';
import './styles/App.css';
import App from './App';

import { QueryClient, QueryClientProvider } from '@tanstack/react-query';

const queryClient = new QueryClient();

const root = ReactDOM.createRoot(document.getElementById('root'));
root.render(
  <React.StrictMode>
    <QueryClientProvider client={queryClient}>
      <App />
    </QueryClientProvider>
  </React.StrictMode>
);
