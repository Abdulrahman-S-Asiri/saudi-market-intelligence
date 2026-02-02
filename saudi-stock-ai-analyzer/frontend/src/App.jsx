import React from 'react';
import { Routes, Route } from 'react-router-dom';
import Navbar from './components/Navbar';
import Dashboard from './pages/Dashboard';
import Analysis from './pages/Analysis';
import Scanner from './pages/Scanner';
import Portfolio from './pages/Portfolio';

function App() {
  return (
    <div className="min-h-screen pb-10">
      <Navbar />

      {/* Spacer for fixed navbar */}
      <div className="h-20"></div>

      {/* Routes */}
      <Routes>
        <Route path="/" element={<Dashboard />} />
        <Route path="/analysis" element={<Analysis />} />
        <Route path="/scanner" element={<Scanner />} />
        <Route path="/portfolio" element={<Portfolio />} />
      </Routes>
    </div>
  );
}

export default App;
