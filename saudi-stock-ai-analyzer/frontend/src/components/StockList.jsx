import React, { useState } from 'react';

const StockList = ({ stocks, onSelect, selectedSymbol }) => {
    const [searchTerm, setSearchTerm] = useState('');

    // Filter stocks based on search term
    const filteredStocks = stocks.filter(stock =>
        stock.symbol.toLowerCase().includes(searchTerm.toLowerCase()) ||
        stock.name?.toLowerCase().includes(searchTerm.toLowerCase())
    );

    return (
        <div className="glass-panel h-full flex flex-col rounded-2xl overflow-hidden">
            <div className="p-4 border-b border-white/5 bg-white/5">
                <h3 className="text-xs font-bold text-gray-400 uppercase tracking-widest mb-3">Market Watch</h3>
                <input
                    type="text"
                    placeholder="Search Symbol..."
                    value={searchTerm}
                    onChange={(e) => setSearchTerm(e.target.value)}
                    className="w-full bg-dark-950/50 border border-white/10 rounded-lg px-3 py-2 text-sm text-white focus:outline-none focus:border-brand-primary/50 transition-colors placeholder-gray-600"
                />
            </div>

            <div className="flex-1 overflow-y-auto custom-scrollbar p-2 space-y-1">
                {filteredStocks.map((stock) => (
                    <button
                        key={stock.symbol}
                        onClick={() => onSelect(stock.symbol)}
                        className={`w-full flex items-center justify-between p-3 rounded-xl transition-all duration-200 group ${selectedSymbol === stock.symbol
                                ? 'bg-brand-primary text-white shadow-lg shadow-brand-primary/20'
                                : 'hover:bg-white/5 text-gray-400 hover:text-white'
                            }`}
                    >
                        <div className="flex items-center gap-3">
                            <div className={`w-8 h-8 rounded-lg flex items-center justify-center text-xs font-bold font-mono ${selectedSymbol === stock.symbol ? 'bg-white/20' : 'bg-dark-800'
                                }`}>
                                {stock.symbol.substring(0, 2)}
                            </div>
                            <div className="text-left">
                                <div className="text-sm font-bold">{stock.symbol}</div>
                                <div className="text-[10px] opacity-60 truncate max-w-[100px]">{stock.name}</div>
                            </div>
                        </div>

                        <div className="text-right">
                            {/* Fake random changes for visuals if real data missing */}
                            <div className="text-sm font-mono font-medium">--</div>
                        </div>
                    </button>
                ))}
            </div>
        </div>
    );
};

export default StockList;
