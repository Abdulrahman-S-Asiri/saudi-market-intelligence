import React from 'react';
import { ArrowTrendingUpIcon, ArrowTrendingDownIcon } from '@heroicons/react/24/solid';

const StatCard = ({ title, value, change, isPositive, icon: Icon, subValue }) => {
    return (
        <div className="glass-card p-5 relative overflow-hidden group hover:border-brand-primary/30 transition-all duration-300">
            {/* Background Glow */}
            <div className={`absolute -right-6 -top-6 w-24 h-24 bg-${isPositive ? 'green' : 'red'}-500/10 rounded-full blur-2xl group-hover:bg-${isPositive ? 'green' : 'red'}-500/20 transition-all duration-500`}></div>

            <div className="relative z-10">
                <div className="flex justify-between items-start mb-2">
                    <span className="text-gray-400 text-xs font-medium uppercase tracking-wider">{title}</span>
                    {Icon && <Icon className="w-4 h-4 text-gray-500" />}
                </div>

                <div className="flex items-baseline gap-2">
                    <span className="text-2xl font-bold font-mono text-white tracking-tight">{value}</span>
                    {change && (
                        <div className={`flex items-center gap-1 text-xs font-bold px-1.5 py-0.5 rounded ${isPositive ? 'text-trade-up bg-trade-up/10' : 'text-trade-down bg-trade-down/10'
                            }`}>
                            {isPositive ? <ArrowTrendingUpIcon className="w-3 h-3" /> : <ArrowTrendingDownIcon className="w-3 h-3" />}
                            {change}
                        </div>
                    )}
                </div>

                {subValue && (
                    <div className="mt-2 text-xs text-gray-500 font-mono">
                        {subValue}
                    </div>
                )}
            </div>
        </div>
    );
};

export default StatCard;
