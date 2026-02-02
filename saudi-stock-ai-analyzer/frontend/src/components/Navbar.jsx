import React from 'react';
import { useNavigate, useLocation } from 'react-router-dom';
import { ChartBarSquareIcon, BellIcon, Cog6ToothIcon } from '@heroicons/react/24/outline';

const navItems = [
    { name: 'Dashboard', path: '/' },
    { name: 'Analysis', path: '/analysis' },
    { name: 'Scanner', path: '/scanner' },
    { name: 'Portfolio', path: '/portfolio' },
];

const Navbar = () => {
    const navigate = useNavigate();
    const location = useLocation();

    const isActive = (path) => {
        if (path === '/') {
            return location.pathname === '/';
        }
        return location.pathname.startsWith(path);
    };

    return (
        <nav className="fixed top-0 left-0 right-0 z-50 h-16 bg-dark-950/80 backdrop-blur-md border-b border-white/5 flex items-center justify-between px-6 transition-all duration-300">
            {/* Brand */}
            <div
                className="flex items-center gap-3 group cursor-pointer"
                onClick={() => navigate('/')}
            >
                <div className="relative w-8 h-8 flex items-center justify-center bg-brand-primary/20 rounded-lg overflow-hidden group-hover:shadow-[0_0_15px_rgba(59,130,246,0.5)] transition-shadow">
                    <ChartBarSquareIcon className="w-5 h-5 text-brand-primary" />
                    <div className="absolute inset-0 bg-brand-primary/10 animate-pulse-slow"></div>
                </div>
                <div className="flex flex-col">
                    <span className="text-sm font-bold tracking-wider text-white">SAUDI<span className="text-brand-primary">.AI</span></span>
                    <span className="text-[10px] text-gray-500 tracking-[0.2em] font-mono">INTELLIGENCE</span>
                </div>
            </div>

            {/* Navigation */}
            <div className="hidden md:flex items-center gap-1 bg-white/5 p-1 rounded-full border border-white/5">
                {navItems.map((item) => (
                    <button
                        key={item.name}
                        onClick={() => navigate(item.path)}
                        className={`px-4 py-1.5 text-xs font-medium rounded-full transition-all duration-300 ${
                            isActive(item.path)
                                ? 'bg-brand-primary text-white shadow-lg shadow-brand-primary/25'
                                : 'text-gray-400 hover:text-white hover:bg-white/5'
                        }`}
                    >
                        {item.name}
                    </button>
                ))}
            </div>

            {/* Actions */}
            <div className="flex items-center gap-4">
                <div className="flex items-center gap-2 text-xs font-mono text-brand-secondary bg-brand-secondary/10 px-3 py-1 rounded-full border border-brand-secondary/20">
                    <span className="relative flex h-2 w-2">
                        <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-brand-secondary opacity-75"></span>
                        <span className="relative inline-flex rounded-full h-2 w-2 bg-brand-secondary"></span>
                    </span>
                    MARKET OPEN
                </div>

                <button className="p-2 text-gray-400 hover:text-white transition-colors relative">
                    <BellIcon className="w-5 h-5" />
                    <span className="absolute top-2 right-2 w-2 h-2 bg-red-500 rounded-full border-2 border-dark-950"></span>
                </button>
                <button className="p-2 text-gray-400 hover:text-white transition-colors">
                    <Cog6ToothIcon className="w-5 h-5" />
                </button>
            </div>
        </nav>
    );
};

export default Navbar;
