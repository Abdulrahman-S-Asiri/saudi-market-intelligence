/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./src/**/*.{js,jsx,ts,tsx}",
  ],
  darkMode: 'class',
  theme: {
    extend: {
      colors: {
        // Ultra-premium dark mode background
        dark: {
          950: '#020617', // Main background (almost black blue)
          900: '#0f172a', // Secondary background (cards)
          800: '#1e293b', // Borders/Separators
          700: '#334155', // Muted text
          600: '#475569', // Hover states
          500: '#64748b', // Lighter text
        },
        // Neon Accents for Financial Data
        trade: {
          up: '#00dc82',     // Vivid Green
          down: '#ff2b2b',   // Vivid Red
          neutral: '#94a3b8', // Slate
          glow: '#00dc8240', // Green Glow opacity
        },
        // Brand Colors
        brand: {
          primary: '#3b82f6', // Bright Blue
          secondary: '#8b5cf6', // Violet
          accent: '#06b6d4', // Cyan
        }
      },
      fontFamily: {
        sans: ['Inter', 'system-ui', 'sans-serif'],
        mono: ['JetBrains Mono', 'Fira Code', 'monospace'], // For numbers/tickers
      },
      backgroundImage: {
        'gradient-radial': 'radial-gradient(var(--tw-gradient-stops))',
        'grid-pattern': "url(\"data:image/svg+xml,%3Csvg width='40' height='40' viewBox='0 0 40 40' xmlns='http://www.w3.org/2000/svg'%3E%3Cg fill='%2394a3b8' fill-opacity='0.05' fill-rule='evenodd'%3E%3Cpath d='M0 40L40 0H20L0 20M40 40V20L20 40'/%3E%3C/g%3E%3C/svg%3E\")",
      },
      boxShadow: {
        'glass': '0 8px 32px 0 rgba(0, 0, 0, 0.37)',
        'neon-green': '0 0 10px rgba(0, 220, 130, 0.5), 0 0 20px rgba(0, 220, 130, 0.3)',
        'neon-red': '0 0 10px rgba(255, 43, 43, 0.5), 0 0 20px rgba(255, 43, 43, 0.3)',
        'neon-blue': '0 0 15px rgba(59, 130, 246, 0.6)',
      },
      animation: {
        'pulse-slow': 'pulse 4s cubic-bezier(0.4, 0, 0.6, 1) infinite',
        'float': 'float 6s ease-in-out infinite',
        'ticker': 'ticker 30s linear infinite',
      },
      keyframes: {
        float: {
          '0%, 100%': { transform: 'translateY(0)' },
          '50%': { transform: 'translateY(-10px)' },
        },
        ticker: {
          '0%': { transform: 'translateX(0)' },
          '100%': { transform: 'translateX(-100%)' },
        }
      }
    },
  },
  plugins: [],
}
