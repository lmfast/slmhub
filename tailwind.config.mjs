/** @type {import('tailwindcss').Config} */
export default {
  content: ['./src/**/*.{astro,html,js,jsx,md,mdx,svelte,ts,tsx,vue}'],
  theme: {
    extend: {
      colors: {
        // 2026 Modern Palette
        accent: {
          DEFAULT: '#FF6B35', // Warm orange - 2026 trending
          low: '#E85A2A',
          high: '#FFA07A',
        },
        primary: {
          DEFAULT: '#FF6B35',
          50: '#FFF5F2',
          100: '#FFE8E0',
          200: '#FFD1C2',
          300: '#FFBA9F',
          400: '#FF9266',
          500: '#FF6B35',
          600: '#E85A2A',
          700: '#CC4A1F',
          800: '#A33B18',
          900: '#7A2C12',
        },
        secondary: {
          DEFAULT: '#4A90E2', // Tech blue
          50: '#EBF5FF',
          100: '#D6EAFF',
          200: '#ADD5FF',
          300: '#85C0FF',
          400: '#5CABFF',
          500: '#4A90E2',
          600: '#3B73B5',
          700: '#2C5688',
          800: '#1D395B',
          900: '#0E1C2E',
        },
        success: {
          DEFAULT: '#2ECC71', // Success green
          50: '#E8F8F0',
          100: '#D1F1E1',
          200: '#A3E3C3',
          300: '#75D5A5',
          400: '#47C787',
          500: '#2ECC71',
          600: '#25A35A',
          700: '#1C7A44',
          800: '#13512D',
          900: '#0A2817',
        },
        // Glassmorphism colors
        glass: {
          light: 'rgba(255, 255, 255, 0.7)',
          dark: 'rgba(23, 23, 23, 0.7)',
          border: 'rgba(255, 255, 255, 0.5)',
          'border-dark': 'rgba(255, 255, 255, 0.1)',
        },
      },
      fontFamily: {
        sans: ['Inter', 'system-ui', '-apple-system', 'sans-serif'],
        display: ['Space Grotesk', 'Inter', 'system-ui', 'sans-serif'],
        mono: ['JetBrains Mono', 'Fira Code', 'Consolas', 'monospace'],
      },
      backgroundImage: {
        'gradient-hero': 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
        'gradient-card': 'linear-gradient(to bottom right, #f8f9fa, #e9ecef)',
        'gradient-primary': 'linear-gradient(135deg, #FF6B35 0%, #8b5cf6 100%)',
        'gradient-secondary': 'linear-gradient(135deg, #4A90E2 0%, #667eea 100%)',
      },
      animation: {
        'morph': 'morph 8s ease-in-out infinite',
        'float': 'float 6s ease-in-out infinite',
        'ripple': 'ripple 0.6s ease-out',
      },
      keyframes: {
        morph: {
          '0%, 100%': { borderRadius: '40% 60% 70% 30% / 40% 50% 60% 50%' },
          '50%': { borderRadius: '54% 46% 38% 62% / 49% 70% 30% 51%' },
        },
        float: {
          '0%, 100%': { transform: 'translateY(0px)' },
          '50%': { transform: 'translateY(-20px)' },
        },
        ripple: {
          'to': {
            transform: 'scale(4)',
            opacity: '0',
          },
        },
      },
      backdropBlur: {
        xs: '2px',
      },
    },
  },
  plugins: [],
};
