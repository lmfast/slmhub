/** @type {import('tailwindcss').Config} */
export default {
  content: ['./src/**/*.{astro,html,js,jsx,md,mdx,svelte,ts,tsx,vue}'],
  theme: {
    extend: {
      colors: {
        accent: {
          DEFAULT: '#6366f1',
          low: '#312e81',
          high: '#a5b4fc',
        },
      },
    },
  },
  plugins: [],
};
