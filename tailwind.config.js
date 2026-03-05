/** @type {import('tailwindcss').Config} */
module.exports = {
  content: [
    "./pages/**/*.{js,ts,jsx,tsx,mdx}",
    "./components/**/*.{js,ts,jsx,tsx,mdx}",
  ],
  theme: {
    extend: {
      fontFamily: {
        sans: ["Inter", "system-ui", "-apple-system", "sans-serif"],
        mono: ['"JetBrains Mono"', '"Courier New"', "monospace"],
      },
      colors: {
        eng: {
          bg: "#060c18",
          panel: "#0a1422",
          card: "#0d1a2e",
          "border-sub": "#132034",
          border: "#1b3050",
          text: "#bdd0e4",
          muted: "#4a6580",
          dim: "#243548",
          cyan: "#00b8d9",
          "cyan-dim": "#00728a",
          amber: "#f59e0b",
          "amber-dim": "#92600a",
          green: "#10b981",
          "green-dim": "#0a7a57",
          red: "#ef4444",
          "red-dim": "#991b1b",
          violet: "#818cf8",
          teal: "#2dd4bf",
          orange: "#fb923c",
          slate: "#64748b",
        },
      },
      boxShadow: {
        "eng-cyan": "0 0 12px rgba(0,184,217,0.25)",
        "eng-amber": "0 0 12px rgba(245,158,11,0.25)",
        "eng-green": "0 0 12px rgba(16,185,129,0.25)",
        "eng-panel": "0 2px 24px rgba(0,0,0,0.6)",
      },
      borderRadius: {
        eng: "2px",
      },
    },
  },
  plugins: [],
};
