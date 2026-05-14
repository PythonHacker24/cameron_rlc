import React, {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useState,
} from "react";

export type ThemeMode = "light" | "dark";

interface ThemeCtx {
  theme: ThemeMode;
  setTheme: (m: ThemeMode) => void;
  toggle: () => void;
}

const ThemeContext = createContext<ThemeCtx | null>(null);
const STORAGE_KEY = "rlc.theme";

/**
 * ThemeProvider — controls the `data-theme` attribute on <html>, persists the
 * user's choice in localStorage, and exposes setTheme/toggle through context.
 *
 * Default is "light" per the engineering-SaaS aesthetic; switching uses a
 * Tailwind-style data attribute so CSS variables in globals.css can swap.
 */
export function ThemeProvider({ children }: { children: React.ReactNode }) {
  // Initial value is "light" on the server (and on the first client render —
  // we resync from localStorage in an effect below so SSR markup matches).
  const [theme, setThemeState] = useState<ThemeMode>("light");

  useEffect(() => {
    const stored = (typeof window !== "undefined"
      ? (window.localStorage.getItem(STORAGE_KEY) as ThemeMode | null)
      : null);
    if (stored === "light" || stored === "dark") setThemeState(stored);
  }, []);

  // Mirror the current theme onto <html data-theme="..."> so CSS variables resolve.
  useEffect(() => {
    if (typeof document === "undefined") return;
    document.documentElement.setAttribute("data-theme", theme);
    document.documentElement.style.colorScheme = theme;
  }, [theme]);

  const setTheme = useCallback((m: ThemeMode) => {
    setThemeState(m);
    if (typeof window !== "undefined") window.localStorage.setItem(STORAGE_KEY, m);
  }, []);

  const toggle = useCallback(() => {
    setThemeState((cur) => {
      const next: ThemeMode = cur === "light" ? "dark" : "light";
      if (typeof window !== "undefined") window.localStorage.setItem(STORAGE_KEY, next);
      return next;
    });
  }, []);

  const value = useMemo(() => ({ theme, setTheme, toggle }), [theme, setTheme, toggle]);

  return <ThemeContext.Provider value={value}>{children}</ThemeContext.Provider>;
}

export function useTheme(): ThemeCtx {
  const ctx = useContext(ThemeContext);
  if (!ctx) throw new Error("useTheme must be used inside <ThemeProvider>");
  return ctx;
}

/**
 * ThemeToggle — sun/moon button suitable for the page header.
 * Pure CSS, no icon library dependency.
 */
export function ThemeToggle({ className }: { className?: string }) {
  const { theme, toggle } = useTheme();
  const isDark = theme === "dark";
  return (
    <button
      type="button"
      onClick={toggle}
      aria-label={`Switch to ${isDark ? "light" : "dark"} theme`}
      title={`Switch to ${isDark ? "light" : "dark"} theme`}
      className={
        "inline-flex items-center justify-center h-9 w-9 rounded-md transition-colors " +
        (className ?? "")
      }
      style={{
        background: "var(--surface-2)",
        border: "1px solid var(--border)",
        color: "var(--text-muted)",
        cursor: "pointer",
      }}
    >
      {isDark ? (
        // Sun icon (when in dark, clicking switches to light)
        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
          <circle cx="12" cy="12" r="4" />
          <path d="M12 2v2M12 20v2M4.93 4.93l1.41 1.41M17.66 17.66l1.41 1.41M2 12h2M20 12h2M4.93 19.07l1.41-1.41M17.66 6.34l1.41-1.41" />
        </svg>
      ) : (
        // Moon icon (when in light, clicking switches to dark)
        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
          <path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z" />
        </svg>
      )}
    </button>
  );
}
