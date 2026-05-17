"""
All custom CSS for the app in one string.

Injected once at startup via ``st.markdown(f"<style>{INLINE_CSS}</style>",
unsafe_allow_html=True)``. Keeping every rule here means ``components.py`` only
emits class names — no inline ``style=`` noise mixed into the markup, so the
visual contract lives in exactly one place (SPEC §5.9).

Class naming: every class is prefixed ``ts-`` to avoid colliding with
Streamlit's own DOM classes, which are unprefixed and unstable across versions.
"""

INLINE_CSS = """
/* --- header strip ------------------------------------------------------- */
.ts-header {
  display: flex; align-items: center; gap: .6rem;
  padding: .55rem .9rem; margin-bottom: .4rem;
  background: #0f172a; color: #f8fafc; border-radius: .5rem;
}
.ts-header-title { font-weight: 700; font-size: 1.05rem; }
.ts-header-badge {
  font-size: .72rem; font-weight: 600; padding: .15rem .5rem;
  background: #1e293b; color: #cbd5e1; border-radius: 999px;
}

/* --- agent identity pill ------------------------------------------------ */
.ts-pill {
  display: inline-block; font-size: .72rem; font-weight: 600;
  padding: .12rem .55rem; border-radius: 999px; color: #fff;
}
.ts-pill--retrieval  { background: #2563eb; }
.ts-pill--comparison { background: #7c3aed; }
.ts-pill--summarizer { background: #0d9488; }
.ts-pill--default    { background: #64748b; }

/* --- source attribution panel (tinted blue) ----------------------------- */
.ts-source-panel {
  background: #eef4fb; border-left: 4px solid #3b82f6;
  border-radius: .35rem; padding: .55rem .8rem; margin: .5rem 0;
}
.ts-source-head {
  font-size: .76rem; font-weight: 700; color: #1e3a8a;
  text-transform: uppercase; letter-spacing: .03em; margin-bottom: .35rem;
}
.ts-source-item { font-size: .82rem; color: #1f2937; margin: .3rem 0; }
.ts-source-id { font-weight: 700; color: #1d4ed8; }
.ts-source-tag {
  font-size: .68rem; background: #dbeafe; color: #1e40af;
  padding: .05rem .4rem; border-radius: 4px; margin-left: .3rem;
}
.ts-source-score {
  float: right; font-size: .72rem; font-weight: 700; color: #2563eb;
}
.ts-source-snippet { color: #475569; font-size: .78rem; margin-top: .1rem; }

/* --- per-answer metadata badge row -------------------------------------- */
.ts-meta { display: flex; flex-wrap: wrap; gap: .4rem; margin-top: .35rem; }
.ts-meta-badge {
  font-size: .72rem; background: #f1f5f9; color: #334155;
  padding: .12rem .5rem; border-radius: 4px;
}
"""
