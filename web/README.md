# Cover Version Retrieval: the website

An interactive research report for this repository: HPCP, the TCN embedding, key
rotation, subsequence DTW, hubness, and all four benchmark evaluations. It is a
static site (React + TypeScript + Vite). No server, no audio, no model at runtime:
everything is precomputed from committed result files (see `src/data/README.md`).

```bash
cd web
npm ci
npm run dev        # local development
npm test           # DTW port + data-integrity tests
npm run build      # static output in dist/
```

## Deploying

**Vercel.** Import the repository and set **Root Directory** to `web`. `vercel.json`
pins the framework (Vite), install (`npm ci`), build (`npm run build`) and output
(`dist`). No environment variables are needed; the base path defaults to `/`.

**GitHub Pages.** `.github/workflows/web.yml` builds with
`BASE_PATH=/<repo>/` and deploys on pushes to `main` once Pages is set to
"GitHub Actions" in the repository settings.

## Data and licence

Da-TACOS metadata and features are CC BY-NC-SA 4.0 (© MTG, UPF) and are not
redistributed. The site shows figures already committed to `reports/figures/` and
values decoded from them, for non-commercial research communication with
attribution. Code is MIT.
