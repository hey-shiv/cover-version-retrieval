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

**Vercel.** Import the GitHub repository in Vercel and set **Root Directory** to
`web`. `vercel.json` pins the framework (Vite), install (`npm ci`), build
(`npm run build`) and output (`dist`). No environment variables are needed. With the
Git integration connected, every push to `main` deploys to production and every PR
gets a preview.

The build is fully static. Any static host works; for a sub-path host (for example
GitHub Pages at `/<repo>/`) set `BASE_PATH=/<repo>/` when building.

CI (`.github/workflows/web.yml`) runs the tests and the production build, and fails
if `src/data/generated/` is out of date with `reports/`.

## Data and licence

Da-TACOS metadata and features are CC BY-NC-SA 4.0 (© MTG, UPF) and are not
redistributed. The site shows figures already committed to `reports/figures/` and
values decoded from them, for non-commercial research communication with
attribution. Code is MIT.
