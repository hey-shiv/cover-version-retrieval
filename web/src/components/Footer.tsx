import { PAPER_URL, REPO_URL } from './Nav'

const REFS = [
  'Yesiler et al. Da-TACOS: A Dataset for Cover Song Identification and Understanding. ISMIR 2019.',
  'Serrà, Gómez, Herrera, Serra. Chroma binary similarity and local alignment applied to cover song identification. IEEE TASLP 2008.',
  'Müller. Fundamentals of Music Processing, 2nd ed. Springer 2021 (DTW, music synchronisation).',
  'Bai, Kolter, Koltun. An empirical evaluation of generic convolutional and recurrent networks for sequence modeling (TCN). 2018.',
  'Khosla et al. Supervised Contrastive Learning. NeurIPS 2020.',
  'Aucouturier & Pachet. A scale-free distribution of false positives for a large class of audio similarity measures. Pattern Recognition 2008.',
  'Seo. Pairwise similarity normalization based on a hubness score for improving cover song retrieval accuracy. IEICE 2022.',
]

export function Footer() {
  return (
    <footer className="footer panel">
      <div className="wrap footer-grid">
        <div>
          <div className="statement footer-statement">
            Can the machine hear <em>through the arrangement?</em>
          </div>
          <p className="footer-answer">Sometimes. This page shows when, and why not.</p>
          <p>
            <a href={REPO_URL}>Code, reports and every result file on GitHub ↗</a>
          </p>
          <p>
            <a href={PAPER_URL}>The paper (draft PDF), built from the same result files ↗</a>
          </p>
        </div>
        <div>
          <div className="label">References</div>
          <ol className="refs">
            {REFS.map((r) => (
              <li key={r}>{r}</li>
            ))}
          </ol>
          <div className="label" style={{ marginTop: 24 }}>
            Colophon
          </div>
          <p className="colophon">
            Code MIT. Da-TACOS metadata and features CC BY-NC-SA 4.0, © Music Technology Group, Universitat Pompeu Fabra; not redistributed. Figures derived
            from them are shown for non-commercial research communication with attribution. Set in Newsreader, IBM Plex Sans and IBM Plex Mono. Static site:
            React, TypeScript, Vite; every chart drawn with SVG or Canvas from repository files.
          </p>
        </div>
      </div>
    </footer>
  )
}
