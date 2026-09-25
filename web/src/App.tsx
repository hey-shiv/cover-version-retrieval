import { Nav } from './components/Nav'
import { Footer } from './components/Footer'
import { Hero } from './sections/Hero'
import { Problem } from './sections/Problem'
import { Representation } from './sections/Representation'
import { Encoder } from './sections/Encoder'
import { Search } from './sections/Search'
import { Rotation } from './sections/Rotation'
import { Rescue } from './sections/Rescue'
import { Explorer } from './sections/Explorer'
import { Hubness } from './sections/Hubness'
import { Benchmark } from './sections/Benchmark'
import { Journey } from './sections/Journey'
import { Beyond } from './sections/Beyond'
import { DataLicense, Limits, Reproducibility } from './sections/Closing'

export default function App() {
  return (
    <>
      <a className="skip" href="#problem">
        Skip to content
      </a>
      <Nav />
      <main id="top">
        <Hero />
        <Problem />
        <Representation />
        <Encoder />
        <Search />
        <Rotation />
        <Rescue />
        <Explorer />
        <Benchmark />
        <Hubness />
        <Journey />
        <Beyond />
        <Reproducibility />
        <DataLicense />
        <Limits />
      </main>
      <Footer />
    </>
  )
}
