import Disclaimer from '../components/Disclaimer.jsx'

export default function About() {
  return (
    <section className="narrow-wide">
      <h1>About &amp; disclaimer</h1>
      <Disclaimer />
      <section className="card"><h3>Academic objectives</h3>
        <p>This project studies deep-learning classification of chest X-rays into five classes (COVID-19, Lung Opacity, Normal, Viral Pneumonia, Tuberculosis), compares several architectures under identical conditions, selects one by validation performance, evaluates it once on a held-out test set, and exposes it through a local web application.</p></section>
      <section className="card"><h3>Dataset attribution</h3>
        <p>The five-class dataset (≈32,687 images) is hosted on Harvard Dataverse and was assembled from public chest X-ray collections. Please cite the original dataset authors and the sources they compiled when reporting results. The dataset is not distributed with this software.</p></section>
      <section className="card"><h3>Intended research use</h3>
        <p>Education, benchmarking and demonstration of model interpretability. Not for diagnosis, triage, treatment decisions, or any clinical workflow.</p></section>
      <section className="card"><h3>Privacy and medical safety</h3>
        <ul>
          <li>Uploaded images are processed in memory by the local server, are not written to disk and their content is never logged. Nothing is sent to external services.</li>
          <li>Recent analysis results (including images shown in the report) are kept in server memory only, to build the PDF, and are lost when the server stops.</li>
          <li>Seek urgent medical attention for severe symptoms. Do not delay emergency care because of an AI prediction. A “Normal” result does not rule out disease.</li>
        </ul></section>
    </section>
  )
}
