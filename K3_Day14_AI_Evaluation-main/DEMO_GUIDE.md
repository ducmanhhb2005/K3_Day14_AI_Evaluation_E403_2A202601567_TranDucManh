# Demo Guide — Northstar Evaluation Command Center

## 1. Start the UI

```bash
python -m pip install -r requirements.txt
python -m streamlit run ui_app.py
```

Open the local URL shown by Streamlit, normally `http://localhost:8501`.

## 2. Recommended 5-minute demo flow

1. **Overview** — introduce the end-to-end pipeline and the five metrics.
2. **Golden Dataset** — show 5 Easy + 7 Medium + 5 Hard + 3 Adversarial,
   10/10 document coverage, and verbatim evidence provenance.
3. **Run API** — paste the OpenAI key into the password field, run one smoke
   question, then run all 20 questions. The key is not written to disk.
4. **Results** — explain the pass rate, metric fingerprint, difficulty chart,
   and case matrix using the newly created real benchmark artifact.
5. **Trace Explorer** — compare the model answer, expected answer, ranked BM25
   chunks, and gold evidence for one case.
6. **Failure Lab** — show failure clusters, prioritized actions, lowest cases,
   and the generated improvement log.

## 3. Truth-first rules

- The UI displays benchmark metrics only after a complete real API run.
- A one-question smoke test never writes a misleading 20-case benchmark.
- A provider error stops the run; it does not create placeholder answers.
- `artifacts/actual_answers.json` records model, retrieval trace and timestamp.
- `artifacts/benchmark_results.json` is computed by the same core covered by
  `tests/test_solution.py`.

## 4. Final verification

```bash
python -m unittest tests.test_solution -v
python validate_golden_dataset.py
```

Expected local results before the live API run:

- 42/42 core tests pass.
- Golden dataset validator reports `PASS` with 20 QA and 10/10 sources.

The live pass rate and metric values must be presented only after running with
your own key because model output can change between runs.
