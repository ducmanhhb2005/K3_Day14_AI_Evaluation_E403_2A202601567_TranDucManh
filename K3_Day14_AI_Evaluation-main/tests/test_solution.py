"""
Day 14 — Evaluation Pipeline & Failure Analysis
Test suite for student solution.

Run from the day folder:
    pytest tests/ -v
"""

import importlib.util
import sys
import unittest
from pathlib import Path

DAY_DIR = Path(__file__).parent.parent
SOLUTION_DIR = DAY_DIR / "solution"


def _load(path: Path, unique_name: str):
    spec = importlib.util.spec_from_file_location(unique_name, str(path))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[unique_name] = mod
    spec.loader.exec_module(mod)
    return mod


if (SOLUTION_DIR / "solution.py").exists():
    _m = _load(SOLUTION_DIR / "solution.py", f"{DAY_DIR.name}.solution")
elif (SOLUTION_DIR / "app.py").exists():
    _m = _load(SOLUTION_DIR / "app.py", f"{DAY_DIR.name}.solution")
else:
    src = "template.py" if (DAY_DIR / "template.py").exists() else "app.py"
    _m = _load(DAY_DIR / src, f"{DAY_DIR.name}.template")

QAPair = getattr(_m, 'QAPair')
EvalResult = getattr(_m, 'EvalResult')
RAGASEvaluator = getattr(_m, 'RAGASEvaluator')
LLMJudge = getattr(_m, 'LLMJudge')
BenchmarkRunner = getattr(_m, 'BenchmarkRunner')
FailureAnalyzer = getattr(_m, 'FailureAnalyzer')
template = _m

def _make_qa(question: str = "What is AI?",
             expected: str = "AI is artificial intelligence",
             context: str = "AI stands for artificial intelligence") -> QAPair:
    return QAPair(question=question, expected_answer=expected, context=context)


def _make_eval_result(faithfulness: float = 0.8, relevance: float = 0.8,
                      completeness: float = 0.8, failure_type: str | None = None) -> EvalResult:
    qa = _make_qa()
    passed = faithfulness >= 0.5 and relevance >= 0.5 and completeness >= 0.5
    return EvalResult(
        qa_pair=qa,
        actual_answer="AI is artificial intelligence.",
        faithfulness=faithfulness,
        relevance=relevance,
        completeness=completeness,
        passed=passed,
        failure_type=failure_type,
    )


def _mock_judge_llm(prompt: str) -> str:
    return '{"accuracy": 0.8, "clarity": 0.7}'


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestRAGASEvaluator(unittest.TestCase):

    def setUp(self):
        self.evaluator = RAGASEvaluator()

    def test_faithfulness_fully_grounded(self):
        context = "Artificial intelligence is a branch of computer science"
        answer = "artificial intelligence"
        score = self.evaluator.evaluate_faithfulness(answer, context)
        self.assertGreaterEqual(score, 0.8)

    def test_faithfulness_unrelated_answer(self):
        context = "Python is a programming language"
        answer = "Jupiter Neptune Saturn Uranus"
        score = self.evaluator.evaluate_faithfulness(answer, context)
        self.assertLess(score, 0.3)

    def test_faithfulness_in_range(self):
        score = self.evaluator.evaluate_faithfulness("some answer", "some context here")
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)

    def test_relevance_high_when_directly_answers(self):
        question = "What is machine learning"
        answer = "machine learning is a subfield of artificial intelligence"
        score = self.evaluator.evaluate_relevance(answer, question)
        self.assertGreater(score, 0.3)

    def test_relevance_in_range(self):
        score = self.evaluator.evaluate_relevance("answer text", "question text")
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)

    def test_completeness_full_match(self):
        expected = "Python is a programming language"
        answer = "Python is a programming language"
        score = self.evaluator.evaluate_completeness(answer, expected)
        self.assertAlmostEqual(score, 1.0, places=1)

    def test_completeness_no_match(self):
        expected = "Python is a programming language"
        answer = "Jupiter is a planet in the solar system"
        score = self.evaluator.evaluate_completeness(answer, expected)
        # Minimal overlap (only stopwords like "is" and "a")
        self.assertLess(score, 0.4)


class TestContextMetrics(unittest.TestCase):
    """Retrieval-side RAGAS metrics: Context Recall + Context Precision."""

    def setUp(self):
        self.evaluator = RAGASEvaluator()

    # --- Context Recall ----------------------------------------------------

    def test_context_recall_full_coverage(self):
        chunks = ["Paris is the capital", "France is located in Europe"]
        expected = "Paris is the capital of France"
        score = self.evaluator.evaluate_context_recall(chunks, expected)
        self.assertGreaterEqual(score, 0.8)

    def test_context_recall_low_when_evidence_missing(self):
        chunks = ["Bananas grow on tropical plantations year round"]
        expected = "Quantum entanglement links distant particles"
        score = self.evaluator.evaluate_context_recall(chunks, expected)
        self.assertLess(score, 0.3)

    def test_context_recall_in_range(self):
        score = self.evaluator.evaluate_context_recall(["some retrieved text"], "expected answer")
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)

    # --- Context Precision (rank-aware) ------------------------------------

    def test_context_precision_rewards_relevant_first(self):
        expected = "Paris is the capital of France"
        relevant = "Paris is the capital city of France"
        noise = "Bananas are yellow tropical fruits grown near the equator"
        good_order = self.evaluator.evaluate_context_precision([relevant, noise], expected)
        bad_order = self.evaluator.evaluate_context_precision([noise, relevant], expected)
        self.assertGreater(good_order, bad_order)

    def test_context_precision_empty_contexts_is_zero(self):
        self.assertEqual(self.evaluator.evaluate_context_precision([], "expected text"), 0.0)

    def test_context_precision_in_range(self):
        score = self.evaluator.evaluate_context_precision(
            ["Paris is the capital of France", "unrelated noise chunk"],
            "Paris is the capital of France",
        )
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)

    # --- Reranking lifts precision ----------------------------------------

    def test_reranking_improves_or_keeps_precision(self):
        rerank = getattr(template, "rerank_by_overlap")
        expected = "Paris is the capital of France"
        retrieved = [
            "Bananas are a tropical fruit rich in potassium",   # noise first
            "Paris is the capital city of France in Europe",    # relevant, buried
        ]
        before = self.evaluator.evaluate_context_precision(retrieved, expected)
        try:
            reranked = rerank(retrieved, expected)
        except NotImplementedError:
            self.skipTest("Exercise 3.5 reranking is optional bonus work")
        after = self.evaluator.evaluate_context_precision(reranked, expected)
        self.assertGreaterEqual(after, before)


class TestRetrievalMetricWiring(unittest.TestCase):
    """The two retrieval metrics must be connected to the full benchmark."""

    def setUp(self):
        self.evaluator = RAGASEvaluator()

    def test_run_full_eval_connects_optional_retrieval_metrics(self):
        kwargs = {
            "answer": "Paris is the capital of France",
            "question": "What is the capital of France?",
            "context": "Paris is the capital city of France.",
            "expected": "Paris is the capital of France",
        }
        without_retrieval = self.evaluator.run_full_eval(**kwargs)
        with_retrieval = self.evaluator.run_full_eval(
            **kwargs,
            contexts=["Paris is the capital city of France."],
        )

        self.assertIsNone(without_retrieval.context_recall)
        self.assertIsNone(without_retrieval.context_precision)
        self.assertIsNotNone(with_retrieval.context_recall)
        self.assertIsNotNone(with_retrieval.context_precision)

    def test_runner_forwards_retrieved_contexts(self):
        received: list[list[str] | None] = []

        class RecordingEvaluator:
            def run_full_eval(
                self,
                answer,
                question,
                context,
                expected,
                contexts=None,
            ):
                received.append(contexts)
                return EvalResult(
                    qa_pair=_make_qa(question, expected, context),
                    actual_answer=answer,
                    faithfulness=1.0,
                    relevance=1.0,
                    completeness=1.0,
                    passed=True,
                )

        pair = QAPair(
            question="What is AI?",
            expected_answer="AI is artificial intelligence",
            context="AI means artificial intelligence",
            retrieved_contexts=["AI means artificial intelligence"],
        )
        BenchmarkRunner().run(
            [pair],
            lambda _: "AI is artificial intelligence",
            RecordingEvaluator(),
        )

        self.assertEqual(received, [pair.retrieved_contexts])

    def test_report_includes_retrieval_averages(self):
        first = _make_eval_result()
        first.context_recall = 1.0
        first.context_precision = 0.5
        second = _make_eval_result()
        second.context_recall = 0.5
        second.context_precision = 1.0

        report = BenchmarkRunner().generate_report([first, second])

        self.assertAlmostEqual(report["avg_context_recall"], 0.75)
        self.assertAlmostEqual(report["avg_context_precision"], 0.75)


class TestBenchmarkRunner(unittest.TestCase):

    def setUp(self):
        self.runner = BenchmarkRunner()
        self.evaluator = RAGASEvaluator()
        self.qa_pairs = [
            _make_qa("What is AI?", "AI is artificial intelligence", "AI means artificial intelligence"),
            _make_qa("What is Python?", "Python is a programming language", "Python is a high-level language"),
        ]
        self.agent_fn = lambda q: "artificial intelligence programming language"

    def test_run_returns_same_count_as_input(self):
        results = self.runner.run(self.qa_pairs, self.agent_fn, self.evaluator)
        self.assertEqual(len(results), len(self.qa_pairs))

    def test_run_returns_eval_results(self):
        results = self.runner.run(self.qa_pairs, self.agent_fn, self.evaluator)
        for r in results:
            self.assertIsInstance(r, EvalResult)

    def test_generate_report_has_required_keys(self):
        results = self.runner.run(self.qa_pairs, self.agent_fn, self.evaluator)
        report = self.runner.generate_report(results)
        self.assertIn("avg_faithfulness", report)
        self.assertIn("avg_relevance", report)
        self.assertIn("pass_rate", report)

    def test_generate_report_total_matches_input(self):
        results = self.runner.run(self.qa_pairs, self.agent_fn, self.evaluator)
        report = self.runner.generate_report(results)
        self.assertEqual(report.get("total"), len(self.qa_pairs))

    def test_identify_failures_returns_subset(self):
        all_pass = [_make_eval_result(0.8, 0.8, 0.8)]
        all_fail = [_make_eval_result(0.2, 0.2, 0.2)]
        results = all_pass + all_fail
        failures = self.runner.identify_failures(results, threshold=0.5)
        self.assertEqual(len(failures), 1)
        self.assertAlmostEqual(failures[0].faithfulness, 0.2)

    def test_identify_failures_empty_on_all_pass(self):
        results = [_make_eval_result(0.8, 0.8, 0.8) for _ in range(5)]
        failures = self.runner.identify_failures(results, threshold=0.5)
        self.assertEqual(len(failures), 0)


class TestFailureAnalyzer(unittest.TestCase):

    def setUp(self):
        self.analyzer = FailureAnalyzer()

    def test_categorize_failures_groups_by_type(self):
        failures = [
            _make_eval_result(0.2, 0.8, 0.8, "hallucination"),
            _make_eval_result(0.2, 0.8, 0.8, "hallucination"),
            _make_eval_result(0.8, 0.2, 0.8, "irrelevant"),
        ]
        categories = self.analyzer.categorize_failures(failures)
        self.assertEqual(categories.get("hallucination"), 2)
        self.assertEqual(categories.get("irrelevant"), 1)

    def test_categorize_failures_empty_list(self):
        categories = self.analyzer.categorize_failures([])
        self.assertIsInstance(categories, dict)

    def test_generate_suggestions_returns_list(self):
        failures = [_make_eval_result(0.2, 0.2, 0.2, "hallucination")]
        suggestions = self.analyzer.generate_improvement_suggestions(failures)
        self.assertIsInstance(suggestions, list)

    def test_generate_suggestions_at_least_3(self):
        failures = [
            _make_eval_result(0.2, 0.8, 0.8, "hallucination"),
            _make_eval_result(0.8, 0.2, 0.8, "irrelevant"),
            _make_eval_result(0.8, 0.8, 0.2, "incomplete"),
        ]
        suggestions = self.analyzer.generate_improvement_suggestions(failures)
        self.assertGreaterEqual(len(suggestions), 3)

    def test_suggestions_are_strings(self):
        failures = [_make_eval_result(0.2, 0.2, 0.2, "hallucination")]
        suggestions = self.analyzer.generate_improvement_suggestions(failures)
        for s in suggestions:
            self.assertIsInstance(s, str)


class TestLLMJudge(unittest.TestCase):

    def setUp(self):
        self.judge = LLMJudge(judge_llm_fn=_mock_judge_llm)

    def test_score_response_returns_dict(self):
        result = self.judge.score_response(
            question="What is AI?",
            answer="AI is artificial intelligence.",
            rubric={"accuracy": "Is the answer correct?"},
        )
        self.assertIsInstance(result, dict)

    def test_score_response_has_scores_key(self):
        result = self.judge.score_response("Q", "A", {"accuracy": "desc"})
        self.assertIn("scores", result)

    def test_score_response_has_reasoning_key(self):
        result = self.judge.score_response("Q", "A", {"accuracy": "desc"})
        self.assertIn("reasoning", result)

    def test_detect_bias_returns_dict_with_bias_types(self):
        scores_batch = [{"scores": {"accuracy": 0.9}, "reasoning": "good"}]
        result = self.judge.detect_bias(scores_batch)
        self.assertIn("positional_bias", result)
        self.assertIn("leniency_bias", result)
        self.assertIn("severity_bias", result)


class TestEvalResultOverallScore(unittest.TestCase):
    def _make_result(self, f, r, c):
        qa = QAPair("q", "expected", None, {})
        return EvalResult(
            qa_pair=qa, actual_answer="actual",
            faithfulness=f, relevance=r, completeness=c,
            passed=True, failure_type=None
        )

    def test_correct_average(self):
        result = self._make_result(0.9, 0.8, 0.7)
        self.assertAlmostEqual(result.overall_score(), (0.9 + 0.8 + 0.7) / 3, places=5)

    def test_all_ones_returns_one(self):
        result = self._make_result(1.0, 1.0, 1.0)
        self.assertAlmostEqual(result.overall_score(), 1.0, places=5)

    def test_all_zeros_returns_zero(self):
        result = self._make_result(0.0, 0.0, 0.0)
        self.assertAlmostEqual(result.overall_score(), 0.0, places=5)


class TestRunRegression(unittest.TestCase):
    def _make_result(self, f, r, c, passed=True):
        qa = QAPair("q", "expected", None, {})
        return EvalResult(
            qa_pair=qa, actual_answer="actual",
            faithfulness=f, relevance=r, completeness=c,
            passed=passed, failure_type=None
        )

    def setUp(self):
        self.runner = BenchmarkRunner()
        self.baseline = [self._make_result(0.9, 0.85, 0.8) for _ in range(5)]

    def test_returns_required_keys(self):
        new = [self._make_result(0.85, 0.82, 0.78) for _ in range(5)]
        result = self.runner.run_regression(new, self.baseline)
        for key in ['new_avg_faithfulness', 'baseline_avg_faithfulness', 'regressions', 'passed']:
            self.assertIn(key, result)

    def test_detects_regression_on_score_drop(self):
        # Drop faithfulness by 0.15 — should be detected as regression
        new = [self._make_result(0.7, 0.85, 0.8) for _ in range(5)]
        result = self.runner.run_regression(new, self.baseline)
        self.assertIn('faithfulness', result['regressions'])

    def test_no_regression_when_scores_stable(self):
        new = [self._make_result(0.89, 0.84, 0.79) for _ in range(5)]
        result = self.runner.run_regression(new, self.baseline)
        self.assertEqual(result['regressions'], [])
        self.assertTrue(result['passed'])


class TestGenerateImprovementLog(unittest.TestCase):
    def setUp(self):
        self.analyzer = FailureAnalyzer()
        qa = QAPair("How does RAG work?", "expected", None, {})
        self.failures = [
            EvalResult(qa, "wrong answer", 0.2, 0.3, 0.1, False, "Hallucination"),
            EvalResult(qa, "partial answer", 0.6, 0.4, 0.3, False, "Low_completeness"),
        ]

    def test_returns_string(self):
        log = self.analyzer.generate_improvement_log(self.failures, ["Add guardrails", "Improve retrieval"])
        self.assertIsInstance(log, str)

    def test_contains_open_status(self):
        log = self.analyzer.generate_improvement_log(self.failures, ["fix1", "fix2"])
        self.assertIn("Open", log)

    def test_contains_failure_types(self):
        log = self.analyzer.generate_improvement_log(self.failures, ["fix1", "fix2"])
        self.assertIn("Hallucination", log)

    def test_contains_markdown_table_syntax(self):
        log = self.analyzer.generate_improvement_log(self.failures, ["fix1"])
        self.assertIn("|", log)


if __name__ == "__main__":
    unittest.main()
