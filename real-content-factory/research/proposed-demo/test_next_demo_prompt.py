"""Model-free response-contract tests; candidate strings are never executed.

Only the standalone audit module is imported. No production module, model,
tokenizer, runtime metadata, or candidate executor is loaded by this suite.
"""
import copy
import hashlib
import json
import unittest
from unittest import mock

from next_demo_prompt import (
    fixed_proposal, make_prompt, profile_manifest,
    serialized_content_bytes, validate_response,
)


TEXT_CAPS = {"reflection": 192, "debate": 192, "meta-review": 192, "overview": 512}
CODE_ROLES = ("generation", "evolution")
CODE = "def merge_intervals(intervals):\n    return []\n"


def at_content_cap(prefix, cap):
    """Construct a boundary specimen using the public JSON wire convention."""
    used = len(json.dumps(prefix, ensure_ascii=True).encode("utf-8")) - 2
    if used > cap:
        raise AssertionError("test specimen prefix exceeds its cap")
    return prefix + "x" * (cap - used)


class SerializedContentBytesTests(unittest.TestCase):
    def test_known_wire_counts_include_escaping_and_exclude_outer_quotes(self):
        specimens = {"": 0, "abc": 3, '"': 2, "\\": 2, "\n": 2,
                     "\t": 2, "\x00": 6, "é": 6, "漢": 6, "😀": 12,
                     "é\n😀": 20}
        for value, expected in specimens.items():
            with self.subTest(value=repr(value)):
                self.assertEqual(serialized_content_bytes(value), expected)

    def test_raw_utf8_length_is_not_the_serialized_content_budget(self):
        self.assertEqual(len("é😀".encode("utf-8")), 6)
        self.assertEqual(serialized_content_bytes("é😀"), 18)


class TextResponseTests(unittest.TestCase):
    def test_every_text_role_accepts_exact_ascii_cap_and_rejects_one_more(self):
        for role, cap in TEXT_CAPS.items():
            with self.subTest(role=role):
                reply = "x" * cap
                self.assertEqual(validate_response(role, reply), reply)
                with self.assertRaises(ValueError):
                    validate_response(role, reply + "x")

    def test_escaped_and_non_ascii_text_uses_serialized_caps(self):
        for role, cap in TEXT_CAPS.items():
            for prefix in ('quoted "slash\\ line\n', "é漢😀"):
                with self.subTest(role=role, prefix=prefix):
                    reply = at_content_cap(prefix, cap)
                    self.assertEqual(serialized_content_bytes(reply), cap)
                    self.assertEqual(validate_response(role, reply), reply)
                    with self.assertRaises(ValueError):
                        validate_response(role, reply + "x")

    def test_empty_and_whitespace_only_text_is_rejected(self):
        for role in TEXT_CAPS:
            for reply in ("", " ", "\n\t"):
                with self.subTest(role=role, reply=repr(reply)):
                    with self.assertRaises(ValueError):
                        validate_response(role, reply)

    def test_response_must_be_a_string(self):
        for role in (*TEXT_CAPS, *CODE_ROLES, "ranking"):
            for reply in (None, 7, [], {"winner": 1, "rationale": "valid"}):
                with self.subTest(role=role, reply=reply):
                    with self.assertRaises((TypeError, ValueError)):
                        validate_response(role, reply, targets=(1, 2))


class CodeResponseTests(unittest.TestCase):
    def test_code_is_returned_unchanged(self):
        for role in CODE_ROLES:
            with self.subTest(role=role):
                self.assertEqual(validate_response(role, CODE), CODE)

    def test_ascii_escaped_and_unicode_code_at_exact_640_content_bytes(self):
        for role in CODE_ROLES:
            for comment in ("# ", '# quoted " slash\\ ', "# é漢😀 "):
                with self.subTest(role=role, comment=comment):
                    reply = at_content_cap(CODE + comment, 640)
                    self.assertEqual(serialized_content_bytes(reply), 640)
                    self.assertEqual(validate_response(role, reply), reply)
                    with self.assertRaises(ValueError):
                        validate_response(role, reply + "x")

    def test_imports_helpers_and_top_level_statements_are_rejected(self):
        invalid = (
            "import math\n" + CODE,
            "from math import floor\n" + CODE,
            "helper = 1\n" + CODE,
            "def helper():\n    return []\n" + CODE,
            "def merge_intervals(intervals):\n    def helper():\n        return []\n    return []\n",
            "class Helper:\n    pass\n" + CODE,
        )
        for source in invalid:
            for role in CODE_ROLES:
                with self.subTest(role=role, source=source):
                    with self.assertRaises(ValueError):
                        validate_response(role, source)

    def test_fixed_signature_and_no_annotations_or_decorators(self):
        headers = (
            "def other(intervals):", "def merge_intervals(items):",
            "def merge_intervals(intervals, extra):",
            "def merge_intervals(intervals=[]):",
            "def merge_intervals(intervals: list):",
            "def merge_intervals(intervals) -> list:",
            "def merge_intervals(intervals, /):",
            "def merge_intervals(*intervals):",
            "def merge_intervals(intervals, **kwargs):",
            "def merge_intervals(intervals, *, extra=1):",
            "@staticmethod\ndef merge_intervals(intervals):",
            "async def merge_intervals(intervals):",
        )
        for header in headers:
            with self.subTest(header=header):
                with self.assertRaises(ValueError):
                    validate_response("generation", header + "\n    return []\n")

    def test_dict_get_items_and_other_unlisted_calls_are_rejected(self):
        expressions = (
            "{}.get(1, 0)", "{}.items()", "sum(intervals)",
            "set(intervals)", "int(1)", "merge_intervals(intervals)",
            "open('/tmp/not-opened')", "print(intervals)",
            "eval('1')", "exec('pass')", "(lambda: [])()",
            "intervals.__class__", "globals()",
        )
        for expression in expressions:
            with self.subTest(expression=expression):
                source = "def merge_intervals(intervals):\n    return " + expression + "\n"
                with self.assertRaises(ValueError):
                    validate_response("generation", source)

    def test_current_safe_builtin_and_attribute_subset_remains_accepted(self):
        source = """def merge_intervals(intervals):
    out = list(intervals)
    out.sort()
    copied = out.copy()
    copied.append((0, 1))
    copied.extend(tuple(zip(range(len(out)), reversed(out))))
    pairs = list(enumerate(sorted(copied)))
    low = min(0, 1)
    high = max(0, 1)
    return [[low, high]]
"""
        self.assertLessEqual(serialized_content_bytes(source), 640)
        self.assertEqual(validate_response("generation", source), source)

    def test_markdown_fences_and_invalid_python_are_rejected(self):
        for source in ("```python\n" + CODE + "```", "def merge_intervals(intervals):\n    return ["):
            with self.subTest(source=source):
                with self.assertRaises((SyntaxError, ValueError)):
                    validate_response("generation", source)

    def test_validation_never_executes_the_candidate_source(self):
        source = "def merge_intervals(intervals):\n    return 1 // 0\n"
        with mock.patch("builtins.exec", side_effect=AssertionError("candidate exec forbidden")) as execute:
            with mock.patch("builtins.eval", side_effect=AssertionError("candidate eval forbidden")) as evaluate:
                self.assertEqual(validate_response("generation", source), source)
                with self.assertRaises(ValueError):
                    validate_response("generation", "raise RuntimeError('must never execute')\n")
                execute.assert_not_called()
                evaluate.assert_not_called()


class RankingResponseTests(unittest.TestCase):
    def test_target_winners_and_null_tie_are_accepted(self):
        for winner in (3, 7, None):
            with self.subTest(winner=winner):
                expected = {"winner": winner, "rationale": "Bounded evidence only."}
                self.assertEqual(validate_response("ranking", json.dumps(expected), targets=(3, 7)), expected)

    def test_exact_96_byte_rationale_caps_cover_escaping_and_unicode(self):
        for prefix in ("ascii ", 'quoted "slash\\ ', "é漢😀"):
            with self.subTest(prefix=prefix):
                rationale = at_content_cap(prefix, 96)
                expected = {"winner": None, "rationale": rationale}
                self.assertEqual(validate_response("ranking", json.dumps(expected), targets=(1, 2)), expected)
                expected["rationale"] += "x"
                with self.assertRaises(ValueError):
                    validate_response("ranking", json.dumps(expected), targets=(1, 2))

    def test_bool_float_string_and_out_of_scope_winners_are_rejected(self):
        for winner in (True, False, 1.0, "1", 0, 3, -1, [], {}):
            with self.subTest(winner=winner):
                with self.assertRaises(ValueError):
                    validate_response("ranking", json.dumps({"winner": winner, "rationale": "Evidence."}), targets=(1, 2))

    def test_duplicate_keys_are_rejected_including_escaped_equivalent_key(self):
        replies = (
            '{"winner":1,"winner":2,"rationale":"Evidence."}',
            '{"winner":1,"rationale":"first","rationale":"second"}',
            '{"winner":1,"\\u0077inner":2,"rationale":"Evidence."}',
        )
        for reply in replies:
            with self.subTest(reply=reply):
                with self.assertRaises(ValueError):
                    validate_response("ranking", reply, targets=(1, 2))

    def test_ranking_requires_exact_keys_object_and_nonempty_string_rationale(self):
        values = (
            [], [1, "Evidence"], None,
            {"winner": 1}, {"rationale": "Evidence"},
            {"winner": 1, "rationale": "Evidence", "score": 2},
            {"winner": 1, "rationale": ""},
            {"winner": 1, "rationale": " \n"},
            {"winner": 1, "rationale": 1},
            {"winner": 1, "rationale": None},
            {"winner": 1, "rationale": {}},
        )
        for value in values:
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    validate_response("ranking", json.dumps(value), targets=(1, 2))

    def test_fences_trailing_prose_and_non_json_are_rejected(self):
        valid = '{"winner":1,"rationale":"Evidence."}'
        for reply in ("```json\n" + valid + "\n```", valid + " explanation", "winner=1"):
            with self.subTest(reply=reply):
                with self.assertRaises(ValueError):
                    validate_response("ranking", reply, targets=(1, 2))


class PromptStateTests(unittest.TestCase):
    def setUp(self):
        proposal = fixed_proposal()
        encoded_evaluator = json.dumps(proposal["evaluator"], sort_keys=True,
                                       separators=(",", ":"), allow_nan=False).encode()
        self.state = {
            "config": proposal,
            "mode": "fixture",
            "evaluator_sha256": hashlib.sha256(encoded_evaluator).hexdigest(),
            "tasks": [], "candidates": [], "matches": [], "meta_reviews": [],
        }

    def add_candidate(self, identity=1):
        source = CODE + "# candidate " + str(identity) + "\n"
        self.state["candidates"].append({
            "id": identity, "source": source,
            "sha256": hashlib.sha256(source.encode()).hexdigest(),
            "duplicate_of": None, "reflection": "Fixture review; no real execution.",
            "evidence": {
                "receipt": {
                    "measurements": {"tests_passed": 884, "tests_total": 884,
                                     "baseline_ns": None, "candidate_ns": None,
                                     "baseline_sha256": None},
                    "retained_fixture_detail": {"marker": "FULL_RECEIPT_REMAINS_IN_STATE"},
                },
                "evaluation": {"correct": True, "evaluator_sha256": self.state["evaluator_sha256"]},
            },
        })

    def add_task(self, role, targets=()):
        task = {"id": "seed-1" if role == "generation" else role + "-fixture",
                "kind": role, "targets": list(targets), "status": "pending"}
        self.state["tasks"].append(task)
        return task

    def test_first_generation_render_is_bounded_and_preserves_state(self):
        task = self.add_task("generation")
        before = copy.deepcopy(self.state)
        rendered = make_prompt(self.state, task)
        payload = json.loads(rendered.split("\n", 1)[1])
        snapshot = json.dumps(before, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
        self.assertEqual(payload["snapshot_sha256"], hashlib.sha256(snapshot).hexdigest())
        self.assertEqual(self.state, before)
        self.assertEqual(rendered, make_prompt(self.state, task))
        self.assertLessEqual(len(rendered.encode()), 4096 - 512 - 136)
        self.assertIn("immutable tuple", rendered)
        self.assertIn("discard empty", rendered)
        self.assertIn("coalesce touching", rendered)
        self.assertIn("no .get/.items", rendered)
        self.assertIn("no fences", rendered)

    def test_reflection_render_keeps_candidate_and_receipt_unchanged(self):
        self.add_candidate()
        task = self.add_task("reflection", (1,))
        before = copy.deepcopy(self.state)
        rendered = make_prompt(self.state, task)
        payload = json.loads(rendered.split("\n", 1)[1])
        self.assertEqual(payload["candidates"][0][1], self.state["candidates"][0]["source"])
        self.assertEqual(self.state, before)
        self.assertLessEqual(len(rendered.encode()), 4096 - 128 - 136)

    def test_aggregate_projects_source_identity_without_discarding_persisted_data(self):
        self.add_candidate()
        task = self.add_task("overview", (1,))
        before = copy.deepcopy(self.state)
        rendered = make_prompt(self.state, task)
        payload = json.loads(rendered.split("\n", 1)[1])
        self.assertEqual(payload["candidates"][0][1], self.state["candidates"][0]["sha256"])
        self.assertNotIn("FULL_RECEIPT_REMAINS_IN_STATE", rendered)
        self.assertEqual(self.state, before)
        self.assertLessEqual(len(rendered.encode()), 4096 - 128 - 136)

    def test_changed_goal_model_options_budget_sources_or_evaluator_are_rejected(self):
        task = self.add_task("generation")
        mutations = (
            lambda s: s["config"].update(goal="A different problem"),
            lambda s: s["config"].update(model_id="different-model"),
            lambda s: s["config"]["options"].update(proximity="model-semantic"),
            lambda s: s["config"]["budget"].update(wall_seconds=2400),
            lambda s: s["config"].update(sources=[{"id": "unreviewed"}]),
            lambda s: s["config"]["evaluator"].update(contract="changed"),
            lambda s: s.update(evaluator_sha256="0" * 64),
        )
        for mutation in mutations:
            changed = copy.deepcopy(self.state)
            mutation(changed)
            before = copy.deepcopy(changed)
            with self.subTest(changed=changed):
                with self.assertRaises(ValueError):
                    make_prompt(changed, task)
                self.assertEqual(changed, before)

    def test_task_must_be_present_in_persisted_state(self):
        task = {"id": "seed-1", "kind": "generation", "targets": [], "status": "pending"}
        before = copy.deepcopy(self.state)
        with self.assertRaises(ValueError):
            make_prompt(self.state, task)
        self.assertEqual(self.state, before)

    def test_evidence_mode_must_be_explicit(self):
        task = self.add_task("generation")
        for mode in (None, "", "qualified"):
            with self.subTest(mode=mode):
                self.state["mode"] = mode
                before = copy.deepcopy(self.state)
                with self.assertRaises(ValueError):
                    make_prompt(self.state, task)
                self.assertEqual(self.state, before)

    def test_changed_evaluator_receipt_or_inconsistent_correctness_is_rejected(self):
        self.add_candidate()
        task = self.add_task("reflection", (1,))
        for field, value in (("evaluator_sha256", "0" * 64), ("correct", False)):
            changed = copy.deepcopy(self.state)
            changed["candidates"][0]["evidence"]["evaluation"][field] = value
            before = copy.deepcopy(changed)
            with self.subTest(field=field):
                with self.assertRaises(ValueError):
                    make_prompt(changed, task)
                self.assertEqual(changed, before)

    def test_duplicate_bool_and_out_of_scope_targets_are_rejected(self):
        self.add_candidate()
        for targets in ((1, 1), (True,), (0,), (2,), (-1,)):
            with self.subTest(targets=targets):
                task = self.add_task("reflection", targets)
                before = copy.deepcopy(self.state)
                with self.assertRaises(ValueError):
                    make_prompt(self.state, task)
                self.assertEqual(self.state, before)

    def test_duplicate_receipt_reuse_preserves_original_and_duplicate_records(self):
        self.add_candidate(1)
        self.add_candidate(2)
        first, duplicate = self.state["candidates"]
        duplicate.update(source=first["source"], sha256=first["sha256"],
                         duplicate_of=1, evidence=None)
        task = self.add_task("reflection", (2,))
        before = copy.deepcopy(self.state)
        payload = json.loads(make_prompt(self.state, task).split("\n", 1)[1])
        self.assertEqual(payload["candidates"][0][2], [884, 884, True])
        self.assertEqual(self.state, before)

    def test_duplicate_bool_out_of_range_self_and_changed_source_are_rejected(self):
        self.add_candidate(1)
        self.add_candidate(2)
        task = self.add_task("reflection", (2,))
        for duplicate_of in (True, False, 0, -1, 2, 3, 1):
            changed = copy.deepcopy(self.state)
            # The two valid source hashes identify different source strings, so
            # even the otherwise in-range predecessor 1 cannot be a duplicate.
            changed["candidates"][1]["duplicate_of"] = duplicate_of
            before = copy.deepcopy(changed)
            with self.subTest(duplicate_of=duplicate_of):
                with self.assertRaises(ValueError):
                    make_prompt(changed, task)
                self.assertEqual(changed, before)


class ProfileManifestTests(unittest.TestCase):
    def test_profile_is_inactive_and_does_not_claim_fit_or_execution(self):
        manifest = profile_manifest()
        self.assertTrue(manifest["status"].startswith("INACTIVE"))
        self.assertIs(manifest["actual_fit_proof"], False)
        self.assertIs(manifest["network"], False)
        self.assertEqual(manifest["actual_model_calls"], 0)
        self.assertEqual(manifest["actual_candidate_executions"], 0)

    def test_profile_records_exact_proposed_caps_and_separate_calibration_scope(self):
        manifest = profile_manifest()
        self.assertEqual(manifest["model_id"], "bonsai-stock")
        self.assertEqual(manifest["context_tokens"], 4096)
        self.assertEqual(manifest["code_output_tokens"], 512)
        self.assertEqual(manifest["other_output_tokens"], 128)
        self.assertEqual(manifest["wrapper_boundary_slack_reserve"], 136)
        self.assertEqual(manifest["serialized_string_content_caps"],
                         {**TEXT_CAPS, "generation": 640, "evolution": 640, "ranking": 96})
        self.assertEqual(manifest["workflow_model_calls"], 18)
        self.assertEqual(manifest["experiment_calls"], 3)
        self.assertEqual(manifest["proposed_workflow_wall_seconds"], 2400)
        self.assertEqual(manifest["separate_calibration_model_calls"], 1)
        self.assertEqual(manifest["separate_calibration_seconds"], 120)
        self.assertEqual(manifest["combined_max_model_calls"], 19)
        self.assertEqual(manifest["combined_max_seconds"], 2520)
        self.assertEqual(manifest["source_proposal_wall_seconds"], 1200)

    def test_returned_manifest_caps_do_not_mutate_future_manifests(self):
        manifest = profile_manifest()
        manifest["serialized_string_content_caps"]["generation"] = 99999
        manifest["code_output_tokens"] = 99999
        subsequent = profile_manifest()
        self.assertEqual(subsequent["serialized_string_content_caps"]["generation"], 640)
        self.assertEqual(subsequent["code_output_tokens"], 512)


if __name__ == "__main__":
    unittest.main()
