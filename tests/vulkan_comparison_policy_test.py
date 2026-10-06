#!/usr/bin/env python3
"""Pure comparison-policy tests; no model files, decoder, backend or GPU."""

import copy
import importlib.util
import math
from pathlib import Path
import sys
import unittest

sys.dont_write_bytecode = True
SOURCE = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("pmi_vulkan_policy", SOURCE / "tests/device_full_model_test.py")
policy = importlib.util.module_from_spec(spec)
spec.loader.exec_module(policy)


def trace():
    records = []
    for evaluation, phase in enumerate(("prefill", "decode")):
        def add(name, values, kind="f32", columns=1):
            records.append({"evaluation": evaluation, "phase": phase, "name": name, "type": kind,
                            "shape": [len(values) // columns, columns, 1, 1], "values": values})
        add("result_output", [.1, .3, -.2, .6])
        add("ffn_input-0", [.2, -.1, .4, .8])
        add("ffn_moe_logits-0", [.2, .1, -.4, -.8])
        add("ffn_moe_topk-0", [0, 1], "i32")
        add("ffn_moe_weights_norm-0", [.6, .4])
        add("ffn_moe_weights_scaled-0", [.6, .4])
        add("ffn_moe_out-0", [.1, -.2, .3, .4])
        add("ffn_out-0", [.15, -.25, .35, .45])
        add("ple_embd", [.3, -.2, .1, .5])
        add("indexer_top_k-3", [0, 2], "i32")
        add("indexer_pool_k_new-3", [1., .5, -1., 2., 2., 1., -.5, .5], columns=2)
    return {"tokens": [7, 9], "records": records}


def row(value, name, evaluation=0):
    matches = [record for record in value["records"] if record["name"] == name and record["evaluation"] == evaluation]
    assert len(matches) == 1
    return matches[0]


def nmse(expected, actual):
    return sum((a - b) ** 2 for a, b in zip(expected, actual)) / sum(value * value for value in expected)


class VulkanComparisonPolicyTest(unittest.TestCase):
    def setUp(self):
        self.reference = trace()
        self.actual = copy.deepcopy(self.reference)

    def compare(self):
        return policy.compare_vulkan_policy(self.reference, self.actual)

    def test_strict_positive_and_complete_accounting(self):
        row(self.actual, "ffn_input-0")["values"][0] += 1e-6
        # An exactly zero reference vector is valid when the actual vector is zero.
        for value in (self.reference, self.actual):
            row(value, "indexer_pool_k_new-3")["values"][:4] = [0.] * 4
        policy.full.compare(self.reference, self.actual, atol=policy.ATOL, rtol=policy.RTOL)
        result = self.compare()
        floats = sum(len(item["values"]) for item in self.reference["records"] if item["type"] == "f32")
        integers = sum(len(item["values"]) for item in self.reference["records"] if item["type"] == "i32")
        self.assertEqual(result["compared_records"], len(self.reference["records"]))
        self.assertEqual(result["compared_floats"], floats)
        self.assertEqual(result["compared_ids"], integers)
        self.assertEqual(result["strict_nonpooled_compared_floats"], floats - 16)
        self.assertEqual(result["pooled_keys"]["vectors"], 4)
        self.assertEqual(result["pooled_keys"]["values"], 16)
        self.assertEqual(result["pooled_keys"]["max_nmse"], 0)

    def test_pooled_policy_positive_is_distinct_from_uniform_strict_parity(self):
        row(self.actual, "indexer_pool_k_new-3")["values"][0] += 2e-4
        with self.assertRaisesRegex(AssertionError, "activation mismatch"):
            policy.full.compare(self.reference, self.actual, atol=policy.ATOL, rtol=policy.RTOL)
        result = self.compare()
        self.assertGreater(result["pooled_keys"]["max_nmse"], 0)
        self.assertLess(result["pooled_keys"]["max_nmse"], 1e-7)

    def test_pooled_vector_large_error_rejected(self):
        row(self.actual, "indexer_pool_k_new-3")["values"][0] += .1
        with self.assertRaisesRegex(AssertionError, "pooled key NMSE"):
            self.compare()

    def test_nonpooled_scalar_error_rejected(self):
        row(self.actual, "ffn_input-0")["values"][0] += .01
        with self.assertRaisesRegex(AssertionError, "activation mismatch"):
            self.compare()

    def test_integer_errors_rejected_for_router_and_indexer(self):
        for name in ("ffn_moe_topk-0", "indexer_top_k-3"):
            with self.subTest(name=name):
                self.actual = copy.deepcopy(self.reference)
                row(self.actual, name)["values"][0] = 99
                with self.assertRaises(AssertionError):
                    self.compare()

    def test_missing_shape_and_duplicate_snapshot_coverage_rejected(self):
        for mutation in ("missing", "shape", "duplicate"):
            with self.subTest(mutation=mutation):
                self.actual = copy.deepcopy(self.reference)
                if mutation == "missing":
                    self.actual["records"].pop()
                elif mutation == "shape":
                    row(self.actual, "indexer_pool_k_new-3")["shape"] = [8, 1, 1, 1]
                else:
                    self.actual["records"].append(copy.deepcopy(row(self.actual, "ffn_moe_topk-0")))
                with self.assertRaisesRegex(AssertionError, "coverage"):
                    self.compare()

    def test_nonfinite_pooled_values_rejected(self):
        for value in (math.nan, math.inf, -math.inf):
            for side in ("reference", "actual"):
                with self.subTest(value=value, side=side):
                    self.reference, self.actual = trace(), trace()
                    row(getattr(self, side), "indexer_pool_k_new-3")["values"][0] = value
                    with self.assertRaises(AssertionError):
                        self.compare()

    def test_zero_reference_requires_exact_zero_actual(self):
        row(self.reference, "indexer_pool_k_new-3")["values"][:4] = [0.] * 4
        row(self.actual, "indexer_pool_k_new-3")["values"][:4] = [1e-30, 0., 0., 0.]
        with self.assertRaisesRegex(AssertionError, "nonzero pooled key against zero reference"):
            self.compare()

    def test_one_bad_column_cannot_hide_among_one_thousand_good_columns(self):
        vector = [1., .5, -1., 2.]
        for value in (self.reference, self.actual):
            record = row(value, "indexer_pool_k_new-3")
            record["shape"] = [4, 1001, 1, 1]
            record["values"] = vector * 1001
        expected = row(self.reference, "indexer_pool_k_new-3")["values"]
        actual = row(self.actual, "indexer_pool_k_new-3")["values"]
        actual[0] += .01
        self.assertLess(nmse(expected, actual), 1e-7, "control must expose a whole-record averaging loophole")
        self.assertGreater(nmse(expected[:4], actual[:4]), 1e-7)
        with self.assertRaisesRegex(AssertionError, "pooled key NMSE"):
            self.compare()

    def test_one_bad_evaluation_cannot_hide_behind_another_evaluation(self):
        row(self.actual, "indexer_pool_k_new-3")["values"][0] += .01
        for value in (self.reference, self.actual):
            record = row(value, "indexer_pool_k_new-3", evaluation=1)
            record["values"] = [x * 1000 for x in record["values"]]
        expected = sum((row(self.reference, "indexer_pool_k_new-3", evaluation=i)["values"] for i in range(2)), [])
        actual = sum((row(self.actual, "indexer_pool_k_new-3", evaluation=i)["values"] for i in range(2)), [])
        self.assertLess(nmse(expected, actual), 1e-7, "control must expose cross-evaluation averaging")
        with self.assertRaisesRegex(AssertionError, "pooled key NMSE"):
            self.compare()

    def test_similarly_named_snapshot_keeps_scalar_tolerance(self):
        for value in (self.reference, self.actual):
            row(value, "indexer_pool_k_new-3")["name"] = "indexer_pool_k_new-3 (view)"
        row(self.actual, "indexer_pool_k_new-3 (view)")["values"][0] += 2e-4
        with self.assertRaisesRegex(AssertionError, "activation mismatch"):
            self.compare()

    def test_tokens_remain_exact(self):
        self.actual["tokens"][-1] += 1
        with self.assertRaisesRegex(AssertionError, "greedy token mismatch"):
            self.compare()


if __name__ == "__main__":
    unittest.main()
