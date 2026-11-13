import json
import os
import unittest
from pathlib import Path

from unittest.mock import patch

import ninja_agent
from ninja_openapi import build_operations


class BundledNinjaSpecTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        spec_path = Path(__file__).resolve().parents[1] / "ninja_spec.json"
        with spec_path.open("r", encoding="utf-8") as handle:
            cls.spec = json.load(handle)
        cls.operations = build_operations(cls.spec)

    def test_load_ops_uses_bundled_spec_without_network_download(self):
        with patch.dict(os.environ, {"NINJA_SPEC_REFRESH": "0"}), patch.object(
            ninja_agent.requests, "get"
        ) as download:
            operations = ninja_agent.load_ops()
        download.assert_not_called()
        self.assertEqual(len(operations), 285)

    def test_bundled_spec_catalog_counts_match_expected_api_surface(self):
        counts = {}
        for operation in self.operations.values():
            method = operation["method"]
            counts[method] = counts.get(method, 0) + 1
        self.assertEqual(len(self.operations), 285)
        self.assertEqual(counts, {"GET": 151, "POST": 84, "PATCH": 30, "PUT": 20})

    def test_delete_operations_are_not_exposed(self):
        self.assertNotIn("deleteContact", self.operations)
        self.assertNotIn("deleteNodeRoles", self.operations)
        self.assertNotIn("deleteTags", self.operations)
        self.assertNotIn("deleteKnowledgeBaseArticles", self.operations)
        self.assertNotIn("removeTicketProducts", self.operations)

    def test_read_only_ticket_board_post_is_distinguished_from_mutations(self):
        self.assertTrue(self.operations["getTicketsByBoard"]["read_only"])
        self.assertFalse(self.operations["create_2"]["read_only"])
        self.assertFalse(self.operations["runScriptOnDevice"]["read_only"])

    def test_wildcard_json_request_media_type_is_normalized(self):
        self.assertEqual(
            self.operations["createInvoice"]["request_body"]["media_type"],
            "application/json",
        )
        self.assertEqual(
            self.operations["approveInvoices"]["request_body"]["media_type"],
            "application/json",
        )


if __name__ == "__main__":
    unittest.main()
