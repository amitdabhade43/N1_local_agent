import json
import unittest
from unittest.mock import patch

import ninja_agent


class NativeTextCallTests(unittest.TestCase):
    def test_python_keyword_tool_call(self):
        self.assertEqual(
            ninja_agent.text_call('resolve_entity(kind="user", name="rene")'),
            ("resolve_entity", {"kind": "user", "name": "rene"}),
        )

    def test_json_argument_tool_call(self):
        self.assertEqual(
            ninja_agent.text_call('resolve_entity({"kind": "user", "name": "rene"})'),
            ("resolve_entity", {"kind": "user", "name": "rene"}),
        )

    def test_rejects_non_literal_python_expression(self):
        self.assertIsNone(
            ninja_agent.text_call('resolve_entity(kind=__import__("os"), name="rene")')
        )

    def test_named_user_count_uses_total_matches_not_preview_length(self):
        payload = {
            "total_matches": 7,
            "matches": [
                {"firstName": "Rene", "lastName": "Example", "email": "rene@example.com"}
            ],
        }
        with patch.object(ninja_agent, "t_resolve", return_value=json.dumps(payload)):
            answer = ninja_agent.recipe_count_named("how many users are there with name rene?")
        self.assertIn("7 NinjaOne user record(s)", answer)
        self.assertIn("rene@example.com", answer)


if __name__ == "__main__":
    unittest.main()
