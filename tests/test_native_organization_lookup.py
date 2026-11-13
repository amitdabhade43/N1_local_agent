import unittest
from unittest.mock import patch

import ninja_agent


class NativeOrganizationLookupTests(unittest.TestCase):
    def test_natural_language_organization_name_request(self):
        rows = [
            {"id": 6, "name": "HOERBIGER"},
            {"id": 7, "name": "HOERBIGER - Unattended"},
        ]
        with patch.object(ninja_agent, "op_by_path", return_value={"path": "/v2/organizations"}), patch.object(
            ninja_agent, "fetch", return_value=rows
        ):
            answer = ninja_agent.recipe_list_organizations("can i get organization name")

        self.assertIn("2 organization(s)", answer)
        self.assertIn("HOERBIGER", answer)
        self.assertIn("HOERBIGER - Unattended", answer)

    def test_count_request_is_left_for_count_handler(self):
        self.assertIsNone(ninja_agent.recipe_list_organizations("how many organizations?"))

    def test_unrelated_question_is_not_claimed(self):
        self.assertIsNone(ninja_agent.recipe_list_organizations("show all technicians"))


if __name__ == "__main__":
    unittest.main()
