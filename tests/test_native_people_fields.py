import unittest
from unittest.mock import patch

import ninja_agent


class NativePeopleFieldLookupTests(unittest.TestCase):
    def _run(self, question, path, rows):
        with patch.object(ninja_agent, "op_by_path", return_value={"path": path}), patch.object(
            ninja_agent, "fetch", return_value=rows
        ):
            return ninja_agent.recipe_list_entity_fields(question)

    def test_end_users_first_and_last_name_with_common_typo(self):
        answer = self._run(
            "list of end users with fisrtname and lastname",
            "/v2/user/end-users",
            [
                {"firstName": "Rene", "lastName": "Example", "email": "rene@example.com"},
                {"firstName": "Alex", "lastName": "Smith"},
            ],
        )
        self.assertIn("2 end user(s)", answer)
        self.assertIn("Rene Example", answer)
        self.assertIn("Alex Smith", answer)
        self.assertNotIn("rene@example.com", answer)

    def test_technician_first_name_only(self):
        answer = self._run(
            "can you give me list of technician with firstname",
            "/v2/user/technicians",
            [{"firstName": "Rene", "lastName": "Example"}, {"firstName": "Alex"}],
        )
        self.assertIn("2 technician(s)", answer)
        self.assertIn("Rene", answer)
        self.assertIn("Alex", answer)

    def test_user_email_field(self):
        answer = self._run(
            "show users email",
            "/v2/users",
            [{"firstName": "Rene", "email": "rene@example.com"}],
        )
        self.assertIn("rene@example.com", answer)
        self.assertNotIn("Rene", answer)

    def test_non_field_question_is_not_claimed(self):
        self.assertIsNone(ninja_agent.recipe_list_entity_fields("how many technicians?"))


if __name__ == "__main__":
    unittest.main()
