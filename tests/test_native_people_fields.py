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

    def test_named_user_device_access_is_resolved_without_llm(self):
        user = {
            "id": 42, "firstName": "Richard", "lastName": "Otrebski",
            "email": "richard@example.com", "accessibleDeviceIds": [10],
            "assignedDeviceIds": [11], "deviceIds": [10, 11], "match": 0.91,
        }
        devices = [
            {"id": 10, "systemName": "RICHARD-LAPTOP"},
            {"id": 11, "systemName": "RICHARD-DESKTOP"},
        ]
        with patch.object(ninja_agent, "t_resolve", return_value='{"matches": [' + __import__("json").dumps(user) + ']}'), patch.object(
            ninja_agent, "op_by_path", return_value={"path": "/v2/devices"}
        ), patch.object(ninja_agent, "fetch", return_value=devices):
            answer = ninja_agent.fast_path("can you give me device access of richard otrebaski")
        self.assertIn("Richard Otrebski", answer)
        self.assertIn("RICHARD-LAPTOP", answer)
        self.assertIn("RICHARD-DESKTOP", answer)
        self.assertIn("Accessible devices (1)", answer)
        self.assertIn("Assigned devices (1)", answer)

    def test_named_user_device_access_asks_for_clarification_when_ambiguous(self):
        payload = {
            "matches": [
                {"firstName": "Richard", "lastName": "Otrebski", "match": 0.94},
                {"firstName": "Richard", "lastName": "Other", "match": 0.90},
            ]
        }
        with patch.object(ninja_agent, "t_resolve", return_value=__import__("json").dumps(payload)):
            answer = ninja_agent.recipe_person_device_access(
                "can you give me device access of richard otrebaski"
            )
        self.assertIn("Multiple NinjaOne users", answer)
        self.assertIn("Please confirm", answer)

    def test_non_field_question_is_not_claimed(self):
        self.assertIsNone(ninja_agent.recipe_list_entity_fields("how many technicians?"))


if __name__ == "__main__":
    unittest.main()
