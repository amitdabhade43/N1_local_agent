import unittest

import ninja_native


class NativeMutationApprovalGuardTests(unittest.TestCase):
    def test_detects_direct_state_change_requests(self):
        samples = [
            "add device 2QN9BY3 to PC Admin Rene Piuk device access list now",
            "Please grant this user access to the device",
            "update the technician role",
            "disable this user",
            "create a new organization",
        ]
        for sample in samples:
            with self.subTest(sample=sample):
                self.assertTrue(ninja_native._is_mutation_request(sample))

    def test_does_not_treat_read_requests_as_mutations(self):
        samples = [
            "show PC Admin Rene Piuk device access list",
            "how many devices are online?",
            "find device 2QN9BY3",
            "what is the user's role?",
        ]
        for sample in samples:
            with self.subTest(sample=sample):
                self.assertFalse(ninja_native._is_mutation_request(sample))


if __name__ == "__main__":
    unittest.main()
