import unittest
from tracking_policy import init_arguments


class TrackingPolicyTests(unittest.TestCase):
    def policy(self):
        return dict(entity='example-team', project='example-research', aggregate_only=True, allow_online=True)

    def test_new_and_resume_are_explicit(self):
        for resume, expected in ((False, 'never'), (True, 'must')):
            args = init_arguments(self.policy(), run_id='a'*32, resume=resume)
            self.assertEqual(args['resume'], expected)
            self.assertEqual(args['entity'], 'example-team')
            self.assertTrue(args['force'])
            self.assertEqual(args['reinit'], 'create_new')

    def test_no_defaults_or_secret_fields(self):
        for key in self.policy():
            value = self.policy()
            del value[key]
            with self.assertRaises(ValueError):
                init_arguments(value, run_id='a'*32, resume=False)
        value = self.policy()
        value['token'] = 'not-a-credential'
        with self.assertRaises(ValueError):
            init_arguments(value, run_id='a'*32, resume=False)

    def test_refuses_unapproved_or_bad_identity(self):
        for key in ('allow_online', 'aggregate_only'):
            value = self.policy()
            value[key] = False
            with self.assertRaises(ValueError):
                init_arguments(value, run_id='a'*32, resume=False)
        with self.assertRaises(ValueError):
            init_arguments(self.policy(), run_id='old/path', resume=False)


if __name__ == '__main__':
    unittest.main()
