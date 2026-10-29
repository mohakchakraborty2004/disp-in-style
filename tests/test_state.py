import unittest

from disappear_in_style.state import State, StateMachine


class StateMachineTest(unittest.TestCase):
    def make(self):
        return StateMachine(disintegrate_seconds=2.0, reintegrate_seconds=1.0)

    def test_full_cycle(self):
        sm = self.make()
        self.assertIs(sm.state, State.VISIBLE)
        self.assertEqual(sm.progress, 0.0)
        self.assertTrue(sm.snap())
        self.assertIs(sm.state, State.DISINTEGRATING)
        sm.update(1.0)
        self.assertIs(sm.state, State.DISINTEGRATING)
        self.assertAlmostEqual(sm.progress, 0.5)
        sm.update(1.0)
        self.assertIs(sm.state, State.GONE)
        self.assertEqual(sm.progress, 1.0)
        self.assertTrue(sm.snap())
        self.assertIs(sm.state, State.REINTEGRATING)
        sm.update(0.5)
        self.assertAlmostEqual(sm.progress, 0.5)
        sm.update(0.6)
        self.assertIs(sm.state, State.VISIBLE)

    def test_snap_ignored_mid_transition(self):
        sm = self.make()
        sm.snap()
        sm.update(0.5)
        self.assertFalse(sm.snap())
        self.assertIs(sm.state, State.DISINTEGRATING)
        self.assertAlmostEqual(sm.progress, 0.25)

    def test_stable_states_do_not_advance(self):
        sm = self.make()
        sm.update(100)
        self.assertIs(sm.state, State.VISIBLE)
        sm.snap()
        sm.update(5)
        sm.update(100)
        self.assertIs(sm.state, State.GONE)

    def test_bad_durations(self):
        with self.assertRaises(ValueError):
            StateMachine(0, 1)


if __name__ == "__main__":
    unittest.main()
