# python -m f1_solver_test

import unittest
import numpy as np
import f1_solver
from pulp import getSolver

class TestF1Solver(unittest.TestCase):
    def test_check_shapes_correct(self):
        cs, ps = np.zeros((3,2)), np.ones((5,3))
        nums = np.array([3, 5])
        self.assertTrue(f1_solver.check_shapes(cs, ps, nums, False))

    def test_check_shapes_incorrect_cs(self):
        cs, ps = np.zeros((2,2)), np.ones((5,3))
        nums = np.array([3, 5])
        with self.assertRaises(AssertionError):
            f1_solver.check_shapes(cs, ps, nums, False)

    def test_check_shapes_incorrect_ps(self):
        cs, ps = np.zeros((3,2)), np.ones((4,3))
        nums = np.array([3, 5])
        with self.assertRaises(AssertionError):
            f1_solver.check_shapes(cs, ps, nums, False)

    def test_check_shapes_incorrect_nums(self):
        cs, ps = np.zeros((3,2)), np.ones((5,3))
        nums = np.ones(3)
        with self.assertRaises(AssertionError):
            f1_solver.check_shapes(cs, ps, nums, False)

    def test_generate_alpha(self):
        scores = [2,1]
        np.testing.assert_array_equal(f1_solver.generate_alpha(scores, 3), np.array([2,1,0]))

    def test_generate_alpha2(self):
        scores = [10, 6, 3, 0]
        np.testing.assert_array_equal(f1_solver.generate_alpha(scores, 6), np.array([10,6,3,0,0,0]))

    def test_generate_alpha_for_less_candidates(self):
        scores = [10, 6, 3, 0]
        np.testing.assert_array_equal(f1_solver.generate_alpha(scores, 2), np.array([10,6]))

    def test_get_max_num_of_ties(self):
        ties = [[1,2], [2,3,4]]
        self.assertEqual(f1_solver.get_max_num_of_ties(ties), 3)

    def test_solve(self):
        C, V = np.array([1, 2, 3]), [np.array([[2, 3, 1]])]
        nums = np.array([3, 1])
        alpha = np.array([3,1,0])
        prob, _ = f1_solver.generate_lp(C, V, nums, 2, alpha, 1)
        solver = getSolver("COIN_CMD", msg=False)
        self.assertTrue(f1_solver.evaluate_lp(prob, solver))

    def test_pos(self):
        v = np.array([5,3,4,2,1])
        self.assertEqual(f1_solver.pos(v, 4), 3)

    def test_pos_tied(self):
        v = np.array([1,2,3]) # 4 and 5 are tied, i.e., on position 4
        self.assertEqual(f1_solver.pos(v, 4), 4)
        self.assertEqual(f1_solver.pos(v, 5), 4)

    def test_pos_multiple_occurances(self):
        v = np.ones(2, dtype=int)
        with self.assertRaises(AssertionError):
            f1_solver.pos(v, 1)

    def test_T_candidate_3_on_position_1_once(self):
        V = [np.array([3,2,1]), np.array([1,2,3])]
        c, k = 3, 1
        self.assertEqual(f1_solver.T(V, c, k), 1)

    def test_T_candidate_3_on_position_1_twice(self):
        V = [np.array([3,2,1]), np.array([3,1,2])]
        c, k = 3, 1
        self.assertEqual(f1_solver.T(V, c, k), 2)

    def test_T_candidate_2_never_on_position_3(self):
        V = [np.array([3,2,1]), np.array([2,1,3])]
        c, k = 2, 3
        self.assertEqual(f1_solver.T(V, c, k), 0)

if __name__ == '__main__':
    unittest.main()
