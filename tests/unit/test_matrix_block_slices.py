import unittest
import torch
from feathermap.utils import get_block_rows
from feathermap.feathernet import LoadLayer


class TestBlockRows(unittest.TestCase):
    """
    Cover all possible matrix slice scenarios and test
    `feathernet.utils.get_block_rows` and `feathernet.LoadLayer._get_operands`
    """

    def setUp(self):
        """ Set up fixture covering index ranges and expected operand returns """
        m = 1
        self.n = 3342
        self.V1 = torch.Tensor(
            [list(range(self.n * q, self.n * (q + 1))) for q in range(m)]
        ).reshape(self.n, m)
        self.V2 = torch.Tensor(
            [list(range(m * r, m * (r + 1))) for r in range(self.n)]
        ).reshape(m, self.n)
        self.V = torch.Tensor(
            [list(range(self.n * q, self.n * (q + 1))) for q in range(self.n)]
        )

        # row1, col1, row2, col2
        self.idxs = [
            # All one row complete
            (42, 0, 42, self.n - 1),
            # All two rows complete
            (652, 0, 653, self.n - 1),
            # All 10 rows complete
            (652, 0, 662, self.n - 1),
            # First row complete, second incomplete
            (0, 0, 1, self.n - 100),
            # First row compete, last row incomplete
            (652, 0, 829, 105),
            # First row incomplete (from right), no additional rows
            (1, 0, 1, 105),
            # First row incomplete (from left), no additional rows
            (123, 123, 123, self.n - 1),
            # First row incomplete (from left), second row incomplete
            (222, 1816, 223, 2018),
            # First row] incomplete (from left), last row incomplete
            (652, 1816, 829, 105),
            # First row incomplete (from left), second row complete
            (3000, 1816, 3001, self.n - 1),
            # First row incomplete (from left), last row complete
            (3000, 1816, 3091, self.n - 1),
        ]

        # expected operands from covered index ranges above
        self.operands = [
            {"top": (self.V1[42, :], self.V2[:, 0 : 3341 + 1])},
            {"block": (self.V1[range(*[652, 654]), :], self.V2)},
            {"block": (self.V1[range(*[652, 663]), :], self.V2)},
            {
                "block": (self.V1[range(*[0, 1]), :], self.V2),
                "bottom": (self.V1[1, :], self.V2[:, : 3242 + 1]),
            },
            {
                "block": (self.V1[range(*[652, 829]), :], self.V2),
                "bottom": (self.V1[829, :], self.V2[:, : 105 + 1]),
            },
            {"top": (self.V1[1, :], self.V2[:, 0 : 105 + 1])},
            {"top": (self.V1[123, :], self.V2[:, 123 : 3341 + 1])},
            {
                "top": (self.V1[222, :], self.V2[:, 1816:]),
                "bottom": (self.V1[223, :], self.V2[:, : 2018 + 1]),
            },
            {
                "block": (self.V1[range(*[653, 829]), :], self.V2),
                "top": (self.V1[652, :], self.V2[:, 1816:]),
                "bottom": (self.V1[829, :], self.V2[:, : 105 + 1]),
            },
            {
                "block": (self.V1[range(*[3001, 3002]), :], self.V2),
                "top": (self.V1[3000, :], self.V2[:, 1816:]),
            },
            {
                "block": (self.V1[range(*[3001, 3092]), :], self.V2),
                "top": (self.V1[3000, :], self.V2[:, 1816:]),
            },
        ]

    def test_get_block_rows(self):
        """ Test block row range """
        # Correct block row range
        res = [
            [42, 43],
            [652, 654],
            [652, 663],
            [0, 1],
            [652, 829],
            [],
            [],
            [],
            [653, 829],
            [3001, 3002],
            [3001, 3092],
        ]

        for i, idx in enumerate(self.idxs):
            with self.subTest():
                self.assertEqual(get_block_rows(*idx, self.n), res[i])

    @unittest.skip("operand keys")
    def test_get_operands_keys(self):
        """ Test operand keys (as sequence) """
        for i, idx in enumerate(self.idxs):
            with self.subTest(name=i):
                load_layer_operands = LoadLayer._get_operands(
                    self.V1, self.V2, *idx, self.n
                )
                self.assertSequenceEqual(
                    load_layer_operands.keys(), self.operands[i].keys()
                )

    @unittest.skip("operand value lengths")
    def test_get_operands_value_len(self):
        """ Test length of operand tensors """
        for i, idx in enumerate(self.idxs):
            load_layer_operands = LoadLayer._get_operands(
                self.V1, self.V2, *idx, self.n
            )
            for key in load_layer_operands:
                for tensor_idx in (0, 1):
                    with self.subTest(name=(i, key, tensor_idx)):
                        self.assertEqual(
                            len(load_layer_operands[key][tensor_idx]),
                            len(self.operands[i][key][tensor_idx]),
                        )

    def test_get_operands_values(self):
        """ Test values of operand tensors """
        for i, idx in enumerate(self.idxs):
            load_layer_operands = LoadLayer._get_operands(
                self.V1, self.V2, *idx, self.n
            )
            for key in load_layer_operands:
                for tensor_idx in (0, 1):
                    with self.subTest(name=(i, key, tensor_idx)):
                        self.assertTrue(
                            torch.equal(
                                load_layer_operands[key][tensor_idx],
                                self.operands[i][key][tensor_idx],
                            )
                        )


if __name__ == "__main__":
    unittest.main()
