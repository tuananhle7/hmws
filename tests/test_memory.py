import torch
import cmws.memory


def test_dims_single_x():
    shape = [2, 3, 4]
    dims = [5, 6]
    k, n = 7, 8

    x = torch.randint(2, size=shape + [n] + dims)
    scores = torch.randn(*[*shape, n])
    x_selected, scores_selected = cmws.memory.get_unique_and_top_k(x, scores, k)

    assert list(x_selected.shape) == shape + [k] + dims
    assert list(scores_selected.shape) == shape + [k]


def test_dims_multiple_x():
    shape = [2, 3, 4]
    dims_0 = [5, 6]
    dims_1 = [7, 8]
    k, n = 9, 10

    x = [torch.randint(2, size=shape + [n] + dims_0), torch.randint(2, size=shape + [n] + dims_1)]
    scores = torch.randn(*[*shape, n])
    x_selected, scores_selected = cmws.memory.get_unique_and_top_k(x, scores, k)

    assert len(x_selected) == 2
    assert list(x_selected[0].shape) == shape + [k] + dims_0
    assert list(x_selected[1].shape) == shape + [k] + dims_1
    assert list(scores_selected.shape) == shape + [k]


def test_value():
    k = 2
    x = torch.tensor([[0, 1, 2, 3], [3, 2, 4, 1], [0, 1, 2, 3], [3, 1, 3, 2], [1, 2, 4, 1]]).long()
    scores = torch.tensor([10, 3, 10, 1, 5]).float()
    x_selected_correct = torch.tensor([[1, 2, 4, 1], [0, 1, 2, 3]])
    scores_selected_correct = torch.tensor([5, 10]).float()

    x_selected, scores_selected = cmws.memory.get_unique_and_top_k(x, scores, k)

    assert torch.equal(x_selected, x_selected_correct)
    assert torch.equal(scores_selected, scores_selected_correct)
