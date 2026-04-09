import torch

from miniagenticrouter.research.training.q_network import QNetwork, QNetworkConfig


def test_q_network_hidden_dims_are_respected() -> None:
    config = QNetworkConfig(hidden_dims=[32, 16], dropout=0.1)
    net = QNetwork(state_dim=8, model_dim=4, config=config)

    # Check backbone structure
    linears = [m for m in net.backbone if isinstance(m, torch.nn.Linear)]
    assert len(linears) == 2
    assert (linears[0].in_features, linears[0].out_features) == (12, 32)
    assert (linears[1].in_features, linears[1].out_features) == (32, 16)

    dropouts = [m for m in net.backbone if isinstance(m, torch.nn.Dropout)]
    assert len(dropouts) == 2

    # Check single head
    assert isinstance(net.head, torch.nn.Linear)
    assert (net.head.in_features, net.head.out_features) == (16, 1)


def test_q_network_falls_back_to_hidden_dim_and_n_layers() -> None:
    config = QNetworkConfig(hidden_dims=None, hidden_dim=10, n_layers=3, dropout=0.0)
    net = QNetwork(state_dim=8, model_dim=4, config=config)

    # Check backbone structure (3 hidden layers)
    linears = [m for m in net.backbone if isinstance(m, torch.nn.Linear)]
    assert len(linears) == 3
    assert (linears[0].in_features, linears[0].out_features) == (12, 10)
    assert (linears[1].in_features, linears[1].out_features) == (10, 10)
    assert (linears[2].in_features, linears[2].out_features) == (10, 10)

    # Check single head
    assert isinstance(net.head, torch.nn.Linear)
    assert (net.head.in_features, net.head.out_features) == (10, 1)
