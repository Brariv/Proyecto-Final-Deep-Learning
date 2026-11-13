"""El servicio debe calcular las mismas variables que el entrenamiento (sin sesgo train/serve).

Requiere los datos procesados (scripts/01_prepare_data.py); si no existen, la prueba se omite.
"""
import numpy as np
import pytest

from fraudscan.config import load_config


@pytest.fixture(scope="module")
def store():
    cfg = load_config()
    try:
        from fraudscan.data import Processed
        from fraudscan.replay import RawStore
        data = Processed(cfg["data"]["processed_dir"])
        return RawStore(data, cfg["data"]["ibm_dir"])
    except FileNotFoundError:
        pytest.skip("No hay datos procesados")


def test_online_features_match_batch(store):
    data = store.data
    rng = np.random.default_rng(0)
    n_cards = int(data.card_idx.max()) + 1
    fraud_cards = np.unique(data.card_idx[data.y == 1])
    cards = np.r_[rng.choice(n_cards, 6, replace=False), rng.choice(fraud_cards, 4, replace=False)]
    checked = 0
    for card in cards:
        _, feats = store.replay(int(card), collect=True)
        rows = np.array([r for r, _, _ in feats])
        online_num = np.stack([x for _, x, _ in feats])
        online_cat = np.stack([c for _, _, c in feats])
        batch_num, batch_cat = data.tabular(rows)
        np.testing.assert_array_equal(online_cat, batch_cat)
        diff = np.abs(online_num - batch_num)
        j = np.unravel_index(diff.argmax(), diff.shape)
        assert diff.max() < 2e-3, (int(card), data.meta["num_features"][j[1]], float(diff.max()))
        checked += len(rows)
    assert checked > 1000


def test_online_sequence_matches_batch(store):
    data = store.data
    card = int(np.unique(data.card_idx[data.y == 1])[0])
    rows = store.card_rows(card)
    target = int(rows[min(len(rows) - 1, 40)])
    state = store.replay(card, until_row=target)
    tx = store.transaction(target)
    x_num, x_cat = state.features(tx, store.vocab)
    num, cat, mask = state.sequence(x_num, x_cat)
    b_num, b_cat, b_mask = data.sequences(np.array([target]))
    np.testing.assert_array_equal(mask, b_mask[0])
    np.testing.assert_array_equal(cat, b_cat[0])
    assert np.abs(num - b_num[0]).max() < 2e-3
