import os

import pytest

import phoebe
from phoebe import u
from phoebe.atmospheres import models
from phoebe.atmospheres.passbands import Passband, group_history


path = os.path.dirname(__file__)
TS = 'Thu Nov 19 18:00:00 2026'


def _passband():
    return Passband(
        ptf=os.path.join(path, 'test.ptf'),
        pbset='test',
        pbname='history',
        wlunits=u.nm,
        calibrated=True,
        reference='passband history tests',
        version=1.0,
    )


def test_long_entry_wraps_and_groups():
    pb = _passband()
    msg = 'phoenix tables superseded by the PHOENIX v3 (Husser et al. 2013) specific intensity grid'
    pb.add_to_history(msg, timestamp=TS)
    cards = pb.history[1:]

    assert len(cards) > 1
    assert all(len(card.split(': ', 1)[1]) <= 46 for card in cards)
    assert all(card.endswith('&') for card in cards[:-1])
    assert group_history(pb.history)[-1] == (TS, msg)


def test_same_timestamp_entries_are_joined():
    pb = _passband()
    pb.add_to_history('phoenix intensities recomputed.', timestamp=TS)
    pb.add_to_history('version bumped to 1.1', timestamp=TS)

    entries = group_history(pb.history)
    assert len(entries) == 2
    assert entries[-1] == (TS, 'phoenix intensities recomputed; version bumped to 1.1')


@pytest.mark.parametrize('msg', ['v1.1: phoenix updated', 'phoenix updated &', 42])
def test_invalid_entries_are_rejected(msg):
    with pytest.raises(ValueError):
        _passband().add_to_history(msg)


def test_history_survives_save_and_load(tmp_path):
    pb = _passband()
    pb.compute_intensities(atm=models.BlackbodyModelAtmosphere(), include_mus=False, include_ld=False, verbose=False)
    pb.add_to_history('a long history entry that has to be wrapped over more than one history card', timestamp=TS)

    fname = str(tmp_path / 'history.fits')
    pb.save(fname)
    loaded = phoebe.atmospheres.passbands.Passband.load(fname, load_content=False)

    assert loaded.history == pb.history
    assert group_history(loaded.history) == group_history(pb.history)
