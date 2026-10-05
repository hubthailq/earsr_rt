import itertools

import pytest

from earsr.runid import PROTOCOLS, RunId


def test_roundtrip_and_seed():
    r = RunId("T6", "span", "reflect-wide", "ps20", "rand", 4, "all", "bic", 3)
    assert str(r) == "T6_span-reflect-wide_ps20_rand_x4_hrall_bic_f3"
    assert RunId.parse(str(r)) == r and r.seed == 3


def test_distinct_configs_never_collide():
    grid = itertools.product(["T6", "S2"], ["span", "rlfn"], ["zero", "reflect", "zero-wide", "zero-deep"],
                             ["pub", "ps20", "pf", "none"], PROTOCOLS, [2, 4], ["96", "144", "all"],
                             ["bic", "est"], [1, 2, 3])
    ids = [str(RunId(*g)) for g in grid]
    assert len(ids) == len(set(ids))


@pytest.mark.parametrize("kw", [dict(backbone="sp_an"), dict(backbone="sp-an"), dict(pretrain="short"),
                                dict(protocol="patch"), dict(tier="big"), dict(fold=9)])
def test_invalid_fields_rejected(kw):
    base = dict(exp="T6", backbone="span", variant="zero", pretrain="pub", protocol="rand", scale=4,
                tier="144", degrade="bic", fold=2)
    base.update(kw)
    with pytest.raises(ValueError):
        RunId(**base)
