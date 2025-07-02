import os

import fitsio
import numpy as np
import scipy.stats
import pytest

import mattspy
from mattspy.twoptdata import TwoPtData

TEST_FNAME = os.path.join(
    mattspy.__file__.replace("/__init__.py", ""),
    "..",
    "data",
    "test_des_two_pt_data.fits",
)
DICT_FIELDS = ["value", "bin1", "bin2", "angbin", "ang", "angmin", "angmax", "msk_dict"]
PROPERTIES = ["dv", "cov", "corr", "msk", "dataid", "ndim"]


print(TEST_FNAME, flush=True)


@pytest.mark.skipif(
    not os.path.exists(TEST_FNAME), reason="Test DES 2pt data cannot be found!"
)
def test_two_pt_data_immutable():
    d = TwoPtData.read_des_twopoint(TEST_FNAME)

    with pytest.raises(Exception):
        d.order[4] = "blah"

    with pytest.raises(Exception):
        d.full_cov[0, 4] = np.nan

    tkey = d.order[0] + "_1_1"
    for attr in DICT_FIELDS:
        with pytest.raises(Exception):
            getattr(d, attr)[tkey] = "10"
        with pytest.raises(Exception):
            getattr(d, attr)[tkey][0] = 2

    for attr in PROPERTIES:
        with pytest.raises(Exception):
            setattr(d, attr, 5)
        with pytest.raises(Exception):
            exec(f"d.{attr} = 10")


@pytest.mark.skipif(
    not os.path.exists(TEST_FNAME), reason="Test DES 2pt data cannot be found!"
)
def test_two_pt_data_shape():
    d = TwoPtData.read_des_twopoint(TEST_FNAME)

    assert d.full_cov.shape[0] == d.full_cov.shape[1]
    assert d.dv.shape[0] == d.full_cov.shape[0]
    assert d.msk.shape[0] == d.full_cov.shape[0]
    assert d.cov.shape == d.full_cov.shape
    assert d.corr.shape == d.full_cov.shape
    assert d.ndim == d.dv.shape[0]
    assert len(d.dataid) == d.dv.shape[0]


@pytest.mark.skipif(
    not os.path.exists(TEST_FNAME), reason="Test DES 2pt data cannot be found!"
)
def test_two_pt_data_chi2():
    d = TwoPtData.read_des_twopoint(TEST_FNAME)
    chi2 = d.chi2_stats(d, 11)
    assert np.allclose(chi2["chi2"], 0)
    assert np.allclose(chi2["pvalue"], 1)
    assert np.allclose(chi2["ndim"], d.ndim)
    assert np.allclose(chi2["dof"], d.ndim - 11)
    assert np.allclose(chi2["nsigma"], 0.0)

    dc = d.cut_wtheta_crosscorr()
    chi2 = dc.chi2_stats(d, 11)
    assert np.allclose(chi2["chi2"], 0)
    assert np.allclose(chi2["pvalue"], 1)
    assert np.allclose(chi2["ndim"], dc.ndim)
    assert np.allclose(chi2["dof"], dc.ndim - 11)
    assert np.allclose(chi2["nsigma"], 0.0)

    dc = d.cut_wtheta_crosscorr()
    chi2 = dc.chi2_stats(d.dv, 11)
    assert np.allclose(chi2["chi2"], 0)
    assert np.allclose(chi2["pvalue"], 1)
    assert np.allclose(chi2["ndim"], dc.ndim)
    assert np.allclose(chi2["dof"], dc.ndim - 11)
    assert np.allclose(chi2["nsigma"], 0.0)

    rng = np.random.default_rng(seed=42)
    pert = rng.normal(size=dc.ndim, scale=0.01)
    pert_dv = dc.dv * (1.0 + pert)
    chi2 = dc.chi2_stats(pert_dv, 11)
    ddv = dc.dv - pert_dv
    pred_chi2 = np.dot(ddv, np.dot(np.linalg.inv(dc.cov), ddv))
    pred_pvalue = scipy.stats.chi2.sf(pred_chi2, dc.ndim - 11)
    pred_nsigma = scipy.stats.norm.isf(pred_pvalue / 2)
    assert np.allclose(chi2["chi2"], pred_chi2)
    assert np.allclose(chi2["pvalue"], pred_pvalue)
    assert np.allclose(chi2["ndim"], dc.ndim)
    assert np.allclose(chi2["dof"], dc.ndim - 11)
    assert np.allclose(chi2["nsigma"], pred_nsigma)

    def _remove_wtheta_x(dct):
        return {
            k: v
            for k, v in dct.items()
            if not (np.all(d.bin1[k] != d.bin2[k]) and k.startswith("wtheta"))
        }

    dcc = TwoPtData(
        [
            s
            for s in d.order
            if not (s.startswith("wtheta") and s.split("_")[1] != s.split("_")[2])
        ],
        _remove_wtheta_x(d.value),
        _remove_wtheta_x(d.bin1),
        _remove_wtheta_x(d.bin2),
        _remove_wtheta_x(d.angbin),
        _remove_wtheta_x(d.ang),
        _remove_wtheta_x(d.angmin),
        _remove_wtheta_x(d.angmax),
        dc.cov.copy(),
        _remove_wtheta_x(d.msk_dict),
    )
    dc = d.cut_wtheta_crosscorr()
    chi2 = dcc.chi2_stats(dc, 11)
    assert np.allclose(chi2["chi2"], 0)
    assert np.allclose(chi2["pvalue"], 1)
    assert np.allclose(chi2["ndim"], dc.ndim)
    assert np.allclose(chi2["dof"], dc.ndim - 11)
    assert np.allclose(chi2["nsigma"], 0.0)

    with pytest.raises(Exception):
        d.chi2_stats(dcc, 11)

    with pytest.raises(Exception):
        dcc.chi2_stats(d.dv, 11)


@pytest.mark.skipif(
    not os.path.exists(TEST_FNAME), reason="Test DES 2pt data cannot be found!"
)
def test_two_pt_data_cut_wtheta_crosscorr():
    d = TwoPtData.read_des_twopoint(TEST_FNAME)
    dc = d.cut_wtheta_crosscorr()

    def _remove_wtheta_x(dct):
        return {
            k: v
            for k, v in dct.items()
            if not (np.all(d.bin1[k] != d.bin2[k]) and k.startswith("wtheta"))
        }

    cov = np.zeros((dc.ndim, dc.ndim))
    minds = np.where(dc.msk)[0]
    for ni, oi in enumerate(minds):
        for nj, oj in enumerate(minds):
            cov[ni, nj] = d.full_cov[oi, oj]

    dcc = TwoPtData(
        [
            s
            for s in d.order
            if not (s.startswith("wtheta") and s.split("_")[1] != s.split("_")[2])
        ],
        _remove_wtheta_x(d.value),
        _remove_wtheta_x(d.bin1),
        _remove_wtheta_x(d.bin2),
        _remove_wtheta_x(d.angbin),
        _remove_wtheta_x(d.ang),
        _remove_wtheta_x(d.angmin),
        _remove_wtheta_x(d.angmax),
        cov,
        _remove_wtheta_x(d.msk_dict),
    )

    assert np.array_equal(dc.dv, dcc.dv)
    assert np.array_equal(dc.cov, dcc.cov)
    assert np.array_equal(dc.corr, dcc.corr)
    assert dc.dataid == dcc.dataid

    for i in range(1, 7):
        for j in range(i, 7):
            if i != j:
                key = f"wtheta_{i}_{j}"
                assert not any(did.startswith(key) for did in dc.dataid)


@pytest.mark.skipif(
    not os.path.exists(TEST_FNAME), reason="Test DES 2pt data cannot be found!"
)
@pytest.mark.parametrize("cutkind", ["str", "int"])
def test_two_pt_data_cut_twopt_stat_wtheta(cutkind):
    d = TwoPtData.read_des_twopoint(TEST_FNAME)
    dc = d.cut_wtheta_crosscorr()

    for i in range(1, 7):
        for j in range(i, 7):
            if i != j:
                if cutkind == "int":
                    d = d.cut_twopt_stat("wtheta", bin1=i, bin2=j)
                else:
                    d = d.cut_twopt_stat(f"wtheta_{i}_{j}")

    assert np.array_equal(d.dv, dc.dv)
    assert np.array_equal(d.cov, dc.cov)
    assert np.array_equal(d.corr, dc.corr)
    assert d.dataid == dc.dataid

    for i in range(1, 7):
        for j in range(i, 7):
            if i != j:
                key = f"wtheta_{i}_{j}"
                assert not any(did.startswith(key) for did in dc.dataid)


@pytest.mark.skipif(
    not os.path.exists(TEST_FNAME), reason="Test DES 2pt data cannot be found!"
)
@pytest.mark.parametrize("cutkind", ["str", "int"])
def test_two_pt_data_cut_twopt_stat_xip(cutkind):
    d = TwoPtData.read_des_twopoint(TEST_FNAME)

    for i in range(1, 5):
        for j in range(i, 5):
            if i != j:
                if cutkind == "int":
                    d = d.cut_twopt_stat("xip", bin1=i, bin2=str(j))
                else:
                    d = d.cut_twopt_stat(f"xip_{i}_{j}")

    for i in range(1, 5):
        for j in range(i, 5):
            if i != j:
                key = f"xip_{i}_{j}"
                assert not any(did.startswith(key) for did in d.dataid)

    with pytest.raises(Exception):
        d.cut_twopt_stat("xip", bin1=6, bin2=5)

    with pytest.raises(Exception):
        d.cut_twopt_stat("xip_6_5")


@pytest.mark.skipif(
    not os.path.exists(TEST_FNAME), reason="Test DES 2pt data cannot be found!"
)
def test_two_pt_data_cut_component_lens():
    d = TwoPtData.read_des_twopoint(TEST_FNAME)
    d = d.cut_component("l", 2)

    for key in ["wtheta_2_2", "gammat_2_"]:
        assert not any(did.startswith(key) for did in d.dataid)


@pytest.mark.skipif(
    not os.path.exists(TEST_FNAME), reason="Test DES 2pt data cannot be found!"
)
def test_two_pt_data_cut_component_source():
    d = TwoPtData.read_des_twopoint(TEST_FNAME)
    d = d.cut_component("s", 3)

    assert not any(
        did.rsplit("_")[2] == "3" and did.startswith("gammat_") for did in d.dataid
    )

    for kind in ["xip", "xim"]:
        for i in range(1, 5):
            for j in range(i, 5):
                if i == 3 or j == 3:
                    key = f"{kind}_{i}_{j}"
                    assert not any(did.startswith(key) for did in d.dataid)


@pytest.mark.skipif(
    not os.path.exists(TEST_FNAME), reason="Test DES 2pt data cannot be found!"
)
def test_two_pt_data_cut_cosmosis():
    d = TwoPtData.read_des_twopoint(TEST_FNAME)
    stat = "xim_1_2"
    angmin = 7.0
    angmax = 50.0
    d = d.cut_cosmosis([f"{stat} = {angmin} {angmax}"])

    angs = d.ang[stat]
    msk = (angs < angmin) | (angs > angmax)
    angbins = d.angbin[stat][msk]
    for angbin in angbins:
        key = f"{stat}_{angbin}"
        assert not any(did == key for did in d.dataid)

    nloss = np.sum(msk)
    assert d.ndim == d.full_cov.shape[0] - nloss


@pytest.mark.skipif(
    not os.path.exists(TEST_FNAME), reason="Test DES 2pt data cannot be found!"
)
def test_two_pt_data_dv_msk_cov_corr():
    d = TwoPtData.read_des_twopoint(TEST_FNAME)
    dorig = d.copy()
    keep_stat = "xip_3_4"
    for stat in dorig.order:
        if not stat.startswith(keep_stat):
            d = d.cut_cosmosis([f"{stat} = -1 -1"])

    assert d.dv.shape[0] == d.value[keep_stat].shape[0]
    assert np.array_equal(d.dv, d.value[keep_stat])
    assert np.array_equal(d.dv, dorig.dv[d.msk])

    minds = np.where(d.msk)[0]
    cov = np.zeros((d.ndim, d.ndim))
    for ni, oi in enumerate(minds):
        for nj, oj in enumerate(minds):
            cov[ni, nj] = dorig.cov[oi, oj]
    assert np.array_equal(cov, d.cov)
    assert np.array_equal(dorig.full_cov, d.full_cov)


if __name__ == "__main__":
    orig_fname = "3x2pt_2025-06-18-09h_UNBLINDED.fits"
    new_fname = TEST_FNAME
    os.system(f"cp {orig_fname} {new_fname}")

    exts = ["xip", "xim", "gammat", "wtheta"]
    rng = np.random.default_rng()
    eps = 0.10
    with fitsio.FITS(new_fname, "rw") as fp:
        for ext in exts:
            d = fp[ext].read(lower=True)
            pert = rng.normal(size=d.shape[0], scale=eps)
            d["value"] = (1.0 + pert) * d["value"]
            fp[ext].write(d)

    for ext in exts:
        orig_d = fitsio.read(orig_fname, ext=ext, lower=True)
        new_d = fitsio.read(new_fname, ext=ext, lower=True)

        assert orig_d.dtype.names == new_d.dtype.names

        for nm in orig_d.dtype.names:
            if nm == "value":
                assert not np.allclose(orig_d[nm], new_d[nm])
            else:
                assert np.allclose(orig_d[nm], new_d[nm]), (ext, nm)
