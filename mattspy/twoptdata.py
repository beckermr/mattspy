import copy
from dataclasses import dataclass

import fitsio
import frozendict
import numpy as np
import scipy.stats


def _copy_dict_arr(darr):
    return {k: v.copy() for k, v in darr.items()}


@dataclass(frozen=True)
class TwoPtData:
    order: tuple
    value: dict
    bin1: dict
    bin2: dict
    angbin: dict
    ang: dict
    angmin: dict
    angmax: dict
    full_cov: np.ndarray
    msk_dict: dict

    def __post_init__(self):
        object.__setattr__(self, "order", tuple(self.order))

        for attr in [
            "value",
            "bin1",
            "bin2",
            "angbin",
            "ang",
            "angmin",
            "angmax",
            "msk_dict",
        ]:
            dct = getattr(self, attr)
            for k in dct:
                dct[k].setflags(write=False)
            object.__setattr__(self, attr, frozendict.frozendict(dct))

        self.full_cov.setflags(write=False)

    def copy(self):
        return TwoPtData(
            copy.deepcopy(self.order),
            _copy_dict_arr(self.value),
            _copy_dict_arr(self.bin1),
            _copy_dict_arr(self.bin2),
            _copy_dict_arr(self.angbin),
            _copy_dict_arr(self.ang),
            _copy_dict_arr(self.angmin),
            _copy_dict_arr(self.angmax),
            self.full_cov.copy(),
            _copy_dict_arr(self.msk_dict),
        )

    @classmethod
    def read_des_twopoint(
        cls, fname: str, default_type_order=("xip", "xim", "gammat", "wtheta")
    ):
        keys_to_keep = [
            "value",
            "bin1",
            "bin2",
            "ang",
            "angbin",
            "anglemin",
            "anglemax",
        ]
        data = {key.replace("angle", "ang"): {} for key in keys_to_keep}
        data["msk"] = {}
        order = []

        with fitsio.FITS(fname) as fp:
            for stat in default_type_order:
                d = fp[stat].read(lower=True)
                raw_order = [
                    (stat, bin1, bin2) for bin1, bin2 in zip(d["bin1"], d["bin2"])
                ]
                stat_b1_b2_order = []
                for r in raw_order:
                    if stat_b1_b2_order:
                        if stat_b1_b2_order[-1] != r:
                            stat_b1_b2_order.append(r)
                    else:
                        stat_b1_b2_order.append(r)
                order.extend([f"{stat}_{b1}_{b2}" for stat, b1, b2 in stat_b1_b2_order])
                for _, b1, b2 in stat_b1_b2_order:
                    key = f"{stat}_{b1}_{b2}"
                    msk = (d["bin1"] == b1) & (d["bin2"] == b2)
                    for dkey in keys_to_keep:
                        data[dkey.replace("angle", "ang")][key] = d[dkey][msk]
                    data["msk"][key] = np.ones(data["value"][key].shape[0], dtype=bool)

            cov = fp["covmat"].read()

        return cls(
            tuple(order),
            data["value"],
            data["bin1"],
            data["bin2"],
            data["angbin"],
            data["ang"],
            data["angmin"],
            data["angmax"],
            cov,
            data["msk"],
        )

    @property
    def dv(self):
        if not hasattr(self, "_cut_dv"):
            dv = []
            for stat in self.order:
                dv.append(self.value[stat][self.msk_dict[stat]])
            _cut_dv = np.concatenate(dv, axis=0)
            _cut_dv.setflags(write=False)
            object.__setattr__(self, "_cut_dv", _cut_dv)

        return self._cut_dv

    @property
    def msk(self):
        if not hasattr(self, "_cut_msk"):
            msk = []
            for stat in self.order:
                msk.append(self.msk_dict[stat])
            _cut_msk = np.concatenate(msk, axis=0)
            _cut_msk.setflags(write=False)
            object.__setattr__(self, "_cut_msk", _cut_msk)

        return self._cut_msk

    @property
    def cov(self):
        if not hasattr(self, "_cut_cov"):
            n_cov = np.sum(self.msk)
            assert n_cov == self.dv.shape[0]
            cov = np.zeros((n_cov, n_cov))
            cov_inds = np.where(self.msk)[0]
            for new_cov_ind_i, cov_ind_i in enumerate(cov_inds):
                for new_cov_ind_j, cov_ind_j in enumerate(cov_inds):
                    cov[new_cov_ind_i, new_cov_ind_j] = self.full_cov[
                        cov_ind_i, cov_ind_j
                    ]
            cov.setflags(write=False)
            object.__setattr__(self, "_cut_cov", cov)

        return self._cut_cov

    @property
    def corr(self):
        if not hasattr(self, "_cut_corr"):
            cov = self.cov
            dg = np.sqrt(np.diag(cov))
            _cut_corr = cov / np.outer(dg, dg)
            _cut_corr.setflags(write=False)
            object.__setattr__(self, "_cut_corr", _cut_corr)

        return self._cut_corr

    @property
    def dataid(self):
        if not hasattr(self, "_dataid"):
            ids = []
            for stat in self.order:
                msk = self.msk_dict[stat]
                if np.any(msk):
                    ids.extend(
                        [f"{stat}_{ind}" for ind in np.where(self.msk_dict[stat])[0]]
                    )

            _dataid = tuple(ids)
            object.__setattr__(self, "_dataid", _dataid)

        return self._dataid

    @property
    def ndim(self):
        return self.dv.shape[0]

    def chi2_stats(self, theory, nparam):
        if isinstance(theory, TwoPtData):
            if self.dataid != theory.dataid:
                assert all(did in theory.dataid for did in self.dataid), (
                    "The theory TwoPtData does not have all of the data points "
                    "in the data TwoPtData!",
                    set(self.dataid) - set(theory.dataid),
                )

                keep = []
                for did in theory.dataid:
                    if did in self.dataid:
                        keep.append(True)
                    else:
                        keep.append(False)
                keep = np.array(keep)
            else:
                keep = None

            theory = theory.dv
            if keep is not None:
                theory = theory[keep]
        else:
            if (
                theory.shape[0] != self.ndim
                and theory.shape[0] == self.full_cov.shape[0]
            ):
                theory = theory[self.msk]

        assert theory.shape[0] == self.ndim, (
            "The theory datavector is the wrong size "
            f"(theory={theory.shape[0]}, data={self.ndim})!"
        )

        dof = self.ndim - nparam
        ddv = self.dv - theory
        chi2 = np.dot(ddv, np.dot(np.linalg.inv(self.cov), ddv))

        pvalue = scipy.stats.chi2.sf(chi2, dof)
        nsigma = scipy.stats.norm.isf(pvalue / 2)
        return dict(
            chi2=float(chi2),
            pvalue=float(pvalue),
            dof=int(dof),
            nsigma=float(nsigma),
            ndim=int(self.ndim),
        )

    def cut_cosmosis(self, cosmosis_cut_list):
        copy_args = [
            self.order,
            self.value,
            self.bin1,
            self.bin2,
            self.angbin,
            self.ang,
            self.angmin,
            self.angmax,
            self.full_cov,
            _copy_dict_arr(self.msk_dict),
        ]
        new_msk = copy_args[-1]

        for cut_str in cosmosis_cut_list:
            dpart, angs = [d.strip() for d in cut_str.split("=")]
            angmin, angmax = [float(ang.strip()) for ang in angs.split()]
            dpart = dpart.split(".")[-1].replace("angle_range_", "").strip()
            assert dpart in new_msk, (
                f"Could not find '{dpart}' in 2pt data ({self.order})!"
            )
            ang_msk = (self.ang[dpart] >= angmin) & (self.ang[dpart] <= angmax)
            new_msk[dpart] = new_msk[dpart] & ang_msk

        copy_args[-1] = new_msk
        return TwoPtData(*copy_args)

    def cut_wtheta_crosscorr(self):
        cuts = []
        for stat in self.order:
            kind, bin1, bin2 = stat.rsplit("_", maxsplit=3)
            if kind == "wtheta" and bin1 != bin2:
                cuts.append(f"{stat} = -1 -1")

        return self.cut_cosmosis(cuts)

    def cut_component(self, comp, bin_num):
        bin_num = f"{bin_num}"

        cuts = []
        for stat in self.order:
            cutit = False
            kind, bin1, bin2 = stat.rsplit("_", maxsplit=3)
            if comp in ["source", "s"]:
                if kind in ["xip", "xim"] and ((bin1 == bin_num) or (bin2 == bin_num)):
                    cutit = True
                elif kind == "gammat" and bin2 == bin_num:
                    cutit = True
            elif comp in ["lens", "l"]:
                if kind == "gammat" and bin1 == bin_num:
                    cutit = True
                elif kind == "wtheta" and ((bin1 == bin_num) or (bin2 == bin_num)):
                    cutit = True

            if cutit:
                cuts.append(f"{stat} = -1 -1")

        assert len(cuts) > 0, f"No cuts found for comp '{comp}' w/ bin '{bin_num}'!"

        return self.cut_cosmosis(cuts)

    def cut_twopt_stat(self, kind, bin1=None, bin2=None):
        if bin1 is not None:
            bin1 = f"{bin1}"
        if bin2 is not None:
            bin2 = f"{bin2}"

        if bin1 is None and bin2 is None:
            kind, bin1, bin2 = kind.rsplit("_", maxsplit=3)

        if (bin1 is None and bin2 is not None) or (bin1 is not None and bin2 is None):
            raise RuntimeError(
                "You must either specify the 2pt statistic "
                "directly (e.g., 'xip_1_1') or "
                "give the kind (e.g., 'xip') and bins (e.g., `bin1=1` and `bin2=3`)."
            )

        cuts = []
        for stat in self.order:
            _kind, _bin1, _bin2 = stat.rsplit("_", maxsplit=3)
            if kind == _kind and bin1 == _bin1 and bin2 == _bin2:
                cuts.append(f"{stat} = -1 -1")

        assert len(cuts) > 0, (
            f"No cuts found for kind '{kind}' w/ bin1 '{bin1}' and bin2 '{bin2}'!"
        )

        return self.cut_cosmosis(cuts)
