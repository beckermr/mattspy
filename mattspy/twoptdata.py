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
    """Container for two-point data measurements with associated
    methods for cuts, chi2 computations, etc.

    The two-point data is stored as dictionaries keyed on the statistic
    plus tomographic bin combination. The possible statistics are `xip`,
    `xim`, `gammat`, and `wtheta`. The dictionary keys then are
    `<stat>_<bin1>_<bin2>` and each maps to numpy array with the value. For
    gammat, the first bin is the lens bin and the second bin is the source bin.

    The overall order of the two-point statistics in the data vector is stored
    in the `order` attribute. To consruct the data vector, you combined the
    individual two-point measurements in the order specified by `order`, applying the
    boolean mask stored per statistics in `msk_dict`:

        dv = np.concatenate(
            [
                data.value[stat][data.msk_dict[stat]]
                for stat in data.order
                if np.any(data.msk_dict[stat])
            ],
            axis=0,
        )

    The `TwoPtData` is immutable and will raise an exception if one attempts
    to modify any of its attributes. To make cuts to the data, use the
    one of the methods below. When a cut is made, the underlying data is unchanged,
    but the boolean mask `msk_data` is updated. Each cut returns a new `TwoPTData`
    object.

    Parameters
    ----------
    order: tuple of str
        A tuple of strings of the form `<stat>_<bin1>_<bin2>` specifying the order
        of the statistics in the data vector.
    value: dict
        Maps strings from `order` to the data values as numpy arrays.
    bin1: dict
        Maps strings from `order` to the first tomographic bin.
    bin2: dict
        Maps strings from `order` to the second tomographic bin.
    angbin: dict
        Maps strings from `order` to the angle bin index.
    ang: dict
        Maps strings from `order` to the angle for the two-point measurement.
    angmin: dict
        Maps strings from `order` to the minimum angle for the data points.
    angmax: dict
        Maps strings from `order` to the maximum angle for the data points.
    full_cov: np.ndarray
        A two-dimensional array giving the covariance matrix for the data vector.
        The size should match the entire size of the data without cuts.
    msk_dict: dict
        Maps strings from `order` to a boolean mask specifying which two-point
        data points are being kept (True) and which ones are cut (False).

    Attributes
    ----------
    dv : np.ndarray
        The data vector with cuts from `msk_dict` applied.
    cov : np.ndarray
        The data vector covariance matrix for `dv`.
    corr : np.ndarray
        The data vector correlation matrix for `dv`.
    msk : np.ndarray
        A boolean mask specifying which elements of the full data vector
        are kept.
    dataid : tuple of strings
        A tuple of strings specifying which data points have been kept in
        their final order. The strings are `<stat>_<bin1>_<bin2>_<angbin>`
        for each statistics in point.
    ndim : int
        The dimension of the data vector after cuts.

    Methods
    -------
    copy()
        Returns a copy of the `TwoPtData`.
    read_des_twopoint(fname)
        Construct a `TwoPtData` instance from a DES Y6 data FITS file.
    chi2_stats(theory, nparam)
        Compute a dictionary of chi2 statistics given a theory prediction
        and number of parameters.
    cut_cosmosis(cosmosis_cut_list)
        Apply cosmosis-style cuts to a data vector.
    cut_wtheta_crosscorr()
        Cut `wtheta` cross-correlations from the data vector.
    cut_component(comp, bin_num)
        Cut all two-point statistics of data vector that use a given lens or source bin.
    cut_twopt_stat(kind, bin1=None, bin2=None)
        Cut a specific two-point stat.
    increase_cov_fractionally(fraction)
        Increase the covariance by an amount `1+fraction`, preserving the
        correlation structure.
    rebuild_only_unmasked()
        Rebuild the `TwoPtData` by applying the masks and remaking the internal data
        with just the unmasked data points.
    replace_full_cov(new_full_cov)
        Replace the full covariance matrix, returning a new `TwoPtData`
        object.
    cut_angle(angmin, angmax)
        Cut all data to within a min and max angle.
    """

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

        ndim = 0
        for stat in self.order:
            ndim += self.value[stat].shape[0]
        assert ndim == self.full_cov.shape[0], (
            "The full covariance matrix does not match the size of the data vector!"
        )

    def copy(self):
        """Return a copy of the TwoPtData."""
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
    def read_des_twopoint(cls, fname: str):
        """Construct a TwoPtData object from a DES FITS file.

        Parameters
        ----------
        fname : str
            The path to the FITS data file.

        Returns
        -------
        TwoPtData
            The two-point data.
        """
        default_type_order = ("xip", "xim", "gammat", "wtheta")

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
        """The data vector w/ cuts as a numpy array."""
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
        """A boolean mask the cuts a full-sized data vector to the cut data vector."""
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
        """The covariance matrix of the data vector `dv`."""
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
        """The correlation matrix of the data vector `dv`."""
        if not hasattr(self, "_cut_corr"):
            cov = self.cov
            dg = np.sqrt(np.diag(cov))
            _cut_corr = cov / np.outer(dg, dg)
            _cut_corr.setflags(write=False)
            object.__setattr__(self, "_cut_corr", _cut_corr)

        return self._cut_corr

    @property
    def dataid(self):
        """The dataid for the data vector.

        The dataid is a tuple of strings af form

            `<stat>_<bin1>_<bin2>_<angbin>`

        in the order of the elements of the data vector.
        """
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
        """The dimenion of the cut data vector."""
        return self.dv.shape[0]

    def chi2_stats(self, theory, nparam):
        """Compute chi2 statistics given a theory prediction and a
        number of parameters.

        Paremeters
        ----------
        theory : TwoPtData | np.ndarray
            A `TwoPtData` object or numpy array containing the theory prediction. We
            attempt to validate the length and/or `dataid` of the input theory
            prediction and the method will raise an error if it detects an issue.
            The lenth of a numpy array input should either be `TwoPtData.ndim` or
            `TwoPTData.full_cov.shape[0]`.
        nparam : int
            The number of parameters to use to compute the degrees of freedom.

        Returns
        -------
        dict of stats

            chi2: the chi2 value
            pvalue: the probability to exceed this value
            dof: the degrees of freedom
            nsigma: the p-value converted into an equivalent number of sigma
                for a Gaussian
            ndim: the number of dimensions of the data vector
        """
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
        """Apply a cosmosis-style cut to the data vector.

        Parameters
        ----------
        cosmosis_cut_list : list of str
            A list of strings specifying the cosmosis cuts. The strings are of the form

                <stat>_<bin1>_<bin2> = <angle min> <angle max>

            which specifies to cut the two-point statistics `<stat>_<bin1>_<bin2>` to
            only the anglular range `angle min <= angle <= angle max`.

        Returns
        -------
        TweoPtData
            The cut two-point data.
        """
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
        """Cut the wtheta tomographic cross-correlations from the data.

        Returns
        -------
        TweoPtData
            The cut two-point data.
        """
        cuts = []
        for stat in self.order:
            kind, bin1, bin2 = stat.rsplit("_", maxsplit=3)
            if kind == "wtheta" and bin1 != bin2:
                cuts.append(f"{stat} = -1 -1")

        return self.cut_cosmosis(cuts)

    def cut_component(self, comp, bin_num):
        """Cut any parts of the data vector that use a given component
        (e.g., lens, source, etc.).

        Parameters
        ----------
        comp : str
            The component, one of "s", "source", "l", or "lens" for sources and lenses.
        bin_num : int | str
            The tomographic bin number.

        Returns
        -------
        TweoPtData
            The cut two-point data.
        """
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
        """Cut a specific two-point statistic from the data vector.

        Parameters
        ----------
        kind : str
            A string specifying which statistic to cut of the form
            `<stat>_<bin1>_<bin2>` or just `<stat>`. If you specify just `<stat>`,
            then you need to also specify `bin1` and `bin2`.
        bin1 : str | int | None
            The first tomographic bin number.
        bin2 : str | int | None
            The second tomographic bin number.

        Returns
        -------
        TwoPtData
            The cut two-point data.
        """
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

    def increase_cov_fractionally(self, fraction):
        """Increase the covariance by an amount `1+fraction`,
        preserving the correlation structure.

        Parameters
        ----------
        fraction : float
            The fraction to increase the covariance.

        Returns
        -------
        TwoPtData
            The data with a larger covariance.
        """
        diag = np.sqrt(np.diag(self.full_cov))
        corr = self.full_cov / np.outer(diag, diag)
        new_diag = diag * (1.0 + fraction)
        new_cov = corr * np.outer(new_diag, new_diag)

        return self.replace_full_cov(new_cov)

    def rebuild_only_unmasked(self):
        """Rebuild the `TwoPtData` by applying the masks and remaking the internal data
        with just the unmasked data points.

        After this operation, all entries in the `msk_dict` data structure will be
        `True` and any data associated with entries that were `False` will be discarded.

        Returns
        -------
        TwoPtData
            The rebuilt two-point data.
        """
        new_order = []
        for stat in self.order:
            if np.any(self.msk_dict[stat]):
                new_order.append(stat)

        def _cut_and_msk_dict(dct):
            new_dct = {}
            for stat in new_order:
                new_dct[stat] = dct[stat][self.msk_dict[stat]]
            return new_dct

        new_d = TwoPtData(
            new_order,
            _cut_and_msk_dict(self.value),
            _cut_and_msk_dict(self.bin1),
            _cut_and_msk_dict(self.bin2),
            _cut_and_msk_dict(self.angbin),
            _cut_and_msk_dict(self.ang),
            _cut_and_msk_dict(self.angmin),
            _cut_and_msk_dict(self.angmax),
            self.cov.copy(),
            _cut_and_msk_dict(self.msk_dict),
        )

        assert set(new_d.order) == set(new_d.msk_dict)
        for stat in new_d.order:
            assert np.all(new_d.msk_dict[stat])

        return new_d

    def replace_full_cov(self, new_full_cov):
        """Replace the full covariance matrix, returning a new `TwoPtData`
        object.

        Parameters
        ----------
        new_full_cov : np.ndarray
            The new full covariance matrix.

        Returns
        -------
        TwoPtData
            The data with a larger covariance.
        """

        assert new_full_cov.shape == self.full_cov.shape, (
            f"New covariance matrix shape {new_full_cov.shape} is not "
            f"equal to the original covariance matrix shape {self.full_cov.shape}!"
        )

        return TwoPtData(
            self.order,
            self.value,
            self.bin1,
            self.bin2,
            self.angbin,
            self.ang,
            self.angmin,
            self.angmax,
            new_full_cov,
            self.msk_dict,
        )

    def cut_angle(self, angmin, angmax):
        """Cut all data to within a min and max angle.

        Parameters
        ----------
        angmin : float
            The minimum angle
        angmax : float
            The maximum angle.

        Returns
        -------
        TwoPtData
            The cut two-point data.
        """

        cuts = []
        for stat in self.order:
            cuts.append(f"{stat} = {angmin} {angmax}")

        assert len(cuts) > 0, f"No cuts found for angle cut {angmin} to {angmax}!"

        return self.cut_cosmosis(cuts)
