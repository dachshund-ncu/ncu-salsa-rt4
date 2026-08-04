from sys import flags
from typing import Literal, BinaryIO, Any

from numpy import dtype, ndarray
from werkzeug.datastructures import FileStorage

from ncu_salsa_rt4.fits_spectrum import FitsSpectrum
import numpy as np
import os
from operator import attrgetter
import glob
import pandas as pd

class FitsCube:
    def __init__(
            self,
            cat_with_source: str = ".",
            fits_files: list[str | BinaryIO | FileStorage] | None = None,
            load_on_creation=True):
        """
        initializes the class
        """

        self.data_catalog = cat_with_source
        self.flagged_obs = self.__read_flagged_obs(self.data_catalog)
        if fits_files is not None:
            self.spectra = self.__load_spectra_from_list(fits_files)
        else:
            self.spectra = self.__load_spectra_from_cat(self.data_catalog)
        self._sort_data()

    def __load_spectra_from_list(self, fits_files_repr: list[str | BinaryIO | FileStorage]) -> list[FitsSpectrum]:
        """
        Loads .fits files from a files given in a list
        :param fits_files: array of .fits files
        :return: list of FitsSpectrum
        """
        fits_files = []
        for fle_representation in fits_files_repr:
            if isinstance(fle_representation, str):
                if os.path.basename(fle_representation) not in self.flagged_obs:
                    fits_files.append(fle_representation)
            else:
                fits_files.append(fle_representation)
        return [FitsSpectrum(fitsFileName) for no, fitsFileName in enumerate(fits_files)]

    def __load_spectra_from_cat(self, cat_with_source: str) -> list[FitsSpectrum]:
        """
        Loads all the .FITS files from given directory
        """
        fits_filenames = glob.glob(os.path.join(cat_with_source, "*.[fF][iI][tT][sS]"))
        return self.__load_spectra_from_list(fits_filenames)

    def _sort_data(self):
        """
        bubble-sorts spectra
        """
        self.spectra.sort(key=attrgetter('mjd'), reverse=False)

    def __read_flagged_obs(self, directory: str = ".") -> np.ndarray:
        """
        Reads the flagged filenames
        """
        filepath = os.path.join(directory, "flagged_obs.dat")
        try:
            return np.loadtxt(filepath, dtype=str, ndmin=1)
        except (FileNotFoundError, ValueError):
            return np.array([], dtype=str)


    def __str__(self):
        if len(self.spectra) > 0:
            return f"{len(self.spectra)}"
        else:
            return "None"

    # =======================
    # === UTILITY METHODS ===
    # =======================

    def get2_ddata_array(self, pol: Literal["I", "V", "LHC", "RHC"] = "I") -> np.ndarray:
        """
        returns the 2D container, containing data from the loaded spectra
        """
        if len(self.spectra) == 0:
            return np.ndarray([])
        z_array = np.empty((len(self.spectra[0].i_tab), len(self.spectra)))
        # -- iterating to get the data --
        for i in range(len(self.spectra)):
            if pol == 'V':
                z_array[:, i] = self.spectra[i].v_tab
            elif pol == 'LHC':
                z_array[:, i] = self.spectra[i].lhc_tab
            elif pol == 'RHC':
                z_array[:, i] = self.spectra[i].rhc_tab
            else:
                z_array[:, i] = self.spectra[i].i_tab
        return z_array

    def get_vel_array(self):
        """
        returns the VELOCITY table from the first spectrum
        """
        if len(self.spectra) == 0:
            raise BufferError("No data loaded!")
        else:
            return self.spectra[0].velocity_table

    def get_mjd_array(self):
        """
        returns the dates array from the loaded dataset
        """
        if len(self.spectra) == 0:
            raise BufferError("No data loaded!")
        else:
            return np.asarray([s.mjd for s in self.spectra])

    def get_mean_spectrum(self):
        """
        Returns the mean spectrum as a data frame
        """
        velocity = self.get_vel_array()
        i_mean = np.mean(np.asarray([sp.i_tab for sp in self.spectra]), axis=0)
        v_mean = np.mean(np.asarray([sp.v_tab for sp in self.spectra]), axis=0)
        lhc_mean = np.mean(np.asarray([sp.lhc_tab for sp in self.spectra]), axis=0)
        rhc_mean = np.mean(np.asarray([sp.rhc_tab for sp in self.spectra]), axis=0)
        return pd.DataFrame(
            np.column_stack(
                (
                    velocity,
                    i_mean,
                    v_mean,
                    lhc_mean,
                    rhc_mean)),
            columns=["Velocity", "I", "V", "LHC", "RHC"])

    def get_integrated_flux_density(
            self,
            min_chan: int,
            max_chan: int,
            df= False) -> np.ndarray | pd.DataFrame:
        """
        Returns the integrated flux density for whole time series
        """
        array = np.asarray([sp.get_integrated_flux_density(min_chan, max_chan) for sp in self.spectra])
        if df:
            return pd.DataFrame(array, columns=["MJD", "I", "V", "LHC", "RHC"])
        else:
            return array

    def make_slice(
            self,
            velocities: tuple,
            epochs: tuple):
        """
        Returns a slice of the current object, constrained by the values in the arguments
        TODO: change slicing from indices to velocity calculated indepedently for every spectrum
        """
        # failsafes
        if epochs[1] < epochs[0]:
            epochs[0], epochs[1] = epochs[1], epochs[0]
        if velocities[0] > velocities[1]:
            velocities[0], velocities[1] = velocities[1], velocities[0]

        # -- create new SetOfFitsSpectra --
        new_slice = FitsCube(self.data_catalog, load_on_creation=False)
        new_slice.flagged_obs = self.flagged_obs

        # -- populate new objects with sliced spectra --
        vels = self.get_vel_array()
        velocity_indices = np.logical_and(vels >= velocities[0], vels <= velocities[1])
        new_slice.spectra = [
            sp.make_slice(velocity_indices) for i, sp in enumerate(self.spectra) if
            epochs[0] <= sp.mjd <= epochs[1]]
        return new_slice

    def get_light_curve(self, velocity: float, df= True) -> np.ndarray | pd.DataFrame:
        """
        Returns the light curve at a given velcoity
        """
        veltab = self.get_vel_array()
        if velocity < veltab.min():
            channel = 0
        elif velocity > veltab.max():
            channel = len(veltab) - 1
        else:
            channel = self._get_channel_for_velocity(veltab, velocity)
        # extract the data
        mjd_table = self.get_mjd_array()
        i_table = np.asarray([sp.i_tab[channel] for sp in self.spectra])
        v_table = np.asarray([sp.v_tab[channel] for sp in self.spectra])
        lhc_table = np.asarray([sp.lhc_tab[channel] for sp in self.spectra])
        rhc_table = np.asarray([sp.rhc_tab[channel] for sp in self.spectra])
        htop = np.column_stack((mjd_table, i_table, v_table, lhc_table, rhc_table))
        if not df:
            return htop
        else:
            return pd.DataFrame(htop, columns=["MJD", "I", "V", "LHC", "RHC"])

    def _get_channel_for_velocity(self, veltab: np.ndarray, velocity: float) -> int:
        """Finds the nearest channel for the given velocity."""
        return int(np.argmin(np.abs(veltab - velocity)))