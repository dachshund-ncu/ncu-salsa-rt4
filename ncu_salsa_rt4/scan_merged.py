"""
Class, that holds merged scan data, which is the result of merging two consecutive scans.
Creation date: 05.03.2022
Creator: Michał Durjasz (md@astro.umk.pl)
"""

from .scan import Scan
import numpy as np


class MergedScan:

    def __init__(self, scan1, scan2):
        self.pols = self.__merge_scans(scan1, scan2)
        self.number_of_channels = self.pols.shape[1]
        self.backupPols = self.pols.copy()
        self.mjd = scan2.mjd
        self.tsys = [scan1.tsys, scan2.tsys]

    def fit_cheby(
            self,
            bbc: int,
            order: int,
            ranges: list[list[int]]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Fits polynomial of the specified order to the data of the specified BBC and
        returns the fitted polynomial values and residuals.
        :param bbc: number of the BBC (typically 1-4)
        :param order: order of the polynomial to fit (typically 7-10)
        :param ranges: ranges of the fit
        :return: tuple of (channels, fitted polynomial values, fit residuals)
        """
        fit_channels, fit_data = self.__get_data_from_ranges(bbc, ranges)
        poly = np.polyfit(fit_channels, fit_data, order)
        pol_tab_x = np.linspace(1, self.number_of_channels, self.number_of_channels)
        pol_tab_y = np.polyval(poly, pol_tab_x)
        pol_tab_residuals = self.pols[bbc - 1] - pol_tab_y
        return pol_tab_x, pol_tab_y, pol_tab_residuals

    def __get_data_from_ranges(
            self,
            bbc: int,
            ranges: list[list[int]]) -> tuple[np.ndarray, np.ndarray]:
        """
        This method is used to get the data from the specified ranges for the specified BBC.
        :param bbc: number of the BBC (typically 1-4)
        :param ranges: ranges of the fit
        :return: channels and data from the specified ranges
        """
        bbc_index = bbc - 1
        channels = np.linspace(1, self.number_of_channels, self.number_of_channels).astype(int)
        indices = np.full(self.pols.shape[0], False)
        for singular_range in ranges:
            indices_tmp = np.logical_and(channels >= singular_range[0], channels <= singular_range[1])
            indices = np.logical_or(indices, indices_tmp)
        return channels[indices], self.pols[bbc_index][indices]


    def __merge_scans(
            self,
            scan1: Scan,
            scan2: Scan) -> np.ndarray:
        """
        This method is used to merge two consecutive scans
        is meant to be used in __INIT__ only and should not be called outside of it.
        ASSUMPTION: every BBC converter results in even number of channels
        :param scan1: even scan from the set
        :param scan2: odd scan from the set (consecutive to scan1)
        :return: array with data merged from scan1 and scan2, which is the result of merging two consecutive scans
        """
        number_of_bbcs = len(scan1.spectr_bbc_final)
        number_of_channels = len(scan1.spectr_bbc_final[0])
        tmp_scans = np.zeros((number_of_bbcs, number_of_channels), dtype=np.float64)
        for i in range(number_of_bbcs):
            tmp_scans[i] = (scan1.spectr_bbc_final[i] / 1000.0 - scan2.spectr_bbc_final[i] / 1000.0) / 2.0
        return tmp_scans

    def remove_channels(
            self,
            bbc: int,
            remove_ranges: list[list[int]]) -> None:
        """
        This method is used to remove channels from the specified BBC and replace them with interpolated values.
        :param bbc: number of the Base Band Converted used (typically 1 - 4)
        :param remove_ranges: ranges of channels to remove (list of lists, where each inner list contains two integers: start and end channel)
        :return: None, but modifies the pols attribute of the MergedScan object by replacing the values in the specified ranges with interpolated values.
        """
        bbc_index = bbc - 1
        for remove_range in remove_ranges:
            start_channel = remove_range[0] - 1
            end_channel = remove_range[1] - 1
            print(f'------> Removing from channels {start_channel} to {end_channel}')
            for j in range(start_channel, end_channel, 1):
                self.pols[bbc_index][j] = self.__interpolate(bbc_index, start_channel, end_channel, j)

    def __interpolate(
            self,
            bbc_index: int,
            start_channel: int,
            end_channel: int,
            target_channel: int) -> float:
        """
        Interpolates the value for the target channel based on the values of the start and end channels.
        :param bbc_index: number of the Base Band Converted used (typically 0 - 3)
        :param start_channel: starting channel of the range to remove (integer)
        :param end_channel: ending channel of the range to remove (integer)
        :param target_channel: channel for which the interpolated value is to be calculated (integer)
        :return: interpolated value for the target channel (float)
        """
        y1 = self.pols[bbc_index][start_channel]
        x1 = start_channel
        y2 = self.pols[bbc_index][end_channel]
        x2 = end_channel
        a = (y1 - y2) / (x1 - x2)
        b = y1 - a * x1
        return a * target_channel + b

    def cancel_remove(
            self,
            bbc: int) -> None:
        bbc_index = bbc - 1
        self.pols[bbc_index] = self.backupPols[bbc_index]