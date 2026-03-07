"""
Class, that holds singular data scan from autocorrelator spectrometer
- and handles its processing
First version creation date: 05.03.2022
Owner: Michał Durjasz (md@astro.umk.pl)
This file is part of the NCU-SALSA-RT4 package
"""

# -- import block --
from numpy import exp, int64, sin, cos, asarray, sqrt, mean, pi, radians, zeros, inf, complex128, linspace
from numpy.fft import fft
from math import copysign
from cmath import sqrt as math_sqrt
from mpmath import nint
from astropy.time import Time
from astropy.coordinates import SkyCoord, FK5
import astropy.units as u
from sys import exit
import barycorrpy
import numpy as np
from datetime import datetime

class Scan:
    def __init__(self, filename, onOFF=False):
        self.isOnOff: bool = onOFF
        # constants
        self.c: float = 2.99792458e+5
        self.NN = 8192
        self.template_restfreqs = [
            1420.405751, # HI
            1612.23101,
            1665.40184,
            1667.35903,
            1720.52998,
            6668.518, # CH3OH
            6030.747,
            6035.092,
            6049.084,
            6016.746,
            4765.562,
            4592.976,
            4593.098,
            4829.664,
            2178.595]
        # read data
        self.fname = filename
        self.read_header_and_data()

    def read_header_and_data(self) -> None:
        """
        Reads data and metadata from singular scan file
        :return: None
        """
        try:
            fle = open(self.fname, "r+")
            file_lines = fle.readlines()
            fle.close()
        except FileNotFoundError:
            print("-----> File \"%s\" does not exist! Exiting..." % self.fname)
            print("-----------------------------------------")
            exit()

        # -- reading metadata block --
        self.read_file_metadata(file_lines)
        self.read_data(file_lines[19:])

    def read_data(self, a):
        # -- deklarujemy kontenery dla konkretnych BBC --
        self.auto = []
        self.bbc1I = []
        self.bbc2I = []
        self.bbc3I = []
        self.bbc4I = []
        self.no_of_channels = 4097
        # -- zapełniamy BBC --
        for i in range(self.no_of_channels):
            tmp = a[i].split()
            self.bbc1I.append(float(tmp[1]))
        for i in range(self.no_of_channels, 2 * self.no_of_channels):
            tmp = a[i].split()
            self.bbc2I.append(float(tmp[1]))
        for i in range(2 * self.no_of_channels, 3 * self.no_of_channels):
            tmp = a[i].split()
            self.bbc3I.append(float(tmp[1]))
        for i in range(3 * self.no_of_channels, 4 * self.no_of_channels):
            tmp = a[i].split()
            self.bbc4I.append(float(tmp[1]))

        # -- wycinamy kanał 1313 --
        self.bbc4I[1312] = self.bbc4I[1311]
        self.bbc4I[1313] = self.bbc4I[1311]
        self.bbc4I[1314] = self.bbc4I[1311]

        # -- agregujemy do jednej tablicy --
        self.auto.append(self.bbc1I)
        self.auto.append(self.bbc2I)
        self.auto.append(self.bbc3I)
        self.auto.append(self.bbc4I)

        # -- zamieniamy na numpy array --
        self.auto = asarray(self.auto)
        # ----- Koniec czytania danych --

    def read_file_metadata(self, file_lines: list[str]) -> None:
        """
        Reads metadata from scan file
        :param file_lines: list of strings, represented line by line
        :return: None
        """
        # -- source name --
        self.sourcename = file_lines[0].split("\'")[1].strip()
        self.INT = float((file_lines[0].split())[1])

        # -- coordinates: right ascension --
        tmp = file_lines[1].split()
        self.RA = float(tmp[1]) + float(tmp[2]) / 60.0 + float(tmp[3]) / 3600.0
        self.rah = int(tmp[1])
        self.ram = int(tmp[2])
        self.ras = int(tmp[3])

        # -- coordinates: declination --
        if float(tmp[4]) > 0:
            self.DEC = float(tmp[4]) + float(tmp[5]) / 60.0 + float(tmp[6]) / 3600.0
        elif float(tmp[4]) == 0.0:
            if tmp[4][0] == '-':
                self.DEC = -1.0 * (-1.0 * float(tmp[4]) + float(tmp[5]) / 60.0 + float(tmp[6]) / 3600.0)
            else:
                self.DEC = float(tmp[4]) + float(tmp[5]) / 60.0 + float(tmp[6]) / 3600.0
        else:
            self.DEC = -1.0 * (-1.0 * float(tmp[4]) + float(tmp[5]) / 60.0 + float(tmp[6]) / 3600.0)
        self.decd = int(tmp[4])
        self.decm = int(tmp[5])
        self.decs = int(tmp[6])

        # -- epoch --
        self.epoch = float(tmp[7])

        # -- coordinates: azimuth and elevation angles --
        tmp = file_lines[2].split()
        self.AZ = float(tmp[1])
        self.EL = float(tmp[2])
        self.azd = int(self.AZ)
        self.azm = int(60 * (self.AZ % 1))
        self.eld = int(self.EL)
        self.elm = int(60 * (self.EL % 1))

        # -- time --
        tmp = file_lines[4].split()
        # UT
        self.UTh = float(tmp[1])  # godzina UT
        self.UTm = float(tmp[2])  # minuta UT
        self.UTs = float(tmp[3])  # sekunda UT
        # ST
        self.STh = int(tmp[4])  # godzina ST
        self.STm = int(tmp[5])  # minuta ST
        self.STs = int(tmp[6])  # sekunda ST
        # decimal year
        tmp = file_lines[5].split()
        self.lsec = float(tmp[1])  # sekunda linukskowa
        self.dayn = float(tmp[2])  # dzień roku
        self.year = float(tmp[7])  # rok
        # search for month number - based on string
        self.monthname = tmp[4]
        self.monthtab = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        self.month = float(self.monthtab.index(self.monthname)) + 1
        self.day = float(tmp[5])

        # -- rest of date - adjust date format to match ISO 8601 standard --
        self.isotime = self._format_time_to_iso_8601_new()

        # -- calculate time using astropy time --
        self.time_of_observation = Time(self.isotime, format="isot", scale="utc")
        self.decimalyear = self.time_of_observation.decimalyear
        self.jd = self.time_of_observation.jd
        self.mjd = self.time_of_observation.mjd

        # -- create << datestring >> for saving in file --
        self.datestring = self._create_datestring()

        # -- frequencies --
        self.freq = []
        self.freqa = []
        self.rest = []
        self.bbcfr = []
        self.bbcnr = []
        self.polnames = []
        self.bw = []
        self.vlsr = []
        self.lo = []
        self.tsys = []
        self._fill_frequency_tables(file_lines)

    def _fill_frequency_tables(self, file_lines: list[str]) -> None:
        tmp = file_lines[6].split()
        for i in range(len(tmp) - 1):
            self.freq.append(float(tmp[i + 1]))

        tmp = file_lines[7].split()
        for i in range(len(tmp) - 1):
            self.freqa.append(float(tmp[i + 1]))

        tmp = file_lines[8].split()
        for i in range(len(tmp) - 1):
            self.rest.append(float(tmp[i + 1]))

        tmp = file_lines[9].split()
        for i in range(len(tmp) - 1):
            self.bbcfr.append(float(tmp[i + 1]))

        tmp = file_lines[10].split()
        for i in range(len(tmp) - 1):
            self.bbcnr.append(int(tmp[i + 1]))

        tmp = file_lines[11].split()
        for i in range(len(tmp) - 1):
            self.bw.append(float(tmp[i + 1]))

        tmp = file_lines[12].split()
        for i in range(len(tmp) - 1):
            self.polnames.append(tmp[i + 1])

        tmp = file_lines[13].split()
        for i in range(len(tmp) - 1):
            self.vlsr.append(float(tmp[i + 1]))

        tmp = file_lines[14].split()
        for i in range(len(tmp) - 1):
            self.lo.append(float(tmp[i + 1]))

        tmp = file_lines[15].split()
        for i in range(len(tmp) - 1):
            self.tsys.append(float(tmp[i + 1]))

        self.freq = asarray(self.freq)
        self.freqa = asarray(self.freqa)
        self.rest = asarray(self.rest)
        self.bbcfr = asarray(self.bbcfr)
        self.bbcnr = asarray(self.bbcnr)
        self.polnames = asarray(self.polnames)
        self.bw = asarray(self.bw)
        self.vlsr = asarray(self.vlsr)
        self.lo = asarray(self.lo)
        self.tsys = asarray(self.tsys)

    def _create_datestring(self) -> str:
        """
        (OBSOLETE) Create datestring in new format (DDMMYY) for saving in file name
        :return: datestring
        """
        datestring = ""
        if len(str(int(self.day))) == 1:
            datestring = datestring + "0" + str(int(self.day))
        else:
            datestring = datestring + str(int(self.day))

        if len(str(int(self.month))) == 1:
            datestring = datestring + "0" + str(int(self.month))
        else:
            datestring = datestring + str(int(self.month))
        datestring = datestring + str(int(self.year - 2000))
        return datestring


    def _format_time_to_iso_8601(self) -> str:
        """
        Based on the avaliable information creates
        time string in the ISO8601 format
        :return: string in ISO8601 format
        """
        isotime = str(int(self.year)) + "-"
        if len(str(int(self.month))) == 1:
            isotime = isotime + "0" + str(int(self.month)) + "-"
        else:
            isotime = isotime + str(int(self.month)) + "-"
        if len(str(int(self.day))) == 1:
            isotime = isotime + "0" + str(int(self.day)) + "T"
        else:
            isotime = isotime + str(int(self.day)) + "T"
        if len(str(int(self.UTh))) == 1:
            isotime = isotime + "0" + str(int(self.UTh)) + ":"
        else:
            isotime = isotime + str(int(self.UTh)) + ":"
        if len(str(int(self.UTm))) == 1:
            isotime = isotime + "0" + str(int(self.UTm)) + ":"
        else:
            isotime = isotime + str(int(self.UTm)) + ":"
        if len(str(int(self.UTs))) == 1:
            isotime = isotime + "0" + str(int(self.UTs))
        else:
            isotime = isotime + str(int(self.UTs))
        return isotime

    def _format_time_to_iso_8601_new(self) -> str:
        microsecond = int((self.UTs % 1) * 1_000_000)
        dt = datetime(
            int(self.year),
            int(self.month),
            int(self.day),
            int(self.UTh),
            int(self.UTm),
            int(self.UTs),
            microsecond
        )
        return dt.isoformat()

    def correct_auto(self) -> None:
        """
        This method is responsible for correcting the autocorrelation function
        This place takes significant amount of time (~ 1/3 of the overall processing time)
        :return: None
        """
        # -- declarations --
        self.average = zeros(4)
        self.auto0tab = zeros(4)
        self.multiple = zeros(4, dtype=int64)
        self.Nmax = zeros(4, dtype=int64)
        self.bias0 = zeros(4)
        self.zero_lag_auto = zeros(4)
        self.r0 = zeros(4)

        # -- loop for ACF correction --
        for i in range(len(self.auto)):
            # average last 240 channels - crucial for statistical estimations
            self.average[i] = mean(self.auto[i][3857:])

            # table with first values
            # in loaded data, first value is not part
            # of the ACF function
            self.auto0tab[i] = self.auto[i][0]

            # calculate real amount of "samples accumulated"
            if self.average[i] == 0.0: # 0 division failsafe
                self.multiple[i] = 0
            else:
                self.multiple[i] = int(nint(self.auto0tab[i] / self.average[i]))

            # calculate Nmax - the maximum number of samples that could be accumulated in the ACF function
            if self.multiple[i] == 0: # 0 division failsafe
                self.Nmax[i] = 0
            else:
                self.Nmax[i] = int(self.auto0tab[i] / self.multiple[i])

            # calculate bias
            if self.Nmax[i] == 0: # 0 division failsafe
                self.bias0[i] = 0.0
            else:
                self.bias0[i] = self.average[i] / self.Nmax[i] - 1

            # subtract the bias from the ACF function
            self.auto[i] -= self.Nmax[i]

            # gather << zero_lag_acf >> - value of the ACF function for tau = 0 (zero delay,
            # should be the highest point of the ACF function)
            self.zero_lag_auto[i] = self.auto[i][1]

            # calculate r0
            if self.Nmax[i] == 0: # 0 division failsafe
                self.r0[i] = 0.0
            else:
                self.r0[i] = self.zero_lag_auto[i] / self.Nmax[i]

            # normalize the ACF function to zero lag value
            if self.zero_lag_auto[i] == 0.0:
                self.auto[i] = zeros(len(self.auto[i])) # 0 division failsafe
            else:
                self.auto[i] = self.auto[i] / self.zero_lag_auto[i]

            for j in range(len(self.auto[i])):
                # to avoid nans and infs, we check the value of the correction before applying it
                tmp_number = self.__correctACF(self.auto[i][j], self.r0[i], self.bias0[i])
                if tmp_number is None:
                    self.auto[i][j] = 0.0
                else:
                    self.auto[i][j] = tmp_number

    def correct_auto_optimized(self) -> None:
        """
        Preprocesses the loaded ACF function for fourier transform
        This is optimized version of the previous utility: correct_auto
        :return:
        """
        # Convert self.auto to a 2D numpy array if it isn't already
        # Shape is (4, channels)
        auto_arr = np.array(self.auto)

        # 1. Batch Mean Calculation (last 240 channels)
        self.average = np.mean(auto_arr[:, 3857:], axis=1)

        # 2. Extract first and second values (tau=0 and tau=1)
        self.auto0tab = auto_arr[:, 0]
        self.zero_lag_auto_raw = auto_arr[:, 1]  # Temporary raw values

        # 3. Vectorized Math (Avoids 0-division using np.where)
        self.multiple = np.where(self.average != 0,
                                 np.round(self.auto0tab / self.average).astype(np.int64),
                                 0)

        self.Nmax = np.where(self.multiple != 0,
                             (self.auto0tab / self.multiple).astype(np.int64),
                             0)

        self.bias0 = np.where(self.Nmax != 0,
                              (self.average / self.Nmax) - 1,
                              0.0)

        # 4. Batch Bias Subtraction
        auto_arr -= self.Nmax[:, np.newaxis]

        # 5. Get corrected zero_lag_auto and r0
        self.zero_lag_auto = auto_arr[:, 1]
        self.r0 = np.where(self.Nmax != 0, self.zero_lag_auto / self.Nmax, 0.0)

        # 6. Normalization
        nonzero_mask = self.zero_lag_auto != 0
        # Create an output array of zeros
        norm_auto = np.zeros_like(auto_arr, dtype=np.float64)
        # Only divide where zero_lag_auto is non-zero
        norm_auto[nonzero_mask] = auto_arr[nonzero_mask] / self.zero_lag_auto[nonzero_mask, np.newaxis]

        self.auto = norm_auto  # Convert back to list if required by your class

        # 7. Van vleck correction
        self.__correct_acf_new()

    def hanning_smooth(self):
        """
        Applies Hanning smoothing to the ACF function
        """
        auto_arr = np.array(self.auto)
        num_cols = auto_arr.shape[1]
        j = np.arange(1, num_cols)
        cosine_window = np.cos(np.pi * (j - 1) / self.NN) ** 2.0
        auto_arr[:, 1:] *= cosine_window
        self.auto = auto_arr

    def doppset(
            self,
            source_ra_now_deg: float,
            source_dec_now_deg: float,
            observatory_lattitude: float,
            observatory_longitude: float,
            observatory_height_m_asl: float) -> None:
        """
        This method calculates doppler shift and rotates the spectrum accordingly
        :param source_ra_now_deg: right-ascension (after precession corretions applied!) - decimal degrees
        :param source_dec_now_deg: declination (after precession corrections applied!) - decimal degrees
        :param observatory_lattitude: observatory lattitude (degrees, float)
        :param observatory_longitude: observatory longitude
        :param observatory_height_m_asl: observatory height above sea level
        :return: None
        """
        # -- barycentric velocity of the observatory --
        # NOTE: we use nominal coordinates here, because barycorrpy performs precession on its own
        bar = barycorrpy.get_BC_vel(
            self.time_of_observation,
            ra=self.RA * 15,
            dec=self.DEC,
            lat=observatory_lattitude,
            longi=observatory_longitude,
            alt=observatory_height_m_asl,
            epoch=2000)
        self.baryvel = bar[0][0] / 1000.0
        # -- velocity in the local standard of the rest - projected into the observation direction --
        self.lsrvel = self.__lsr_motion(source_ra_now_deg, source_dec_now_deg, self.decimalyear)
        # -- final velocity for doppler shift calculation --
        self.Vdop = self.baryvel + self.lsrvel

        # -- rest freq correction --
        # replace the file rest frequency with values from the standard table (if they match)
        for i in range(len(self.auto)):  # iteracja po indeksach
            for tmp_ind in self.template_restfreqs:  # iteracja po templatkach
                if int(self.rest[i]) == int(tmp_ind):
                    self.rest[i] = tmp_ind
                    break

        # -- doppler shift correction --
        # -- or so called rotating the spectrum --
        # declare tables
        self.fvideo = zeros(4)
        self.kanalf = zeros(4, dtype=int64)
        self.q = zeros(4)
        self.kanalv = zeros(4)

        # --- rotowanie oryginalnego widma ---
        # move the emission line to 1 / 4 of the bandwidth
        self.lo[0] = self.lo[0] - (self.bw[0] / 4)
        # real observed frequency of the line (after doppler shift)
        self.fsky = self.rest - self.rest * (-self.Vdop + self.vlsr) / self.c
        # frequency of the line in the intermediate frequency (IF) domain
        self.f_IF = self.fsky - self.lo[0]

        # miscellaneous parameters
        self.NNch = len(self.auto[0]) - 1  # faktyczna ilość kanałów
        for i in range(len(self.auto)):  # po BBC
            # fvideo
            self.fvideo[i] = self.f_IF[i] - copysign(self.bbcfr[i], self.f_IF[i])
            # kanalf (line in frequency domain)
            self.kanalf[i] = int(self.NNch * abs(self.fvideo[i]) / self.bw[i] + 1)
            # q (line position in the spectrum, in channels, with sign)
            if self.fvideo[i] < 0.0:
                self.kanalf[i] = self.NNch - self.kanalf[i] + 1
                self.q[i] = (-self.fvideo[i] / self.bw[i])
            else:
                self.q[i] = 1.0 - self.fvideo[i] / self.bw[i] - 1.0 / self.NNch

        # kanalv (line in velocity domain)
        self.kanalv = self.NNch - self.kanalf + 1
        # line velocity in frequency domain (???)
        targetChan = 1024
        self.v1024f = self.vlsr + (targetChan - self.kanalf) * (-self.c * self.bw) / (self.rest * self.NNch)
        # velocity in 1024 channel in velocity domain
        self.v1024v = self.vlsr + (targetChan - self.kanalv) * (self.c * self.bw) / (self.rest * self.NNch)

        # no of channels to rotate
        self.fc = self.q * self.NNch - targetChan
        self.fcBBC = self.fc

        # -- fft preparation --
        self.fr = - (self.fc + 1) * 2.0 * pi / self.NN

        # -- FFT table - declare a table to hold data after fourier transform --
        self.auto_prepared_to_fft = zeros((4, self.NN), dtype=complex128)  # docelowa

        # -- spectrum rotation and FFT preparation --
        # -- rotation is done by multiplying the ACF function by exp(sqrt(-1) * self.fr) --
        for w in range(len(self.auto)):  # iteruje po BBC
            # calculate phases and rotate spectrum
            phases = linspace(0, int(self.NN / 2) - 1, int(self.NN / 2)) * self.fr[w]  # fazy
            shift_coeffs = exp(math_sqrt(-1) * phases)
            tmpwp = self.auto[w][1:] * shift_coeffs
            # prepare the table for fft
            # non-mirror part
            self.auto_prepared_to_fft[w][:int(self.NN / 2)] = tmpwp
            # mirror part
            self.auto_prepared_to_fft[w][int(self.NN / 2) + 1:] = tmpwp[::-1][
                :-1].conjugate()  # odwracamy znakiem część zespoloną
            # last sample
            self.auto_prepared_to_fft[w][4096] = (0 + 0j)
        # --------------

    def do_statistics(self) -> None:
        """
        Miscelanneous statistics to calculate, not included in other procedures
        :return: None
        """
        self.rMean = self.bias0 * 100.0
        self.ACF0 = self.r0
        self.V_sigma = zeros(4)
        for i in range(len(self.auto)):
            self.V_sigma[i] = self.__clip_level(self.r0[i])

    def scale_tsys_to_mK(self) -> None:
        """
        Scale system temperature to mK, and apply a failsafe for negative values (set them to 1000 K)
        :return: None
        """
        for i in range(len(self.auto)):
            self.tsys[i] = self.tsys[i] * 1000.0
            if self.tsys[i] < 0.0:
                self.tsys[i] = 1000.0 * 1000.0

    def perform_fourier_transform(self) -> None:
        """
        Performs fourier transform on the ACF function, and prepares the final spectrum for further processing
        """
        # declare tables to save some time
        self.spectr_bbc = zeros((4, self.NN))
        self.spectr_bbc_final = zeros((4, int(self.NN / 2)))
        # iterate through BBC
        for i in range(len(self.auto)):
            self.spectr_bbc[i] = fft(self.auto_prepared_to_fft[i]).real
            # check to take lower or higher spectrium
            # it depends on the sign of fvideo - if it's positive, we take upper spectrum, if it's negative, we take lower spectrum
            # most certainly fvideo will be always high, but one can never tell
            if self.fvideo[i] > 0:
                self.spectr_bbc_final[i] = self.spectr_bbc[i][int(self.NN / 2):]
            else:
                self.spectr_bbc_final[i] = self.spectr_bbc[i][:int(self.NN / 2)]

    def calibrate_in_tsys(self) -> None:
        """
        Calibrate specturm in system temperature units
        """
        for i in range(len(self.auto)):
            self.spectr_bbc_final[i] = self.spectr_bbc_final[i] * self.tsys[i]


    def extended_print(self):
        """
        Extended spectrum information - normally disabled, but useful for debugging and checking the results of doppler shift correction
        """
        print('f(LSR)/MHz   f(sky)      LO1(RF)    LO2(BBC)   fvideo   v(Dopp) [km/s] V(LSR)')
        # print(tab[i].rest[0], tab[i].fsky[0], tab[i].lo[0], tab[i].bbcfr[0], tab[i].fvideo[0], -Vdop, tab[i].vlsr)
        print('%.3f    %.3f    %.3f    %.3f    %.3f    %.3f       %.3f' % (self.rest[0], self.fsky[0], self.lo[0],
                                                                           self.bbcfr[0], self.fvideo[0], -self.Vdop,
                                                                           self.vlsr[0]))
        print('====> Frequency domain: line is in', self.kanalf[0], '=', round(self.v1024f[0], 3), 'km/s')
        print('====> Velocity domain: line is in', self.kanalv[0], '=', round(self.v1024v[0], 3), 'km/s')
        print('Output spectra were rotated by', round(self.fcBBC[0], 3), 'channels')
        if self.fvideo[0] > 0:
            date6 = ' (USBeff)'
        else:
            date6 = ' (LSBeff)'
        print('ACFs', date6, 'Nmax =', int(self.Nmax[3]), '    BBC1   ', '  BBC2   ', '   BBC3   ', '  BBC4')
        print("r0 =                                %.4f    %.4f    %.4f    %.4f" % (self.ACF0[0], self.ACF0[1],
                                                                                    self.ACF0[2], self.ACF0[3]))
        print("rmean (bias of 0) =                 %.4f    %.4f    %.4f    %.4f" % (self.rMean[0], self.rMean[1],
                                                                                    self.rMean[2], self.rMean[3]))
        print("Threshold (u=V/rms) =               %.4f    %.4f    %.4f    %.4f" % (self.V_sigma[0], self.V_sigma[1],
                                                                                    self.V_sigma[2], self.V_sigma[3]))

    def __lsr_motion(
            self,
            target_ra_deg: float,
            target_dec_deg: float,
            decimalyear: float) -> float:
        """
        Calculates sun velocity relative to the local standard of rest, projected into the direction of the source
        :param target_ra_deg: right ascension of the source (after precession corrections applied!) - in degrees
        :param target_dec_deg: declination of the source (after precession corrections applied!) - in degrees
        :param decimalyear: decimal year of the observation (after precession corrections applied!)
        :return: velocity of the sun relative to the local standard of rest, projected into the direction of the source (in km/s)
        """
        v_sun_0 = 20.0
        # -- sun apex coordinates in the year 1900 (in radians) --
        ras = 18.0 * pi / 12.0
        decs = 30.0 * pi / 180.0
        sunc = SkyCoord(ras * u.rad, decs * u.rad, frame=FK5, equinox="B1900")

        # -- actual observation epoch FK5 frame --
        sunc_now = FK5(equinox="J" + str(decimalyear))
        sunc_new = sunc.transform_to(sunc_now) # transformation from 1900 to obs. epoch
        sun_dec_now_rad = radians((sunc_new.dec * u.degree).value)
        sun_ra_now_rad = radians((sunc_new.ra * u.degree).value)

        # -- prepare some values to simplify final formula --
        target_ra_rad = radians(target_ra_deg)
        target_dec_rad = radians(target_dec_deg)
        cdec = cos(target_dec_rad)
        sdec = sin(target_dec_rad)

        # -- final calculation --
        v_sun = v_sun_0 * (sin(sun_dec_now_rad) * sdec + cos(sun_dec_now_rad) * cdec * cos(target_ra_rad - sun_ra_now_rad))
        return v_sun


    def __correctACF(
            self,
            autof: float,
            r0: float,
            r_mean: float) -> float:
        """
        Calculate correction to the autocorrelation function for several cases of 3- and 2-level autocorrelator
        This method is 1 to 1 ported from K. Borkowski's A2S program, which is written in Fortran77
        :param autof: acf. function point
        :param r0: correlation coefficient for zero delay
        :param r_mean: mean coefficient for the tail of the acf function (higher delays)
        :return: corrected acf function point
        """

        if r_mean <= 1e-5:
            r = min([1.0, abs(autof)])
            if r0 > 0 and r0 < 0.3:
                r = r * 0.0574331
                rho = r * (60.50861 + r * (-1711.23607 + r * (26305.13517 - r * 167213.89458))) / 0.99462695104383
                correct_auto = copysign(rho, autof)
                return correct_auto
            elif r0 > 0.3 and r0 < 0.9:
                r = r * 0.548506
                rho = (r * (2.214 + r * (0.85701 + r * (-7.67838 + r * (22.42186 - r * 24.896))))) / 0.998609598617374
                correct_auto = copysign(rho, autof)
                return correct_auto
            elif r0 > 0.9:
                rho = sin(1.570796326794897 * r)
                correct_auto = copysign(rho, autof)
                return correct_auto
        else:
            autof2 = autof ** 2.0
            if (abs(autof)) < 0.5:
                fac = 4.167810515925 - r0 * 7.8518131881775
                a = -0.0007292201019684441 - 0.0005671518541787936 * fac
                b = 1.2358980680949918 + 0.03931789097196692 * fac
                c = -0.11565632506887912 + 0.08747950965746415 * fac
                d = 0.01573239969731158 - 0.06572872697836053 * fac
                correct_auto = a + (b + (c + d * autof2) * autof2) * autof
                return correct_auto
            elif autof > 0.5:
                correct_auto = -1.1568973833585783 + 10.27012449073475 * autof - 27.537554958512125 * autof2 + 40.54762923890069 * autof ** 3 - 28.758995213769058 * autof2 ** 2 + 7.635693826008257 * autof ** 5 + 0.218044850 * (
                            0.53080867 - r0) * cos(3.12416 * (autof - 0.49721))
                return correct_auto
            else:
                correct_auto = -0.0007466171982634772 + autof * (1.2660000881004778 + autof2 * (
                            -0.4237089538779861 + autof2 * (1.0910718879007775 + autof2 * (
                                -1.452946181286572 + autof2 * 0.520334578982730)))) + 0.22613222 * (
                                           0.53080867 - r0) * cos(3.11635 * (autof - 0.49595))
                return correct_auto

    def __correct_acf_new(self) -> None:
        # Ensure auto is a 2D numpy array: (4, N)
        auto_arr = np.array(self.auto)

        # Reshape scalar-per-row values to (4, 1) for 2D broadcasting
        r0 = self.r0[:, np.newaxis]
        r_mean = self.bias0[:, np.newaxis]

        # Initialize output
        result = np.zeros_like(auto_arr)

        # --- BRANCH A: Low r_mean (r_mean <= 1e-5) ---
        # We create a mask for the rows that meet the r_mean condition
        row_mask_a = (self.bias0 <= 1e-5)
        if np.any(row_mask_a):
            # Slice only the rows needed
            a_autof = auto_arr[row_mask_a]
            a_r0 = self.r0[row_mask_a][:, np.newaxis]
            a_res = np.zeros_like(a_autof)

            r = np.clip(np.abs(a_autof), 0, 1.0)
            sgn = np.sign(a_autof)

            # Sub-branch 1: 0 < r0 < 0.3
            m1 = (self.r0[row_mask_a] > 0) & (self.r0[row_mask_a] < 0.3)
            if np.any(m1):
                rs = r[m1] * 0.0574331
                rho = rs * (60.50861 + rs * (-1711.23607 + rs * (26305.13517 - rs * 167213.89458))) / 0.99462695104383
                a_res[m1] = sgn[m1] * rho

            # Sub-branch 2: 0.3 <= r0 < 0.9
            m2 = (self.r0[row_mask_a] >= 0.3) & (self.r0[row_mask_a] < 0.9)
            if np.any(m2):
                rs = r[m2] * 0.548506
                rho = (rs * (2.214 + rs * (
                            0.85701 + rs * (-7.67838 + rs * (22.42186 - rs * 24.896))))) / 0.998609598617374
                a_res[m2] = sgn[m2] * rho

            # Sub-branch 3: r0 >= 0.9
            m3 = (self.r0[row_mask_a] >= 0.9)
            if np.any(m3):
                a_res[m3] = sgn[m3] * np.sin(1.570796326794897 * r[m3])

            result[row_mask_a] = a_res

        # --- BRANCH B: High r_mean (r_mean > 1e-5) ---
        row_mask_b = (self.bias0 > 1e-5)
        if np.any(row_mask_b):
            b_autof = auto_arr[row_mask_b]
            b_r0 = self.r0[row_mask_b][:, np.newaxis]
            b_autof2 = b_autof ** 2
            b_res = np.zeros_like(b_autof)

            # Mask 1: abs < 0.5
            m_small = np.abs(b_autof) < 0.5
            fac = 4.167810515925 - b_r0 * 7.8518131881775
            # Pre-calculating constants for the row
            a = -0.0007292201019684441 - 0.0005671518541787936 * fac
            b = 1.2358980680949918 + 0.03931789097196692 * fac
            c = -0.11565632506887912 + 0.08747950965746415 * fac
            d = 0.01573239969731158 - 0.06572872697836053 * fac
            b_res[m_small] = (a + (b + (c + d * b_autof2) * b_autof2) * b_autof)[m_small]

            # Mask 2: > 0.5
            m_pos = b_autof >= 0.5
            if np.any(m_pos):
                b_res[m_pos] = (-1.1568973833585783 + 10.27012449073475 * b_autof - 27.537554958512125 * b_autof2
                                + 40.54762923890069 * b_autof ** 3 - 28.758995213769058 * b_autof2 ** 2
                                + 7.635693826008257 * b_autof ** 5
                                + 0.218044850 * (0.53080867 - b_r0) * np.cos(3.12416 * (b_autof - 0.49721)))[m_pos]

            # Mask 3: <= -0.5
            m_neg = b_autof <= -0.5
            if np.any(m_neg):
                b_res[m_neg] = (-0.0007466171982634772 + b_autof * (1.2660000881004778 + b_autof2 * (
                            -0.4237089538779861 + b_autof2 * (
                                1.0910718879007775 + b_autof2 * (-1.452946181286572 + b_autof2 * 0.520334578982730))))
                                + 0.22613222 * (0.53080867 - b_r0) * np.cos(3.11635 * (b_autof - 0.49595)))[m_neg]

            result[row_mask_b] = b_res

        # Final cleanup and store
        self.auto = np.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0)

    def __clip_level(
            self,
            ratio: float) -> float | None:
        """
        Clip_level method
        copied from original A2S program
        :param ratio:  ???
        :return: corrected clip level
        """
        err = 1 - ratio
        x = 0
        for i in range(100):
            dE = self.__erf(x + 0.01) - self.__erf(x)
            if dE == 0.0:
                return inf
            x = x + (err - self.__erf(x)) * 0.01 / dE
            if abs(err - self.__erf(x)) < 0.0001:
                return sqrt(2.0) * x
            else:
                continue
        return None


    def __erf(
            self,
            x: float) -> float:
        """
        clip_level helper method
        :param x:
        :return:
        """
        t = 1.0 / (1 + 0.3275911 * x)
        erf = 1.0 - t * (
                    0.254829592 + t * (-0.284496736 + t * (1.421413741 + t * (-1.453152027 + t * 1.061405429)))) * exp(
            -x ** 2)
        return erf
