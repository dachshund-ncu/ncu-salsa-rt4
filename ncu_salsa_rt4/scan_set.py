"""
Module: ncu_salsa_rt4.scan_set
Creation date: 2026-02-01
Owner: Michał Durjasz
"""
import tarfile
from astropy.coordinates import SkyCoord, FK5
import astropy.units as u
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from .scan import Scan
from .scan_merged import MergedScan
import os

class ScanSet:
    def __init__(
            self,
            archive_filename: str,
            on_off: bool = False,
            debug: bool = False):
        self.isOnOff = on_off
        self.debug = debug
        self.archive_directory = os.path.dirname(archive_filename)
        self.archive_filename = archive_filename
        self.__tmp_dir_name = os.path.join(self.archive_directory, ".tmp_auto_scans_data")
        self.__load_data(self.archive_filename)
        self.__process_data()

    def __load_data(self, archive_filename: str):
        # unpack scan filenames
        self.scan_filenames = self.__unpack_archive(archive_filename)
        # load data
        self.scans = self.read_scans_from_filenames(self.scan_filenames)
        self.__make_initial_settings()
        self.__remove_unpacked_data()

    def __process_data(self):
        self.noOfScans = len(self.scans)
        self.scans = self.proceed_scans_sequential()
        # self.scans = self.proceed_scans()
        self.mergedScans: list[MergedScan] = self.merge_scans(self.scans)
        self.mjd = self.__calculate_mjd()

    def __unpack_archive(self, archive_filename: str):
        archive = tarfile.open(archive_filename, "r:bz2")
        list_of_files = [os.path.join(self.__tmp_dir_name, f) for f in archive.getnames()]
        if len(list_of_files) %2 != 0:
            list_of_files = list_of_files[:-1]
        try:
            os.mkdir(self.__tmp_dir_name)
        except FileExistsError:
            for i in os.listdir(self.__tmp_dir_name):
                os.remove(os.path.join(self.__tmp_dir_name, i))
        archive.extractall(self.__tmp_dir_name)
        return list_of_files

    def __remove_unpacked_data(self):
        for f in self.scan_filenames:
            os.remove(f)
        os.rmdir(self.__tmp_dir_name)

    def read_scans_from_filenames(self, list_of_filenames):
        return [Scan(f, self.isOnOff) for f in list_of_filenames]

    def __make_initial_settings(self):
        """
        Here we perform a set of initial settings, needed for further processing
        """
        # CONSTANTS
        # Piwnice 32 m radio telescop - location
        self.longitude_deg = 18.56406  # deg
        self.latitude_deg = 53.09546  # deg
        self.height_m_asl = 133.61  # meters above sea level
        # COORDINATES
        # source (from scan file header in J2000)
        self.source_J2000 = SkyCoord(
            ra=self.scans[0].RA * u.hourangle,
            dec=self.scans[0].DEC * u.degree,
            frame=FK5,
            equinox='J2000')
        # precession and nutation
        self.frame_now = FK5(equinox="J" + str(self.scans[0].decimalyear)) # coordinates for scan epoch
        self.source_JNOW = self.source_J2000.transform_to(self.frame_now) # precession to actual position
        # galactic coordinates
        self.l_ga = self.source_JNOW.galactic
        self.source_L = (self.l_ga.l * u.degree).value
        self.source_B = (self.l_ga.b * u.degree).value
        self.source_ld = int(self.source_L)
        self.source_lm = int(60.0 * (self.source_L % 1))
        self.source_bd = int(self.source_B)
        self.source_bm = int(60.0 * (self.source_B % 1))

    def _process_single_scan(self, scan, source_J2000, latitude_deg, longitude_deg, height_m_asl, debug, i):
        # processing position
        frame_now = FK5(equinox="J" + str(scan.decimalyear))
        source_JNOW = source_J2000.transform_to(frame_now)
        source_JNOW_RA = (source_JNOW.ra * u.degree).value
        source_JNOW_DEC = (source_JNOW.dec * u.degree).value
        # signal processing
        scan.correct_auto(scannr=i + 1)
        scan.hanning_smooth()
        scan.doppset(source_JNOW_RA, source_JNOW_DEC, latitude_deg, longitude_deg, height_m_asl)
        if debug:
            print("-----> scan %d: line rotated by %4.3f channels" % (i + 1, round(scan.fcBBC[0], 3)))
        scan.do_statistics()
        scan.scale_tsys_to_mK()
        scan.make_transformata_furiata()
        scan.calibrate_in_tsys()
        return scan

    def proceed_scans(self):
        with ThreadPoolExecutor() as executor:
            results = list(executor.map(
                self._process_single_scan,
                self.scans,
                [self.source_J2000] * len(self.scans),
                [self.latitude_deg] * len(self.scans),
                [self.longitude_deg] * len(self.scans),
                [self.height_m_asl] * len(self.scans),
                [self.debug] * len(self.scans),
                range(len(self.scans))
            ))
        return results

    def proceed_scans_sequential(self):
        for i in range(len(self.scans)):
            self.scans[i] = self._process_single_scan(
                self.scans[i],
                self.source_J2000,
                self.latitude_deg,
                self.longitude_deg,
                self.height_m_asl,
                self.debug,
                i
            )
        return self.scans

    def merge_scans(self, scans):
        return [MergedScan(scans[i], scans[i + 1]) for i in range(0, int(self.noOfScans), 2)]

    def __calculate_mjd(self):
        return np.mean([s.mjd for s in self.scans])
