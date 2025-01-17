"""
Models for Transmision Electron Microscopy
    * VERY IMPORTANT: this package makes usage of some IMOD binaries, so they must be installed in the system

"""

__author__ = "Antonio Martinez-Sanchez"

import os
import time
import math
import subprocess
import numpy as np
import scipy as sp
from polnet import lio

## IMOD commands
IMOD_CMD_XYZPROJ = "xyzproj"
IMOD_CMD_TILT = "tilt"
IMOD_CMD_AHEADER = "alterheader"


class TEM:
    """
    Class for modeling a Transmission Electron Microscope
    """

    def __init__(self, work_dir):
        """
        :param work_dir: path to working directory where intermediate files are stored
        """
        self.__work_dir = work_dir
        self.__log_file = self.__work_dir + "/TEM.log"
        self.__create_work_dir()
        self.__vol_file = self.__work_dir + "/in_vol.mrc"
        self.__micgraphs_file = self.__work_dir + "/out_micrographs.mrc"
        self.__tangs_file = self.__work_dir + "/out_tangs.tlt"
        self.__rec3d_file = self.__work_dir + "/out_rec3d.mrc"

    def __create_work_dir(self):
        """Create the working directory"""
        if not os.path.exists(self.__work_dir):
            os.makedirs(self.__work_dir, exist_ok=True)
            
    def __save_tangs_file(self, angs):
        """
        Stores the tilt angles file according IMOD format
        :param angs: non-empty iterable with the tilt angles
        """
        with open(self.__tangs_file, "w") as file:
            for ang in angs:
                file.write(f"{float(ang):.1f}\n")

    def __load_tangs_file(self):
        """
        Load the tilt angles file into an array

        :return: output array with the tilt angles
        """
        angs = list()
        with open(self.__tangs_file, "r") as file:
            for line in file:
                angs.append(float(line.strip()))
        return np.asarray(angs)

    def __format_tilt_angles(self, angs):
        """
        Format tilt angles for command line usage
        :param angs: iterable of angles or range object
        :return: properly formatted string of angles
        """
        if isinstance(angs, range):
            return f"{angs.start},{angs.stop},{angs.step}"
        return ",".join(f"{float(ang):.1f}" for ang in angs)

    def gen_tilt_series_imod(self, vol, angs, ax="X", mode="real"):
        """
        Generates the 2D projection series from a 3D volume using 'xyzproj' IMOD binary

        :param vol: input 3D volume
        :param angs: non-empty iterable with the tilt angles or a range
        :param ax: tilt axis, either 'X', 'Y' or 'Z' (default 'X')
        :param mode: mode of output file, valid: 'byte', 'int' or 'real' (default)
        """
        # Input validation
        if not isinstance(vol, np.ndarray) or len(vol.shape) != 3:
            raise ValueError("Volume must be a 3D numpy array")
        if not hasattr(angs, "__len__") or len(angs) == 0:
            raise ValueError("Angles must be a non-empty iterable")
        if ax not in ["X", "Y", "Z"]:
            raise ValueError("Axis must be 'X', 'Y', or 'Z'")
        if mode not in ["byte", "int", "real"]:
            raise ValueError("Mode must be 'byte', 'int', or 'real'")

        # Save input volume
        lio.write_mrc(vol, self.__vol_file)

        # Build command with proper argument separation
        cmd = [
            IMOD_CMD_XYZPROJ,
            "-inp", self.__vol_file,
            "-o", self.__micgraphs_file,
            "-ax", ax
        ]

        # Add tilt angles with proper formatting
        if isinstance(angs, range):
            cmd.extend(["-an", self.__format_tilt_angles(angs)])
        else:
            cmd.extend(["-ta", self.__format_tilt_angles(angs)])

        # Add mode as separate argument
        mode_map = {"byte": "0", "int": "1", "real": "2"}
        cmd.extend(["-m", mode_map[mode]])

        # Execute command
        try:
            with open(self.__log_file, "a") as log_file:
                log_file.write(f"\n[{time.strftime('%c')}]RUNNING COMMAND:-> {' '.join(cmd)}\n")
                result = subprocess.run(
                    cmd,
                    stdout=log_file,
                    stderr=subprocess.STDOUT,
                    check=True,
                    text=True
                )

            # Save tilt angles file if command succeeded
            self.__save_tangs_file(angs)
            
            # Verify output file exists and is non-empty
            if not os.path.exists(self.__micgraphs_file):
                raise RuntimeError(f"Output file not created: {self.__micgraphs_file}")
            if os.path.getsize(self.__micgraphs_file) == 0:
                raise RuntimeError(f"Output file is empty: {self.__micgraphs_file}")

        except subprocess.CalledProcessError as e:
            print(f"Error running IMOD command: {' '.join(cmd)}")
            print(f"Return code: {e.returncode}")
            if e.output:
                print(f"Output: {e.output}")
            raise
        except IOError as e:
            print(f"I/O error during command execution: {e}")
            raise

    def add_detector_noise(self, snr):
        """
        Add detector noise to micrographs. Readout noise has Gaussian distribution and dark current is typically
        one order magnitude lower, so the last one is neglected.

        :param snr: target snr to determine the level of noise to be added, linear scale so greater than zero
        """

        # Input parsing
        assert os.path.exists(self.__micgraphs_file)
        assert snr > 0

        # Load micrographs
        mics = lio.load_mrc(self.__micgraphs_file)

        rng = np.random.default_rng()
        # Adding readout noise with Gaussian distribution to every micrograph
        for i in range(mics.shape[2]):
            mic = mics[:, :, i]
            mask = mic > 0
            mn = mic[mask].mean()
            sg_fg = mn / snr
            mics[:, :, i] = mic + rng.normal(mn, sg_fg, mic.shape)

        # Update micrographs file
        lio.write_mrc(mics, self.__micgraphs_file)

    def recon3D_imod(self, thick=None):
        """
        Performs a 3D reconstruction from the tilted series micrograph using 'tilt' IMOD binary
        :param thick: (optional) to enable a tomogram thickness (along Z-axis) different from the original density.
        """

        # Call to IMOD binary (tilt)

        # Building the command
        tilt_cmd = [IMOD_CMD_TILT]
        vol = lio.load_mrc(self.__vol_file, mmap=True, no_saxes=False)
        tilt_cmd += ["-inp", self.__micgraphs_file]
        tilt_cmd += ["-output", self.__rec3d_file]
        tilt_cmd += ["-TILTFILE", self.__tangs_file]
        if thick is None:
            tilt_cmd += ["-THICKNESS", str(vol.shape[0])]
        else:
            assert thick > 0
            tilt_cmd += ["-THICKNESS", str(thick)]

        # Command calling
        try:
            with open(self.__log_file, "a") as file_log:
                file_log.write(
                    "\n["
                    + time.strftime("%c")
                    + "]RUNNING COMMAND:-> "
                    + " ".join(tilt_cmd)
                    + "\n"
                )
                subprocess.call(tilt_cmd, stdout=file_log, stderr=file_log)
        except subprocess.CalledProcessError:
            print("ERROR: Error calling the command:", tilt_cmd)
            raise subprocess.CalledProcessError
        except IOError:
            print("ERROR: Log file could not be written:", self.__log_file)
            raise IOError

        # Swap Y-Z axes from the output given by IMOD
        hold_rec = np.swapaxes(lio.load_mrc(self.__rec3d_file), 1, 2)
        # Flip Z-axis
        lio.write_mrc(np.flip(hold_rec, axis=2), self.__rec3d_file)

    def set_header(self, data="mics", p_size=None, origin=None):
        """
        Set 3D reconstructed tomogram pixel (voxel) size using alter 'alterheader' IMOD script

        :param data: determines the data where the changes will be applied, valid: 'mics' or 'rec3d'
        :param p_size: pixel size X, Y and Z dimensions
        :param origin: origin X, Y and Z dimensions
        """
        assert (data == "mics") or (data == "rec3d")
        if p_size is not None:
            assert hasattr(p_size, "__len__") and len(p_size) == 3
        if origin is not None:
            assert hasattr(origin, "__len__") and len(origin) == 3

        # Call to IMOD binary (tilt)

        # Building the command
        aheader_cmd = [IMOD_CMD_AHEADER]
        if data == "mics":
            aheader_cmd += [
                self.__micgraphs_file,
            ]
        else:
            aheader_cmd += [
                self.__rec3d_file,
            ]
        if p_size is not None:
            aheader_cmd += [
                "-del",
                str(p_size[0]) + "," + str(p_size[1]) + "," + str(p_size[2]),
            ]
        if origin is not None:
            aheader_cmd += [
                "-org",
                str(origin[0]) + "," + str(origin[1]) + "," + str(origin[2]),
            ]

        # Command calling
        try:
            with open(self.__log_file, "a") as file_log:
                file_log.write(
                    "\n["
                    + time.strftime("%c")
                    + "]RUNNING COMMAND:-> "
                    + " ".join(aheader_cmd)
                    + "\n"
                )
                subprocess.call(aheader_cmd, stdout=file_log, stderr=file_log)
        except subprocess.CalledProcessError:
            print("ERROR: Error calling the command:", aheader_cmd)
            raise subprocess.CalledProcessError
        except IOError:
            print("ERROR: Log file could not be written:", self.__log_file)
            raise IOError

    def invert_mics_den(self):
        """
        Invert micrographs density
        """
        mics = lio.load_mrc(self.__micgraphs_file)
        lio.write_mrc(-1 * mics, self.__micgraphs_file)

    def add_mics_misalignment(self, mn, mx, n_sigma=0):
        """
        Add random X and Y misalignment to each micrograh in the tilt series following a sinusoidal model:
        f(t_angle) = mn + mx(sin(t_angle)/sin(max_t_angle))

        :param mn: minimun mislignment value (t_angle = 0)
        :param mx: maximum mislignment value (t_angle = max_t_angle)
        :param n_sigma: sigma value for Gaussian noise.
        :return: None
        """

        assert mx >= mn

        rng = np.random.default_rng()

        # Micrographs loop
        mics = lio.load_mrc(self.__micgraphs_file)
        angs = np.abs(np.radians(self.__load_tangs_file()))
        n_angs = len(angs)
        shifts = (
            mn
            + np.sin(angs) / np.sin(angs.max())
            + rng.normal(0, n_sigma, n_angs)
        )
        split_fs = rng.uniform(0, 1, n_angs)
        for i, shift, split_f in zip(range(n_angs), shifts, split_fs):
            shift_x = shift / math.sqrt(split_f + 1)
            shift_y = split_f * shift_x
            mics[:, :, i] = sp.ndimage.shift(
                mics[:, :, i],
                (shift_x, shift_y),
                output=None,
                order=3,
                mode="constant",
                cval=0.0,
                prefilter=True,
            )

        # Save shited micrographs
        lio.write_mrc(mics, self.__micgraphs_file)
