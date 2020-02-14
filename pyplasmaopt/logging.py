import logging
import os
from mpi4py import MPI
comm = MPI.COMM_WORLD

__all__ = ("debug", "info", "warning", "error", "set_file_logger")

logger = logging.getLogger('PyPlasmaOpt')

handler = logging.StreamHandler()
formatter = logging.Formatter(fmt="%(levelname)s %(message)s")
handler.setFormatter(formatter)
if comm is not None and comm.rank != 0:
    handler = logging.NullHandler()
logger.addHandler(handler)

def set_file_logger(path):
    filename, file_extension = os.path.splitext(path)
    from math import log10, ceil
    digits = ceil(log10(comm.size))
    fileHandler = logging.FileHandler(filename + "-rank" + ("%i" % comm.rank).zfill(digits) + file_extension, mode='a')
    formatter = logging.Formatter(fmt="%(asctime)s:%(name)s:%(levelname)s %(message)s")
    fileHandler.setFormatter(formatter)
    logger.addHandler(fileHandler)

logger.setLevel(logging.INFO)

debug = logger.debug
info = logger.info
warning = logger.warning
error = logger.error
