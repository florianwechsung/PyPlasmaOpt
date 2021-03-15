import logging
import os
from mpi4py import MPI
comm = MPI.COMM_WORLD

__all__ = ("debug", "info", "warning", "error", "debug_all", "info_all", "warning_all", "error_all", "info_all_sync", "set_file_logger")

logger = logging.getLogger('PyPlasmaOpt')

handler = logging.StreamHandler()
formatter = logging.Formatter(fmt="%(levelname)s %(message)s")
handler.setFormatter(formatter)
if comm is not None and comm.rank != 0:
    handler = logging.NullHandler()
logger.addHandler(handler)

from math import log10, ceil
digits = ceil(log10(comm.size))
def set_file_logger(path):
    filename, file_extension = os.path.splitext(path)
    fileHandler = logging.FileHandler(filename + "-rank" + ("%i" % comm.rank).zfill(digits) + file_extension, mode='a')
    formatter = logging.Formatter(fmt="%(asctime)s:%(name)s:%(levelname)s %(message)s")
    fileHandler.setFormatter(formatter)
    logger.addHandler(fileHandler)

logger.setLevel(logging.INFO)

debug = logger.debug
info = logger.info
warning = logger.warning
error = logger.error
logger.propagate = False



logger_all = logging.getLogger('PyPlasmaOptAll')
handler = logging.StreamHandler()
formatter = logging.Formatter(fmt=f"%(levelname)s [" + f"{comm.rank}".zfill(digits) + f"] %(message)s")
handler.setFormatter(formatter)
logger_all.addHandler(handler)
logger_all.setLevel(logging.INFO)

debug_all = logger_all.debug
info_all = logger_all.info
warning_all = logger_all.warning
error_all = logger_all.error
logger_all.propagate = False

def info_all_sync(*args):
    for i in range(comm.size):
        comm.barrier()
        if i == comm.rank:
            info_all(*args)
