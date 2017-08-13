
from ctypes import cdll
import ctypes
import numpy as np
from numpy.ctypeslib import ndpointer
import ConfigParser
config = ConfigParser.ConfigParser()
config.readfp(open('config.ini'))

lib = cdll.LoadLibrary(config.get('simulation', 'library'))
lib.OffNFACAg_start_episode.argtypes = [ ctypes.c_int, ndpointer(ctypes.c_double), ctypes.c_bool]
lib.OffNFACAg_run.argtypes = [ ctypes.c_int, ctypes.c_double, ndpointer(ctypes.c_double), ctypes.c_bool, ctypes.c_bool, ctypes.c_bool]
lib.OffNFACAg_dump.argtypes = [ ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_double]
lib.OffNFACAg_display.argtypes = [ ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_double]
lib.CaclaAg_start_episode.argtypes = [ ctypes.c_int, ndpointer(ctypes.c_double), ctypes.c_bool]
lib.CaclaAg_run.argtypes = [ ctypes.c_int, ctypes.c_double, ndpointer(ctypes.c_double), ctypes.c_bool, ctypes.c_bool, ctypes.c_bool]
lib.CaclaAg_dump.argtypes = [ ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_double]
lib.CaclaAg_display.argtypes = [ ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_double]

class OffNFACAg(object):
    def __init__(self, nb_motors, nb_sensors):
        self.obj = lib.OffNFACAg_new(nb_motors, nb_sensors)
        lib.OffNFACAg_unique_invoke(self.obj, 0, "")
        lib.OffNFACAg_run.restype = ndpointer(ctypes.c_double, shape=(nb_motors,))

    def run(self, reward, state, learning, goal, last):
        return lib.OffNFACAg_run(self.obj, reward, np.ascontiguousarray(state, np.float64), learning, goal , last)

    def start_ep(self, state, learning):
        lib.OffNFACAg_start_episode(self.obj, np.ascontiguousarray(state, np.float64), learning)

    def end_ep(self, learning):
        lib.OffNFACAg_end_episode(self.obj, learning)

    def dumpdisplay(self, learning, episode, step, treward):
        lib.OffNFACAg_dump(self.obj, learning, episode, step, treward)
        lib.OffNFACAg_display(self.obj, learning, episode, step, treward)
        
    def save(self, episode):
        lib.OffNFACAg_save(self.obj, episode)

    def load(self, episode):
        lib.OffNFACAg_load(self.obj, episode)

class CaclaAg(object):
    def __init__(self, nb_motors, nb_sensors):
        self.obj = lib.CaclaAg_new(nb_motors, nb_sensors)
        lib.CaclaAg_unique_invoke(self.obj, 0, "")
        lib.CaclaAg_run.restype = ndpointer(ctypes.c_double, shape=(nb_motors,))

    def run(self, reward, state, learning, goal, last):
        return lib.CaclaAg_run(self.obj, reward, np.ascontiguousarray(state, np.float64), learning, goal , last)

    def start_ep(self, state, learning):
        lib.CaclaAg_start_episode(self.obj, np.ascontiguousarray(state, np.float64), learning)

    def end_ep(self, learning):
        lib.CaclaAg_end_episode(self.obj, learning)

    def dumpdisplay(self, learning, episode, step, treward):
        lib.CaclaAg_dump(self.obj, learning, episode, step, treward)
        lib.CaclaAg_display(self.obj, learning, episode, step, treward)

    def save(self, episode):
        lib.CaclaAg_save(self.obj, episode)

    def load(self, episode):
        lib.CaclaAg_load(self.obj, episode)
