from ctypes import cdll
import ctypes
import numpy as np
from numpy.ctypeslib import ndpointer
import configparser

config = configparser.ConfigParser()
config.read('config.ini')

#works only on x86_64 system where pointer are stored into a int64
#need either numpy 1.16 or numpy 1.14 : 1.15 broke ctypes

lib = cdll.LoadLibrary(config.get('simulation', 'library'))
if "libddrl-nfac" in config.get('simulation', 'library') :
    lib.OfflineCaclaAg_new.restype = ctypes.c_int64
    lib.OfflineCaclaAg_start_episode.argtypes = [ ctypes.c_int64, ndpointer(ctypes.c_double), ctypes.c_bool]
    lib.OfflineCaclaAg_run.argtypes = [ ctypes.c_int64, ctypes.c_double, ndpointer(ctypes.c_double), ctypes.c_bool, ctypes.c_bool, ctypes.c_bool]
    lib.OfflineCaclaAg_dump.argtypes = [ ctypes.c_int64, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_double]
    lib.OfflineCaclaAg_display.argtypes = [ ctypes.c_int64, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_double]
    lib.OfflineCaclaAg_end_episode.argtypes = [ ctypes.c_int64, ctypes.c_bool ]
    
    class NFACAg(object):
        def __init__(self, nb_motors, nb_sensors, argv=[]):
            self.obj = lib.OfflineCaclaAg_new(nb_motors, nb_sensors)
            argv.append("")
            string_length = len(argv)

            select_type = (ctypes.c_char_p * string_length)
            select = select_type()
            for key, item in enumerate(argv):
                select[key] = item.encode('utf-8')
            
            lib.OfflineCaclaAg_unique_invoke.argtypes = [ctypes.c_int64, ctypes.c_int, select_type]
            lib.OfflineCaclaAg_unique_invoke(self.obj, len(argv), select)
            lib.OfflineCaclaAg_run.restype = ndpointer(ctypes.c_double, shape=(nb_motors,))
    
        def run(self, reward, state, learning, goal, last):
            return lib.OfflineCaclaAg_run(self.obj, reward, np.ascontiguousarray(state, np.float64), learning, goal , last)
    
        def start_ep(self, state, learning):
            lib.OfflineCaclaAg_start_episode(self.obj, np.ascontiguousarray(state, np.float64), learning)
    
        def end_ep(self, learning):
            lib.OfflineCaclaAg_end_episode(self.obj, learning)
    
        def dump(self, learning, episode, step, treward):
            lib.OfflineCaclaAg_dump(self.obj, learning, episode, step, treward)
            
        def display(self, learning, episode, step, treward):
            lib.OfflineCaclaAg_display(self.obj, learning, episode, step, treward)
            
        def save(self, episode):
            lib.OfflineCaclaAg_save(self.obj, episode)
    
        def load(self, episode):
            lib.OfflineCaclaAg_load(self.obj, episode)