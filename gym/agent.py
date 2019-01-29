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
if "libddrl-nfac" in config.get('simulation', 'library') or "libddrl-penfac" in config.get('simulation', 'library'):
    lib.OfflineCaclaAg_new.restype = ctypes.c_int64
    lib.OfflineCaclaAg_start_episode.argtypes = [ ctypes.c_int64, ndpointer(ctypes.c_double), ctypes.c_bool]
    lib.OfflineCaclaAg_run.argtypes = [ ctypes.c_int64, ctypes.c_double, ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"), ctypes.c_bool, ctypes.c_bool, ctypes.c_bool]
    lib.OfflineCaclaAg_dump.argtypes = [ ctypes.c_int64, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_double]
    lib.OfflineCaclaAg_display.argtypes = [ ctypes.c_int64, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_double]
    lib.OfflineCaclaAg_end_episode.argtypes = [ ctypes.c_int64, ctypes.c_bool ]
    
    class DDRLAg(object):
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
        
        def name(self):
            return "NFAC(lambda)-V" if "libddrl-nfac" in config.get('simulation', 'library') else "PeNFAC(lambda)-V"

elif "libddrl-cacla" in config.get('simulation', 'library') :
    lib.BaseCaclaAg_new.restype = ctypes.c_int64
    lib.BaseCaclaAg_start_episode.argtypes = [ ctypes.c_int64, ndpointer(ctypes.c_double), ctypes.c_bool]
    lib.BaseCaclaAg_run.argtypes = [ ctypes.c_int64, ctypes.c_double, ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"), ctypes.c_bool, ctypes.c_bool, ctypes.c_bool]
    lib.BaseCaclaAg_dump.argtypes = [ ctypes.c_int64, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_double]
    lib.BaseCaclaAg_display.argtypes = [ ctypes.c_int64, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_double]
    lib.BaseCaclaAg_end_episode.argtypes = [ ctypes.c_int64, ctypes.c_bool ]
    
    class DDRLAg(object):
        def __init__(self, nb_motors, nb_sensors, argv=[]):
            self.obj = lib.BaseCaclaAg_new(nb_motors, nb_sensors)
            argv.append("")
            string_length = len(argv)

            select_type = (ctypes.c_char_p * string_length)
            select = select_type()
            for key, item in enumerate(argv):
                select[key] = item.encode('utf-8')
            
            lib.BaseCaclaAg_unique_invoke.argtypes = [ctypes.c_int64, ctypes.c_int, select_type]
            lib.BaseCaclaAg_unique_invoke(self.obj, len(argv), select)
            lib.BaseCaclaAg_run.restype = ndpointer(ctypes.c_double, shape=(nb_motors,))
    
        def run(self, reward, state, learning, goal, last):
            return lib.BaseCaclaAg_run(self.obj, reward, np.ascontiguousarray(state, np.float64), learning, goal , last)
    
        def start_ep(self, state, learning):
            lib.BaseCaclaAg_start_episode(self.obj, np.ascontiguousarray(state, np.float64), learning)
    
        def end_ep(self, learning):
            lib.BaseCaclaAg_end_episode(self.obj, learning)
    
        def dump(self, learning, episode, step, treward):
            lib.BaseCaclaAg_dump(self.obj, learning, episode, step, treward)
            
        def display(self, learning, episode, step, treward):
            lib.BaseCaclaAg_display(self.obj, learning, episode, step, treward)
            
        def save(self, episode):
            lib.BaseCaclaAg_save(self.obj, episode)
    
        def load(self, episode):
            lib.BaseCaclaAg_load(self.obj, episode)
            
        def name(self):
            return "CACLA"
elif "libddrl-ddpg" in config.get('simulation', 'library') :
    lib.DDPGAg_new.restype = ctypes.c_int64
    lib.DDPGAg_start_episode.argtypes = [ ctypes.c_int64, ndpointer(ctypes.c_double), ctypes.c_bool]
    lib.DDPGAg_run.argtypes = [ ctypes.c_int64, ctypes.c_double, ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"), ctypes.c_bool, ctypes.c_bool, ctypes.c_bool]
    lib.DDPGAg_dump.argtypes = [ ctypes.c_int64, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_double]
    lib.DDPGAg_display.argtypes = [ ctypes.c_int64, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_double]
    lib.DDPGAg_end_episode.argtypes = [ ctypes.c_int64, ctypes.c_bool ]
    
    class DDRLAg(object):
        def __init__(self, nb_motors, nb_sensors, argv=[]):
            self.obj = lib.DDPGAg_new(nb_motors, nb_sensors)
            argv.append("")
            string_length = len(argv)

            select_type = (ctypes.c_char_p * string_length)
            select = select_type()
            for key, item in enumerate(argv):
                select[key] = item.encode('utf-8')
            
            lib.DDPGAg_unique_invoke.argtypes = [ctypes.c_int64, ctypes.c_int, select_type]
            lib.DDPGAg_unique_invoke(self.obj, len(argv), select)
            lib.DDPGAg_run.restype = ndpointer(ctypes.c_double, shape=(nb_motors,))
    
        def run(self, reward, state, learning, goal, last):
            return lib.DDPGAg_run(self.obj, reward, np.ascontiguousarray(state, np.float64), learning, goal , last)
    
        def start_ep(self, state, learning):
            lib.DDPGAg_start_episode(self.obj, np.ascontiguousarray(state, np.float64), learning)
    
        def end_ep(self, learning):
            lib.DDPGAg_end_episode(self.obj, learning)
    
        def dump(self, learning, episode, step, treward):
            lib.DDPGAg_dump(self.obj, learning, episode, step, treward)
            
        def display(self, learning, episode, step, treward):
            lib.DDPGAg_display(self.obj, learning, episode, step, treward)
            
        def save(self, episode):
            lib.DDPGAg_save(self.obj, episode)
    
        def load(self, episode):
            lib.DDPGAg_load(self.obj, episode)
            
        def name(self):
            return "DDPG"
