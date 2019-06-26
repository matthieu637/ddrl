from ctypes import cdll
import ctypes
import numpy as np
import platform
from numpy.ctypeslib import ndpointer

if not platform.architecture()[0] == '64bit':
    print("works only on x86_64 system where pointer are stored into a int64")
    exit()

if np.version.version.split(".")[0] == "1" and np.version.version.split(".")[1] == "15":
    print("need either numpy 1.16 or numpy 1.14 because 1.15 broke ctypes")
    exit()

def load_so_libray(config):
    lib_path=config.get('simulation', 'library')
    lib = cdll.LoadLibrary(lib_path)
    if "libddrl-nfac" in lib_path or "libddrl-penfac" in lib_path or "libddrl-psepenfac" in lib_path or "libddrl-dpenfac" in lib_path:
        lib.OfflineCaclaAg_new.restype = ctypes.c_int64
        lib.OfflineCaclaAg_start_episode.argtypes = [ ctypes.c_int64, ndpointer(ctypes.c_double), ctypes.c_bool]
        lib.OfflineCaclaAg_run.argtypes = [ ctypes.c_int64, ctypes.c_double, ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"), ctypes.c_bool, ctypes.c_bool, ctypes.c_bool]
        lib.OfflineCaclaAg_dump.argtypes = [ ctypes.c_int64, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_double]
        lib.OfflineCaclaAg_display.argtypes = [ ctypes.c_int64, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_double]
        lib.OfflineCaclaAg_end_episode.argtypes = [ ctypes.c_int64, ctypes.c_bool ]
        lib.OfflineCaclaAg_save.argtypes = [ ctypes.c_int64 ]
        lib.OfflineCaclaAg_load.argtypes = [ ctypes.c_int64 ]
        
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
                if "libddrl-psepenfac" in lib_path:
                    return "PSEPeNFAC(lambda)-V"
                if "libddrl-dpenfac" in lib_path:
                    return "DPeNFAC(lambda)-V"
                return "NFAC(lambda)-V" if "libddrl-nfac" in lib_path else "PeNFAC(lambda)-V"
    elif "libddrl-hpenfac" in lib_path:
        lib.OfflineCaclaAg_new.restype = ctypes.c_int64
        lib.OfflineCaclaAg_start_episode.argtypes = [ ctypes.c_int64, ndpointer(ctypes.c_double), ctypes.c_bool]
        lib.OfflineCaclaAg_run.argtypes = [ ctypes.c_int64, ctypes.c_double, ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"), ctypes.c_bool, ctypes.c_bool, ctypes.c_bool]
        lib.OfflineCaclaAg_dump.argtypes = [ ctypes.c_int64, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_double]
        lib.OfflineCaclaAg_display.argtypes = [ ctypes.c_int64, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_double]
        lib.OfflineCaclaAg_end_episode.argtypes = [ ctypes.c_int64, ctypes.c_bool ]
        lib.OfflineCaclaAg_save.argtypes = [ ctypes.c_int64 ]
        lib.OfflineCaclaAg_load.argtypes = [ ctypes.c_int64 ]
        
        class DDRLAg(object):
            def __init__(self, nb_motors, nb_sensors, goal_size, goal_start, goal_achieved_start, argv=[]):
                self.obj = lib.OfflineCaclaAg_new(nb_motors, nb_sensors, goal_size, goal_start, goal_achieved_start)
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
                return "HPeNFAC"
    elif "libddrl-cacla" in lib_path :
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
    elif "libddrl-ddpg" in lib_path or "libddrl-td3" in lib_path :
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
                return "DDPG" if "libddrl-ddpg" in lib_path else "TD3"
    elif "libddrl-foo" in lib_path :
        lib.FOOAg_new.restype = ctypes.c_int64
        lib.FOOAg_start_episode.argtypes = [ ctypes.c_int64, ndpointer(ctypes.c_double), ctypes.c_bool]
        lib.FOOAg_run.argtypes = [ ctypes.c_int64, ctypes.c_double, ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"), ctypes.c_bool, ctypes.c_bool, ctypes.c_bool]
        lib.FOOAg_dump.argtypes = [ ctypes.c_int64, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_double]
        lib.FOOAg_display.argtypes = [ ctypes.c_int64, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_double]
        lib.FOOAg_end_episode.argtypes = [ ctypes.c_int64, ctypes.c_bool ]
        
        class DDRLAg(object):
            def __init__(self, nb_motors, nb_sensors, argv=[]):
                self.obj = lib.FOOAg_new(nb_motors, nb_sensors)
                argv.append("")
                string_length = len(argv)
    
                select_type = (ctypes.c_char_p * string_length)
                select = select_type()
                for key, item in enumerate(argv):
                    select[key] = item.encode('utf-8')
                
                lib.FOOAg_unique_invoke.argtypes = [ctypes.c_int64, ctypes.c_int, select_type]
                lib.FOOAg_unique_invoke(self.obj, len(argv), select)
                lib.FOOAg_run.restype = ndpointer(ctypes.c_double, shape=(nb_motors,))
        
            def run(self, reward, state, learning, goal, last):
                return lib.FOOAg_run(self.obj, reward, np.ascontiguousarray(state, np.float64), learning, goal , last)
        
            def start_ep(self, state, learning):
                lib.FOOAg_start_episode(self.obj, np.ascontiguousarray(state, np.float64), learning)
        
            def end_ep(self, learning):
                lib.FOOAg_end_episode(self.obj, learning)
        
            def dump(self, learning, episode, step, treward):
                lib.FOOAg_dump(self.obj, learning, episode, step, treward)
                
            def display(self, learning, episode, step, treward):
                lib.FOOAg_display(self.obj, learning, episode, step, treward)
                
            def save(self, episode):
                lib.FOOAg_save(self.obj, episode)
        
            def load(self, episode):
                lib.FOOAg_load(self.obj, episode)
                
            def name(self):
                return "FOO"
    return DDRLAg
