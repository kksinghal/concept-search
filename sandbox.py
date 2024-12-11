from dsl import *
import traceback
import numpy as np
import multiprocessing
import signal

np.random.seed(10)

TIMEOUT = 2

class TimeoutError(Exception):
  """Custom exception to indicate a function timeout."""
  pass

"""
Return: 
    Bool, whether code is syntactically for input
    Tuple[Tuple], output of function, if syntactically correct
"""
def run(function, inp):
    try:
        def handler(signum, frame):
            print(f"Function execution exceeded {TIMEOUT} seconds.")
            raise TimeoutError(f"Function execution exceeded {TIMEOUT} seconds.")
        
        # Set the signal handler and a timer
        signal.signal(signal.SIGALRM, handler)
        signal.alarm(TIMEOUT)

        try:
            out = function(inp)
        except TimeoutError as e:
            print(e)
            return False, e
        finally:
            # Cancel the alarm in case the function completes before the timeout
            signal.alarm(0)

        # out = function(inp)
        
        # check for homogeneity of shape
        if np.array(out).size == 0: # throws error on inhomogeneous shape
            return False, "Zero sized output grid"
        elif len(np.array(out).shape) > 2 or len(np.array(out).shape) < 2: 
            return False, "Invalid size of output grid"
        
        return True, out
    except Exception as e:
        return False, traceback.format_exc()
    

def target_function_wrapper(func, queue, inp):
    """Wrapper function to put the return value in the queue."""
    try:
        result = func(inp)
        queue.put(result)
    except Exception as e:
        # Handle potential errors within the target function
        queue.put(e) 