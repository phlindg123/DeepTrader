import numpy as np


class Env:
    #ANVÄNDER JAG EJ TROR JAG.

    def step(self, action):
        """
        Returnerar en tuple (obs, reward, done, info)
        """
        raise NotImplementedError()
    def reset(self):
        """
        Resettar allt. returnerar initial obserivation
        """
        raise NotImplementedError()

    