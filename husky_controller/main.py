from omni.isaac.kit import SimulationApp

simulation_app = SimulationApp({"headless": False})

from omni.isaac.core import World
from omni.isaac.core.utils.stage import add_reference_to_stage
import numpy as np
from wheelcontroller.controller import WheelController

np.set_printoptions(precision=3, suppress=True)
import time
import sys
import select
import tty
import termios
import os

def isData():
    return select.select([sys.stdin], [], [], 0) == ([sys.stdin], [], [])

old_settings = termios.tcgetattr(sys.stdin)

hz = 50
pkg_path = os.path.dirname(os.path.abspath(__file__))
world = World(stage_units_in_meters=1.0)
world.scene.add_default_ground_plane()
add_reference_to_stage(usd_path=pkg_path+"/model/husky.usd", prim_path="/World/husky")
world.set_simulation_dt(1/hz, 1/hz)
wc = WheelController(hz=hz, world=world, pkg_path=pkg_path, is_two_wheel=False)
world.reset()


is_simulation_run = True
exit_flag = False
is_first = True
try:
    tty.setcbreak(sys.stdin.fileno())
    while (simulation_app.is_running() and (not exit_flag)):
        world.step(render=True)
        wc.UpdateData()
        if(is_first):
            world.reset()
            wc.UpdateData()
            print("Initial q : " )
            print(wc.getPose()) 
            is_first = False
            wc.initPosition()
        if isData():
            key = sys.stdin.read(1)

            if key == "i":
                wc.setMode("init")
            elif key == "1":
                wc.setMode("Kanayama Circle")
            elif key == "2":
                wc.setMode("Kanayama Square")
            elif key == "3":
                wc.setMode("Kanayama Eight")
            elif key == "4":
                wc.setMode("Velocity Circle")
            elif key == "5":
                wc.setMode("Velocity Square")
            elif key == "6":
                wc.setMode("Velocity Eight")
            elif key == "v":
                wc.setMode("Velocity Command")

            elif key == "p":
                if is_simulation_run:
                    time.sleep(0.1)
                    world.pause()  
                    print("Simulation Paused")
                    is_simulation_run = False
                else:
                    time.sleep(0.1)
                    world.play()
                    print("Simulation Play")
                    is_simulation_run = True
            elif key == "q":
                time.sleep(0.1)
                is_simulation_run = False
                exit_flag = True
            
        if is_simulation_run:
            wc.compute()
            wc.write()
finally:
    wc.closeFile()
    simulation_app.close()
