from omni.isaac.kit import SimulationApp

simulation_app = SimulationApp({"headless": False})

from omni.isaac.core import World
from omni.isaac.core.utils.stage import add_reference_to_stage
import time
import numpy as np
from armcontroller.controller import ArmController
from omni.isaac.core.articulations.articulation_view import ArticulationView
import sys
import select
import tty
import termios
import os
import time



def isData():
    return select.select([sys.stdin], [], [], 0) == ([sys.stdin], [], [])

old_settings = termios.tcgetattr(sys.stdin)


hz = 100
pkg_path = os.path.dirname(os.path.abspath(__file__))
world = World(stage_units_in_meters=1.0)
world.scene.add_default_ground_plane()
add_reference_to_stage(usd_path=pkg_path+"/model/panda_arm_hand_wo_gripper.usd", prim_path="/World/panda")
world.set_simulation_dt(1/hz, 1/hz) 
ac = ArmController(hz, world, pkg_path)
world.reset()


is_simulation_run = True
exit_flag = False
is_first = True
try:
    tty.setcbreak(sys.stdin.fileno())
    while (simulation_app.is_running() and (not exit_flag)):
        world.step(render=True)
        ac.UpdateData()
        if(is_first):
            is_first = False
            world.reset()
            ac.UpdateData()
            print("Initial q : " )
            print(ac.getPosition()) 
            ac.initPosition()

        if isData():
            key = sys.stdin.read(1)
            if key == 'i':
                ac.setMode("joint_ctrl_init")

            elif key == 'p':
                if is_simulation_run:
                    world.pause()  
                    print("Simulation Paused")
                    is_simulation_run = False
                else:
                    world.play()
                    print("Simulation Play")
                    is_simulation_run = True
            elif key == 'm':
                ac.setMode("MPC_tracking")    
        
            elif key == '\x1b': # x1b is ESC
                is_simulation_run = False
                exit_flag = True  
  
        if is_simulation_run:
            # tic=time.time()
            ac.compute()
            ac.write()
            # toc=time.time()
            # print(toc-tic)
finally:
    termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)
    ac.closeFile()
    simulation_app.close()