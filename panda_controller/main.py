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

def isData():
    return select.select([sys.stdin], [], [], 0) == ([sys.stdin], [], [])

old_settings = termios.tcgetattr(sys.stdin)


hz = 100
pkg_path = "/home/yoonjunheon/Isaac_Sim/panda_controller"
world = World(stage_units_in_meters=1.0)
world.scene.add_default_ground_plane()
add_reference_to_stage(usd_path=pkg_path+"/model/panda_arm_hand.usd", prim_path="/World/panda")
world.set_simulation_dt(1/hz, 1/hz) 
ac = ArmController(hz, world, pkg_path)
ft = ArticulationView(prim_paths_expr="/World/panda", name="ft_viewer")
world.scene.add(ft)
world.reset()


is_simulation_run = True
exit_flag = False
is_first = True
try:
    tty.setcbreak(sys.stdin.fileno())
    while (simulation_app.is_running() and (not exit_flag)):
        world.step(render=True)
        ac.UpdateData()
        ac.getFTdata(ft._physics_view.get_force_sensor_forces()[0])
        if(is_first):
            is_first = False
            world.reset()
            ac.UpdateData()
            ac.getFTdata(ft._physics_view.get_force_sensor_forces()[0])
            print("Initial q : " )
            print(ac.getPosition()) 
            ac.initPosition()

        if isData():
            key = sys.stdin.read(1)
            if key == 'i':
                ac.setMode("joint_ctrl_init")
            elif key == '1':
                ac.setMode("CLIK Circle")
            elif key == '2':
                ac.setMode("CLIK Square")
            elif key == '3':
                ac.setMode("CLIK Eight")
            elif key == 'o':
                ac.setMode("gripper_open")
            elif key == 'c':
                ac.setMode("gripper_close")
            elif key == 'g':
                ac.setMode("pick_up")
            elif key == 'n':
                ac.setMode("collision_avoidance")
            elif key == 'p':
                if is_simulation_run:
                    world.pause()  
                    print("Simulation Paused")
                    is_simulation_run = False
                else:
                    world.play()
                    print("Simulation Play")
                    is_simulation_run = True
            elif key == '\x1b': # x1b is ESC
                is_simulation_run = False
                exit_flag = True  
  
        if is_simulation_run:
            ac.compute()
            ac.write()
finally:
    termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)
    ac.closeFile()
    simulation_app.close()