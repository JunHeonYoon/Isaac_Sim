import numpy as np
from omni.isaac.core import World
from omni.isaac.sensor import ContactSensor

class ContactSensors:
    def __init__(self, world:World, prim_path:list, translations:np.array, sensor_radius:float) -> None:
        self.world = world
        self.prim_path = prim_path
        self.translations = translations
        self.radius = sensor_radius
        self.setSensor()
        
    
    def setSensor(self) ->None:
        self.sensors = []
        for i in range(0, self.translations.shape[0]):
            self.sensors.append(self.world.scene.add(ContactSensor(prim_path=self.prim_path[i],
                                                                   name="contact_sensor_{}".format(i),
                                                                   min_threshold=0,
                                                                   max_threshold=10000000,
                                                                   radius=self.radius,
                                                                   translation=self.translations[i])))
    
        
    def getForce(self)->np.array:
        force = np.zeros(len(self.sensors))
        for i in range(0,len(self.sensors)):
            if self.sensors[i].get_current_frame()['in_contact']:
                force[i] = self.sensors[i].get_current_frame()['force']
        return force