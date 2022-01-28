# from ctypes.wintypes import RGB
from datetime import date, datetime 
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pprint import pprint
import copy
import matplotlib.animation as animation
from numpy.lib.function_base import average
import numpy as np
import hashlib
import os


class DoublePendulum(object):

    def __init__(self, random=False, savefig=False):
        """
        Class to animate the path of a double-pendulum system via matplotlib animate

        parameters:
            random : bool, default False; if True,
                        initial values for the pendula's 
                        mass and length are chosen randomly (via np.random.random())
            savefig : bool, default False; if True, save the output to a gif file (in the output directory)
        """

        self.g = 9.81
        self.frame_delay = 75 # Delay between frames (in ms)
        self.steps_per_frame = 16
        self.deltaT = float(self.frame_delay) * 0.001 / float(self.steps_per_frame)
        self.nframes = 100
        self.savefig = savefig

        # Setting pendulum arm lens and masses
        self.m_1 = np.random.random() if random else 0.3
        self.l_1 = np.random.random() if random else 0.3

        self.m_2 = np.random.random() if random else 0.1
        self.l_2 = np.random.random() if random else 0.3

        print('Initial masses (M) & arm lengths (L)')
        print(f"M1 = {self.m_1} | M2 = {self.m_2}")
        print(f"L1 = {self.l_1} | L2 = {self.l_2}")

        # at t = 0
        self.theta_1_0 = 2
        self.theta_2_0 = 2
        self.thetaDot_1_0 = 2
        self.thetaDot_2_0 = 2

        self.u_vector = [self.theta_1_0, self.theta_2_0, self.thetaDot_1_0, self.thetaDot_2_0]
        self.u_vectorTimeSnapshots = []

        self.graphAxisRadius = 1
        self.backgroundColorRGB = (0, 0, 0)
        self.line_color = (1, 1, 1)

        # Setting up figure for later plotting
        self.f = plt.figure()
        self.ax = plt.axes(projection='3d')
        self.ax.set_xlim3d(-self.graphAxisRadius, self.graphAxisRadius)
        self.ax.set_ylim3d(-self.graphAxisRadius, self.graphAxisRadius)
        self.ax.set_zlim3d(-self.graphAxisRadius + self.graphAxisRadius*0.23, self.graphAxisRadius - self.graphAxisRadius*0.23)
        self.ax.view_init(elev=90)

        self.ax.set_facecolor(self.backgroundColorRGB)
        self.f.patch.set_facecolor(self.backgroundColorRGB)
        plt.axis('off')

        # Drawing path of second (bottom) pendulum
        self.line1, = self.ax.plot(0, 0, 0, lw=0.5, color=self.line_color)

        # Create animation: https://matplotlib.org/stable/api/_as_gen/matplotlib.animation.FuncAnimation.html
        ani = animation.FuncAnimation(self.f, func=self.animation_frame,# init_func=self.init,
                                      frames=self.nframes,
                                      interval=self.frame_delay, save_count=200, repeat=False)#, blit=True)
        plt.show()

        # print('Animation shown!')
        if self.savefig:
            initial_values = bytes(str({'m1': self.m_1, 'm2': self.m_2, 'l1': self.l_1, 'l2': self.l_2}), encoding='utf-8')
            file_id = hashlib.sha256(initial_values).hexdigest()
            file_path = os.path.join('..', 'output', f'doublePendulum_{file_id}.gif')
            print('Saving fig with imagemagick!')
            ani.save(file_path, writer='imagemagick', fps=30)

        # plt.close()

        

    def theta_double_dot1(self, theta_1, theta_2, thetaDot_1, thetaDot_2):
        """Calculates value of second derivative of penduum 1\'s angular position"""
        return (-self.m_2*self.l_1*thetaDot_1**2*np.sin(theta_1 - theta_2)*np.cos(theta_1 - theta_2) + self.m_2*self.g*np.sin(theta_2)*np.cos(theta_1 - theta_2) - self.m_2*self.l_2*thetaDot_2**2*np.sin(theta_1 - theta_2) - (self.m_1 + self.m_2)*self.g*np.sin(theta_1)) / ((self.m_1 + self.m_2)*self.l_1 - self.m_2*self.l_1*np.cos(theta_1 - theta_2)**2)

    def theta_double_dot2(self, theta_1, theta_2, thetaDot_1, thetaDot_2):
        """Calculates value of second derivative of penduum 2\'s angular position"""
        return (self.m_2*self.l_2*thetaDot_2**2*np.sin(theta_1 - theta_2)*np.cos(theta_1 - theta_2) + (self.m_1 + self.m_2)*self.g*np.sin(theta_1)*np.cos(theta_1 - theta_2) + self.l_1*thetaDot_1**2*np.sin(theta_1-theta_2)*(self.m_1 + self.m_2) - self.g*np.sin(theta_2)*(self.m_1 + self.m_2)) / (self.l_1*(self.m_1 + self.m_2) - self.m_2*self.l_2*np.cos(theta_1 - theta_2)**2)

    def symplecticEulerOneStep(self):
    
        self.u_vector[2] += self.theta_double_dot1(self.u_vector[0], self.u_vector[1], self.u_vector[2], self.u_vector[3]) * self.deltaT
        self.u_vector[3] += self.theta_double_dot2(self.u_vector[0], self.u_vector[1], self.u_vector[2], self.u_vector[3]) * self.deltaT

        self.u_vector[0] += self.u_vector[2] * self.deltaT
        self.u_vector[1] += self.u_vector[3] * self.deltaT

        self.u_vectorTimeSnapshots.append(copy.deepcopy(self.u_vector))

    def getAxisCoordinatesOverTimeForParticle(self, _particle, _axis):
        line = []
        if _axis == 2:
            for i in range(len(self.u_vectorTimeSnapshots)):
                line.append(0)
            # return line
        if _axis == 0 and _particle == 0:
            for i in range(len(self.u_vectorTimeSnapshots)):
                line.append(self.l_1*np.sin(self.u_vectorTimeSnapshots[i][0]))
            # return line
        if _axis == 1 and _particle == 0:
            for i in range(len(self.u_vectorTimeSnapshots)):
                line.append(-self.l_1*np.cos(self.u_vectorTimeSnapshots[i][0]))
            # return line
        if _axis == 0 and _particle == 1:
            for i in range(len(self.u_vectorTimeSnapshots)):
                line.append(self.l_1*np.sin(self.u_vectorTimeSnapshots[i][0]) + self.l_1*np.sin(self.u_vectorTimeSnapshots[i][1]))
            # return line
        if _axis == 1 and _particle == 1:
            for i in range(len(self.u_vectorTimeSnapshots)):
                line.append(-self.l_1*np.cos(self.u_vectorTimeSnapshots[i][0]) - self.l_1*np.cos(self.u_vectorTimeSnapshots[i][1]))
        return line

    # def init(self):
    #     self.line1.set_data([], [])
    #     return self.line1

    def animation_frame(self, i):  
        '''Creates animation frame for gif
        
        parameters:
            i : integer: ranges from 0 to self.nframes
        '''

        # starttime = datetime.now()
        self.ax.view_init(0, -90)

        for x in range(self.steps_per_frame):
            
            self.symplecticEulerOneStep()

        # self.line0.set_data(self.sCoordinatesOverTimeForParticle(0, 0), self.getAxisCoordinatesOverTimeForParticle(0, 2))
        # self.line0.set_3d_properties(selgetAxif.getAxisCoordinatesOverTimeForParticle(0, 1))

            self.line1.set_data(self.getAxisCoordinatesOverTimeForParticle(1, 0), self.getAxisCoordinatesOverTimeForParticle(1, 2))
            self.line1.set_3d_properties(self.getAxisCoordinatesOverTimeForParticle(1, 1))

        # return self.line1





if __name__ == "__main__":
    d = DoublePendulum(random=True, savefig=True)