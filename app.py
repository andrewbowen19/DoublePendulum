# from ctypes.wintypes import RGB
from datetime import date, datetime 
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pprint import pprint
import copy
import matplotlib.animation as animation
from numpy.lib.function_base import average


class DoublePendulum(object):

    def __init__(self, savefig=False):
        self.g = 9.81
        self.milliSecondDelayBetweenFrames = 20
        self.stepsPerFrame = 32
        self.deltaT = float(self.milliSecondDelayBetweenFrames) * 0.001 / float(self.stepsPerFrame)

        # Setting pendulum arm lens and masses
        self.m_1 = 0.1
        self.l_1 = 0.3

        self.m_2 = 0.10000000000000000000001
        self.l_2 = 0.3

        # at t = 0
        self.theta_1_0 = 2.1
        self.theta_2_0 = 2
        self.thetaDot_1_0 = 2
        self.thetaDot_2_0 = 2

        self.u_vector = [self.theta_1_0, self.theta_2_0, self.thetaDot_1_0, self.thetaDot_2_0]
        self.u_vectorTimeSnapshots = []

        self.radiusOfGraphAxises = 0.5
        self.backgroundColorRGB = (1, 1, 1)
        self.lineColor = (0, 0, 0)


        self.f = plt.figure()
        self.ax = plt.axes(projection='3d')
        self.ax.set_xlim3d(-self.radiusOfGraphAxises, self.radiusOfGraphAxises)
        self.ax.set_ylim3d(-self.radiusOfGraphAxises, self.radiusOfGraphAxises)
        self.ax.set_zlim3d(-self.radiusOfGraphAxises + self.radiusOfGraphAxises*0.23, self.radiusOfGraphAxises - self.radiusOfGraphAxises*0.23)
        self.ax.view_init(40, -5)
        # self.ax.set_title('Double Pendulum')
        self.ax.set_facecolor(self.backgroundColorRGB)
        self.f.patch.set_facecolor(self.backgroundColorRGB)
        plt.axis('off')

        self.line0, = self.ax.plot(0, 0, 0, lw=0.5, color=((0, 0, 1)))
        self.line1, = self.ax.plot(0, 0, 0, lw=0.5, color=((1, 0, 0)))
        self.bar0, = self.ax.plot(0, 0, 0, lw=3, color=(self.lineColor))
        self.bar1, = self.ax.plot(0, 0, 0, lw=3, color=(self.lineColor))

        # Create animation
        ani = animation.FuncAnimation(self.f, func=self.animation_frame, frames=np.arange(0, 750, 1), interval=self.milliSecondDelayBetweenFrames)
        plt.show()

        # print('Animation shown!')
        if savefig:
            ani.save('doublePendulum.gif')

    # @staticmethod
    def findThetaDoubleDot_1(self, theta_1, theta_2, thetaDot_1, thetaDot_2):
        return (-self.m_2*self.l_1*thetaDot_1**2*np.sin(theta_1 - theta_2)*np.cos(theta_1 - theta_2) + self.m_2*self.g*np.sin(theta_2)*np.cos(theta_1 - theta_2) - self.m_2*self.l_2*thetaDot_2**2*np.sin(theta_1 - theta_2) - (self.m_1 + self.m_2)*self.g*np.sin(theta_1)) / ((self.m_1 + self.m_2)*self.l_1 - self.m_2*self.l_1*np.cos(theta_1 - theta_2)**2)

    # @staticmethod
    def findThetaDoubleDot_2(self, theta_1, theta_2, thetaDot_1, thetaDot_2):
        return (self.m_2*self.l_2*thetaDot_2**2*np.sin(theta_1 - theta_2)*np.cos(theta_1 - theta_2) + (self.m_1 + self.m_2)*self.g*np.sin(theta_1)*np.cos(theta_1 - theta_2) + self.l_1*thetaDot_1**2*np.sin(theta_1-theta_2)*(self.m_1 + self.m_2) - self.g*np.sin(theta_2)*(self.m_1 + self.m_2)) / (self.l_1*(self.m_1 + self.m_2) - self.m_2*self.l_2*np.cos(theta_1 - theta_2)**2)


    def symplecticEulerOneStep(self):
    # def findThetaDoubleDot_1(theta_1, theta_2, thetaDot_1, thetaDot_2):
    #     return (-m_2*l_1*thetaDot_1**2*np.sin(theta_1 - theta_2)*np.cos(theta_1 - theta_2) + m_2*g*np.sin(theta_2)*np.cos(theta_1 - theta_2) - m_2*l_2*thetaDot_2**2*np.sin(theta_1 - theta_2) - (m_1 + m_2)*g*np.sin(theta_1)) / ((m_1 + m_2)*l_1 - m_2*l_1*np.cos(theta_1 - theta_2)**2)

    # def findThetaDoubleDot_2(theta_1, theta_2, thetaDot_1, thetaDot_2):
    #     return (m_2*l_2*thetaDot_2**2*np.sin(theta_1 - theta_2)*np.cos(theta_1 - theta_2) + (m_1 + m_2)*g*np.sin(theta_1)*np.cos(theta_1 - theta_2) + l_1*thetaDot_1**2*np.sin(theta_1-theta_2)*(m_1 + m_2) - g*np.sin(theta_2)*(m_1 + m_2)) / (l_1*(m_1 + m_2) - m_2*l_2*np.cos(theta_1 - theta_2)**2)

        self.u_vector[2] += self.findThetaDoubleDot_1(self.u_vector[0], self.u_vector[1], self.u_vector[2], self.u_vector[3]) * self.deltaT
        self.u_vector[3] += self.findThetaDoubleDot_2(self.u_vector[0], self.u_vector[1], self.u_vector[2], self.u_vector[3]) * self.deltaT

        self.u_vector[0] += self.u_vector[2] * self.deltaT
        self.u_vector[1] += self.u_vector[3] * self.deltaT

        self.u_vectorTimeSnapshots.append(copy.deepcopy(self.u_vector))



    def getAxisCoordinatesOverTimeForParticle(self, _particle, _axis):
        line = []
        if _axis == 2:
            for i in range(len(self.u_vectorTimeSnapshots)):
                line.append(0)
            return line
        if _axis == 0 and _particle == 0:
            for i in range(len(self.u_vectorTimeSnapshots)):
                line.append(self.l_1*np.sin(self.u_vectorTimeSnapshots[i][0]))
            return line
        if _axis == 1 and _particle == 0:
            for i in range(len(self.u_vectorTimeSnapshots)):
                line.append(-self.l_1*np.cos(self.u_vectorTimeSnapshots[i][0]))
            return line
        if _axis == 0 and _particle == 1:
            for i in range(len(self.u_vectorTimeSnapshots)):
                line.append(self.l_1*np.sin(self.u_vectorTimeSnapshots[i][0]) + self.l_1*np.sin(self.u_vectorTimeSnapshots[i][1]))
            return line
        if _axis == 1 and _particle == 1:
            for i in range(len(self.u_vectorTimeSnapshots)):
                line.append(-self.l_1*np.cos(self.u_vectorTimeSnapshots[i][0]) - self.l_1*np.cos(self.u_vectorTimeSnapshots[i][1]))
            return line

    def create_figure(self):
        fig = plt.figure()
        self.ax = plt.axes(projection='3d')
        self.ax.set_xlim3d(-self.radiusOfGraphAxises, self.radiusOfGraphAxises)
        self.ax.set_ylim3d(-self.radiusOfGraphAxises, self.radiusOfGraphAxises)
        self.ax.set_zlim3d(-self.radiusOfGraphAxises + self.radiusOfGraphAxises*0.23, self.radiusOfGraphAxises - self.radiusOfGraphAxises*0.23)
        self.ax.view_init(40, -5)
        self.ax.set_title('Double Pendulum')
        self.ax.set_facecolor(self.backgroundColorRGB)
        fig.patch.set_facecolor(self.backgroundColorRGB)
        plt.axis('off')

        self.line0, = self.ax.plot(0, 0, 0, lw=0.5, color=((0, 0, 1)))
        self.line1, = self.ax.plot(0, 0, 0, lw=0.5, color=((1, 0, 0)))
        self.bar0, = self.ax.plot(0, 0, 0, lw=3, color=(self.lineColor))
        self.bar1, = self.ax.plot(0, 0, 0, lw=3, color=(self.lineColor))

        return fig, self.ax

    def animation_frame(self, scatter_points=False, savefig=False):  
        '''Creates animation frame for gif'''
        # scats = []
        starttime = datetime.now()
        # f, self.ax = self.create_figure()
        self.ax.view_init(0, -90)

        for x in range(self.stepsPerFrame):
            self.symplecticEulerOneStep()

        #averageFps = round(i / ((datetime.now() - starttime).total_seconds()))
        #self.ax.set_title(f'average fps: {averageFps}')
        self.ax.set_title('Double Pendulum')

        # self.line0.set_data(self.getAxisCoordinatesOverTimeForParticle(0, 0), self.getAxisCoordinatesOverTimeForParticle(0, 2))
        # self.line0.set_3d_properties(self.getAxisCoordinatesOverTimeForParticle(0, 1))

        self.line1.set_data(self.getAxisCoordinatesOverTimeForParticle(1, 0), self.getAxisCoordinatesOverTimeForParticle(1, 2))
        self.line1.set_3d_properties(self.getAxisCoordinatesOverTimeForParticle(1, 1))

        # self.bar0.set_data([0, self.l_1*np.sin(self.u_vector[0])], [0, 0])
        # self.bar0.set_3d_properties([0, -self.l_1*np.cos(self.u_vector[0])])

        # self.bar1.set_data([self.l_1*np.sin(self.u_vector[0]), self.l_1*np.sin(self.u_vector[0]) + self.l_1*np.sin(self.u_vector[1])], [0, 0])
        # self.bar1.set_3d_properties([-self.l_1*np.cos(self.u_vector[0]), -self.l_1*np.cos(self.u_vector[0]) - self.l_1*np.cos(self.u_vector[1])])

        # if scatter_points:
        global scats
        scats = []
        # first remove all old scatters
        for scat in scats:
            scat.remove()
        scats = []

        # if scatter pointf at each frame desired
        # scats.append((self.ax.scatter(0, 0, 0, color=(0, 0, 0), s=3)))
        # scats.append((self.ax.scatter(self.l_1*np.sin(self.u_vector[0]), 0, -self.l_1*np.cos(self.u_vector[0]), color=(0, 0, 1), s=10)))
        # scats.append((self.ax.scatter(self.l_1*np.sin(self.u_vector[0]) + self.l_1*np.sin(self.u_vector[1]), 0, -self.l_1*np.cos(self.u_vector[0]) - self.l_1*np.cos(self.u_vector[1]), color=(1, 0, 0), s=10)))

        print('Creating animation....')


    

    # return ani
         



if __name__ == "__main__":
    d = DoublePendulum()
    # a = d.animation_frame(True)   
    # plt.show()

#ani.save('doublePendulum.gif')