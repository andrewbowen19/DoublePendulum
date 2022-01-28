# doublePendulum/randomPendulumGenerator.py

from doublePendulum.doublePendulum import DoublePendulum
import matplotlib.pyplot as plt

if __name__ == "__main__":
    for i in range(0,5): 
        d = DoublePendulum(random=True)
        plt.close()
        
        print('Initial Values:')
        print(f"M1: {d.m_1}")
        print(f"M2: {d.m_2}")
        print(f"L1: {d.l_1}")
        print(f"L2: {d.l_2}")
        print('------------------------------------------------------')



