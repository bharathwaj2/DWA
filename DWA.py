"""
This code implements Dynamic Window Approach (DWA) algorithm with a rectangular mobile robot 
in a world space with customized obstacles(static), the code is based on the research article 
published by Dieter Fox, Wolfram Burgard and Sebastiam Thrun, "The Dynamic Window Approach to
Collision avoidance" in 1997.

Please do execute and see the results for yourself.

"""


import math
from enum import Enum
import random
import matplotlib.pyplot as plt
import numpy as np
import random
import time

random.seed(11)
show_animation = True


class RobotType(Enum):
    circle = 0
    rectangle = 1


class Robot_configurations:
    """
    simulation parameter class
    """

    def __init__(self):
        
        # robot's parameters
        self.max_speed = 1.0  # [m/s]
        self.min_speed = -0.5  # [m/s]
        self.max_yaw_rate = 40.0 * math.pi / 180.0  # [rad/s]
        self.max_accel = 0.2  # [m/ss]
        self.max_delta_yaw_rate = 40.0 * math.pi / 180.0  # [rad/ss]
        self.v_resolution = 0.01  # [m/s]
        self.yaw_rate_resolution = 0.1 * math.pi / 180.0  # [rad/s]
        self.dt = 0.1  # [s] Time tick for motion prediction
        self.predict_time = 3.0  # [s]
        self.to_goal_cost_gain = 0.15
        self.speed_cost_gain = 1.0
        self.obstacle_cost_gain = 1.0
        self.robot_stuck_flag_cons = 0.001  # constant to prevent robot stucked
        self.robot_type = RobotType.circle

        # if robot_type == RobotType.circle
        # Also used to check if goal is reached in both types
        self.robot_radius = 1.0  # [m] for collision check

        # if robot_type == RobotType.rectangle
        self.robot_width = 0.8  # [m] for collision check
        self.robot_length = 2.5  # [m] for collision check
        # obstacles [x(m) y(m), ....]
        self.ob = np.array([[-1, -1],
                            [1.5,2.0],
                            [-1.9,2.4],
                            [8.0,4.0],
                            [4.0, 1.0],
                            [5.0, 5.0],
                            [7.0, 10.0],
                            [13.0, 6.0]
                            ])

    @property
    def robot_type(self):
        return self._robot_type

    @robot_type.setter
    def robot_type(self, value):
        if not isinstance(value, RobotType):
            raise TypeError("robot_type must be an instance of RobotType")
        self._robot_type = value


config = Robot_configurations()

def dwa_control(x, config, goal, ob):
    """
    Dynamic Window Approach control
    """
    dw = calc_dynamic_window(x, config)

    u, trajectory = calc_control_and_trajectory(x, dw, config, goal, ob)

    return u, trajectory


def motion(x, u, dt):
    """
    motion model updation
    """

    x[2] += u[1] * dt
    x[0] += u[0] * math.cos(x[2]) * dt
    x[1] += u[0] * math.sin(x[2]) * dt
    x[3] = u[0]
    x[4] = u[1]

    return x


def calc_dynamic_window(x, config):
    """
    calculating dynamic window based on current state x
    """

    # Dynamic window from robot specification
    Vs = [config.min_speed, config.max_speed,
          -config.max_yaw_rate, config.max_yaw_rate]

    # Dynamic window from motion model
    Vd = [x[3] - config.max_accel * config.dt,
          x[3] + config.max_accel * config.dt,
          x[4] - config.max_delta_yaw_rate * config.dt,
          x[4] + config.max_delta_yaw_rate * config.dt]

    #  [v_min, v_max, yaw_rate_min, yaw_rate_max]
    dw = [max(Vs[0], Vd[0]), min(Vs[1], Vd[1]),
          max(Vs[2], Vd[2]), min(Vs[3], Vd[3])]

    return dw


def predict_trajectory(x_init, v, y, config):
    """
    predict trajectories with an input
    """

    x = np.array(x_init)
    trajectory = np.array(x)
    time = 0
    while time <= config.predict_time:
        x = motion(x, [v, y], config.dt)
        trajectory = np.vstack((trajectory, x))
        time += config.dt

    return trajectory


def calc_control_and_trajectory(x, dw, config, goal, ob):
    """
    calculation final input with dynamic window
    """
    x_init = x[:]
    min_cost = float("inf")
    best_u = [0.0, 0.0]
    best_trajectory = np.array([x])

    # evaluate all trajectory with sampled input in dynamic window
    for v in np.arange(dw[0], dw[1], config.v_resolution):
        for y in np.arange(dw[2], dw[3], config.yaw_rate_resolution):

            # Predict a new trajectory for the bot
            trajectory = predict_trajectory(x_init, v, y, config)
            
            # calculate all costs
            to_goal_cost = config.to_goal_cost_gain * calc_to_goal_cost(trajectory, goal)
            speed_cost = config.speed_cost_gain * (config.max_speed - trajectory[-1,3])
            ob_cost = config.obstacle_cost_gain * calc_obstacle_cost(trajectory, ob, config)
                
            # cost function to detect collision probability 
            final_cost = to_goal_cost + speed_cost + ob_cost       

            # search minimum trajectory
            if min_cost >= final_cost:
                min_cost = final_cost
                best_u = [v, y]
                best_trajectory = trajectory
                if abs(best_u[0]) < config.robot_stuck_flag_cons \
                        and abs(x[3]) < config.robot_stuck_flag_cons:
                    # to ensure the robot do not get stuck in
                    # best v=0 m/s (in front of an obstacle) and
                    # best omega=0 rad/s (heading to the goal with
                    # angle difference of 0)
                    best_u[1] = -config.max_delta_yaw_rate
    return best_u, best_trajectory


def calc_obstacle_cost(trajectory, ob, config):
    """
    calculate obstacle cost : collision = inf
    """
    ox = ob[:,0]
    oy = ob[:,1]
    dx = trajectory[:,0] - ox[:,None]
    dy = trajectory[:,1] - oy[:,None]
    r = np.hypot(dx, dy)

    # Check if any or all part of rectangular bot is colliding or not 
    if config.robot_type == RobotType.rectangle:
        yaw = trajectory[:,2]
        rot = np.array([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]])
        rot = np.transpose(rot, [2, 0, 1])
        local_ob = ob[:, None] - trajectory[:,0:2]
        local_ob = local_ob.reshape(-1, local_ob.shape[-1])
        local_ob = np.array([local_ob @ x for x in rot])
        local_ob = local_ob.reshape(-1, local_ob.shape[-1])
        upper_check = local_ob[:,0] <= config.robot_length / 2
        right_check = local_ob[:,1] <= config.robot_width / 2
        bottom_check = local_ob[:,0] >= -config.robot_length / 2
        left_check = local_ob[:,1] >= -config.robot_width / 2
        if (np.logical_and(np.logical_and(upper_check, right_check),
                           np.logical_and(bottom_check, left_check))).any():
            return float("Inf")
    
    # Check if any or all part of circular bot is colliding or not 
    elif config.robot_type == RobotType.circle:
        if np.array(r <= config.robot_radius).any():
            return float("Inf")

    min_r = np.min(r)
    return 1.0 / min_r  


def calc_to_goal_cost(trajectory, goal):
    """
        calculate goal cost with angle difference
    """

    dx = (goal[0] - trajectory[-1,0])
    dy = (goal[1] - trajectory[-1,1])
    error_angle = math.atan2(dy, dx)
    cost_angle = error_angle - trajectory[-1,2]
    cost = abs(math.atan2(math.sin(cost_angle), math.cos(cost_angle)))

    return cost

# To plot the heading trajectory
def plot_arrow(x, y, yaw, length=0.5, width=0.1):  
    plt.arrow(x, y, length * math.cos(yaw), length * math.sin(yaw),
              head_length=width, head_width=width)
    plt.plot(x, y)

# To plot robot
def plot_robot(x, y, yaw, config):  
    if config.robot_type == RobotType.rectangle:
        outline = np.array([[-config.robot_length / 2, config.robot_length / 2,
                             (config.robot_length / 2), -config.robot_length / 2,
                             -config.robot_length / 2],
                            [config.robot_width / 2, config.robot_width / 2,
                             - config.robot_width / 2, -config.robot_width / 2,
                             config.robot_width / 2]])
        Rot1 = np.array([[math.cos(yaw), math.sin(yaw)],
                         [-math.sin(yaw), math.cos(yaw)]])
        outline = (outline.T.dot(Rot1)).T
        outline[0, :] += x
        outline[1, :] += y
        plt.plot(np.array(outline[0, :]).flatten(),
                 np.array(outline[1, :]).flatten(), "-k")
    elif config.robot_type == RobotType.circle:
        circle = plt.Circle((x, y), config.robot_radius)
        plt.gcf().gca().add_artist(circle)
        out_x, out_y = (np.array([x, y]) +
                        np.array([np.cos(yaw), np.sin(yaw)]) * config.robot_radius)
        plt.plot([x, out_x], [y, out_y], "-r")




def main(gx=10.0, gy=10.0, robot_type=RobotType.circle):
    print(__file__ + " start!!")
    
    # Initialize bot's kinematics, goal and obstacles
    x_b = np.zeros((5))
    x_b = np.array([-2.5, 0.0, np.pi/8.0, 0.0, 0.0])

    goal_z = np.array([10,6])
    dist_to_goal = 0

    u = np.zeros((2))
    predicted_trajectory = np.zeros((31,5))

    reached = 0
  
    config.robot_type = robot_type

    ob=config.ob

    bot_no = np.zeros((0))
    i = 0

    figure = plt.figure()

    trajlist = []
    while True:
        plt.cla()
        ob = config.ob

        # Find the distance to reach goal and check if reached by predicted motion and trajectory
        dist_to_goal = math.hypot(x_b[0] - goal_z[0], x_b[1] - goal_z[1])

        if(dist_to_goal >= 1.5*config.robot_radius):
            #  Calculate predicted trajectory and motion of bot
            u, predicted_trajectory = dwa_control(x_b, config, goal_z, (np.delete(ob,(len(ob)-1),axis=0)))
            x_b = motion(x_b, u, config.dt)  # simulate robot
            
            # Update the trajectory and obstacle list
            trajlist.append([x_b[0],x_b[1],x_b[2],x_b[3],x_b[4]])
            ob = np.vstack((ob,[x_b[0],x_b[1]]))


        elif(dist_to_goal < 1.5*config.robot_radius):
            ob = np.vstack((ob,[x_b[0],x_b[1]]))
            if i not in bot_no:
                bot_no = np.append(bot_no,[i+1])
                reached += 1
                i+=1
  
        if show_animation:

            # for stopping simulation with the esc key.
            plt.gcf().canvas.mpl_connect(
                'key_release_event',
                lambda event: [exit(0) if event.key == 'escape' else None])

            plt.plot(predicted_trajectory[:, 0], predicted_trajectory[:, 1], "-g")
            plt.plot(x_b[0], x_b[1], "xr")
            plt.plot(goal_z[0], goal_z[1], "xb")
            plt.plot(ob[:, 0], ob[:, 1], "ok")
            plot_robot(x_b[0], x_b[1], x_b[2], config)
            plot_arrow(x_b[0], x_b[1], x_b[2])
                
            plt.axis("equal")
            plt.grid(True)
            plt.pause(0.0001)
            figure.canvas.draw()
            figure.canvas.flush_events()

        if (reached==1):
            print("GOAL!!!")
            break

    print("Done, succesfully reached goal without collision....!!")

# Initialize the robot type and execute the main function
if __name__ == '__main__':
    main(robot_type=RobotType.rectangle)