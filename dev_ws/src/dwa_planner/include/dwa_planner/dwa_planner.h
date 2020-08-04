/*! \file dwa_planner.h
 *  \brief A class to perform collision avoidance based on the dynamic window approach
 *  By: Juan David Galvis
 *  https://github.com/jdgalviss
 */
#include<iostream>
#include<vector>
#include<array>
#include<cmath>
#include<math.h>
#include<opencv2/opencv.hpp>
#include<opencv2/core/core.hpp>
#include<opencv2/highgui/highgui.hpp>
#include <tf2/transform_datatypes.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>
#include <tf2/convert.h>

#define PI 3.141592653


using Trajectory = std::vector<std::array<float, 5>>;
using Obstacle = std::vector<std::array<float, 2>>;
using State = std::array<float, 5>;
using Window = std::array<float, 4>;
using Point = std::array<float, 2>;
using Control = std::array<float, 2>;

/*! Configuration class with the parameters of the algorithm */
class Config{
public:
  float max_speed = 0.45;
  float min_speed = -0.2;
  float max_yawrate = 60.0 * PI / 180.0;
  float max_accel = 0.3;
  float robot_radius = 0.45;
  float max_dyawrate = 40.0 * PI / 180.0;

  float v_reso = 0.01;
  float yawrate_reso = 0.1 * PI / 180.0;

  float dt = 0.1;
  float predict_time = 4.0;
  float to_goal_cost_gain = 0.75;
  float speed_cost_gain = 1.5;
};

/*! DWAPlanner class */
class DWAPlanner{
    public:
        DWAPlanner(State init_state);
        void SetObstacles(std::vector<float> scan_distances, float angle_increment, float angle_min, 
                          float angle_max, float range_min, float range_max); //!< Stores obstacle points from scan msgs
        void SetState(State state);       //!< Stores the current state of the robot for planning
        void SetGoal(Point goal);         //!< Stores the desired goal 
        bool IsGoalReached();             //!< Returns a bool that defines if the goal has been reached by the robot          
        Trajectory GetTrajectory();       //!< Returns current trajectory to be followed by the robot according to the DWA
        Control GetCmd();                 //!< Returns speed and yaw_rate commands defined by the DWA

    private:
        State Motion(State x, Control u, float dt);   //!< Executes a motion simulation step to predict robot's state
        Window CalcDynamicWindow();                   //!< Calculates the dynamic window depending on constrains defined in Config class
        Trajectory CalcTrajectory(float v, float y);  //!< Calculates trajectory followed with speed and yaw_rate
        float CalcObstacleCost(Trajectory traj);      //!< Calculate obstacle cost defined by the current trajectory
        float CalcToGoalCost(Trajectory traj);        //!< Calculate cost depending on distance to goal of the current trajectory
        Trajectory CalcFinalInput(Window dw);         //!< Calculate motion command by evaluating the cost of trajectories inside window
        Trajectory DWAControl();                      //!< Calculate dynamic window and trajectory with the smallest cost

        State x_;                       /*!< Vector containing state of the robot (position and velocities) */
        Point goal_;                    /*!< Vector containing x and y coordinates of the current goal */
        Control u_;                     /*!< Motion command (speed and yaw rate) defined by DWA */
        Obstacle ob_;                   /*!< Vector with scan points corresponding to obstacles*/
        Trajectory trajectory_;         /*!< Current best trajectory */
        int count_ = 0;
        Config config_;
        bool goal_reached_ = false;     /*!< Bool that defines if the goal has been reached */
};
