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
  float predict_time = 3.0;
  float to_goal_cost_gain = 0.75;
  float speed_cost_gain = 1.5;
};

class DWAPlanner{
    public:
        DWAPlanner(State init_state);
        void SetObstacles(std::vector<float> scan_distances, float angle_increment, float angle_min, float angle_max, float range_min, float range_max);
        void SetState(State state);
        void SetGoal(Point goal);

        Control GetCmd();

    private:
        State Motion(State x, Control u, float dt);
        Window CalcDynamicWindow(State x);
        Trajectory CalcTrajectory(State x, float v, float y);
        float CalcObstacleCost(Trajectory traj, Obstacle ob);
        float CalcToGoalCost(Trajectory traj, Point goal);
        Trajectory CalcFinalInput( State x, Control& u, Window dw, Point goal,
                                                std::vector<std::array<float, 2>>ob);
        Trajectory DWAControl(State x, Control & u, Point goal, Obstacle ob);
        

        State x_;
        Point goal_;
        Control u_;
        Obstacle ob_;
        Obstacle ob_closest_;

        Obstacle ob_local_;
        Trajectory trajectory_;
        int count_ = 0;
        Config config_;

};
