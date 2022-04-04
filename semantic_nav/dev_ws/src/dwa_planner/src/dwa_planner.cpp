/*! \file dwa_planner.cpp
 *  \brief A class to perform collision avoidance based on the dynamic window approach
 *  By: Juan David Galvis
 *  https://github.com/jdgalviss
 */
#include "dwa_planner/dwa_planner.h"

//! DWAPlanner Constructor.
    /*!
      Initialize the state, the goal and the control output
      \param init_state Initial State of the robot, when an instance of the planner is created
    */
DWAPlanner::DWAPlanner(State init_state)
{
  std::cout << "DWAPLanner" << std::endl;
  x_ = init_state;
  goal_ = Point({{init_state[0], init_state[1]}});
  u_ = Control({{0.0, 0.0}});
  trajectory_.push_back(x_);
}

//! Motion function
    /*!
      Executes a motion simulation step to predict robot's state
      \param x State of the robot
      \param u Motion cmd
      \param dt Time delta
      \return State after executing a motion step from the state x with motion cmd u
    */
State DWAPlanner::Motion(State x, Control u, float dt)
{
  x[2] += 1.9*u[1] * dt;
  x[0] += 0.9*u[0] * std::cos(x[2]) * dt;
  x[1] += 0.9*u[0] * std::sin(x[2]) * dt;
  x[3] = 0.9*u[0];
  x[4] = 1.9*u[1];
  return x;
}

//! Motion function
    /*!
      Calculates the dynamic window depending on constrains defined in Config class
      \return window of possible speeds depending on config parameters
    */
Window DWAPlanner::CalcDynamicWindow()
{
  return {{std::max((u_[0] - config_.max_accel * config_.dt), config_.min_speed),
           std::min((u_[0] + config_.max_accel * config_.dt), config_.max_speed),
           std::max((u_[1] - config_.max_dyawrate * config_.dt), -config_.max_yawrate),
           std::min((u_[1] + config_.max_dyawrate * config_.dt), config_.max_yawrate)}};
}

//! CalcTrajectory function
    /*!
      Calculates trajectory followed with speed and yaw_rate
      /param v speed
      /param y yaw_rate
      \return trajectory predicted when moving with speeds v and y 
              on a time frime (predict_time) 
    */
Trajectory DWAPlanner::CalcTrajectory(float v, float y)
{
  State x(x_);             // Copy of current state
  Trajectory traj;
  traj.push_back(x);       // First element of the trajectory is current state
  float time = 0.0;
  while (time <= config_.predict_time)  //Simulate trajectory for predict_time seconds
  {
    x = Motion(x, std::array<float, 2>{{v, y}}, config_.dt);  // Perform one motion step
    traj.push_back(x);
    time += config_.dt;
  }
  return traj;
}

//! CalcObstacleCost function
    /*!
      Calculate obstacle cost defined by the current trajectory
      /param traj Trajectory that we want to calculate the cost of
      \return cost float obstacle cost
    */
float DWAPlanner::CalcObstacleCost(Trajectory traj)
{
  // calc obstacle cost inf: collistion, 0:free
  int skip_n = 4; //skip some points of the trajectory
  float minr = std::numeric_limits<float>::max();

  // Evaluate distance to obstacles for evert point int he trajectory and 
  // make sure that the robot won't crash
  for (unsigned int ii = 0; ii < traj.size(); ii += skip_n)
  {
    for (unsigned int i = 0; i < ob_.size(); i++)
    {
        float ox = ob_[i][0];
        float oy = ob_[i][1];  
        float dx = traj[ii][0] - ox;
        float dy = traj[ii][1] - oy;

        float r = std::sqrt(dx * dx + dy * dy);
        if (r <= config_.robot_radius)
        {
          return std::numeric_limits<float>::max();
        }

        if (minr >= r)
        {
          minr = r;
        }
    }
  }
  return 1.0 / minr;
}

//! CalcToGoalCost function
    /*!
      Calculate cost depending on distance to goal of the current trajectory
      /param traj Trajectory that we want to calculate the cost of
      \return cost float to goal cost
    */
float DWAPlanner::CalcToGoalCost(Trajectory traj)
{
  // Cost is defined by the angle to goal
  float goal_magnitude = std::sqrt(pow(goal_[0]-x_[0],2) + pow(goal_[1]-x_[1],2));
  float traj_magnitude = std::sqrt(std::pow(traj.back()[0]-x_[0], 2) + std::pow(traj.back()[1]-x_[1], 2));
  float dot_product = ((goal_[0]-x_[0]) * (traj.back()[0]-x_[0])) + ((goal_[1]-x_[1]) * (traj.back()[1]-x_[1]));
  float error = dot_product / (goal_magnitude * traj_magnitude);
  float error_angle = std::acos(error);
  float cost = config_.to_goal_cost_gain * error_angle;

  return cost;
}

//! CalcFinalInput function
    /*!
      Calculate motion command by evaluating the cost of trajectories inside window
      /param dw Dynamic window
      \return trajectory best trajectory
    */
Trajectory DWAPlanner::CalcFinalInput(Window dw)
{
  float min_cost = 10000.0;
  Control min_u = u_;
  min_u[0] = 0.0;
  Trajectory best_traj;
  // evalucate all trajectory with sampled input in dynamic window
  for (float v = dw[0]; v <= dw[1]; v += config_.v_reso)
  {
    for (float y = dw[2]; y <= dw[3]; y += config_.yawrate_reso)
    {
      Trajectory traj = CalcTrajectory(v, y);

      // Add all costs
      float to_goal_cost = CalcToGoalCost(traj);
      float dist_to_goal = sqrt(pow((goal_[0] - x_[0]),2) + pow((goal_[1] - x_[1]), 2));
      dist_to_goal = dist_to_goal < 1.0f ? dist_to_goal : 1.0f;
      float speed_cost = dist_to_goal*config_.speed_cost_gain * (config_.max_speed - traj.back()[3]);
      float ob_cost = 1.0*CalcObstacleCost(traj);
      float final_cost = to_goal_cost + speed_cost + ob_cost;

      // Save motion command that produces smallest cost
      if (min_cost >= final_cost)
      {
        min_cost = final_cost;
        min_u = Control{{v, y}};
        best_traj = traj;
      }
    }
  }

  u_ = min_u;
  return best_traj;
}

//! DWAControl function
    /*!
      Calculate dynamic window and trajectory with the smallest cost
      \return trajectory best trajectory
    */
Trajectory DWAPlanner::DWAControl()
{
  // # Dynamic Window control
  Window dw = CalcDynamicWindow();
  Trajectory traj = CalcFinalInput(dw);
  return traj;
}

//! SetObstacles function
    /*!
      Stores obstacle points from scan msgs
      \param scan_distances Vector with the scan distances
      \param angle_increment Angle increment for each scan point in the scan_distances vector
      \param angle_min Minimum angle of the scan points
      \param angle_max Maximum angle of the scan points
      \param range_min Min posible distance possible in the scan distances
      \param range_max Max posible distance possible in the scan distances
    */
void DWAPlanner::SetObstacles(std::vector<float> scan_distances, float angle_increment, float angle_min, float angle_max, float range_min, float range_max)
{
  std::vector<float>::iterator d_ptr;
  // Iterate through scans and calculate location of points
  ob_.clear();
  float current_angle = angle_min;
  float min_dist = 10000.0f;
  Point closest_obstacle({min_dist, min_dist});

  // Calculate the point closest to the robot
  for (d_ptr = scan_distances.begin(); d_ptr < scan_distances.end(); d_ptr++)
  {
    if ((*d_ptr < range_max) && (*d_ptr > range_min))
    {
      float x_local = *d_ptr * cos(current_angle);
      float y_local = *d_ptr * sin(current_angle);
      float ox = x_[0] + x_local * cos(x_[2]) - y_local * sin(x_[2]);
      float oy = x_[1] + x_local * sin(x_[2]) + y_local * cos(x_[2]);

      if(*d_ptr <=min_dist){
        min_dist = *d_ptr;
        closest_obstacle = Point{{ox, oy}};
      }
      if (current_angle <= angle_max)
        current_angle += angle_increment;
    }
  }
  // Save closest obstacle in vector (save last 25 points)
  ob_.push_back(closest_obstacle);
    if(ob_.size()>=25)
      ob_.erase(ob_.begin());
}

//! SetState function
    /*!
      Stores the current state of the robot for planning
      \param state 
    */
void DWAPlanner::SetState(State state)
{
  x_ = state;
  //trajectory_.push_back(x);

}

//! SetGoal function
    /*!
      Stores the desired goal 
      \param goal 
    */
void DWAPlanner::SetGoal(Point goal){
  goal_ = goal;
  goal_reached_ = false;
}

//! IsGoalReached function
    /*!
      Returns a bool that defines if the goal has been reached by the robot 
    */
bool DWAPlanner::IsGoalReached(){
  return goal_reached_;
}

//! GetTrajectory function
    /*!
      Returns current trajectory to be followed by the robot according to the DWA
    */
Trajectory DWAPlanner::GetTrajectory(){
  return trajectory_;
}

//! GetCmd function
    /*!
      Returns speed and yaw_rate commands defined by the DWA
    */
Control DWAPlanner::GetCmd()
{
  trajectory_ = DWAControl();
  //when goal is reached
  if(sqrt(pow(goal_[0]-x_[0],2) + pow(goal_[1]-x_[1],2))<0.5){
    u_ = Control{{0, 0.0}};
    if(!goal_reached_)
      goal_reached_ = true;
  }
  return u_;
}