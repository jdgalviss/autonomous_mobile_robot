#include "dwa_planner/dwa_planner.h"

DWAPlanner::DWAPlanner(State init_state)
{
  std::cout << "DWAPLanner" << std::endl;
  x_ = init_state;
  goal_ = Point({{init_state[0], init_state[1]}});
  u_ = Control({{0.0, 0.0}});
  trajectory_.push_back(x_);
}

State DWAPlanner::Motion(State x, Control u, float dt)
{
  x[2] += 1.9*u[1] * dt;
  x[0] += 0.9*u[0] * std::cos(x[2]) * dt;
  x[1] += 0.9*u[0] * std::sin(x[2]) * dt;
  x[3] = 0.9*u[0];
  x[4] = 1.9*u[1];
  return x;
}

Window DWAPlanner::CalcDynamicWindow()
{
  // std::cout<<"vx: "<<x[3]<<", omega: "<<x_[4]<<std::endl;
  return {{std::max((u_[0] - config_.max_accel * config_.dt), config_.min_speed),
           std::min((u_[0] + config_.max_accel * config_.dt), config_.max_speed),
           std::max((u_[1] - config_.max_dyawrate * config_.dt), -config_.max_yawrate),
           std::min((u_[1] + config_.max_dyawrate * config_.dt), config_.max_yawrate)}};
}

Trajectory DWAPlanner::CalcTrajectory(float v, float y)
{
  State x(x_);
  Trajectory traj;
  traj.push_back(x);
  float time = 0.0;
  while (time <= config_.predict_time)
  {
    x = Motion(x, std::array<float, 2>{{v, y}}, config_.dt);
    traj.push_back(x);
    time += config_.dt;
  }
  return traj;
}

float DWAPlanner::CalcObstacleCost(Trajectory traj)
{
  // calc obstacle cost inf: collistion, 0:free
  int skip_n = 4;
  float minr = std::numeric_limits<float>::max();

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

float DWAPlanner::CalcToGoalCost(Trajectory traj)
{

  float goal_magnitude = std::sqrt(pow(goal_[0]-x_[0],2) + pow(goal_[1]-x_[1],2));

  float traj_magnitude = std::sqrt(std::pow(traj.back()[0]-x_[0], 2) + std::pow(traj.back()[1]-x_[1], 2));

  float dot_product = ((goal_[0]-x_[0]) * (traj.back()[0]-x_[0])) + ((goal_[1]-x_[1]) * (traj.back()[1]-x_[1]));
  float error = dot_product / (goal_magnitude * traj_magnitude);
  //std::cout<<"error: "<<error<<std::endl;
  float error_angle = std::acos(error);
  //std::cout<<"error: "<<error_angle<<std::endl;

  // float dist = sqrt(pow((goal[0] - traj.back()[0]),2) + pow((goal[1] - traj.back()[1]), 2));
  float cost = config_.to_goal_cost_gain * error_angle;
  //std::cout<<"error: "<<cost<<" / goal: "<<goal[0]<<", "<<goal[1]<< " / traj: " <<traj.back()[0]<<", "<<traj.back()[1]<<std::endl;

  return cost;
}

Trajectory DWAPlanner::CalcFinalInput(Window dw)
{

  float min_cost = 10000.0;
  Control min_u = u_;
  min_u[0] = 0.0;
  Trajectory best_traj;
  // std::cout<<dw[0]<<", "<<dw[1]<<", "<<dw[2]<<", "<<dw[3]<<std::endl;
  // evalucate all trajectory with sampled input in dynamic window
  for (float v = dw[0]; v <= dw[1]; v += config_.v_reso)
  {
    //std::cout<<"hola2"<<std::endl;

    for (float y = dw[2]; y <= dw[3]; y += config_.yawrate_reso)
    {
      //std::cout<<"hola"<<std::endl;
      Trajectory traj = CalcTrajectory(v, y);

      float to_goal_cost = CalcToGoalCost(traj);
      float dist_to_goal = sqrt(pow((goal_[0] - x_[0]),2) + pow((goal_[1] - x_[1]), 2));
      dist_to_goal = dist_to_goal < 1.0f ? dist_to_goal : 1.0f;
      float speed_cost = dist_to_goal*config_.speed_cost_gain * (config_.max_speed - traj.back()[3]);
      float ob_cost = 1.0*CalcObstacleCost(traj);
      float final_cost = to_goal_cost + speed_cost + ob_cost;

      if (min_cost >= final_cost)
      {
        min_cost = final_cost;
        min_u = Control{{v, y}};
        best_traj = traj;
      }
    }
  }
  // std::cout<<"error: "<<min_cost<< " obs_cost:"<< obstacle_cost<<std::endl;

  u_ = min_u;
  return best_traj;
}

Trajectory DWAPlanner::DWAControl()
{
  // # Dynamic Window control
  Window dw = CalcDynamicWindow();
  Trajectory traj = CalcFinalInput(dw);

  return traj;
}

void DWAPlanner::SetObstacles(std::vector<float> scan_distances, float angle_increment, float angle_min, float angle_max, float range_min, float range_max)
{
  std::vector<float>::iterator d_ptr;
  // Iterate through scans and calculate location of points
  ob_.clear();
  float current_angle = angle_min;
  float min_dist = 10000.0f;
  Point closest_obstacle({min_dist, min_dist});
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
      // if (current_angle < angle_min + 0.2)
      // {
      //   std::cout << "dist: " << *d_ptr << "x: " << x << " y:" << y << std::endl;
      // }
    }
  }
  ob_.push_back(closest_obstacle);
    if(ob_.size()>=25)
      ob_.erase(ob_.begin());
}

void DWAPlanner::SetState(State state)
{
  x_ = state;
  //trajectory_.push_back(x);

}

void DWAPlanner::SetGoal(Point goal){
  goal_ = goal;
  goal_reached_ = false;
}

bool DWAPlanner::IsGoalReached(){
  return goal_reached_;
}

Trajectory DWAPlanner::GetTrajectory(){
  return trajectory_;
}

Control DWAPlanner::GetCmd()
{
  trajectory_ = DWAControl();
  //when goal is reached
  if(sqrt(pow(goal_[0]-x_[0],2) + pow(goal_[1]-x_[1],2))<0.5){
    u_ = Control{{0, 0.0}};
    if(!goal_reached_)
      goal_reached_ = true;
  }
  
  // if(trajectory_.size()>1)
    // std::cout<<"state: "<<x_[0]<<", "<<x_[1]<<" traj: "<<trajectory_[0][0]<<", "<<trajectory_[0][1]<<std::endl;
  // std::cout<<"============="<<std::endl;
  // std::cout<<"x: "<<x_[0]<<" vx: "<<u_[0]<<", omega: "<<u_[1]<<std::endl;
  return u_;
}