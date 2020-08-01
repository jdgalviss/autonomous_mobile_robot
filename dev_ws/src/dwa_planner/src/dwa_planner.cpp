#include "dwa_planner/dwa_planner.h"

DWAPlanner::DWAPlanner(State init_state)
{
  std::cout << "DWAPLanner" << std::endl;
  x_ = init_state;
  goal_ = Point({{init_state[0]+1.0f, init_state[1]}});
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

Window DWAPlanner::CalcDynamicWindow(State x)
{
  // std::cout<<"vx: "<<x[3]<<", omega: "<<x_[4]<<std::endl;
  return {{std::max((u_[0] - config_.max_accel * config_.dt), config_.min_speed),
           std::min((u_[0] + config_.max_accel * config_.dt), config_.max_speed),
           std::max((u_[1] - config_.max_dyawrate * config_.dt), -config_.max_yawrate),
           std::min((u_[1] + config_.max_dyawrate * config_.dt), config_.max_yawrate)}};
}

Trajectory DWAPlanner::CalcTrajectory(State x, float v, float y)
{

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

float DWAPlanner::CalcObstacleCost(Trajectory traj, Obstacle ob)
{
  // calc obstacle cost inf: collistion, 0:free
  int skip_n = 2;
  float minr = std::numeric_limits<float>::max();
  float min_dist = 10000.0f;
  int index_obs = -1;
  for (unsigned int i = 0; i < ob.size(); i++)
  {

    float ox = ob[i][0];
    float oy = ob[i][1];

    float dist = sqrt(pow((ox - x_[0]),2) + pow((oy - x_[1]), 2));

    if(dist <=min_dist){
      min_dist = dist;
      index_obs = i;
    }
  }
  if(index_obs != -1){
    float ox = ob[index_obs][0];
    float oy = ob[index_obs][1];
    std::array<float, 2> obstacle({ox, oy});
    ob_closest_.push_back(obstacle);
    if(ob_closest_.size()>=25)
      ob_closest_.erase(ob_closest_.begin());
  }

  for (unsigned int ii = 0; ii < traj.size(); ii += skip_n)
  {
    for (unsigned int i = 0; i < ob_closest_.size(); i++)
    {
        float ox = ob_closest_[i][0];
        float oy = ob_closest_[i][1];  
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

float DWAPlanner::CalcToGoalCost(Trajectory traj, Point goal)
{

  float goal_magnitude = std::sqrt(pow(goal[0]-x_[0],2) + pow(goal[1]-x_[1],2));

  float traj_magnitude = std::sqrt(std::pow(traj.back()[0]-x_[0], 2) + std::pow(traj.back()[1]-x_[1], 2));

  float dot_product = ((goal[0]-x_[0]) * (traj.back()[0]-x_[0])) + ((goal[1]-x_[1]) * (traj.back()[1]-x_[1]));
  float error = dot_product / (goal_magnitude * traj_magnitude);
  //std::cout<<"error: "<<error<<std::endl;
  float error_angle = std::acos(error);
  //std::cout<<"error: "<<error_angle<<std::endl;

  // float dist = sqrt(pow((goal[0] - traj.back()[0]),2) + pow((goal[1] - traj.back()[1]), 2));
  float cost = config_.to_goal_cost_gain * error_angle;
  //std::cout<<"error: "<<cost<<" / goal: "<<goal[0]<<", "<<goal[1]<< " / traj: " <<traj.back()[0]<<", "<<traj.back()[1]<<std::endl;

  return cost;
}

Trajectory DWAPlanner::CalcFinalInput(
    State x, Control &u,
    Window dw, Point goal,
    std::vector<std::array<float, 2>> ob)
{

  float min_cost = 10000.0;
  Control min_u = u;
  min_u[0] = 0.0;
  Trajectory best_traj;
  // std::cout<<dw[0]<<", "<<dw[1]<<", "<<dw[2]<<", "<<dw[3]<<std::endl;
  float obstacle_cost = 0.0f;
  // evalucate all trajectory with sampled input in dynamic window
  for (float v = dw[0]; v <= dw[1]; v += config_.v_reso)
  {
    //std::cout<<"hola2"<<std::endl;

    for (float y = dw[2]; y <= dw[3]; y += config_.yawrate_reso)
    {
      //std::cout<<"hola"<<std::endl;
      Trajectory traj = CalcTrajectory(x, v, y);

      float to_goal_cost = CalcToGoalCost(traj, goal);
      float dist_to_goal = sqrt(pow((goal_[0] - x_[0]),2) + pow((goal_[1] - x_[1]), 2));
      dist_to_goal = dist_to_goal < 1.0f ? dist_to_goal : 1.0f;
      float speed_cost = dist_to_goal*config_.speed_cost_gain * (config_.max_speed - traj.back()[3]);
      float ob_cost = 1.0*CalcObstacleCost(traj, ob);
      float final_cost = to_goal_cost + speed_cost + ob_cost;

      if (min_cost >= final_cost)
      {
        min_cost = final_cost;
        min_u = Control{{v, y}};
        best_traj = traj;
        obstacle_cost = ob_cost;
      }
    }
  }
  // std::cout<<"error: "<<min_cost<< " obs_cost:"<< obstacle_cost<<std::endl;

  u = min_u;
  return best_traj;
}

Trajectory DWAPlanner::DWAControl(State x, Control &u,
                                  Point goal, Obstacle ob)
{
  // # Dynamic Window control
  Window dw = CalcDynamicWindow(x);
  Trajectory traj = CalcFinalInput(x, u, dw, goal, ob);

  return traj;
}

void DWAPlanner::SetObstacles(std::vector<float> scan_distances, float angle_increment, float angle_min, float angle_max, float range_min, float range_max)
{
  std::vector<float>::iterator d_ptr;
  // Iterate through scans and calculate location of points
  ob_.clear();
  float current_angle = angle_min;
  for (d_ptr = scan_distances.begin(); d_ptr < scan_distances.end(); d_ptr++)
  {
    if ((*d_ptr < range_max) && (*d_ptr > range_min))
    {
      float x_local = *d_ptr * cos(current_angle);
      float y_local = *d_ptr * sin(current_angle);
      float x = x_[0] + x_local * cos(x_[2]) - y_local * sin(x_[2]);
      float y = x_[1] + x_local * sin(x_[2]) + y_local * cos(x_[2]);
      std::array<float, 2> obstacle({x, y});
      ob_.push_back(obstacle);
      if (current_angle <= angle_max)
        current_angle += angle_increment;
      // if (current_angle < angle_min + 0.2)
      // {
      //   std::cout << "dist: " << *d_ptr << "x: " << x << " y:" << y << std::endl;
      // }
    }
  }
}

void DWAPlanner::SetState(State state)
{
  x_ = state;
  //trajectory_.push_back(x);

}

void DWAPlanner::SetGoal(Point goal){
  goal_ = goal;
}


Control DWAPlanner::GetCmd()
{
  Trajectory ltraj = DWAControl(x_, u_, goal_, ob_);
  if(sqrt(pow(goal_[0]-x_[0],2) + pow(goal_[1]-x_[1],2))<0.1)
    u_ = Control{{0, 0}};
  // std::cout<<"============="<<std::endl;
  // std::cout<<"x: "<<x_[0]<<" vx: "<<u_[0]<<", omega: "<<u_[1]<<std::endl;
  return u_;
}

// int main()
// {
//   State x({{0.0, 0.0, PI / 8.0, 0.0, 0.0}});
//   Point goal({{10.0, 10.0}});
//   Obstacle ob({{{-1, -1}},
//                {{0, 2}},
//                {{4.0, 2.0}},
//                {{5.0, 4.0}},
//                {{5.0, 5.0}},
//                {{5.0, 6.0}},
//                {{5.0, 9.0}},
//                {{8.0, 9.0}},
//                {{7.0, 9.0}},
//                {{12.0, 12.0}}});

//   Control u({{0.0, 0.0}});
//   Config config_;
//   Trajectory traj;
//   traj.push_back(x);

//   bool terminal = false;

//   int count = 0;

//   for (int i = 0; i < 1000 && !terminal; i++)
//   {
//     Trajectory ltraj = DWAControl(x, u, goal, ob);
//     // x = Motion(x, u, config_.dt);
//     traj.push_back(x);

//     // // visualization
//     // cv::Mat bg(3500,3500, CV_8UC3, cv::Scalar(255,255,255));
//     // cv::circle(bg, cv_offset(goal[0], goal[1], bg.cols, bg.rows),
//     //            30, cv::Scalar(255,0,0), 5);
//     // for(unsigned int j=0; j<ob.size(); j++){
//     //   cv::circle(bg, cv_offset(ob[j][0], ob[j][1], bg.cols, bg.rows),
//     //              20, cv::Scalar(0,0,0), -1);
//     // }
//     // for(unsigned int j=0; j<ltraj.size(); j++){
//     //   cv::circle(bg, cv_offset(ltraj[j][0], ltraj[j][1], bg.cols, bg.rows),
//     //              7, cv::Scalar(0,255,0), -1);
//     // }
//     // cv::circle(bg, cv_offset(x[0], x[1], bg.cols, bg.rows),
//     //            30, cv::Scalar(0,0,255), 5);

//     // cv::arrowedLine(
//     //   bg,
//     //   cv_offset(x[0], x[1], bg.cols, bg.rows),
//     //   cv_offset(x[0] + std::cos(x[2]), x[1] + std::sin(x[2]), bg.cols, bg.rows),
//     //   cv::Scalar(255,0,255),
//     //   7);

//     // if (std::sqrt(std::pow((x[0] - goal[0]), 2) + std::pow((x[1] - goal[1]), 2)) <= config_.robot_radius){
//     //   terminal = true;
//     //   for(unsigned int j=0; j<traj.size(); j++){
//     //     cv::circle(bg, cv_offset(traj[j][0], traj[j][1], bg.cols, bg.rows),
//     //                 7, cv::Scalar(0,0,255), -1);
//     //   }
//     // }

//     // cv::imshow("dwa", bg);
//     // cv::waitKey(5);

//     // std::string int_count = std::to_string(count);
//     // cv::imwrite("./pngs/"+std::string(5-int_count.length(), '0').append(int_count)+".png", bg);

//     count++;
//   }
// }