#include <chrono>
#include <functional>
#include <memory>
#include <string>

#include <rclcpp/rclcpp.hpp>
#include <memory>
#include "dwa_planner/dwa_planner.h"
#include <sensor_msgs/msg/laser_scan.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <nav_msgs/msg/path.hpp>

#include <geometry_msgs/msg/twist.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include "std_msgs/msg/string.hpp"
#include "std_msgs/msg/bool.hpp"


using namespace std::chrono_literals;
using std::placeholders::_1;

class DWAPlannerNode : public rclcpp::Node
{
  public:
    DWAPlannerNode()
    : Node("dwa_planner_node"), count_(0)
    {
      // Subscribers

      scan_subscriber_ = this->create_subscription<sensor_msgs::msg::LaserScan>(
      "vision/image_segmented/scan", 10, std::bind(&DWAPlannerNode::ScanCallback, this, _1));

      odom_subscriber_ = this->create_subscription<nav_msgs::msg::Odometry>(
      "/dolly/odom", 10, std::bind(&DWAPlannerNode::OdomCallback, this, _1));

      goal_subscriber_ = this->create_subscription<geometry_msgs::msg::PoseStamped>(
      "/move_base_simple/goal", 10, std::bind(&DWAPlannerNode::GoalCallback, this, _1));

      // Publishers
      cmd_publisher_ = this->create_publisher<geometry_msgs::msg::Twist>("dolly/cmd_vel", 10);
      goal_reached_publisher_ = this->create_publisher<std_msgs::msg::Bool>("planner/goal_reached", 10);
      path_publisher_ = this->create_publisher<nav_msgs::msg::Path>("planner/path", 10);

      timer_ = this->create_wall_timer(
      100ms, std::bind(&DWAPlannerNode::TimerCallback, this));
      State init_state({{-39.1f, 5.1f, 0.0f , 0.0f, 0.0f}});
      dwa_planner_ = new DWAPlanner(init_state);
    }

  private:

    void ScanCallback(const sensor_msgs::msg::LaserScan::SharedPtr msg) const
    {
      std::vector<float> scan_distances = msg->ranges;
      dwa_planner_->SetObstacles(scan_distances, msg->angle_increment, msg->angle_min, msg->angle_max, msg->range_min, msg->range_max);

    }

    void OdomCallback(const nav_msgs::msg::Odometry::SharedPtr msg) const
    {
      float x = msg->pose.pose.position.x;
      float y = msg->pose.pose.position.y;
      // Get yaw from quaternions
      tf2::Quaternion q(
        msg->pose.pose.orientation.x,
        msg->pose.pose.orientation.y,
        msg->pose.pose.orientation.z,
        msg->pose.pose.orientation.w);
      tf2::Matrix3x3 m(q);
      double roll, pitch, yaw;
      m.getRPY(roll, pitch, yaw);
      float vx = msg->twist.twist.linear.x;
      float omega = msg->twist.twist.angular.z;

      State state({{x, y, (float)yaw, vx, omega}});
      dwa_planner_->SetState(state);
    }

    void GoalCallback(const geometry_msgs::msg::PoseStamped::SharedPtr msg) const
    {
      Point goal({{(float)msg->pose.position.x, (float)msg->pose.position.y}});
      dwa_planner_->SetGoal(goal);
    }


    void TimerCallback()
    {
      auto control_msg = geometry_msgs::msg::Twist();
      Control control = dwa_planner_->GetCmd();
      control_msg.linear.x = control[0];
      control_msg.angular.z = control[1];
      cmd_publisher_->publish(control_msg);

      //Check if goal was reached
      if(!is_goal_reached && dwa_planner_->IsGoalReached()){
        auto goal_reached_msg = std_msgs::msg::Bool();
        goal_reached_msg.data = dwa_planner_->IsGoalReached();
        goal_reached_publisher_->publish(goal_reached_msg);
        RCLCPP_INFO(this->get_logger(), "Goal Reached");
      } 
      is_goal_reached = dwa_planner_->IsGoalReached();

      //Publish trajectory
      Trajectory traj = dwa_planner_->GetTrajectory();
      auto path_msg = nav_msgs::msg::Path();
      path_msg.header.frame_id = "odom_demo";
      path_msg.header.stamp = rclcpp::Clock().now();
      int skip_n = 4;
      std::vector<geometry_msgs::msg::PoseStamped> path;
      // std::cout<<traj.size()<<std::endl;
      for (unsigned int i = 0; i < traj.size(); i += skip_n)
      {
        float x = traj[i][0];
        float y = traj[i][1];
        auto pose = geometry_msgs::msg::PoseStamped();
        pose.header.stamp = rclcpp::Clock().now();

        // pose.header.frame_id = "odom_demo";

        pose.pose.position.x = x;
        pose.pose.position.y = y;
        pose.pose.orientation.w = 1.0;
        pose.pose.position.z = 5.2;


        path.push_back(pose);
      }
      path_msg.poses = path;
      path_publisher_->publish(path_msg);

      //control_msg.header.stamp = this->get_clock();

      // publisher_->publish(message);
    }
    rclcpp::TimerBase::SharedPtr timer_;
    rclcpp::Subscription<sensor_msgs::msg::LaserScan>::SharedPtr scan_subscriber_;
    rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr odom_subscriber_;
    rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr goal_subscriber_;

    rclcpp::Publisher<geometry_msgs::msg::Twist>::SharedPtr cmd_publisher_;
    rclcpp::Publisher<std_msgs::msg::Bool>::SharedPtr goal_reached_publisher_;
    rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr path_publisher_;


    DWAPlanner * dwa_planner_;
    size_t count_;
    bool is_goal_reached = false;
  };

  int main(int argc, char * argv[])
  {
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<DWAPlannerNode>());
    rclcpp::shutdown();
    return 0;
  }