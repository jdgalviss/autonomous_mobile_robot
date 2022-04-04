#include <chrono>
#include <functional>
#include <memory>
#include <string>
#include<vector>
#include<array>

#include "rclcpp/rclcpp.hpp"
#include <geometry_msgs/msg/pose_stamped.hpp>
#include "std_msgs/msg/bool.hpp"

using namespace std::chrono_literals;
using std::placeholders::_1;

using Waypoints = std::vector<std::array<float, 2>>;
/* This example creates a subclass of Node and uses std::bind() to register a
* member function as a callback from the timer. */

class TrajectoryPublisher : public rclcpp::Node
{
  public:
    TrajectoryPublisher()
    : Node("trajectory_publisher"), count_(0)
    {
      goal_reached_sub_ = this->create_subscription<std_msgs::msg::Bool>(
      "planner/goal_reached", 10, std::bind(&TrajectoryPublisher::GoalReachedCallback, this, _1));
      trajectory_publisher_ = this->create_publisher<geometry_msgs::msg::PoseStamped>("move_base_simple/goal", 10);
      Waypoints waypoints({{{-20.3, 5.0}}, {{-20.0, 39.5}}, {{-39.9, 38.5}},{{-39.5, 5.0}}});
      waypoints_ = waypoints;

      timer_ = this->create_wall_timer(
      500ms, std::bind(&TrajectoryPublisher::PublishWaypoint, this));
    }

  private:
    void PublishWaypoint() const
    {
      auto trajectory_message = geometry_msgs::msg::PoseStamped();      
      trajectory_message.pose.position.x = waypoints_[waypoint_count][0];
      trajectory_message.pose.position.y = waypoints_[waypoint_count][1];
      trajectory_message.header.stamp = rclcpp::Clock().now();
      trajectory_publisher_->publish(trajectory_message);

    }
    void GoalReachedCallback(const std_msgs::msg::Bool::SharedPtr msg) const
    {
      RCLCPP_INFO(this->get_logger(), "Goal Reached");
      if(msg->data){
        if(waypoint_count == 3)
          waypoint_count = 0;
        else
          waypoint_count++;
        
        PublishWaypoint();
      }
    }
    rclcpp::TimerBase::SharedPtr timer_;
    rclcpp::Subscription<std_msgs::msg::Bool>::SharedPtr goal_reached_sub_;
    rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr trajectory_publisher_;
    Waypoints waypoints_;
    mutable int waypoint_count = 0;
    size_t count_;
  };

  int main(int argc, char * argv[])
  {
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<TrajectoryPublisher>());
    rclcpp::shutdown();
    return 0;
  }