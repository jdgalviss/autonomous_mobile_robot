#include <chrono>
#include <functional>
#include <memory>
#include <string>

#include "rclcpp/rclcpp.hpp"
#include <geometry_msgs/msg/pose_stamped.hpp>
#include "std_msgs/msg/bool.hpp"

using namespace std::chrono_literals;

/* This example creates a subclass of Node and uses std::bind() to register a
* member function as a callback from the timer. */

class TrajectoryPublisher : public rclcpp::Node
{
  public:
    TrajectoryPublisher()
    : Node("trajectory_publisher"), count_(0)
    {
      trajectory_publisher_ = this->create_publisher<geometry_msgs::msg::PoseStamped>("move_base_simple/goal", 10);
      
      timer_ = this->create_wall_timer(
      500ms, std::bind(&TrajectoryPublisher::timer_callback, this));
    }

  private:
    void timer_callback()
    {
      auto trajectory_message = geometry_msgs::msg::PoseStamped();      
      trajectory_message.pose.position.x = -20.3f;
      trajectory_message.pose.position.y = 5.0f;
      trajectory_message.header.stamp = rclcpp::Clock().now();
      trajectory_publisher_->publish(trajectory_message);
      RCLCPP_INFO(this->get_logger(), "Publishing: ");
    }
    rclcpp::TimerBase::SharedPtr timer_;
    rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr trajectory_publisher_;
    size_t count_;
  };

  int main(int argc, char * argv[])
  {
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<TrajectoryPublisher>());
    rclcpp::shutdown();
    return 0;
  }