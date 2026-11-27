#ifndef OBSTACLE_ADDER_HPP
#define OBSTACLE_ADDER_HPP

#include "rclcpp/rclcpp.hpp"
#include "curobo_msgs/srv/add_object.hpp"
#include "curobo_msgs/srv/get_collision_distance.hpp"
#include "curobo_msgs/srv/remove_object.hpp"
#include "curobo_msgs/srv/scene_generator.hpp"
#include "geometry_msgs/msg/pose.hpp"
#include "geometry_msgs/msg/vector3.hpp"
#include "std_msgs/msg/color_rgba.hpp"
#include "std_srvs/srv/trigger.hpp"

#include <optional>
#include <set>
#include <string>
#include <vector>
using namespace std;

#include <random>

class ObstacleAdder : public rclcpp::Node
{
public:
    ObstacleAdder();
    bool add_random_cubes(int nb_object, float max_reach);

private:
    rclcpp::Node::SharedPtr node_;
    // DataGeneration data_generation_;
    rclcpp::Client<curobo_msgs::srv::AddObject>::SharedPtr client_add_obj;
    rclcpp::Client<std_srvs::srv::Trigger>::SharedPtr client_get_obstacles;
    rclcpp::Client<curobo_msgs::srv::RemoveObject>::SharedPtr client_remove_obj;
    rclcpp::Client<curobo_msgs::srv::GetCollisionDistance>::SharedPtr client_get_collision_dist;
    rclcpp::Service<curobo_msgs::srv::SceneGenerator>::SharedPtr service_;

    rclcpp::CallbackGroup::SharedPtr client_cb_add_obj, client_cb_rm_obj, client_cb_obstacles, client_cb_collision, server_scene_generator;
    std::mt19937 rng_; // Générateur de nombres aléatoires
    std::uniform_int_distribution<int> num_cubes_dist_; // Distribution pour le nombre de cubes
    std::uniform_real_distribution<double> position_dist_; // Distribution pour les positions
    std::uniform_real_distribution<double> size_dist_; // Distribution pour les dimensions

    

    static constexpr int kMaxAttemptsPerCube = 1000;

    // The Station's obstacles (world file, loaded by curobo_ros), as found before the first
    // Scene; every Scene starts from exactly them.
    std::set<std::string> station_names_;
    bool station_known_ = false;
    bool baseline_warned_ = false;
    std::vector<std::string> own_cubes_;   // cubes this node added to the current Scene

    bool add_cube(const std::string &name, double x, double y, double z, double dx, double dy, double dz);
    bool remove_object(const std::string &name);
    std::optional<std::set<std::string>> obstacle_names();
    std::optional<double> base_collision_cost();
    bool reset_to_station();
};

#endif // OBSTACLE_ADDER_HPP
