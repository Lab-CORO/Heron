// this class create a scene to curobo, check the collisions and launch the data generator. it save the scene to a file near the data generator

#include "../include/obstacle_adder.h"
#include "../include/curobo_node.h"

#include <sstream>

ObstacleAdder::ObstacleAdder() : Node("obstacle_adder")
{
    client_cb_add_obj = this->create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);
    client_cb_rm_obj = this->create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);
    server_scene_generator = this->create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);
    client_cb_obstacles = this->create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);
    client_cb_collision = this->create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);

    // Clients of the cuRobo node (parameter curobo_node, default unified_planner)
    const std::string curobo = curobo_node::declare(*this);
    client_add_obj = this->create_client<curobo_msgs::srv::AddObject>(curobo_node::service(curobo, "add_object"), rmw_qos_profile_services_default, client_cb_add_obj);
    client_get_obstacles = this->create_client<std_srvs::srv::Trigger>(curobo_node::service(curobo, "get_obstacles"), rmw_qos_profile_services_default, client_cb_obstacles);

    client_get_collision_dist = this->create_client<curobo_msgs::srv::GetCollisionDistance>(curobo_node::service(curobo, "get_collision_distance"), rmw_qos_profile_services_default, client_cb_collision);
    client_remove_obj = this->create_client<curobo_msgs::srv::RemoveObject>(curobo_node::service(curobo, "remove_object"), rmw_qos_profile_services_default, client_cb_rm_obj);

    // Attendre que le service soit disponible
    while (!client_get_collision_dist->wait_for_service(std::chrono::seconds(5)))
    {
        if (!rclcpp::ok())
        {
            RCLCPP_ERROR(this->get_logger(), "Interrompu pendant l'attente du service. Sortie.");
            return;
        }
        RCLCPP_INFO(this->get_logger(), "Services non disponible, nouvelle tentative...");
    }

    // Initialiser le générateur de nombres aléatoires avec une graine basée sur le temps
    std::random_device rd;
    rng_ = std::mt19937(rd());

    auto callback_scene_generator = [this](const std::shared_ptr<curobo_msgs::srv::SceneGenerator::Request> request,
                                           std::shared_ptr<curobo_msgs::srv::SceneGenerator::Response> response)
    {
        auto res = add_random_cubes(request->nb_object, request->max_reach);
        response->success = res;
        response->message = res ? "Scene generated" : "Scene generation failed (see obstacle_adder log)";
        return response;
    };

    service_ = this->create_service<curobo_msgs::srv::SceneGenerator>(
        "generate_scene",
        callback_scene_generator,
        rmw_qos_profile_services_default,
        server_scene_generator);
}

bool ObstacleAdder::add_cube(const std::string &name, double x, double y, double z, double dx, double dy, double dz)
{
    auto request = std::make_shared<curobo_msgs::srv::AddObject::Request>();
    request->type = 0; // CUBOID
    request->name = name;
    request->pose.position.x = x;
    request->pose.position.y = y;
    request->pose.position.z = z;
    request->pose.orientation.w = 1.0;
    request->dimensions.x = dx;
    request->dimensions.y = dy;
    request->dimensions.z = dz;
    request->color.r = 1.0;
    request->color.a = 1.0f;
    auto future = client_add_obj->async_send_request(request);
    if (future.wait_for(10s) != std::future_status::ready)
    {
        RCLCPP_ERROR(this->get_logger(), "add_object '%s': no answer", name.c_str());
        return false;
    }
    auto response = future.get();
    if (!response->success)
    {
        RCLCPP_ERROR(this->get_logger(), "add_object '%s' refused: %s", name.c_str(), response->message.c_str());
        return false;
    }
    return true;
}

bool ObstacleAdder::remove_object(const std::string &name)
{
    auto request = std::make_shared<curobo_msgs::srv::RemoveObject::Request>();
    request->name = name;
    auto future = client_remove_obj->async_send_request(request);
    if (future.wait_for(10s) != std::future_status::ready)
    {
        RCLCPP_ERROR(this->get_logger(), "remove_object '%s': no answer", name.c_str());
        return false;
    }
    auto response = future.get();
    if (!response->success)
    {
        RCLCPP_ERROR(this->get_logger(), "remove_object '%s' refused: %s", name.c_str(), response->message.c_str());
        return false;
    }
    return true;
}

std::optional<std::set<std::string>> ObstacleAdder::obstacle_names()
{
    auto future = client_get_obstacles->async_send_request(std::make_shared<std_srvs::srv::Trigger::Request>());
    if (future.wait_for(10s) != std::future_status::ready)
    {
        RCLCPP_ERROR(this->get_logger(), "get_obstacles: no answer");
        return std::nullopt;
    }
    std::set<std::string> names;
    std::istringstream lines(future.get()->message); // one name per line
    for (std::string line; std::getline(lines, line);)
    {
        if (!line.empty())
        {
            names.insert(line);
        }
    }
    return names;
}

/// Collision cost of the robot's first sphere (its base) against every obstacle. cuRobo v2
/// sums, per sphere, a cost over obstacles that is > 0 within collision_activation_distance
/// (0.05 m, set by the launch files) and 0 beyond.
std::optional<double> ObstacleAdder::base_collision_cost()
{
    auto future = client_get_collision_dist->async_send_request(
        std::make_shared<curobo_msgs::srv::GetCollisionDistance::Request>());
    if (future.wait_for(10s) != std::future_status::ready)
    {
        RCLCPP_ERROR(this->get_logger(), "get_collision_distance: no answer");
        return std::nullopt;
    }
    auto response = future.get();
    if (response->data.empty())
    {
        // curobo_ros answers an empty list when it has no solver to query
        RCLCPP_ERROR(this->get_logger(), "get_collision_distance returned no sphere");
        return std::nullopt;
    }
    return response->data[0];
}

/// Back to the Station alone: remove this node's cubes, then anything else that is not one
/// of the Station's obstacles. False if the scene still differs.
bool ObstacleAdder::reset_to_station()
{
    for (const auto &name : own_cubes_)
    {
        remove_object(name); // checked below, whatever happened
    }
    own_cubes_.clear();
    auto names = obstacle_names();
    if (!names)
    {
        return false;
    }
    for (const auto &name : *names)
    {
        if (!station_names_.count(name))
        {
            RCLCPP_WARN(this->get_logger(), "'%s' left over from the previous Scene, removing it", name.c_str());
            remove_object(name);
        }
    }
    names = obstacle_names();
    if (!names || *names != station_names_)
    {
        RCLCPP_ERROR(this->get_logger(), "the scene is not the Station's obstacles alone, giving up on this Scene");
        return false;
    }
    return true;
}

bool ObstacleAdder::add_random_cubes(int nb_object, float max_reach)
{
    if (!station_known_)
    {
        // First Scene: nothing added yet, so what curobo_ros holds is the Station (world file).
        auto names = obstacle_names();
        if (!names)
        {
            return false;
        }
        station_names_ = *names;
        station_known_ = true;
        std::string list;
        for (const auto &n : station_names_)
        {
            list += " " + n;
        }
        RCLCPP_INFO(this->get_logger(), "Station obstacles kept in every Scene (%zu):%s", station_names_.size(), list.c_str());
    }
    if (!reset_to_station())
    {
        return false;
    }
    if (nb_object <= 0)
    {
        RCLCPP_INFO(this->get_logger(), "scene: no cube (obj_max=0), the Station alone");
        return true;
    }

    // A cube is refused when it brings an obstacle within 5 cm of the robot's base: the
    // base's collision cost rises above its value with the Station alone.
    auto baseline = base_collision_cost();
    if (!baseline)
    {
        return false;
    }
    if (*baseline > 0.0 && !baseline_warned_)
    {
        RCLCPP_WARN(this->get_logger(),
                    "the Station's obstacles are already within 5 cm of the robot base (cost %g): "
                    "cubes are judged on the increase of that cost", *baseline);
        baseline_warned_ = true;
    }

    this->num_cubes_dist_ = std::uniform_int_distribution<int>(std::min(3, nb_object), nb_object);
    this->position_dist_ = std::uniform_real_distribution<double>(-max_reach, max_reach);
    this->size_dist_ = std::uniform_real_distribution<double>(0.05, 0.3);
    const int num_cubes = this->num_cubes_dist_(rng_);
    int refused = 0;

    for (int i = 0; i < num_cubes; ++i)
    {
        const std::string name = "cube_" + std::to_string(i);
        bool accepted = false;
        for (int attempt = 1; !accepted; ++attempt)
        {
            // Bounded: a check that never succeeds must end the Scene, not spin forever.
            if (attempt > kMaxAttemptsPerCube)
            {
                RCLCPP_ERROR(this->get_logger(), "%s: no free position after %d attempts, giving up on this Scene",
                             name.c_str(), kMaxAttemptsPerCube);
                return false;
            }
            const double x = this->position_dist_(rng_);
            const double y = this->position_dist_(rng_);
            const double z = this->position_dist_(rng_);
            const double dx = this->size_dist_(rng_);
            const double dy = this->size_dist_(rng_);
            const double dz = this->size_dist_(rng_);
            // A cube curobo_ros does not take would leave the Scene one cube short.
            if (!add_cube(name, x, y, z, dx, dy, dz))
            {
                return false;
            }
            own_cubes_.push_back(name);
            auto cost = base_collision_cost();
            if (!cost)
            {
                return false;
            }
            accepted = *cost <= *baseline + 1e-6;
            if (!accepted)
            {
                ++refused;
                own_cubes_.pop_back();
                if (!remove_object(name))
                {
                    return false;
                }
            }
        }
    }
    RCLCPP_INFO(this->get_logger(), "scene: %d cubes placed, %d draws refused (closer than 5 cm to the robot base)",
                num_cubes, refused);
    return true;
}

int main(int argc, char **argv)
{
    // Initialisation de ROS2
    rclcpp::init(argc, argv);

    // Création du nœud
    auto obstacle_adder_node = std::make_shared<ObstacleAdder>();
    if (!curobo_node::wait_until_ready(obstacle_adder_node, obstacle_adder_node->get_parameter("curobo_node").as_string()))
    {
        return 1;
    }

    // Création de l'executor
    rclcpp::executors::MultiThreadedExecutor executor;
    executor.add_node(obstacle_adder_node);

    RCLCPP_INFO(obstacle_adder_node->get_logger(), "Starting client node, shut down with CTRL-C");
    executor.spin();
    RCLCPP_INFO(obstacle_adder_node->get_logger(), "Keyboard interrupt, shutting down.\n");

    rclcpp::shutdown();
    return 0;
}
