#pragma once

// Names of the cuRobo services data_generation calls.
//
// curobo_ros (ROS 2 Jazzy, cuRobo v2) serves everything from one node,
// `curobo_trajectory_planner`, under its node name (`unified_planner` by default),
// e.g. /unified_planner/ik_batch. Every node that calls cuRobo declares the
// `curobo_node` parameter and builds its service names with curobo_service().

#include <string>

#include "rclcpp/rclcpp.hpp"

namespace curobo_node
{
inline const char *kDefaultNode = "unified_planner";

/// Declare (if needed) and read the `curobo_node` parameter of `node`.
inline std::string declare(rclcpp::Node &node)
{
    if (!node.has_parameter("curobo_node"))
    {
        node.declare_parameter<std::string>("curobo_node", kDefaultNode);
    }
    std::string name = node.get_parameter("curobo_node").as_string();
    while (!name.empty() && name.front() == '/')
    {
        name.erase(0, 1);
    }
    return name.empty() ? std::string(kDefaultNode) : name;
}

/// "/<curobo node>/<service>"
inline std::string service(const std::string &curobo_node, const std::string &srv)
{
    return "/" + curobo_node + "/" + srv;
}

/// Block until the cuRobo node has finished its warmup: its services exist from the start,
/// but it only answers once its `node_is_available` parameter is true. Call before `node`
/// is added to an executor. The parameter client runs on a throwaway node of its own:
/// spinning `node` here would tie its callback groups to a temporary executor, and the
/// replies to its service clients would then never be processed.
inline bool wait_until_ready(const rclcpp::Node::SharedPtr &node, const std::string &curobo_node)
{
    using namespace std::chrono_literals;
    // use_global_arguments(false): launch's name remap (__node:=...) would otherwise rename the
    // probe to `node`'s own name.
    auto probe = rclcpp::Node::make_shared(std::string(node->get_name()) + "_wait_for_curobo",
                                           rclcpp::NodeOptions().use_global_arguments(false));
    auto params = std::make_shared<rclcpp::SyncParametersClient>(probe, curobo_node);
    while (rclcpp::ok())
    {
        if (params->wait_for_service(5s))
        {
            auto values = params->get_parameters({"node_is_available"}, 5s);
            if (!values.empty() && values[0].get_type() == rclcpp::ParameterType::PARAMETER_BOOL &&
                values[0].as_bool())
            {
                RCLCPP_INFO(node->get_logger(), "%s is ready", curobo_node.c_str());
                return true;
            }
        }
        RCLCPP_INFO(node->get_logger(), "waiting for %s to finish its warmup...", curobo_node.c_str());
    }
    return false;
}
} // namespace curobo_node
