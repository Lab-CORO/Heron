#include <chrono>
#include <memory>
#include "rclcpp/rclcpp.hpp"
#include "std_srvs/srv/empty.hpp"
#include <iostream>
#include <math.h>
#include <bits/stdc++.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <fstream>
#include "../include/utils.h"
#include "curobo_msgs/srv/generate_rm.hpp"
#include <curobo_msgs/srv/get_voxel_grid.hpp>
#include "curobo_msgs/srv/ik_batch.hpp"
#include "curobo_msgs/srv/warmup_ik.hpp"
#include "../include/curobo_node.h"

#include <ament_index_cpp/get_package_share_directory.hpp>
#include <ctime>

#include <highfive/H5File.hpp>

#include <boost/multi_array.hpp>
#include <highfive/highfive.hpp>

#include <iostream>
using namespace std;
using namespace HighFive;

using namespace std::chrono_literals;

namespace cb_data_generator
{
    class DataGenerator : public rclcpp::Node
    {
    public:
        DataGenerator() : Node("client_node")
        {
            // create h45 file name from time
            auto t = std::time(nullptr);
            auto tm = *std::localtime(&t);
            std::ostringstream oss;
            oss << std::put_time(&tm, "%d_%m_%Y_%H_%M_%S");
            auto date_str = oss.str();
            this->data_file_path = ament_index_cpp::get_package_share_directory("data_generation") + "/data/" + date_str + ".h5";

            client_cb_group_ = this->create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);
            service_cb_group_ = this->create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);
            client_voxel_cb_group_ = this->create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);
            this->dataset_id = 0;
            // Get resolution from ros param
            this->voxel_size = 0;
            this->declare_parameter("voxel_size", 0.5);
            this->get_parameter("voxel_size", voxel_size);
            this->voxel_size = voxel_size;
            // Half side of the Scene cube (m), from the Station (scene_half_extent).
            // The voxel grid must hold ceil(2 * scene_half_extent / voxel_size)
            // voxels per side; curobo_ros sizes it from its own mapper_extent_xyz
            // parameter, which the launch files set from the same value.
            this->declare_parameter("scene_half_extent", 1.52);
            this->get_parameter("scene_half_extent", this->scene_half_extent);
            this->curobo_node_ = curobo_node::declare(*this);
            std::stringstream resolution_string;
            resolution_string << voxel_size;              // appending the float value to the streamclass
            // Load the reachablity map from the resolution
            utils::load_poses_from_file(ament_index_cpp::get_package_share_directory("data_generation") + "/data" + "/master_ik_data" + resolution_string.str() + ".npz", this->raw_datas);
            for (const auto& data_pt : this->raw_datas) {
                // on prend le risque de la dedondance mais au piure on fera un round sur les coords x y z
                utils::QuantizedPoint3D pt(data_pt.position.x, data_pt.position.y, data_pt.position.z, this->voxel_size);
                this->map_rm[pt] = 0.0;
            }

            auto callback_generate_rm = [this](const std::shared_ptr<curobo_msgs::srv::GenerateRM::Request> request,
                                               std::shared_ptr<curobo_msgs::srv::GenerateRM::Response> response)
            {
                auto res = data_generation(request->batch_size, request->resolution);
                response->success = res;
                response->message = res ? "Reachability map generated successfully." : "Reachability map generation failed (see generate_data log).";

                return response;
            };

            this->client_ik = this->create_client<curobo_msgs::srv::IkBatch>(
                curobo_node::service(curobo_node_, "ik_batch"), rmw_qos_profile_services_default, client_cb_group_);
            this->client_warmup_ik = this->create_client<curobo_msgs::srv::WarmupIK>(
                curobo_node::service(curobo_node_, "warmup_ik"), rmw_qos_profile_services_default, client_cb_group_);
            client_voxel_grid = this->create_client<curobo_msgs::srv::GetVoxelGrid>(
                curobo_node::service(curobo_node_, "get_voxel_grid"), rmw_qos_profile_services_default, client_cb_group_);
            while (!this->client_ik->wait_for_service(std::chrono::seconds(5)))
            {
                if (!rclcpp::ok()) {
                    RCLCPP_ERROR(this->get_logger(), "client interrupted while waiting for service to appear.");
                return ;
                }
                RCLCPP_WARN(this->get_logger(), "%s is not available, retrying...", client_ik->get_service_name());
            }
            RCLCPP_INFO(this->get_logger(), "%s is now available.", client_ik->get_service_name());

            service_ = this->create_service<curobo_msgs::srv::GenerateRM>(
                "generate_rm",
                callback_generate_rm,
                rmw_qos_profile_services_default,
                service_cb_group_);
        }

        ~DataGenerator()
        {
            if (this->data_file_) // never opened when no Reachability map was asked for
            {
                this->data_file_->flush();
            }
            RCLCPP_INFO(this->get_logger(), "DataGenerator is shutting down.");
        }
        
        bool load_reachability_map(float voxel_size)
        {
            return true;
        }

        /**
         * @brief Make sure the cuRobo v2 IK solver exists for batches of `batch_size` poses.
         *
         * curobo_ros v2 creates its IK solver only on warmup_ik, and recreates it whenever a
         * batch of another size arrives (see ik_services.py). Warm it up once here, and pad
         * every batch to `batch_size` (data_generation()), so the solver is built once per run.
         */
        bool ensure_ik_ready(int batch_size)
        {
            if (this->ik_batch_size_ == batch_size)
            {
                return true;
            }
            if (!client_warmup_ik->wait_for_service(std::chrono::seconds(30)))
            {
                RCLCPP_ERROR(this->get_logger(), "%s not available", client_warmup_ik->get_service_name());
                return false;
            }
            auto request = std::make_shared<curobo_msgs::srv::WarmupIK::Request>();
            request->batch_size = batch_size;
            RCLCPP_INFO(this->get_logger(), "Warming up the IK solver for batches of %d poses...", batch_size);
            auto future = client_warmup_ik->async_send_request(request);
            // Building the solver compiles its kernels on the first run: allow minutes.
            if (future.wait_for(600s) != std::future_status::ready)
            {
                RCLCPP_ERROR(this->get_logger(), "IK warmup timed out");
                return false;
            }
            auto res = future.get();
            if (!res->success)
            {
                RCLCPP_ERROR(this->get_logger(), "IK warmup failed: %s", res->message.c_str());
                return false;
            }
            this->ik_batch_size_ = batch_size;
            return true;
        }

        bool data_generation(int batch_size, float resolution)
        {
            this->data_file_ = std::make_shared<HighFive::File>(
                this->data_file_path,
                HighFive::File::ReadWrite | HighFive::File::Create);
            auto start_time = std::chrono::high_resolution_clock::now();

            std::vector<geometry_msgs::msg::Pose> data;

            // split data into batches
            std::vector<std::vector<geometry_msgs::msg::Pose>> batches;

            utils::split_data(this->raw_datas, batch_size, batches);
            if (!ensure_ik_ready(batch_size))
            {
                return false;
            }
            // print batch size
            RCLCPP_WARN(this->get_logger(), "Number of Batch: %i", batches.size());
            RCLCPP_WARN(this->get_logger(), "Data size: %i", this->raw_datas.size());

            // Befor start use a clone of this->map_rm
            auto local_map_rm = this->map_rm;
            // iterate all batches

            for (const auto &batch : batches)
            {
                // send the batch to robot ik
                std::vector<sensor_msgs::msg::JointState> joint_states;
                std::vector<std_msgs::msg::Bool> joint_states_valid;

                auto request = std::make_shared<curobo_msgs::srv::IkBatch::Request>();

                request->poses = batch;
                // Pad the last, shorter batch with copies of its last pose: a batch of another
                // size would make curobo_ros rebuild its IK solver. The padding is ignored below.
                while ((int)request->poses.size() < batch_size)
                {
                    request->poses.push_back(batch.back());
                }
                auto result_future = this->client_ik->async_send_request(request);

                std::future_status status = result_future.wait_for(120s); // timeout to guarantee a graceful finish
                if (status == std::future_status::ready)
                {
                    auto res = result_future.get();
                    if (!res->success)
                    {
                        RCLCPP_ERROR(this->get_logger(), "Ik batch failed: %s", res->error_msg.data.c_str());
                        return false;
                    }
                    if (res->joint_states_valid.size() < batch.size())
                    {
                        RCLCPP_ERROR(this->get_logger(), "Ik batch returned %zu results for %zu poses",
                                     res->joint_states_valid.size(), batch.size());
                        return false;
                    }
                    joint_states = res->joint_states;
                    joint_states_valid = res->joint_states_valid;
                }
                else
                {
                    RCLCPP_ERROR(this->get_logger(), "Service call failed");
                    return false;
                }

                for (size_t i = 0; i < batch.size(); ++i)
                {
                    // get the x, y, z and if joint state valide, add 1 to score 
                    utils::QuantizedPoint3D pt(batch[i].position.x, batch[i].position.y, batch[i].position.z, resolution);
                    
                    if (joint_states_valid[i].data)
                    {
                        local_map_rm[pt] += 1.0 / 50.0;
                    }
                 
                }
            }

            // End the timer
            auto end_time = std::chrono::high_resolution_clock::now();

            // Calculate the elapsed time in milliseconds
            auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();
            RCLCPP_INFO(this->get_logger(), "RM generation time: %ld ms", duration);

            std::vector<std::array<double, 4>> voxel_map = {};
            int voxel_grid_sizes[3];
            double voxel_grid_origin[3];

            start_time = std::chrono::high_resolution_clock::now();

            if (!this->get_voxel_map(voxel_map, voxel_grid_sizes, voxel_grid_origin))
            {
                return false;
            }

            // End the timer
            end_time = std::chrono::high_resolution_clock::now();
            // Calculate the elapsed time in milliseconds
            duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();
            RCLCPP_INFO(this->get_logger(), "Voxel map gen time: %ld ms", duration);

            start_time = std::chrono::high_resolution_clock::now();

            utils::saveToHDF5(local_map_rm, voxel_map, resolution, voxel_grid_sizes, voxel_grid_origin, this->data_file_, this->dataset_id);

            this->dataset_id += 1;
            // End the timer
            end_time = std::chrono::high_resolution_clock::now();
            // Calculate the elapsed time in milliseconds
            duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();
            RCLCPP_INFO(this->get_logger(), "Save hdf5 time: %ld ms", duration);

            RCLCPP_INFO(this->get_logger(), "Reachability generated !");
            
            this->data_file_->flush();

            return true;
        }

        /**
         * @brief Generate the voxel grid. Here we generate the voxel grid to save it. For AI porpuse, an occupied voxel is 0 and a free voxel is 1.
         */

        bool get_voxel_map(std::vector<std::array<double, 4>> &voxel_grid, int (&voxel_grid_sizes)[3], double (&voxel_grid_origin)[3])
        {
            // Wait for the service to be available
            if (!client_voxel_grid->wait_for_service(std::chrono::seconds(5)))
            {
                RCLCPP_ERROR(this->get_logger(), "%s not available", client_voxel_grid->get_service_name());
                return false;
            }

            // The Scene cube, in the robot base frame. curobo_ros 06aef20 does not read this box
            // yet (it sizes the grid from its mapper_extent_xyz parameter); it is filled so the
            // request means the same thing once it does, and the size is checked below either way.
            auto request = std::make_shared<curobo_msgs::srv::GetVoxelGrid::Request>();
            const double h = this->scene_half_extent;
            request->bbox_min_x = -h;
            request->bbox_min_y = -h;
            request->bbox_min_z = -h;
            request->bbox_max_x = h;
            request->bbox_max_y = h;
            request->bbox_max_z = h;
            // Call the service asynchronously
            auto result_future = client_voxel_grid->async_send_request(request);
            std::future_status status = result_future.wait_for(120s); // timeout to guarantee a graceful finish
            if (status == std::future_status::ready)
            {
                auto response = result_future.get();
                // ceil, as curobo_ros (and the Humble patch) size the grid; the margin absorbs
                // float error (2 * 1.52 / 0.02f = 152.0000034 -> 152, not 153).
                const long expected = (long)std::ceil(2.0 * h / this->voxel_size - 1e-3);
                const auto &g = response->voxel_grid;
                if ((long)g.size_x != expected || (long)g.size_y != expected || (long)g.size_z != expected ||
                    g.data.size() != (size_t)g.size_x * g.size_y * g.size_z)
                {
                    RCLCPP_ERROR(this->get_logger(),
                                 "voxel grid is %ux%ux%u (%zu values), expected %ld^3 for scene_half_extent=%.3f m "
                                 "at voxel_size=%.3f m: set mapper_extent_xyz of %s to 2 * scene_half_extent",
                                 g.size_x, g.size_y, g.size_z, g.data.size(), expected, h, this->voxel_size,
                                 this->curobo_node_.c_str());
                    return false;
                }

                // Iterate through the voxel grid data and add points for occupied voxels
                int index = 0;
                for (int x = 0; x < (int)response->voxel_grid.size_x; ++x)
                {
                    for (int y = 0; y < (int)response->voxel_grid.size_y; ++y)
                    {
                        for (int z = 0; z < (int)response->voxel_grid.size_z; ++z)
                        {
                            voxel_grid.push_back(std::array<double, 4>{response->voxel_grid.origin.x + x * response->voxel_grid.resolutions.x,
                                                                       response->voxel_grid.origin.y + y * response->voxel_grid.resolutions.y,
                                                                       response->voxel_grid.origin.z + z * response->voxel_grid.resolutions.z,
                                                                       (double)1 - response->voxel_grid.data[index]});
                            ++index;
                        }
                    }
                }
                voxel_grid_sizes[0] = (int)response->voxel_grid.size_x;
                RCLCPP_WARN(this->get_logger(), "voxel size: %d", response->voxel_grid.size_x);
                voxel_grid_sizes[1] = (int)response->voxel_grid.size_y;
                voxel_grid_sizes[2] = (int)response->voxel_grid.size_z;

                voxel_grid_origin[0] = response->voxel_grid.origin.x;
                voxel_grid_origin[1] = response->voxel_grid.origin.y;
                voxel_grid_origin[2] = response->voxel_grid.origin.z;
            }
            else
            {
                RCLCPP_ERROR(this->get_logger(), "Failed to call %s.", client_voxel_grid->get_service_name());
                return false;
            }
            return true;
        }


    private:
        rclcpp::CallbackGroup::SharedPtr client_cb_group_;
        rclcpp::CallbackGroup::SharedPtr service_cb_group_;
        rclcpp::CallbackGroup::SharedPtr client_voxel_cb_group_;

        rclcpp::Client<std_srvs::srv::Empty>::SharedPtr client_ptr_;
        rclcpp::TimerBase::SharedPtr timer_ptr_;
        rclcpp::Service<curobo_msgs::srv::GenerateRM>::SharedPtr service_;
        rclcpp::Client<curobo_msgs::srv::IkBatch>::SharedPtr client_ik;
        rclcpp::Client<curobo_msgs::srv::GetVoxelGrid>::SharedPtr client_voxel_grid;
        rclcpp::Client<curobo_msgs::srv::WarmupIK>::SharedPtr client_warmup_ik;
        std::string curobo_node_;
        double scene_half_extent = 1.52;
        int ik_batch_size_ = 0; // batch size the cuRobo IK solver is warmed up for (0 = not yet)

        std::shared_ptr<HighFive::File> data_file_;
        std::string data_file_path;
        int dataset_id;
        float voxel_size;
        std::map<utils::QuantizedPoint3D, double> map_rm;
        std::vector<geometry_msgs::msg::Pose>raw_datas;
    };
}

int main(int argc, char *argv[])
{
    rclcpp::init(argc, argv);
    auto client_node = std::make_shared<cb_data_generator::DataGenerator>();
    if (!curobo_node::wait_until_ready(client_node, client_node->get_parameter("curobo_node").as_string()))
    {
        return 1;
    }
    rclcpp::executors::MultiThreadedExecutor executor;
    executor.add_node(client_node);

    RCLCPP_INFO(client_node->get_logger(), "Starting client node, shut down with CTRL-C");
    executor.spin();
    RCLCPP_INFO(client_node->get_logger(), "Keyboard interrupt, shutting down.\n");

    rclcpp::shutdown();
    return 0;
}
