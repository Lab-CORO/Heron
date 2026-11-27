from functools import partial
from std_srvs.srv import Trigger
from curobo_msgs.srv import AddObject, RemoveObject, GetVoxelGrid, GetCollisionDistance, SetCollisionCache
from visualization_msgs.msg import MarkerArray, Marker


class RosServiceManager:
    """
    Manages all ROS services, publishers, and timers.
    Responsible for:
    - Creating ROS services for obstacle management
    - Publishing visualization markers (collision spheres)
    - Managing timers for periodic publishing
    - Delegating service callbacks to appropriate managers
    """

    def __init__(self, node, obstacle_manager, robot_model_manager, config_manager, config_wrapper=None):
        """
        Initialize ROS service manager.

        Args:
            node: ROS2 node instance
            obstacle_manager: ObstacleManager for obstacle operations
            robot_model_manager: RobotModelManager for robot collision spheres
            config_manager: ConfigManager for base_link access
            config_wrapper: ConfigWrapper parent for calling update_world_config()
        """
        self.node = node
        self.obstacle_manager = obstacle_manager
        self.robot_model_manager = robot_model_manager
        self.config_manager = config_manager
        self.config_wrapper = config_wrapper

        # Services (initialized in init_services)
        self.add_object_srv = None
        self.remove_object_srv = None
        self.get_obstacles_srv = None
        self.node_available_srv = None
        self.remove_all_objects_srv = None
        self.get_voxel_map_srv = None
        self.get_collision_distance_srv = None
        self.set_collision_cache_srv = None

        # Publisher for collision spheres visualization
        self.publish_collision_spheres_pub = None
        self.publish_collision_spheres_timer = None

    def init_services(self):
        """Create all ROS services, publishers, and timers"""
        # Create services
        self.add_object_srv = self.node.create_service(
            AddObject,
            self.node.get_name() + '/add_object',
            partial(self._callback_add_object, self.node)
        )

        self.remove_object_srv = self.node.create_service(
            RemoveObject,
            self.node.get_name() + '/remove_object',
            partial(self._callback_remove_object, self.node)
        )

        self.get_obstacles_srv = self.node.create_service(
            Trigger,
            self.node.get_name() + '/get_obstacles',
            partial(self._callback_get_obstacles, self.node)
        )

        self.node_available_srv = self.node.create_service(
            Trigger,
            self.node.get_name() + '/is_available',
            partial(self._callback_is_available, self.node)
        )

        self.remove_all_objects_srv = self.node.create_service(
            Trigger,
            self.node.get_name() + '/remove_all_objects',
            partial(self._callback_remove_all_objects, self.node)
        )

        self.get_voxel_map_srv = self.node.create_service(
            GetVoxelGrid,
            self.node.get_name() + '/get_voxel_grid',
            partial(self._callback_get_voxel_grid, self.node)
        )

        self.get_collision_distance_srv = self.node.create_service(
            GetCollisionDistance,
            self.node.get_name() + '/get_collision_distance',
            partial(self._callback_get_collision_distance, self.node)
        )

        self.set_collision_cache_srv = self.node.create_service(
            SetCollisionCache,
            self.node.get_name() + '/set_collision_cache',
            partial(self._callback_set_collision_cache, self.node)
        )

        # CapaciNet 'scene' pipeline mode: load a dense ESDF scene from
        # $VOXEL_SCENE_PATH (written by pipeline/load_scene.py) as a native
        # curobo VoxelGrid obstacle, bypassing cuboid decomposition entirely.
        self.load_voxel_scene_srv = self.node.create_service(
            Trigger,
            self.node.get_name() + '/load_voxel_scene',
            partial(self._callback_load_voxel_scene, self.node)
        )

        # Create publisher for collision spheres
        self.publish_collision_spheres_pub = self.node.create_publisher(
            MarkerArray,
            self.node.get_name() + '/collision_spheres',
            1
        )

        # Create timer for periodic collision sphere publishing
        self.publish_collision_spheres_timer = self.node.create_timer(
            0.5,
            partial(self.publish_collision_spheres, self.node)
        )

    def _callback_add_object(self, node, request: AddObject, response):
        """Delegate add_object service to ObstacleManager and update world config"""
        response = self.obstacle_manager.add_object(node, request, response)

        # Update world configuration to propagate obstacle to cuRobo world_model
        if response.success and self.config_wrapper is not None:
            self.config_wrapper.update_world_config(node)
            
        return response

    def _callback_remove_object(self, node, request: RemoveObject, response):
        """Delegate remove_object service to ObstacleManager and update world config"""
        response = self.obstacle_manager.remove_object(node, request, response)

        # Update world configuration to propagate changes to cuRobo world_model
        if response.success and self.config_wrapper is not None:
            self.config_wrapper.update_world_config(node)

        return response

    def _callback_get_obstacles(self, node, request: Trigger, response):
        """Delegate get_obstacles service to ObstacleManager"""
        return self.obstacle_manager.get_obstacles(node, request, response)

    def _callback_is_available(self, node, request: Trigger, response):
        """Return node availability status"""
        response.success = self.node.node_is_available
        return response

    def _callback_remove_all_objects(self, node, request: Trigger, response):
        """Delegate remove_all_objects service to ObstacleManager and update world config"""
        response = self.obstacle_manager.remove_all_objects(node, request, response)

        # Update world configuration to propagate changes to cuRobo world_model
        if response.success and self.config_wrapper is not None:
            self.config_wrapper.update_world_config(node)

        return response

    def _callback_get_voxel_grid(self, node, request: GetVoxelGrid, response):
        """Delegate get_voxel_grid service to ObstacleManager"""
        return self.obstacle_manager.get_voxel_grid(node, request, response)

    def _callback_get_collision_distance(self, node, request: GetCollisionDistance, response):
        """Delegate get_collision_distance service to ConfigWrapper"""
        return self.config_wrapper.callback_get_collision_distance(node, request, response)

    def _callback_set_collision_cache(self, node, request: SetCollisionCache, response):
        """Delegate set_collision_cache service to ObstacleManager"""
        return self.obstacle_manager.set_collision_cache(node, request, response)

    def _callback_load_voxel_scene(self, node, request: Trigger, response):
        """
        Load a scene ESDF (see ObstacleManager.load_voxel_scene) and make it take
        effect in the live IK solver.

        Unlike add_object/remove_object/remove_all_objects, a plain
        config_wrapper.update_world_config(node) is NOT enough here: that call reuses
        the currently-built IKSolver as-is (see ConfigWrapperIK.update_world_config in
        config_wrapper_motion.py), and the checker type (BLOX by default) is fixed at
        IKSolver construction time. So after staging the VoxelGrid we explicitly rebuild
        the solver with the VOXEL checker (set_ik_gen_config -- the same method the
        existing '/update_motion_gen_config' service exposes) and then push the ESDF
        values in, since load_collision_model() only registers voxel grid geometry, not
        its feature data (see curobo.geom.sdf.world_voxel.WorldVoxelCollision).
        """
        response = self.obstacle_manager.load_voxel_scene(node, request, response)
        if not response.success or self.config_wrapper is None:
            return response

        try:
            self.config_wrapper.set_ik_gen_config(node, None, None)

            # ik_init() is NOT optional here. ik.py's get_ik() builds its goal Pose
            # from plain `torch.tensor(...)`, i.e. CPU tensors, and cuRobo asserts
            # `goal_position.is_cuda()`. That only works because the warmup solve in
            # ik_init() first populates the solver's internal goal buffer with CUDA
            # tensors, which later CPU calls copy into. A freshly built solver has no
            # such buffer, so the very first ik_batch_poses would die with
            # "goal_position must be a CUDA tensor" (seen as "Ik batch failed", the
            # real exception swallowed by get_ik's bare `except:`).
            # Everywhere else in curobo_ros, set_ik_gen_config is likewise always
            # immediately followed by ik_init() -- keep that pairing.
            node.ik_init()

            # Features last: load_collision_model (inside the rebuild) registers voxel
            # grid geometry but leaves feature values at their default, so the real
            # ESDF has to be pushed in after the solver exists.
            vg = self.obstacle_manager.get_scene_voxel_grid()
            node.ik_solver.world_coll_checker.update_voxel_data(vg, env_idx=0)
        except Exception as e:
            response.success = False
            response.message = f"Voxel scene staged but failed to apply to IK solver: {e}"
            return response

        return response

    def publish_collision_spheres(self, node):
        """
        Publishes the robot's collision spheres as markers for visualization in RViz.
        Useful for debugging and ensuring proper masking of the robot in the point cloud.
        """
        # Get collision spheres from robot model manager
        robot_spheres = self.robot_model_manager.get_collision_spheres()

        # Create marker array
        marker_array = MarkerArray()

        for i, sphere in enumerate(robot_spheres):
            marker = Marker()
            marker.header.frame_id = self.config_manager.base_link
            marker.type = Marker.SPHERE
            marker.action = Marker.ADD
            marker.id = i
            marker.pose.position.x = sphere[0]
            marker.pose.position.y = sphere[1]
            marker.pose.position.z = sphere[2]
            marker.scale.x = sphere[3] * 2  # Diameter
            marker.scale.y = sphere[3] * 2
            marker.scale.z = sphere[3] * 2
            marker.color.a = 0.5  # Transparency
            marker.color.r = 1.0
            marker.color.g = 0.0
            marker.color.b = 0.0
            marker_array.markers.append(marker)

        # Publish marker array
        self.publish_collision_spheres_pub.publish(marker_array)
