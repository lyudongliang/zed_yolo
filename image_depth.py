"""
    This sample shows how to detect objects and draw 3D bounding boxes around them
    in an OpenGL window
"""
import sys
import ogl_viewer.viewer as gl
import pyzed.sl as sl
import numpy as np


# get rectangle region depth.
def get_rectangle_region_depth(pc:sl.Mat, x_center_index:int, y_center_index:int, rec_width:int, rec_height:int) ->float:
    remote_depth = -100000000
    half_width, half_height = round(rec_width / 2.), round(rec_height / 2.)
    depth_result_list = [pc.get_value(x_center_index - half_width + i, y_center_index - half_height + j)  for i in range(rec_width) for j in range(rec_height)]
    depth_list = [line[1][2] for line in depth_result_list if line[0] == sl.ERROR_CODE.SUCCESS and remote_depth <  line[1][2] < 0]
    
    rec_depth = -100000000
    if len(depth_list) > 0:
        depth_list = sorted(depth_list, key=lambda d: -d)[:100]
        rec_depth = np.mean(np.array(depth_list))
        # rec_depth = max(depth_list)
        
    return rec_depth


if __name__ == "__main__":
    print("start-------------")
    # Create a InitParameters object and set configuration parameters
    init_params = sl.InitParameters()
    init_params.camera_resolution = sl.RESOLUTION.HD720  # Use HD720 video mode
    init_params.coordinate_units = sl.UNIT.METER
    init_params.depth_mode = sl.DEPTH_MODE.ULTRA
    init_params.camera_fps = 15
    init_params.coordinate_system = sl.COORDINATE_SYSTEM.RIGHT_HANDED_Y_UP

    print("initialize camera")
    # Create a Camera object
    zed = sl.Camera()
    # Open the camera
    err = zed.open(init_params)
    if err != sl.ERROR_CODE.SUCCESS:
        exit(1)

    print("zed camera info")
    camera_info = zed.get_camera_information()
    # Create OpenGL viewer
    viewer = gl.GLViewer()
    viewer.init(camera_info.calibration_parameters.left_cam)
    print("init viewer")

    # Create ZED objects filled in the main loop
    objects = sl.Objects()
    image = sl.Mat()

    print("initialize point cloud.")
    # initialize point cloud.
    res = sl.Resolution()
    res.width = 1280
    res.height = 720
    point_cloud = sl.Mat(res.width, res.height, sl.MAT_TYPE.F32_C4, sl.MEM.CPU)

    print("poind_cloud", point_cloud.get_width())

    no_index  = 1
    while viewer.is_available():
        # Grab an image, a RuntimeParameters object must be given to grab()
        if zed.grab() == sl.ERROR_CODE.SUCCESS:
            # Retrieve left image
            zed.retrieve_image(image, sl.VIEW.LEFT)
            print("image width, image.height", image.get_width(), image.get_height(), image.get_channels())

            # retrieve point cloud
            zed.retrieve_measure(point_cloud, sl.MEASURE.XYZRGBA,sl.MEM.CPU, res)
            print("point cloud width, height", point_cloud.get_width(), point_cloud.get_height(), point_cloud.get_channels())

            # get point cloud value
            center_depth =  point_cloud.get_value(640, 360)
            print("center depth", center_depth)
            
            # get region depth
            region_depth = get_rectangle_region_depth(point_cloud, 640, 360, 50, 50)
            print("region depth", region_depth)

            # # Update GL view
            viewer.update_view(image, objects)

            print("no index", no_index)
            no_index += 1

    viewer.exit()

    image.free(memory_type=sl.MEM.CPU)
    # Disable modules and close camera
    zed.disable_object_detection()
    zed.disable_positional_tracking()

    zed.close()
