import argparse
import os
import glob
import random
# import darknet
import time
import cv2
import numpy as np
import pyzed.sl as sl
import ogl_viewer.viewer as gl
from utils import get_pos_2d, get_rectangle_region_depth


def parser():
    parser = argparse.ArgumentParser(description="YOLO Object Detection")
    parser.add_argument("--batch_size", default=1, type=int,
                        help="number of images to be processed at the same time")
    parser.add_argument("--weights", default="yolov4.weights",
                        help="yolo weights path")
    parser.add_argument("--dont_show", action='store_true',
                        help="windown inference display. For headless systems")
    parser.add_argument("--ext_output", action='store_true',
                        help="display bbox coordinates of detected objects")
    parser.add_argument("--config_file", default="./cfg/yolov4.cfg",
                        help="path to config file")
    parser.add_argument("--data_file", default="./cfg/coco.data",
                        help="path to data file")
    parser.add_argument("--thresh", type=float, default=.25,
                        help="remove detections with lower confidence")
    return parser.parse_args()


def check_arguments_errors(args):
    assert 0 < args.thresh < 1, "Threshold should be a float between zero and one (non-inclusive)"
    if not os.path.exists(args.config_file):
        raise(ValueError("Invalid config path {}".format(os.path.abspath(args.config_file))))
    if not os.path.exists(args.weights):
        raise(ValueError("Invalid weight path {}".format(os.path.abspath(args.weights))))
    if not os.path.exists(args.data_file):
        raise(ValueError("Invalid data file path {}".format(os.path.abspath(args.data_file))))


def image_detection(image:sl.Mat, network, class_names, class_colors, thresh):
    import darknet
    # Darknet doesn't accept numpy images.
    # Create one with image we reuse for each detect
    width = darknet.network_width(network)
    height = darknet.network_height(network)
    darknet_image = darknet.make_image(width, height, 3)

    image_rgb = cv2.cvtColor(image.get_data(), cv2.COLOR_BGR2RGB)
    image_resized = cv2.resize(image_rgb, (width, height), interpolation=cv2.INTER_LINEAR)

    darknet.copy_image_from_bytes(darknet_image, image_resized.tobytes())

    detections = darknet.detect_image(network, class_names, darknet_image, thresh=thresh)

    darknet.free_image(darknet_image)

    _image_with_box = darknet.draw_boxes(detections, image_resized, class_colors)

    return cv2.cvtColor(_image_with_box, cv2.COLOR_BGR2RGB), detections


def print_detection_depth(detection_list: list, pc:sl.Mat, scale_2d:np.ndarray):
    for label, confidence, bbox in detection_list:
        scaled_bbox = [bbox[i] * scale_2d[i % 2] for i in range(len(bbox))]
        x, y, w, h = scaled_bbox
        x_center, y_center = int(round(x)), int(round(y))
        if (x_center < 0 or y_center < 0):
            print('bbox', bbox)
            print("{}: {}%".format(label, confidence))
            continue
        
        _pos_2d = get_pos_2d(pc, x_center, y_center)
        start_region_time = time.time()
        _depth = get_rectangle_region_depth(pc, round(x_center), round(y_center), 5, 5)
        end_region_time = time.time()
        # print('region depth time', end_region_time - start_region_time)

        print("{}: {}%    (x_center: {:.4f}  y_center:  {:.4f}  depth: {:.6f})".format(label, confidence, _pos_2d[0], _pos_2d[1], _depth))


def get_param_matrix(_zed):
    cam_param = _zed.get_camera_information().calibration_parameters.left_cam
    print('fx: {}, fy: {}, cx: {}, cy: {}'.format(cam_param.fx, cam_param.fy, cam_param.cx, cam_param.cy))
    
    inner_param = np.array([[cam_param.fx, 0, cam_param.cx], [0, cam_param.fy, cam_param.cy], [0, 0, 1]])
    inner_param = np.linalg.inv(inner_param)
    print('inner_param', inner_param)

    return np.mat(inner_param)


def main():
    args = parser()
    check_arguments_errors(args)

    print("start-------------")
    # Create a InitParameters object and set configuration parameters
    init_params = sl.InitParameters()
    init_params.camera_resolution = sl.RESOLUTION.HD720  # Use HD720 video mode
    init_params.coordinate_units = sl.UNIT.METER
    init_params.depth_mode = sl.DEPTH_MODE.ULTRA
    init_params.camera_fps = 5
    init_params.coordinate_system = sl.COORDINATE_SYSTEM.RIGHT_HANDED_Y_UP
    
    # sl.Camera.get_init_parameters()

    print("initialize camera")
    # Create a Camera object
    zed = sl.Camera()
    # Open the camera
    err = zed.open(init_params)
    if err != sl.ERROR_CODE.SUCCESS:
        exit(1)

    # get zed camera param matrix.
    runtime_matrix = get_param_matrix(zed)

    camera_info = zed.get_camera_information()
    # # Create OpenGL viewer
    viewer0 = gl.GLViewer()
    # print("init viewer")
    viewer0.init(camera_info.calibration_parameters.left_cam)
    
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

    import darknet
    random.seed(3)  # deterministic bbox colors
    network, class_names, class_colors = darknet.load_network(
        args.config_file,
        args.data_file,
        args.weights,
        batch_size=args.batch_size
    )

    pc_resolution = np.array([res.width, res.height], dtype=np.float32)

    no_index  = 1
    while viewer0.is_available():
        print("no index", no_index)
        # Grab an image, a RuntimeParameters object must be given to grab()
        if zed.grab() == sl.ERROR_CODE.SUCCESS:
            # Retrieve left image
            zed.retrieve_image(image, sl.VIEW.LEFT)

            # detect image
            start_detect_time = time.time()
            image_with_box, detections = image_detection(image, network, class_names, class_colors, args.thresh)
            print("image_with_box shape", image_with_box.shape)
            detection_resolution = np.array([image_with_box.shape[0], image_with_box.shape[1]], dtype=np.float32)
            res_scale = pc_resolution / detection_resolution
            end_detect_time = time.time()
            # darknet.print_detections(detections, args.ext_output)
            print("detection image time:", end_detect_time - start_detect_time)
            
            # fps
            fps = int(1/(end_detect_time - start_detect_time))
            print("FPS: {}".format(fps))

            start_pc_time = time.time()
            # retrieve point cloud
            zed.retrieve_measure(point_cloud, sl.MEASURE.XYZRGBA,sl.MEM.CPU, res)
            end_pc_time  = time.time()

            start_depth_time = time.time()
            print_detection_depth(detections, point_cloud, res_scale)
            end_depth_time = time.time()
            print('depth time', end_depth_time - start_depth_time)

            start_view_time = time.time()
            # # Update GL view
            viewer0.update_scene_view(image, point_cloud, detections, res_scale, runtime_matrix)
            # viewer0.update_view(image, objects)
            end_view_time = time.time()
            print('view time', end_view_time - start_view_time)

            no_index += 1

    viewer0.exit()

    image.free(memory_type=sl.MEM.CPU)
    # Disable modules and close camera
    zed.disable_object_detection()
    zed.disable_positional_tracking()

    zed.close()


if __name__ == "__main__":
    main()
