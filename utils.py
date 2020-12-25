import numpy as np
import pyzed.sl as sl


def get_pos_2d(pc:sl.Mat, x_center_index:int, y_center_index:int):
    _pos_3d = pc.get_value(x_center_index, y_center_index)

    max_dist = 100000000.0
    _pos_2d = (max_dist, max_dist)
    if (_pos_3d[0]== sl.ERROR_CODE.SUCCESS and -max_dist <  _pos_3d[1][0] < max_dist and -max_dist <  _pos_3d[1][1] < max_dist):
        _pos_2d = (_pos_3d[1][0], _pos_3d[1][1])
    
    return _pos_2d


def get_pos_3d(pc:sl.Mat, x_center_index:int, y_center_index:int):
    coord_value = pc.get_value(x_center_index, y_center_index)

    max_dist = 100000000.0
    _pos_3d = (max_dist, max_dist, max_dist)
    if (coord_value[0]== sl.ERROR_CODE.SUCCESS and -max_dist <  coord_value[1][0] < max_dist and -max_dist <  coord_value[1][1] < max_dist and  -max_dist <  coord_value[1][2] < 0):
        _pos_3d = coord_value[1]
    
    return _pos_3d


# get rectangle region depth.
def get_rectangle_region_depth(pc:sl.Mat, x_center_index:int, y_center_index:int, rec_width:int, rec_height:int) ->float:
    remote_depth = -100000000.0
    half_width, half_height = round(rec_width / 2.), round(rec_height / 2.)
    # region_pixels = [(x_center_index - half_width + i, y_center_index - half_height + j) for i in range(rec_width) for j in range(rec_height)]
    # region_pixels.filter(lambda x: x[0] > 0 and x[1] > 0)

    # region_pixels = [(x_center_index - half_width + i, y_center_index - half_height + j)
    #                                     if (x_center_index - half_width + i > 0 and y_center_index - half_height + j > 0) else (sl.ERROR_CODE.FAILURE, (0, 0, 0)) 
    #                                     for i in range(rec_width) for j in range(rec_height)]
    depth_result_list = [pc.get_value(x_center_index - half_width + i, y_center_index - half_height + j) 
                                             if (x_center_index - half_width + i > 0 and y_center_index - half_height + j > 0) else (sl.ERROR_CODE.FAILURE, (0, 0, 0)) 
                                             for i in range(rec_width) for j in range(rec_height)]
    depth_list = [line[1][2] for line in depth_result_list if line[0] == sl.ERROR_CODE.SUCCESS and remote_depth <  line[1][2] < 0]
    
    rec_depth = remote_depth
    if len(depth_list) > 0:
        depth_list = sorted(depth_list, key=lambda d: -d)[:100]
        rec_depth = np.mean(np.array(depth_list))
        # rec_depth = max(depth_list)
    
    return rec_depth