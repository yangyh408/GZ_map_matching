# -*- coding:utf-8 -*-
import json
import os
import warnings

import geopandas as gpd
import numpy as np
import pandas as pd
# import matplotlib.pyplot as plt
# import transbigdata as tbd
from leuvenmapmatching import visualization as mmviz
from leuvenmapmatching.map.inmem import InMemMap
from leuvenmapmatching.matcher.distance import DistanceMatcher

warnings.filterwarnings("ignore")

mid_points_path = os.path.join('.', 'src', 'mid_points.npy')
task_path = os.path.join('.', 'log', 'task_info.json')
big_node_path = os.path.join('.', 'src', '20220730大路网toyyh', 'node_l_7featuresdone.shx')
big_link_path = os.path.join('.', 'src', '20220730大路网toyyh', 'link_l_7featuresdone.shx')
small_node_path = os.path.join('.', 'src', '20220730小路网toyyh', 'node_s_5turningdone.shx')
small_link_path = os.path.join('.', 'src', '20220730小路网toyyh', 'link_s_6uturn.shx')


def generate_task(div_num):
    mid_points = np.load(mid_points_path, allow_pickle=True)
    task_info = {
        "tasks": [],
    }
    batch_size = len(mid_points) // div_num
    for i in range(0, div_num):
        if i == div_num-1:
            task_info["tasks"].append([batch_size * i, len(mid_points)])
        else:
            task_info["tasks"].append([batch_size*i, batch_size*(i+1)])
    save_json(task_info, task_path)


def save_json(dict, path):
    if isinstance(dict, str):
        dict = eval(dict)
    with open(path, 'w', encoding='utf-8') as f:
        str_ = json.dumps(dict, ensure_ascii=False)
        f.write(str_)


def load_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        data = f.readline().strip()
        dict = json.loads(data)
        return dict
    
    
def init_map():
    node_gdf_big = gpd.read_file(big_node_path)
    link_gdf_big = gpd.read_file(big_link_path)

    node_gdf_small = gpd.read_file(small_node_path)
    link_gdf_small = gpd.read_file(small_link_path)
    node_id_bias = 2000
    node_gdf_small['nodeid'] += node_id_bias
    link_gdf_small['s_nodeid'] = link_gdf_small.sid + node_id_bias
    link_gdf_small['e_nodeid'] = link_gdf_small.eid + node_id_bias
    link_gdf_small['L_NODE'] = link_gdf_small.L_NODE + node_id_bias
    link_gdf_small['T_NODE'] = link_gdf_small.T_NODE + node_id_bias
    link_gdf_small['R_NODE'] = link_gdf_small.R_NODE + node_id_bias
    link_gdf_small['O1_NODE'] = link_gdf_small.O1_NODE + node_id_bias
    link_gdf_small['O2_NODE'] = link_gdf_small.O2_NODE + node_id_bias
    link_gdf_small['r_L_NODE'] = link_gdf_small.R_L_NODE + node_id_bias
    link_gdf_small['r_T_NODE'] = link_gdf_small.R_T_NODE + node_id_bias
    link_gdf_small['r_R_NODE'] = link_gdf_small.R_R_NODE + node_id_bias
    link_gdf_small['r_O1_NODE'] = link_gdf_small.R_O1_NODE + node_id_bias
    link_gdf_small['r_O2_NODE'] = link_gdf_small.R_O2_NODE + node_id_bias

    node_gdf_4326 = pd.concat([node_gdf_big[['nodeid', 'geometry']], node_gdf_small[['nodeid', 'geometry']]], ignore_index=True)
    link_info = ['s_nodeid', 'e_nodeid', 'L_NODE', 'T_NODE', 'R_NODE', 'O1_NODE', 'O2_NODE', 'r_L_NODE', 'r_T_NODE', 'r_R_NODE', 'r_O1_NODE', 'r_O2_NODE', 'geometry']
    link_gdf_4326 = pd.concat([link_gdf_big[link_info], link_gdf_small[link_info]], ignore_index=True)
    
    node_gdf_4326.crs = {'init': 'epsg:4326'}
    node_gdf = node_gdf_4326.to_crs(2414)

    link_gdf_4326['lon'] = link_gdf_4326.centroid.x
    link_gdf_4326['lat'] = link_gdf_4326.centroid.y
    link_gdf_4326.crs = {'init': 'epsg:4326'}
    link_gdf = link_gdf_4326.to_crs(2414)

    map_con = InMemMap(name='GZ_zhongguan', use_latlon=False)

    for _, info in node_gdf.iterrows():
        map_con.add_node(info.nodeid, (info.geometry.y, info.geometry.x))

    for _, info in link_gdf.iterrows():
        map_con.add_edge(info.s_nodeid, info.e_nodeid)
        map_con.add_edge(info.e_nodeid, info.s_nodeid)
        for i in range(2, 7):
            if info[i] != 0 and info[i] != node_id_bias:
                map_con.add_edge(info.e_nodeid, info[i])
        for i in range(7, 12):
            if info[i] != 0 and info[i] != node_id_bias:
                map_con.add_edge(info.s_nodeid, info[i])
    return link_gdf_4326, map_con


def get_path(points):
    lon = []
    lat = []
    for p in points:
        bias_lon = -0.005539
        bias_lat = 0.002725
        lon.append(p[0]+bias_lon)
        lat.append(p[1]+bias_lat)
    if len(lon) == 0:
        return [], []
    tmp_df = pd.DataFrame({'lon': lon, 'lat': lat})
    tmp_df['geometry'] = gpd.points_from_xy(tmp_df['lon'], tmp_df['lat'])

    tmp_gdf = gpd.GeoDataFrame(tmp_df)
    tmp_gdf.crs = {'init': 'epsg:4326'}
    tmp_gdf = tmp_gdf.to_crs(2414)
    tmp_path = list(zip(tmp_gdf.geometry.y, tmp_gdf.geometry.x))
    return tmp_gdf, tmp_path


def map_matching(map_con, path, show_image=False):
    matcher = DistanceMatcher(map_con, max_dist=10000, max_dist_init=170, min_prob_norm=0.0001,
                              non_emitting_length_factor=0.95, obs_noise=50, obs_noise_ne=50,
                              dist_noise=50, max_lattice_width=20, non_emitting_states=True)
    states, _ = matcher.match(path, unique=False)
    
    if show_image:
        mmviz.plot_map(map_con, matcher=matcher, show_labels=True, show_matching=True, )
                       # filename=f"/Users/yangyh408/Desktop/FCD/matching_image/out_basemap_{plot_num}.png")
            
    return matcher.path_pred_onlynodes


def plot_result_with_map(link_gdf_4326, tmp_gdf, match_result, save_file=False):
    pathdf = pd.DataFrame(match_result, columns=["u"])
    pathdf["v"] = pathdf["u"].shift(-1)
    pathdf = pathdf[-pathdf["v"].isnull()]
    tmpdf = pathdf
    for _, i in pathdf.iterrows():
        tmpdf = pd.DataFrame(
            np.insert(tmpdf.values, len(tmpdf.index), values=[i.v, i.u], axis=0)
        )
    tmpdf.columns = ["s_nodeid", "e_nodeid"]
    
    link_gdf = link_gdf_4326.to_crs(2414)
    pathgdf = pd.merge(tmpdf, link_gdf.reset_index())
    pathgdf = gpd.GeoDataFrame(pathgdf)
    # pathgdf.plot()
    pathgdf.crs = {"init": "epsg:2414"}
    pathgdf_4326 = pathgdf.to_crs(4326)

    fig = plt.figure(1, (8, 8), dpi=100)
    ax = plt.subplot(111)
    plt.sca(ax)
    fig.tight_layout(rect=(0.05, 0.1, 1, 0.9))
    # 设定可视化边界
    bounds = pathgdf_4326.unary_union.bounds
    gap = 0.003
    bounds = [bounds[0] - gap, bounds[1] - gap, bounds[2] + gap, bounds[3] + gap]
    # 绘制匹配的路径
    pathgdf_4326.plot(ax=ax, zorder=1)
    # 绘制底图路网
    tbd.clean_outofbounds(link_gdf_4326, bounds, col=["lon", "lat"]).plot(
        ax=ax, color="#333", lw=0.1
    )
    # 绘制GPS点
    tmp_gdf.to_crs(4326).plot(ax=ax, color="r", markersize=5, zorder=2)

    plt.axis("off")
    plt.xlim(bounds[0], bounds[2])
    plt.ylim(bounds[1], bounds[3])
    if save_file:
        plt.savefig(f"/Users/yangyh408/Desktop/FCD/matching_image/out_map_{plot_num}.png")
    else:
        plt.show()
        

def single_match(plot_num, show_image=True):
    link_gdf_4326, map_con = init_map()

    mid_points = np.load(mid_points_path, allow_pickle=True)

    tmp_gdf, path = get_path(mid_points[plot_num])
    match_result = map_matching(plot_num, path, show_image)
    if show_image:
        plot_result_with_map(link_gdf_4326, tmp_gdf, match_result, save_file=False)


def load_result():
    os.system('clear')
    try:
        task_info = load_json(task_path)
        match_result = load_json(os.path.join('.', 'result', f'result_task{task_info["my_task"]}.json'))
        print(f"You're running task {task_info['my_task']} with process "
              f"{'%.2f' % ((task_info['cur_num']-task_info['start_num'])/(task_info['end_num']-task_info['start_num'])*100)}% ({task_info['cur_num']-task_info['start_num']}"
              f"/{task_info['end_num']-task_info['start_num']})")
    except:
        generate_task(5)
        task_info = load_json(task_path)

        print("==========================================================")
        print("Thanks for helping me run this file!")
        task_info['my_task'] = int(input("  -->  Choose the task ID [1,2,3,4,5]: "))
        print("==========================================================")
        task_info['start_num'] = task_info['tasks'][task_info['my_task']-1][0]
        task_info['end_num'] = task_info['tasks'][task_info['my_task']-1][1]
        task_info['cur_num'] = task_info['start_num']
        save_json(task_info, task_path)
        print(f"You're running task {task_info['my_task']} [index from {task_info['start_num']}--{task_info['end_num']}]")
        match_result = {}
        save_json(match_result, os.path.join('.', 'result', f'result_task{task_info["my_task"]}.json'))
    finally:
        return task_info, match_result


def batch_match():

    task_info, match_result = load_result()

    link_gdf_4326, map_con = init_map()
    mid_points = np.load(mid_points_path, allow_pickle=True)

    try:
        task_num = task_info['end_num'] - task_info['start_num']
        for plot_num in range(task_info['cur_num'], task_info['end_num']):
            if plot_num == task_info['cur_num'] or plot_num % 100 == 0:
                os.system('clear')
                cur_num = plot_num-task_info['start_num']
                match_rate = cur_num / task_num
                finish = "▓" * int(match_rate*80)
                need_do = "-" * (80 - int(match_rate))
                print("[{}->{}]{:^4.2f}%({}/{})".format(finish, need_do, match_rate*100, cur_num, task_num))

            # print(f"[{plot_num}] --> ", end="")
            tmp_gdf, path = get_path(mid_points[plot_num])
            if len(path) == 0:
                match_result[str(plot_num)] = []
                continue
            match_result[str(plot_num)] = map_matching(map_con, path)
        os.system('clear')
        finish = "▓" * 100
        print("[{}]{:^5.2f}%".format(finish, 100))
        print("matching done!")
    finally:
        save_json(match_result, os.path.join('.', 'result', f'result_task{task_info["my_task"]}.json'))
        task_info['cur_num'] = int(list(match_result.keys())[-1]) + 1
        save_json(task_info, task_path)

        
if __name__ == '__main__':
    output_num = None
    if output_num:
        single_match(output_num)
    else:
        batch_match()
