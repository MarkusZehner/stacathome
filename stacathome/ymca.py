# yet more cubing attempts!
from shapely import Point
from .combine import combine_to_cube
from .download import download_request_from_probe
from .request import build_request_from_probe, probe_request
from .utils import parse_dec_to_lon_lat_point, parse_dms_to_lon_lat_point


def get_minicube(pos_coordinates, time_range, edge_length_m, target_res_m, sel_bands, workdir):
    for position_str in pos_coordinates:
        fname = None
        if len(position_str) == 2:
            fname, position_str = position_str

        print(fname, position_str)

        try:
            position = parse_dms_to_lon_lat_point(position_str)
        except TypeError:
            pass
        try:
            position = parse_dec_to_lon_lat_point(position_str)
        except (TypeError, ValueError):
            pass
        assert position is not None
        assert isinstance(position, Point)

        probe_dict = probe_request(
            point_wgs84=position, distance_in_m=edge_length_m, collection=list(sel_bands.keys()), return_box=True
        )
        request_from_probe = build_request_from_probe(
            center_point=position,
            time_range=time_range,
            edge_length_m=edge_length_m,
            target_res_m=target_res_m,
            probe_dict=probe_dict,
            save_dir=workdir,
            fname=fname
        )
        file_list = download_request_from_probe(request_from_probe, sel_bands=sel_bands, workdir=workdir)
        # return file_list, position, time_range, probe_dict, request_from_probe, workdir, edge_length_m
        combine_to_cube(file_list, position, time_range, probe_dict, request_from_probe, workdir, edge_length_m)
