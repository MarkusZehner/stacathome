"""
This command line tool can be used to prefetch metadata from a Provider and store it as
MetadataCollection objects in python files. These can then be augmented by hand to include
missing values or correct erroneous values.

Example:
    python -m stacathome.tools.download -p planetary_computer -c sentinel-2-l2a -r '{"type": "Polygon","coordinates": [[[9.131870256207634, 47.758991256013616],[9.185185625306275, 47.758991256013616],[9.185185625306275, 47.78498307002232],[9.131870256207634, 47.78498307002232],[9.131870256207634, 47.758991256013616]]]}' -s "2025-07-10" -e "2025-07-13"
"""

import argparse
import json
from pathlib import Path

import pandas as pd
import odc.geo.geom as geom

import stacathome
import stacathome.geo


def print_items(items, geometry):
    strs = []
    for item in items:
        item_geometry = geom.Geometry(item.geometry, crs='EPSG:4326')
        overlap = stacathome.geo.wgs84_overlap_percentage(geometry, item_geometry)
        s = f"  + {item.id} ({item.datetime}, {overlap*100:.1f}% overlap, {len(item.assets)} assets"
        if 'proj:code' in item.properties:
            s += f', {item.properties["proj:code"]}'
        if 'sat:relative_orbit' in item.properties:
            s += f', orbit {item.properties["sat:relative_orbit"]}'
        s += ')'
        strs.append(s)
    print('\n'.join(strs), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--provider', '-p', required=True, help='The provider to download from')
    parser.add_argument('--collection', '-c', required=True, help='Which collection to download')
    parser.add_argument('--start', '-s', help='Start date of query')
    parser.add_argument('--end', '-e', help='End date of query')
    parser.add_argument('--roi', '-r', help='Region of interest as GeoJSON')
    parser.add_argument('--outfile', '-o', help='Filename of download.', default='out.zarr')
    parser.add_argument('--dry', '-d', action='store_true', help='If set, do not store but just print to stdout')
    parser.add_argument('--no-default', action='store_true', help='If set, do not use the default processor if availables')
    parser.add_argument('--search-only', action='store_true', help='If true, only search for STAC items, do not download')
    args = parser.parse_args()


    provider_name: str = args.provider
    collection: str = args.collection
    start = pd.to_datetime(args.start).to_pydatetime()
    end =  pd.to_datetime(args.end).to_pydatetime()
    roi: str = args.roi
    outfile = Path(args.outfile)
    dryrun: bool = args.dry
    no_default: bool = args.no_default
    search_only: bool = args.search_only    

    provider = stacathome.get_provider(provider_name)
    if not provider.has_collection(collection):
        parser.exit(1, f'Unknown collection: {collection}')

    geojson = json.loads(roi)
    geometry = geom.Geometry(geojson, crs='EPSG:4326')

    if search_only:
        items = stacathome.search_items(
            provider_name, 
            collection, 
            starttime=start,
            endtime=end,
            roi=geometry,
            no_default_processor=no_default
        )
        data = None
    else:
        items, data = stacathome.load(
            provider_name, 
            collection, 
            starttime=start,
            endtime=end,
            roi=geometry,
            no_default_processor=no_default
        )

    if data is None:
        print(f'Found {len(items)} items: ', flush=True)
        print_items(items, geometry)
        return

    print(f'Downloaded {len(items)} items: ', flush=True)
    print_items(items, geometry)
    print('-'*120)
    print(data, flush=True)

    if not dryrun:
        data.to_zarr(outfile)


if __name__ == '__main__':
    main()
