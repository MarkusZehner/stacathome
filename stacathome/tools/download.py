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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--provider', '-p', required=True, help='The provider to download from')
    parser.add_argument('--collection', '-c', required=True, help='Which collection to download')
    parser.add_argument('--start', '-s', help='Start date of query')
    parser.add_argument('--end', '-e', help='End date of query')
    parser.add_argument('--roi', '-r', help='Region of interest as GeoJSON')
    parser.add_argument('--outfile', '-o', help='Filename of download.', default='out.zarr')
    parser.add_argument('--dry', '-d', action='store_true', help='If set, do not store but just print to stdout')
    args = parser.parse_args()


    provider_name: str = args.provider
    collection: str = args.collection
    start = pd.to_datetime(args.start).to_pydatetime()
    end =  pd.to_datetime(args.end).to_pydatetime()
    roi: str = args.roi
    outfile = Path(args.outfile)
    dryrun: bool = args.dry

    provider = stacathome.get_provider(provider_name)
    if not provider.has_collection(collection):
        parser.exit(1, f'Unknown collection: {collection}')


    geojson = json.loads(roi)
    geometry = geom.Geometry(geojson)

    items, data = stacathome.load(
        provider_name, 
        collection, 
        starttime=start,
        endtime=end,
        roi=geometry
    )

    print(f'Downloaded {len(items)} items: ', flush=True)
    print('\n'.join([item.id for item in items]))
    print('-'*120)
    print(data, flush=True)

    if not dryrun:
        data.to_zarr(outfile)

if __name__ == '__main__':
    main()
