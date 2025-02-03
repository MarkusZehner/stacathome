from os import listdir, rmdir, path as os_path
from functools import partial

import json
import pickle

from pyproj import Proj, Transformer
from requests import get as requests_get

def remove_empty_folders(paths, base_folder):
    # Normalize base folder path to avoid issues with relative paths
    base_folder = os_path.abspath(base_folder)

    def delete_if_empty(folder):
        """Recursively delete folder and its empty parents within base folder"""
        folder = os_path.abspath(folder)
        
        # Ensure folder is within base folder
        if folder.startswith(base_folder) and os_path.isdir(folder):
            try:
                # If the folder is empty, delete it
                if not listdir(folder):
                    rmdir(folder)
                    print(f"Deleted empty folder: {folder}")
                    
                    # Check parent folder if it's not the base folder
                    parent_folder = os_path.dirname(folder)
                    if parent_folder != base_folder:
                        delete_if_empty(parent_folder)  # Recursively check the parent
                else:
                    print(f"Skipped (not empty): {folder}")
            except Exception as e:
                print(f"Failed to delete {folder}: {e}")
        else:
            print(f"Skipped (outside base folder or not a directory): {folder}")

    # Process each folder in the provided paths
    for folder in paths:
        delete_if_empty(folder)


def get_transform(from_crs, to_crs, always_xy=True):
    project = Transformer.from_proj(
        Proj(f"epsg:{from_crs}"),  # source coordinate system
        Proj(f"epsg:{to_crs}"),
        always_xy=always_xy,
    )
    return partial(transform_coords, project=project)


def transform_coords(x_y, project):
    for i in range(len(x_y)):
        x_y[i] = project.transform(x_y[i][0], x_y[i][1])
    return x_y


def get_countries_json(name):
    if check_if_country_in_repo(name):
        url = f"https://raw.githubusercontent.com/georgique/world-geojson/main/countries/{
            name}.json"
        json = requests_get(url).json()
        json["features"][0]["properties"]["crs"] = 4326
        return json


def check_if_country_in_repo(name):

    country_set = set(['afghanistan', 'albania', 'algeria', 'andorra', 'angola',
                    'antigua_and_barbuda', 'argentina', 'armenia', 'australia',
                    'austria', 'azerbaijan', 'bahamas', 'bahrain', 'bangladesh',
                    'barbados', 'belarus', 'belgium', 'belize', 'benin', 'bhutan',
                    'bolivia', 'bosnia_and_herzegovina', 'botswana', 'brazil',
                    'brunei', 'bulgaria', 'burkina_faso', 'burundi', 'cambodia',
                    'cameroon', 'canada', 'cape_verde', 'central_african_republic',
                    'chad', 'chile', 'china', 'colombia', 'comoros', 'congo',
                    'cook_islands', 'costa_rica', 'croatia', 'cuba', 'cyprus',
                    'czech', 'democratic_congo', 'denmark', 'djibouti', 'dominica',
                    'dominican_republic', 'east_timor', 'ecuador', 'egypt',
                    'el_salvador', 'equatorial_guinea', 'eritrea', 'estonia',
                    'eswatini', 'ethiopia', 'fiji', 'finland', 'france', 'gabon',
                    'gambia', 'georgia', 'germany', 'ghana', 'greece', 'grenada',
                    'guatemala', 'guinea', 'guinea_bissau', 'guyana', 'haiti',
                    'honduras', 'hungary', 'iceland', 'india', 'indonesia', 'iran',
                    'iraq', 'ireland', 'israel', 'italy', 'ivory_coast', 'jamaica',
                    'japan', 'jordan', 'kazakhstan', 'kenya', 'kiribati', 'kuwait',
                    'kyrgyzstan', 'laos', 'latvia', 'lebanon', 'lesotho', 'liberia',
                    'libya', 'liechtenstein', 'lithuania', 'luxembourg', 'madagascar',
                    'malawi', 'malaysia', 'maldives', 'mali', 'malta', 'marshall_islands',
                    'mauritania', 'mauritius', 'mexico', 'micronesia', 'moldova', 'monaco',
                    'mongolia', 'montenegro', 'morocco', 'mozambique', 'myanmar', 'namibia',
                    'nauru', 'nepal', 'netherlands', 'new_zealand', 'nicaragua', 'niger',
                    'nigeria', 'niue', 'north_korea', 'north_macedonia', 'norway', 'oman',
                    'pakistan', 'palau', 'palestine', 'panama', 'papua_new_guinea', 'paraguay',
                    'peru', 'philippines', 'poland', 'portugal', 'qatar', 'romania', 'russia',
                    'rwanda', 'saint_kitts_and_nevis', 'saint_lucia', 'saint_vincent_and_the_grenadines',
                    'samoa', 'san_marino', 'sao_tome_and_principe', 'saudi_arabia', 'senegal',
                    'serbia', 'seychelles', 'sierra_leone', 'singapore', 'slovakia', 'slovenia',
                    'solomon_islands', 'somalia', 'south_africa', 'south_korea', 'south_sudan',
                    'spain', 'sri_lanka', 'sudan', 'suriname', 'sweden', 'switzerland', 'syria',
                    'tajikistan', 'tanzania', 'thailand', 'togo', 'tonga', 'trinidad_and_tobago',
                    'tunisia', 'turkey', 'turkmenistan', 'tuvalu', 'uganda', 'ukraine',
                    'united_arab_emirates', 'united_kingdom', 'uruguay', 'usa', 'uzbekistan',
                    'vanuatu', 'vatican', 'venezuela', 'vietnam', 'western_sahara', 'yemen',
                    'zambia', 'zimbabwe'])
    if name.lower() in country_set:
        return True
    else:
        owner = "georgique"  # GitHub username or organization
        repo = "world-geojson"  # Repository name
        # Path within the repository (use an empty string if listing the root)
        path = "countries"
        # Branch name (e.g., 'main', 'master', or other branches)
        branch = "main"

        url = f"https://api.github.com/repos/{owner}/{
            repo}/contents/{path}?ref={branch}"

        # Make a GET request to the GitHub API
        response = requests_get(url)

        # Check if the request was successful
        if response.status_code == 200:
            contents = response.json()
            # List all filenames
            filenames = [
                item["name"].split(".")[0]
                for item in contents
                if item["type"] == "file"
            ]
            if name.lower() in filenames:
                return True
        else:
            print(f"Failed to retrieve data: {response.status_code}")

        possible_matches = [
            country for country in country_set if country.startswith(name[0])
        ]
        print(
            f"Country {name} not found in the repository, maybe you meant one of these: {
            possible_matches}"
        )
        return None



def pickled_items_to_json(pickled_items, json_file):
    stac_json = []
    for item in pickled_items:
        stac_json.append(item.to_dict())
    json_dict = {'type': 'FeatureCollection', 'features': stac_json}    
    json.dump(json_dict, open(json_file, 'w'))