from platform import system
from json import loads
from pathlib import Path
from configparser import ConfigParser, SectionProxy
from pkg_resources import resource_filename
from requests import get
from tqdm import tqdm
from tdw.controller import Controller
from tdw.librarian import SceneLibrarian, ModelLibrarian, MaterialLibrarian, HDRISkyboxLibrarian, HumanoidLibrarian, HumanoidAnimationLibrarian
from tdw.backend.platforms import SYSTEM_TO_S3
from transport_challenge_multi_agent.paths import CONTAINERS_PATH, TARGET_OBJECTS_PATH, TARGET_OBJECT_MATERIALS_PATH, USER_CONFIG_PATH, DEFAULT_CONFIG_PATH


def download_asset_bundles() -> None:
    """
    Read a config file and download asset bundles, skipping local asset bundles that have already been downloaded.

    This function will also set the TDW librarian classes to use local asset bundles instead of remote S3 asset bundles.

    The expected config file path is: `~/transport_challenge_multi_agent/config.ini`. If that file doesn't exist, a default config file will be used.

    See the README for more information about the config file.
    """

    parser = ConfigParser()
    if USER_CONFIG_PATH.exists():
        path = str(USER_CONFIG_PATH)
    else:
        path = str(DEFAULT_CONFIG_PATH)
    parser.read(path)
    config: SectionProxy = parser["DEFAULT"]
    if config["local_asset_bundles"] != "1":
        return
    # Get the scenes and models from the floorplan layouts.
    floorplan_layouts_path = resource_filename("tdw", "add_ons/floorplan_layouts.json")
    floorplan_layouts = loads(Path(floorplan_layouts_path).read_text())
    scenes = []
    models = []
    materials = []
    for scene in floorplan_layouts:
        for variant in ["a", "b"]:
            scenes.append(f"floorplan_{scene}{variant}")
        for layout in floorplan_layouts[scene]:
            models.extend([m["name"] for m in floorplan_layouts[scene][layout]])
    # Add the containers.
    models.extend(CONTAINERS_PATH.read_text().split("\n"))
    # Add the target objects.
    target_objects_csv = TARGET_OBJECTS_PATH.read_text().split("\n")
    models.extend([m.split(",")[0] for m in target_objects_csv[1:]])
    # Remove duplicate models.
    models = list(sorted(set(models)))
    # Add the materials.
    materials.extend(TARGET_OBJECT_MATERIALS_PATH.read_text().split("\n"))
    materials = list(sorted(materials))
    # Add the HDRI skyboxes.
    skyboxes_path = resource_filename("tdw", "add_ons/interior_scene_lighting_data/hdri_skyboxes.json")
    skyboxes_data = loads(Path(skyboxes_path).read_text())
    skyboxes = list(sorted(skyboxes_data.keys()))
    # Add the Replicants.
    replicants = ["replicant_0"]
    # Add the animations.
    animations = ["walking_2"]
    # Get the output directory.
    asset_bundles_directory: Path = Path(config["asset_bundles_directory"])
    platform = SYSTEM_TO_S3[system()]
    download = config["download"] == "1"
    for asset_bundle_names, library_type, library, dictionary in zip(
            [scenes, models, materials, skyboxes, replicants, animations],
            [SceneLibrarian, ModelLibrarian, MaterialLibrarian, HDRISkyboxLibrarian, HumanoidLibrarian,
             HumanoidAnimationLibrarian],
            ["scenes", "models_core", "materials_low", "hdri_skyboxes", "replicants", "humanoid_animations"],
            [Controller.SCENE_LIBRARIANS, Controller.MODEL_LIBRARIANS, Controller.MATERIAL_LIBRARIANS,
             Controller.HDRI_SKYBOX_LIBRARIANS, Controller.HUMANOID_LIBRARIANS,
             Controller.HUMANOID_ANIMATION_LIBRARIANS]):
        output_directory = asset_bundles_directory.joinpath(library)
        if not output_directory.exists():
            output_directory.mkdir(parents=True)
        library_path = output_directory.joinpath(f"{library}.json").resolve()
        # Create a new library.
        if not library_path.exists():
            library_type.create_library(description=library, path=str(library_path))
        # Load the local and remote librarians.
        librarian_local = library_type(str(library_path))
        # Set the local librarian as the default.
        f = f"{library}.json"
        dictionary[f] = librarian_local
        # Check if we need to download any asset bundles.
        asset_bundles_exist = True
        for asset_bundle_name in asset_bundle_names:
            asset_bundle_path = output_directory.joinpath(asset_bundle_name).resolve()
            if not asset_bundle_path.exists():
                asset_bundles_exist = False
                break
        if asset_bundles_exist:
            continue
        librarian_remote = library_type(f)
        pbar = tqdm(total=len(asset_bundle_names))
        for asset_bundle_name in asset_bundle_names:
            # We already have downloaded this asset bundle.
            local_record = librarian_local.get_record(asset_bundle_name)
            if local_record is not None:
                pbar.update(1)
                continue
            # Get the asset bundle.
            pbar.set_description_str(asset_bundle_name)
            record = librarian_remote.get_record(asset_bundle_name)
            asset_bundle_path = output_directory.joinpath(asset_bundle_name).resolve()
            # Download the asset bundle.
            if download:
                resp = get(record.get_url()).content
                asset_bundle_path.write_bytes(resp)
            # Add the record.
            record.urls[platform] = "file:///" + str(asset_bundle_path).replace("\\", "/")
            librarian_local.add_or_update_record(record=record, overwrite=True, write=True)
            pbar.update(1)
        pbar.close()
