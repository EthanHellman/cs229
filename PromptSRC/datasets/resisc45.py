import os
import pickle

from dassl.data.datasets import DATASET_REGISTRY, Datum, DatasetBase
from dassl.utils import mkdir_if_missing

from .oxford_pets import OxfordPets
from .dtd import DescribableTextures as DTD

NEW_CNAMES = {
    "wetland": "Wetland",
    "thermal_power_station": "Thermal Power Station",
    "terrace": "Terrace",
    "tennis_court": "Tennis Court",
    "storage_tank": "Storage Tank",
    "stadium": "Stadium",
    "sparse_residential": "Sparse Residential Area",
    "snowberg": "Snowberg",
    "ship": "Ship",
    "sea_ice": "Sea Ice",
    "runway": "Runway",
    "roundabout": "Roundabout",
    "river": "River",
    "rectangular_farmland": "Rectangular Farmland",
    "railway_station": "Railway Station",
    "railway": "Railway",
    "parking_lot": "Parking Lot",
    "palace": "Palace",
    "overpass": "Overpass",
    "mountain": "Mountain",
    "mobile_home_park": "Mobile Home Park",
    "medium_residential": "Medium Residential Area",
    "meadow": "Meadow",
    "lake": "Lake",
    "island": "Island",
    "intersection": "Intersection",
    "industrial_area": "Industrial Area",
    "harbor": "Harbor",
    "ground_track_field": "Ground Track Field",
    "golf_course": "Golf Course",
    "freeway": "Freeway",
    "forest": "Forest",
    "desert": "Desert",
    "dense_residential": "Dense Residential Area",
    "commercial_area": "Commercial Area",
    "cloud": "Cloud",
    "circular_farmland": "Circular Farmland",
    "church": "Church",
    "chaparral": "Chaparral",
    "bridge": "Bridge",
    "beach": "Beach",
    "basketball_court": "Basketball Court",
    "baseball_diamond": "Baseball Diamond",
    "airport": "Airport",
    "airplane": "Airplane",
}



@DATASET_REGISTRY.register()
class Resisc45(DatasetBase):

    dataset_dir = "resisc45"

    def __init__(self, cfg):
        root = os.path.abspath(os.path.expanduser(cfg.DATASET.ROOT))
        self.dataset_dir = os.path.join(root, self.dataset_dir)
        self.image_dir = os.path.join(self.dataset_dir, "3100")
        self.split_path = os.path.join(self.dataset_dir, "split_zhou_Resisc45.json")
        self.split_fewshot_dir = os.path.join(self.dataset_dir, "split_fewshot")
        mkdir_if_missing(self.split_fewshot_dir)

        if os.path.exists(self.split_path):
            train, val, test = OxfordPets.read_split(self.split_path, self.image_dir)
        else:
            train, val, test = DTD.read_and_split_data(self.image_dir, new_cnames=NEW_CNAMES)
            OxfordPets.save_split(train, val, test, self.split_path, self.image_dir)

        num_shots = cfg.DATASET.NUM_SHOTS
        if num_shots >= 1:
            seed = cfg.SEED
            preprocessed = os.path.join(self.split_fewshot_dir, f"shot_{num_shots}-seed_{seed}.pkl")
            
            if os.path.exists(preprocessed):
                print(f"Loading preprocessed few-shot data from {preprocessed}")
                with open(preprocessed, "rb") as file:
                    data = pickle.load(file)
                    train, val = data["train"], data["val"]
            else:
                train = self.generate_fewshot_dataset(train, num_shots=num_shots)
                val = self.generate_fewshot_dataset(val, num_shots=min(num_shots, 4))
                data = {"train": train, "val": val}
                print(f"Saving preprocessed few-shot data to {preprocessed}")
                with open(preprocessed, "wb") as file:
                    pickle.dump(data, file, protocol=pickle.HIGHEST_PROTOCOL)

        subsample = cfg.DATASET.SUBSAMPLE_CLASSES
        train, val, test = OxfordPets.subsample_classes(train, val, test, subsample=subsample)

        super().__init__(train_x=train, val=val, test=test)

    def update_classname(self, dataset_old):
        dataset_new = []
        for item_old in dataset_old:
            cname_old = item_old.classname
            cname_new = NEW_CLASSNAMES[cname_old]
            item_new = Datum(impath=item_old.impath, label=item_old.label, classname=cname_new)
            dataset_new.append(item_new)
        return dataset_new
