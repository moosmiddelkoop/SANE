from pathlib import Path

import torch

from torch.utils.data import Dataset

from SANE.datasets.zoo_split import select_models

import copy
import json
import tqdm

import logging


class PropertyDataset(Dataset):
    """
    This dataset class loads model properties from path, and skips the actual checkpoints.
    Interfaces with the same dataset.properties logic of the other datasets, but cheaper to laod and much smaller.
    """

    ## class arguments

    # init
    def __init__(
        self,
        root,  # path from which to load the dataset
        epoch_lst=[5, 10],  # list of epochs to load
        train_val_test="train",  # "train", "val" or "test" from the zoo's split.json; "all" for populations
        property_keys=None,  # keys of properties to load
        num_threads=4,
        verbosity=0,
    ):
        self.epoch_lst = epoch_lst
        self.verbosity = verbosity
        self.property_keys = copy.deepcopy(property_keys)
        self.train_val_test = train_val_test

        ### prepare directories and path list ################################################################

        ## check if root is list. if not, make root a list
        if not isinstance(root, list):
            root = [root]

        self.root = [Path(rdx) for rdx in root]

        # the split is fixed per zoo in <zoo>/split.json, see SANE.datasets.zoo_split
        self.path_list, _, self.split_id = select_models(self.root, train_val_test)

        ### initialize data over epochs #####################
        if not isinstance(epoch_lst, list):
            epoch_lst = [epoch_lst]

        # ### prepare data lists ###############
        self.paths = []
        # epochs = []

        if self.property_keys is not None:
            logging.info(f"Load properties for samples from paths.")

            # get propertys from path
            result_keys = self.property_keys.get("result_keys", [])
            config_keys = self.property_keys.get("config_keys", [])
            # figure out offset
            try:
                self.read_properties(
                    results_key_list=result_keys,
                    config_key_list=config_keys,
                    idx_offset=1,
                )
            except AssertionError as e:
                logging.error(e)
                self.read_properties(
                    results_key_list=result_keys,
                    config_key_list=config_keys,
                    idx_offset=0,
                )
            logging.info(f"Properties loaded.")
        else:
            self.properties = None

    ## getitem ####################################################################################################################################################################
    def __getitem__(self, index):
        # not implemented in base class
        raise NotImplementedError(
            "the __getitem__ function is not implemented in the base class. "
        )
        pass

    ## len ####################################################################################################################################################################
    def __len__(self):
        return len(self.data_in)

    ## read properties from path ##############################################################################################################################################
    def read_properties(self, results_key_list, config_key_list, idx_offset=1):
        """
        iterate over all paths in path_list and load the properties
        """
        # init dict
        properties = {}
        for key in results_key_list:
            properties[key] = []
        for key in config_key_list:
            properties[key] = []
        # remove ggap from results_key_list -> cannot be read, has to be computed.
        read_ggap = False
        if "ggap" in results_key_list:
            results_key_list.remove("ggap")
            read_ggap = True
        # iterate over samples
        for iidx, ppdx in tqdm.tqdm(enumerate(self.path_list)):
            # iterate over epochs
            for eedx in self.epoch_lst:
                try:
                    res_tmp = read_properties_from_path(
                        ppdx, eedx, idx_offset=idx_offset, verbosity=self.verbosity
                    )
                    if res_tmp is None:
                        continue
                    else:
                        for key in results_key_list:
                            properties[key].append(res_tmp[key])
                        for key in config_key_list:
                            properties[key].append(res_tmp["config"][key])
                        # compute ggap
                        if read_ggap:
                            gap = res_tmp["train_acc"] - res_tmp["test_acc"]
                            properties["ggap"].append(gap)
                        # assert epoch == training_iteration -> match correct data
                        # if iidx == 0:
                        #     train_it = int(res_tmp["training_iteration"])
                        #     assert (
                        #         int(eedx) == train_it
                        #     ), f"training iteration {train_it} and epoch {eedx} don't match."
                    self.paths.append(ppdx)
                except Exception as e:
                    logging.error(e)
                    logging.error(f"couldn't read data from {ppdx}. skip.")
        self.properties = properties


## helper function for property reading
def read_properties_from_path(path, idx, idx_offset, verbosity=5):
    """
    reads path/result.json
    returns the dict for training_iteration=idx
    idx_offset=0 if checkpoint_0 was written, else idx_offset=1
    """
    # read json
    try:
        fname = Path(path).joinpath("result.json")
        results = []
        for line in fname.open():
            results.append(json.loads(line))
        # trial_id = results[0]["trial_id"]
    except Exception as e:
        if verbosity > 5:
            logging.error(f"error loading {fname}")
            logging.error(e)
    # pick results
    jdx = idx - idx_offset
    try:
        resdx = results[jdx]
        return resdx
    except Exception as e:
        logging.debug(e)
        return None
