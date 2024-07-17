from dataclasses import dataclass, field
from typing import List

@dataclass
class Config:
    num_waypoints: int = 3

    num_vf_angles: int = 18
    num_vf_mags: int = 2
    num_vf_choices: int = (num_vf_angles)*(num_vf_mags)+1
    num_af_angles: int = 19
    num_af_mags: int = 2
    num_af_choices: int = (num_af_angles)*(num_af_mags)+1

    # After trimming
    vf_desc_col: int = 999
    af_desc_col: int = 999
    csv_input_col: int = 6 # upper bound is exlusive
    csv_label_col: int = 7

@dataclass
class TrainConfig(Config):
    train_csv_file: List[str] = field(default_factory=lambda: ['data/srand9/training_data_8192paths_3wps_10scale_EXPplanner_9seed.csv'])
    val_csv_file: List[str] = field(default_factory=lambda: ['data/1scale/val_data_4096paths_3wps_1scale_EXPplanner_10seed.csv'])
    
    num_examples_per_class: int = 100
    batch_size: int = 64
    input_size: int = 3*Config.num_waypoints + 9
    output_size: int = Config.num_vf_choices

    num_epochs: int = 500

@dataclass
class TestConfig(Config):    
    test_csv_file: List[str] = field(default_factory=lambda: ['data/1scale/test_data_1070paths_3wps_1scale_EXPplanner_10seed.csv'])
    test_model_path: str = ''

    batch_size: int = 64
    
    num_examples_per_class: int = 50
    
    # af_desc_col: int = 19