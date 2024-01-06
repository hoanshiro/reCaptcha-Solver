import os

base_dir = os.path.dirname(os.path.abspath(__file__))
CFG = {
    "model_name": "lcnet_050.ra2_in1k",
    "learning_rate": 1e-4,
    "weight_decay": 1e-6,
    "num_classes": 20,
    "batch_size": 32,
    "num_workers": 4,
    "epochs": 10,
    "device": "cpu",
    "dataset_path": f"{base_dir}",
    "data_dir": f"{base_dir}/google-recaptcha-image/ds/",
    "labels": [
        "bicycle",
        "bridge",
        "bus",
        "car",
        "chimney",
        "cross",
        "hydrant",
        "motorcycle",
        "other",
        "palm",
        "stair",
        "tlight",
    ],
}
print(CFG)
