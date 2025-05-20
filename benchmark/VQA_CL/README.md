# VQA_CL Benchmark

## Introduction

The VQA_CL (Visual Question Answering in Continual Learning) benchmark is designed to evaluate the ability of models to perform visual question answering tasks in a continual learning setting. This means the model is expected to learn from a sequence of VQA datasets, corresponding to different visual domains or task types, without catastrophically forgetting previously learned knowledge. This benchmark helps assess model adaptability, knowledge retention, and forward transfer capabilities in evolving VQA scenarios.

## Datasets

Please download the images from the respective datasets using the links below.

**The JSON annotation files for the train/test splits of each dataset (e.g., `scienceqa_train.json`, `vqav2_test.json`, etc.) can be downloaded from the Hugging Face Hub at: [ecnu-icalk/VQA_CL](https://huggingface.co/datasets/ecnu-icalk/VQA_CL).**

|  Image Source   | Download Path  |
|  :----:  | :----:  |
| COCO | [train2014](http://images.cocodataset.org/zips/train2014.zip), [test2015](http://images.cocodataset.org/zips/test2015.zip), [val2014](http://images.cocodataset.org/zips/val2014.zip) |
| ScienceQA  | [images](https://drive.google.com/drive/folders/1w8imCXWYn2LxajmGeGH_g5DaL2rabHev) |
| VizWiz  | [train](https://vizwiz.cs.colorado.edu/VizWiz_final/images/train.zip), [val](https://vizwiz.cs.colorado.edu/VizWiz_final/images/val.zip), [test](https://vizwiz.cs.colorado.edu/VizWiz_final/images/test.zip) |

## Data Organization

The benchmark expects a specific directory structure for the images and JSON annotation files.

### JSON File Format

The JSON files (e.g., `scienceqa_train.json`, `vizwiz_test.json`, `vqav2_train.json`) are structured as a list of objects, where each object represents a VQA instance. These JSON files are located within their respective dataset subdirectories (e.g., `scienceqa/scienceqa_train.json`, `vqav2/vqav2_train.json`).

Each object typically contains:

*   `"image"`: A string representing the relative path to the image file. **This path is relative to the directory containing the JSON file itself.** For instance, if `scienceqa_train.json` is in the `benchmark/VQA_CL/scienceqa/` directory, an `image` path like `"./scienceqa/images/train/1/image.png"` inside it will point to `benchmark/VQA_CL/scienceqa/scienceqa/images/train/1/image.png`.
*   `"conversations"`: A list of dialogue turns. Each turn is an object with:
    *   `"from"`: A string indicating the source of the text, either `"human"` (for questions) or `"gpt"` (for answers).
    *   `"value"`: A string containing the actual question or answer text. Questions may include an `<image>` token, which is a placeholder for the visual context.

**Example JSON structure (from an entry in `vqav2_train.json`, assuming it's located in `benchmark/VQA_CL/vqav2/`):**
```json
[
  {
    "image": "./COCO2014/train2014/COCO_train2014_000000458752.jpg", // Path relative to benchmark/VQA_CL/vqav2/
    "conversations": [
      {
        "from": "human",
        "value": "<image>\nWhat is this photo taken looking through?\nAnswer the question using a single word or phrase."
      },
      {
        "from": "gpt",
        "value": "net"
      }
      // ... more conversation turns ...
    ]
  }
  // ... more VQA instances ...
]
```

### Directory Structure

It is recommended to organize your downloaded image data as follows. JSON files should be placed in their respective dataset subdirectories within `benchmark/VQA_CL/`.

```
benchmark/
└── VQA_CL/
    ├── README.md
    ├── scienceqa/  # Subdirectory for ScienceQA dataset
    │   ├── scienceqa_train.json
    │   ├── scienceqa_test.json
    │   └── scienceqa/  # Image directory for ScienceQA, as referenced by paths in its JSONs
    │       └── images/
    │           ├── train/
    │           │   ├── 1/
    │           │   │   └── image.png
    │           │   └── ... (other ScienceQA train images)
    │           └── test/
    │               ├── 5/
    │               │   └── image.png
    │               └── ... (other ScienceQA test images)
    ├── vqav2/      # Subdirectory for VQAv2 dataset
    │   ├── vqav2_train.json
    │   ├── vqav2_test.json
    │   └── COCO2014/ # Image directory for COCO, as referenced by paths in VQAv2 JSONs
    │       ├── train2014/
    │       │   ├── COCO_train2014_000000458752.jpg
    │       │   └── ... (other COCO train images)
    │       └── val2014/
    │           ├── COCO_val2014_000000262148.jpg
    │           └── ... (other COCO val images)
    └── vizwiz/     # Subdirectory for VizWiz dataset
        ├── vizwiz_train.json
        ├── vizwiz_test.json
        └── VizWiz/   # Image directory for VizWiz, as referenced by paths in its JSONs
            ├── train/
            │   ├── VizWiz_train_00000000.jpg
            │   └── ... (other VizWiz train images)
            ├── val/
            │   ├── VizWiz_val_00000000.jpg
            │   └── ... (other VizWiz val images)
            └── test/ 
                └── ... (VizWiz test images, if applicable)

```

**Explanation:**

*   The main JSON annotation files (e.g., `scienceqa_train.json`, `vqav2_train.json`) are located within dataset-specific subdirectories like `benchmark/VQA_CL/scienceqa/`, `benchmark/VQA_CL/vqav2/`, etc.
*   The `image` paths within the JSON files (e.g., `"./scienceqa/images/train/1/image.png"` in `scienceqa_train.json`) are relative to the location of the JSON file itself.
*   **COCO (for VQAv2):**
    *   VQAv2 JSON files (e.g., `vqav2_train.json`) are in `benchmark/VQA_CL/vqav2/`.
    *   Image paths like `"./COCO2014/train2014/..."` inside these JSONs mean COCO images should be extracted into `benchmark/VQA_CL/vqav2/COCO2014/`.
    *   So, `train2014.zip` contents go into `benchmark/VQA_CL/vqav2/COCO2014/train2014/`.
    *   `val2014.zip` contents go into `benchmark/VQA_CL/vqav2/COCO2014/val2014/`.
*   **ScienceQA:**
    *   ScienceQA JSON files (e.g., `scienceqa_train.json`) are in `benchmark/VQA_CL/scienceqa/`.
    *   Image paths like `"./scienceqa/images/train/..."` inside these JSONs mean ScienceQA images should be extracted such that the `images` directory is under `benchmark/VQA_CL/scienceqa/scienceqa/`.
    *   So, the downloaded `images` directory (containing `train/` and `test/` subfolders) from ScienceQA should be placed at `benchmark/VQA_CL/scienceqa/scienceqa/images/`.
*   **VizWiz:**
    *   VizWiz JSON files (e.g., `vizwiz_train.json`) are in `benchmark/VQA_CL/vizwiz/`.
    *   Image paths like `"./VizWiz/train/..."` inside these JSONs mean VizWiz images should be extracted into `benchmark/VQA_CL/vizwiz/VizWiz/`.
    *   So, `train.zip` contents go into `benchmark/VQA_CL/vizwiz/VizWiz/train/`.
    *   `val.zip` contents go into `benchmark/VQA_CL/vizwiz/VizWiz/val/`.
    *   `test.zip` contents go into `benchmark/VQA_CL/vizwiz/VizWiz/test/`.

Ensure the image paths in your local setup match this structure for the benchmark to function correctly.
