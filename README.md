# My-Perception:
## Introduction

The My-Perception repository focuses on embodied perception, covering both benchmarks and models for embodied scene understanding and reasoning. Currently, it includes:
### Models
- [LLaVA-3D](https://zcmax.github.io/projects/LLaVA-3D/)
### Benchmaeks
- [MMScan](https://tai-wang.github.io/mmscan/), [OST-Bench](https://rbler1234.github.io/OSTBench.github.io/), [MMSI-Bench](https://runsenxu.com/projects/MMSI_Bench/), and EgoExo-Bench.







## What's New

### Highlight
2025/7 - The first version of My-Perception includes the model LLaVA-3D, and benchmarks MMScan, OST-Bench, MMSI-Bench, and EgoExo-Bench.

## Getting Start

Clone this Github repo and try using our model and benchmark by following the quickstart tutorial below, for more detail, you can refer to 
```shell
git clone https://github.com/rbler1234/my-perception.git
cd my-perception
```
<details>
  <summary>(Model) LLaVA-3D</summary>

</details>

<details>
  <summary>(Benchmark) MMScan</summary>

1. **Install requirements**

   Your environment needs to include Python version 3.8 or higher.

   ```shell
   cd MMScan
   conda activate your_env_name
   python intall.py all/VG/QA
   ```

   Use `"all"` to install all components and specify `"VG"` or `"QA"` if you only need to install the components for Visual Grounding or Question Answering, respectively.
2. **Data Preparation**

    (a) Download the Embodiedscan and MMScan annotation. (Fill in the [form](https://docs.google.com/forms/d/e/1FAIpQLScUXEDTksGiqHZp31j7Zp7zlCNV7p_08uViwP_Nbzfn3g6hhw/viewform) to apply for downloading).
    Create a folder `mmscan_data/` and then unzip the files. For the first zip file, put `embodiedscan` under `mmscan_data/embodiedscan_split` and rename it to `embodiedscan-v1`. For the second zip file, put `MMScan-beta-release` under `mmscan_data/MMScan-beta-release` and `embodiedscan-v2` under `mmscan_data/embodiedscan_split`.

    The directory structure should be as below:

    ```
    mmscan_data
    ├── embodiedscan_split
    │   ├──embodiedscan-v1/   # EmbodiedScan v1 data in 'embodiedscan.zip'
    │   ├──embodiedscan-v2/   # EmbodiedScan v2 data in 'embodiedscan-v2-beta.zip'
    ├── MMScan-beta-release   # MMScan data in 'embodiedscan-v2-beta.zip'
    ```

    (b) Prepare the point clouds files, refer to the [guide](data_preparation/README.md) here.

    (c) (Option) Download mmscan llava-form from [huggingface]() if needed, which , you can also generate the llava-form files by running:
3. **Toolkit Usage**

    (a) Dataset tool: The dataset tool in MMScan allows seamless access to data required for various tasks within MMScan.

    - Initialize the dataset for a specific task with:

        ```shell
        from mmscan import MMScan

        # (1) The dataset tool
        my_dataset = MMScan(split='train'/'test'/'val', task='MMScan-VG'/'MMScan-QA')
        # Access a specific sample
        print(my_dataset[index])
        ```
        *Note*: For the test split, we have only made the VG portion publicly available, while the QA portion has not been released.
    - Each dataset item is a dictionary containing data information from three modalities: language, 2D, and 3D.（[Details](https://rbler1234.gitbook.io/mmscan-devkit-tutorial#data-access)）

    (b) Evaluation tool: It is designed to streamline the assessment of model outputs for the MMScan task, providing essential metrics to gauge model performance effectively. We provide three evaluation tools: `VisualGroundingEvaluator`, `QuestionAnsweringEvaluator`, and `GPTEvaluator`. For more details, please refer to the [documentation](https://rbler1234.gitbook.io/mmscan-devkit-tutorial/evaluator).

    ```bash
    from mmscan import MMScan

    # (2) The evaluator tool ('VisualGroundingEvaluator', 'QuestionAnsweringEvaluator', 'GPTEvaluator')
    from mmscan import VisualGroundingEvaluator, QuestionAnsweringEvaluator, GPTEvaluator

    # For VisualGroundingEvaluator and QuestionAnsweringEvaluator, initialize the evaluator in the following way, update the model output to the evaluator, and finally perform the evaluation and save the final results.
    my_evaluator = VisualGroundingEvaluator(show_results=True) / QuestionAnsweringEvaluaton(show_results=True)
    my_evaluator.update(model_output)
    metric_dict = my_evaluator.start_evaluation()

    # For GPTEvaluator, initialize the Evaluator in the following way, and evaluate the model's output using multithreading, finally saving the results to the specified path (tmp_path).
    gpt_evaluator = GPTEvaluator(API_key='XXX')
    metric_dict = gpt_evaluator.load_and_eval(model_output, num_threads=1, tmp_path='XXX')

    ```


</details>

<details>
  <summary>(Benchmark) OST-Bench</summary>


1. **Install requirements**

   ```shell
   cd OST-Bench
   conda activate your_env_name
   pip install -r requirements.txt
   ```

   *Note:* If you want to evaluate open-source models, you need to set up their corresponding environments.

2. **Data Preparation**

    Download the datas of OST-Bench from [kaggle](https://www.kaggle.com/datasets/jinglilin/ost-bench/) / [huggingface](https://huggingface.co/datasets/rbler/OST-Bench) and unzip the image files and the json file, place them as followed:

    ```
    |-data/
    |-- OST_bench.json
    |-- images/
    |----<scan_id folder>
    ```

    For more detail about the json-format data, refer to [documention](https://huggingface.co/datasets/rbler/OST-Bench).
3. **Multi-round Evaluation**

    We provide inference code compatible with both closed-source models (e.g., GPT, Gemini, Claude series) and open-source models (e.g., InternVL2.5, QwenVL2.5, LLaVA-Video, LLaVA-OneVision) on our OST-Bench.

    For closed-source models, please fill in the appropriate API keys in `models/utils/openai_api.py` according to the model you plan to use. For open-source models, follow the Quickstart of [QwenVL](https://github.com/QwenLM/Qwen2.5-VL) / [InternVL](https://internvl.readthedocs.io/en/latest/internvl2.5/quick_start.html) / [LLaVA](https://github.com/LLaVA-VL/LLaVA-NeXT/tree/main) to set up the required environment and download the corresponding checkpoints.

    - Inference

      (1) To perform inference with closed-source models, run the following command:

        ```shell
        python proprietary_baseline.py --rank_num int --model_name str --save_root str
        ```
        Closed-source model inference supports multi-process execution, where `rank_num` specifies the number of processes, `model_name` indicates the model to use, `save_root` is the directory to save the inference results.

      (2) To perform inference with open-source models, run the following command:

        ```shell
        python InternVL/LLaVA/QwenVL_baseline.py --rank_index int --rank_num int --model_path str --save_root str
        ```

        Open-source model inference also supports multi-process execution, where `rank_index` specifies the index of the current process and `model_path` is the path to the model and its weights.

      (3) Our inference code groups the input data into multi-turn dialogues, where each scene corresponds to one dialogue session. These multi-turn dialogues are fed into the model to generate multi-round responses. The results will be saved in `output_dir` as multiple files named `<scan_id>.json`, each containing the model's responses for all turns in that scene, which can be used for inspection or evaluation. Welcome to implement your method under the `models/your_method.py`!

    - Evaluator

      Use our OST evaluator to get the results, the evaluator will return full results over all question types and the average results across three main categories (*Agent Visible Info*, *Agent Object Spatial*，*Agent State* ) and four question formats.

      ```bash
      cd evaluation
      python OST_evaluator.py --result_dir /path/to/save
      ```

4. **Interleaved Evaluation (VLMEvalkit)**

    Our OST-Bench has been integrated into VLMEvalKit. Follow the [QuickStart](https://github.com/open-compass/VLMEvalKit/blob/main/docs/en/Quickstart.md) to get started with VLMEvalKit and evaluate OST-Bench!

    ```
    LMUDATA/
    ├──OST.tsv
    ├──images/
    ├────OST/
    ├──────<scan_id folder>
    ```

    Place the images under `LMUDATA`.When using VLMEvalKit to evaluate OST-Bench, When evaluating the performance of models `llava/qwenvl/InternVL` series, set `max_new_tokens` to 4096 to ensure complete reproducibility of the results.  Additionally, when using the LLaVA_OneVision series of models, set `self.model.config.image_aspect_ratio` = 'pt'  (under `vlmeval/vlm/llava/llava.py`).

    Run the following command to perform evaluation:

    ```shell
    python run.py --model GPT-4o --data OST
    ```

    *Note*: As most VLMEvalKit models do not support multi-turn inference, we provide an interleaved version of OST-Bench, where each sample merges the system prompt, history, and current question into a single turn. Evaluation results may slightly differ from true multi-round settings.

</details>

<details>
  <summary>(Benchmark) MMSI-Bench</summary>

1. **Install requirements**

    ```shell
    cd MMSI-Bench
    pip install -e .
    ```


2. **Data Preparation**

    Download the dataset from [huggingface](https://huggingface.co/datasets/RunsenXu/MMSI-Bench) or load the dataset via huggingface API.
      ```
      from datasets import load_dataset

      dataset = load_dataset("RunsenXu/MMSI-Bench")
      print(dataset)

      # After downloading the parquet file, read each record, decode images from binary, and save them as JPG files.
      import pandas as pd
      import os

      df = pd.read_parquet('MMSI_Bench.parquet')

      output_dir = './images'
      os.makedirs(output_dir, exist_ok=True)

      for idx, row in df.iterrows():
          id_val = row['id']
          images = row['images']  
          question_type = row['question_type']
          question = row['question']
          answer = row['answer']
          thought = row['thought']

          image_paths = []
          if images is not None:
              for n, img_data in enumerate(images):
                  image_path = f"{output_dir}/{id_val}_{n}.jpg"
                  with open(image_path, "wb") as f:
                      f.write(img_data)
                  image_paths.append(image_path)
          else:
              image_paths = []

          print(f"id: {id_val}")
          print(f"images: {image_paths}")
          print(f"question_type: {question_type}")
          print(f"question: {question}")
          print(f"answer: {answer}")
          print(f"thought: {thought}")
          print("-" * 50)
      ```
3. Evaluation

    Please refer to the [evaluation guidelines](https://github.com/open-compass/VLMEvalKit/blob/main/docs/en/Quickstart.md) of [VLMEvalKit](https://github.com/open-compass/VLMEvalKit).

    ```shell
    # api model
    python run.py --model Seed1.5-VL --data MMSI_Bench

    # huggingface model
    python run.py --model Qwen2.5-VL-7B-Instruct --data MMSI_Bench
    ```



</details>

<details>
  <summary>(Benchmark) EgoExo-Bench</summary>

1. **Install requirements**

    ```shell
    cd EgoExo-Bench
    pip install -e .
    ```
2. **Data Preparation**


    (1) EgoExoBench builds upon six publicly available ego–exo datasets. Please download the videos from the following sources:

    * [Ego-Exo4D](https://ego-exo4d-data.org/)
    * [LEMMA](https://sites.google.com/view/lemma-activity)
    * [EgoExoLearn](https://huggingface.co/datasets/hyf015/EgoExoLearn)
    * [TF2023](https://github.com/facebookresearch/Ego-Exo)
    * [EgoMe](https://huggingface.co/datasets/HeqianQiu/EgoMe)
    * [CVMHAT](https://github.com/RuizeHan/CVMHT)

    Place all datasets under the `data/` directory. The dataset structure is as follows:
    ```
    EgoExoBench/
    ├── data/
    │   ├── CVMHAT
    │   	├── data
    │   ├── Ego-Exo4D
    │   	├── takes
    │   ├── EgoExoLearn
    │   ├── EgoMe
    │   ├── LEMMA
    │   ├── TF2023
    │   	├── data
    ```

    (2) For the CVMHAT and TF2023 datasets, we utilize the bounding box annotations to augment the original frames by overlaying bounding boxes that indicate the target person. To generate these bboxes, run the following commands:
    ```
    python data/CVMHAT/tools/process_bbox.py
    python data/TF2023/tools/process_bbox.py
    ```
    (3) Download the EgoExoBench **multiple-choice questions (MCQs)** file [(link)](https://www.kaggle.com/datasets/d481439076f14580fc0fd85fda68e0c832e85fd7600d93d7f90e624731bebdfc) and place it in the `MCQ/` directory.

3. **Evaluation**

    Evaluation is built upon [VLMEvalKit](https://github.com/open-compass/VLMEvalKit).
    ```shell
    # for VLMs that consume small amounts of GPU memory
    torchrun --nproc-per-node=1 run.py --data EgoExoBench_MCQ --model Qwen2.5-VL-7B-Instruct-ForVideo

    # for very large VLMs
    python run.py --data EgoExoBench_MCQ --model Qwen2.5-VL-72B-Instruct-ForVideo
    ```


</details>

## Overview of My-Perception Models
### LLaVA-3D

## Overview of My-Perception Benchmarks

| Benchmark       | Domain                | Tasks                                                   | Input Modality                                       | Data Split & Scale                                     | Access                                      |
|-----------------|-----------------------|----------------------------------------------------------|------------------------------------------------------|--------------------------------------------------------|---------------------------------------------|
| `MMScan`        | Scene Understanding      | Visual Grounding, Question Answering | `point cloud`, `image`, `video`, `text`              | Train/Val/Test, 3M  | [Code](#) / [Data](#) / [Eval](#)           |
| `OST-Bench`     | Scene Spatial Reasoning          | Online Question Answering         | `video`, `text`                                      | Test,10k    | [Code](#) / [Data](#) / [Eval](#)           |
| `MMSI-Bench`    | Scene Spatial Reasoning       | Question Answering                     | `image`, `text`                      | Test,1k    | [Code](#) / [Data](#) / [Eval](#)           |
| `EgoExo-Bench`  | Scene Spatial Reasoning | Question Answering                       | `video`,  `text`                             | Test, 5k          | [Code](#) / [Data](#) / [Eval](#)           |

<div align="center">

## MMScan: A Multi-Modal 3D Scene Dataset with Hierarchical Grounded Language Annotations
[**🌐 Homepage**](https://tai-wang.github.io/mmscan/)  | [**💻 Code&Dataset**](https://github.com/OpenRobotLab/EmbodiedScan/tree/mmscan)  | [**📖 arXiv**](https://arxiv.org/abs/2406.09401)

<img src="assets/MMScan_teaser.png" alt="Dialogue_Teaser" width=70% >
</div>

### **Description :**

MMScan is the **first** largest ever multi-modal 3D scene dataset and benchmark with hierarchical grounded language annotations designed to advance embodied perception. It addresses the limitations of existing datasets by introducing hierarchical, grounded language annotations that span regions, objects, and inter-object relationships. Built with VLM-assisted annotation and human refinement, MMScan contains 1.4M captions across 109k objects and 7.7k regions, supporting over 3.04M samples for Visual Grounding and Question Answering tasks. Extensive benchmarking reveals key challenges and shows that training on MMScan significantly boosts model performance in both standard and real-world settings.

### **Input Modalities :** `🌐 Point Cloud` `🖼️🎞️ image/video` `💬 text`

### **Tasks in MMScan :**
| Task Name              | Input Format                             | Output Format                               | Evaluation Metrics                  |
|------------------------|---------------------------------------------|---------------------------------------------|-------------------------------------|
| 3D Visual Grounding    | `point cloud`/`rgbd images`+`text prompt`                               | `List [3d bounding boxes]`                           | **gTop-k**, AP, AR, AP_C, AR_C                          |
| 3D Question Answering  | `point cloud`/`rgbd images`+`text prompt`(+`List [3d bounding boxes]`)                      | `text response`          | **GPT-score**, EM , Bleu-X, Meteor, CIDEr, SPICE, SimCSE, SBERT                             |



### **Dataset Statics**
MMScan covers both *Inter-target and Single-target* cases, spans two reasoning aspects: *Spatial and Attribute*, and operates at both the *Object and Region levels*.

For the Question Answering task, the dataset contains 1M training samples, 300k validation samples, and 300k test samples. For Visual Grounding, the training/validation/test split includes 850k, 200k, and 200k samples respectively.

<div align="center">
    <img src="assets/mix.png" alt="Dialogue_Teaser" width=90% >
</div>


### **Access**
The code and data for MMScan are located under `./benchmarks/MMScan`.
This directory includes instructions for data download, as well as the mmscan-devkit tools for data loading and evaluation, along with comprehensive documentation.


<div align="center">

## OST-Bench: Evaluating the Capabilities of MLLMs in Online Spatio-temporal Scene Understanding
[**🌐 Homepage**](https://rbler1234.github.io/OSTBench.github.io/)  | [**💻 Code&Dataset**](https://github.com/rbler1234/OST-Bench)  | [**📖 arXiv**](https://arxiv.org/abs/<>)

<img src="assets/OSTBench_teaser.png" alt="Dialogue_Teaser" width=90% >

</div>

### **Description :**

Recent advances in multimodal large language models (MLLMs) have shown remarkable capabilities in integrating vision and language for complex reasoning. While most existing benchmarks evaluate models under offline settings with a fixed set of pre-recorded inputs, we introduce OST-Bench, a benchmark designed to evaluate Online Spatio-Temporal understanding from the perspective of an agent actively exploring a scene. The **Online** aspect emphasizes the need to process and reason over incrementally acquired observations, while the **Spatio-Temporal** component requires integrating current visual inputs with historical memory to support dynamic spatial reasoning. OST-Bench better reflects the challenges of real-world embodied perception. 

### **Input Modalities :** `🎞️ video` `💬 text`

### **Tasks in OST-Bench :**
| Task Name              | Input Format                             | Output Format                               | Evaluation Metrics                  |
|------------------------|---------------------------------------------|---------------------------------------------|-------------------------------------|
| Question Answering    | `rgb video`+`question`(+`options`)                               | `string`/`int`/`float`                           | `EM(string)`/`EM(int)`/`MRA`                         |




### **Dataset Statics**
OST-Bench categorizes questions into three main types: *Agent State*, *Agent-Object Spatial Relationship*, and *Agent Visible Information*, which are further divided into 15 specific subtypes (as shown in the figure).

It covers 1.4K multi-turn dialogues across scenes from ScanNet, Matterport3D, and ARKitScenes, resulting in a total of 10,000 question-answer pairs used as the test set for this benchmark.

<div align="center">
    <img src="assets/OSTBench_data.png" alt="Dialogue_Teaser" width=80% >
</div>


### **Access**
The code and data for OST-Bench are located under `./benchmarks/MMScan`.
This directory includes instructions for data download, as well as the evaluation tools along with comprehensive documentation. OST-Bench is also integrated with [VLMEvalKit](https://github.com/open-compass/VLMEvalKit/tree/main), enabling standardized evaluation within the framework.



<div align="center">

## MMSI-Bench: A Benchmark for Multi-Image Spatial Intelligence
[**🌐 Homepage**](https://runsenxu.com/projects/MMSI_Bench/)  | [**💻 Code&Dataset**](https://github.com/OpenRobotLab/MMSI-Bench)  | [**📖 arXiv**](https://arxiv.org/abs/2505.23764)

<img src="assets/MMSIBench_teaser.jpg" alt="Dialogue_Teaser" width=90% >
</div>

### **Description :**
MMSI-Bench possesses the following unique features:
1. **Multi-image.** We target multi-image spatial reasoning: each of the ten fundamental tasks involves two images, while the multi-step reasoning tasks use more.
2. **High quality.** Every question is fully human-designed—selecting images, crafting questions, carefully designing distractors, and annotating step-by-step reasoning processes.
3. **Aligned with real-world scenarios.** All images depict real-world scenes from domains such as autonomous driving, robotic manipulation, and scene scanning, and every question demands real-world scene understanding and reasoning. We do not use any synthetic data.
4. **Comprehensive and challenging.** We benchmark 34 MLLMs—nearly all leading proprietary and open-source models—and observe a large gap between model and human performance. Most open-source models perform at roughly random-choice level. To the best of our knowledge, our benchmark shows the largest reported model-human gap.
5. **Reasoning processes.** Each sample is annotated with a step-by-step reasoning trace that justifies the correct answer and helps diagnose model errors.

### **Input Modalities :** `🖼️ image` `💬 text`

### **Tasks in OST-Bench :**
| Task Name              | Input Format                             | Output Format                               | Evaluation Metrics                  |
|------------------------|---------------------------------------------|---------------------------------------------|-------------------------------------|
| Question Answering    | `rgb image`+`question`+`options`                               | `string`                           | `EM(string)`                         |




### **Dataset Statics**
MMSI-Bench categorizes tasks around three core spatial elements: camera, object, and region, focusing on their positional relationships, attributes, and motion. There are six types of positional relationships: camera-camera, camera-object, camera-region, object-object, object-region, and region-region. The benchmark also includes two types of attributes (measurement and appearance), two types of motion (camera and object), and one multi-step reasoning category. 

Annotated by six 3D vision researchers from diverse real-world scene images, it includes 1,000 challenging questions used as the test set.

<div align="center">
    <img src="assets/MMSIBench_data.png" alt="Dialogue_Teaser" width=80% >
</div>


### **Access**
The code and data for MMSI-Bench are located under `./benchmarks/MMScan`.
This directory includes instructions for data download, as well as the evaluation tools along with comprehensive documentation. MMSI-Bench is integrated with [VLMEvalKit](https://github.com/open-compass/VLMEvalKit/tree/main), enabling standardized evaluation within the framework.