# Awesome Gaussian Splatting [![Awesome](https://awesome.re/badge.svg)](https://awesome.re)

A curated list of latest research papers, projects and resources related to Gaussian Splatting. Content is automatically updated daily.

> Last Update: 2026-02-10 01:35:33

## 📰 Latest Updates

🚀 **[2026-02] Major Feature Update — v2.0**
- **Unified CLI**: Single entry point `python main.py` with subcommands: `init`, `search`, `suggest`, `export-bib`, `readme`
- **Interactive Configuration Wizard**: Run `python main.py init` to set up keywords, domains, time range, and API keys step-by-step
- **Custom Time Range Filtering**: Support relative periods (`6m`, `1y`, `2y`) and absolute date ranges (`2024-01-01` to `2025-06-01`)
- **Smart Link Extraction**: Automatically extracts and classifies GitHub, project page, dataset, video, demo, and HuggingFace links from paper abstracts
- **BibTeX Export**: Fetch BibTeX from arXiv and export to `.bib` files with category/date filters
- **LLM Keyword Suggestion**: Paste a few paper titles or arXiv IDs, and an LLM automatically generates optimized search keywords
- **arXiv Domain Filtering**: Restrict searches to specific arXiv categories (e.g., `cs.CV`, `cs.GR`)

🔧 **[2025-06-26] HTTP 301 Redirect Issue Completely Resolved!** 
- Implemented multi-layer fallback strategy to thoroughly solve network compatibility issues

🔧 **[2025-06-26] Configurable Search Keywords Feature Added!**
- You can now customize search keywords by modifying `data/search_config.json`

- View detailed updates: [News.md](News.md) 📋

---

## Categories

- [Cross-Modal Generation](#cross-modal-generation) (90 papers) - Methods transforming text or 2D images into 3D assets.
- [Dynamic & Articulated Modeling](#dynamic-&-articulated-modeling) (9 papers) - Creation of 3D objects with moving parts or controllable skeletal structures.
- [Scene & View Synthesis](#scene-&-view-synthesis) (114 papers) - Focus on reconstructing complex scenes or synthesizing new perspectives from limited data.
- [Surface & Appearance Modeling](#surface-&-appearance-modeling) (25 papers) - Techniques for generating realistic textures, materials, and surface details.



## Table of Contents

- [Categorized Papers](#categorized-papers)
- [Classic Papers](#classic-papers)
- [Open Source Projects](#open-source-projects)
- [Applications](#applications)
- [Tutorials & Blogs](#tutorials--blogs)





## Categorized Papers

### Cross-Modal Generation

*Showing the latest 50 out of 90 papers*

- **[FastPhysGS: Accelerating Physics-based Dynamic 3DGS Simulation via Interior Completion and Adaptive Optimization](https://arxiv.org/abs/2602.01723v1)**  
  Authors: Yikun Ma, Yiqing Li, Jingwen Ye, Zhongkai Wu, Weidong Zhang, Lin Gao, Zhi Jin  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2602.01723v1.pdf)  
  Keywords: image-to-3d  
- **[RoamScene3D: Immersive Text-to-3D Scene Generation via Adaptive Object-aware Roaming](https://arxiv.org/abs/2601.19433v1)**  
  Authors: Jisheng Chu, Wenrui Li, Rui Zhao, Wangmeng Zuo, Shifeng Chen, Xiaopeng Fan  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2601.19433v1.pdf) | [![GitHub](https://img.shields.io/github/stars/JS-CHU/RoamScene3D?style=social)](https://github.com/JS-CHU/RoamScene3D)  
  Keywords: text-to-3d  
- **[Spherical Geometry Diffusion: Generating High-quality 3D Face Geometry via Sphere-anchored Representations](https://arxiv.org/abs/2601.13371v1)**  
  Authors: Junyi Zhang, Yiming Wang, Yunhong Lu, Qichao Wang, Wenzhe Qian, Xiaoyin Xu, David Gu, Min Zhang  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2601.13371v1.pdf)  
  Keywords: texture synthesis, text-to-3d  
- **[A Comparative Study of 3D Model Acquisition Methods for Synthetic Data Generation of Agricultural Products](https://arxiv.org/abs/2601.03784v1)**  
  Authors: Steven Moonen, Rob Salaets, Kenneth Batstone, Abdellatif Bey-Temsamani, Nick Michiels  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2601.03784v1.pdf)  
  Keywords: image-to-3d  
- **[UnrealPose: Leveraging Game Engine Kinematics for Large-Scale Synthetic Human Pose Data](https://arxiv.org/abs/2601.00991v1)**  
  Authors: Joshua Kawaguchi, Saad Manzur, Emily Gao Wang, Maitreyi Sinha, Bryan Vela, Yunxi Wang, Brandon Vela, Wayne B. Hayes  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2601.00991v1.pdf)  
  Keywords: image-to-3d  
- **[SAM 3D for 3D Object Reconstruction from Remote Sensing Images](https://arxiv.org/abs/2512.22452v1)**  
  Authors: Junsheng Yao, Lichao Mou, Qingyu Li  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2512.22452v1.pdf)  
  Keywords: image-to-3d  
- **[SS4D: Native 4D Generative Model via Structured Spacetime Latents](https://arxiv.org/abs/2512.14284v1)**  
  Authors: Zhibing Li, Mengchen Zhang, Tong Wu, Jing Tan, Jiaqi Wang, Dahua Lin  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2512.14284v1.pdf)  
  Keywords: image-to-3d  
- **[Feedforward 3D Editing via Text-Steerable Image-to-3D](https://arxiv.org/abs/2512.13678v1)**  
  Authors: Ziqi Ma, Hongqiao Chen, Yisong Yue, Georgia Gkioxari  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2512.13678v1.pdf) | [![Project](https://img.shields.io/badge/-Project-blue)](https://glab-caltech.github.io/steer3d)  
  Keywords: image-to-3d  
- **[Particulate: Feed-Forward 3D Object Articulation](https://arxiv.org/abs/2512.11798v1)**  
  Authors: Ruining Li, Yuxin Yao, Chuanxia Zheng, Christian Rupprecht, Joan Lasenby, Shangzhe Wu, Andrea Vedaldi  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2512.11798v1.pdf)  
  Keywords: image-to-3d, 3d generator  
- **[Are We Ready for RL in Text-to-3D Generation? A Progressive Investigation](https://arxiv.org/abs/2512.10949v1)**  
  Authors: Yiwen Tang, Zoey Guo, Kaixin Zhu, Ray Zhang, Qizhi Chen, Dongzhi Jiang, Junli Liu, Bohan Zeng, Haoming Song, Delin Qu, Tianyi Bai, Dan Xu, Wentao Zhang, Bin Zhao  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2512.10949v1.pdf) | [![GitHub](https://img.shields.io/github/stars/Ivan-Tang-3D/3DGen-R1?style=social)](https://github.com/Ivan-Tang-3D/3DGen-R1)  
  Keywords: text-to-3d  

### Dynamic & Articulated Modeling

- **[GAOT: Generating Articulated Objects Through Text-Guided Diffusion Models](https://arxiv.org/abs/2512.03566v1)**  
  Authors: Hao Sun, Lei Fan, Donglin Di, Shaohui Liu  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2512.03566v1.pdf)  
  Keywords: articulated object generation  
- **[FreeArt3D: Training-Free Articulated Object Generation using 3D Diffusion](https://arxiv.org/abs/2510.25765v2)**  
  Authors: Chuhao Chen, Isabella Liu, Xinyue Wei, Hao Su, Minghua Liu  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2510.25765v2.pdf) | [![Project](https://img.shields.io/badge/-Project-blue)](https://czzzzh.github.io/FreeArt3D)  
  Keywords: articulated object generation  
- **[Pain in 3D: Generating Controllable Synthetic Faces for Automated Pain Assessment](https://arxiv.org/abs/2509.16727v4)**  
  Authors: Xin Lei Lin, Soroush Mehraban, Abhishek Moturu, Babak Taati  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2509.16727v4.pdf)  
  Keywords: rigging  
- **[Empowering Children to Create AI-Enabled Augmented Reality Experiences](https://arxiv.org/abs/2508.08467v1)**  
  Authors: Lei Zhang, Shuyao Zhou, Amna Liaqat, Tinney Mak, Brian Berengard, Emily Qian, Andrés Monroy-Hernández  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2508.08467v1.pdf)  
  Keywords: rigging, text-to-3d  
- **[Guiding Diffusion-Based Articulated Object Generation by Partial Point Cloud Alignment and Physical Plausibility Constraints](https://arxiv.org/abs/2508.00558v1)**  
  Authors: Jens U. Kreber, Joerg Stueckler  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2508.00558v1.pdf)  
  Keywords: articulated object generation  
- **[From One to More: Contextual Part Latents for 3D Generation](https://arxiv.org/abs/2507.08772v2)**  
  Authors: Shaocong Dong, Lihe Ding, Xiao Chen, Yaokun Li, Yuxin Wang, Yucheng Wang, Qi Wang, Jaehyeok Kim, Chenjian Gao, Zhanpeng Huang, Zibin Wang, Tianfan Xue, Dan Xu  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2507.08772v2.pdf)  
  Keywords: articulated object generation  
- **[EmbodiedGen: Towards a Generative 3D World Engine for Embodied Intelligence](https://arxiv.org/abs/2506.10600v2)**  
  Authors: Xinjie Wang, Liu Liu, Yu Cao, Ruiqi Wu, Wenkang Qin, Dehui Wang, Wei Sui, Zhizhong Su  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2506.10600v2.pdf) | [![Project](https://img.shields.io/badge/-Project-blue)](https://horizonrobotics.github.io/robot_lab/embodied_gen/index.html)  
  Keywords: image-to-3d, text-to-3d, articulated object generation  
- **[AdaHuman: Animatable Detailed 3D Human Generation with Compositional Multiview Diffusion](https://arxiv.org/abs/2505.24877v1)**  
  Authors: Yangyi Huang, Ye Yuan, Xueting Li, Jan Kautz, Umar Iqbal  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2505.24877v1.pdf)  
  Keywords: rigging, image-to-3d  
- **[DIPO: Dual-State Images Controlled Articulated Object Generation Powered by Diverse Data](https://arxiv.org/abs/2505.20460v2)**  
  Authors: Ruiqi Wu, Xinjie Wang, Liu Liu, Chunle Guo, Jiaxiong Qiu, Chongyi Li, Lichao Huang, Zhizhong Su, Ming-Ming Cheng  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2505.20460v2.pdf)  
  Keywords: articulated object generation  

### Scene & View Synthesis

*Showing the latest 50 out of 114 papers*

- **[Seeing Through Clutter: Structured 3D Scene Reconstruction via Iterative Object Removal](https://arxiv.org/abs/2602.04053v1)**  
  Authors: Rio Aguina-Kang, Kevin James Blackburn-Matzen, Thibault Groueix, Vladimir Kim, Matheus Gadelha  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2602.04053v1.pdf) | [![Project](https://img.shields.io/badge/-Project-blue)](https://rioak.github.io/seeingthroughclutter)  
  Keywords: 3d scene reconstruction  
- **[EUGens: Efficient, Unified, and General Dense Layers](https://arxiv.org/abs/2601.22563v2)**  
  Authors: Sang Min Kim, Byeongchan Kim, Arijit Sehanobish, Somnath Basu Roy Chowdhury, Rahul Kidambi, Dongseok Shim, Avinava Dubey, Snigdha Chaturvedi, Min-hwan Oh, Krzysztof Choromanski  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2601.22563v2.pdf)  
  Keywords: 3d scene reconstruction  
- **[Splat-Portrait: Generalizing Talking Heads with Gaussian Splatting](https://arxiv.org/abs/2601.18633v1)**  
  Authors: Tong Shi, Melonie de Almeida, Daniela Ivanova, Nicolas Pugeault, Paul Henderson  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2601.18633v1.pdf) | [![GitHub](https://img.shields.io/github/stars/stonewalking/Splat-portrait?style=social)](https://github.com/stonewalking/Splat-portrait)  
  Keywords: novel view synthesis  
- **[POTR: Post-Training 3DGS Compression](https://arxiv.org/abs/2601.14821v1)**  
  Authors: Bert Ramlot, Martijn Courteaux, Peter Lambert, Glenn Van Wallendael  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2601.14821v1.pdf)  
  Keywords: 3d scene reconstruction, novel view synthesis  
- **[studentSplat: Your Student Model Learns Single-view 3D Gaussian Splatting](https://arxiv.org/abs/2601.11772v1)**  
  Authors: Yimu Pan, Hongda Mao, Qingshuang Chen, Yelin Kim  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2601.11772v1.pdf)  
  Keywords: 3d scene reconstruction  
- **[SyncTwin: Fast Digital Twin Construction and Synchronization for Safe Robotic Grasping](https://arxiv.org/abs/2601.09920v1)**  
  Authors: Ruopeng Huang, Boyu Yang, Wenlong Gui, Jeremy Morgan, Erdem Biyik, Jiachen Li  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2601.09920v1.pdf)  
  Keywords: 3d scene reconstruction  
- **[HERE: Hierarchical Active Exploration of Radiance Field with Epistemic Uncertainty Minimization](https://arxiv.org/abs/2601.07242v1)**  
  Authors: Taekbeom Lee, Dabin Kim, Youngseok Jang, H. Jin Kim  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2601.07242v1.pdf)  
  Keywords: 3d scene reconstruction  
- **[Rotate Your Character: Revisiting Video Diffusion Models for High-Quality 3D Character Generation](https://arxiv.org/abs/2601.05722v1)**  
  Authors: Jin Wang, Jianxiang Lu, Comi Chen, Guangzheng Xu, Haoyu Yang, Peng Chen, Na Zhang, Yifan Xu, Longhuang Wu, Shuai Shao, Qinglin Lu, Ping Luo  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2601.05722v1.pdf)  
  Keywords: novel view synthesis  
- **[360-GeoGS: Geometrically Consistent Feed-Forward 3D Gaussian Splatting Reconstruction for 360 Images](https://arxiv.org/abs/2601.02102v1)**  
  Authors: Jiaqi Yao, Zhongmiao Yan, Jingyi Xu, Songpengcheng Xia, Yan Xiang, Ling Pei  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2601.02102v1.pdf)  
  Keywords: 3d scene reconstruction  
- **[Learning Dynamic Scene Reconstruction with Sinusoidal Geometric Priors](https://arxiv.org/abs/2512.22295v1)**  
  Authors: Tian Guo, Hui Yuan, Philip Xu, David Elizondo  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2512.22295v1.pdf)  
  Keywords: 3d scene reconstruction  

### Surface & Appearance Modeling

- **[Neural Particle Automata: Learning Self-Organizing Particle Dynamics](https://arxiv.org/abs/2601.16096v1)**  
  Authors: Hyunsoo Kim, Ehsan Pajouheshgar, Sabine Süsstrunk, Wenzel Jakob, Jinah Park  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2601.16096v1.pdf)  
  Keywords: texture synthesis  
- **[ThermoSplat: Cross-Modal 3D Gaussian Splatting with Feature Modulation and Geometry Decoupling](https://arxiv.org/abs/2601.15897v1)**  
  Authors: Zhaoqi Su, Shihai Chen, Xinyan Lin, Liqin Huang, Zhipeng Su, Xiaoqiang Lu  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2601.15897v1.pdf)  
  Keywords: texture synthesis  
- **[Spherical Geometry Diffusion: Generating High-quality 3D Face Geometry via Sphere-anchored Representations](https://arxiv.org/abs/2601.13371v1)**  
  Authors: Junyi Zhang, Yiming Wang, Yunhong Lu, Qichao Wang, Wenzhe Qian, Xiaoyin Xu, David Gu, Min Zhang  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2601.13371v1.pdf)  
  Keywords: texture synthesis, text-to-3d  
- **[OSCAR: Optical-aware Semantic Control for Aleatoric Refinement in Sar-to-Optical Translation](https://arxiv.org/abs/2601.06835v1)**  
  Authors: Hyunseo Lee, Sang Min Kim, Ho Kyung Shin, Taeheon Kim, Woo-Jeoung Nam  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2601.06835v1.pdf)  
  Keywords: texture synthesis  
- **[Conditional Morphogenesis: Emergent Generation of Structural Digits via Neural Cellular Automata](https://arxiv.org/abs/2512.08360v1)**  
  Authors: Ali Sakour  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2512.08360v1.pdf)  
  Keywords: texture synthesis  
- **[TEXTRIX: Latent Attribute Grid for Native Texture Generation and Beyond](https://arxiv.org/abs/2512.02993v1)**  
  Authors: Yifei Zeng, Yajie Bao, Jiachen Qian, Shuang Wu, Youtian Lin, Hao Zhu, Buyu Li, Feihu Zhang, Xun Cao, Yao Yao  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2512.02993v1.pdf)  
  Keywords: texture synthesis  
- **[DensiCrafter: Physically-Constrained Generation and Fabrication of Self-Supporting Hollow Structures](https://arxiv.org/abs/2511.09298v2)**  
  Authors: Shengqi Dang, Fu Chai, Jiaxin Li, Chao Yuan, Wei Ye, Nan Cao  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2511.09298v2.pdf)  
  Keywords: texture synthesis, text-to-3d  
- **[AvatarTex: High-Fidelity Facial Texture Reconstruction from Single-Image Stylized Avatars](https://arxiv.org/abs/2511.06721v1)**  
  Authors: Yuda Qiu, Zitong Xiao, Yiwei Zuo, Zisheng Ye, Weikai Chen, Xiaoguang Han  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2511.06721v1.pdf)  
  Keywords: texture synthesis  
- **[Oitijjo-3D: Generative AI Framework for Rapid 3D Heritage Reconstruction from Street View Imagery](https://arxiv.org/abs/2511.00362v1)**  
  Authors: Momen Khandoker Ope, Akif Islam, Mohd Ruhul Ameen, Abu Saleh Musa Miah, Md Rashedul Islam, Jungpil Shin  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2511.00362v1.pdf)  
  Keywords: texture synthesis, image-to-3d  
- **[Scale-DiT: Ultra-High-Resolution Image Generation with Hierarchical Local Attention](https://arxiv.org/abs/2510.16325v1)**  
  Authors: Yuyao Zhang, Yu-Wing Tai  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2510.16325v1.pdf)  
  Keywords: texture synthesis  



## Classic Papers
- **[3D Gaussian Splatting for Real-Time Radiance Field Rendering](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/)** (SIGGRAPH 2023)  
  Authors: Bernhard Kerbl, Georgios Kopanas, Thomas Leimkühler, George Drettakis  
  Code: 🔗 [GitHub](https://github.com/graphdeco-inria/gaussian-splatting)  
  Keywords: Real-time Rendering, Neural Rendering, Point-based Graphics

- **[Instruct-4DGS: Efficient Dynamic Scene Editing via 4D Gaussian-based Static-Dynamic Separation](https://hanbyelcho.info/instruct-4dgs/)** (CVPR 2025)  
  Authors: Hanbyel Cho, Juhyeon Kwon, et al.  
  Paper: 📄 [arXiv](https://arxiv.org/abs/2502.02091)  
  Code: 🔗 [GitHub](https://github.com/juhyeon-kwon/efficient_4d_gaussian_editing)  
  Keywords: Dynamic Scene Editing, 4D Gaussian Splatting, Static-Dynamic Separation

## Open Source Projects
- [gaussian-splatting](https://github.com/graphdeco-inria/gaussian-splatting) - Original implementation of 3D Gaussian Splatting
- [taichi-3d-gaussian-splatting](https://github.com/wanmeihuali/taichi-3d-gaussian-splatting) - 3D Gaussian Splatting implemented in Taichi

## Applications
- [3D Gaussian Splatting for Real-Time Radiance Field Rendering Demo](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/) - Online Demo

## Tutorials & Blogs
- [Introduction to 3D Gaussian Splatting](https://github.com/graphdeco-inria/gaussian-splatting) - Official Tutorial

## 📋 Project Features

### 🛠️ Core Features
- **Unified CLI** (`main.py`): Single entry point with `init`, `search`, `suggest`, `export-bib`, `readme` subcommands
- **Interactive Config Wizard**: Guided setup for keywords, domains, time range, and API keys via `python main.py init`
- **Custom Search Keywords**: Configure keywords for title, abstract, or both; with arXiv domain filtering (`cs.CV`, `cs.GR`, etc.)
- **Time Range Filtering**: Relative periods (`30d`, `6m`, `1y`, `2y`) or absolute date ranges (`YYYY-MM-DD` to `YYYY-MM-DD`)
- **Smart Link Extraction**: Auto-classifies URLs from abstracts into GitHub, project page, dataset, video, demo, HuggingFace links
- **BibTeX Export**: Fetch BibTeX from arXiv official API; export to `.bib` files with category and date filters
- **LLM Keyword Suggestion**: Input paper titles or arXiv IDs to auto-generate optimized search keywords via OpenAI-compatible API
- **Automated Paper Collection**: Daily automatic crawling with GitHub Actions
- **Intelligent Classification**: Auto-categorize papers into 14+ topics (Acceleration, Dynamic Scenes, SLAM, etc.)

### 🛠️ Technical Features
- **Robust Error Handling**: Multi-layer retry and fallback strategies ensure stable operation
- **GitHub Actions Integration**: Automated CI/CD workflows for daily updates
- **Multi-type Link Badges**: README entries display PDF, GitHub (with stars), Project, Dataset, Video, Demo, HuggingFace, and Citation badges
- **Detailed Logging**: Comprehensive logging for debugging and monitoring
- **Cross-Platform**: Support for Windows/Linux/macOS

### 📚 Data Output
- **Paper JSON files** (`data/papers_YYYY-MM-DD.json`): Full paper metadata with title, authors, abstract, links, keywords, BibTeX
- **BibTeX files** (`output/*.bib`): Ready-to-use bibliography files for LaTeX
- **Auto-generated README**: Categorized and formatted paper listings

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Interactive Setup (Recommended)

```bash
python main.py init
```

This wizard walks you through:
- Setting search keywords (for title, abstract, or both)
- Selecting arXiv domains (e.g., `cs.CV`, `cs.GR`, `cs.AI`)
- Configuring time range (relative like `6m`/`1y`, or absolute dates)
- Setting max results
- Optionally configuring an OpenAI-compatible API key for keyword suggestion

### 3. Search Papers

```bash
# Search with settings from user_config.json
python main.py search

# Override: fetch 200 papers from the last 6 months, include BibTeX
python main.py search --max-results 200 --recent 6m --bibtex

# Search with absolute date range
python main.py search --date-from 2024-01-01 --date-to 2025-01-01

# Include citation counts from Semantic Scholar
python main.py search --citations
```

### 4. Export BibTeX

```bash
# Export all papers from the latest data file
python main.py export-bib --output output/references.bib

# Export only "Dynamic Scene" papers
python main.py export-bib --category "Dynamic Scene" --output output/dynamic.bib

# Export papers from a specific date range
python main.py export-bib --date-from 2024-06-01 --date-to 2025-01-01 --output output/recent.bib
```

### 5. LLM Keyword Suggestion

```bash
# Generate keywords from paper titles
python main.py suggest --titles "3D Gaussian Splatting for Real-Time Rendering" "Dynamic 3D Gaussians"

# Generate from arXiv IDs (auto-fetches titles)
python main.py suggest --arxiv-ids 2308.04079 2311.12897

# Auto-write suggested keywords to config
python main.py suggest --titles "NeRF" "Gaussian Splatting" --apply

# Use a custom API endpoint (e.g., DeepSeek)
python main.py suggest --titles "Paper Title" --base-url https://api.deepseek.com/v1 --api-key sk-xxx --model deepseek-chat
```

### 6. Generate README

```bash
# Basic README
python main.py readme

# Include latest papers section and abstracts
python main.py readme --show-latest --show-abstracts
```

### Configuration File

All settings are stored in `data/user_config.json`:

```json
{
  "search": {
    "keywords": {
      "both_abstract_and_title": ["gaussian splatting", "3d gaussian"],
      "abstract_only": ["neural radiance field gaussian"],
      "title_only": ["3D scene reconstruction"]
    },
    "domains": ["cs.CV", "cs.GR"],
    "time_range": {
      "mode": "relative",
      "relative": "1y"
    },
    "max_results": 500
  },
  "api_keys": {
    "openai_api_key": "",
    "openai_base_url": "https://api.openai.com/v1",
    "openai_model": "gpt-4o-mini"
  }
}
```

## Contribution Guidelines
Feel free to submit Pull Requests to improve this list! Please follow these formats:
- Paper entry format: `**[Paper Title](link)** - Brief description`
- Project entry format: `[Project Name](link) - Project description`

## License
[![CC0](https://licensebuttons.net/p/zero/1.0/88x31.png)](https://creativecommons.org/publicdomain/zero/1.0/) 