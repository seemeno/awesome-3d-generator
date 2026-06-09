# Awesome Gaussian Splatting [![Awesome](https://awesome.re/badge.svg)](https://awesome.re)

A curated list of latest research papers, projects and resources related to Gaussian Splatting. Content is automatically updated daily.

> Last Update: 2026-06-09 01:48:16

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

- [Cross-Modal Generation](#cross-modal-generation) (96 papers) - Methods transforming text or 2D images into 3D assets.
- [Dynamic & Articulated Modeling](#dynamic-&-articulated-modeling) (13 papers) - Creation of 3D objects with moving parts or controllable skeletal structures.
- [Scene & View Synthesis](#scene-&-view-synthesis) (107 papers) - Focus on reconstructing complex scenes or synthesizing new perspectives from limited data.
- [Surface & Appearance Modeling](#surface-&-appearance-modeling) (24 papers) - Techniques for generating realistic textures, materials, and surface details.



## Table of Contents

- [Latest Papers](#latest-papers)
- [Categorized Papers](#categorized-papers)
- [Classic Papers](#classic-papers)
- [Open Source Projects](#open-source-projects)
- [Applications](#applications)
- [Tutorials & Blogs](#tutorials--blogs)



## Latest Papers
> 🔄 Updated Daily

### September 2025
- **[PAD3R: Pose-Aware Dynamic 3D Reconstruction from Casual Videos](https://arxiv.org/abs/2509.25183v1)**  
  Authors: Ting-Hsuan Liao, Haowen Liu, Yiran Xu, Songwei Ge, Gengshan Yang, Jia-Bin Huang  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2509.25183v1.pdf)  
  <details><summary>Abstract</summary>

  We present PAD3R, a method for reconstructing deformable 3D objects from casually captured, unposed monocular videos. Unlike existing approaches, PAD3R handles long video sequences featuring substantial object deformation, large-scale camera movement, and limited view coverage that typically challenge conventional systems. At its core, our approach trains a personalized, object-centric pose estimator, supervised by a pre-trained image-to-3D model. This guides the optimization of deformable 3D Gaussian representation. The optimization is further regularized by long-term 2D point tracking over the entire input video. By combining generative priors and differentiable rendering, PAD3R reconstructs high-fidelity, articulated 3D representations of objects in a category-agnostic way. Extensive qualitative and quantitative results show that PAD3R is robust and generalizes well across challenging scenarios, highlighting its potential for dynamic scene understanding and 3D content creation.
  </details>  
  Keywords: image-to-3d  
- **[Unsupervised Representation Learning for 3D Mesh Parameterization with Semantic and Visibility Objectives](https://arxiv.org/abs/2509.25094v3)**  
  Authors: AmirHossein Zamani, Bruno Roy, Arianna Rampini  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2509.25094v3.pdf) | [![GitHub](https://img.shields.io/github/stars/AHHHZ975/Semantic-Visibility-UV-Param?style=social)](https://github.com/AHHHZ975/Semantic-Visibility-UV-Param)  
  <details><summary>Abstract</summary>

  Recent 3D generative models produce high-quality textures for 3D mesh objects. However, they commonly rely on the heavy assumption that input 3D meshes are accompanied by manual mesh parameterization (UV mapping), a manual task that requires both technical precision and artistic judgment. Industry surveys show that this process often accounts for a significant share of asset creation, creating a major bottleneck for 3D content creators. Moreover, existing automatic methods often ignore two perceptually important criteria: (1) semantic awareness (UV charts should align semantically similar 3D parts across shapes) and (2) visibility awareness (cutting seams should lie in regions unlikely to be seen). To overcome these shortcomings and to automate the mesh parameterization process, we present an unsupervised differentiable framework that augments standard geometry-preserving UV learning with semantic- and visibility-aware objectives. For semantic-awareness, our pipeline (i) segments the mesh into semantic 3D parts, (ii) applies an unsupervised learned per-part UV-parameterization backbone,...
  </details>  
  Keywords: uv mapping  
- **[UniLat3D: Geometry-Appearance Unified Latents for Single-Stage 3D Generation](https://arxiv.org/abs/2509.25079v1)**  
  Authors: Guanjun Wu, Jiemin Fang, Chen Yang, Sikuang Li, Taoran Yi, Jia Lu, Zanwei Zhou, Jiazhong Cen, Lingxi Xie, Xiaopeng Zhang, Wei Wei, Wenyu Liu, Xinggang Wang, Qi Tian  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2509.25079v1.pdf) | [![Project](https://img.shields.io/badge/-Project-blue)](https://unilat3d.github.io)  
  <details><summary>Abstract</summary>

  High-fidelity 3D asset generation is crucial for various industries. While recent 3D pretrained models show strong capability in producing realistic content, most are built upon diffusion models and follow a two-stage pipeline that first generates geometry and then synthesizes appearance. Such a decoupled design tends to produce geometry-texture misalignment and non-negligible cost. In this paper, we propose UniLat3D, a unified framework that encodes geometry and appearance in a single latent space, enabling direct single-stage generation. Our key contribution is a geometry-appearance Unified VAE, which compresses high-resolution sparse features into a compact latent representation -- UniLat. UniLat integrates structural and visual information into a dense low-resolution latent, which can be efficiently decoded into diverse 3D formats, e.g., 3D Gaussians and meshes. Based on this unified representation, we train a single flow-matching model to map Gaussian noise directly into UniLat, eliminating redundant stages. Trained solely on public datasets, UniLat3D produces high-quality 3D...
  </details>  
  Keywords: 3d asset generation  
- **[GEM: 3D Gaussian Splatting for Efficient and Accurate Cryo-EM Reconstruction](https://arxiv.org/abs/2509.25075v2)**  
  Authors: Huaizhi Qu, Xiao Wang, Gengwei Zhang, Jie Peng, Tianlong Chen  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2509.25075v2.pdf) | [![GitHub](https://img.shields.io/github/stars/UNITES-Lab/GEM?style=social)](https://github.com/UNITES-Lab/GEM)  
  <details><summary>Abstract</summary>

  Cryo-electron microscopy (cryo-EM) has become a central tool for high-resolution structural biology, yet the massive scale of datasets (often exceeding 100k particle images) renders 3D reconstruction both computationally expensive and memory intensive. Traditional Fourier-space methods are efficient but lose fidelity due to repeated transforms, while recent real-space approaches based on neural radiance fields (NeRFs) improve accuracy but incur cubic memory and computation overhead. Therefore, we introduce GEM, a novel cryo-EM reconstruction framework built on 3D Gaussian Splatting (3DGS) that operates directly in real-space while maintaining high efficiency. Instead of modeling the entire density volume, GEM represents proteins with compact 3D Gaussians, each parameterized by only 11 values. To further improve the training efficiency, we designed a novel gradient computation to 3D Gaussians that contribute to each voxel. This design substantially reduced both memory footprint and training cost. On standard cryo-EM benchmarks, GEM achieves up to 48% faster training and 12%...
  </details>  
- **[Light-SQ: Structure-aware Shape Abstraction with Superquadrics for Generated Meshes](https://arxiv.org/abs/2509.24986v1)**  
  Authors: Yuhan Wang, Weikai Chen, Zeyu Hu, Runze Zhang, Yingda Yin, Ruoyu Wu, Keyang Luo, Shengju Qian, Yiyan Ma, Hongyi Li, Yuan Gao, Yuhuan Zhou, Hao Luo, Wan Wang, Xiaobin Shen, Zhaowei Li, Kuixin Zhu, Chuanlang Hong, Yueyue Wang, Lijie Feng, Xin Wang, Chen Change Loy  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2509.24986v1.pdf)  
  <details><summary>Abstract</summary>

  In user-generated-content (UGC) applications, non-expert users often rely on image-to-3D generative models to create 3D assets. In this context, primitive-based shape abstraction offers a promising solution for UGC scenarios by compressing high-resolution meshes into compact, editable representations. Towards this end, effective shape abstraction must therefore be structure-aware, characterized by low overlap between primitives, part-aware alignment, and primitive compactness. We present Light-SQ, a novel superquadric-based optimization framework that explicitly emphasizes structure-awareness from three aspects. (a) We introduce SDF carving to iteratively udpate the target signed distance field, discouraging overlap between primitives. (b) We propose a block-regrow-fill strategy guided by structure-aware volumetric decomposition, enabling structural partitioning to drive primitive placement. (c) We implement adaptive residual pruning based on SDF update history to surpress over-segmentation and ensure compact results. In addition, Light-SQ supports multiscale fitting, enabling localized refinement to preserve fine geometric details. To evaluate our method, we introduce 3DGen-Prim, a benchmark extending...
  </details>  
  Keywords: image-to-3d  
- **[UP2You: Fast Reconstruction of Yourself from Unconstrained Photo Collections](https://arxiv.org/abs/2509.24817v3)**  
  Authors: Zeyu Cai, Ziyang Li, Xiaoben Li, Boqian Li, Zeyu Wang, Zhenyu Zhang, Yuliang Xiu  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2509.24817v3.pdf) | [![Project](https://img.shields.io/badge/-Project-blue)](https://zcai0612.github.io/UP2You)  
  <details><summary>Abstract</summary>

  We present UP2You, the first tuning-free solution for reconstructing high-fidelity 3D clothed portraits from extremely unconstrained in-the-wild 2D photos. Unlike previous approaches that require "clean" inputs (e.g., full-body images with minimal occlusions, or well-calibrated cross-view captures), UP2You directly processes raw, unstructured photographs, which may vary significantly in pose, viewpoint, cropping, and occlusion. Instead of compressing data into tokens for slow online text-to-3D optimization, we introduce a data rectifier paradigm that efficiently converts unconstrained inputs into clean, orthogonal multi-view images in a single forward pass within seconds, simplifying the 3D reconstruction. Central to UP2You is a pose-correlated feature aggregation module (PCFA), that selectively fuses information from multiple reference images w.r.t. target poses, enabling better identity preservation and nearly constant memory footprint, with more observations. We also introduce a perceiver-based multi-reference shape predictor, removing the need for pre-captured body templates. Extensive experiments on 4D-Dress, PuzzleIOI, and in-the-wild captures demonstrate that UP2You consistently...
  </details>  
  Keywords: text-to-3d  
- **[RapidMV: Leveraging Spatio-Angular Representations for Efficient and Consistent Text-to-Multi-View Synthesis](https://arxiv.org/abs/2509.24410v1)**  
  Authors: Seungwook Kim, Yichun Shi, Kejie Li, Minsu Cho, Peng Wang  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2509.24410v1.pdf)  
  <details><summary>Abstract</summary>

  Generating synthetic multi-view images from a text prompt is an essential bridge to generating synthetic 3D assets. In this work, we introduce RapidMV, a novel text-to-multi-view generative model that can produce 32 multi-view synthetic images in just around 5 seconds. In essence, we propose a novel spatio-angular latent space, encoding both the spatial appearance and angular viewpoint deviations into a single latent for improved efficiency and multi-view consistency. We achieve effective training of RapidMV by strategically decomposing our training process into multiple steps. We demonstrate that RapidMV outperforms existing methods in terms of consistency and latency, with competitive quality and text-image alignment.
  </details>  
  Keywords: multi-view synthesis  
- **[LatXGen: Towards Radiation-Free and Accurate Quantitative Analysis of Sagittal Spinal Alignment Via Cross-Modal Radiographic View Synthesis](https://arxiv.org/abs/2509.24165v1)**  
  Authors: Moxin Zhao, Nan Meng, Jason Pui Yin Cheung, Chris Yuk Kwan Tang, Chenxi Yu, Wenting Zhong, Pengyu Lu, Chang Shi, Yipeng Zhuang, Teng Zhang  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2509.24165v1.pdf)  
  <details><summary>Abstract</summary>

  Adolescent Idiopathic Scoliosis (AIS) is a complex three-dimensional spinal deformity, and accurate morphological assessment requires evaluating both coronal and sagittal alignment. While previous research has made significant progress in developing radiation-free methods for coronal plane assessment, reliable and accurate evaluation of sagittal alignment without ionizing radiation remains largely underexplored. To address this gap, we propose LatXGen, a novel generative framework that synthesizes realistic lateral spinal radiographs from posterior Red-Green-Blue and Depth (RGBD) images of unclothed backs. This enables accurate, radiation-free estimation of sagittal spinal alignment. LatXGen tackles two core challenges: (1) inferring sagittal spinal morphology changes from a lateral perspective based on posteroanterior surface geometry, and (2) performing cross-modality translation from RGBD input to the radiographic domain. The framework adopts a dual-stage architecture that progressively estimates lateral spinal structure and synthesizes corresponding radiographs. To enhance anatomical consistency, we introduce an attention-based Fast Fourier Convolution (FFC) module for integrating anatomical features...
  </details>  
- **[SIE3D: Single-Image Expressive 3D Avatar Generation via Semantic Embedding and Perceptual Expression Loss](https://arxiv.org/abs/2509.24004v2)**  
  Authors: Zhiqi Huang, Dulongkai Cui, Jinglu Hu  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2509.24004v2.pdf) | [![Project](https://img.shields.io/badge/-Project-blue)](https://huang-zhiqi.github.io/SIE3D)  
  <details><summary>Abstract</summary>

  Generating high-fidelity 3D head avatars from a single image is challenging, as current methods lack fine-grained, intuitive control over expressions via text. This paper proposes SIE3D, a framework that generates expressive 3D avatars from a single image and descriptive text. SIE3D fuses identity features from the image with semantic embedding from text through a novel conditioning scheme, enabling detailed control. To ensure generated expressions accurately match the text, it introduces an innovative perceptual expression loss function. This loss uses a pre-trained expression classifier to regularize the generation process, guaranteeing expression accuracy. Extensive experiments show SIE3D significantly improves controllability and realism, outperforming competitive methods in identity preservation and expression fidelity on a single consumer-grade GPU. Project page: https://huang-zhiqi.github.io/SIE3D/
  </details>  
- **[Towards Fine-Grained Text-to-3D Quality Assessment: A Benchmark and A Two-Stage Rank-Learning Metric](https://arxiv.org/abs/2509.23841v2)**  
  Authors: Bingyang Cui, Yujie Zhang, Qi Yang, Zhu Li, Yiling Xu  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2509.23841v2.pdf) | [![Project](https://img.shields.io/badge/-Project-blue)](https://cbysjtu.github.io/Rank2Score)  
  <details><summary>Abstract</summary>

  Recent advances in Text-to-3D (T23D) generative models have enabled the synthesis of diverse, high-fidelity 3D assets from textual prompts. However, existing challenges restrict the development of reliable T23D quality assessment (T23DQA). First, existing benchmarks are outdated, fragmented, and coarse-grained, making fine-grained metric training infeasible. Moreover, current objective metrics exhibit inherent design limitations, resulting in non-representative feature extraction and diminished metric robustness. To address these limitations, we introduce T23D-CompBench, a comprehensive benchmark for compositional T23D generation. We define five components with twelve sub-components for compositional prompts, which are used to generate 3,600 textured meshes from ten state-of-the-art generative models. A large-scale subjective experiment is conducted to collect 129,600 reliable human ratings across different perspectives. Based on T23D-CompBench, we further propose Rank2Score, an effective evaluator with two-stage training for T23DQA. Rank2Score enhances pairwise training via supervised contrastive regression and curriculum learning in the first stage, and subsequently refines predictions using mean opinion...
  </details>  
  Keywords: text-to-3d  
- **[M3DLayout: A Multi-Source Dataset of 3D Indoor Layouts and Structured Descriptions for 3D Generation](https://arxiv.org/abs/2509.23728v3)**  
  Authors: Yiheng Zhang, Zhuojiang Cai, Mingdao Wang, Meitong Guo, Tianxiao Li, Li Lin, Yuwang Wang  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2509.23728v3.pdf)  
  <details><summary>Abstract</summary>

  In text-driven 3D scene generation, object layout serves as a crucial intermediate representation that bridges high-level language instructions with detailed geometric output. It not only provides a structural blueprint for ensuring physical plausibility but also supports semantic controllability and interactive editing. However, the learning capabilities of current 3D indoor layout generation models are constrained by the limited scale, diversity, and annotation quality of existing datasets. To address this, we introduce M3DLayout, a large-scale, multi-source dataset for 3D indoor layout generation. M3DLayout comprises 21,367 layouts and over 433k object instances, integrating three distinct sources: real-world scans, professional CAD designs, and procedurally generated scenes. Each layout is paired with detailed structured text describing global scene summaries, relational placements of large furniture, and fine-grained arrangements of smaller items. This diverse and richly annotated resource enables models to learn complex spatial and semantic patterns across a wide variety of indoor environments. To assess the...
  </details>  
- **[StrucADT: Generating Structure-controlled 3D Point Clouds with Adjacency Diffusion Transformer](https://arxiv.org/abs/2509.23709v1)**  
  Authors: Zhenyu Shu, Jiajun Shen, Zhongui Chen, Xiaoguang Han, Shiqing Xin  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2509.23709v1.pdf)  
  <details><summary>Abstract</summary>

  In the field of 3D point cloud generation, numerous 3D generative models have demonstrated the ability to generate diverse and realistic 3D shapes. However, the majority of these approaches struggle to generate controllable 3D point cloud shapes that meet user-specific requirements, hindering the large-scale application of 3D point cloud generation. To address the challenge of lacking control in 3D point cloud generation, we are the first to propose controlling the generation of point clouds by shape structures that comprise part existences and part adjacency relationships. We manually annotate the adjacency relationships between the segmented parts of point cloud shapes, thereby constructing a StructureGraph representation. Based on this StructureGraph representation, we introduce StrucADT, a novel structure-controllable point cloud generation model, which consists of StructureGraphNet module to extract structure-aware latent features, cCNF Prior module to learn the distribution of the latent features controlled by the part adjacency, and Diffusion Transformer module conditioned...
  </details>  
- **[Sparse-Up: Learnable Sparse Upsampling for 3D Generation with High-Fidelity Textures](https://arxiv.org/abs/2509.23646v1)**  
  Authors: Lu Xiao, Jiale Zhang, Yang Liu, Taicheng Huang, Xin Tian  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2509.23646v1.pdf)  
  <details><summary>Abstract</summary>

  The creation of high-fidelity 3D assets is often hindered by a 'pixel-level pain point': the loss of high-frequency details. Existing methods often trade off one aspect for another: either sacrificing cross-view consistency, resulting in torn or drifting textures, or remaining trapped by the resolution ceiling of explicit voxels, forfeiting fine texture detail. In this work, we propose Sparse-Up, a memory-efficient, high-fidelity texture modeling framework that effectively preserves high-frequency details. We use sparse voxels to guide texture reconstruction and ensure multi-view consistency, while leveraging surface anchoring and view-domain partitioning to break through resolution constraints. Surface anchoring employs a learnable upsampling strategy to constrain voxels to the mesh surface, eliminating over 70% of redundant voxels present in traditional voxel upsampling. View-domain partitioning introduces an image patch-guided voxel partitioning scheme, supervising and back-propagating gradients only on visible local patches. Through these two strategies, we can significantly reduce memory consumption during high-resolution voxel training...
  </details>  
- **[ZeroScene: A Zero-Shot Framework for 3D Scene Generation from a Single Image and Controllable Texture Editing](https://arxiv.org/abs/2509.23607v2)**  
  Authors: Xiang Tang, Ruotong Li, Xiaopeng Fan  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2509.23607v2.pdf)  
  <details><summary>Abstract</summary>

  In the field of 3D content generation, single image scene reconstruction methods still struggle to simultaneously ensure the quality of individual assets and the coherence of the overall scene in complex environments, while texture editing techniques often fail to maintain both local continuity and multi-view consistency. In this paper, we propose a novel system ZeroScene, which leverages the prior knowledge of large vision models to accomplish both single image-to-3D scene reconstruction and texture editing in a zero-shot manner. ZeroScene extracts object-level 2D segmentation and depth information from input images to infer spatial relationships within the scene. It then jointly optimizes 3D and 2D projection losses of the point cloud to update object poses for precise scene alignment, ultimately constructing a coherent and complete 3D scene that encompasses both foreground and background. Moreover, ZeroScene supports texture editing of objects in the scene. By imposing constraints on the diffusion model and introducing...
  </details>  
  Keywords: 3d scene reconstruction, image-to-3d  
- **[FM-SIREN & FM-FINER: Implicit Neural Representation Using Nyquist-based Orthogonality](https://arxiv.org/abs/2509.23438v3)**  
  Authors: Mohammed Alsakabi, Wael Mobeirek, John M. Dolan, Ozan K. Tonguz  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2509.23438v3.pdf)  
  <details><summary>Abstract</summary>

  Existing periodic activation-based implicit neural representation (INR) networks, such as SIREN and FINER, suffer from hidden feature redundancy, where neurons within a layer capture overlapping frequency components due to the use of a fixed frequency multiplier. This redundancy limits the expressive capacity of multilayer perceptrons (MLPs). Drawing inspiration from classical signal processing methods such as the Discrete Sine Transform (DST), in this paper, we propose FM-SIREN and FM-FINER, which assign Nyquist-informed, neuron-specific frequency multipliers to periodic activations. Contrary to existing approaches, our design introduces frequency diversity without requiring hyperparameter tuning or additional network depth. This simple yet principled approach reduces the redundancy of features by nearly 50% and consistently improves signal reconstruction across diverse INR tasks, such as fitting 1D audio, 2D image and 3D shape, and video, outperforming their baseline counterparts while maintaining efficiency.
  </details>  
- **[NIFTY: a Non-Local Image Flow Matching for Texture Synthesis](https://arxiv.org/abs/2509.22318v1)**  
  Authors: Pierrick Chatillon, Julien Rabin, David Tschumperlé  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2509.22318v1.pdf) | [![GitHub](https://img.shields.io/github/stars/PierrickCh/Nifty.git?style=social)](https://github.com/PierrickCh/Nifty.git)  
  <details><summary>Abstract</summary>

  This paper addresses the problem of exemplar-based texture synthesis. We introduce NIFTY, a hybrid framework that combines recent insights on diffusion models trained with convolutional neural networks, and classical patch-based texture optimization techniques. NIFTY is a non-parametric flow-matching model built on non-local patch matching, which avoids the need for neural network training while alleviating common shortcomings of patch-based methods, such as poor initialization or visual artifacts. Experimental results demonstrate the effectiveness of the proposed approach compared to representative methods from the literature. Code is available at https://github.com/PierrickCh/Nifty.git
  </details>  
  Keywords: texture synthesis  
- **[Large Material Gaussian Model for Relightable 3D Generation](https://arxiv.org/abs/2509.22112v1)**  
  Authors: Jingrui Ye, Lingting Zhu, Runze Zhang, Zeyu Hu, Yingda Yin, Lanjiong Li, Lequan Yu, Qingmin Liao  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2509.22112v1.pdf)  
  <details><summary>Abstract</summary>

  The increasing demand for 3D assets across various industries necessitates efficient and automated methods for 3D content creation. Leveraging 3D Gaussian Splatting, recent large reconstruction models (LRMs) have demonstrated the ability to efficiently achieve high-quality 3D rendering by integrating multiview diffusion for generation and scalable transformers for reconstruction. However, existing models fail to produce the material properties of assets, which is crucial for realistic rendering in diverse lighting environments. In this paper, we introduce the Large Material Gaussian Model (MGM), a novel framework designed to generate high-quality 3D content with Physically Based Rendering (PBR) materials, ie, albedo, roughness, and metallic properties, rather than merely producing RGB textures with uncontrolled light baking. Specifically, we first fine-tune a new multiview material diffusion model conditioned on input depth and normal maps. Utilizing the generated multiview PBR images, we explore a Gaussian material representation that not only aligns with 2D Gaussian Splatting but also...
  </details>  
  Keywords: pbr materials  
- **[Drag4D: Align Your Motion with Text-Driven 3D Scene Generation](https://arxiv.org/abs/2509.21888v1)**  
  Authors: Minjun Kang, Inkyu Shin, Taeyeop Lee, In So Kweon, Kuk-Jin Yoon  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2509.21888v1.pdf)  
  <details><summary>Abstract</summary>

  We introduce Drag4D, an interactive framework that integrates object motion control within text-driven 3D scene generation. This framework enables users to define 3D trajectories for the 3D objects generated from a single image, seamlessly integrating them into a high-quality 3D background. Our Drag4D pipeline consists of three stages. First, we enhance text-to-3D background generation by applying 2D Gaussian Splatting with panoramic images and inpainted novel views, resulting in dense and visually complete 3D reconstructions. In the second stage, given a reference image of the target object, we introduce a 3D copy-and-paste approach: the target instance is extracted in a full 3D mesh using an off-the-shelf image-to-3D model and seamlessly composited into the generated 3D scene. The object mesh is then positioned within the 3D scene via our physics-aware object position learning, ensuring precise spatial alignment. Lastly, the spatially aligned object is temporally animated along a user-defined 3D trajectory. To mitigate...
  </details>  
  Keywords: text-to-3d, image-to-3d  
- **[SRHand: Super-Resolving Hand Images and 3D Shapes via View/Pose-aware Neural Image Representations and Explicit 3D Meshes](https://arxiv.org/abs/2509.21859v1)**  
  Authors: Minje Kim, Tae-Kyun Kim  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2509.21859v1.pdf) | [![Project](https://img.shields.io/badge/-Project-blue)](https://yunminjin2.github.io/projects/srhand)  
  <details><summary>Abstract</summary>

  Reconstructing detailed hand avatars plays a crucial role in various applications. While prior works have focused on capturing high-fidelity hand geometry, they heavily rely on high-resolution multi-view image inputs and struggle to generalize on low-resolution images. Multi-view image super-resolution methods have been proposed to enforce 3D view consistency. These methods, however, are limited to static objects/scenes with fixed resolutions and are not applicable to articulated deformable hands. In this paper, we propose SRHand (Super-Resolution Hand), the method for reconstructing detailed 3D geometry as well as textured images of hands from low-resolution images. SRHand leverages the advantages of implicit image representation with explicit hand meshes. Specifically, we introduce a geometric-aware implicit image function (GIIF) that learns detailed hand prior by upsampling the coarse input images. By jointly optimizing the implicit image function and explicit 3D hand shapes, our method preserves multi-view and pose consistency among upsampled hand images, and achieves fine-detailed...
  </details>  
- **[Hunyuan3D-Omni: A Unified Framework for Controllable Generation of 3D Assets](https://arxiv.org/abs/2509.21245v1)**  
  Authors: Team Hunyuan3D, :, Bowen Zhang, Chunchao Guo, Haolin Liu, Hongyu Yan, Huiwen Shi, Jingwei Huang, Junlin Yu, Kunhong Li, Linus, Penghao Wang, Qingxiang Lin, Sicong Liu, Xianghui Yang, Yixuan Tang, Yunfei Zhao, Zeqiang Lai, Zhihao Liang, Zibo Zhao  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2509.21245v1.pdf)  
  <details><summary>Abstract</summary>

  Recent advances in 3D-native generative models have accelerated asset creation for games, film, and design. However, most methods still rely primarily on image or text conditioning and lack fine-grained, cross-modal controls, which limits controllability and practical adoption. To address this gap, we present Hunyuan3D-Omni, a unified framework for fine-grained, controllable 3D asset generation built on Hunyuan3D 2.1. In addition to images, Hunyuan3D-Omni accepts point clouds, voxels, bounding boxes, and skeletal pose priors as conditioning signals, enabling precise control over geometry, topology, and pose. Instead of separate heads for each modality, our model unifies all signals in a single cross-modal architecture. We train with a progressive, difficulty-aware sampling strategy that selects one control modality per example and biases sampling toward harder signals (e.g., skeletal pose) while downweighting easier ones (e.g., point clouds), encouraging robust multi-modal fusion and graceful handling of missing inputs. Experiments show that these additional controls improve generation accuracy,...
  </details>  
  Keywords: 3d asset generation  
- **[Distant Object Localisation from Noisy Image Segmentation Sequences](https://arxiv.org/abs/2509.20906v2)**  
  Authors: Julius Pesonen, Arno Solin, Eija Honkavaara  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2509.20906v2.pdf)  
  <details><summary>Abstract</summary>

  3D object localisation based on a sequence of camera measurements is essential for safety-critical surveillance tasks, such as drone-based wildfire monitoring. Localisation of objects detected with a camera can typically be solved with specialised sensor configurations or 3D scene reconstruction. However, in the context of distant objects or tasks limited by the amount of available computational resources, neither solution is feasible. In this paper, we show that the task can be solved with either multi-view triangulation or particle filters, with the latter also providing shape and uncertainty estimates. We studied the solutions using 3D simulation and drone-based image segmentation sequences with global navigation satellite system (GNSS) based camera pose estimates. The results suggest that combining the proposed methods with pre-existing image segmentation models and drone-carried computational resources yields a reliable system for drone-based wildfire monitoring. The proposed solutions are independent of the detection method, also enabling quick adaptation to similar...
  </details>  
  Keywords: 3d scene reconstruction  
- **[FreeInsert: Personalized Object Insertion with Geometric and Style Control](https://arxiv.org/abs/2509.20756v1)**  
  Authors: Yuhong Zhang, Han Wang, Yiwen Wang, Rong Xie, Li Song  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2509.20756v1.pdf)  
  <details><summary>Abstract</summary>

  Text-to-image diffusion models have made significant progress in image generation, allowing for effortless customized generation. However, existing image editing methods still face certain limitations when dealing with personalized image composition tasks. First, there is the issue of lack of geometric control over the inserted objects. Current methods are confined to 2D space and typically rely on textual instructions, making it challenging to maintain precise geometric control over the objects. Second, there is the challenge of style consistency. Existing methods often overlook the style consistency between the inserted object and the background, resulting in a lack of realism. In addition, the challenge of inserting objects into images without extensive training remains significant. To address these issues, we propose \textit{FreeInsert}, a novel training-free framework that customizes object insertion into arbitrary scenes by leveraging 3D geometric information. Benefiting from the advances in existing 3D generation models, we first convert the 2D object into...
  </details>  
- **[SeamCrafter: Enhancing Mesh Seam Generation for Artist UV Unwrapping via Reinforcement Learning](https://arxiv.org/abs/2509.20725v2)**  
  Authors: Duoteng Xu, Yuguang Chen, Jing Li, Xinhai Liu, Xueqi Ma, Zhuo Chen, Dongyu Zhang, Chunchao Guo  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2509.20725v2.pdf)  
  <details><summary>Abstract</summary>

  Mesh seams play a pivotal role in partitioning 3D surfaces for UV parametrization and texture mapping. Poorly placed seams often result in severe UV distortion or excessive fragmentation, thereby hindering texture synthesis and disrupting artist workflows. Existing methods frequently trade one failure mode for another-producing either high distortion or many scattered islands. To address this, we introduce SeamCrafter, an autoregressive GPT-style seam generator conditioned on point cloud inputs. SeamCrafter employs a dual-branch point-cloud encoder that disentangles and captures complementary topological and geometric cues during pretraining. To further enhance seam quality, we fine-tune the model using Direct Preference Optimization (DPO) on a preference dataset derived from a novel seam-evaluation framework. This framework assesses seams primarily by UV distortion and fragmentation, and provides pairwise preference labels to guide optimization. Extensive experiments demonstrate that SeamCrafter produces seams with substantially lower distortion and fragmentation than prior approaches, while preserving topological consistency and visual fidelity.
  </details>  
  Keywords: texture synthesis  
- **[Lyra: Generative 3D Scene Reconstruction via Video Diffusion Model Self-Distillation](https://arxiv.org/abs/2509.19296v1)**  
  Authors: Sherwin Bahmani, Tianchang Shen, Jiawei Ren, Jiahui Huang, Yifeng Jiang, Haithem Turki, Andrea Tagliasacchi, David B. Lindell, Zan Gojcic, Sanja Fidler, Huan Ling, Jun Gao, Xuanchi Ren  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2509.19296v1.pdf)  
  <details><summary>Abstract</summary>

  The ability to generate virtual environments is crucial for applications ranging from gaming to physical AI domains such as robotics, autonomous driving, and industrial AI. Current learning-based 3D reconstruction methods rely on the availability of captured real-world multi-view data, which is not always readily available. Recent advancements in video diffusion models have shown remarkable imagination capabilities, yet their 2D nature limits the applications to simulation where a robot needs to navigate and interact with the environment. In this paper, we propose a self-distillation framework that aims to distill the implicit 3D knowledge in the video diffusion models into an explicit 3D Gaussian Splatting (3DGS) representation, eliminating the need for multi-view training data. Specifically, we augment the typical RGB decoder with a 3DGS decoder, which is supervised by the output of the RGB decoder. In this approach, the 3DGS decoder can be purely trained with synthetic data generated by video diffusion...
  </details>  
  Keywords: 3d scene reconstruction  

### October 2025
- **[A Multi-Modal Neuro-Symbolic Approach for Spatial Reasoning-Based Visual Grounding in Robotics](https://arxiv.org/abs/2510.27033v1)**  
  Authors: Simindokht Jahangard, Mehrzad Mohammadi, Abhinav Dhall, Hamid Rezatofighi  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2510.27033v1.pdf)  
  <details><summary>Abstract</summary>

  Visual reasoning, particularly spatial reasoning, is a challenging cognitive task that requires understanding object relationships and their interactions within complex environments, especially in robotics domain. Existing vision_language models (VLMs) excel at perception tasks but struggle with fine-grained spatial reasoning due to their implicit, correlation-driven reasoning and reliance solely on images. We propose a novel neuro_symbolic framework that integrates both panoramic-image and 3D point cloud information, combining neural perception with symbolic reasoning to explicitly model spatial and logical relationships. Our framework consists of a perception module for detecting entities and extracting attributes, and a reasoning module that constructs a structured scene graph to support precise, interpretable queries. Evaluated on the JRDB-Reasoning dataset, our approach demonstrates superior performance and reliability in crowded, human_built environments while maintaining a lightweight design suitable for robotics and embodied AI applications.
  </details>  
- **[FullPart: Generating each 3D Part at Full Resolution](https://arxiv.org/abs/2510.26140v1)**  
  Authors: Lihe Ding, Shaocong Dong, Yaokun Li, Chenjian Gao, Xiao Chen, Rui Han, Yihao Kuang, Hong Zhang, Bo Huang, Zhanpeng Huang, Zibin Wang, Dan Xu, Tianfan Xue  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2510.26140v1.pdf)  
  <details><summary>Abstract</summary>

  Part-based 3D generation holds great potential for various applications. Previous part generators that represent parts using implicit vector-set tokens often suffer from insufficient geometric details. Another line of work adopts an explicit voxel representation but shares a global voxel grid among all parts; this often causes small parts to occupy too few voxels, leading to degraded quality. In this paper, we propose FullPart, a novel framework that combines both implicit and explicit paradigms. It first derives the bounding box layout through an implicit box vector-set diffusion process, a task that implicit diffusion handles effectively since box tokens contain little geometric detail. Then, it generates detailed parts, each within its own fixed full-resolution voxel grid. Instead of sharing a global low-resolution space, each part in our method - even small ones - is generated at full resolution, enabling the synthesis of intricate details. We further introduce a center-point encoding strategy to...
  </details>  
- **[FreeArt3D: Training-Free Articulated Object Generation using 3D Diffusion](https://arxiv.org/abs/2510.25765v2)**  
  Authors: Chuhao Chen, Isabella Liu, Xinyue Wei, Hao Su, Minghua Liu  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2510.25765v2.pdf) | [![Project](https://img.shields.io/badge/-Project-blue)](https://czzzzh.github.io/FreeArt3D)  
  <details><summary>Abstract</summary>

  Articulated 3D objects are central to many applications in robotics, AR/VR, and animation. Recent approaches to modeling such objects either rely on optimization-based reconstruction pipelines that require dense-view supervision or on feed-forward generative models that produce coarse geometric approximations and often overlook surface texture. In contrast, open-world 3D generation of static objects has achieved remarkable success, especially with the advent of native 3D diffusion models such as Trellis. However, extending these methods to articulated objects by training native 3D diffusion models poses significant challenges. In this work, we present FreeArt3D, a training-free framework for articulated 3D object generation. Instead of training a new model on limited articulated data, FreeArt3D repurposes a pre-trained static 3D diffusion model (e.g., Trellis) as a powerful shape prior. It extends Score Distillation Sampling (SDS) into the 3D-to-4D domain by treating articulation as an additional generative dimension. Given a few images captured in different articulation states,...
  </details>  
  Keywords: articulated object generation  
- **[4-Doodle: Text to 3D Sketches that Move!](https://arxiv.org/abs/2510.25319v1)**  
  Authors: Hao Chen, Jiaqi Wang, Yonggang Qi, Ke Li, Kaiyue Pang, Yi-Zhe Song  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2510.25319v1.pdf)  
  <details><summary>Abstract</summary>

  We present a novel task: text-to-3D sketch animation, which aims to bring freeform sketches to life in dynamic 3D space. Unlike prior works focused on photorealistic content generation, we target sparse, stylized, and view-consistent 3D vector sketches, a lightweight and interpretable medium well-suited for visual communication and prototyping. However, this task is very challenging: (i) no paired dataset exists for text and 3D (or 4D) sketches; (ii) sketches require structural abstraction that is difficult to model with conventional 3D representations like NeRFs or point clouds; and (iii) animating such sketches demands temporal coherence and multi-view consistency, which current pipelines do not address. Therefore, we propose 4-Doodle, the first training-free framework for generating dynamic 3D sketches from text. It leverages pretrained image and video diffusion models through a dual-space distillation scheme: one space captures multi-view-consistent geometry using differentiable Bézier curves, while the other encodes motion dynamics via temporally-aware priors. Unlike prior...
  </details>  
  Keywords: text-to-3d  
- **[TurboPortrait3D: Single-step diffusion-based fast portrait novel-view synthesis](https://arxiv.org/abs/2510.23929v1)**  
  Authors: Emily Kim, Julieta Martinez, Timur Bagautdinov, Jessica Hodgins  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2510.23929v1.pdf)  
  <details><summary>Abstract</summary>

  We introduce TurboPortrait3D: a method for low-latency novel-view synthesis of human portraits. Our approach builds on the observation that existing image-to-3D models for portrait generation, while capable of producing renderable 3D representations, are prone to visual artifacts, often lack of detail, and tend to fail at fully preserving the identity of the subject. On the other hand, image diffusion models excel at generating high-quality images, but besides being computationally expensive, are not grounded in 3D and thus are not directly capable of producing multi-view consistent outputs. In this work, we demonstrate that image-space diffusion models can be used to significantly enhance the quality of existing image-to-avatar methods, while maintaining 3D-awareness and running with low-latency. Our method takes a single frontal image of a subject as input, and applies a feedforward image-to-avatar generation pipeline to obtain an initial 3D representation and corresponding noisy renders. These noisy renders are then fed to...
  </details>  
  Keywords: image-to-3d  
- **[Adaptive Keyframe Selection for Scalable 3D Scene Reconstruction in Dynamic Environments](https://arxiv.org/abs/2510.23928v3)**  
  Authors: Raman Jha, Yang Zhou, Giuseppe Loianno  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2510.23928v3.pdf)  
  <details><summary>Abstract</summary>

  In this paper, we propose an adaptive keyframe selection method for improved 3D scene reconstruction in dynamic environments. The proposed method integrates two complementary modules: an error-based selection module utilizing photometric and structural similarity (SSIM) errors, and a momentum-based update module that dynamically adjusts keyframe selection thresholds according to scene motion dynamics. By dynamically curating the most informative frames, our approach addresses a key data bottleneck in real-time perception. This allows for the creation of high-quality 3D world representations from a compressed data stream, a critical step towards scalable robot learning and deployment in complex, dynamic environments. Experimental results demonstrate significant improvements over traditional static keyframe selection strategies, such as fixed temporal intervals or uniform frame skipping. These findings highlight a meaningful advancement toward adaptive perception systems that can dynamically respond to complex and evolving visual scenes. We evaluate our proposed adaptive keyframe selection module on two recent state-of-the-art 3D...
  </details>  
  Keywords: 3d scene reconstruction  
- **[TRELLISWorld: Training-Free World Generation from Object Generators](https://arxiv.org/abs/2510.23880v2)**  
  Authors: Hanke Chen, Yuan Liu, Minchen Li  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2510.23880v2.pdf)  
  <details><summary>Abstract</summary>

  Text-driven 3D scene generation holds promise for a wide range of applications, from virtual prototyping to AR/VR and simulation. However, existing methods are often constrained to single-object generation, require domain-specific training, or lack support for full 360-degree viewability. In this work, we present a training-free approach to 3D scene synthesis by repurposing general-purpose text-to-3D object diffusion models as modular tile generators. We reformulate scene generation as a multi-tile denoising problem, where overlapping 3D regions are independently generated and seamlessly blended via weighted averaging. This enables scalable synthesis of large, coherent scenes while preserving local semantic control. Our method eliminates the need for scene-level datasets or retraining, relies on minimal heuristics, and inherits the generalization capabilities of object-level priors. We demonstrate that our approach supports diverse scene layouts, efficient generation, and flexible editing, establishing a simple yet powerful foundation for general-purpose, language-driven 3D scene construction.
  </details>  
  Keywords: text-to-3d  
- **[Track, Inpaint, Resplat: Subject-driven 3D and 4D Generation with Progressive Texture Infilling](https://arxiv.org/abs/2510.23605v1)**  
  Authors: Shuhong Zheng, Ashkan Mirzaei, Igor Gilitschenski  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2510.23605v1.pdf) | [![Project](https://img.shields.io/badge/-Project-blue)](https://zsh2000.github.io/track-inpaint-resplat.github.io)  
  <details><summary>Abstract</summary>

  Current 3D/4D generation methods are usually optimized for photorealism, efficiency, and aesthetics. However, they often fail to preserve the semantic identity of the subject across different viewpoints. Adapting generation methods with one or few images of a specific subject (also known as Personalization or Subject-driven generation) allows generating visual content that align with the identity of the subject. However, personalized 3D/4D generation is still largely underexplored. In this work, we introduce TIRE (Track, Inpaint, REsplat), a novel method for subject-driven 3D/4D generation. It takes an initial 3D asset produced by an existing 3D generative model as input and uses video tracking to identify the regions that need to be modified. Then, we adopt a subject-driven 2D inpainting model for progressively infilling the identified regions. Finally, we resplat the modified 2D multi-view observations back to 3D while still maintaining consistency. Extensive experiments demonstrate that our approach significantly improves identity preservation in...
  </details>  
- **[RobotArena $\infty$: Scalable Robot Benchmarking via Real-to-Sim Translation](https://arxiv.org/abs/2510.23571v3)**  
  Authors: Yash Jangir, Yidi Zhang, Pang-Chi Lo, Kashu Yamazaki, Chenyu Zhang, Kuan-Hsun Tu, Tsung-Wei Ke, Lei Ke, Yonatan Bisk, Katerina Fragkiadaki  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2510.23571v3.pdf)  
  <details><summary>Abstract</summary>

  The pursuit of robot generalists, agents capable of performing diverse tasks across diverse environments, demands rigorous and scalable evaluation. Yet real-world testing of robot policies remains fundamentally constrained: it is labor-intensive, slow, unsafe at scale, and difficult to reproduce. As policies expand in scope and complexity, these barriers only intensify, since defining "success" in robotics often hinges on nuanced human judgments of execution quality. We introduce RobotArena Infinity, a new benchmarking framework that overcomes these challenges by shifting vision-language-action (VLA) evaluation into large-scale simulated environments augmented with online human feedback. Leveraging advances in vision-language models, 2D-to-3D generative modeling, and differentiable rendering, our approach automatically converts video demonstrations from widely used robot datasets into simulated counterparts. Within these digital twins, we assess VLA policies using both automated vision-language-model-guided scoring and scalable human preference judgments collected from crowdworkers, transforming human involvement from tedious scene setup, resetting, and safety supervision into lightweight preference...
  </details>  
- **[ReconViaGen: Towards Accurate Multi-view 3D Object Reconstruction via Generation](https://arxiv.org/abs/2510.23306v1)**  
  Authors: Jiahao Chang, Chongjie Ye, Yushuang Wu, Yuantao Chen, Yidan Zhang, Zhongjin Luo, Chenghong Li, Yihao Zhi, Xiaoguang Han  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2510.23306v1.pdf) | [![Project](https://img.shields.io/badge/-Project-blue)](https://jiahao620.github.io/reconviagen)  
  <details><summary>Abstract</summary>

  Existing multi-view 3D object reconstruction methods heavily rely on sufficient overlap between input views, where occlusions and sparse coverage in practice frequently yield severe reconstruction incompleteness. Recent advancements in diffusion-based 3D generative techniques offer the potential to address these limitations by leveraging learned generative priors to hallucinate invisible parts of objects, thereby generating plausible 3D structures. However, the stochastic nature of the inference process limits the accuracy and reliability of generation results, preventing existing reconstruction frameworks from integrating such 3D generative priors. In this work, we comprehensively analyze the reasons why diffusion-based 3D generative methods fail to achieve high consistency, including (a) the insufficiency in constructing and leveraging cross-view connections when extracting multi-view image features as conditions, and (b) the poor controllability of iterative denoising during local detail generation, which easily leads to plausible but inconsistent fine geometric and texture details with inputs. Accordingly, we propose ReconViaGen to innovatively integrate...
  </details>  
- **[VR-Drive: Viewpoint-Robust End-to-End Driving with Feed-Forward 3D Gaussian Splatting](https://arxiv.org/abs/2510.23205v1)**  
  Authors: Hoonhee Cho, Jae-Young Kang, Giwon Lee, Hyemin Yang, Heejun Park, Seokwoo Jung, Kuk-Jin Yoon  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2510.23205v1.pdf)  
  <details><summary>Abstract</summary>

  End-to-end autonomous driving (E2E-AD) has emerged as a promising paradigm that unifies perception, prediction, and planning into a holistic, data-driven framework. However, achieving robustness to varying camera viewpoints, a common real-world challenge due to diverse vehicle configurations, remains an open problem. In this work, we propose VR-Drive, a novel E2E-AD framework that addresses viewpoint generalization by jointly learning 3D scene reconstruction as an auxiliary task to enable planning-aware view synthesis. Unlike prior scene-specific synthesis approaches, VR-Drive adopts a feed-forward inference strategy that supports online training-time augmentation from sparse views without additional annotations. To further improve viewpoint consistency, we introduce a viewpoint-mixed memory bank that facilitates temporal interaction across multiple viewpoints and a viewpoint-consistent distillation strategy that transfers knowledge from original to synthesized views. Trained in a fully end-to-end manner, VR-Drive effectively mitigates synthesis-induced noise and improves planning under viewpoint shifts. In addition, we release a new benchmark dataset to evaluate...
  </details>  
  Keywords: 3d scene reconstruction  
- **[Look and Tell: A Dataset for Multimodal Grounding Across Egocentric and Exocentric Views](https://arxiv.org/abs/2510.22672v2)**  
  Authors: Anna Deichler, Jonas Beskow  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2510.22672v2.pdf)  
  <details><summary>Abstract</summary>

  We introduce Look and Tell, a multimodal dataset for studying referential communication across egocentric and exocentric perspectives. Using Meta Project Aria smart glasses and stationary cameras, we recorded synchronized gaze, speech, and video as 25 participants instructed a partner to identify ingredients in a kitchen. Combined with 3D scene reconstructions, this setup provides a benchmark for evaluating how different spatial representations (2D vs. 3D; ego vs. exo) affect multimodal grounding. The dataset contains 3.67 hours of recordings, including 2,707 richly annotated referential expressions, and is designed to advance the development of embodied agents that can understand and engage in situated dialogue.
  </details>  
  Keywords: 3d scene reconstruction  
- **[LAMP: Data-Efficient Linear Affine Weight-Space Models for Parameter-Controlled 3D Shape Generation and Extrapolation](https://arxiv.org/abs/2510.22491v3)**  
  Authors: Ghadi Nehme, Yanxia Zhang, Dule Shu, Matt Klenk, Faez Ahmed  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2510.22491v3.pdf)  
  <details><summary>Abstract</summary>

  Generating high-fidelity 3D geometries under explicit parameter constraints is central to engineering design, yet current methods often require large datasets and fail to provide reliable control beyond the training distribution. We introduce LAMP, a data-efficient framework for controllable and interpretable 3D generation that aligns signed distance function (SDF) decoders by overfitting each exemplar from a shared initialization, then generates new designs by solving a parameter-constrained affine mixing problem in the aligned weight space. To improve reliability, we propose a linearity-mismatch safety metric that detects when mixed decoders leave the valid local regime. We evaluate LAMP on DrivAerNet++, BlendedNet, and additional industry-level vehicle families, including sports cars, SUVs, and convertibles. LAMP enables controlled interpolation with as few as 50 samples, safe extrapolation up to 100% beyond training ranges, and performance-guided optimization under fixed parameters, outperforming conditional autoencoder and Deep Network Interpolation (DNI) baselines in extrapolation, data efficiency, and parameter fidelity. Our...
  </details>  
- **[DynaPose4D: High-Quality 4D Dynamic Content Generation via Pose Alignment Loss](https://arxiv.org/abs/2510.22473v1)**  
  Authors: Jing Yang, Yufeng Yang  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2510.22473v1.pdf)  
  <details><summary>Abstract</summary>

  Recent advancements in 2D and 3D generative models have expanded the capabilities of computer vision. However, generating high-quality 4D dynamic content from a single static image remains a significant challenge. Traditional methods have limitations in modeling temporal dependencies and accurately capturing dynamic geometry changes, especially when considering variations in camera perspective. To address this issue, we propose DynaPose4D, an innovative solution that integrates 4D Gaussian Splatting (4DGS) techniques with Category-Agnostic Pose Estimation (CAPE) technology. This framework uses 3D Gaussian Splatting to construct a 3D model from single images, then predicts multi-view pose keypoints based on one-shot support from a chosen view, leveraging supervisory signals to enhance motion consistency. Experimental results show that DynaPose4D achieves excellent coherence, consistency, and fluidity in dynamic motion generation. These findings not only validate the efficacy of the DynaPose4D framework but also indicate its potential applications in the domains of computer vision and animation production.
  </details>  
- **[Better Tokens for Better 3D: Advancing Vision-Language Modeling in 3D Medical Imaging](https://arxiv.org/abs/2510.20639v1)**  
  Authors: Ibrahim Ethem Hamamci, Sezgin Er, Suprosanna Shit, Hadrien Reynaud, Dong Yang, Pengfei Guo, Marc Edgar, Daguang Xu, Bernhard Kainz, Bjoern Menze  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2510.20639v1.pdf) | [![GitHub](https://img.shields.io/github/stars/ibrahimethemhamamci/BTB3D?style=social)](https://github.com/ibrahimethemhamamci/BTB3D)  
  <details><summary>Abstract</summary>

  Recent progress in vision-language modeling for 3D medical imaging has been fueled by large-scale computed tomography (CT) corpora with paired free-text reports, stronger architectures, and powerful pretrained models. This has enabled applications such as automated report generation and text-conditioned 3D image synthesis. Yet, current approaches struggle with high-resolution, long-sequence volumes: contrastive pretraining often yields vision encoders that are misaligned with clinical language, and slice-wise tokenization blurs fine anatomy, reducing diagnostic performance on downstream tasks. We introduce BTB3D (Better Tokens for Better 3D), a causal convolutional encoder-decoder that unifies 2D and 3D training and inference while producing compact, frequency-aware volumetric tokens. A three-stage training curriculum enables (i) local reconstruction, (ii) overlapping-window tiling, and (iii) long-context decoder refinement, during which the model learns from short slice excerpts yet generalizes to scans exceeding 300 slices without additional memory overhead. BTB3D sets a new state-of-the-art on two key tasks: it improves BLEU scores and...
  </details>  
- **[COS3D: Collaborative Open-Vocabulary 3D Segmentation](https://arxiv.org/abs/2510.20238v1)**  
  Authors: Runsong Zhu, Ka-Hei Hui, Zhengzhe Liu, Qianyi Wu, Weiliang Tang, Shi Qiu, Pheng-Ann Heng, Chi-Wing Fu  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2510.20238v1.pdf) | [![GitHub](https://img.shields.io/github/stars/Runsong123/COS3D?style=social)](https://github.com/Runsong123/COS3D)  
  <details><summary>Abstract</summary>

  Open-vocabulary 3D segmentation is a fundamental yet challenging task, requiring a mutual understanding of both segmentation and language. However, existing Gaussian-splatting-based methods rely either on a single 3D language field, leading to inferior segmentation, or on pre-computed class-agnostic segmentations, suffering from error accumulation. To address these limitations, we present COS3D, a new collaborative prompt-segmentation framework that contributes to effectively integrating complementary language and segmentation cues throughout its entire pipeline. We first introduce the new concept of collaborative field, comprising an instance field and a language field, as the cornerstone for collaboration. During training, to effectively construct the collaborative field, our key idea is to capture the intrinsic relationship between the instance field and language field, through a novel instance-to-language feature mapping and designing an efficient two-stage training strategy. During inference, to bridge distinct characteristics of the two fields, we further design an adaptive language-to-instance prompt refinement, promoting high-quality prompt-segmentation inference....
  </details>  
- **[Seed3D 1.0: From Images to High-Fidelity Simulation-Ready 3D Assets](https://arxiv.org/abs/2510.19944v1)**  
  Authors: Jiashi Feng, Xiu Li, Jing Lin, Jiahang Liu, Gaohong Liu, Weiqiang Lou, Su Ma, Guang Shi, Qinlong Wang, Jun Wang, Zhongcong Xu, Xuanyu Yi, Zihao Yu, Jianfeng Zhang, Yifan Zhu, Rui Chen, Jinxin Chi, Zixian Du, Li Han, Lixin Huang, Kaihua Jiang, Yuhan Li, Guan Luo, Shuguang Wang, Qianyi Wu, Fan Yang, Junyang Zhang, Xuanmeng Zhang  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2510.19944v1.pdf) | [![Project](https://img.shields.io/badge/-Project-blue)](https://console.volcengine.com/ark/region:ark+cn-beijing/experience/vision?modelId=doubao-seed3d-1-0-250928&tab=Gen3D)  
  <details><summary>Abstract</summary>

  Developing embodied AI agents requires scalable training environments that balance content diversity with physics accuracy. World simulators provide such environments but face distinct limitations: video-based methods generate diverse content but lack real-time physics feedback for interactive learning, while physics-based engines provide accurate dynamics but face scalability limitations from costly manual asset creation. We present Seed3D 1.0, a foundation model that generates simulation-ready 3D assets from single images, addressing the scalability challenge while maintaining physics rigor. Unlike existing 3D generation models, our system produces assets with accurate geometry, well-aligned textures, and realistic physically-based materials. These assets can be directly integrated into physics engines with minimal configuration, enabling deployment in robotic manipulation and simulation training. Beyond individual objects, the system scales to complete scene generation through assembling objects into coherent environments. By enabling scalable simulation-ready content creation, Seed3D 1.0 provides a foundation for advancing physics-based world simulators. Seed3D 1.0 is now available...
  </details>  
- **[Advances in 4D Representation: Geometry, Motion, and Interaction](https://arxiv.org/abs/2510.19255v2)**  
  Authors: Mingrui Zhao, Sauradip Nag, Kai Wang, Aditya Vora, Guangda Ji, Peter Chun, Ali Mahdavi-Amiri, Hao Zhang  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2510.19255v2.pdf) | [![Project](https://img.shields.io/badge/-Project-blue)](https://mingrui-zhao.github.io/4DRep-GMI)  
  <details><summary>Abstract</summary>

  We present a survey on 4D generation and reconstruction, a fast-evolving subfield of computer graphics whose developments have been propelled by recent advances in neural fields, geometric and motion deep learning, as well as 3D generative artificial intelligence (GenAI). While our survey is not the first of its kind, we build our coverage of the domain from a unique and distinctive perspective of 4D representations, to model 3D geometry evolving over time while exhibiting motion and interaction. Specifically, instead of offering an exhaustive enumeration of many works, we take a more selective approach by focusing on representative works to highlight both the desirable properties and ensuing challenges of each representation under different computation, application, and data scenarios. The main take-away message we aim to convey to the readers is on how to select and then customize the appropriate 4D representations for their tasks. Organizationally, we separate the 4D representations based...
  </details>  
- **[ShapeCraft: LLM Agents for Structured, Textured and Interactive 3D Modeling](https://arxiv.org/abs/2510.17603v1)**  
  Authors: Shuyuan Zhang, Chenhan Jiang, Zuoou Li, Jiankang Deng  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2510.17603v1.pdf)  
  <details><summary>Abstract</summary>

  3D generation from natural language offers significant potential to reduce expert manual modeling efforts and enhance accessibility to 3D assets. However, existing methods often yield unstructured meshes and exhibit poor interactivity, making them impractical for artistic workflows. To address these limitations, we represent 3D assets as shape programs and introduce ShapeCraft, a novel multi-agent framework for text-to-3D generation. At its core, we propose a Graph-based Procedural Shape (GPS) representation that decomposes complex natural language into a structured graph of sub-tasks, thereby facilitating accurate LLM comprehension and interpretation of spatial relationships and semantic shape details. Specifically, LLM agents hierarchically parse user input to initialize GPS, then iteratively refine procedural modeling and painting to produce structured, textured, and interactive 3D assets. Qualitative and quantitative experiments demonstrate ShapeCraft's superior performance in generating geometrically accurate and semantically rich 3D assets compared to existing LLM-based agents. We further show the versatility of ShapeCraft through examples...
  </details>  
  Keywords: text-to-3d  
- **[GACO-CAD: Geometry-Augmented and Conciseness-Optimized CAD Model Generation from Single Image](https://arxiv.org/abs/2510.17157v1)**  
  Authors: Yinghui Wang, Xinyu Zhang, Peng Du  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2510.17157v1.pdf)  
  <details><summary>Abstract</summary>

  Generating editable, parametric CAD models from a single image holds great potential to lower the barriers of industrial concept design. However, current multi-modal large language models (MLLMs) still struggle with accurately inferring 3D geometry from 2D images due to limited spatial reasoning capabilities. We address this limitation by introducing GACO-CAD, a novel two-stage post-training framework. It is designed to achieve a joint objective: simultaneously improving the geometric accuracy of the generated CAD models and encouraging the use of more concise modeling procedures. First, during supervised fine-tuning, we leverage depth and surface normal maps as dense geometric priors, combining them with the RGB image to form a multi-channel input. In the context of single-view reconstruction, these priors provide complementary spatial cues that help the MLLM more reliably recover 3D geometry from 2D observations. Second, during reinforcement learning, we introduce a group length reward that, while preserving high geometric fidelity, promotes the...
  </details>  
  Keywords: single-view reconstruction  
- **[GuideFlow3D: Optimization-Guided Rectified Flow For Appearance Transfer](https://arxiv.org/abs/2510.16136v1)**  
  Authors: Sayan Deb Sarkar, Sinisa Stekovic, Vincent Lepetit, Iro Armeni  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2510.16136v1.pdf)  
  <details><summary>Abstract</summary>

  Transferring appearance to 3D assets using different representations of the appearance object - such as images or text - has garnered interest due to its wide range of applications in industries like gaming, augmented reality, and digital content creation. However, state-of-the-art methods still fail when the geometry between the input and appearance objects is significantly different. A straightforward approach is to directly apply a 3D generative model, but we show that this ultimately fails to produce appealing results. Instead, we propose a principled approach inspired by universal guidance. Given a pretrained rectified flow model conditioned on image or text, our training-free method interacts with the sampling process by periodically adding guidance. This guidance can be modeled as a differentiable loss function, and we experiment with two different types of guidance including part-aware losses for appearance and self-similarity. Our experiments show that our approach successfully transfers texture and geometric details to...
  </details>  
- **[SANR: Scene-Aware Neural Representation for Light Field Image Compression with Rate-Distortion Optimization](https://arxiv.org/abs/2510.15775v1)**  
  Authors: Gai Zhang, Xinfeng Zhang, Lv Tang, Hongyu An, Li Zhang, Qingming Huang  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2510.15775v1.pdf)  
  <details><summary>Abstract</summary>

  Light field images capture multi-view scene information and play a crucial role in 3D scene reconstruction. However, their high-dimensional nature results in enormous data volumes, posing a significant challenge for efficient compression in practical storage and transmission scenarios. Although neural representation-based methods have shown promise in light field image compression, most approaches rely on direct coordinate-to-pixel mapping through implicit neural representation (INR), often neglecting the explicit modeling of scene structure. Moreover, they typically lack end-to-end rate-distortion optimization, limiting their compression efficiency. To address these limitations, we propose SANR, a Scene-Aware Neural Representation framework for light field image compression with end-to-end rate-distortion optimization. For scene awareness, SANR introduces a hierarchical scene modeling block that leverages multi-scale latent codes to capture intrinsic scene structures, thereby reducing the information gap between INR input coordinates and the target light field image. From a compression perspective, SANR is the first to incorporate entropy-constrained quantization-aware training...
  </details>  
  Keywords: 3d scene reconstruction  
- **[H2OFlow: Grounding Human-Object Affordances with 3D Generative Models and Dense Diffused Flows](https://arxiv.org/abs/2510.21769v2)**  
  Authors: Harry Zhang, Luca Carlone  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2510.21769v2.pdf)  
  <details><summary>Abstract</summary>

  Understanding how humans interact with the surrounding environment, and specifically reasoning about object interactions and affordances, is a critical challenge in computer vision, robotics, and AI. Current approaches often depend on labor-intensive, hand-labeled datasets capturing real-world or simulated human-object interaction (HOI) tasks, which are costly and time-consuming to produce. Furthermore, most existing methods for 3D affordance understanding are limited to contact-based analysis, neglecting other essential aspects of human-object interactions, such as orientation (\eg, humans might have a preferential orientation with respect certain objects, such as a TV) and spatial occupancy (\eg, humans are more likely to occupy certain regions around an object, like the front of a microwave rather than its back). To address these limitations, we introduce \emph{H2OFlow}, a novel framework that comprehensively learns 3D HOI affordances -- encompassing contact, orientation, and spatial occupancy -- using only synthetic data generated from 3D generative models. H2OFlow employs a dense 3D-flow-based...
  </details>  
- **[Comprehensive language-image pre-training for 3D medical image understanding](https://arxiv.org/abs/2510.15042v2)**  
  Authors: Tassilo Wald, Ibrahim Ethem Hamamci, Yuan Gao, Sam Bond-Taylor, Harshita Sharma, Maximilian Ilse, Cynthia Lo, Olesya Melnichenko, Anton Schwaighofer, Noel C. F. Codella, Maria Teodora Wetscherek, Klaus H. Maier-Hein, Panagiotis Korfiatis, Valentina Salvatelli, Javier Alvarez-Valle, Fernando Pérez-García  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2510.15042v2.pdf) | [![Project](https://img.shields.io/badge/-Project-blue)](https://huggingface.co/microsoft/colipri.) | [![HuggingFace](https://img.shields.io/badge/-HuggingFace-yellow)](https://huggingface.co/microsoft/colipri)  
  <details><summary>Abstract</summary>

  Vision-language pre-training, i.e., aligning images with paired text, is a powerful paradigm to create encoders that can be directly used for tasks such as classification, retrieval, and segmentation. In the 3D medical image domain, these capabilities allow vision-language encoders (VLEs) to support radiologists by retrieving patients with similar abnormalities, predicting likelihoods of abnormality, or, with downstream adaptation, generating radiological reports. While the methodology holds promise, data availability and domain-specific hurdles limit the capabilities of current 3D VLEs. In this paper, we overcome these challenges by injecting additional supervision via a report generation objective and combining vision-language with vision-only pre-training. This allows us to leverage both image-only and paired image-text 3D datasets, increasing the total amount of data to which our model is exposed. Through these additional objectives, paired with best practices of the 3D medical imaging domain, we develop the Comprehensive Language-Image Pre-training (COLIPRI) encoder family. Our COLIPRI encoders achieve...
  </details>  
- **[Text-to-3D by Stitching a Multi-view Reconstruction Network to a Video Generator](https://arxiv.org/abs/2510.13454v2)**  
  Authors: Hyojun Go, Dominik Narnhofer, Goutam Bhat, Prune Truong, Federico Tombari, Konrad Schindler  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2510.13454v2.pdf)  
  <details><summary>Abstract</summary>

  The rapid progress of large, pretrained models for both visual content generation and 3D reconstruction opens up new possibilities for text-to-3D generation. Intuitively, one could obtain a formidable 3D scene generator if one were able to combine the power of a modern latent text-to-video model as "generator" with the geometric abilities of a recent (feedforward) 3D reconstruction system as "decoder". We introduce VIST3A, a general framework that does just that, addressing two main challenges. First, the two components must be joined in a way that preserves the rich knowledge encoded in their weights. We revisit model stitching, i.e., we identify the layer in the 3D decoder that best matches the latent representation produced by the text-to-video generator and stitch the two parts together. That operation requires only a small dataset and no labels. Second, the text-to-video generator must be aligned with the stitched 3D decoder, to ensure that the generated...
  </details>  
  Keywords: text-to-3d  
- **[BSGS: Bi-stage 3D Gaussian Splatting for Camera Motion Deblurring](https://arxiv.org/abs/2510.12493v2)**  
  Authors: An Zhao, Piaopiao Yu, Zhe Zhu, Mingqiang Wei  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2510.12493v2.pdf) | [![GitHub](https://img.shields.io/github/stars/wsxujm/bsgs?style=social)](https://github.com/wsxujm/bsgs)  
  <details><summary>Abstract</summary>

  3D Gaussian Splatting has exhibited remarkable capabilities in 3D scene reconstruction. However, reconstructing high-quality 3D scenes from motion-blurred images caused by camera motion poses a significant challenge.The performance of existing 3DGS-based deblurring methods are limited due to their inherent mechanisms, such as extreme dependence on the accuracy of camera poses and inability to effectively control erroneous Gaussian primitives densification caused by motion blur. To solve these problems, we introduce a novel framework, Bi-Stage 3D Gaussian Splatting, to accurately reconstruct 3D scenes from motion-blurred images. BSGS contains two stages. First, Camera Pose Refinement roughly optimizes camera poses to reduce motion-induced distortions. Second, with fixed rough camera poses, Global RigidTransformation further corrects motion-induced blur distortions. To alleviate multi-subframe gradient conflicts, we propose a subframe gradient aggregation strategy to optimize both stages. Furthermore, a space-time bi-stage optimization strategy is introduced to dynamically adjust primitive densification thresholds and prevent premature noisy Gaussian generation in...
  </details>  
  Keywords: 3d scene reconstruction  
- **[G4Splat: Geometry-Guided Gaussian Splatting with Generative Prior](https://arxiv.org/abs/2510.12099v2)**  
  Authors: Junfeng Ni, Yixin Chen, Zhifei Yang, Yu Liu, Ruijie Lu, Song-Chun Zhu, Siyuan Huang  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2510.12099v2.pdf) | [![Project](https://img.shields.io/badge/-Project-blue)](https://dali-jack.github.io/g4splat-web)  
  <details><summary>Abstract</summary>

  Despite recent advances in leveraging generative prior from pre-trained diffusion models for 3D scene reconstruction, existing methods still face two critical limitations. First, due to the lack of reliable geometric supervision, they struggle to produce high-quality reconstructions even in observed regions, let alone in unobserved areas. Second, they lack effective mechanisms to mitigate multi-view inconsistencies in the generated images, leading to severe shape-appearance ambiguities and degraded scene geometry. In this paper, we identify accurate geometry as the fundamental prerequisite for effectively exploiting generative models to enhance 3D scene reconstruction. We first propose to leverage the prevalence of planar structures to derive accurate metric-scale depth maps, providing reliable supervision in both observed and unobserved regions. Furthermore, we incorporate this geometry guidance throughout the generative pipeline to improve visibility mask estimation, guide novel view selection, and enhance multi-view consistency when inpainting with video diffusion models, resulting in accurate and consistent scene completion....
  </details>  
  Keywords: 3d scene reconstruction  
- **[High-resolution Photo Enhancement in Real-time: A Laplacian Pyramid Network](https://arxiv.org/abs/2510.11613v1)**  
  Authors: Feng Zhang, Haoyou Deng, Zhiqiang Li, Lida Li, Bin Xu, Qingbo Lu, Zisheng Cao, Minchen Wei, Changxin Gao, Nong Sang, Xiang Bai  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2510.11613v1.pdf) | [![GitHub](https://img.shields.io/github/stars/fengzhang427/LLF-LUT?style=social)](https://github.com/fengzhang427/LLF-LUT)  
  <details><summary>Abstract</summary>

  Photo enhancement plays a crucial role in augmenting the visual aesthetics of a photograph. In recent years, photo enhancement methods have either focused on enhancement performance, producing powerful models that cannot be deployed on edge devices, or prioritized computational efficiency, resulting in inadequate performance for real-world applications. To this end, this paper introduces a pyramid network called LLF-LUT++, which integrates global and local operators through closed-form Laplacian pyramid decomposition and reconstruction. This approach enables fast processing of high-resolution images while also achieving excellent performance. Specifically, we utilize an image-adaptive 3D LUT that capitalizes on the global tonal characteristics of downsampled images, while incorporating two distinct weight fusion strategies to achieve coarse global image enhancement. To implement this strategy, we designed a spatial-frequency transformer weight predictor that effectively extracts the desired distinct weights by leveraging frequency features. Additionally, we apply local Laplacian filters to adaptively refine edge details in high-frequency components....
  </details>  
- **[P-4DGS: Predictive 4D Gaussian Splatting with 90$\times$ Compression](https://arxiv.org/abs/2510.10030v1)**  
  Authors: Henan Wang, Hanxin Zhu, Xinliang Gong, Tianyu He, Xin Li, Zhibo Chen  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2510.10030v1.pdf)  
  <details><summary>Abstract</summary>

  3D Gaussian Splatting (3DGS) has garnered significant attention due to its superior scene representation fidelity and real-time rendering performance, especially for dynamic 3D scene reconstruction (\textit{i.e.}, 4D reconstruction). However, despite achieving promising results, most existing algorithms overlook the substantial temporal and spatial redundancies inherent in dynamic scenes, leading to prohibitive memory consumption. To address this, we propose P-4DGS, a novel dynamic 3DGS representation for compact 4D scene modeling. Inspired by intra- and inter-frame prediction techniques commonly used in video compression, we first design a 3D anchor point-based spatial-temporal prediction module to fully exploit the spatial-temporal correlations across different 3D Gaussian primitives. Subsequently, we employ an adaptive quantization strategy combined with context-based entropy coding to further reduce the size of the 3D anchor points, thereby achieving enhanced compression efficiency. To evaluate the rate-distortion performance of our proposed P-4DGS in comparison with other dynamic 3DGS representations, we conduct extensive experiments on both...
  </details>  
  Keywords: 3d scene reconstruction  
- **[FLOWING: Implicit Neural Flows for Structure-Preserving Morphing](https://arxiv.org/abs/2510.09537v1)**  
  Authors: Arthur Bizzi, Matias Grynberg, Vitor Matias, Daniel Perazzo, João Paulo Lima, Luiz Velho, Nuno Gonçalves, João Pereira, Guilherme Schardong, Tiago Novello  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2510.09537v1.pdf) | [![Project](https://img.shields.io/badge/-Project-blue)](http://schardong.github.io/flowing)  
  <details><summary>Abstract</summary>

  Morphing is a long-standing problem in vision and computer graphics, requiring a time-dependent warping for feature alignment and a blending for smooth interpolation. Recently, multilayer perceptrons (MLPs) have been explored as implicit neural representations (INRs) for modeling such deformations, due to their meshlessness and differentiability; however, extracting coherent and accurate morphings from standard MLPs typically relies on costly regularizations, which often lead to unstable training and prevent effective feature alignment. To overcome these limitations, we propose FLOWING (FLOW morphING), a framework that recasts warping as the construction of a differential vector flow, naturally ensuring continuity, invertibility, and temporal coherence by encoding structural flow properties directly into the network architectures. This flow-centric approach yields principled and stable transformations, enabling accurate and structure-preserving morphing of both 2D images and 3D shapes. Extensive experiments across a range of applications - including face and image morphing, as well as Gaussian Splatting morphing - show...
  </details>  
- **[Synthetic Object Compositions for Scalable and Accurate Learning in Detection, Segmentation, and Grounding](https://arxiv.org/abs/2510.09110v4)**  
  Authors: Weikai Huang, Jieyu Zhang, Taoyang Jia, Chenhao Zheng, Ziqi Gao, Jae Sung Park, Winson Han, Ranjay Krishna  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2510.09110v4.pdf)  
  <details><summary>Abstract</summary>

  Visual grouping -- operationalized through tasks such as instance segmentation, visual grounding, and object detection -- enables applications ranging from robotic perception to photo editing. These fundamental problems in computer vision are powered by large-scale, painstakingly annotated datasets. Despite their impact, these datasets are costly to build, biased in coverage, and difficult to scale. Synthetic datasets offer a promising alternative but struggle with flexibility, accuracy, and compositional diversity. We introduce Synthetic Object Compositions (SOC), an accurate and scalable data synthesis pipeline via a novel object-centric composition strategy. It composes high-quality synthetic object segments into new images using 3D geometric layout augmentation and camera configuration augmentation with generative harmonization and mask-area-weighted blending, yielding accurate and diverse masks, boxes, and referring expressions. Models trained on just 100K of our synthetic images outperform those trained on larger real datasets (GRIT 20M, V3Det 200K) and synthetic pipelines (Copy-Paste, X-Paste, SynGround, SegGen) by +24-36% --...
  </details>  
- **[Understanding Exoplanet Habitability: A Bayesian ML Framework for Predicting Atmospheric Absorption Spectra](https://arxiv.org/abs/2510.08766v1)**  
  Authors: Vasuda Trehan, Kevin H. Knuth, M. J. Way  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2510.08766v1.pdf)  
  <details><summary>Abstract</summary>

  The evolution of space technology in recent years, fueled by advancements in computing such as Artificial Intelligence (AI) and machine learning (ML), has profoundly transformed our capacity to explore the cosmos. Missions like the James Webb Space Telescope (JWST) have made information about distant objects more easily accessible, resulting in extensive amounts of valuable data. As part of this work-in-progress study, we are working to create an atmospheric absorption spectrum prediction model for exoplanets. The eventual model will be based on both collected observational spectra and synthetic spectral data generated by the ROCKE-3D general circulation model (GCM) developed by the climate modeling program at NASA's Goddard Institute for Space Studies (GISS). In this initial study, spline curves are used to describe the bin heights of simulated atmospheric absorption spectra as a function of one of the values of the planetary parameters. Bayesian Adaptive Exploration is then employed to identify areas...
  </details>  
- **[SViM3D: Stable Video Material Diffusion for Single Image 3D Generation](https://arxiv.org/abs/2510.08271v2)**  
  Authors: Andreas Engelhardt, Mark Boss, Vikram Voleti, Chun-Han Yao, Hendrik P. A. Lensch, Varun Jampani  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2510.08271v2.pdf)  
  <details><summary>Abstract</summary>

  We present Stable Video Materials 3D (SViM3D), a framework to predict multi-view consistent physically based rendering (PBR) materials, given a single image. Recently, video diffusion models have been successfully used to reconstruct 3D objects from a single image efficiently. However, reflectance is still represented by simple material models or needs to be estimated in additional steps to enable relighting and controlled appearance edits. We extend a latent video diffusion model to output spatially varying PBR parameters and surface normals jointly with each generated view based on explicit camera control. This unique setup allows for relighting and generating a 3D asset using our model as neural prior. We introduce various mechanisms to this pipeline that improve quality in this ill-posed setting. We show state-of-the-art relighting and novel view synthesis performance on multiple object-centric datasets. Our method generalizes to diverse inputs, enabling the generation of relightable 3D assets useful in AR/VR, movies,...
  </details>  
  Keywords: novel view synthesis  
- **[A 3D Generation Framework from Cross Modality to Parameterized Primitive](https://arxiv.org/abs/2510.08656v1)**  
  Authors: Yiming Liang, Huan Yu, Zili Wang, Shuyou Zhang, Guodong Yi, Jin Wang, Jianrong Tan  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2510.08656v1.pdf)  
  <details><summary>Abstract</summary>

  Recent advancements in AI-driven 3D model generation have leveraged cross modality, yet generating models with smooth surfaces and minimizing storage overhead remain challenges. This paper introduces a novel multi-stage framework for generating 3D models composed of parameterized primitives, guided by textual and image inputs. In the framework, A model generation algorithm based on parameterized primitives, is proposed, which can identifies the shape features of the model constituent elements, and replace the elements with parameterized primitives with high quality surface. In addition, a corresponding model storage method is proposed, it can ensure the original surface quality of the model, while retaining only the parameters of parameterized primitives. Experiments on virtual scene dataset and real scene dataset demonstrate the effectiveness of our method, achieving a Chamfer Distance of 0.003092, a VIoU of 0.545, a F1-Score of 0.9139 and a NC of 0.8369, with primitive parameter files approximately 6KB in size. Our approach...
  </details>  
- **[SyncHuman: Synchronizing 2D and 3D Generative Models for Single-view Human Reconstruction](https://arxiv.org/abs/2510.07723v2)**  
  Authors: Wenyue Chen, Peng Li, Wangguandong Zheng, Chengfeng Zhao, Mengfei Li, Yaolong Zhu, Zhiyang Dou, Ronggang Wang, Yuan Liu  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2510.07723v2.pdf)  
  <details><summary>Abstract</summary>

  Photorealistic 3D full-body human reconstruction from a single image is a critical yet challenging task for applications in films and video games due to inherent ambiguities and severe self-occlusions. While recent approaches leverage SMPL estimation and SMPL-conditioned image generative models to hallucinate novel views, they suffer from inaccurate 3D priors estimated from SMPL meshes and have difficulty in handling difficult human poses and reconstructing fine details. In this paper, we propose SyncHuman, a novel framework that combines 2D multiview generative model and 3D native generative model for the first time, enabling high-quality clothed human mesh reconstruction from single-view images even under challenging human poses. Multiview generative model excels at capturing fine 2D details but struggles with structural consistency, whereas 3D native generative model generates coarse yet structurally consistent 3D shapes. By integrating the complementary strengths of these two approaches, we develop a more effective generation framework. Specifically, we first jointly...
  </details>  
- **[Generating Surface for Text-to-3D using 2D Gaussian Splatting](https://arxiv.org/abs/2510.06967v1)**  
  Authors: Huanning Dong, Fan Li, Ping Kuang, Jianwen Min  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2510.06967v1.pdf)  
  <details><summary>Abstract</summary>

  Recent advancements in Text-to-3D modeling have shown significant potential for the creation of 3D content. However, due to the complex geometric shapes of objects in the natural world, generating 3D content remains a challenging task. Current methods either leverage 2D diffusion priors to recover 3D geometry, or train the model directly based on specific 3D representations. In this paper, we propose a novel method named DirectGaussian, which focuses on generating the surfaces of 3D objects represented by surfels. In DirectGaussian, we utilize conditional text generation models and the surface of a 3D object is rendered by 2D Gaussian splatting with multi-view normal and texture priors. For multi-view geometric consistency problems, DirectGaussian incorporates curvature constraints on the generated surface during optimization process. Through extensive experiments, we demonstrate that our framework is capable of achieving diverse and high-fidelity 3D content creation.
  </details>  
  Keywords: text-to-3d  
- **[OBJVanish: Physically Realizable Text-to-3D Adv. Generation of LiDAR-Invisible Objects](https://arxiv.org/abs/2510.06952v1)**  
  Authors: Bing Li, Wuqi Wang, Yanan Zhang, Jingzheng Li, Haigen Min, Wei Feng, Xingyu Zhao, Jie Zhang, Qing Guo  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2510.06952v1.pdf)  
  <details><summary>Abstract</summary>

  LiDAR-based 3D object detectors are fundamental to autonomous driving, where failing to detect objects poses severe safety risks. Developing effective 3D adversarial attacks is essential for thoroughly testing these detection systems and exposing their vulnerabilities before real-world deployment. However, existing adversarial attacks that add optimized perturbations to 3D points have two critical limitations: they rarely cause complete object disappearance and prove difficult to implement in physical environments. We introduce the text-to-3D adversarial generation method, a novel approach enabling physically realizable attacks that can generate 3D models of objects truly invisible to LiDAR detectors and be easily realized in the real world. Specifically, we present the first empirical study that systematically investigates the factors influencing detection vulnerability by manipulating the topology, connectivity, and intensity of individual pedestrian 3D models and combining pedestrians with multiple objects within the CARLA simulation environment. Building on the insights, we propose the physically-informed text-to-3D adversarial generation...
  </details>  
  Keywords: text-to-3d  
- **[A Comparative Study of Vision Transformers and CNNs for Few-Shot Rigid Transformation and Fundamental Matrix Estimation](https://arxiv.org/abs/2510.04794v1)**  
  Authors: Alon Kaya, Igal Bilik, Inna Stainvas  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2510.04794v1.pdf)  
  <details><summary>Abstract</summary>

  Vision-transformers (ViTs) and large-scale convolution-neural-networks (CNNs) have reshaped computer vision through pretrained feature representations that enable strong transfer learning for diverse tasks. However, their efficiency as backbone architectures for geometric estimation tasks involving image deformations in low-data regimes remains an open question. This work considers two such tasks: 1) estimating 2D rigid transformations between pairs of images and 2) predicting the fundamental matrix for stereo image pairs, an important problem in various applications, such as autonomous mobility, robotics, and 3D scene reconstruction. Addressing this intriguing question, this work systematically compares large-scale CNNs (ResNet, EfficientNet, CLIP-ResNet) with ViT-based foundation models (CLIP-ViT variants and DINO) in various data size settings, including few-shot scenarios. These pretrained models are optimized for classification or contrastive learning, encouraging them to focus mostly on high-level semantics. The considered tasks require balancing local and global features differently, challenging the straightforward adoption of these models as the backbone. Empirical...
  </details>  
  Keywords: 3d scene reconstruction  
- **[MetaFind: Scene-Aware 3D Asset Retrieval for Coherent Metaverse Scene Generation](https://arxiv.org/abs/2510.04057v1)**  
  Authors: Zhenyu Pan, Yucheng Lu, Han Liu  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2510.04057v1.pdf)  
  <details><summary>Abstract</summary>

  We present MetaFind, a scene-aware tri-modal compositional retrieval framework designed to enhance scene generation in the metaverse by retrieving 3D assets from large-scale repositories. MetaFind addresses two core challenges: (i) inconsistent asset retrieval that overlooks spatial, semantic, and stylistic constraints, and (ii) the absence of a standardized retrieval paradigm specifically tailored for 3D asset retrieval, as existing approaches mainly rely on general-purpose 3D shape representation models. Our key innovation is a flexible retrieval mechanism that supports arbitrary combinations of text, image, and 3D modalities as queries, enhancing spatial reasoning and style consistency by jointly modeling object-level features (including appearance) and scene-level layout structures. Methodologically, MetaFind introduces a plug-and-play equivariant layout encoder ESSGNN that captures spatial relationships and object appearance features, ensuring retrieved 3D assets are contextually and stylistically coherent with the existing scene, regardless of coordinate frame transformations. The framework supports iterative scene construction by continuously adapting retrieval results to...
  </details>  
- **[Towards Scalable and Consistent 3D Editing](https://arxiv.org/abs/2510.02994v1)**  
  Authors: Ruihao Xia, Yang Tang, Pan Zhou  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2510.02994v1.pdf) | [![Project](https://img.shields.io/badge/-Project-blue)](https://www.lv-lab.org/3DEditFormer)  
  <details><summary>Abstract</summary>

  3D editing - the task of locally modifying the geometry or appearance of a 3D asset - has wide applications in immersive content creation, digital entertainment, and AR/VR. However, unlike 2D editing, it remains challenging due to the need for cross-view consistency, structural fidelity, and fine-grained controllability. Existing approaches are often slow, prone to geometric distortions, or dependent on manual and accurate 3D masks that are error-prone and impractical. To address these challenges, we advance both the data and model fronts. On the data side, we introduce 3DEditVerse, the largest paired 3D editing benchmark to date, comprising 116,309 high-quality training pairs and 1,500 curated test pairs. Built through complementary pipelines of pose-driven geometric edits and foundation model-guided appearance edits, 3DEditVerse ensures edit locality, multi-view consistency, and semantic alignment. On the model side, we propose 3DEditFormer, a 3D-structure-preserving conditional transformer. By enhancing image-to-3D generation with dual-guidance attention and time-adaptive gating, 3DEditFormer...
  </details>  
  Keywords: image-to-3d  
- **[SemMorph3D: Unsupervised Semantic-Aware 3D Morphing via Mesh-Guided Gaussians](https://arxiv.org/abs/2510.02034v2)**  
  Authors: Mengtian Li, Yunshu Bai, Yimin Chu, Xinru Guo, Haolin Liu, Zhifeng Xie, Chaofeng Chen  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2510.02034v2.pdf) | [![Project](https://img.shields.io/badge/-Project-blue)](https://baiyunshu.github.io/GAUSSIANMORPHING.github.io)  
  <details><summary>Abstract</summary>

  We introduce METHODNAME, a novel framework for semantic-aware 3D shape and texture morphing directly from multi-view images. While 3D Gaussian Splatting (3DGS) enables photorealistic rendering, its unstructured nature often leads to catastrophic geometric fragmentation during morphing. Conversely, traditional mesh-based morphing enforces structural integrity but mandates pristine input topology and struggles with complex appearances. Our method resolves this dichotomy by employing a mesh-guided strategy where a coarse, extracted base mesh acts as a flexible geometric anchor. This anchor provides the necessary topological scaffolding to guide unstructured Gaussians, successfully compensating for mesh extraction artifacts and topological limitations. Furthermore, we propose a novel dual-domain optimization strategy that leverages this hybrid representation to establish unsupervised semantic correspondence, synergizing geodesic regularizations for shape preservation with texture-aware constraints for coherent color evolution. This integrated approach ensures stable, physically plausible transformations without requiring labeled data, specialized 3D assets, or category-specific templates. On the proposed TexMorph benchmark, METHODNAME...
  </details>  
- **[LoBE-GS: Load-Balanced and Efficient 3D Gaussian Splatting for Large-Scale Scene Reconstruction](https://arxiv.org/abs/2510.01767v2)**  
  Authors: Sheng-Hsiang Hung, Ting-Yu Yen, Wei-Fang Sun, Simon See, Shih-Hsuan Hung, Hung-Kuo Chu  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2510.01767v2.pdf)  
  <details><summary>Abstract</summary>

  3D Gaussian Splatting (3DGS) has established itself as an efficient representation for real-time, high-fidelity 3D scene reconstruction. However, scaling 3DGS to large and unbounded scenes such as city blocks remains difficult. Existing divide-and-conquer methods alleviate memory pressure by partitioning the scene into blocks and training on multiple, non-communicating GPUs, but introduce new bottlenecks: (i) partitions suffer from severe load imbalance since uniform or heuristic splits do not reflect actual computational demands, and (ii) coarse-to-fine pipelines fail to exploit the coarse stage efficiently, often reloading the entire model and incurring high overhead. In this work, we introduce LoBE-GS, a novel Load-Balanced and Efficient 3D Gaussian Splatting framework, that re-engineers the large-scale 3DGS pipeline. Specifically, LoBE-GS introduces a load-balanced KD-tree scene partitioning scheme with optimized cutlines that balance per-block camera counts. To accelerate preprocessing, it employs depth-based back-projection for fast camera assignment, reducing processing time from hours to minutes. It further reduces...
  </details>  
  Keywords: 3d scene reconstruction  

### November 2025
- **[LISA-3D: Lifting Language-Image Segmentation to 3D via Multi-View Consistency](https://arxiv.org/abs/2512.01008v1)**  
  Authors: Zhongbin Guo, Jiahe Liu, Wenyu Gao, Yushan Li, Chengzhi Li, Ping Jian  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2512.01008v1.pdf) | [![GitHub](https://img.shields.io/github/stars/binisalegend/LISA-3D?style=social)](https://github.com/binisalegend/LISA-3D)  
  <details><summary>Abstract</summary>

  Text-driven 3D reconstruction demands a mask generator that simultaneously understands open-vocabulary instructions and remains consistent across viewpoints. We present LISA-3D, a two-stage framework that lifts language-image segmentation into 3D by retrofitting the instruction-following model LISA with geometry-aware Low-Rank Adaptation (LoRA) layers and reusing a frozen SAM-3D reconstructor. During training we exploit off-the-shelf RGB-D sequences and their camera poses to build a differentiable reprojection loss that enforces cross-view agreement without requiring any additional 3D-text supervision. The resulting masks are concatenated with RGB images to form RGBA prompts for SAM-3D, which outputs Gaussian splats or textured meshes without retraining. Across ScanRefer and Nr3D, LISA-3D improves language-to-3D accuracy by up to +15.6 points over single-view baselines while adapting only 11.6M parameters. The system is modular, data-efficient, and supports zero-shot deployment on unseen categories, providing a practical recipe for language-guided 3D content creation. Our code will be available at https://github.com/binisalegend/LISA-3D.
  </details>  
- **[EAG3R: Event-Augmented 3D Geometry Estimation for Dynamic and Extreme-Lighting Scenes](https://arxiv.org/abs/2512.00771v2)**  
  Authors: Xiaoshan Wu, Yifei Yu, Xiaoyang Lyu, Yihua Huang, Bo Wang, Baoheng Zhang, Zhongrui Wang, Xiaojuan Qi  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2512.00771v2.pdf)  
  <details><summary>Abstract</summary>

  Robust 3D geometry estimation from videos is critical for applications such as autonomous navigation, SLAM, and 3D scene reconstruction. Recent methods like DUSt3R demonstrate that regressing dense pointmaps from image pairs enables accurate and efficient pose-free reconstruction. However, existing RGB-only approaches struggle under real-world conditions involving dynamic objects and extreme illumination, due to the inherent limitations of conventional cameras. In this paper, we propose EAG3R, a novel geometry estimation framework that augments pointmap-based reconstruction with asynchronous event streams. Built upon the MonST3R backbone, EAG3R introduces two key innovations: (1) a retinex-inspired image enhancement module and a lightweight event adapter with SNR-aware fusion mechanism that adaptively combines RGB and event features based on local reliability; and (2) a novel event-based photometric consistency loss that reinforces spatiotemporal coherence during global optimization. Our method enables robust geometry estimation in challenging dynamic low-light scenes without requiring retraining on night-time data. Extensive experiments demonstrate that...
  </details>  
  Keywords: 3d scene reconstruction  
- **[Asset-Driven Sematic Reconstruction of Dynamic Scene with Multi-Human-Object Interactions](https://arxiv.org/abs/2512.00547v1)**  
  Authors: Sandika Biswas, Qianyi Wu, Biplab Banerjee, Hamid Rezatofighi  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2512.00547v1.pdf)  
  <details><summary>Abstract</summary>

  Real-world human-built environments are highly dynamic, involving multiple humans and their complex interactions with surrounding objects. While 3D geometry modeling of such scenes is crucial for applications like AR/VR, gaming, and embodied AI, it remains underexplored due to challenges like diverse motion patterns and frequent occlusions. Beyond novel view rendering, 3D Gaussian Splatting (GS) has demonstrated remarkable progress in producing detailed, high-quality surface geometry with fast optimization of the underlying structure. However, very few GS-based methods address multihuman, multiobject scenarios, primarily due to the above-mentioned inherent challenges. In a monocular setup, these challenges are further amplified, as maintaining structural consistency under severe occlusion becomes difficult when the scene is optimized solely based on GS-based rendering loss. To tackle the challenges of such a multihuman, multiobject dynamic scene, we propose a hybrid approach that effectively combines the advantages of 1) 3D generative models for generating high-fidelity meshes of the scene elements,...
  </details>  
- **[CC-FMO: Camera-Conditioned Zero-Shot Single Image to 3D Scene Generation with Foundation Model Orchestration](https://arxiv.org/abs/2512.00493v1)**  
  Authors: Boshi Tang, Henry Zheng, Rui Huang, Gao Huang  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2512.00493v1.pdf)  
  <details><summary>Abstract</summary>

  High-quality 3D scene generation from a single image is crucial for AR/VR and embodied AI applications. Early approaches struggle to generalize due to reliance on specialized models trained on curated small datasets. While recent advancements in large-scale 3D foundation models have significantly enhanced instance-level generation, coherent scene generation remains a challenge, where performance is limited by inaccurate per-object pose estimations and spatial inconsistency. To this end, this paper introduces CC-FMO, a zero-shot, camera-conditioned pipeline for single-image to 3D scene generation that jointly conforms to the object layout in input image and preserves instance fidelity. CC-FMO employs a hybrid instance generator that combines semantics-aware vector-set representation with detail-rich structured latent representation, yielding object geometries that are both semantically plausible and high-quality. Furthermore, CC-FMO enables the application of foundational pose estimation models in the scene generation task via a simple yet effective camera-conditioned scale-solving algorithm, to enforce scene-level coherence. Extensive experiments demonstrate...
  </details>  
- **[SplatFont3D: Structure-Aware Text-to-3D Artistic Font Generation with Part-Level Style Control](https://arxiv.org/abs/2512.00413v1)**  
  Authors: Ji Gan, Lingxu Chen, Jiaxu Leng, Xinbo Gao  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2512.00413v1.pdf)  
  <details><summary>Abstract</summary>

  Artistic font generation (AFG) can assist human designers in creating innovative artistic fonts. However, most previous studies primarily focus on 2D artistic fonts in flat design, leaving personalized 3D-AFG largely underexplored. 3D-AFG not only enables applications in immersive 3D environments such as video games and animations, but also may enhance 2D-AFG by rendering 2D fonts of novel views. Moreover, unlike general 3D objects, 3D fonts exhibit precise semantics with strong structural constraints and also demand fine-grained part-level style control. To address these challenges, we propose SplatFont3D, a novel structure-aware text-to-3D AFG framework with 3D Gaussian splatting, which enables the creation of 3D artistic fonts from diverse style text prompts with precise part-level style control. Specifically, we first introduce a Glyph2Cloud module, which progressively enhances both the shapes and styles of 2D glyphs (or components) and produces their corresponding 3D point clouds for Gaussian initialization. The initialized 3D Gaussians are further...
  </details>  
  Keywords: text-to-3d  
- **[Object-Centric Data Synthesis for Category-level Object Detection](https://arxiv.org/abs/2511.23450v1)**  
  Authors: Vikhyat Agarwal, Jiayi Cora Guo, Declan Hoban, Sissi Zhang, Nicholas Moran, Peter Cho, Srilakshmi Pattabiraman, Shantanu Joshi  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2511.23450v1.pdf)  
  <details><summary>Abstract</summary>

  Deep learning approaches to object detection have achieved reliable detection of specific object classes in images. However, extending a model's detection capability to new object classes requires large amounts of annotated training data, which is costly and time-consuming to acquire, especially for long-tailed classes with insufficient representation in existing datasets. Here, we introduce the object-centric data setting, when limited data is available in the form of object-centric data (multi-view images or 3D models), and systematically evaluate the performance of four different data synthesis methods to finetune object detection models on novel object categories in this setting. The approaches are based on simple image processing techniques, 3D rendering, and image diffusion models, and use object-centric data to synthesize realistic, cluttered images with varying contextual coherence and complexity. We assess how these methods enable models to achieve category-level generalization in real-world data, and demonstrate significant performance boosts within this data-constrained experimental setting.
  </details>  
- **[Learning to Predict Aboveground Biomass from RGB Images with 3D Synthetic Scenes](https://arxiv.org/abs/2511.23249v1)**  
  Authors: Silvia Zuffi  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2511.23249v1.pdf)  
  <details><summary>Abstract</summary>

  Forests play a critical role in global ecosystems by supporting biodiversity and mitigating climate change via carbon sequestration. Accurate aboveground biomass (AGB) estimation is essential for assessing carbon storage and wildfire fuel loads, yet traditional methods rely on labor-intensive field measurements or remote sensing approaches with significant limitations in dense vegetation. In this work, we propose a novel learning-based method for estimating AGB from a single ground-based RGB image. We frame this as a dense prediction task, introducing AGB density maps, where each pixel represents tree biomass normalized by the plot area and each tree's image area. We leverage the recently introduced synthetic 3D SPREAD dataset, which provides realistic forest scenes with per-image tree attributes (height, trunk and canopy diameter) and instance segmentation masks. Using these assets, we compute AGB via allometric equations and train a model to predict AGB density maps, integrating them to recover the AGB estimate for...
  </details>  
- **[GeoWorld: Unlocking the Potential of Geometry Models to Facilitate High-Fidelity 3D Scene Generation](https://arxiv.org/abs/2511.23191v1)**  
  Authors: Yuhao Wan, Lijuan Liu, Jingzhi Zhou, Zihan Zhou, Xuying Zhang, Dongbo Zhang, Shaohui Jiao, Qibin Hou, Ming-Ming Cheng  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2511.23191v1.pdf) | [![Project](https://img.shields.io/badge/-Project-blue)](https://peaes.github.io/GeoWorld)  
  <details><summary>Abstract</summary>

  Previous works leveraging video models for image-to-3D scene generation tend to suffer from geometric distortions and blurry content. In this paper, we renovate the pipeline of image-to-3D scene generation by unlocking the potential of geometry models and present our GeoWorld. Instead of exploiting geometric information obtained from a single-frame input, we propose to first generate consecutive video frames and then take advantage of the geometry model to provide full-frame geometry features, which contain richer information than single-frame depth maps or camera embeddings used in previous methods, and use these geometry features as geometrical conditions to aid the video generation model. To enhance the consistency of geometric structures, we further propose a geometry alignment loss to provide the model with real-world geometric constraints and a geometry adaptation module to ensure the effective utilization of geometry features. Extensive experiments show that our GeoWorld can generate high-fidelity 3D scenes from a single image...
  </details>  
  Keywords: image-to-3d  
- **[Fast Multi-view Consistent 3D Editing with Video Priors](https://arxiv.org/abs/2511.23172v2)**  
  Authors: Liyi Chen, Ruihuang Li, Guowen Zhang, Pengfei Wang, Lei Zhang  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2511.23172v2.pdf)  
  <details><summary>Abstract</summary>

  Text-driven 3D editing enables user-friendly 3D object or scene editing with text instructions. Due to the lack of multi-view consistency priors, existing methods typically resort to employing 2D generation or editing models to process each view individually, followed by iterative 2D-3D-2D updating. However, these methods are not only time-consuming but also prone to over-smoothed results because the different editing signals gathered from different views are averaged during the iterative process. In this paper, we propose generative Video Prior based 3D Editing (ViP3DE) to employ the temporal consistency priors from pre-trained video generation models for multi-view consistent 3D editing in a single forward pass. Our key insight is to condition the video generation model on a single edited view to generate other consistent edited views for 3D updating directly, thereby bypassing the iterative editing paradigm. Since 3D updating requires edited views to be paired with specific camera poses, we propose motion-preserved...
  </details>  
- **[Image Valuation in NeRF-based 3D reconstruction](https://arxiv.org/abs/2511.23052v1)**  
  Authors: Grigorios Aris Cheimariotis, Antonis Karakottas, Vangelis Chatzis, Angelos Kanlis, Dimitrios Zarpalas  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2511.23052v1.pdf)  
  <details><summary>Abstract</summary>

  Data valuation and monetization are becoming increasingly important across domains such as eXtended Reality (XR) and digital media. In the context of 3D scene reconstruction from a set of images -- whether casually or professionally captured -- not all inputs contribute equally to the final output. Neural Radiance Fields (NeRFs) enable photorealistic 3D reconstruction of scenes by optimizing a volumetric radiance field given a set of images. However, in-the-wild scenes often include image captures of varying quality, occlusions, and transient objects, resulting in uneven utility across inputs. In this paper we propose a method to quantify the individual contribution of each image to NeRF-based reconstructions of in-the-wild image sets. Contribution is assessed through reconstruction quality metrics based on PSNR and MSE. We validate our approach by removing low-contributing images during training and measuring the resulting impact on reconstruction fidelity.
  </details>  
  Keywords: 3d scene reconstruction  
- **[MICCAI STS 2024 Challenge: Semi-Supervised Instance-Level Tooth Segmentation in Panoramic X-ray and CBCT Images](https://arxiv.org/abs/2511.22911v1)**  
  Authors: Yaqi Wang, Zhi Li, Chengyu Wu, Jun Liu, Yifan Zhang, Jiaxue Ni, Qian Luo, Jialuo Chen, Hongyuan Zhang, Jin Liu, Can Han, Kaiwen Fu, Changkai Ji, Xinxu Cai, Jing Hao, Zhihao Zheng, Shi Xu, Junqiang Chen, Qianni Zhang, Dahong Qian, Shuai Wang, Huiyu Zhou  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2511.22911v1.pdf) | [![GitHub](https://img.shields.io/github/stars/ricoleehduu/STS-Challenge-2024?style=social)](https://github.com/ricoleehduu/STS-Challenge-2024)  
  <details><summary>Abstract</summary>

  Orthopantomogram (OPGs) and Cone-Beam Computed Tomography (CBCT) are vital for dentistry, but creating large datasets for automated tooth segmentation is hindered by the labor-intensive process of manual instance-level annotation. This research aimed to benchmark and advance semi-supervised learning (SSL) as a solution for this data scarcity problem. We organized the 2nd Semi-supervised Teeth Segmentation (STS 2024) Challenge at MICCAI 2024. We provided a large-scale dataset comprising over 90,000 2D images and 3D axial slices, which includes 2,380 OPG images and 330 CBCT scans, all featuring detailed instance-level FDI annotations on part of the data. The challenge attracted 114 (OPG) and 106 (CBCT) registered teams. To ensure algorithmic excellence and full transparency, we rigorously evaluated the valid, open-source submissions from the top 10 (OPG) and top 5 (CBCT) teams, respectively. All successful submissions were deep learning-based SSL methods. The winning semi-supervised models demonstrated impressive performance gains over a fully-supervised nnU-Net baseline...
  </details>  
- **[ITS3D: Inference-Time Scaling for Text-Guided 3D Diffusion Models](https://arxiv.org/abs/2511.22456v1)**  
  Authors: Zhenglin Zhou, Fan Ma, Xiaobo Xia, Hehe Fan, Yi Yang, Tat-Seng Chua  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2511.22456v1.pdf) | [![GitHub](https://img.shields.io/github/stars/ZhenglinZhou/ITS3D?style=social)](https://github.com/ZhenglinZhou/ITS3D)  
  <details><summary>Abstract</summary>

  We explore inference-time scaling in text-guided 3D diffusion models to enhance generative quality without additional training. To this end, we introduce ITS3D, a framework that formulates the task as an optimization problem to identify the most effective Gaussian noise input. The framework is driven by a verifier-guided search algorithm, where the search algorithm iteratively refines noise candidates based on verifier feedback. To address the inherent challenges of 3D generation, we introduce three techniques for improved stability, efficiency, and exploration capability. 1) Gaussian normalization is applied to stabilize the search process. It corrects distribution shifts when noise candidates deviate from a standard Gaussian distribution during iterative updates. 2) The high-dimensional nature of the 3D search space increases computational complexity. To mitigate this, a singular value decomposition-based compression technique is employed to reduce dimensionality while preserving effective search directions. 3) To further prevent convergence to suboptimal local minima, a singular space reset...
  </details>  
  Keywords: text-to-3d  
- **[Cue3D: Quantifying the Role of Image Cues in Single-Image 3D Generation](https://arxiv.org/abs/2511.22121v1)**  
  Authors: Xiang Li, Zirui Wang, Zixuan Huang, James M. Rehg  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2511.22121v1.pdf)  
  <details><summary>Abstract</summary>

  Humans and traditional computer vision methods rely on a diverse set of monocular cues to infer 3D structure from a single image, such as shading, texture, silhouette, etc. While recent deep generative models have dramatically advanced single-image 3D generation, it remains unclear which image cues these methods actually exploit. We introduce Cue3D, the first comprehensive, model-agnostic framework for quantifying the influence of individual image cues in single-image 3D generation. Our unified benchmark evaluates seven state-of-the-art methods, spanning regression-based, multi-view, and native 3D generative paradigms. By systematically perturbing cues such as shading, texture, silhouette, perspective, edges, and local continuity, we measure their impact on 3D output quality. Our analysis reveals that shape meaningfulness, not texture, dictates generalization. Geometric cues, particularly shading, are crucial for 3D generation. We further identify over-reliance on provided silhouettes and diverse sensitivities to cues such as perspective and local continuity across model families. By dissecting these dependencies,...
  </details>  
- **[PAT3D: Physics-Augmented Text-to-3D Scene Generation](https://arxiv.org/abs/2511.21978v2)**  
  Authors: Guying Lin, Kemeng Huang, Michael Liu, Ruihan Gao, Hanke Chen, Lyuhao Chen, Beijia Lu, Taku Komura, Yuan Liu, Jun-Yan Zhu, Minchen Li  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2511.21978v2.pdf) | [![GitHub](https://img.shields.io/github/stars/Simulation-Intelligence/PAT3D?style=social)](https://github.com/Simulation-Intelligence/PAT3D)  
  <details><summary>Abstract</summary>

  We introduce PAT3D, the first physics-augmented text-to-3D scene generation framework that integrates vision-language models with physics-based simulation to produce physically plausible, simulation-ready, and intersection-free 3D scenes. Given a text prompt, PAT3D generates 3D objects, infers their spatial relations, and organizes them into a hierarchical scene tree, which is then converted into initial conditions for simulation. A differentiable rigid-body simulator ensures realistic object interactions under gravity, driving the scene toward static equilibrium without interpenetrations. To further enhance scene quality, we introduce a simulation-in-the-loop optimization procedure that guarantees physical stability and non-intersection, while improving semantic consistency with the input prompt. Experiments demonstrate that PAT3D substantially outperforms prior approaches in physical plausibility, semantic consistency, and visual quality. Beyond high-quality generation, PAT3D uniquely enables simulation-ready 3D scenes for downstream tasks such as scene editing and robotic manipulation. Code and data are available at: https://github.com/Simulation-Intelligence/PAT3D.
  </details>  
  Keywords: text-to-3d  
- **[GENA3D: Generative Amodal 3D Modeling by Bridging 2D Priors and 3D Coherence](https://arxiv.org/abs/2511.21945v2)**  
  Authors: Junwei Zhou, Yu-Wing Tai  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2511.21945v2.pdf)  
  <details><summary>Abstract</summary>

  Generating complete 3D objects under partial occlusions (i.e., amodal scenarios) is a practically important yet challenging problem, as large portions of object geometry are unobserved in real-world scenarios. Existing approaches either operate directly in 3D, which ensures geometric consistency but often lacks generative expressiveness, or rely on 2D amodal completion, which provides strong appearance priors but does not guarantee reliable 3D structure. This raises a key question: how can we achieve both generative plausibility and geometric coherence in amodal 3D modeling? To answer this question, we introduce GENA3D (GENarative Amodal 3D), a framework that integrates learned 2D generative priors with explicit 3D geometric reasoning within a conditional 3D generation paradigm. The 2D priors enable the model to plausibly infer diverse occluded content, while the 3D representation enforces multi-view consistency and spatial validity. Our design incorporates a novel View-Wise Cross-Attention for multi-view alignment and a Stereo-Conditioned Cross-Attention to anchor generative predictions...
  </details>  
- **[HTTM: Head-wise Temporal Token Merging for Faster VGGT](https://arxiv.org/abs/2511.21317v1)**  
  Authors: Weitian Wang, Lukas Meiner, Rai Shubham, Cecilia De La Parra, Akash Kumar  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2511.21317v1.pdf)  
  <details><summary>Abstract</summary>

  The Visual Geometry Grounded Transformer (VGGT) marks a significant leap forward in 3D scene reconstruction, as it is the first model that directly infers all key 3D attributes (camera poses, depths, and dense geometry) jointly in one pass. However, this joint inference mechanism requires global attention layers that perform all-to-all attention computation on tokens from all views. For reconstruction of large scenes with long-sequence inputs, this causes a significant latency bottleneck. In this paper, we propose head-wise temporal merging (HTTM), a training-free 3D token merging method for accelerating VGGT. Existing merging techniques merge tokens uniformly across different attention heads, resulting in identical tokens in the layers' output, which hinders the model's representational ability. HTTM tackles this problem by merging tokens in multi-head granularity, which preserves the uniqueness of feature tokens after head concatenation. Additionally, this enables HTTM to leverage the spatial locality and temporal correspondence observed at the head level...
  </details>  
  Keywords: 3d scene reconstruction  
- **[MODEST: Multi-Optics Depth-of-Field Stereo Dataset](https://arxiv.org/abs/2511.20853v3)**  
  Authors: Nisarg K. Trivedi, Vinayak A. Belludi, Li-Yun Wang  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2511.20853v3.pdf)  
  <details><summary>Abstract</summary>

  Reliable depth estimation under real optical conditions remains a core challenge for camera vision in systems such as autonomous robotics and augmented reality. Despite recent progress in depth estimation and depth-of-field rendering, research remains constrained by the lack of large-scale, high-fidelity, real stereo DSLR datasets, limiting real-world generalization and evaluation of models trained on synthetic data as shown extensively in literature. We present the first high-resolution (5472$\times$3648px) stereo DSLR dataset with 18000 images, systematically varying focal length and aperture across complex real scenes and capturing the optical realism and complexity of professional camera systems. For 9 scenes with varying scene complexity, lighting and background, images are captured with two identical camera assemblies at 10 focal lengths (28-70mm) and 5 apertures (f/2.8-f/22), spanning 50 optical configurations in 2000 images per scene. This full-range optics coverage enables controlled analysis of geometric and optical effects for monocular and stereo depth estimation, shallow depth-of-field...
  </details>  
  Keywords: 3d scene reconstruction, novel view synthesis  
- **[ShapeGen: Towards High-Quality 3D Shape Synthesis](https://arxiv.org/abs/2511.20624v1)**  
  Authors: Yangguang Li, Xianglong He, Zi-Xin Zou, Zexiang Liu, Wanli Ouyang, Ding Liang, Yan-Pei Cao  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2511.20624v1.pdf)  
  <details><summary>Abstract</summary>

  Inspired by generative paradigms in image and video, 3D shape generation has made notable progress, enabling the rapid synthesis of high-fidelity 3D assets from a single image. However, current methods still face challenges, including the lack of intricate details, overly smoothed surfaces, and fragmented thin-shell structures. These limitations leave the generated 3D assets still one step short of meeting the standards favored by artists. In this paper, we present ShapeGen, which achieves high-quality image-to-3D shape generation through 3D representation and supervision improvements, resolution scaling up, and the advantages of linear transformers. These advancements allow the generated assets to be seamlessly integrated into 3D pipelines, facilitating their widespread adoption across various applications. Through extensive experiments, we validate the impact of these improvements on overall performance. Ultimately, thanks to the synergistic effects of these enhancements, ShapeGen achieves a significant leap in image-to-3D generation, establishing a new state-of-the-art performance.
  </details>  
  Keywords: image-to-3d  
- **[MFM-point: Multi-scale Flow Matching for Point Cloud Generation](https://arxiv.org/abs/2511.20041v1)**  
  Authors: Petr Molodyk, Jaemoo Choi, David W. Romero, Ming-Yu Liu, Yongxin Chen  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2511.20041v1.pdf)  
  <details><summary>Abstract</summary>

  In recent years, point cloud generation has gained significant attention in 3D generative modeling. Among existing approaches, point-based methods directly generate point clouds without relying on other representations such as latent features, meshes, or voxels. These methods offer low training cost and algorithmic simplicity, but often underperform compared to representation-based approaches. In this paper, we propose MFM-Point, a multi-scale Flow Matching framework for point cloud generation that substantially improves the scalability and performance of point-based methods while preserving their simplicity and efficiency. Our multi-scale generation algorithm adopts a coarse-to-fine generation paradigm, enhancing generation quality and scalability without incurring additional training or inference overhead. A key challenge in developing such a multi-scale framework lies in preserving the geometric structure of unordered point clouds while ensuring smooth and consistent distributional transitions across resolutions. To address this, we introduce a structured downsampling and upsampling strategy that preserves geometry and maintains alignment between coarse...
  </details>  
- **[GigaWorld-0: World Models as Data Engine to Empower Embodied AI](https://arxiv.org/abs/2511.19861v2)**  
  Authors: GigaWorld Team, Angen Ye, Boyuan Wang, Chaojun Ni, Guan Huang, Guosheng Zhao, Haoyun Li, Jiagang Zhu, Kerui Li, Mengyuan Xu, Qiuping Deng, Siting Wang, Wenkang Qin, Xinze Chen, Xiaofeng Wang, Yankai Wang, Yu Cao, Yifan Chang, Yuan Xu, Yun Ye, Yang Wang, Yukun Zhou, Zhengyuan Zhang, Zhehao Dong, Zheng Zhu  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2511.19861v2.pdf)  
  <details><summary>Abstract</summary>

  World models are emerging as a foundational paradigm for scalable, data-efficient embodied AI. In this work, we present GigaWorld-0, a unified world model framework designed explicitly as a data engine for Vision-Language-Action (VLA) learning. GigaWorld-0 integrates two synergistic components: GigaWorld-0-Video, which leverages large-scale video generation to produce diverse, texture-rich, and temporally coherent embodied sequences under fine-grained control of appearance, camera viewpoint, and action semantics; and GigaWorld-0-3D, which combines 3D generative modeling, 3D Gaussian Splatting reconstruction, physically differentiable system identification, and executable motion planning to ensure geometric consistency and physical realism. Their joint optimization enables the scalable synthesis of embodied interaction data that is visually compelling, spatially coherent, physically plausible, and instruction-aligned. Training at scale is made feasible through our efficient GigaTrain framework, which exploits FP8-precision and sparse attention to drastically reduce memory and compute requirements. We conduct comprehensive evaluations showing that GigaWorld-0 generates high-quality, diverse, and controllable data across multiple...
  </details>  
- **[4DWorldBench: A Comprehensive Evaluation Framework for 3D/4D World Generation Models](https://arxiv.org/abs/2511.19836v1)**  
  Authors: Yiting Lu, Wei Luo, Peiyan Tu, Haoran Li, Hanxin Zhu, Zihao Yu, Xingrui Wang, Xinyi Chen, Xinge Peng, Xin Li, Zhibo Chen  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2511.19836v1.pdf) | [![Project](https://img.shields.io/badge/-Project-blue)](https://yeppp27.github.io/4DWorldBench.github.io)  
  <details><summary>Abstract</summary>

  World Generation Models are emerging as a cornerstone of next-generation multimodal intelligence systems. Unlike traditional 2D visual generation, World Models aim to construct realistic, dynamic, and physically consistent 3D/4D worlds from images, videos, or text. These models not only need to produce high-fidelity visual content but also maintain coherence across space, time, physics, and instruction control, enabling applications in virtual reality, autonomous driving, embodied intelligence, and content creation. However, prior benchmarks emphasize different evaluation dimensions and lack a unified assessment of world-realism capability. To systematically evaluate World Models, we introduce the 4DWorldBench, which measures models across four key dimensions: Perceptual Quality, Condition-4D Alignment, Physical Realism, and 4D Consistency. The benchmark covers tasks such as Image-to-3D/4D, Video-to-4D, Text-to-3D/4D. Beyond these, we innovatively introduce adaptive conditioning across multiple modalities, which not only integrates but also extends traditional evaluation paradigms. To accommodate different modality-conditioned inputs, we map all modality conditions into a unified...
  </details>  
  Keywords: text-to-3d, image-to-3d  
- **[Ref-SAM3D: Bridging SAM3D with Text for Reference 3D Reconstruction](https://arxiv.org/abs/2511.19426v1)**  
  Authors: Yun Zhou, Yaoting Wang, Guangquan Jie, Jinyu Liu, Henghui Ding  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2511.19426v1.pdf) | [![GitHub](https://img.shields.io/github/stars/FudanCVL/Ref-SAM3D?style=social)](https://github.com/FudanCVL/Ref-SAM3D)  
  <details><summary>Abstract</summary>

  SAM3D has garnered widespread attention for its strong 3D object reconstruction capabilities. However, a key limitation remains: SAM3D cannot reconstruct specific objects referred to by textual descriptions, a capability that is essential for practical applications such as 3D editing, game development, and virtual environments. To address this gap, we introduce Ref-SAM3D, a simple yet effective extension to SAM3D that incorporates textual descriptions as a high-level prior, enabling text-guided 3D reconstruction from a single RGB image. Through extensive qualitative experiments, we show that Ref-SAM3D, guided only by natural language and a single 2D view, delivers competitive and high-fidelity zero-shot reconstruction performance. Our results demonstrate that Ref-SAM3D effectively bridges the gap between 2D visual cues and 3D geometric understanding, offering a more flexible and accessible paradigm for reference-guided 3D reconstruction. Code is available at: https://github.com/FudanCVL/Ref-SAM3D.
  </details>  
- **[Three-Dimensional Anatomical Data Generation Based on Artificial Neural Networks](https://arxiv.org/abs/2511.19198v1)**  
  Authors: Ann-Sophia Müller, Moonkwang Jeong, Meng Zhang, Jiyuan Tian, Arkadiusz Miernik, Stefanie Speidel, Tian Qiu  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2511.19198v1.pdf)  
  <details><summary>Abstract</summary>

  Surgical planning and training based on machine learning requires a large amount of 3D anatomical models reconstructed from medical imaging, which is currently one of the major bottlenecks. Obtaining these data from real patients and during surgery is very demanding, if even possible, due to legal, ethical, and technical challenges. It is especially difficult for soft tissue organs with poor imaging contrast, such as the prostate. To overcome these challenges, we present a novel workflow for automated 3D anatomical data generation using data obtained from physical organ models. We additionally use a 3D Generative Adversarial Network (GAN) to obtain a manifold of 3D models useful for other downstream machine learning tasks that rely on 3D data. We demonstrate our workflow using an artificial prostate model made of biomimetic hydrogels with imaging contrast in multiple zones. This is used to physically simulate endoscopic surgery. For evaluation and 3D data generation, we...
  </details>  
- **[Yo'City: Personalized and Boundless 3D Realistic City Scene Generation via Self-Critic Expansion](https://arxiv.org/abs/2511.18734v3)**  
  Authors: Keyang Lu, Sifan Zhou, Hongbin Xu, Gang Xu, Zhifei Yang, Yikai Wang, Zhen Xiao, Jieyi Long, Ming Li  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2511.18734v3.pdf)  
  <details><summary>Abstract</summary>

  Realistic 3D city generation is fundamental to a wide range of applications, including virtual reality and digital twins. However, most existing methods rely on training a single diffusion model, which limits their ability to generate personalized and boundless city-scale scenes. In this paper, we present Yo'City, a novel agentic framework that enables user-customized and infinitely expandable 3D city generation by leveraging the reasoning and compositional capabilities of off-the-shelf large models. Specifically, Yo'City first conceptualizes the city through a top-down planning strategy that defines a hierarchical "City-District-Grid" structure. The Global Planner determines the overall layout and potential functional districts, while the Local Designer further refines each district with detailed grid-level descriptions. Subsequently, the grid-level 3D generation is achieved through a "produce-refine-evaluate" isometric image synthesis loop, followed by image-to-3D generation. To simulate continuous city evolution, Yo'City further introduces a user-interactive, relationship-guided expansion mechanism, which performs scene graph-based distance- and semantics-aware layout optimization,...
  </details>  
  Keywords: image-to-3d  
- **[LATTICE: Democratize High-Fidelity 3D Generation at Scale](https://arxiv.org/abs/2512.03052v1)**  
  Authors: Zeqiang Lai, Yunfei Zhao, Zibo Zhao, Haolin Liu, Qingxiang Lin, Jingwei Huang, Chunchao Guo, Xiangyu Yue  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2512.03052v1.pdf)  
  <details><summary>Abstract</summary>

  We present LATTICE, a new framework for high-fidelity 3D asset generation that bridges the quality and scalability gap between 3D and 2D generative models. While 2D image synthesis benefits from fixed spatial grids and well-established transformer architectures, 3D generation remains fundamentally more challenging due to the need to predict both spatial structure and detailed geometric surfaces from scratch. These challenges are exacerbated by the computational complexity of existing 3D representations and the lack of structured and scalable 3D asset encoding schemes. To address this, we propose VoxSet, a semi-structured representation that compresses 3D assets into a compact set of latent vectors anchored to a coarse voxel grid, enabling efficient and position-aware generation. VoxSet retains the simplicity and compression advantages of prior VecSet methods while introducing explicit structure into the latent space, allowing positional embeddings to guide generation and enabling strong token-level test-time scaling. Built upon this representation, LATTICE adopts a...
  </details>  
  Keywords: 3d asset generation  
- **[Single Image to High-Quality 3D Object via Latent Features](https://arxiv.org/abs/2511.19512v1)**  
  Authors: Huanning Dong, Yinuo Huang, Fan Li, Ping Kuang  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2511.19512v1.pdf)  
  <details><summary>Abstract</summary>

  3D assets are essential in the digital age. While automatic 3D generation, such as image-to-3d, has made significant strides in recent years, it often struggles to achieve fast, detailed, and high-fidelity generation simultaneously. In this work, we introduce LatentDreamer, a novel framework for generating 3D objects from single images. The key to our approach is a pre-trained variational autoencoder that maps 3D geometries to latent features, which greatly reducing the difficulty of 3D generation. Starting from latent features, the pipeline of LatentDreamer generates coarse geometries, refined geometries, and realistic textures sequentially. The 3D objects generated by LatentDreamer exhibit high fidelity to the input images, and the entire generation process can be completed within a short time (typically in 70 seconds). Extensive experiments show that with only a small amount of training, LatentDreamer demonstrates competitive performance compared to contemporary approachs.
  </details>  
  Keywords: image-to-3d  
- **[InfiniBench: Infinite Benchmarking for Visual Spatial Reasoning with Customizable Scene Complexity](https://arxiv.org/abs/2511.18200v2)**  
  Authors: Haoming Wang, Qiyao Xue, Wei Gao  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2511.18200v2.pdf)  
  <details><summary>Abstract</summary>

  Modern vision-language models (VLMs) are expected to have abilities of spatial reasoning with diverse scene complexities, but evaluating such abilities is difficult due to the lack of benchmarks that are not only diverse and scalable but also fully customizable. Existing benchmarks offer limited customizability over the scene complexity and are incapable of isolating and analyzing specific VLM failure modes under distinct spatial conditions. To address this gap, instead of individually presenting benchmarks for different scene complexities, in this paper we present InfiniBench, a fully automated, customizable and user-friendly benchmark generator that can synthesize a theoretically infinite variety of 3D scenes with parameterized control on scene complexity. InfiniBench uniquely translates scene descriptions in natural language into photo-realistic videos with complex and physically plausible 3D layouts. This is achieved through three key innovations: 1) a LLM-based agentic framework that iteratively refines procedural scene constraints from scene descriptions; 2) a flexible cluster-based layout...
  </details>  
- **[RAISECity: A Multimodal Agent Framework for Reality-Aligned 3D World Generation at City-Scale](https://arxiv.org/abs/2511.18005v1)**  
  Authors: Shengyuan Wang, Zhiheng Zheng, Yu Shang, Lixuan He, Yangcheng Yu, Fan Hangyu, Jie Feng, Qingmin Liao, Yong Li  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2511.18005v1.pdf)  
  <details><summary>Abstract</summary>

  City-scale 3D generation is of great importance for the development of embodied intelligence and world models. Existing methods, however, face significant challenges regarding quality, fidelity, and scalability in 3D world generation. Thus, we propose RAISECity, a \textbf{R}eality-\textbf{A}ligned \textbf{I}ntelligent \textbf{S}ynthesis \textbf{E}ngine that creates detailed, \textbf{C}ity-scale 3D worlds. We introduce an agentic framework that leverages diverse multimodal foundation tools to acquire real-world knowledge, maintain robust intermediate representations, and construct complex 3D scenes. This agentic design, featuring dynamic data processing, iterative self-reflection and refinement, and the invocation of advanced multimodal tools, minimizes cumulative errors and enhances overall performance. Extensive quantitative experiments and qualitative analyses validate the superior performance of RAISECity in real-world alignment, shape precision, texture fidelity, and aesthetics level, achieving over a 90% win-rate against existing baselines for overall perceptual quality. This combination of 3D quality, reality alignment, scalability, and seamless compatibility with computer graphics pipelines makes RAISECity a promising foundation for...
  </details>  
- **[ArticFlow: Generative Simulation of Articulated Mechanisms](https://arxiv.org/abs/2511.17883v1)**  
  Authors: Jiong Lin, Jinchen Ruan, Hod Lipson  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2511.17883v1.pdf)  
  <details><summary>Abstract</summary>

  Recent advances in generative models have produced strong results for static 3D shapes, whereas articulated 3D generation remains challenging due to action-dependent deformations and limited datasets. We introduce ArticFlow, a two-stage flow matching framework that learns a controllable velocity field from noise to target point sets under explicit action control. ArticFlow couples (i) a latent flow that transports noise to a shape-prior code and (ii) a point flow that transports points conditioned on the action and the shape prior, enabling a single model to represent diverse articulated categories and generalize across actions. On MuJoCo Menagerie, ArticFlow functions both as a generative model and as a neural simulator: it predicts action-conditioned kinematics from a compact prior and synthesizes novel morphologies via latent interpolation. Compared with object-specific simulators and an action-conditioned variant of static point-cloud generators, ArticFlow achieves higher kinematic accuracy and better shape quality. Results show that action-conditioned flow matching is...
  </details>  
- **[WorldGen: From Text to Traversable and Interactive 3D Worlds](https://arxiv.org/abs/2511.16825v1)**  
  Authors: Dilin Wang, Hyunyoung Jung, Tom Monnier, Kihyuk Sohn, Chuhang Zou, Xiaoyu Xiang, Yu-Ying Yeh, Di Liu, Zixuan Huang, Thu Nguyen-Phuoc, Yuchen Fan, Sergiu Oprea, Ziyan Wang, Roman Shapovalov, Nikolaos Sarafianos, Thibault Groueix, Antoine Toisoul, Prithviraj Dhar, Xiao Chu, Minghao Chen, Geon Yeong Park, Mahima Gupta, Yassir Azziz, Rakesh Ranjan, Andrea Vedaldi  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2511.16825v1.pdf)  
  <details><summary>Abstract</summary>

  We introduce WorldGen, a system that enables the automatic creation of large-scale, interactive 3D worlds directly from text prompts. Our approach transforms natural language descriptions into traversable, fully textured environments that can be immediately explored or edited within standard game engines. By combining LLM-driven scene layout reasoning, procedural generation, diffusion-based 3D generation, and object-aware scene decomposition, WorldGen bridges the gap between creative intent and functional virtual spaces, allowing creators to design coherent, navigable worlds without manual modeling or specialized 3D expertise. The system is fully modular and supports fine-grained control over layout, scale, and style, producing worlds that are geometrically consistent, visually rich, and efficient to render in real time. This work represents a step towards accessible, generative world-building at scale, advancing the frontier of 3D generative AI for applications in gaming, simulation, and immersive social environments.
  </details>  
- **[TRIM: Scalable 3D Gaussian Diffusion Inference with Temporal and Spatial Trimming](https://arxiv.org/abs/2511.16642v1)**  
  Authors: Zeyuan Yin, Xiaoming Liu  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2511.16642v1.pdf) | [![GitHub](https://img.shields.io/github/stars/zeyuanyin/TRIM?style=social)](https://github.com/zeyuanyin/TRIM)  
  <details><summary>Abstract</summary>

  Recent advances in 3D Gaussian diffusion models suffer from time-intensive denoising and post-denoising processing due to the massive number of Gaussian primitives, resulting in slow generation and limited scalability along sampling trajectories. To improve the efficiency of 3D diffusion models, we propose $\textbf{TRIM}$ ($\textbf{T}$rajectory $\textbf{R}$eduction and $\textbf{I}$nstance $\textbf{M}$ask denoising), a post-training approach that incorporates both temporal and spatial trimming strategies, to accelerate inference without compromising output quality while supporting the inference-time scaling for Gaussian diffusion models. Instead of scaling denoising trajectories in a costly end-to-end manner, we develop a lightweight selector model to evaluate latent Gaussian primitives derived from multiple sampled noises, enabling early trajectory reduction by selecting candidates with high-quality potential. Furthermore, we introduce instance mask denoising to prune learnable Gaussian primitives by filtering out redundant background regions, reducing inference computation at each denoising step. Extensive experiments and analysis demonstrate that TRIM significantly improves both the efficiency and quality...
  </details>  
- **[SAM 3D: 3Dfy Anything in Images](https://arxiv.org/abs/2511.16624v2)**  
  Authors: SAM 3D Team, Xingyu Chen, Fu-Jen Chu, Pierre Gleize, Kevin J Liang, Alexander Sax, Hao Tang, Weiyao Wang, Michelle Guo, Thibaut Hardin, Xiang Li, Aohan Lin, Jiawei Liu, Ziqi Ma, Anushka Sagar, Bowen Song, Xiaodong Wang, Jianing Yang, Bowen Zhang, Piotr Dollár, Georgia Gkioxari, Matt Feiszli, Jitendra Malik  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2511.16624v2.pdf)  
  <details><summary>Abstract</summary>

  We present SAM 3D, a generative model for visually grounded 3D object reconstruction, predicting geometry, texture, and layout from a single image. SAM 3D excels in natural images, where occlusion and scene clutter are common and visual recognition cues from context play a larger role. We achieve this with a human- and model-in-the-loop pipeline for annotating object shape, texture, and pose, providing visually grounded 3D reconstruction data at unprecedented scale. We learn from this data in a modern, multi-stage training framework that combines synthetic pretraining with real-world alignment, breaking the 3D "data barrier". We obtain significant gains over recent work, with at least a 5:1 win rate in human preference tests on real-world objects and scenes. We will release our code and model weights, an online demo, and a new challenging benchmark for in-the-wild 3D object reconstruction.
  </details>  
- **[From Prompts to Printable Models: Support-Effective 3D Generation via Offset Direct Preference Optimization](https://arxiv.org/abs/2511.16434v2)**  
  Authors: Chenming Wu, Xiaofan Li, Chengkai Dai  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2511.16434v2.pdf)  
  <details><summary>Abstract</summary>

  Current text-to-3D models prioritize visual fidelity but often neglect physical fabricability, resulting in geometries requiring excessive support structures. This paper introduces SEG (\textit{\underline{S}upport-\underline{E}ffective \underline{G}eneration}), a novel framework that integrates Direct Preference Optimization with an Offset (ODPO) into the 3D generation pipeline to directly optimize models for minimal support material usage. By incorporating support structure simulation into the training process, SEG encourages the generation of geometries that inherently require fewer supports, thus reducing material waste and production time. We demonstrate SEG's effectiveness through extensive experiments on two benchmark datasets, Thingi10k-Val and GPT-3DP-Val, showing that SEG significantly outperforms baseline models such as TRELLIS, DPO, and DRO in terms of support volume reduction and printability. Qualitative results further reveal that SEG maintains high fidelity to input prompts while minimizing the need for support structures. Our findings highlight the potential of SEG to transform 3D printing by directly optimizing models during the generative process,...
  </details>  
  Keywords: text-to-3d  
- **[Optimizing 3D Gaussian Splattering for Mobile GPUs](https://arxiv.org/abs/2511.16298v1)**  
  Authors: Md Musfiqur Rahman Sanim, Zhihao Shu, Bahram Afsharmanesh, AmirAli Mirian, Jiexiong Guan, Wei Niu, Bin Ren, Gagan Agrawal  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2511.16298v1.pdf)  
  <details><summary>Abstract</summary>

  Image-based 3D scene reconstruction, which transforms multi-view images into a structured 3D representation of the surrounding environment, is a common task across many modern applications. 3D Gaussian Splatting (3DGS) is a new paradigm to address this problem and offers considerable efficiency as compared to the previous methods. Motivated by this, and considering various benefits of mobile device deployment (data privacy, operating without internet connectivity, and potentially faster responses), this paper develops Texture3dgs, an optimized mapping of 3DGS for a mobile GPU. A critical challenge in this area turns out to be optimizing for the two-dimensional (2D) texture cache, which needs to be exploited for faster executions on mobile GPUs. As a sorting method dominates the computations in 3DGS on mobile platforms, the core of Texture3dgs is a novel sorting algorithm where the processing, data movement, and placement are highly optimized for 2D memory. The properties of this algorithm are analyzed...
  </details>  
  Keywords: 3d scene reconstruction  
- **[GeoSceneGraph: Geometric Scene Graph Diffusion Model for Text-guided 3D Indoor Scene Synthesis](https://arxiv.org/abs/2511.14884v1)**  
  Authors: Antonio Ruiz, Tao Wu, Andrew Melnik, Qing Cheng, Xuqin Wang, Lu Liu, Yongliang Wang, Yanfeng Zhang, Helge Ritter  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2511.14884v1.pdf)  
  <details><summary>Abstract</summary>

  Methods that synthesize indoor 3D scenes from text prompts have wide-ranging applications in film production, interior design, video games, virtual reality, and synthetic data generation for training embodied agents. Existing approaches typically either train generative models from scratch or leverage vision-language models (VLMs). While VLMs achieve strong performance, particularly for complex or open-ended prompts, smaller task-specific models remain necessary for deployment on resource-constrained devices such as extended reality (XR) glasses or mobile phones. However, many generative approaches that train from scratch overlook the inherent graph structure of indoor scenes, which can limit scene coherence and realism. Conversely, methods that incorporate scene graphs either demand a user-provided semantic graph, which is generally inconvenient and restrictive, or rely on ground-truth relationship annotations, limiting their capacity to capture more varied object interactions. To address these challenges, we introduce GeoSceneGraph, a method that synthesizes 3D scenes from text prompts by leveraging the graph structure...
  </details>  
- **[Let Language Constrain Geometry: Vision-Language Models as Semantic and Spatial Critics for 3D Generation](https://arxiv.org/abs/2511.14271v1)**  
  Authors: Weimin Bai, Yubo Li, Weijian Luo, Zeqiang Lai, Yequan Wang, Wenzheng Chen, He Sun  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2511.14271v1.pdf)  
  <details><summary>Abstract</summary>

  Text-to-3D generation has advanced rapidly, yet state-of-the-art models, encompassing both optimization-based and feed-forward architectures, still face two fundamental limitations. First, they struggle with coarse semantic alignment, often failing to capture fine-grained prompt details. Second, they lack robust 3D spatial understanding, leading to geometric inconsistencies and catastrophic failures in part assembly and spatial relationships. To address these challenges, we propose VLM3D, a general framework that repurposes large vision-language models (VLMs) as powerful, differentiable semantic and spatial critics. Our core contribution is a dual-query critic signal derived from the VLM's Yes or No log-odds, which assesses both semantic fidelity and geometric coherence. We demonstrate the generality of this guidance signal across two distinct paradigms: (1) As a reward objective for optimization-based pipelines, VLM3D significantly outperforms existing methods on standard benchmarks. (2) As a test-time guidance module for feed-forward pipelines, it actively steers the iterative sampling process of SOTA native 3D models to...
  </details>  
  Keywords: text-to-3d  
- **[PhysX-Anything: Simulation-Ready Physical 3D Assets from Single Image](https://arxiv.org/abs/2511.13648v1)**  
  Authors: Ziang Cao, Fangzhou Hong, Zhaoxi Chen, Liang Pan, Ziwei Liu  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2511.13648v1.pdf)  
  <details><summary>Abstract</summary>

  3D modeling is shifting from static visual representations toward physical, articulated assets that can be directly used in simulation and interaction. However, most existing 3D generation methods overlook key physical and articulation properties, thereby limiting their utility in embodied AI. To bridge this gap, we introduce PhysX-Anything, the first simulation-ready physical 3D generative framework that, given a single in-the-wild image, produces high-quality sim-ready 3D assets with explicit geometry, articulation, and physical attributes. Specifically, we propose the first VLM-based physical 3D generative model, along with a new 3D representation that efficiently tokenizes geometry. It reduces the number of tokens by 193x, enabling explicit geometry learning within standard VLM token budgets without introducing any special tokens during fine-tuning and significantly improving generative quality. In addition, to overcome the limited diversity of existing physical 3D datasets, we construct a new dataset, PhysX-Mobility, which expands the object categories in prior physical 3D datasets by...
  </details>  
- **[3DAlign-DAER: Dynamic Attention Policy and Efficient Retrieval Strategy for Fine-grained 3D-Text Alignment at Scale](https://arxiv.org/abs/2511.13211v2)**  
  Authors: Yijia Fan, Jusheng Zhang, Kaitong Cai, Jing Yang, Jian Wang, Keze Wang  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2511.13211v2.pdf) | [![GitHub](https://img.shields.io/github/stars/waltstephen/Cost-Effective-Communication?style=social)](https://github.com/waltstephen/Cost-Effective-Communication)  
  <details><summary>Abstract</summary>

  Despite recent advancements in 3D-text cross-modal alignment, existing state-of-the-art methods still struggle to align fine-grained textual semantics with detailed geometric structures, and their alignment performance degrades significantly when scaling to large-scale 3D databases. To overcome this limitation, we introduce 3DAlign-DAER, a unified framework designed to align text and 3D geometry via the proposed dynamic attention policy and the efficient retrieval strategy, capturing subtle correspondences for diverse cross-modal retrieval and classification tasks. Specifically, during the training, our proposed dynamic attention policy (DAP) employs the Hierarchical Attention Fusion (HAF) module to represent the alignment as learnable fine-grained token-to-point attentions. To optimize these attentions across different tasks and geometric hierarchies, our DAP further exploits the Monte Carlo tree search to dynamically calibrate HAF attention weights via a hybrid reward signal and further enhances the alignment between textual descriptions and local 3D geometry. During the inference, our 3DAlign-DAER introduces an Efficient Retrieval Strategy (ERS)...
  </details>  
- **[Visible Structure Retrieval for Lightweight Image-Based Relocalisation](https://arxiv.org/abs/2511.12503v1)**  
  Authors: Fereidoon Zangeneh, Leonard Bruns, Amit Dekel, Alessandro Pieropan, Patric Jensfelt  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2511.12503v1.pdf)  
  <details><summary>Abstract</summary>

  Accurate camera pose estimation from an image observation in a previously mapped environment is commonly done through structure-based methods: by finding correspondences between 2D keypoints on the image and 3D structure points in the map. In order to make this correspondence search tractable in large scenes, existing pipelines either rely on search heuristics, or perform image retrieval to reduce the search space by comparing the current image to a database of past observations. However, these approaches result in elaborate pipelines or storage requirements that grow with the number of past observations. In this work, we propose a new paradigm for making structure-based relocalisation tractable. Instead of relying on image retrieval or search heuristics, we learn a direct mapping from image observations to the visible scene structure in a compact neural network. Given a query image, a forward pass through our novel visible structure retrieval network allows obtaining the subset of...
  </details>  
- **[DenseAnnotate: Enabling Scalable Dense Caption Collection for Images and 3D Scenes via Spoken Descriptions](https://arxiv.org/abs/2511.12452v1)**  
  Authors: Xiaoyu Lin, Aniket Ghorpade, Hansheng Zhu, Justin Qiu, Dea Rrozhani, Monica Lama, Mick Yang, Zixuan Bian, Ruohan Ren, Alan B. Hong, Jiatao Gu, Chris Callison-Burch  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2511.12452v1.pdf)  
  <details><summary>Abstract</summary>

  With the rapid adoption of multimodal large language models (MLLMs) across diverse applications, there is a pressing need for task-centered, high-quality training data. A key limitation of current training datasets is their reliance on sparse annotations mined from the Internet or entered via manual typing that capture only a fraction of an image's visual content. Dense annotations are more valuable but remain scarce. Traditional text-based annotation pipelines are poorly suited for creating dense annotations: typing limits expressiveness, slows annotation speed, and underrepresents nuanced visual features, especially in specialized areas such as multicultural imagery and 3D asset annotation. In this paper, we present DenseAnnotate, an audio-driven online annotation platform that enables efficient creation of dense, fine-grained annotations for images and 3D assets. Annotators narrate observations aloud while synchronously linking spoken phrases to image regions or 3D scene parts. Our platform incorporates speech-to-text transcription and region-of-attention marking. To demonstrate the effectiveness of...
  </details>  
- **[LSS3D: Learnable Spatial Shifting for Consistent and High-Quality 3D Generation from Single-Image](https://arxiv.org/abs/2511.12202v1)**  
  Authors: Zhuojiang Cai, Yiheng Zhang, Meitong Guo, Mingdao Wang, Yuwang Wang  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2511.12202v1.pdf)  
  <details><summary>Abstract</summary>

  Recently, multi-view diffusion-based 3D generation methods have gained significant attention. However, these methods often suffer from shape and texture misalignment across generated multi-view images, leading to low-quality 3D generation results, such as incomplete geometric details and textural ghosting. Some methods are mainly optimized for the frontal perspective and exhibit poor robustness to oblique perspective inputs. In this paper, to tackle the above challenges, we propose a high-quality image-to-3D approach, named LSS3D, with learnable spatial shifting to explicitly and effectively handle the multiview inconsistencies and non-frontal input view. Specifically, we assign learnable spatial shifting parameters to each view, and adjust each view towards a spatially consistent target, guided by the reconstructed mesh, resulting in high-quality 3D generation with more complete geometric details and clean textures. Besides, we include the input view as an extra constraint for the optimization, further enhancing robustness to non-frontal input angles, especially for elevated viewpoint inputs. We...
  </details>  
  Keywords: image-to-3d  
- **[Rethinking Multimodal Point Cloud Completion: A Completion-by-Correction Perspective](https://arxiv.org/abs/2511.12170v2)**  
  Authors: Wang Luo, Di Wu, Hengyuan Na, Yinlin Zhu, Miao Hu, Guocong Quan  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2511.12170v2.pdf)  
  <details><summary>Abstract</summary>

  Point cloud completion aims to reconstruct complete 3D shapes from partial observations, which is a challenging problem due to severe occlusions and missing geometry. Despite recent advances in multimodal techniques that leverage complementary RGB images to compensate for missing geometry, most methods still follow a Completion-by-Inpainting paradigm, synthesizing missing structures from fused latent features. We empirically show that this paradigm often results in structural inconsistencies and topological artifacts due to limited geometric and semantic constraints. To address this, we rethink the task and propose a more robust paradigm, termed Completion-by-Correction, which begins with a topologically complete shape prior generated by a pretrained image-to-3D model and performs feature-space correction to align it with the partial observation. This paradigm shifts completion from unconstrained synthesis to guided refinement, enabling structurally consistent and observation-aligned reconstruction. Building upon this paradigm, we introduce PGNet, a multi-stage framework that conducts dual-feature encoding to ground the generative prior,...
  </details>  
  Keywords: image-to-3d  
- **[Large Language Models and 3D Vision for Intelligent Robotic Perception and Autonomy](https://arxiv.org/abs/2511.11777v2)**  
  Authors: Vinit Mehta, Charu Sharma, Karthick Thiyagarajan  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2511.11777v2.pdf)  
  <details><summary>Abstract</summary>

  With the rapid advancement of artificial intelligence and robotics, the integration of Large Language Models (LLMs) with 3D vision is emerging as a transformative approach to enhancing robotic sensing technologies. This convergence enables machines to perceive, reason and interact with complex environments through natural language and spatial understanding, bridging the gap between linguistic intelligence and spatial perception. This review provides a comprehensive analysis of state-of-the-art methodologies, applications and challenges at the intersection of LLMs and 3D vision, with a focus on next-generation robotic sensing technologies. We first introduce the foundational principles of LLMs and 3D data representations, followed by an in-depth examination of 3D sensing technologies critical for robotics. The review then explores key advancements in scene understanding, text-to-3D generation, object grounding and embodied agents, highlighting cutting-edge techniques such as zero-shot 3D segmentation, dynamic scene synthesis and language-guided manipulation. Furthermore, we discuss multimodal LLMs that integrate 3D data with touch,...
  </details>  
  Keywords: text-to-3d  
- **[Hyperbolic Hierarchical Alignment Reasoning Network for Text-3D Retrieval](https://arxiv.org/abs/2511.11045v1)**  
  Authors: Wenrui Li, Yidan Lu, Yeyu Chai, Rui Zhao, Hengyu Man, Xiaopeng Fan  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2511.11045v1.pdf) | [![GitHub](https://img.shields.io/github/stars/liwrui/H2ARN?style=social)](https://github.com/liwrui/H2ARN)  
  <details><summary>Abstract</summary>

  With the daily influx of 3D data on the internet, text-3D retrieval has gained increasing attention. However, current methods face two major challenges: Hierarchy Representation Collapse (HRC) and Redundancy-Induced Saliency Dilution (RISD). HRC compresses abstract-to-specific and whole-to-part hierarchies in Euclidean embeddings, while RISD averages noisy fragments, obscuring critical semantic cues and diminishing the model's ability to distinguish hard negatives. To address these challenges, we introduce the Hyperbolic Hierarchical Alignment Reasoning Network (H$^{2}$ARN) for text-3D retrieval. H$^{2}$ARN embeds both text and 3D data in a Lorentz-model hyperbolic space, where exponential volume growth inherently preserves hierarchical distances. A hierarchical ordering loss constructs a shrinking entailment cone around each text vector, ensuring that the matched 3D instance falls within the cone, while an instance-level contrastive loss jointly enforces separation from non-matching samples. To tackle RISD, we propose a contribution-aware hyperbolic aggregation module that leverages Lorentzian distance to assess the relevance of each local...
  </details>  
  Keywords: text-to-3d  
- **[Optimizing Input of Denoising Score Matching is Biased Towards Higher Score Norm](https://arxiv.org/abs/2511.11727v1)**  
  Authors: Tongda Xu  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2511.11727v1.pdf)  
  <details><summary>Abstract</summary>

  Many recent works utilize denoising score matching to optimize the conditional input of diffusion models. In this workshop paper, we demonstrate that such optimization breaks the equivalence between denoising score matching and exact score matching. Furthermore, we show that this bias leads to higher score norm. Additionally, we observe a similar bias when optimizing the data distribution using a pre-trained diffusion model. Finally, we discuss the wide range of works across different domains that are affected by this bias, including MAR for auto-regressive generation, PerCo for image compression, and DreamFusion for text to 3D generation.
  </details>  
- **[Target-Balanced Score Distillation](https://arxiv.org/abs/2511.11710v1)**  
  Authors: Zhou Xu, Qi Wang, Yuxiao Yang, Luyuan Zhang, Zhang Liang, Yang Li  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2511.11710v1.pdf)  
  <details><summary>Abstract</summary>

  Score Distillation Sampling (SDS) enables 3D asset generation by distilling priors from pretrained 2D text-to-image diffusion models, but vanilla SDS suffers from over-saturation and over-smoothing. To mitigate this issue, recent variants have incorporated negative prompts. However, these methods face a critical trade-off: limited texture optimization, or significant texture gains with shape distortion. In this work, we first conduct a systematic analysis and reveal that this trade-off is fundamentally governed by the utilization of the negative prompts, where Target Negative Prompts (TNP) that embed target information in the negative prompts dramatically enhancing texture realism and fidelity but inducing shape distortions. Informed by this key insight, we introduce the Target-Balanced Score Distillation (TBSD). It formulates generation as a multi-objective optimization problem and introduces an adaptive strategy that effectively resolves the aforementioned trade-off. Extensive experiments demonstrate that TBSD significantly outperforms existing state-of-the-art methods, yielding 3D assets with high-fidelity textures and geometrically accurate shape.
  </details>  
  Keywords: 3d asset generation  
- **[DensiCrafter: Physically-Constrained Generation and Fabrication of Self-Supporting Hollow Structures](https://arxiv.org/abs/2511.09298v2)**  
  Authors: Shengqi Dang, Fu Chai, Jiaxin Li, Chao Yuan, Wei Ye, Nan Cao  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2511.09298v2.pdf)  
  <details><summary>Abstract</summary>

  The rise of 3D generative models has enabled automatic 3D geometry and texture synthesis from multimodal inputs (e.g., text or images). However, these methods often ignore physical constraints and manufacturability considerations. In this work, we address the challenge of producing 3D designs that are both lightweight and self-supporting. We present DensiCrafter, a framework for generating lightweight, self-supporting 3D hollow structures by optimizing the density field. Starting from coarse voxel grids produced by Trellis, we interpret these as continuous density fields to optimize and introduce three differentiable, physically constrained, and simulation-free loss terms. Additionally, a mass regularization penalizes unnecessary material, while a restricted optimization domain preserves the outer surface. Our method seamlessly integrates with pretrained Trellis-based models (e.g., Trellis, DSO) without any architectural changes. In extensive evaluations, we achieve up to 43% reduction in material mass on the text-to-3D task. Compared to state-of-the-art baselines, our method could improve the stability and...
  </details>  
  Keywords: text-to-3d, texture synthesis  
- **[AnchorDS: Anchoring Dynamic Sources for Semantically Consistent Text-to-3D Generation](https://arxiv.org/abs/2511.11692v1)**  
  Authors: Jiayin Zhu, Linlin Yang, Yicong Li, Angela Yao  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2511.11692v1.pdf)  
  <details><summary>Abstract</summary>

  Optimization-based text-to-3D methods distill guidance from 2D generative models via Score Distillation Sampling (SDS), but implicitly treat this guidance as static. This work shows that ignoring source dynamics yields inconsistent trajectories that suppress or merge semantic cues, leading to "semantic over-smoothing" artifacts. As such, we reformulate text-to-3D optimization as mapping a dynamically evolving source distribution to a fixed target distribution. We cast the problem into a dual-conditioned latent space, conditioned on both the text prompt and the intermediately rendered image. Given this joint setup, we observe that the image condition naturally anchors the current source distribution. Building on this insight, we introduce AnchorDS, an improved score distillation mechanism that provides state-anchored guidance with image conditions and stabilizes generation. We further penalize erroneous source estimates and design a lightweight filter strategy and fine-tuning strategy that refines the anchor with negligible overhead. AnchorDS produces finer-grained detail, more natural colours, and stronger semantic...
  </details>  
  Keywords: text-to-3d  
- **[Fuse3D: Generating 3D Assets Controlled by Multi-Image Fusion](https://arxiv.org/abs/2602.17040v1)**  
  Authors: Xuancheng Jin, Rengan Xie, Wenting Zheng, Rui Wang, Hujun Bao, Yuchi Huo  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2602.17040v1.pdf) | [![Project](https://img.shields.io/badge/-Project-blue)](https://jinnmnm.github.io/Fuse3d.github.io)  
  <details><summary>Abstract</summary>

  Recently, generating 3D assets with the control of condition images has achieved impressive quality. However, existing 3D generation methods are limited to handling a single control objective and lack the ability to utilize multiple images to independently control different regions of a 3D asset, which hinders their flexibility in applications. We propose Fuse3D, a novel method that enables generating 3D assets under the control of multiple images, allowing for the seamless fusion of multi-level regional controls from global views to intricate local details. First, we introduce a Multi-Condition Fusion Module to integrate the visual features from multiple image regions. Then, we propose a method to automatically align user-selected 2D image regions with their associated 3D regions based on semantic cues. Finally, to resolve control conflicts and enhance local control features from multi-condition images, we introduce a Local Attention Enhancement Strategy that flexibly balances region-specific feature fusion. Overall, we introduce the...
  </details>  
  Keywords: 3d asset generation  
- **[DT-NVS: Diffusion Transformers for Novel View Synthesis](https://arxiv.org/abs/2511.08823v1)**  
  Authors: Wonbong Jang, Jonathan Tremblay, Lourdes Agapito  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2511.08823v1.pdf)  
  <details><summary>Abstract</summary>

  Generating novel views of a natural scene, e.g., every-day scenes both indoors and outdoors, from a single view is an under-explored problem, even though it is an organic extension to the object-centric novel view synthesis. Existing diffusion-based approaches focus rather on small camera movements in real scenes or only consider unnatural object-centric scenes, limiting their potential applications in real-world settings. In this paper we move away from these constrained regimes and propose a 3D diffusion model trained with image-only losses on a large-scale dataset of real-world, multi-category, unaligned, and casually acquired videos of everyday scenes. We propose DT-NVS, a 3D-aware diffusion model for generalized novel view synthesis that exploits a transformer-based architecture backbone. We make significant contributions to transformer and self-attention architectures to translate images to 3d representations, and novel camera conditioning strategies to allow training on real-world unaligned datasets. In addition, we introduce a novel training paradigm swapping the...
  </details>  
  Keywords: novel view synthesis  
- **[Twist and Compute: The Cost of Pose in 3D Generative Diffusion](https://arxiv.org/abs/2511.08203v1)**  
  Authors: Kyle Fogarty, Jack Foster, Boqiao Zhang, Jing Yang, Cengiz Öztireli  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2511.08203v1.pdf)  
  <details><summary>Abstract</summary>

  Despite their impressive results, large-scale image-to-3D generative models remain opaque in their inductive biases. We identify a significant limitation in image-conditioned 3D generative models: a strong canonical view bias. Through controlled experiments using simple 2D rotations, we show that the state-of-the-art Hunyuan3D 2.0 model can struggle to generalize across viewpoints, with performance degrading under rotated inputs. We show that this failure can be mitigated by a lightweight CNN that detects and corrects input orientation, restoring model performance without modifying the generative backbone. Our findings raise an important open question: Is scale enough, or should we pursue modular, symmetry-aware designs?
  </details>  
  Keywords: image-to-3d  
- **[YoNoSplat: You Only Need One Model for Feedforward 3D Gaussian Splatting](https://arxiv.org/abs/2511.07321v1)**  
  Authors: Botao Ye, Boqi Chen, Haofei Xu, Daniel Barath, Marc Pollefeys  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2511.07321v1.pdf) | [![Project](https://img.shields.io/badge/-Project-blue)](https://botaoye.github.io/yonosplat)  
  <details><summary>Abstract</summary>

  Fast and flexible 3D scene reconstruction from unstructured image collections remains a significant challenge. We present YoNoSplat, a feedforward model that reconstructs high-quality 3D Gaussian Splatting representations from an arbitrary number of images. Our model is highly versatile, operating effectively with both posed and unposed, calibrated and uncalibrated inputs. YoNoSplat predicts local Gaussians and camera poses for each view, which are aggregated into a global representation using either predicted or provided poses. To overcome the inherent difficulty of jointly learning 3D Gaussians and camera parameters, we introduce a novel mixing training strategy. This approach mitigates the entanglement between the two tasks by initially using ground-truth poses to aggregate local Gaussians and gradually transitioning to a mix of predicted and ground-truth poses, which prevents both training instability and exposure bias. We further resolve the scale ambiguity problem by a novel pairwise camera-distance normalization scheme and by embedding camera intrinsics into the...
  </details>  
  Keywords: 3d scene reconstruction  
- **[4DSTR: Advancing Generative 4D Gaussians with Spatial-Temporal Rectification for High-Quality and Consistent 4D Generation](https://arxiv.org/abs/2511.07241v1)**  
  Authors: Mengmeng Liu, Jiuming Liu, Yunpeng Zhang, Jiangtao Li, Michael Ying Yang, Francesco Nex, Hao Cheng  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2511.07241v1.pdf)  
  <details><summary>Abstract</summary>

  Remarkable advances in recent 2D image and 3D shape generation have induced a significant focus on dynamic 4D content generation. However, previous 4D generation methods commonly struggle to maintain spatial-temporal consistency and adapt poorly to rapid temporal variations, due to the lack of effective spatial-temporal modeling. To address these problems, we propose a novel 4D generation network called 4DSTR, which modulates generative 4D Gaussian Splatting with spatial-temporal rectification. Specifically, temporal correlation across generated 4D sequences is designed to rectify deformable scales and rotations and guarantee temporal consistency. Furthermore, an adaptive spatial densification and pruning strategy is proposed to address significant temporal variations by dynamically adding or deleting Gaussian points with the awareness of their pre-frame movements. Extensive experiments demonstrate that our 4DSTR achieves state-of-the-art performance in video-to-4D generation, excelling in reconstruction quality, spatial-temporal consistency, and adaptation to rapid temporal movements.
  </details>  
- **[ProcGen3D: Learning Neural Procedural Graph Representations for Image-to-3D Reconstruction](https://arxiv.org/abs/2511.07142v1)**  
  Authors: Xinyi Zhang, Daoyi Gao, Naiqi Li, Angela Dai  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2511.07142v1.pdf)  
  <details><summary>Abstract</summary>

  We introduce ProcGen3D, a new approach for 3D content creation by generating procedural graph abstractions of 3D objects, which can then be decoded into rich, complex 3D assets. Inspired by the prevalent use of procedural generators in production 3D applications, we propose a sequentialized, graph-based procedural graph representation for 3D assets. We use this to learn to approximate the landscape of a procedural generator for image-based 3D reconstruction. We employ edge-based tokenization to encode the procedural graphs, and train a transformer prior to predict the next token conditioned on an input RGB image. Crucially, to enable better alignment of our generated outputs to an input image, we incorporate Monte Carlo Tree Search (MCTS) guided sampling into our generation process, steering output procedural graphs towards more image-faithful reconstructions. Our approach is applicable across a variety of objects that can be synthesized with procedural generators. Extensive experiments on cacti, trees, and bridges...
  </details>  
  Keywords: image-to-3d  
- **[RaLD: Generating High-Resolution 3D Radar Point Clouds with Latent Diffusion](https://arxiv.org/abs/2511.07067v1)**  
  Authors: Ruijie Zhang, Bixin Zeng, Shengpeng Wang, Fuhui Zhou, Wei Wang  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2511.07067v1.pdf)  
  <details><summary>Abstract</summary>

  Millimeter-wave radar offers a promising sensing modality for autonomous systems thanks to its robustness in adverse conditions and low cost. However, its utility is significantly limited by the sparsity and low resolution of radar point clouds, which poses challenges for tasks requiring dense and accurate 3D perception. Despite that recent efforts have shown great potential by exploring generative approaches to address this issue, they often rely on dense voxel representations that are inefficient and struggle to preserve structural detail. To fill this gap, we make the key observation that latent diffusion models (LDMs), though successful in other modalities, have not been effectively leveraged for radar-based 3D generation due to a lack of compatible representations and conditioning strategies. We introduce RaLD, a framework that bridges this gap by integrating scene-level frustum-based LiDAR autoencoding, order-invariant latent representations, and direct radar spectrum conditioning. These insights lead to a more compact and expressive generation...
  </details>  
- **[GFix: Perceptually Enhanced Gaussian Splatting Video Compression](https://arxiv.org/abs/2511.06953v1)**  
  Authors: Siyue Teng, Ge Gao, Duolikun Danier, Yuxuan Jiang, Fan Zhang, Thomas Davis, Zoe Liu, David Bull  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2511.06953v1.pdf)  
  <details><summary>Abstract</summary>

  3D Gaussian Splatting (3DGS) enhances 3D scene reconstruction through explicit representation and fast rendering, demonstrating potential benefits for various low-level vision tasks, including video compression. However, existing 3DGS-based video codecs generally exhibit more noticeable visual artifacts and relatively low compression ratios. In this paper, we specifically target the perceptual enhancement of 3DGS-based video compression, based on the assumption that artifacts from 3DGS rendering and quantization resemble noisy latents sampled during diffusion training. Building on this premise, we propose a content-adaptive framework, GFix, comprising a streamlined, single-step diffusion model that serves as an off-the-shelf neural enhancer. Moreover, to increase compression efficiency, We propose a modulated LoRA scheme that freezes the low-rank decompositions and modulates the intermediate hidden states, thereby achieving efficient adaptation of the diffusion backbone with highly compressible updates. Experimental results show that GFix delivers strong perceptual quality enhancement, outperforming GSVC with up to 72.1% BD-rate savings in LPIPS and...
  </details>  
  Keywords: 3d scene reconstruction  
- **[Rethinking Rainy 3D Scene Reconstruction via Perspective Transforming and Brightness Tuning](https://arxiv.org/abs/2511.06734v1)**  
  Authors: Qianfeng Yang, Xiang Chen, Pengpeng Li, Qiyuan Guan, Guiyue Jin, Jiyu Jin  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2511.06734v1.pdf)  
  <details><summary>Abstract</summary>

  Rain degrades the visual quality of multi-view images, which are essential for 3D scene reconstruction, resulting in inaccurate and incomplete reconstruction results. Existing datasets often overlook two critical characteristics of real rainy 3D scenes: the viewpoint-dependent variation in the appearance of rain streaks caused by their projection onto 2D images, and the reduction in ambient brightness resulting from cloud coverage during rainfall. To improve data realism, we construct a new dataset named OmniRain3D that incorporates perspective heterogeneity and brightness dynamicity, enabling more faithful simulation of rain degradation in 3D scenes. Based on this dataset, we propose an end-to-end reconstruction framework named REVR-GSNet (Rain Elimination and Visibility Recovery for 3D Gaussian Splatting). Specifically, REVR-GSNet integrates recursive brightness enhancement, Gaussian primitive optimization, and GS-guided rain elimination into a unified architecture through joint alternating optimization, achieving high-fidelity reconstruction of clean 3D scenes from rain-degraded inputs. Extensive experiments show the effectiveness of our dataset...
  </details>  
  Keywords: 3d scene reconstruction  
- **[AvatarTex: High-Fidelity Facial Texture Reconstruction from Single-Image Stylized Avatars](https://arxiv.org/abs/2511.06721v1)**  
  Authors: Yuda Qiu, Zitong Xiao, Yiwei Zuo, Zisheng Ye, Weikai Chen, Xiaoguang Han  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2511.06721v1.pdf)  
  <details><summary>Abstract</summary>

  We present AvatarTex, a high-fidelity facial texture reconstruction framework capable of generating both stylized and photorealistic textures from a single image. Existing methods struggle with stylized avatars due to the lack of diverse multi-style datasets and challenges in maintaining geometric consistency in non-standard textures. To address these limitations, AvatarTex introduces a novel three-stage diffusion-to-GAN pipeline. Our key insight is that while diffusion models excel at generating diversified textures, they lack explicit UV constraints, whereas GANs provide a well-structured latent space that ensures style and topology consistency. By integrating these strengths, AvatarTex achieves high-quality topology-aligned texture synthesis with both artistic and geometric coherence. Specifically, our three-stage pipeline first completes missing texture regions via diffusion-based inpainting, refines style and structure consistency using GAN-based latent optimization, and enhances fine details through diffusion-based repainting. To address the need for a stylized texture dataset, we introduce TexHub, a high-resolution collection of 20,000 multi-style UV textures...
  </details>  
  Keywords: texture synthesis  
- **[VLAD-Grasp: Zero-shot Grasp Detection via Vision-Language Models](https://arxiv.org/abs/2511.05791v2)**  
  Authors: Manav Kulshrestha, S. Talha Bukhari, Damon Conover, Aniket Bera  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2511.05791v2.pdf)  
  <details><summary>Abstract</summary>

  Robotic grasping is a fundamental capability for enabling autonomous manipulation, with usually infinite solutions. State-of-the-art approaches for grasping rely on learning from large-scale datasets comprising expert annotations of feasible grasps. Curating such datasets is challenging, and hence, learning-based methods are limited by the solution coverage of the dataset, and require retraining to handle novel objects. Towards this, we present VLAD-Grasp, a Vision-Language model Assisted zero-shot approach for Detecting Grasps. Our method (1) prompts a large vision-language model to generate a goal image where a virtual cylindrical proxy intersects the object's geometry, explicitly encoding an antipodal grasp axis in image space, then (2) predicts depth and segmentation to lift this generated image into 3D, and (3) aligns generated and observed object point clouds via principal components and correspondence-free optimization to recover an executable grasp pose. Unlike prior work, our approach is training-free and does not require curated grasp datasets, while achieving...
  </details>  
- **[Walking the Schrödinger Bridge: A Direct Trajectory for Text-to-3D Generation](https://arxiv.org/abs/2511.05609v1)**  
  Authors: Ziying Li, Xuequan Lu, Xinkui Zhao, Guanjie Cheng, Shuiguang Deng, Jianwei Yin  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2511.05609v1.pdf)  
  <details><summary>Abstract</summary>

  Recent advancements in optimization-based text-to-3D generation heavily rely on distilling knowledge from pre-trained text-to-image diffusion models using techniques like Score Distillation Sampling (SDS), which often introduce artifacts such as over-saturation and over-smoothing into the generated 3D assets. In this paper, we address this essential problem by formulating the generation process as learning an optimal, direct transport trajectory between the distribution of the current rendering and the desired target distribution, thereby enabling high-quality generation with smaller Classifier-free Guidance (CFG) values. At first, we theoretically establish SDS as a simplified instance of the Schrödinger Bridge framework. We prove that SDS employs the reverse process of an Schrödinger Bridge, which, under specific conditions (e.g., a Gaussian noise as one end), collapses to SDS's score function of the pre-trained diffusion model. Based upon this, we introduce Trajectory-Centric Distillation (TraCe), a novel text-to-3D generation framework, which reformulates the mathematically trackable framework of Schrödinger Bridge to...
  </details>  
  Keywords: text-to-3d  
- **[Can Foundation Models Revolutionize Mobile AR Sparse Sensing?](https://arxiv.org/abs/2511.02215v1)**  
  Authors: Yiqin Zhao, Tian Guo  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2511.02215v1.pdf)  
  <details><summary>Abstract</summary>

  Mobile sensing systems have long faced a fundamental trade-off between sensing quality and efficiency due to constraints in computation, power, and other limitations. Sparse sensing, which aims to acquire and process only a subset of sensor data, has been a key strategy for maintaining performance under such constraints. However, existing sparse sensing methods often suffer from reduced accuracy, as missing information across space and time introduces uncertainty into many sensing systems. In this work, we investigate whether foundation models can change the landscape of mobile sparse sensing. Using real-world mobile AR data, our evaluations demonstrate that foundation models offer significant improvements in geometry-aware image warping, a central technique for enabling accurate reuse of cross-frame information. Furthermore, our study demonstrates the scalability of foundation model-based sparse sensing and shows its leading performance in 3D scene reconstruction. Collectively, our study reveals critical aspects of the promises and the open challenges of integrating...
  </details>  
  Keywords: 3d scene reconstruction  
- **[Text to Robotic Assembly of Multi Component Objects using 3D Generative AI and Vision Language Models](https://arxiv.org/abs/2511.02162v4)**  
  Authors: Alexander Htet Kyaw, Richa Gupta, Dhruv Shah, Anoop Sinha, Kory Mathewson, Stefanie Pender, Sachin Chitta, Yotto Koga, Faez Ahmed, Lawrence Sass, Randall Davis  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2511.02162v4.pdf)  
  <details><summary>Abstract</summary>

  Advances in 3D generative AI have enabled the creation of physical objects from text prompts, but challenges remain in creating objects involving multiple component types. We present a pipeline that integrates 3D generative AI with vision-language models (VLMs) to enable the robotic assembly of multi-component objects from natural language. Our method leverages VLMs for zero-shot, multi-modal reasoning about geometry and functionality to decompose AI-generated meshes into multi-component 3D models using predefined structural and panel components. We demonstrate that a VLM is capable of determining which mesh regions need panel components in addition to structural components, based on the object's geometry and functionality. Evaluation across test objects shows that users preferred the VLM-generated assignments 90.6% of the time, compared to 59.4% for rule-based and 2.5% for random assignment. Lastly, the system allows users to refine component assignments through conversational feedback, enabling greater human control and agency in making physical objects with...
  </details>  
- **[Wonder3D++: Cross-domain Diffusion for High-fidelity 3D Generation from a Single Image](https://arxiv.org/abs/2511.01767v2)**  
  Authors: Yuxiao Yang, Xiao-Xiao Long, Zhiyang Dou, Cheng Lin, Yuan Liu, Qingsong Yan, Yuexin Ma, Haoqian Wang, Zhiqiang Wu, Wei Yin  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2511.01767v2.pdf) | [![GitHub](https://img.shields.io/github/stars/xxlong0/Wonder3D?style=social)](https://github.com/xxlong0/Wonder3D)  
  <details><summary>Abstract</summary>

  In this work, we introduce \textbf{Wonder3D++}, a novel method for efficiently generating high-fidelity textured meshes from single-view images. Recent methods based on Score Distillation Sampling (SDS) have shown the potential to recover 3D geometry from 2D diffusion priors, but they typically suffer from time-consuming per-shape optimization and inconsistent geometry. In contrast, certain works directly produce 3D information via fast network inferences, but their results are often of low quality and lack geometric details. To holistically improve the quality, consistency, and efficiency of single-view reconstruction tasks, we propose a cross-domain diffusion model that generates multi-view normal maps and the corresponding color images. To ensure the consistency of generation, we employ a multi-view cross-domain attention mechanism that facilitates information exchange across views and modalities. Lastly, we introduce a cascaded 3D mesh extraction algorithm that drives high-quality surfaces from the multi-view 2D representations in only about $3$ minute in a coarse-to-fine manner. Our...
  </details>  
  Keywords: single-view reconstruction  
- **[Semantic BIM enrichment for firefighting assets: Fire-ART dataset and panoramic image-based 3D reconstruction](https://arxiv.org/abs/2511.01399v1)**  
  Authors: Ya Wen, Yutong Qiao, Chi Chiu Lam, Ioannis Brilakis, Sanghoon Lee, Mun On Wong  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2511.01399v1.pdf)  
  <details><summary>Abstract</summary>

  Inventory management of firefighting assets is crucial for emergency preparedness, risk assessment, and on-site fire response. However, conventional methods are inefficient due to limited capabilities in automated asset recognition and reconstruction. To address the challenge, this research introduces the Fire-ART dataset and develops a panoramic image-based reconstruction approach for semantic enrichment of firefighting assets into BIM models. The Fire-ART dataset covers 15 fundamental assets, comprising 2,626 images and 6,627 instances, making it an extensive and publicly accessible dataset for asset recognition. In addition, the reconstruction approach integrates modified cube-map conversion and radius-based spherical camera projection to enhance recognition and localization accuracy. Through validations with two real-world case studies, the proposed approach achieves F1-scores of 73% and 88% and localization errors of 0.620 and 0.428 meters, respectively. The Fire-ART dataset and the reconstruction approach offer valuable resources and robust technical solutions to enhance the accurate digital management of fire safety equipment.
  </details>  
- **[MoSa: Motion Generation with Scalable Autoregressive Modeling](https://arxiv.org/abs/2511.01200v1)**  
  Authors: Mengyuan Liu, Sheng Yan, Yong Wang, Yingjie Li, Gui-Bin Bian, Hong Liu  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2511.01200v1.pdf) | [![Project](https://img.shields.io/badge/-Project-blue)](https://mosa-web.github.io/MoSa-web)  
  <details><summary>Abstract</summary>

  We introduce MoSa, a novel hierarchical motion generation framework for text-driven 3D human motion generation that enhances the Vector Quantization-guided Generative Transformers (VQ-GT) paradigm through a coarse-to-fine scalable generation process. In MoSa, we propose a Multi-scale Token Preservation Strategy (MTPS) integrated into a hierarchical residual vector quantization variational autoencoder (RQ-VAE). MTPS employs interpolation at each hierarchical quantization to effectively retain coarse-to-fine multi-scale tokens. With this, the generative transformer supports Scalable Autoregressive (SAR) modeling, which predicts scale tokens, unlike traditional methods that predict only one token at each step. Consequently, MoSa requires only 10 inference steps, matching the number of RQ-VAE quantization layers. To address potential reconstruction degradation from frequent interpolation, we propose CAQ-VAE, a lightweight yet expressive convolution-attention hybrid VQ-VAE. CAQ-VAE enhances residual block design and incorporates attention mechanisms to better capture global dependencies. Extensive experiments show that MoSa achieves state-of-the-art generation quality and efficiency, outperforming prior methods in both...
  </details>  
- **[Oitijjo-3D: Generative AI Framework for Rapid 3D Heritage Reconstruction from Street View Imagery](https://arxiv.org/abs/2511.00362v1)**  
  Authors: Momen Khandoker Ope, Akif Islam, Mohd Ruhul Ameen, Abu Saleh Musa Miah, Md Rashedul Islam, Jungpil Shin  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2511.00362v1.pdf)  
  <details><summary>Abstract</summary>

  Cultural heritage restoration in Bangladesh faces a dual challenge of limited resources and scarce technical expertise. Traditional 3D digitization methods, such as photogrammetry or LiDAR scanning, require expensive hardware, expert operators, and extensive on-site access, which are often infeasible in developing contexts. As a result, many of Bangladesh's architectural treasures, from the Paharpur Buddhist Monastery to Ahsan Manzil, remain vulnerable to decay and inaccessible in digital form. This paper introduces Oitijjo-3D, a cost-free generative AI framework that democratizes 3D cultural preservation. By using publicly available Google Street View imagery, Oitijjo-3D reconstructs faithful 3D models of heritage structures through a two-stage pipeline - multimodal visual reasoning with Gemini 2.5 Flash Image for structure-texture synthesis, and neural image-to-3D generation through Hexagen for geometry recovery. The system produces photorealistic, metrically coherent reconstructions in seconds, achieving significant speedups compared to conventional Structure-from-Motion pipelines, without requiring any specialized hardware or expert supervision. Experiments on landmarks...
  </details>  
  Keywords: texture synthesis, image-to-3d  

### May 2026
- **[Towards Interactive Video World Modeling: Frontiers, Challenges, Benchmarks, and Future Trends](https://arxiv.org/abs/2606.01164v1)**  
  Authors: Jiuming Liu, Chaojun Ni, Mengmeng Liu, Chensheng Peng, Fangjinhua Wang, Sitian Shen, Marc Pollefeys, Masayoshi Tomizuka, Ayush Tewari, Per Ola Kristensson  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2606.01164v1.pdf) | [![GitHub](https://img.shields.io/github/stars/liujiuming123/Awesome-Interactive-World-Model?style=social)](https://github.com/liujiuming123/Awesome-Interactive-World-Model)  
  <details><summary>Abstract</summary>

  With rapid development of large language models and diffusion-based content generation, world modeling has attracted increasing research attention, benefiting various downstream domains such as game engines, embodied AI, autonomous driving, etc. Through explicitly incorporating user actions into world state transition, recent literature empowers world modeling with interactivity in an action-conditioned video or 3D generation paradigm, further enhancing controllability over world evolutions and facilitating users to freely traverse, manipulate, navigate, and personalize the state evolution. In this paper, we aim to systematically review recent research trends, technical developments, evaluation benchmarks, and also propose future potential directions in interactive world modeling. Specifically, we first summarize recent efforts and trends in terms of application scenarios, world state evolution, and scene modality. Afterwards, we delve into three crucial technical challenges, including action-conditioned controllability, long-horizon interactions and memory, and action-following responsiveness for real-time interactivity. Furthermore, we also thoroughly compare existing benchmarks and metrics in four...
  </details>  
- **[3DCodeBench: Benchmarking Agentic Procedural 3D Modeling Via Code](https://arxiv.org/abs/2606.01057v1)**  
  Authors: Yipeng Gao, Lei Shu, Genzhi Ye, Xi Xiong, Ameesh Makadia, Meiqi Guo, Laurent Itti, Jindong Chen  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2606.01057v1.pdf)  
  <details><summary>Abstract</summary>

  Procedural 3D modeling through code is emerging as a versatile paradigm, offering deterministic, engine-ready, and precisely editable assets that neural 3D generators inherently lack. Authoring such procedural content, however, demands deep expertise in 3D software APIs, parametric design, and code-level geometric reasoning. In this paper, we propose 3DCodeBench, a systematic benchmark for evaluating vision-language model (VLM) agents for procedural 3D generation in 3D modeling software. Specifically, 3DCodeBench evaluates how effectively 12 advanced VLMs can serve as procedural 3D modelers by translating text and image references into procedural code for 3D modeling software. Recognizing that automated metrics may not fully capture the perceptual quality of 3D shapes, we build 3DCodeArena, a ranking platform based on pairwise human preferences over generated 3D outputs. From extensive evaluations and results, we observe that: (1) Failures mostly arise from API mismatches, while successful renders still suffer from disconnected or floating 3D geometric components. (2) Test-time...
  </details>  
  Keywords: 3d generator  
- **[Cross-Axis Feature Fusion with Joint-Wise Motion Difference Prediction for Text-Based 3D Human Motion Editing](https://arxiv.org/abs/2606.01014v1)**  
  Authors: Gyojin Han, Junmo Kim  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2606.01014v1.pdf)  
  <details><summary>Abstract</summary>

  We address text-based 3D human motion editing, where the goal is to preserve the style and structure of a source motion while applying edits described in natural language. The release of the MotionFix dataset has spurred active research into training-based diffusion models that directly generate an edited motion from a source motion and a text instruction. While previous works have focused primarily on learning when an edit should occur temporally, our goal is to create a model that understands not only this temporal aspect but also which specific joints are responsible for the change. Targeting this, we propose a novel architecture and a complementary auxiliary task to aid its training. Our architecture consists of two axis-anchored transformers, which extract distinct features along the joint and time dimensions respectively, and a cross-axis fusion block that integrates these representations. We further introduce an auxiliary task that trains the joint-anchored transformer to regress...
  </details>  
- **[Beyond Static Gaussians: An Empirical Investigation of Architectural Paradigms for Dynamic 3D Scene Reconstruction](https://arxiv.org/abs/2606.00452v1)**  
  Authors: Adrian Ramlal, John S. Zelek  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2606.00452v1.pdf)  
  <details><summary>Abstract</summary>

  Dynamic scene reconstruction via 3D Gaussian Splatting (3DGS) has emerged as a compelling approach for representing evolving environments, yet understanding trade-offs between methodologies remains crucial. This paper presents a comprehensive analysis of dynamic 3DGS methods, categorizing them into two paradigms: structure-guided methods employing auxiliary representations (deformation fields, canonical spaces, grids) to model temporal changes, and gaussian-centric methods encoding dynamics directly into primitives via continuous functions or 4D representations. We evaluate representative methods from both paradigms on the D-NeRF benchmark. Our findings reveal that structure-guided methods achieve superior reconstruction fidelity and compact model sizes, while gaussian-centric approaches demonstrate significantly higher rendering speeds enabling real-time performance, though with greater quality variability and potentially substantial storage overhead. This analysis highlights a fundamental trade-off between reconstruction quality/compactness versus rendering speed, providing insights to guide future research and application development in dynamic scene reconstruction.
  </details>  
  Keywords: 3d scene reconstruction  
- **[Feature-Optimized Vision for Adaptive 3D Scene Reconstruction](https://arxiv.org/abs/2605.31534v1)**  
  Authors: Eric Liang  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2605.31534v1.pdf)  
  <details><summary>Abstract</summary>

  Three-dimensional scene reconstruction depends on local image evidence that is both visually discriminative and geometrically useful. Fixed feature thresholds and uniform feature budgets are easy to deploy, but they can waste computation on repeated texture, low-parallax regions, or unstable points. This paper proposes an adaptive feature-optimized vision front end for 3D reconstruction. The method scores candidate features by texture, repeatability, distinctiveness, expected triangulation angle, and spatial coverage, then allocates a per-view feature budget to maximize useful tracks under a fixed reconstruction pipeline. A small synthetic multi-view prototype evaluates four selection policies across corridor, facade, object-table, and cluttered scenes. Compared with random, texture-only, and uniform-grid baselines, the adaptive policy obtains the best quality-aware completeness and the lowest aggregate reconstruction RMSE while preserving broad image coverage. The result is not a replacement for modern learned matching or neural reconstruction systems; it is a modular front-end policy that can make classical and learned...
  </details>  
  Keywords: 3d scene reconstruction  
- **[VolFill: Single-View Amodal 3D Scene Reconstruction with Volumetric Flow Matching](https://arxiv.org/abs/2605.31466v1)**  
  Authors: Tuan Duc Ngo, Chuang Gan, Evangelos Kalogerakis  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2605.31466v1.pdf)  
  <details><summary>Abstract</summary>

  Reconstructing the complete geometry of a scene from a single RGB image remains challenging - especially when inferring hidden structures where visual evidence is incomplete. We introduce VolFill, a generative framework that predicts the 3D structure of the complete scene rather than relying on traditional pixel-aligned regression. Our method utilizes a hybrid 3D VAE to compress sparse truncated unsigned distance function grids into a compact latent space, paired with a latent Diffusion Transformer that denoises this representation to recover the complete scene. We condition the generation on geometry foundation models, leveraging rich spatial priors for robust reasoning. Unlike existing methods limited by per-ray constraints or unstructured point-cloud queries, VolFill provides a structured representation that supports direct surface extraction and occupancy queries at scale. Extensive experiments on the SCRREAM and NRGB-D datasets demonstrate that our approach significantly outperforms current baselines, providing a robust foundation for holistic spatial understanding.
  </details>  
  Keywords: 3d scene reconstruction  
- **[Function2Scene: 3D Indoor Scene Layout from Functional Specifications](https://arxiv.org/abs/2605.30819v1)**  
  Authors: Ruiqi Wang, Qimin Chen, Daniel Ritchie, Angel X. Chang, Manolis Savva, Kai Wang, Hao Zhang  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2605.30819v1.pdf)  
  <details><summary>Abstract</summary>

  Most text-driven 3D indoor scene synthesis methods generate rooms from object-centric prompts, asking what furniture should be placed rather than how the space is used. Yet in real interior design, a layout is judged by how well it supports its occupants, e.g., their activities and physical needs. We introduce Function2Scene, a framework for generating 3D indoor layouts from functional specifications, i.e., natural-language design briefs describing who will use a room and what they need to do there. Given such a specification, our system parses occupant personas and activities, derives a customized set of functional design constraints from a taxonomy of 17 criteria spanning spatial, ergonomic, activity, and environmental considerations, and uses these constraints to guide layout generation. Rather than relying on an LLM to directly produce a final scene, Function2Scene performs iterative evaluation and refinement through a tool-augmented check-and-repair loop, combining geometric measurements, LLM-based contextual reasoning, and VLM-based visual assessment....
  </details>  
- **[Advances in Neural 3D Mesh Texturing: A Survey](https://arxiv.org/abs/2606.00137v1)**  
  Authors: Sai Raj Kishore Perla, Hao Zhang, Ali Mahdavi-Amiri  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2606.00137v1.pdf)  
  <details><summary>Abstract</summary>

  Texturing 3D meshes plays a vital role in determining the visual realism of digital objects and scenes. Although recent generative 3D approaches based on Neural Radiance Fields and Gaussian Splatting can produce textured assets directly, polygonal meshes remain the core representation across modeling, animation, visual effects, and gaming pipelines. Neural 3D mesh texturing therefore continues to be an essential and active area of research. In this survey, we present a comprehensive review of recent advances in neural 3D mesh texturing, covering methods for texture synthesis, transfer, and completion. We first summarize key foundations in mesh geometry, texture mapping, differentiable rendering, and neural generative models, and then organize the literature into a unified taxonomy spanning early GAN-based methods to modern diffusion-based pipelines. We further analyze common architectures and supervision strategies, review datasets and evaluation protocols, and discuss emerging applications, practical/commercial systems, and open challenges. Together, these insights provide a structured perspective...
  </details>  
  Keywords: texture synthesis  
- **[DynaFLIP: Rethinking Robotics Perception via Tri-Modal-Dynamics Guided Representation](https://arxiv.org/abs/2605.30350v1)**  
  Authors: Jusuk Lee, Seungjae Lee, Jonghun Shin, Hoseong Jung, Sungha Kim, Daesol Cho, H. Jin Kim, Jia-Bin Huang, Furong Huang  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2605.30350v1.pdf)  
  <details><summary>Abstract</summary>

  Robot manipulation critically depends on perception that preserves the action-relevant aspects of a scene. Yet most robot learning pipelines are built upon visual encoders pre-trained for static recognition or vision-language alignment, leaving motion understanding to downstream policies. We introduce DynaFLIP, a dynamics-aware multimodal pre-training framework that pushes motion understanding upstream into perception. We construct image-language-3D flow triplets from heterogeneous human and robot videos, and use these triplets as training-time supervision to shape an image-only encoder. Our key idea is to encourage the three modalities to span a small simplex volume in the shared hyperspherical space -- a smaller simplex volume indicating stronger alignment. To avoid the geometric ambiguity and trivial collapse of naive volume minimization, we combine simplex-volume minimization with a cosine regularizer and a contrastive objective. Our analyses show that DynaFLIP focuses on control-relevant regions critical for manipulation. The resulting dynamics-aware representations serve as reusable visual backbones and consistently...
  </details>  
- **[REST3D: Reconstructing Physically Stable 3D Scenes from a Single Image](https://arxiv.org/abs/2605.30338v1)**  
  Authors: Xiaoxuan Ma, Jiashun Wang, Nicolas Ugrinovic, Yehonathan Litman, Kris Kitani  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2605.30338v1.pdf)  
  <details><summary>Abstract</summary>

  Reconstructing physically stable 3D scenes from a single RGB image enables casual images to be converted into simulation-ready digital assets for applications such as immersive interaction and content creation. However, existing single-image reconstruction methods fall short in capturing the physical structure of a scene. As a result, they often produce geometrically plausible but physically inconsistent results, including object floating and penetration, which lead to unstable behavior in physics simulations. Image-conditioned scene generation methods improve physical plausibility but often rely on strong scene priors, yielding plausible yet inaccurate object arrangements that fail to match the input image. We propose REST3D, a single-image reconstruction framework that can reconstruct physically stable 3D scenes by integrating physical scene understanding with physics-constrained refinement. We first introduce an agentic physical scene understanding technique that constructs a scene-tree representation capturing object physical states and inter-object relationships from a gravity-support perspective, providing a structural prior for reconstruction. Leveraging...
  </details>  
  Keywords: image-to-3d  
- **[SuperVoxelGPT: Adaptive and Ordered 3D Tokenization for Autoregressive Shape Generation](https://arxiv.org/abs/2605.29655v2)**  
  Authors: Yuan Li, Congyi Zhang, Xifeng Gao, Xiaohu Guo  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2605.29655v2.pdf)  
  <details><summary>Abstract</summary>

  Autoregressive multimodal large language models (MLLMs) enable 3D generation but struggle to scale to high-resolution shapes due to inadequate 3D tokenizations. Compact set-based representations discard deterministic spatial ordering, leading to ambiguous sequence prediction, while uniform or octree-based voxel grids preserve ordering at the cost of severe redundancy and excessively long sequences. This structural trade-off limits stable and efficient autoregressive 3D generation. We present SuperVoxelGPT, a representation-first framework that resolves this tension through adaptive and deterministically ordered supervoxel tokenization. Given a prompt, we first predict a coarse geometric saliency distribution and construct a shape-adaptive supervoxel partition using saliency-guided centroidal Voronoi tessellation, allocating fine-grained cells to complex regions and larger cells to smooth regions. Conditioned on the text and ordered supervoxel layout, we introduce a SuperVoxelVAE and fine-tune a pretrained MLLM to autoregressively generate supervoxel tokens. Experiments on Trellis-500K show that SuperVoxelGPT reduces token sequence length to 12.8% of uniform voxel tokenization...
  </details>  
- **[Comparative evaluation of photogrammetric reconstruction methods and 3D Gaussian Splatting for road surface roughness analysis](https://arxiv.org/abs/2605.29452v1)**  
  Authors: Marouane Elmegdar, Teng Xiao  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2605.29452v1.pdf)  
  <details><summary>Abstract</summary>

  Image-based 3D reconstruction offers a low-cost alternative to traditional sensor-based techniques for road surface assessment. This study compares four reconstruction pipelines--COLMAP, Meshroom, Metashape, and 3D Gaussian Splatting (3DGS)--to evaluate their ability to estimate road surface roughness from smartphone imagery. All point clouds were processed in CloudCompare using a consistent workflow involving orientation alignment, segmentation, normal estimation, and roughness computation at neighborhood radiuses of 0.2, 0.4, and 0.6 model units. The results show that COLMAP provides the highest sensitivity to micro-texture, while Meshroom yields balanced reconstructions with moderate roughness variation. Metashape produces the smoothest geometry due to its internal filtering, and 3DGS captures visible irregularities but exhibits higher noise and lower density. The comparison demonstrates that open-source pipelines are viable for relative roughness evaluation, offering a practical approach for low-cost pavement monitoring.
  </details>  
- **[GAP3D: Generative Alignment of VLM Latents to Patch-Level Embeddings for 3D Generation](https://arxiv.org/abs/2605.28995v2)**  
  Authors: Polytimi Anna Gkotsi, Andrii Zadaianchuk, Mohammad Mahdi Derakhshani  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2605.28995v2.pdf)  
  <details><summary>Abstract</summary>

  Recent approaches integrating vision-language models (VLMs) as prompt encoders for generative model conditioning typically rely on expensive end-to-end training or map features to compressed representations, discarding the dense spatial structure required for geometry-aware tasks like 3D asset generation. To address this, we propose GAP3D, a modular, diffusion-based approach that aligns VLM-generated latents directly to the complete, patch-level feature space of a pre-trained image encoder, enabling a frozen downstream generative model to utilize a VLM as prompt encoder while maintaining a spatially structured conditioning signal. Evaluated on 3D asset generation, our method bypasses the need for large-scale 3D data by training mainly on general-domain image-text pairs. It also exhibits emergent zero-shot capabilities for multimodal prompts, despite being trained exclusively on text input. Finally, while currently prioritizing high-level semantics over fine-grained detail, GAP3D demonstrates that the representation gap between VLM and image-encoder feature spaces can be partially bridged through diffusion-based alignment, taking...
  </details>  
  Keywords: 3d asset generation  
- **[CubePart: An Open-Vocabulary Part-Controllable 3D Generator](https://arxiv.org/abs/2605.28763v1)**  
  Authors: Yiheng Zhu, Kangle Deng, Jean-Philippe Fauconnier, Inaki Navarro, Daiqing Li, Ava Pun, Yinan Zhang, Peiye Zhuang, Xiaoxia Sun, Maneesh Agrawala, Kiran Bhat, Tinghui Zhou  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2605.28763v1.pdf) | [![Project](https://img.shields.io/badge/-Project-blue)](https://cubepart.github.io)  
  <details><summary>Abstract</summary>

  Interactive 3D assets used in games and simulation are typically decomposed into specific semantic parts to support animation, physics, and scripted behaviors, yet most generative 3D models produce either monolithic meshes or arbitrary part decompositions that cannot be aligned with application-specific requirements. We present CubePart, a generative framework for open-vocabulary, part-controllable 3D mesh generation that exposes part structure as an explicit inference-time control signal. Given a global text prompt and a user-defined parts schema expressed as an open-ended list of part names, our method generates a set of meshes - one per schema element - that assemble into a coherent object while respecting the specified semantic structure. To enable this capability, we introduce a scalable data pipeline to construct a large open-vocabulary, part-labeled 3D dataset, along with a two-stage generative architecture that separates global shape synthesis from part-level decoding. We demonstrate that the resulting assets can be directly integrated into...
  </details>  
  Keywords: 3d generator  
- **[MUSE: Benchmarking Manufacturable, Functional, and Assemblable Text-to-CAD Generation](https://arxiv.org/abs/2605.28579v2)**  
  Authors: Xiaoyu Dong, Zhi Li, Xiao-Ming Wu  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2605.28579v2.pdf) | [![Project](https://img.shields.io/badge/-Project-blue)](https://dong7313.github.io/muse-benchmark/.) | [![Dataset](https://img.shields.io/badge/-Dataset-orange)](https://dong7313.github.io/muse-benchmark)  
  <details><summary>Abstract</summary>

  Large language models (LLMs) have recently advanced text-driven 3D generation, yet Text-to-CAD remains far from supporting industrial product design. Existing benchmarks focus primarily on generating single-part CAD models and evaluate them using geometric similarity metrics that fail to capture functionality, manufacturability, and assemblability. To address this gap, we introduce MUSE, a Text-to-CAD benchmark focused on complex, editable boundary representation (B-Rep) assemblies. MUSE pairs practical design instances with structured Design Specifications and evaluates generated models through a three-stage protocol: code check, geometric check, and design-intent alignment. The final stage uses design-specific rubrics to assess functionality, manufacturability, and assemblability, moving beyond shape matching toward practical design quality. To enable scalable evaluation, we use a rubric-based visual language model (VLM) judge and validate its reliability through human annotation. Experiments on closed-source and open-source LLMs reveal a clear failure cascade from executable code to valid geometry and finally to engineering-ready design, with even the...
  </details>  
- **[Feedforward 3D Editing Learns from Semantic-Part Transformation](https://arxiv.org/abs/2605.27351v2)**  
  Authors: Jiawei Weng, Saining Zhang, Zhenxin Diao, Peishuo Li, Henghaofan Zhang, Junhao Chen, Hao Zhao  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2605.27351v2.pdf)  
  <details><summary>Abstract</summary>

  3D editing is a fundamental capability for scalable 3D content creation. While image editing has rapidly evolved toward large-scale feedforward generative paradigms, 3D AI generation remains dominated by training-free editing pipelines. A central challenge of feedforward 3D editing lies in the lack of high-quality paired supervision. Editable 3D assets require simultaneous preservation of geometry, multi-view consistency, structural coherence, and localized edit controllability. Existing 3D editing datasets often rely on independently generated assets, image-mediated reconstruction or narrow edit taxonomies, leading to inaccurate localization, weak preservation, blurred edit boundaries, and limited semantic consistency. In this work, we introduce a new perspective: scalable feedforward 3D editing should be learned from semantic-part transformations. Based on this insight, we propose Pxform, a high-quality 3D editing dataset with over 100K consistent before/after editing pairs across seven edit types. Instead of treating objects as unstructured shapes, our pipeline grounds edits directly in semantic 3D parts. Built upon...
  </details>  
- **[DelowlightSplat: Feed-Forward Gaussian Splatting for Lowlight 3D Scene Reconstruction](https://arxiv.org/abs/2605.26629v1)**  
  Authors: Fuzhen Jiang, Zengtian Xie, Zhuoran Li  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2605.26629v1.pdf)  
  <details><summary>Abstract</summary>

  Novel-view synthesis and 3D reconstruction from sparse posed images are central to robotics and AR/VR. Yet, feed-forward 3D Gaussian reconstruction fails under lowlight due to noise, color shifts, and unreliable correspondence. We propose DelowlightSplat, a lowlight-aware feed-forward Gaussian splatting framework for clean novel-view rendering. We build a controllable multi-view lowlight benchmark by degrading only context views while keeping target views clean. We introduce a lightweight Lowlight Adapter for residual enhancement to improve matchability, and couple it with cost-volume-based multi-view inference to directly predict clean 3D Gaussians. Experiments show that DelowlightSplat significantly outperforms previous feed-forward method and two-stage pipeline under lowlight conditions.
  </details>  
  Keywords: 3d scene reconstruction  
- **[Underwater360: Reconstructing Underwater Scenes from Panoramic Images with Omnidirectional Gaussian Splatting](https://arxiv.org/abs/2605.26447v1)**  
  Authors: Jiangbei Hu, Weichao Song, Shibo Yu, Mohan Wang, Zihan Yi, Rui Wu, Mingkang Xiang, Na Lei, Shengfa Wang, Zhongxuan Luo, Ying He  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2605.26447v1.pdf) | [![GitHub](https://img.shields.io/github/stars/SwcK423/Underwater360?style=social)](https://github.com/SwcK423/Underwater360)  
  <details><summary>Abstract</summary>

  Underwater scene reconstruction is essential for immersive exploration of aquatic environments, yet remains challenging due to complex participating-media effects such as absorption and scattering, as well as the limited field of view (FoV) of conventional cameras. Although combining panoramic imaging with 3D Gaussian Splatting (3DGS) offers a promising direction for photorealistic underwater rendering, traditional 3DGS struggles with both spherical projection distortion and underwater medium degradation. In this paper, we propose \textbf{Underwater360}, a physics-informed omnidirectional 3DGS framework for underwater panoramic scene reconstruction. First, we introduce an Omnidirectional Gaussian Splatting module that performs ray casting directly in spherical camera space instead of relying on 2D projection approximations, thereby reducing geometric distortions under 360$^\circ$ FoV. Second, we design a physics-based appearance-medium modeling architecture with pose-conditioned appearance embeddings to explicitly decouple intrinsic scene radiance from depth-dependent backscatter and attenuation, enabling physically grounded scene appearance restoration. Finally, we establish a new panoramic underwater benchmark dataset...
  </details>  
  Keywords: novel view synthesis  
- **[TriSplat: Simulation-Ready Feed-Forward 3D Scene Reconstruction](https://arxiv.org/abs/2605.26115v1)**  
  Authors: Weijie Wang, Zimu Li, Jinchuan Shi, Zeyu Zhang, Botao Ye, Marc Pollefeys, Donny Y. Chen, Bohan Zhuang  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2605.26115v1.pdf)  
  <details><summary>Abstract</summary>

  Sparse-view 3D reconstruction is increasingly addressed with feed-forward splatting networks that predict explicit primitives directly from images. Yet most existing methods remain centered on Gaussian primitives and expose surfaces only indirectly: extracting a usable mesh for downstream simulation, physics reasoning, or embodied interaction still requires expensive post-hoc steps that break the feed-forward promise. This limitation is especially pronounced in pose-free settings, where scene structure and camera parameters must be estimated jointly from sparse observations. We present TriSplat, a feed-forward reconstruction network that represents scenes with oriented triangle primitives and directly exports simulation-ready mesh scenes from a single forward pass. Given input images, the network predicts local 3D point maps, triangle attributes, camera poses, and optional intrinsics. Rather than regressing triangle orientation as an unconstrained latent variable, our approach constructs geometry normals from the predicted point maps, refines them with an image-conditioned normal head, and converts them into stable local frames...
  </details>  
  Keywords: 3d scene reconstruction  
- **[Helix4D: Complex 4D Mesh Generation](https://arxiv.org/abs/2605.26109v1)**  
  Authors: Jiraphon Yenphraphai, Jianqi Chen, Jian Wang, Gordon Qian, Sergey Tulyakov, Rameen Abdal, Raymond A. Yeh, Peter Wonka, Chaoyang Wang  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2605.26109v1.pdf)  
  <details><summary>Abstract</summary>

  Current video-to-4D methods struggle with complex topology changes, transparent materials, thin structures, and inner surfaces. We present Helix4D, a dynamic mesh generation framework by inheriting the expressive representation of Trellis2, adapting it from image-to-3D to video-conditioned 4D generation. Our design arises from two key questions: (a) how to enable Trellis2's frame-local attention to share information across frames while preserving its pretrained quality on rare cases such as transparent objects and inner surfaces, and (b) how to inject temporal information into a purely 3D positional encoding without breaking pretrained capabilities. We address (a) with a sliding-window cross-frame attention and anchor on the first frame. The first frame is generated by the base Trellis2 model and injected into our model, letting it inherit Trellis2's quality in rare cases through cross-frame attention. We address (b) with a 4D temporal encoding that repurposes redundant low-frequency spatial RoPE bands for time, extending the encoding from...
  </details>  
  Keywords: image-to-3d  
- **[Fishbone: From One 3D Asset to a Million Controllable Edits](https://arxiv.org/abs/2605.24805v1)**  
  Authors: Yumeng He, Xiaoying Wang, Peihao Li, Yanjia Huang, Joe Masterjohn, Jiajun Wu, Leonidas Guibas, Yin Yang, Ying Jiang, Chenfanfu Jiang  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2605.24805v1.pdf)  
  <details><summary>Abstract</summary>

  Large-scale controllable 3D assets are critical for computer graphics, embodied AI, robotics, and interactive content creation, yet creating diverse 3D assets remains challenging due to the high cost of manual modeling and rigging. Shape deformation offers a natural way to generate variations from existing meshes, but existing data-driven methods often rely on sparse user inputs, while parametric editing frameworks require manually designed control structures and category-specific configurations. Inspired by natural creatures, where a central spine governs global shape and cross-sectional ribs control local variation, we introduce Fishbone, a unified rib-spine representation for general shapes that supports controllable parametric mesh deformation, reduced-space dynamics, and animation. Given an input mesh, Fishbone computes a geodesic scalar field with an adaptive heat method, extracts iso-contours as cross-sectional ribs, constructs a smooth geometry-aware spine through rib centers, and associates surface vertices with nearby rib and spine structures using Gaussian-weighted skinning. The resulting representation enables real-time...
  </details>  
  Keywords: controllable 3d generation, rigging  
- **[Artiverse: A Diverse and Physically Grounded Dataset for Articulated Objects](https://arxiv.org/abs/2605.24403v1)**  
  Authors: Denys Iliash, Jiayi Liu, Egor Fokin, Qirui Wu, Ali Mahdavi-Amiri, Manolis Savva, Angel X. Chang  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2605.24403v1.pdf)  
  <details><summary>Abstract</summary>

  We present Artiverse, a diverse and physically grounded dataset of high-quality articulated 3D objects designed for realistic functional modeling and simulation. Artiverse contains 5.4K human-authored objects across a broad range of 88 categories, aggregated from multiple 3D static repositories. Objects are annotated with functional parts, interior structures, realistic kinematic relationships and articulated joints including multi-DoF joints, and physical attributes such as metric scale, material, and mass. We develop a semi-automated annotation pipeline that combines few-shot segmentation, geometric reasoning, and multi-stage human verification to achieve high-quality and efficient annotation, reducing manual annotation time by over 30%. We demonstrate the value of Artiverse on tasks of part mobility analysis, articulated object generation, and physics-based interaction. Artiverse provides a data resource to advance functional understanding for articulated objects.
  </details>  
  Keywords: articulated object generation  
- **[AnySurf: Any Surface Generation with Directed Edge](https://arxiv.org/abs/2605.26149v1)**  
  Authors: Wenda Shi, Chenyuan Pan, Dengming Zhang, Yiren Song, Biao Zhang, Xingxing Zou  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2605.26149v1.pdf)  
  <details><summary>Abstract</summary>

  Open surface components prevail in real industrial 3D content and support rendering, physical simulation and geometric editing. Garments serve as a typical open surface type, with numerous existing generation methods leveraging sewing patterns to generate 2D panels and stitch them into 3D shapes. Such domain-specific designs lack scalability and cannot generalize to shoes and accessories. Common field-based 3D generators prioritize watertight meshes and tend to create flawed double-layer structures on open surfaces. Though Trellis2 adopts field-free representation, its open surface results still contain normal and topology errors. We present AnySurf, a unified framework generating open, closed and hybrid 3D surfaces with accurate face orientation. Built on directed-edge enhanced Flexible Dual Grid (FDG-D), our representation retains normal direction information via oriented grid edges. We also propose ROS-FT post-training and a lightweight DE-Adapter with merely 1% extra parameters, facilitating directed edge learning while preserving original generation performance. We further construct Outfit3D dataset...
  </details>  
  Keywords: 3d generator  
- **[GenRecon: Bridging Generative Priors for Multi-View 3D Scene Reconstruction](https://arxiv.org/abs/2605.23888v1)**  
  Authors: Katharina Schmid, Nicolas von Lützow, Jozef Hladký, Angela Dai, Matthias Nießner  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2605.23888v1.pdf)  
  <details><summary>Abstract</summary>

  We introduce a new approach to high-fidelity 3D scene reconstruction from multi-view RGB images that tightly couples reconstruction with a strong generative 3D prior. We cast scene reconstruction as conditional 3D generation over a set of spatially-localized, overlapping chunks that together tile the scene, scaling generation to large scene extents. Crucially, we inherit the fidelity and completeness of state-of-the-art generative shape models -- we use Trellis.2 as an example -- which we generalize to the scene level. To this end, we propose a projection-based conditioning mechanism that lifts posed multi-view image features into a coherent 3D representation aligned with the generative model, independent of view ordering and spatially anchored to the scene, yielding high-fidelity, multi-view consistent generated geometry. This enables lifting the strong object-level prior of Trellis.2 to multi-view, scene-scale generation, producing faithful, editable PBR mesh reconstructions of indoor environments. As a result, we obtain high-fidelity results that outperform cutting-edge...
  </details>  
  Keywords: 3d scene reconstruction  
- **[Perceptually Lossless Tactile Texture Synthesis with Compact Spectral Envelope Models](https://arxiv.org/abs/2605.23804v1)**  
  Authors: Jagan K. Balasubramanian, Yasemin Vardar  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2605.23804v1.pdf)  
  <details><summary>Abstract</summary>

  Modern audio-visual media rely on compact representations for efficient storage and transmission, whereas realistic digital touch still depends on high-resolution tactile recordings. Existing approaches for representing tactile signals constrain manipulation and limit the generation of new content. Here, we introduce two compact representations, spectral beta and spectral slope, that capture the temporal spectral structure of finger-surface friction signals while preserving perceptually relevant information. Spectral beta models spectral skewness using a two-parameter beta distribution, whereas spectral slope approximates the spectrum with an asymmetric bandpass filter defined by low- and high-pass orders. We evaluated these representations in a perceptual study with 14 participants using five virtual textures rendered on a friction-modulation display and compared them with physical textures and high-fidelity reproductions of recorded signals. Spectral beta achieved perceptual similarity ratings comparable to those of the original high-fidelity reproductions. Regression analysis further showed that matching spectral energy across nine critical frequency bands was...
  </details>  
  Keywords: texture synthesis  
- **[VDE: Training-Free Accelerating Rectified Flow Model via Velocity Decomposition and Estimation](https://arxiv.org/abs/2605.23381v1)**  
  Authors: Junwen Tan, Jinglin Liang, Hongyuan Chen, Shuangping Huang  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2605.23381v1.pdf)  
  <details><summary>Abstract</summary>

  Though rectified flow models have achieved remarkable performance in image, video, and 3D generation, their practical deployments are challenged by slow inference speeds. Prior acceleration methods reuse cached features from previous steps, which neglects the growing mismatch between static caches and the evolving input, leading to reduced output fidelity. This work proposes Velocity Decomposition and Estimation (VDE), a training-free acceleration method that shifts the paradigm from caching-and-reusing to decomposing-and-estimating. Specifically, VDE decomposes the model's velocity into components parallel and orthogonal to the input, exploiting their temporal predictability and directional stability for precise, input-adaptive estimation. To prevent error accumulation, it periodically anchors the model's state via full forward passes. Extensive experiments on image and video generation tasks demonstrate that VDE achieves substantial acceleration with minimal loss in visual quality. Notably, VDE accelerates Flux by 3.22 times and achieves an LPIPS of 0.069 on Qwen-Image, outperforming the best baseline with a 52.2%...
  </details>  
- **[LangFlash: Feed-forward 3D Language Gaussian Splatting from Sparse Unposed Images](https://arxiv.org/abs/2605.23287v1)**  
  Authors: Yilong Liu, Wanhua Li, Chen Zhu-Tian, Hanspeter Pfister  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2605.23287v1.pdf) | [![Project](https://img.shields.io/badge/-Project-blue)](https://liylo.github.io/langflash.github.io)  
  <details><summary>Abstract</summary>

  We present LangFlash, a feed-forward framework for 3D Language Gaussian Splatting that reconstructs 3D scenes parameterized by Gaussian primitives enriched with language-aligned semantic features from sparse unposed multi-view images. Unlike optimization-based 3D methods, LangFlash directly predicts the geometry and semantics in a single forward pass, enabling low-latency 3D reconstruction and language-consistent scene understanding. To support large-scale training, we enriched the RealEstate10k dataset with coherent and dense semantic information for 3D semantic supervision. Furthermore, we propose a sparse semantic encoding scheme that combines a global semantic dictionary with locally varying per-primitive weights, preserving high-level linguistic information, while reducing representation complexity. Experimental results show that LangFlash achieves superior novel view synthesis and semantic consistency compared with previous methods. This study establishes a new paradigm for pose-free, language-grounded 3D scene reconstruction, advancing generalizable 3D vision and multimodal scene understanding. Demo is available at https://liylo.github.io/langflash.github.io/.
  </details>  
  Keywords: 3d scene reconstruction, novel view synthesis  
- **[AssetGen: Deployable 3D Asset Generation at Interactive Speed](https://arxiv.org/abs/2605.26137v1)**  
  Authors: Dilin Wang, Xiaoyu Xiang, Kihyuk Sohn, Tom Monnier, Yu-Ying Yeh, Thu Nguyen-Phuoc, Jiawen Zhang, Yuchen Fan, Antoine Toisoul, Hyunyoung Jung, Prithviraj Dhar, Michael Bunnell, Nikolaos Sarafianos, Chuhang Zou, Roman Shapovalov, Andrea Vedaldi, Rakesh Ranjan  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2605.26137v1.pdf)  
  <details><summary>Abstract</summary>

  While 3D generation is progressing rapidly, recent work has often focused on obtaining high-resolution assets, leaving user experience and deployability as afterthoughts. We present AssetGen, a 3D generator that focuses instead on these two aspects. Given one reference image, in 30 seconds it produces a high-quality mesh with baked normals, a color texture, and a controlled polygon budget suitable for real-time rendering, including mobile use cases. The AssetGen Flash variant further reduces latency to 14 seconds for interactive and agentic creation loops. Our model generates the object geometry with a coarse-to-refine VecSet framework, which implements mesh simplification, cleaning, and normal baking on the GPU, and a fast parallel UV unwrapping. It then generates textures in a multi-view fashion, followed by backprojection and 3D inpainting. Model distillation, kernel optimization, and pipeline parallelization are co-designed to accelerate the system end-to-end. We introduce numerous automated and blind human evaluations and demonstrate competitive visual...
  </details>  
  Keywords: 3d generator, 3d asset generation  
- **[No Pose, No Problem in 4D: Feed-Forward Dynamic Gaussians from Unposed Multi-View Videos](https://arxiv.org/abs/2605.22190v1)**  
  Authors: Matteo Balice, Yanik Kunzi, Chenyangguang Zhang, Matteo Matteucci, Marc Pollefeys, Sungwhan Hong  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2605.22190v1.pdf)  
  <details><summary>Abstract</summary>

  Recent feed-forward 3D gaussian splatting methods have made dramatic progress on individual aspects of 3D scene reconstruction, but no existing method jointly addresses dynamic content, multi-view input, and unknown camera poses in a single feed-forward pass. Methods that handle dynamics either require accurate camera poses or accept only monocular input; pose-free multi-view methods address only static scenes; and per-scene optimization methods bridge some of these gaps but at minutes-to-hours cost per scene. We introduce NoPo4D, the first feed-forward system that addresses this empty quadrant. Building on a pretrained geometry backbone and recent 4D Gaussian frameworks, NoPo4D introduces a velocity decomposition that splits Gaussian motion into per-pixel image-plane shifts and depth changes, allowing direct supervision from pseudo ground-truth optical flow on the 2D component. This sidesteps both the differentiable rendering that couples prior posed methods to pose accuracy and the 3D motion ground truth that prior pose-free methods require. The system...
  </details>  
  Keywords: 3d scene reconstruction  
- **[Variance Reduction for Expectations with Diffusion Teachers](https://arxiv.org/abs/2605.21489v2)**  
  Authors: Jesse Bettencourt, Xindi Wu, Matan Atzmon, James Lucas, Jonathan Lorraine  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2605.21489v2.pdf)  
  <details><summary>Abstract</summary>

  Pretrained diffusion models serve as frozen teachers feeding downstream pipelines such as text-to-3D, single-step distillation, and data attribution. The teacher gradients these pipelines consume are Monte Carlo (MC) expectations over noise levels and Gaussian noise samples; their estimator variance dominates compute cost because each draw requires expensive upstream work (rendering, simulation, encoding). We introduce CARV, a compute-aware variance-accounting framework that motivates a hierarchical MC estimator: amortize the expensive upstream computation over cheap diffusion-noise resamples, sharpened by timestep importance sampling and a stratified-inverse-CDF construction. In our text-to-3D distillation and attribution experiments, CARV delivers 2-3x effective compute multipliers (most from amortized reuse; ~25% additional from IS+stratification) without changing the objective; in single-step distillation, the same techniques cut gradient variance by an order of magnitude but do not improve downstream FID, marking the regime where MC variance is no longer the bottleneck.
  </details>  
  Keywords: text-to-3d  
- **[PhysX-Omni: Unified Simulation-Ready Physical 3D Generation for Rigid, Deformable, and Articulated Objects](https://arxiv.org/abs/2605.21572v1)**  
  Authors: Ziang Cao, Yinghao Liu, Haitian Li, Runmao Yao, Fangzhou Hong, Zhaoxi Chen, Liang Pan, Ziwei Liu  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2605.21572v1.pdf)  
  <details><summary>Abstract</summary>

  Simulation-ready physical 3D assets have emerged as a promising direction owing to their broad applicability in downstream tasks. However, most existing 3D generation methods either neglect physical properties or are limited to a single asset category, e.g., rigid, deformable, or articulated objects. To address these limitations, we introduce PhysX-Omni, a unified framework for simulation-ready physical 3D generation across diverse asset types. Specifically, we develop a novel and efficient geometry representation tailored for Vision-Language Models, which directly encodes high-resolution 3D structures without compression, significantly improving generation performance. In addition, we construct the first general simulation-ready 3D dataset, PhysXVerse, covering diverse indoor and outdoor categories. Furthermore, to comprehensively and flexibly evaluate both generative and understanding capabilities in the wild, we propose PhysX-Bench, which encompasses six key attributes: geometry, absolute scale, material, affordance, kinematics, and function description. Extensive experiments with conventional metrics and PhysX-Bench show that PhysX-Omni performs strongly in both generation and...
  </details>  
- **[Stream3D: Sequential Multi-View 3D Generation via Evidential Memory](https://arxiv.org/abs/2605.21472v3)**  
  Authors: Kaichen Zhou, Zeyang Bai, Xinhai Chang, Mengyu Wang, Paul Liang, Fangneng Zhan  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2605.21472v3.pdf) | [![Project](https://img.shields.io/badge/-Project-blue)](https://stream-3d.github.io/stream3d.github.io)  
  <details><summary>Abstract</summary>

  View-conditioned 3D generators such as SAM 3D, TRELLIS, and Hunyuan3D produce high-quality object reconstructions from a single view, but real-world visual observation often arrives as long monocular streams. Naively applying these generators to each streaming frame independently leads to severe temporal inconsistency in the generated results. To address this problem, we propose Stream3D, the first training-free streaming mechanism that turns a frozen view-conditioned 3D generator into a streaming generator with constant cross-chunk memory. Stream3D achieves this by maintaining a compact evidential memory, which selectively caches the most informative historical frames based on a proposed evidence score mechanism. As the stream progresses, the memory dynamically updates to retain a fixed number of informative frames, preventing the memory footprint from growing linearly with sequence length. This also prevents degradation over long sequences and keeps the underlying generator completely unchanged without retraining, architectural modifications, or auxiliary losses. Evaluated on both realistic and synthetic...
  </details>  
  Keywords: 3d generator  
- **[ROAR-3D: Routing Arbitrary Views for High-Fidelity 3D Generation](https://arxiv.org/abs/2605.21121v1)**  
  Authors: Hanxiao Sun, Mingxin Yang, Shuhui Yang, Zebin He, Xintong Han, Hongbo Fu, Chunchao Guo, Wenhan Luo  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2605.21121v1.pdf)  
  <details><summary>Abstract</summary>

  Single-image-to-3D generative models can now produce high-quality geometry, yet conditioning on a single view inevitably introduces ambiguity about unseen regions. Multi-view conditioning can reduce this ambiguity, but existing methods either require fixed canonical viewpoints or rely on external reconstruction modules that impose heavy training costs and limit generation quality. We observe that pretrained single-view models already possess strong 2D-to-3D grounding that can be reused for multi-view conditioning. However, a closer analysis reveals that their conditioning mechanism entangles orientation control with geometry transfer, two functions that conflict when images from different viewpoints are naively combined. Based on this analysis, we propose ROAR-3D, a lightweight method that upgrades a pretrained single-view model to accept an arbitrary number of unposed images. A token-wise view router assigns each 3D latent token to its most relevant view, implicitly establishing 2D-to-3D correspondences without explicit pose input. A dual-stream attention design preserves the pretrained primary-view behavior while...
  </details>  
  Keywords: image-to-3d  
- **[Structural Energy Guidance for View-Consistent Text-to-3D Generation](https://arxiv.org/abs/2605.19876v1)**  
  Authors: Qing Zhang, Jinguang Tong, Jing Zhang, Jie Hong, Xuesong Li  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2605.19876v1.pdf)  
  <details><summary>Abstract</summary>

  Text-to-3D generation based on diffusion models often suffers from the Janus problem, leading to inconsistent geometry across viewpoints. This work identifies viewpoint bias in 2D diffusion priors as the main cause and proposes Structural Energy-Guided Sampling (SEGS), a training-free and plug-and-play framework to improve multi-view consistency. SEGS constructs a structural energy in the PCA subspace of U-Net features and injects its gradient into the denoising process. It can be easily integrated into SDS/VSD pipelines without retraining. Experiments show that SEGS reduces the Janus Rate by about 10% on average and improves View-CS scores across multiple baselines, including DreamFusion, Magic3D, and LucidDreamer. This method effectively alleviates viewpoint artifacts while preserving appearance fidelity, providing a flexible solution for high-quality text-to-3D content generation.
  </details>  
  Keywords: text-to-3d  
- **[TelePhysics: Physics-Grounded Multi-Object Scene Generation from a Single Image with Real-Time Interaction](https://arxiv.org/abs/2605.20290v1)**  
  Authors: Xin Zhang, Yabo Chen, Yijie Fang, Wanying Qu, Haibin Huang, Chi Zhang, Feng Xu, Xuelong Li  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2605.20290v1.pdf) | [![GitHub](https://img.shields.io/github/stars/xinzhang007/TelePhysics?style=social)](https://github.com/xinzhang007/TelePhysics)  
  <details><summary>Abstract</summary>

  Recent generative video models achieve impressive visual quality but remain constrained by limited physical consistency and controllability. Existing video generation methods provide minimal physical control, and single-image-to-3D conversion approaches often suffer from object interpenetration. Furthermore, physics-based scene-level 3D generation methods exhibit spatial misalignment, stylized artifacts, and inconsistencies with the input data, restricting their use in realistic interactive video synthesis. We propose TelePhysics, a training-free framework that converts a single image into a physically consistent and controllable video through holistic scene-level 3D reconstruction. By representing the full scene geometry in a unified spatial coordinate system, TelePhysics resolves object penetration and alignment ambiguity. Unlike prior methods, this formulation enables accurate scenelevel multi-object interactions and introduces richer, complex control types for advanced mechanicsbased manipulation. By decoupling simulation from rendering, TelePhysics bypasses latency-heavy priors, achieving real-time physical interaction previews paired while preserving photorealistic visual fidelity. Experimental results demonstrate that TelePhysics substantially outperforms prior methods...
  </details>  
  Keywords: image-to-3d  
- **[GaussianZoom: Progressive Zoom-in Generative 3D Gaussian Splatting with Geometric and Semantic Guidance](https://arxiv.org/abs/2605.18252v1)**  
  Authors: Jiale Shi, Jiarui Hu, Zesong Yang, Kaixuan Luan, Hujun Bao, Zhaopeng Cui  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2605.18252v1.pdf)  
  <details><summary>Abstract</summary>

  We introduce GaussianZoom, a generative zoom-in 3D reconstruction system with an iterative progressive framework that combines geometry-consistent scene modeling and multi-scale semantic reasoning to enable high-fidelity extreme zoom-in rendering from low-resolution inputs. To achieve this, we develop a novel multi-view consistent super-resolution module with depth-based feature warping and VLM-driven detail synthesis, ensuring accurate multi-view correspondence while enriching fine-scale appearance beyond the observed resolution. To support zooming across large magnification ranges, we further introduce a new expandable continuous Level-of-Detail hierarchy that dynamically modulates Gaussian visibility for smooth, alias-free cross-scale rendering. Experiments on Mip-NeRF360 and Tanks\&Temples demonstrate that GaussianZoom achieves superior perceptual quality, multi-view consistency, and robustness under extreme magnification, establishing a strong baseline for generative zoom-in 3D scene reconstruction.
  </details>  
  Keywords: 3d scene reconstruction  
- **[Who Generated This 3D Asset? Learning Source Attribution for Generative 3D Models](https://arxiv.org/abs/2605.18132v1)**  
  Authors: Sihan Ma, Siyuan Liang, Dacheng Tao  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2605.18132v1.pdf)  
  <details><summary>Abstract</summary>

  Generative 3D models are deployed in gaming, robotics, and immersive creation, making source attribution critical: given a 3D asset, can we identify whether and which generative model created it? This problem faces two core challenges: dispersed attribution signals, where 3D fingerprints are distributed across multi-view, geometric, and frequency-domain cues; and realistic deployment constraints, where scarce labels, degraded prompts, and mixed real/synthetic assets undermine attribution reliability. To systematically study this problem, we construct, to the best of our knowledge, the first passive source attribution benchmark for modern generated assets, covering 22 representative 3D generators under standard, few-shot, and realistic deployment protocols. Based on this benchmark, we find that generative 3D models leave two types of stable fingerprints: cross-view inconsistency and structural artifacts reflected in geometric statistics and frequency-domain cues. To capture these dispersed signals, we propose a hierarchical multi-view multi-modal Transformer that fuses appearance, geometric, and frequency-domain features within each view...
  </details>  
  Keywords: 3d generator  
- **[Efficient 3D Content Reconstruction and Generation](https://arxiv.org/abs/2605.18052v1)**  
  Authors: Jiahao Li  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2605.18052v1.pdf)  
  <details><summary>Abstract</summary>

  Automatic 3D content creation seeks to replace labor-intensive modeling and scanning pipelines with systems that can synthesize or recover 3D assets directly from text or images. Its applications span video games, virtual reality, robotics, and simulation, enabling rapid asset prototyping, diverse interactive world generation, and efficient 3D data collection for training foundation models. Contemporary solutions largely follow two complementary paradigms: (i) text- or image-to-3D generation, which learns priors over 3D geometry and appearance to create novel assets from natural language or a single view image; and (ii) 3D reconstruction, which estimates camera poses and geometry from RGB images. This thesis advances both directions. On the generation side, I introduce Instant3D, which combines multi-view diffusion with feed-forward sparse-view 3D reconstruction to produce high-quality assets in 5-20 seconds. On the reconstruction side, I develop FastMap, a structure-from-motion pipeline that achieves up to 10x speedup over prior state-of-the-art by using first-order optimization with...
  </details>  
  Keywords: novel view synthesis, image-to-3d  
- **[PanoWorld: A Generative Spatial World Model for Consistent Whole-House Panorama Synthesis](https://arxiv.org/abs/2605.17916v2)**  
  Authors: Jinrang Jia, Zhenjia Li, Yijiang Hu, Yifeng Shi  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2605.17916v2.pdf) | [![Project](https://img.shields.io/badge/-Project-blue)](https://jjrcn.github.io/PanoWorld-project-home)  
  <details><summary>Abstract</summary>

  Generating a consistent whole-house VR tour from a floorplan and style reference requires both photorealistic panoramas and cross-view spatial coherence. Pure 2D generators produce appealing single panoramas but re-imagine geometry and materials when the viewpoint changes, whereas monolithic 3D generation becomes expensive and loses fine texture at multi-room scale. We introduce PanoWorld, a generative spatial world model that treats whole-house synthesis as autoregressive generation of node-based 360-degree panoramas, matching the discrete navigation used by real VR tour products. PanoWorld uses a floorplan-derived 3D shell as a global geometric proxy and a dynamic 3D Gaussian Splatting cache as renderable spatial memory. A feed-forward panoramic LRM designed for metric-scale multi-room 360-degree inputs lifts generated panoramas into local 3DGS updates, while Room-aware Group Attention suppresses cross-room feature interference. A topology-aware progressive caching strategy fuses these local updates without repeatedly reconstructing the full history. By decoupling shell-based geometry guidance from cache-rendered visual memory, PanoWorld...
  </details>  
- **[Accelerating 3D Gaussian Splatting using Tensor Cores](https://arxiv.org/abs/2605.17855v2)**  
  Authors: Sheng Li, Yang Sui, Yue Wu, Zhuoran Song, Bo Yuan, Xulong Tang, Yue Dai  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2605.17855v2.pdf)  
  <details><summary>Abstract</summary>

  3D Gaussian Splatting (3DGS) has become a leading technique for real-time neural rendering and 3D scene reconstruction, but its rendering cost remains too high for many latency-sensitive scenarios. In particular, the rasterization stage in 3DGS dominates end-to-end rendering time, during which the renderer repeatedly evaluates each Gaussian's contribution to each covered pixel, making this stage compute-bound. At the same time, modern GPUs provide high-throughput Tensor Cores for low-precision matrix operations, yet existing 3DGS systems execute rasterization entirely on CUDA cores and leave Tensor Cores idle. We find that 3DGS rendering can be executed in FP16 with negligible quality degradation, suggesting a promising opportunity for Tensor Core acceleration. However, exploiting Tensor Cores for 3DGS is non-trivial because rasterization does not naturally match their execution model. Existing 3DGS rasterization is expressed as irregular per-pixel scalar operations, whereas Tensor Cores require dense, regular, and reuse-rich matrix workloads. Moreover, conventional tile-by-tile execution fails to...
  </details>  
  Keywords: 3d scene reconstruction  
- **[Fine-tuning Pocket-Aware Diffusion Models via Denoising Policy Optimization](https://arxiv.org/abs/2605.17693v1)**  
  Authors: Yuan Xue, Daniel Kudenko, Megha Khosla  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2605.17693v1.pdf) | [![GitHub](https://img.shields.io/github/stars/xy9485/DePPA?style=social)](https://github.com/xy9485/DePPA)  
  <details><summary>Abstract</summary>

  Structure-based drug design has been accelerated by pocket-aware 3D generative models, yet most methods primarily fit the training distribution and may fall short of satisfying multiple properties required in real-world therapeutic drug discovery. Recently, increasing attention has focused on structure-based molecule optimization (SBMO), which targets fine-grained control over multiple specified molecular properties. In this paper, we present DEPPA, a novel SBMO approach building upon Denoising Diffusion Policy Optimization for fine-tuning a pre-trained pocket-aware diffusion model via reinforcement learning. DEPPA enables optimization over multiple properties, including binding affinity, drug-likeness, synthesizability and diversity. We formulate the reverse denoising process of the pretrained pocket-aware diffusion model as a multi-step Markov Decision Process, where the desired properties that serve as reward signals are evaluated on the final generated ligand molecules. DEPPA incorporates a coarse denoising scheduler during the RL fine-tuning to achieve efficient and effective molecule optimization. Experimental results on the CrossDocked2020 benchmark demonstrate...
  </details>  
- **[Mamba-VGGT: Persistent Long-Sequence Video Geometry Grounded Transformer via External Sliding Window Mamba Memory](https://arxiv.org/abs/2605.17478v1)**  
  Authors: Tianchen Deng, Zhenxiang Xiong, Nailin Wang, Fangjinhua Wang, Jiuming Liu, Jianfei Yang, Hesheng Wang  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2605.17478v1.pdf)  
  <details><summary>Abstract</summary>

  Visual Geometry Grounded Transformers (VGGT) have set new benchmarks in high-fidelity 3D scene reconstruction. However, as the sequence length increases, these models suffer from catastrophic geometric forgetting and accumulation drift, primarily due to the quadratic complexity of global attention which necessitates truncated temporal windows. To overcome the resulting geometric drift, we present Mamba-VGGT, an enhanced VGGT framework capable of persistent long-range reasoning. Our key contribution is a Sliding Window Mamba (SWM) memory module that maintains an explicit external memory token across temporal windows. This module leverages selective state-space modeling to distill and propagate global geometric priors, effectively bypassing the memory constraints of traditional transformers. To integrate these long-term temporal cues without disrupting the highly optimized spatial features of the pre-trained VGGT, we propose a Zero-Init Spatial Memory Injector. Utilizing zero-convolutional layers, this injector adaptively fuses persistent memory into the patch token stream, ensuring structural stability and seamless feature alignment. Extensive...
  </details>  
  Keywords: 3d scene reconstruction  
- **[DreamEdit3D: Personalization of Multi-View Diffusion Models for 3D Editing](https://arxiv.org/abs/2605.16990v1)**  
  Authors: Jinxin Ai, Matthias Nießner, Ziya Erkoç  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2605.16990v1.pdf)  
  <details><summary>Abstract</summary>

  While 2D diffusion models have achieved remarkable success in identity-preserving personalization, extending this capability to 3D assets remains a significant challenge due to the complexities of multi-view consistency and spatial control. Inspired by these 2D advancements, we present a novel personalization method for text-guided 3D editing that enables compositional, object-level control through natural language. Given a 3D input, we render orthogonal views and extract object-level segmentation masks to isolate semantic components. We then learn distinct token embeddings for each component through a tailored two-phase optimization strategy: multi-view textual inversion with attention alignment, followed by full fine-tuning of multi-view diffusion model. During inference, these disentangled tokens seamlessly compose with editing prompts to generate multi-view consistent images, which are subsequently lifted into high-fidelity textured 3D meshes. Extensive evaluations across diverse editing scenarios demonstrate that our method successfully transfers the flexibility of 2D personalization to 3D, achieving state-of-the-art edit faithfulness and identity preservation...
  </details>  
- **[DecoRec: Decomposed 3D Scene Reconstruction from Single-View Images via Object-Level Diffusion](https://arxiv.org/abs/2605.16807v1)**  
  Authors: Yuhan Ping, Yuan Liu, Xiaoxiao Long, Peng Wang, Junhui Hou, Jianyi Zheng, Jia Pan, Xin Li, Cheng Lin  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2605.16807v1.pdf)  
  <details><summary>Abstract</summary>

  In this paper, we introduce \textit{DecoRec}, a novel system designed to elevate single-view 2D images to a decomposed 3D scene mesh. Current methods for single-view scene reconstruction typically rely on object retrieval or the regression of coarse 3D voxels or surfaces, leading to inaccuracies in capturing the appearance and geometry of the input image. The lack of high-quality large-scale scene-level datasets further complicates direct 3D scene generation from single-view images. To achieve high-quality 3D scene generation from a single-view image, DecoRec takes advantage of recent diffusion-based single-view object reconstruction methods to reconstruct individual objects separately. Subsequently, a refinement pipeline is proposed to effectively merge these reconstructed objects, enhancing appearance and geometry through a differentiable rendering technique and diffusion-guided refinement. Our results demonstrate that DecoRec facilitates high-quality single-view scene reconstruction in both geometry and novel synthesis, offering significant benefits for downstream applications like room interior design.
  </details>  
  Keywords: 3d scene reconstruction  
- **[3DPhysVideo: Consistency-Guided Flow SDE for Video Generation via 3D Scene Reconstruction and Physical Simulation](https://arxiv.org/abs/2605.16795v1)**  
  Authors: Hwidong Kim, Yunho Kim, Tae-Kyun Kim  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2605.16795v1.pdf)  
  <details><summary>Abstract</summary>

  Video generative models have made remarkable progress, yet they often yield visual artifacts that violate grounding in physical dynamics. Recent works such as PhysGen3D tackle single image-to-3D physics through mesh reconstruction and Physically-Based Rendering, but challenges remain in modeling fluid dynamics, multi-object interactions and photorealism. This work introduces 3DPhysVideo, a novel training-free pipeline that generates physically realistic videos from a single image. We repurpose an off-the-shelf video model for two stages. First, we use it as a novel view synthesizer to reconstruct complete 360-degree 3D scene geometry by guiding the image-to-video (I2V) flow model with rendered point clouds. Second, after applying physics solvers to this geometry, the physically simulated point cloud is used to guide the same I2V flow model to synthesize final, high-quality videos. Consistency-Guided Flow SDE, which decomposes the predicted velocity of the I2V flow model into denoising and consistency bias, enforces consistency to the conditional inputs, allowing...
  </details>  
  Keywords: 3d scene reconstruction, image-to-3d  
- **[EVA01: Unified Native 3D Understanding and Generation via Mixture-of-Transformers](https://arxiv.org/abs/2605.16745v1)**  
  Authors: Zongyuan Yang, Mingjing Yi, Wanli Ma, Chenzhuo Fan, Bocheng Li, Baolin Liu, Yuke Lou, Yingde Song, Yongping Xiong, Zhengdong Guo, Shimu Wang  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2605.16745v1.pdf) | [![Project](https://img.shields.io/badge/-Project-blue)](https://www.seeles.ai/research/pages/EVA01)  
  <details><summary>Abstract</summary>

  This paper addresses the challenge of integrating 3D meshes as a native modality within Multimodal Large Language Models (MLLMs). Diffusion-based large reconstruction models decouple semantic understanding from geometric reasoning, operating as stateless reconstructors conditioned on dense 2D pixel priors. Recent MLLM-based methods treat the 3D modality as an external output rather than a native component of the multimodal sequence, making incremental adaptations without a systematic analysis of how geometric manifolds align with MLLM feature spaces. We introduce EVA01, a unified framework that extends the modality boundary of MLLMs to natively incorporate 3D mesh understanding, generation, and context-aware editing. Built upon a Mixture-of-Transformers (MoT) architecture, EVA01 decouples the model into a pre-trained Understanding Expert ($E_{\mathrm{und}}$) and a structurally mirrored Generation Expert ($E_{\mathrm{gen}}$), coupled through shared global self-attention with hard modality routing. This design aligns the semantic latent space of the MLLM backbone with the geometric manifold, enabling direct transfer of multimodal...
  </details>  
  Keywords: text-to-3d  
- **[Robust Prior-Guided Segmentation for Editable 3D Gaussian Splatting](https://arxiv.org/abs/2605.16065v1)**  
  Authors: Raushan Joshi, Jean-Yves Guillemaut  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2605.16065v1.pdf)  
  <details><summary>Abstract</summary>

  3D Gaussian Splatting (3D-GS) enables real-time 3D scene reconstruction but lacks robust segmentation for editing tasks such as object removal, extraction, and recoloring. Existing approaches that lift 2D segmentations to the 3D domain suffer from view inconsistencies and coarse masks. In this paper, we propose a novel framework that leverages the Segment Anything Model High Quality (SAM-HQ) to generate accurate 2D masks, addressing the limitations of the standard SAM in boundary fidelity and fine-structure preservation. To achieve robust 3D segmentation of any target object in a given scene, we introduce a prior-guided label reassignment method that assigns labels to 3D Gaussians by enforcing multiview consistency with learned priors. Our approach achieves state-of-the-art segmentation accuracy and enables interactive, real-time object editing while maintaining high visual fidelity. Qualitative results demonstrate superior boundary preservation and practical utility in Virtual Reality (VR) and robotics, advancing 3D scene editing.
  </details>  
  Keywords: 3d scene reconstruction  
- **[3DEditSafe: Defending 3D Editing Pipelines from Unsafe Generation](https://arxiv.org/abs/2605.15398v1)**  
  Authors: Nicole Meng, Zheyuan Liu, Meng Jiang, Yingjie Lao  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2605.15398v1.pdf)  
  <details><summary>Abstract</summary>

  Recent advances in 3D generative editing, particularly pipelines based on 3D Gaussian Splatting (3DGS), have achieved high-fidelity, multi-view-consistent scene manipulation from text prompts. However, we find that these pipelines also introduce new safety risks when unsafe prompts produce edits that are propagated and optimized across views. In this work, we study unsafe generation in 3D editing pipelines and show that such behavior can lead to coherent, undesirable Not-Safe-For-Work (NSFW) content in the final 3D representation. To address this, we propose 3DEditSafe, a safety-regularized 3D editing framework that constrains unsafe semantic propagation during optimization. 3DEditSafe combines generation-stage safety guidance with rendered-view 3D safety regularization, safe semantic projection, residue suppression, and mask-aware preservation to steer optimization away from unsafe editing directions. We evaluate our approach on EditSplat scenes using an object-compatible unsafe prompt benchmark and show that 2D safety guidance alone is not consistently sufficient to prevent unsafe 3D edits. 3DEditSafe reduces...
  </details>  
- **[Articraft: An Agentic System for Scalable Articulated 3D Asset Generation](https://arxiv.org/abs/2605.15187v1)**  
  Authors: Matt Zhou, Ruining Li, Xiaoyang Lyu, Zhaomou Song, Zhening Huang, Chuanxia Zheng, Christian Rupprecht, Andrea Vedaldi, Shangzhe Wu  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2605.15187v1.pdf)  
  <details><summary>Abstract</summary>

  A bottleneck in learning to understand articulated 3D objects is the lack of large and diverse datasets. In this paper, we propose to leverage large language models (LLMs) to close this gap and generate articulated assets at scale. We reduce the problem of generating an articulated 3D asset to that of writing a program that builds it. We then introduce a new agentic system, Articraft, that writes such programs automatically. We design a programmatic interface and harness to help the LLM do so effectively. The LLM writes code against a domain-specific SDK for defining parts, composing geometry, specifying joints, and writing tests to validate the resulting assets. The harness exposes a restricted workspace and interface to the LLM, validates the resulting assets, and returns structured feedback. In this way, the LLM is not distracted by details such as authoring a URDF file or managing a complex software environment. We show...
  </details>  
  Keywords: 3d asset generation  
- **[VGGT-Edit: Feed-forward Native 3D Scene Editing with Residual Field Prediction](https://arxiv.org/abs/2605.15186v2)**  
  Authors: Kaixin Zhu, Yiwen Tang, Yifan Yang, Renrui Zhang, Bohan Zeng, Ziyu Guo, Ruichuan An, Zhou Liu, Qizhi Chen, Delin Qu, Jaehong Yoon, Wentao Zhang  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2605.15186v2.pdf) | [![Project](https://img.shields.io/badge/-Project-blue)](https://chriszkxxx.github.io/VGGT-Edit)  
  <details><summary>Abstract</summary>

  High-quality 3D scene reconstruction has recently advanced toward generalizable feed-forward architectures, enabling the generation of complex environments in a single forward pass. However, despite their strong performance in static scene perception, these models remain limited in responding to dynamic human instructions, which restricts their use in interactive applications. Existing editing methods typically rely on a 2D-lifting strategy, where individual views are edited independently and then lifted back into 3D space. This indirect pipeline often leads to blurry textures and inconsistent geometry, as 2D editors lack the spatial awareness required to preserve structure across viewpoints. To address these limitations, we propose VGGT-Edit, a feed-forward framework for text-conditioned native 3D scene editing. VGGT-Edit introduces depth-synchronized text injection to align semantic guidance with the backbone's spatial poses, ensuring stable instruction grounding. This semantic signal is then processed by a residual transformation head, which directly predicts 3D geometric displacements to deform the scene while...
  </details>  
  Keywords: 3d scene reconstruction  
- **[Sat3DGen: Comprehensive Street-Level 3D Scene Generation from Single Satellite Image](https://arxiv.org/abs/2605.14984v1)**  
  Authors: Ming Qian, Zimin Xia, Changkun Liu, Shuailei Ma, Wen Wang, Zeran Ke, Bin Tan, Hang Zhang, Gui-Song Xia  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2605.14984v1.pdf) | [![GitHub](https://img.shields.io/github/stars/qianmingduowan/Sat3DGen?style=social)](https://github.com/qianmingduowan/Sat3DGen)  
  <details><summary>Abstract</summary>

  Generating a street-level 3D scene from a single satellite image is a crucial yet challenging task. Current methods present a stark trade-off: geometry-colorization models achieve high geometric fidelity but are typically building-focused and lack semantic diversity. In contrast, proxy-based models use feed-forward image-to-3D frameworks to generate holistic scenes by jointly learning geometry and texture, a process that yields rich content but coarse and unstable geometry. We attribute these geometric failures to the extreme viewpoint gap and sparse, inconsistent supervision inherent in satellite-to-street data. We introduce Sat3DGen to address these fundamental challenges, which embodies a geometry-first methodology. This methodology enhances the feed-forward paradigm by integrating novel geometric constraints with a perspective-view training strategy, explicitly countering the primary sources of geometric error. This geometry-centric strategy yields a dramatic leap in both 3D accuracy and photorealism. For validation, we first constructed a new benchmark by pairing the VIGOR-OOD test set with high-resolution DSM...
  </details>  
  Keywords: image-to-3d  
- **[TOPOS: High-Fidelity and Efficient Industry-Grade 3D Head Generation](https://arxiv.org/abs/2605.14594v1)**  
  Authors: Bojun Xiong, Zoubin Bi, Xinghui Peng, Yunmu Wang, Junchen Deng, Jun Liang, Jing Li, Bowen Cai, Huan Fu  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2605.14594v1.pdf)  
  <details><summary>Abstract</summary>

  High-fidelity 3D head generation plays a crucial role in the film, animation and video game industries. In industrial pipelines, studios typically enforce a fixed reference topology across all head assets, as such a clean and uniform topology is a prerequisite for production-level rigging, skinning and animation. In this paper, we present TOPOS, a framework tailored for single image conditioned 3D head generation that jointly recovers geometry and appearance under such an industry-standard topology. In contrast to general 3D generative models which produce triangle meshes with inconsistent topology and numerous vertices, hindering semantic correspondence and asset-level reuse, TOPOS generates head meshes with a fixed, studio-style topology, enabling consistent vertex-level correspondence across all generated heads. To model heads under this unified topology, we proposed a novel variational autoencoder structure, termed TOPOS-VAE. Inspired by multi-model large language models (MLLMs), our TOPOS-VAE leverages the Perceiver Resampler to convert input pointclouds sampled from head meshes...
  </details>  
  Keywords: rigging  
- **[Fast and Robust Mesh Simplification for Generated and Real-World 3D Assets](https://arxiv.org/abs/2605.14029v1)**  
  Authors: Kunal Bhosikar, Preet Savalia, Lokender Tiwari, Brojeshwar Bhowmick  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2605.14029v1.pdf)  
  <details><summary>Abstract</summary>

  The rapid growth of 3D content from modern reconstruction and generative pipelines, such as neural rendering and large-scale 3D asset generation, has led to an abundance of dense, noisy, and often non-manifold meshes. While these representations achieve high visual fidelity, their complexity poses significant challenges for downstream applications in simulation, AR/VR, and scientific computing, where efficient and reliable geometry is essential. This necessitates mesh simplification methods that are not only fast and robust to "in-the-wild" inputs, but also capable of preserving fine geometric structures and high-quality appearance. In this paper, we propose Feature-Aware Quadric Error Metric (FA-QEM), a comprehensive mesh simplification pipeline designed for modern 3D assets. Our approach introduces a novel multi-term quadric error formulation that jointly encodes geometric deviation, boundary curvature, and surface normal consistency, enabling optimal vertex placement that preserves sharp features even under aggressive simplification. Furthermore, we show that high-fidelity geometric simplification significantly improves downstream appearance...
  </details>  
  Keywords: 3d asset generation  
- **[Generative Texture Diversification of 3D Pedestrians for Robust Autonomous Driving Perception](https://arxiv.org/abs/2605.13755v1)**  
  Authors: Arka Bhowmick, Enes Ozeren, Ahmed Abdullah, Oliver Wasenmuller  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2605.13755v1.pdf)  
  <details><summary>Abstract</summary>

  In recent years, autonomous driving has significantly in creased the demand for high-quality data to train 2D and 3D perception models for safety-critical scenarios. Real world datasets struggle to meet this demand as require ments continuously evolve and large-scale annotated data collection remains costly and time-consuming making syn thetic data a scalable, practical and controllable alterna tive. Pedestrian detection is among the most safety-critical tasks in autonomous driving. In this paper, we propose a simple yet effective method for scaling variability in 3D pedestrian assets for synthetic scene generation. Starting from a single 3D base asset, we generate multiple distinct pedestrian instances by synthesizing diverse facial textures and identity-level appearance variations using StyleGAN2 and automatically mapping them onto 3D meshes. This ap proach enables scalable appearance-level asset diversifica tion without requiring the design of new geometries for each instance. Using the assets, we construct synthetic datasets and study the impact...
  </details>  
  Keywords: texture synthesis  
- **[Texture Regenerating and Grafting Using Genome-Driven Neural Cellular Automata](https://arxiv.org/abs/2605.13630v1)**  
  Authors: Mirela-Magdalena Catrina, Ioana Cristina Plajer, Alexandra Băicoianu  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2605.13630v1.pdf)  
  <details><summary>Abstract</summary>

  This study significantly advances multi-texture synthesis using Neural Cellular Automata (NCAs) by introducing a novel training methodology that enables robust self-regeneration of textures in damaged regions. This inherent healing mechanism, essential for dynamic and adaptive systems, extends beyond traditional computer graphics applications, highlighting the fundamental self-organizing properties of NCAs. Furthermore, we present a versatile grafting technique, enabling the seamless combination of distinct textures. This is achieved efficiently during the inference phase, without requiring specialized retraining, through precise initialization of the NCA's genome channels. Our findings demonstrate the generation of high-quality, complex textures with fluid transitions, showcasing a powerful and efficient paradigm for dynamic texture composition and self-repair in autonomous systems.
  </details>  
  Keywords: texture synthesis  
- **[Rigel3D: Rig-aware Latents for Animation-Ready 3D Asset Generation](https://arxiv.org/abs/2605.13129v1)**  
  Authors: Nikitas Chatzis, Marios Loizou, Evangelos Kalogerakis  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2605.13129v1.pdf)  
  <details><summary>Abstract</summary>

  Recent 3D generative models can synthesize high-quality assets, but their outputs are typically static: they lack the skeletal rigs, joint hierarchies, and skinning weights required for animation. This limits their use in games, film, simulation, virtual agents, and embodied AI, where assets must not only look plausible but also move plausibly. We introduce Rigel3D, a generative method for animation-ready 3D assets represented as rigged meshes. Unlike post-hoc auto-rigging methods that attach rigs to completed shapes, our method jointly models geometry and rig structure through coupled surface and skeleton structured latent representations. A rig-aware autoencoder decodes these representations into mesh geometry, skeleton topology, joint coordinates, and skinning weights, while a two-stage latent generative model synthesizes both surface and skeleton representations for image-conditioned generation. To support downstream animation workflows, we further introduce an open-vocabulary joint labeling module that embeds generated joints into a shared vision-language space, enabling correspondence to arbitrary retargeting templates....
  </details>  
  Keywords: 3d asset generation, rigging  
- **[GTA: Advancing Image-to-3D World Generation via Geometry Then Appearance Video Diffusion](https://arxiv.org/abs/2605.12957v1)**  
  Authors: Hanxin Zhu, Cong Wang, Peiyan Tu, Jiayi Luo, Tianyu He, Xin Jin, Zhibo Chen  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2605.12957v1.pdf) | [![Project](https://img.shields.io/badge/-Project-blue)](https://hanxinzhu-lab.github.io/GTA)  
  <details><summary>Abstract</summary>

  Recent developments in generative models and large-scale datasets have substantially advanced 3D world generation, facilitating a broad range of domains including spatial intelligence, embodied intelligence, and autonomous driving. While achieving remarkable progress, existing approaches to 3D world generation typically prioritize appearance prediction with limited modeling of the underlying geometry, leading to issues such as unreliable scene structure estimation and degraded cross-view consistency. To address these limitations, motivated by the coarse-to-fine nature of human visual perception, we propose GTA, a novel image-to-3D world generation method following a Geometry-Then-Appearance paradigm. Specifically, given a single input image, to improve the structural fidelity of synthesized 3D scenes, GTA adopts a two-stage framework with two dedicated video diffusion models, which first generate coarse geometric structure from novel viewpoints and then synthesize fine-grained appearance conditioned on the predicted geometry. To further enhance cross-view appearance consistency, we introduce a random latent shuffle strategy during the training process,...
  </details>  
  Keywords: image-to-3d  
- **[3D Primitives are a Spatial Language for VLMs](https://arxiv.org/abs/2605.12586v1)**  
  Authors: Junze Liu, Kun Qian, Florian Dubost, Kai Zhong, Arvind Srinivasan, Nan Chen, Anping Wang, Sam Zhang, Alejandro Mottini, Qingjun Cui, Tian Wang  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2605.12586v1.pdf)  
  <details><summary>Abstract</summary>

  Vision-language models (VLMs) exhibit a striking paradox: they can generate executable code that reconstructs a 3D scene from geometric primitives with correct object counts, classes, and approximate positions, yet the same models fail at simpler spatial questions on the same image. We show that 3D geometric primitives (cubes, spheres, cylinders, expressed in executable code) serve as a powerful intermediate representation for spatial understanding, and exploit this through three contributions. First, we introduce \textbf{\textsc{SpatialBabel}}, a benchmark evaluating fourteen VLMs on primitive-based 3D scene reconstruction across six \emph{scene-code languages} (programming languages and declarative formats for 3D primitive scenes), revealing that a single model's object-detection F1 can vary by up to $5.7\times$ across languages. Second, we propose \textbf{Code-CoT} (Code Chain-of-Thought), a training-free inference strategy that routes spatial reasoning through primitive-based code generation. Code-CoT lifts the SpatialBabel-QA-Score by up to $+6.4$\% on primitive scenes and real-photo CV-Bench-3D accuracy by $+5.0$\% for VLMs with strong...
  </details>  
  Keywords: 3d scene reconstruction  
- **[Optimizing 4D Wires for Sparse 3D Abstraction](https://arxiv.org/abs/2605.11977v1)**  
  Authors: Dong-Yi Wu, Tong-Yee Lee  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2605.11977v1.pdf)  
  <details><summary>Abstract</summary>

  We present a unified framework for 3D geometric abstraction using a single continuous 4D wire, parameterized as a B-spline with spatial coordinates and variable width $(x,y,z,w)$. Existing approaches typically represent shapes as collections of many independent curve segments, which often leads to fragmented structures and limited physical realizability. In contrast, we show that a single continuous spline is sufficiently expressive to capture complex volumetric forms while enforcing global topological coherence. By imposing continuity, our method transforms 3D sketching from a local density-accumulation process into a global routing problem, providing a strong inductive bias toward cleaner aesthetics and improved structural coherence. To enable gradient-based optimization, we introduce a differentiable rendering pipeline that efficiently rasterizes variable-width curves with bounded projection error. This formulation supports robust optimization using modern guidance signals such as Score Distillation Sampling (SDS) or CLIP. We demonstrate applications including image-to-3D abstraction, multi-view wire art generation, and differentiable stylized surface...
  </details>  
  Keywords: image-to-3d  
- **[Pixal3D: Pixel-Aligned 3D Generation from Images](https://arxiv.org/abs/2605.10922v1)**  
  Authors: Dong-Yang Li, Wang Zhao, Yuxin Chen, Wenbo Hu, Meng-Hao Guo, Fang-Lue Zhang, Ying Shan, Shi-Min Hu  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2605.10922v1.pdf) | [![Project](https://img.shields.io/badge/-Project-blue)](https://ldyang694.github.io/projects/pixal3d)  
  <details><summary>Abstract</summary>

  Recent advances in 3D generative models have rapidly improved image-to-3D synthesis quality, enabling higher-resolution geometry and more realistic appearance. Yet fidelity, which measures pixel-level faithfulness of the generated 3D asset to the input image, still remains a central bottleneck. We argue this stems from an implicit 2D-3D correspondence issue: most 3D-native generators synthesize shape in canonical space and inject image cues via attention, leaving pixel-to-3D associations ambiguous. To tackle this issue, we draw inspiration from 3D reconstruction and propose Pixal3D, a pixel-aligned 3D generation paradigm for high-fidelity 3D asset creation from images. Instead of generating in a canonical pose, Pixal3D directly generates 3D in a pixel-aligned way, consistent with the input view. To enable this, we introduce a pixel back-projection conditioning scheme that explicitly lifts multi-scale image features into a 3D feature volume, establishing direct pixel-to-3D correspondence without ambiguity. We show that Pixal3D is not only scalable and capable of...
  </details>  
  Keywords: image-to-3d  
- **[CADBench: A Multimodal Benchmark for AI-Assisted CAD Program Generation](https://arxiv.org/abs/2605.10873v1)**  
  Authors: Anna C. Doris, Jacob Thomas Sony, Ghadi Nehme, Era Syla, Amin Heyrani Nobari, Faez Ahmed  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2605.10873v1.pdf) | [![Project](https://img.shields.io/badge/-Project-blue)](https://huggingface.co/datasets/DeCoDELab/CADBench.) | [![Dataset](https://img.shields.io/badge/-Dataset-orange)](https://huggingface.co/datasets/DeCoDELab/CADBench)  
  <details><summary>Abstract</summary>

  Recovering editable CAD programs from images or 3D observations is central to AI-assisted design, but progress is difficult to measure because existing evaluations are fragmented across datasets, modalities, and metrics. We introduce CADBench, a unified benchmark for multimodal CAD program generation. CADBench contains 18,000 evaluation samples spanning six benchmark families derived from DeepCAD, Fusion 360, ABC, MCB, and Objaverse; five input modalities including clean meshes, noisy meshes, single-view renders, photorealistic renders, and multi-view renders; and six metrics covering geometric fidelity, executability, and program compactness. STEP-based families are stratified by B-rep face count and all families are diversity-sampled to support controlled analysis across complexity and object variation. We benchmark eleven CAD-specialized and general-purpose vision-language systems, generating more than 1.4 million CAD programs. Under idealized inputs, specialized mesh-to-CAD models substantially outperform code-generating VLMs, which remain far from reliable CAD program reconstruction. CADBench further reveals three recurring failure modes: reconstruction quality degrades with...
  </details>  
- **[Beyond Spatial Compression: Interface-Centric Generative States for Open-World 3D Structure](https://arxiv.org/abs/2605.10438v1)**  
  Authors: Xiang Chen, Alexander Binder  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2605.10438v1.pdf)  
  <details><summary>Abstract</summary>

  Current 3D tokenizers largely treat representation as spatial compression: compact codes reconstruct surface geometry, but leave component ownership and attachment validity implicit. In open-world assets with intersecting components, noisy topology, and weak canonical structure, this creates a representation mismatch: local shape, component identity, and assembly relations become entangled in a latent stream and are not natively addressable during decoding. We formulate an alternative view, interface-centric generative states, in which tokenization constructs an operational state rather than a passive compressed code. The state exposes local geometry, component ownership, and attachment validity as variables that can be queried, constrained, and repaired during decoding. We instantiate this formulation with Component-Conditioned Canonical Local Tokens (C2LT-3D), factorizing representation into canonical local geometry, partition-conditioned context, and relational seam variables. Each factor targets a distinct failure mode of compression-centric tokens: pose leakage, cross-component interference, or invalid local attachment. This exposed state supports attachment validation, latent structural repair,...
  </details>  
- **[On the Generation and Mitigation of Harmful Geometry in Image-to-3D Models](https://arxiv.org/abs/2605.09606v1)**  
  Authors: Yule Liu, Yilong Yang, Jiale Teng, Hanze Jia, Zeren Luo, Jingyi Zheng, Zifan Peng, Ke Li, Yifan Liao, Zhen Sun, Jiaheng Wei, Yang Liu, Zhuo Ma, Xinlei He  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2605.09606v1.pdf)  
  <details><summary>Abstract</summary>

  Recent advances in image-to-3D models have significantly improved the fidelity and accessibility of 3D content creation. Such a powerful reconstruction capability that enables creative design can also be misused by the adversary to generate harmful geometries, which can be further fabricated via 3D printers and pose real-world risks. However, such risks are largely underexplored: it remains unclear how well current image-to-3D models can produce these harmful geometries, and whether existing safeguards can reliably prevent such generation. To fill this gap, we conduct a systematic measurement study of harmful geometry generation and mitigation. We first describe this risk through three kinds of unsafe categories: direct-use physical hazards, risky templates or components, and deceptive replicas. Each category is instantiated with representative objects. We evaluate both open-source and commercial image-to-3D models under original, degraded, viewpoint-shifted, and semantically camouflaged inputs. We consider different evaluation metrics, including geometric validity, multi-view VLM-based semantic scoring, targeted human...
  </details>  
  Keywords: image-to-3d  
- **[SimWorld Studio: Automatic Environment Generation with Evolving Coding Agent for Embodied Agent Learning](https://arxiv.org/abs/2605.09423v2)**  
  Authors: Haoqiang Kang, Xiaokang Ye, Yuhan Liu, Siddhant Hitesh Mantri, Lingjun Mao, James Fleming, Drishti Regmi, Lianhui Qin  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2605.09423v2.pdf)  
  <details><summary>Abstract</summary>

  LLM/VLM-based digital agents have advanced rapidly thanks to scalable sandboxes for coding, web navigation, and computer use, which provide rich interactive training grounds. In contrast, embodied agents still lack abundant, diverse, and automatically generated 3D environments for interactive learning. Existing embodied simulators rely on manually crafted scenes or procedural templates, while recent LLM-based 3D generation systems mainly produce static scenes rather than deployable environments with verifiable tasks and standard learning interfaces. We introduce SimWorld Studio, an open-source platform built on Unreal Engine 5 for generating evolving embodied learning environments. At its core is SimCoder, a tool/skill-augmented coding agent that writes and executes engine-level code to construct physically grounded 3D worlds from language/image instructions. SimCoder self-evolves by using verifier feedback (e.g., compilation errors, physics checks, VLM critiques) to revise environments and autonomously add reusable tools and skills to its library. Generated worlds are exported as Gym-style environments for embodied agent learning....
  </details>  
- **[Noise-Started One-Step Real-World Super-Resolution via LR-Conditioned SplitMeanFlow and GAN Refinement](https://arxiv.org/abs/2605.09328v1)**  
  Authors: Wei Zhu, Kai Zhang, Yu Zheng, Lei Luo, Yong Guo, Jian Yang  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2605.09328v1.pdf)  
  <details><summary>Abstract</summary>

  Pre-trained text-to-image (T2I) diffusion models have shown strong potential for real-world image super-resolution (Real-ISR), owing to their noise-started generation process that enables realistic texture synthesis and captures the one-to-many nature of super-resolution. However, diffusion-based Real-ISR methods still face a fundamental efficiency-quality trade-off. Multi-step methods generate high-quality results by iteratively denoising random Gaussian noise under LR conditioning, but suffer from slow sampling. Recent one-step methods greatly improve efficiency, yet they typically replace noise-started generation with direct LR-to-HR restoration, which weakens stochasticity and limits realistic detail synthesis. To address this issue, we propose SMFSR, a noise-started one-step Real-ISR framework via LR-conditioned SplitMeanFlow and GAN refinement. SMFSR preserves the random-noise starting point of diffusion models and learns a direct noise-to-HR mapping conditioned on the LR image. To this end, Interval Splitting Consistency distills the multi-step generative trajectory into a single average-velocity prediction, enabling efficient one-step generation. To compensate for the reduced opportunity for...
  </details>  
  Keywords: texture synthesis  
- **[An Elastic Shape Variational Autoencoder for Skeleton Pose Trajectories](https://arxiv.org/abs/2605.09231v3)**  
  Authors: Arafat Rahman, Shashwat Kumar, Laura E. Barnes, Anuj Srivastava  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2605.09231v3.pdf)  
  <details><summary>Abstract</summary>

  Deep generative models provide flexible frameworks for modeling complex, structured data such as images, videos, 3D objects, and texts. However, when applied to sequences of human skeletons, standard variational autoencoders (VAEs) often allocate substantial capacity to nuisance factors-such as camera orientation, subject scale, viewpoint, and execution speed-rather than the intrinsic geometry of shapes and their motion. We propose the Elastic Shape - Variational Autoencoder (ES-VAE), a geometry-aware generative model for skeletal trajectories that leverages the transported square-root velocity field (TSRVF) representation on Kendall's shape manifold. This representation inherently removes rigid translations, rotations, and global scaling of shapes, and temporal rate variability of sequences, isolating the underlying shape dynamics. The ES-VAE encoder maps skeletal sequences to a low-dimensional latent space incorporating the Riemannian logarithm map, while the decoder reconstructs sequences using the corresponding exponential map. We demonstrate the effectiveness of ES-VAE on two datasets. First, we analyze skeletal gait cycles to...
  </details>  
- **[Probability-Flow Distillation: Exact Wasserstein Gradient Flow for High-Fidelity 3D Generation](https://arxiv.org/abs/2605.09071v1)**  
  Authors: Rohith Ramanan, A. N. Rajagopalan  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2605.09071v1.pdf)  
  <details><summary>Abstract</summary>

  Score Distillation Sampling (SDS) and its variants have been widely used for text-to-3D generation by distilling 2D image diffusion priors. However, the standard SDS objective is prone to severe mode collapse, frequently yielding over-smoothed and over-saturated results. Although recent advancements, such as Score Distillation via Inversion (SDI), mitigate these artifacts and produce visually sharper models, they ultimately fail to faithfully capture the full target distribution. In this work, we show that the bottleneck limiting the sampling capacity of SDI stems from its reliance on the posterior mean estimator, which is mathematically equivalent to a single-step Euler approximation of the deterministic reverse DDIM trajectory. To address this, we propose a naturally motivated extension termed Probability-Flow Distillation (PFD). We establish that PFD corresponds exactly to a Wasserstein gradient flow, thereby inducing principled distribution-matching dynamics. Finally, we show that PFD can synthesize 3D assets with fine-grained, high-fidelity details and achieve improved quality compared...
  </details>  
  Keywords: text-to-3d  
- **[HairGPT: Strand-as-Language Autoregressive Modeling for Realistic 3D Hairstyle Synthesis](https://arxiv.org/abs/2605.08824v1)**  
  Authors: Haimin Luo, Min Ouyang, Lan Xu, Jingyi Yu  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2605.08824v1.pdf) | [![Project](https://img.shields.io/badge/-Project-blue)](https://haiminluo.github.io/hairgpt)  
  <details><summary>Abstract</summary>

  Hair is a rich medium of visual and cultural expression, yet its digital modeling remains challenging due to the duality of fluidity and structure. Many existing generative approaches rely primarily on continuous diffusion fields, which entangle global topology with local texture and obscure the semantic and structural organization of hairstyles. To address this, we propose HairGPT, a strand-centric framework that treats strands as generative primitives and formulates realistic 3D hairstyle synthesis as a dual-decoupled autoregressive sequence modeling problem. Our method applies spatial decoupling across semantic scalp regions and structural decoupling along a hierarchical strand representation, progressing from global layout to fine-grained style. We further introduce a geometric tokenizer and region-aware semantic annotations to guide strand-level generation, enabling compositional editing, synthesis of rare and complex hairstyles, and adaptation to stylized domains. By aligning generative modeling with the workflow of digital grooming, HairGPT turns hair generation from opaque texture synthesis into a...
  </details>  
  Keywords: texture synthesis  
- **[NeuroGAN-3D: Enhancing Intrinsic Functional Brain Networks via High-Fidelity 3D Generative Super-Resolution](https://arxiv.org/abs/2605.08373v1)**  
  Authors: M. Moein Esfahani, Sepehr Salem Ghahfarokhi, Mohammed Alser, Jingyu Liu, Vince Calhoun  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2605.08373v1.pdf)  
  <details><summary>Abstract</summary>

  Recent advances in neuroimaging have deepened our understanding of the brain's complex functional and structural organization. Among these, functional Magnetic Resonance Imaging (fMRI) - particularly resting-state fMRI (rs-fMRI) - has emerged as a tool for identifying biomarkers of intrinsic brain connectivity and delineating large-scale neural networks. These networks are typically represented as volumetric spatial maps that capture functionally coherent brain regions and reflect individual differences in brain activity and structure. The spatial resolution of these maps plays an important role, as it determines the ability to localize functional units with precision, perform reliable brain parcellation, and detect subtle, spatially specific neurobiological alterations associated with development, aging, or disease. Therefore, improving the effective resolution of neuroimaging-derived maps holds significant promise for enabling more detailed insights into brain architecture and its relationship to behavior and pathology. To address this need, we propose NeuroGAN-3D, a novel 3D generative super-resolution model tailored to the...
  </details>  
- **[Generative 3D Gaussians with Learned Density Control](https://arxiv.org/abs/2605.16355v1)**  
  Authors: Runjie Yan, Yan-Pei Cao, Peng Wang, Ding Liang, Yuan-Chen Guo  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2605.16355v1.pdf)  
  <details><summary>Abstract</summary>

  We present Density-Sampled Gaussians (DeG), a novel 3D representation designed to bridge the gap between adaptive rendering primitives and scalable generative modeling. Unlike existing approaches that constrain 3D Gaussians to fixed voxel grids or arrays, DeG models Gaussian centers as samples from a learnable probability density function defined over an octree. This formulation provides a rigorous mathematical framework for adaptive density control: by jointly optimizing the spatial density and Gaussian attributes under rendering supervision, our model naturally concentrates primitives in regions of high geometric complexity. We achieve this via a new render loss contribution gradient that serves as a fully differentiable analogue to the discrete densification and pruning heuristics used in standard Gaussian Splatting. The resulting representation is highly flexible, supporting variable-resolution decoding from a single latent code by simply adjusting the sampling budget. To enable generative synthesis, we train a latent diffusion model on DeG. We identify a critical...
  </details>  
  Keywords: image-to-3d  
- **[DVD: Discrete Voxel Diffusion for 3D Generation and Editing](https://arxiv.org/abs/2605.07971v2)**  
  Authors: Zhengrui Xiang, Jiaqi Wu, Fupeng Sun, Heliang Zheng, Yingzhen Li  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2605.07971v2.pdf)  
  <details><summary>Abstract</summary>

  We introduce Discrete Voxel Diffusion (DVD), a discrete diffusion framework to generate, assess, and edit sparse voxels for SLat (Structured LATent) based 3D generative pipelines. Although discrete diffusion has not generally displaced continuous diffusion in image-like generation, we show that it can be an effective first-stage prior for sparse voxel scaffolds. By treating voxel occupancy as a native discrete variable, DVD avoids continuous-to-discrete thresholding and provides a simple framework for voxel generation, uncertainty estimation, and editing. Beyond quality gains, DVD provides more interpretable generation dynamics through explicit categorical modeling. Furthermore, we leverage the predictive entropy as a robust uncertainty metric to identify ambiguous voxel regions and complicated samples, facilitating tasks such as data filtering and quality assessment. Finally, we propose a lightweight fine-tuning strategy using block-structured perturbation patterns. This approach empowers the model to inpaint and edit voxels within a single sampling round, requiring negligible auxiliary computation and no additional...
  </details>  
- **[Rollback-Free Stable Brick Structures Generation](https://arxiv.org/abs/2605.06947v1)**  
  Authors: Chenhui Xu, Ziyue Bai, Fuxun Yu, Heng Huang, Jinjun Xiong  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2605.06947v1.pdf) | [![GitHub](https://img.shields.io/github/stars/miniHuiHui/STABLE?style=social)](https://github.com/miniHuiHui/STABLE) | [![Project](https://img.shields.io/badge/-Project-blue)](https://huggingface.co/miniHui/STABLE.) | [![HuggingFace](https://img.shields.io/badge/-HuggingFace-yellow)](https://huggingface.co/miniHui/STABLE)  
  <details><summary>Abstract</summary>

  While autoregressive models have advanced 3D generation, creating physically stable brick structures remains a challenge due to the strict requirements of gravity and interconnectivity. Existing approaches rely on external physical simulators during inference to perform rejection sampling and brick-by-brick rollbacks, which severely bottlenecks efficiency. To address this, we propose a reinforcement learning paradigm that shifts physical validity enforcement from test-time correction to training-time policy optimization. By utilizing assembly-level rewards, the model optimizes for collision avoidance, global connectivity, structural interlocking, and shape conformity. This paradigm allows the model to internalize physical priors, enabling the first rollback-free generation of stable brick structures. Experimental results demonstrate that our approach achieves state-of-the-art generation quality while accelerating inference speed by orders of magnitude. Our code and dataset are available at https://github.com/miniHuiHui/STABLE. Our models are available at https://huggingface.co/miniHui/STABLE.
  </details>  
- **[LiVeAction: a Lightweight, Versatile, and Asymmetric Neural Codec Design for Real-time Operation](https://arxiv.org/abs/2605.06628v1)**  
  Authors: Dan Jacobellis, Neeraja J. Yadwadkar  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2605.06628v1.pdf) | [![GitHub](https://img.shields.io/github/stars/UT-SysML/liveaction?style=social)](https://github.com/UT-SysML/liveaction)  
  <details><summary>Abstract</summary>

  Modern sensors generate rich, high-fidelity data, yet applications operating on wearable or remote sensing devices remain constrained by bandwidth and power budgets. Standardized codecs such as JPEG and MPEG achieve efficient trade-offs between bitrate and perceptual quality but are designed for human perception, limiting their applicability to machine-perception tasks and non-traditional modalities such as spatial audio arrays, hyperspectral images, and 3D medical images. General-purpose compression schemes based on scalar quantization or resolution reduction are broadly applicable but fail to exploit inherent signal redundancies, resulting in suboptimal rate-distortion performance. Recent generative neural codecs, or tokenizers, model complex signal dependencies but are often over-parameterized, data-hungry, and modality-specific, making them impractical for resource-constrained environments. We introduce a Lightweight, Versatile, and Asymmetric neural codec architecture (LiVeAction), that addresses these limitations through two key ideas. (1) To reduce the complexity of the encoder to meet the resource constraints of the execution environments, we impose an...
  </details>  
- **[Structured 3D Latents Are Surprisingly Powerful: Unleashing Generalizable Style with 2D Diffusion](https://arxiv.org/abs/2605.04412v2)**  
  Authors: Yiran Qiao, Yiren Lu, Yunlai Zhou, Disheng Liu, Linlin Hou, Rui Yang, Yu Yin, Jing Ma  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2605.04412v2.pdf)  
  <details><summary>Abstract</summary>

  3D asset generation plays a pivotal role in fields such as gaming and virtual reality, enabling the rapid synthesis of high-fidelity 3D objects from a single or multiple images. Building on this capability, enabling style-controllable generation naturally emerges as an important and desirable direction. However, existing approaches typically rely on style images that lie within or are similar to the training distribution of 3D generation models. When presented with out-of-distribution (OOD) styles, their performance degrades significantly or even fails. To address this limitation, we introduce \textbf{DiLAST}: 2D Diffusion-based Latent Awakening for 3D Style Transfer. Specifically, we leverage a pretrained 2D diffusion model as a teacher to provide rich and generalizable style priors. By aligning rendered views with the target style under diffusion-based guidance, our method optimizes the structured 3D latent representations for stylization. We observe that this limitation stems not from insufficient model capacity, but from the underutilization of structured...
  </details>  
  Keywords: 3d asset generation  
- **[Mix3R: Mixing Feed-forward Reconstruction and Generative 3D Priors for Joint Multi-view Aligned 3D Reconstruction and Pose Estimation](https://arxiv.org/abs/2605.03359v1)**  
  Authors: Siyou Lin, Zhou Xue, Hongwen Zhang, Liang An, Dongping Li, Shaohui Jiao, Yebin Liu  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2605.03359v1.pdf) | [![Project](https://img.shields.io/badge/-Project-blue)](https://jsnln.github.io/mix3r)  
  <details><summary>Abstract</summary>

  Recent trends in sparse-view 3D reconstruction have taken two different paths: feed-forward reconstruction that predicts pixel-aligned point maps without a complete geometry, and generative 3D reconstruction that generates complete geometry but often with poor input-alignment. We present Mix3R, a novel generative 3D reconstruction method which mixes feed-forward reconstruction and 3D generation into a single framework in an aligned manner. Mix3R generates a 3D shape in two stages: a sparse voxel generation stage and a textured geometry generation stage. Unlike pure generative methods, our first-stage generation jointly produces a coarse 3D structure (sparse voxels), per-view point maps and camera parameters aligned to that 3D structure. This is made possible by introducing a Mixture-of-Transformers architecture that inserts global self-attentions to a feed-forward reconstruction model and a 3D generative model, both pretrained on large-scale data. This design effectively retains the pretrained priors but enables better 2D-3D alignment. Based on the initial aligned generations...
  </details>  
- **[MOC-3D: Manifold-Order Consistency for Text-to-3D Generation](https://arxiv.org/abs/2605.01743v1)**  
  Authors: Chenyang Fan, Junshi Cheng, Wen Yang, Zihong Li, Wenfeng Zhang, Wei Hu, Yi Zhang, Pan Zeng  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2605.01743v1.pdf)  
  <details><summary>Abstract</summary>

  With the burgeoning development of fields such as the Metaverse, Virtual Reality (VR), and Digital Twins, text-to-3D generation has emerged as a research hotspot in both academia and industry. Currently, optimization methods based on Score Distillation Sampling (SDS) utilizing 2D diffusion priors have become the mainstream technological paradigm in this field. However, due to the view bias of 2D priors and the mode-seeking ambiguity combined with gradient noise induced by high Classifier-Free Guidance (CFG), these methods still suffer from macro-topological inconsistency (e.g., the Janus problem) and micro-geometric discontinuity. To address these challenges, we propose MOC-3D, a text-to-3D generation method based on geometric manifold and semantic view-order consistency. Built upon the ScaleDreamer framework, our method incorporates a Semantic View-Order Constraint Module and a Manifold-based Feature Continuity Module. The former aims to rectify macro-topological inconsistency, while the latter focuses on eliminating micro-geometric discontinuity. Specifically, the Semantic View-Order Constraint Module leverages the prior...
  </details>  
  Keywords: text-to-3d  
- **[The Antipodal Method: Fast, Accurate, and Robust 3D Generalized Winding Numbers](https://arxiv.org/abs/2605.01536v1)**  
  Authors: Cedric Martens, Philip Trettner, Mikhail Bessmeltsev  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2605.01536v1.pdf) | [![GitHub](https://img.shields.io/github/stars/MartensCedric/antipodal?style=social)](https://github.com/MartensCedric/antipodal)  
  <details><summary>Abstract</summary>

  Generalized winding numbers provide a robust measure of point insidedness for 3D surfaces - whether open, self-intersecting, or non-manifold - and are central to numerous geometry processing tasks. However, existing methods trade off between accuracy and computational efficiency, limiting their use in interactive and large-scale applications. We introduce a new formulation and algorithm for computing generalized winding numbers that is both fast and accurate to arbitrary precision, applicable to meshes and parametric surfaces. Our approach expresses the winding number as the sum of two intuitive geometric quantities: the signed number of ray-surface intersections and a boundary integral over the surface's projection onto the unit sphere. This insight leads to an efficient discretization that avoids expensive surface integrals and spherical arrangements. For meshes, our method achieves average speedups of $22\times$ on a CPU compared to the fastest precise methods and $3\times$ compared to the fastest approximation method, while maintaining full precision....
  </details>  
- **[Map2World: Segment Map Conditioned Text to 3D World Generation](https://arxiv.org/abs/2605.00781v1)**  
  Authors: Jaeyoung Chung, Suyoung Lee, Jianfeng Xiang, Jiaolong Yang, Kyoung Mu Lee  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2605.00781v1.pdf)  
  <details><summary>Abstract</summary>

  3D world generation is essential for applications such as immersive content creation or autonomous driving simulation. Recent advances in 3D world generation have shown promising results; however, these methods are constrained by grid layouts and suffer from inconsistencies in object scale throughout the entire world. In this work, we introduce a novel framework, Map2World, that first enables 3D world generation conditioned on user-defined segment maps of arbitrary shapes and scales, ensuring global-scale consistency and flexibility across expansive environments. To further enhance the quality, we propose a detail enhancer network that generates fine details of the world. The detail enhancer enables the addition of fine-grained details without compromising overall scene coherence by incorporating global structure information. We design the entire pipeline to leverage strong priors from asset generators, achieving robust generalization across diverse domains, even under limited training data for scene generation. Extensive experiments demonstrate that our method significantly outperforms existing...
  </details>  
- **[PhysiGen: Integrating Collision-Aware Physical Constraints for High-Fidelity Human-Human Interaction Generation](https://arxiv.org/abs/2605.00517v1)**  
  Authors: Nan Lei, Yuan-Ming Li, Ling-An Zeng, Liang Xu, Zhi-Wei Xia, Hui-Wen Huang, Fa-Ting Hong, Wei-Shi Zheng  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2605.00517v1.pdf)  
  <details><summary>Abstract</summary>

  Despite substantial progress in text-driven 3D human motion synthesis, generating realistic multi-person interaction sequences remains challenging. Notably, body inter-penetration is a pervasive issue from both data acquisition to the generated results, which significantly undermines the realism and usability. Previous generative models either ignored this issue or introduced computationally expensive mesh-level loss functions to alleviate inter-body collisions. In this paper, we propose a general-purpose and computationally efficient optimization strategy named PhysiGen to explicitly integrate collision-aware physical constraints for human-human interaction generation. Specifically, we simplify the high-resolution human body mesh into geometric primitives to greatly reduce the cost of inter-person collision detection. Moreover, we identify the collision regions as the guidance of the optimization directions. PhysiGen is plug-and-play and can be readily integrated into existing human interaction generation models. Extensive cross-dataset and cross-model experiments show that our method can effectively reduce interpenetration and significantly improve visual coherence and physical plausibility compared to...
  </details>  
- **[Pose-Aware Diffusion for 3D Generation](https://arxiv.org/abs/2605.00345v1)**  
  Authors: Zihan Zhou, Luxi Chen, Jingzhi Zhou, Yuhao Wan, Min Zhao, Baoyu Fan, Chongxuan Li  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2605.00345v1.pdf)  
  <details><summary>Abstract</summary>

  Generating pose-aligned 3D objects is challenging due to the spatial mismatches and transformation ambiguities inherent in decoupled canonical-then-rotate paradigms. To this end, we introduce Pose-Aware Diffusion (PAD), a novel end-to-end diffusion framework that synthesizes 3D geometry directly within the observation space. By unprojecting monocular depth into a partial point cloud and explicitly injecting it as a 3D geometric anchor, PAD abandons canonical assumptions to enforce rigorous spatial supervision. This native generation intrinsically resolves pose ambiguity, producing high-fidelity pose-aligned assets. Extensive experiments demonstrate that PAD achieves superior geometric alignment and image-to-3D correspondence compared to state-of-the-art methods. Additionally, PAD naturally extends to compositional 3D scene reconstruction via a simple union of independently generated objects, highlighting its robust ability to preserve precise spatial layouts.
  </details>  
  Keywords: 3d scene reconstruction, image-to-3d  

### March 2026
- **[Extend3D: Town-Scale 3D Generation](https://arxiv.org/abs/2603.29387v1)**  
  Authors: Seungwoo Yoon, Jinmo Kim, Jaesik Park  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2603.29387v1.pdf)  
  <details><summary>Abstract</summary>

  In this paper, we propose Extend3D, a training-free pipeline for 3D scene generation from a single image, built upon an object-centric 3D generative model. To overcome the limitations of fixed-size latent spaces in object-centric models for representing wide scenes, we extend the latent space in the $x$ and $y$ directions. Then, by dividing the extended latent space into overlapping patches, we apply the object-centric 3D generative model to each patch and couple them at each time step. Since patch-wise 3D generation with image conditioning requires strict spatial alignment between image and latent patches, we initialize the scene using a point cloud prior from a monocular depth estimator and iteratively refine occluded regions through SDEdit. We discovered that treating the incompleteness of 3D structure as noise during 3D refinement enables 3D completion via a concept, which we term under-noising. Furthermore, to address the sub-optimality of object-centric models for sub-scene generation, we...
  </details>  
- **[HVG-3D: Bridging Real and Simulation Domains for 3D-Conditional Hand-Object Interaction Video Synthesis](https://arxiv.org/abs/2604.03305v1)**  
  Authors: Mingjin Chen, Junhao Chen, Zhaoxin Fan, Yujian Lee, Zichen Dang, Lili Wang, Yawen Cui, Lap-Pui Chau, Yi Wang  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2604.03305v1.pdf)  
  <details><summary>Abstract</summary>

  Recent methods have made notable progress in the visual quality of hand-object interaction video synthesis. However, most approaches rely on 2D control signals that lack spatial expressiveness and limit the utilization of synthetic 3D conditional data. To address these limitations, we propose HVG-3D, a unified framework for 3D-aware hand-object interaction (HOI) video synthesis conditioned on explicit 3D representations. Specifically, we develop a diffusion-based architecture augmented with a 3D ControlNet, which encodes geometric and motion cues from 3D inputs to enable explicit 3D reasoning during video synthesis. To achieve high-quality synthesis, HVG-3D is designed with two core components: (i) a 3D-aware HOI video generation diffusion architecture that encodes geometric and motion cues from 3D inputs for explicit 3D reasoning; and (ii) a hybrid pipeline for constructing input and condition signals, enabling flexible and precise control during both training and inference. During inference, given a single real image and a 3D control...
  </details>  
- **[WorldFlow3D: Flowing Through 3D Distributions for Unbounded World Generation](https://arxiv.org/abs/2603.29089v1)**  
  Authors: Amogh Joshi, Julian Ost, Felix Heide  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2603.29089v1.pdf) | [![Project](https://img.shields.io/badge/-Project-blue)](https://light.princeton.edu/worldflow3d)  
  <details><summary>Abstract</summary>

  Unbounded 3D world generation is emerging as a foundational task for scene modeling in computer vision, graphics, and robotics. In this work, we present WorldFlow3D, a novel method capable of generating unbounded 3D worlds. Building upon a foundational property of flow matching - namely, defining a path of transport between two data distributions - we model 3D generation more generally as a problem of flowing through 3D data distributions, not limited to conditional denoising. We find that our latent-free flow approach generates causal and accurate 3D structure, and can use this as an intermediate distribution to guide the generation of more complex structure and high-quality texture - all while converging more rapidly than existing methods. We enable controllability over generated scenes with vectorized scene layout conditions for geometric structure control and visual texture control through scene attributes. We confirm the effectiveness of WorldFlow3D on both real outdoor driving scenes and...
  </details>  
- **[ObjectMorpher: 3D-Aware Image Editing via Deformable 3DGS Models](https://arxiv.org/abs/2603.28152v1)**  
  Authors: Yuhuan Xie, Aoxuan Pan, Yi-Hua Huang, Chirui Chang, Peng Dai, Xin Yu, Xiaojuan Qi  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2603.28152v1.pdf)  
  <details><summary>Abstract</summary>

  Achieving precise, object-level control in image editing remains challenging: 2D methods lack 3D awareness and often yield ambiguous or implausible results, while existing 3D-aware approaches rely on heavy optimization or incomplete monocular reconstructions. We present ObjectMorpher, a unified, interactive framework that converts ambiguous 2D edits into geometry-grounded operations. ObjectMorpher lifts target instances with an image-to-3D generator into editable 3D Gaussian Splatting (3DGS), enabling fast, identity-preserving manipulation. Users drag control points; a graph-based non-rigid deformation with as-rigid-as-possible (ARAP) constraints ensures physically sensible shape and pose changes. A composite diffusion module harmonizes lighting, color, and boundaries for seamless reintegration. Across diverse categories, ObjectMorpher delivers fine-grained, photorealistic edits with superior controllability and efficiency, outperforming 2D drag and 3D-aware baselines on KID, LPIPS, SIFID, and user preference.
  </details>  
  Keywords: 3d generator, image-to-3d  
- **[SVGS: Single-View to 3D Object Editing via Gaussian Splatting](https://arxiv.org/abs/2603.28126v1)**  
  Authors: Pengcheng Xue, Yan Tian, Qiutao Song, Ziyi Wang, Linyang He, Weiping Ding, Mahmoud Hassaballah, Karen Egiazarian, Wei-Fa Yang, Leszek Rutkowski  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2603.28126v1.pdf) | [![Project](https://img.shields.io/badge/-Project-blue)](https://amateurc.github.io/svgs.github.io)  
  <details><summary>Abstract</summary>

  Text-driven 3D scene editing has attracted considerable interest due to its convenience and user-friendliness. However, methods that rely on implicit 3D representations, such as Neural Radiance Fields (NeRF), while effective in rendering complex scenes, are hindered by slow processing speeds and limited control over specific regions of the scene. Moreover, existing approaches, including Instruct-NeRF2NeRF and GaussianEditor, which utilize multi-view editing strategies, frequently produce inconsistent results across different views when executing text instructions. This inconsistency can adversely affect the overall performance of the model, complicating the task of balancing the consistency of editing results with editing efficiency. To address these challenges, we propose a novel method termed Single-View to 3D Object Editing via Gaussian Splatting (SVGS), which is a single-view text-driven editing technique based on 3D Gaussian Splatting (3DGS). Specifically, in response to text instructions, we introduce a single-view editing strategy grounded in multi-view diffusion models, which reconstructs 3D scenes by...
  </details>  
- **[Hg-I2P: Bridging Modalities for Generalizable Image-to-Point-Cloud Registration via Heterogeneous Graphs](https://arxiv.org/abs/2603.27969v1)**  
  Authors: Pei An, Junfeng Ding, Jiaqi Yang, Yulong Wang, Jie Ma, Liangliang Nan  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2603.27969v1.pdf) | [![GitHub](https://img.shields.io/github/stars/anpei96/hg-i2p-demo?style=social)](https://github.com/anpei96/hg-i2p-demo)  
  <details><summary>Abstract</summary>

  Image-to-point-cloud (I2P) registration aims to align 2D images with 3D point clouds by establishing reliable 2D-3D correspondences. The drastic modality gap between images and point clouds makes it challenging to learn features that are both discriminative and generalizable, leading to severe performance drops in unseen scenarios. We address this challenge by introducing a heterogeneous graph that enables refining both cross-modal features and correspondences within a unified architecture. The proposed graph represents a mapping between segmented 2D and 3D regions, which enhances cross-modal feature interaction and thus improves feature discriminability. In addition, modeling the consistency among vertices and edges within the graph enables pruning of unreliable correspondences. Building on these insights, we propose a heterogeneous graph embedded I2P registration method, termed Hg-I2P. It learns a heterogeneous graph by mining multi-path feature relationships, adapts features under the guidance of heterogeneous edges, and prunes correspondences using graph-based projection consistency. Experiments on six indoor...
  </details>  
- **[Which Reconstruction Model Should a Robot Use? Routing Image-to-3D Models for Cost-Aware Robotic Manipulation](https://arxiv.org/abs/2603.27797v1)**  
  Authors: Akash Anand, Aditya Agarwal, Leslie Pack Kaelbling  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2603.27797v1.pdf)  
  <details><summary>Abstract</summary>

  Robotic manipulation tasks require 3D mesh reconstructions of varying quality: dexterous manipulation demands fine-grained surface detail, while collision-free planning tolerates coarser representations. Multiple reconstruction methods offer different cost-quality tradeoffs, from Image-to-3D models - whose output quality depends heavily on the input viewpoint - to view-invariant methods such as structured light scanning. Querying all models is computationally prohibitive, motivating per-input model selection. We propose SCOUT, a novel routing framework that decouples reconstruction scores into two components: (1) the relative performance of viewpoint-dependent models, captured by a learned probability distribution, and (2) the overall image difficulty, captured by a scalar partition function estimate. As the learned network operates only over the viewpoint-dependent models, view-invariant pipelines can be added, removed, or reconfigured without retraining. SCOUT also supports arbitrary cost constraints at inference time, accommodating the multi-dimensional cost constraints common in robotics. We evaluate on the Google Scanned Objects, BigBIRD, and YCB datasets under...
  </details>  
  Keywords: image-to-3d  
- **[NimbusGS: Unified 3D Scene Reconstruction under Hybrid Weather](https://arxiv.org/abs/2603.27228v1)**  
  Authors: Yanying Li, Jinyang Li, Shengfeng He, Yangyang Xu, Junyu Dong, Yong Du  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2603.27228v1.pdf) | [![GitHub](https://img.shields.io/github/stars/lyy-ovo/NimbusGS?style=social)](https://github.com/lyy-ovo/NimbusGS)  
  <details><summary>Abstract</summary>

  We present NimbusGS, a unified framework for reconstructing high-quality 3D scenes from degraded multi-view inputs captured under diverse and mixed adverse weather conditions. Unlike existing methods that target specific weather types, NimbusGS addresses the broader challenge of generalization by modeling the dual nature of weather: a continuous, view-consistent medium that attenuates light, and dynamic, view-dependent particles that cause scattering and occlusion. To capture this structure, we decompose degradations into a global transmission field and per-view particulate residuals. The transmission field represents static atmospheric effects shared across views, while the residuals model transient disturbances unique to each input. To enable stable geometry learning under severe visibility degradation, we introduce a geometry-guided gradient scaling mechanism that mitigates gradient imbalance during the self-supervised optimization of 3D Gaussian representations. This physically grounded formulation allows NimbusGS to disentangle complex degradations while preserving scene structure, yielding superior geometry reconstruction and outperforming task-specific methods across diverse and...
  </details>  
  Keywords: 3d scene reconstruction  
- **[Hierarchy-Guided Topology Latent Flow for Molecular Graph Generation](https://arxiv.org/abs/2603.27113v1)**  
  Authors: Urvi Awasthi, Alexander Arjun Lobo, Leonid Zhukov  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2603.27113v1.pdf)  
  <details><summary>Abstract</summary>

  Generating chemically valid 3D molecules is hindered by discrete bond topology: small local bond errors can cause global failures (valence violations, disconnections, implausible rings), especially for drug-like molecules with long-range constraints. Many unconditional 3D generators emphasize coordinates and then infer bonds or rely on post-processing, leaving topology feasibility weakly controlled. We propose Hierarchy-Guided Latent Topology Flow (HLTF), a planner-executor model that generates bond graphs with 3D coordinates, using a latent multi-scale plan for global context and a constraint-aware sampler to suppress topology-driven failures. On QM9, HLTF achieves 98.8% atom stability and 92.9% valid-and-unique, improving PoseBusters validity to 94.0% (+0.9 over the strongest reported baseline). On GEOM-DRUGS, HLTF attains 85.5%/85.0% validity/valid-unique-novel without post-processing and 92.2%/91.2% after standardized relaxation, within 0.9 points of the best post-processed baseline. Explicit topology generation also reduces "false-valid" samples that pass RDKit sanitization but fail stricter checks.
  </details>  
  Keywords: 3d generator  
- **[GaussianGPT: Towards Autoregressive 3D Gaussian Scene Generation](https://arxiv.org/abs/2603.26661v1)**  
  Authors: Nicolas von Lützow, Barbara Rössle, Katharina Schmid, Matthias Nießner  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2603.26661v1.pdf)  
  <details><summary>Abstract</summary>

  Most recent advances in 3D generative modeling rely on diffusion or flow-matching formulations. We instead explore a fully autoregressive alternative and introduce GaussianGPT, a transformer-based model that directly generates 3D Gaussians via next-token prediction, thus facilitating full 3D scene generation. We first compress Gaussian primitives into a discrete latent grid using a sparse 3D convolutional autoencoder with vector quantization. The resulting tokens are serialized and modeled using a causal transformer with 3D rotary positional embedding, enabling sequential generation of spatial structure and appearance. Unlike diffusion-based methods that refine scenes holistically, our formulation constructs scenes step-by-step, naturally supporting completion, outpainting, controllable sampling via temperature, and flexible generation horizons. This formulation leverages the compositional inductive biases and scalability of autoregressive modeling while operating on explicit representations compatible with modern neural rendering pipelines, positioning autoregressive transformers as a complementary paradigm for controllable and context-aware 3D generation.
  </details>  
- **[ArtHOI: Taming Foundation Models for Monocular 4D Reconstruction of Hand-Articulated-Object Interactions](https://arxiv.org/abs/2603.25791v1)**  
  Authors: Zikai Wang, Zhilu Zhang, Yiqing Wang, Hui Li, Wangmeng Zuo  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2603.25791v1.pdf) | [![Project](https://img.shields.io/badge/-Project-blue)](https://arthoi-reconstruction.github.io)  
  <details><summary>Abstract</summary>

  Existing hand-object interactions (HOI) methods are largely limited to rigid objects, while 4D reconstruction methods of articulated objects generally require pre-scanning the object or even multi-view videos. It remains an unexplored but significant challenge to reconstruct 4D human-articulated-object interactions from a single monocular RGB video. Fortunately, recent advancements in foundation models present a new opportunity to address this highly ill-posed problem. To this end, we introduce ArtHOI, an optimization-based framework that integrates and refines priors from multiple foundation models. Our key contribution is a suite of novel methodologies designed to resolve the inherent inaccuracies and physical unreality of these priors. In particular, we introduce an Adaptive Sampling Refinement (ASR) method to optimize object's metric scale and pose for grounding its normalized mesh in world space. Furthermore, we propose a Multimodal Large Language Model (MLLM) guided hand-object alignment method, utilizing contact reasoning information as constraints of hand-object mesh composition optimization. To...
  </details>  
- **[ViewSplat: View-Adaptive Dynamic Gaussian Splatting for Feed-Forward Synthesis](https://arxiv.org/abs/2603.25265v1)**  
  Authors: Moonyeon Jeong, Seunggi Min, Suhyeon Lee, Hongje Seong  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2603.25265v1.pdf)  
  <details><summary>Abstract</summary>

  We present ViewSplat, a view-adaptive 3D Gaussian splatting network for novel view synthesis from unposed images. While recent feed-forward 3D Gaussian splatting has significantly accelerated 3D scene reconstruction by bypassing per-scene optimization, a fundamental fidelity gap remains. We attribute this bottleneck to the limited capacity of single-step feed-forward networks to regress static Gaussian primitives that satisfy all viewpoints. To address this limitation, we shift the paradigm from static primitive regression to view-adaptive dynamic splatting. Instead of a rigid Gaussian representation, our pipeline learns a view-adaptable latent representation. Specifically, ViewSplat initially predicts base Gaussian primitives alongside the weights of dynamic MLPs. During rendering, these MLPs take target view coordinates as input and predict view-dependent residual updates for each Gaussian attribute (i.e., 3D position, scale, rotation, opacity, and color). This mechanism, which we term view-adaptive dynamic splatting, allows each primitive to rectify initial estimation errors, effectively capturing high-fidelity appearances. Extensive experiments demonstrate...
  </details>  
  Keywords: 3d scene reconstruction, novel view synthesis  
- **[Teacher-Student Diffusion Model for Text-Driven 3D Hand Motion Generation](https://arxiv.org/abs/2603.24407v1)**  
  Authors: Ching-Lam Cheng, Bin Zhu, Shengfeng He  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2603.24407v1.pdf)  
  <details><summary>Abstract</summary>

  Generating realistic 3D hand motion from natural language is vital for VR, robotics, and human-computer interaction. Existing methods either focus on full-body motion, overlooking detailed hand gestures, or require explicit 3D object meshes, limiting generality. We propose TSHaMo, a model-agnostic teacher-student diffusion framework for text-driven hand motion generation. The student model learns to synthesize motions from text alone, while the teacher leverages auxiliary signals (e.g., MANO parameters) to provide structured guidance during training. A co-training strategy enables the student to benefit from the teacher's intermediate predictions while remaining text-only at inference. Evaluated using two diffusion backbones on GRAB and H2O, TSHaMo consistently improves motion quality and diversity. Ablations confirm its robustness and flexibility in using diverse auxiliary inputs without requiring 3D objects at test time.
  </details>  
- **[TopoMesh: High-Fidelity Mesh Autoencoding via Topological Unification](https://arxiv.org/abs/2603.24278v2)**  
  Authors: Guan Luo, Xiu Li, Rui Chen, Xuanyu Yi, Jing Lin, Chia-Hao Chen, Jiahang Liu, Song-Hai Zhang, Jianfeng Zhang  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2603.24278v2.pdf)  
  <details><summary>Abstract</summary>

  The dominant paradigm for high-fidelity 3D generation relies on a VAE-Diffusion pipeline, where the VAE's reconstruction capability sets a firm upper bound on generation quality. A fundamental challenge limiting existing VAEs is the representation mismatch between ground-truth meshes and network predictions: GT meshes have arbitrary, variable topology, while VAEs typically predict fixed-structure implicit fields (\eg, SDF on regular grids). This inherent misalignment prevents establishing explicit mesh-level correspondences, forcing prior work to rely on indirect supervision signals such as SDF or rendering losses. Consequently, fine geometric details, particularly sharp features, are poorly preserved during reconstruction. To address this, we introduce TopoMesh, a sparse voxel-based VAE that unifies both GT and predicted meshes under a shared Dual Marching Cubes (DMC) topological framework. Specifically, we convert arbitrary input meshes into DMC-compliant representations via a remeshing algorithm that preserves sharp edges using an L$\infty$ distance metric. Our decoder outputs meshes in the same DMC...
  </details>  
- **[Realiz3D: 3D Generation Made Photorealistic via Domain-Aware Learning](https://arxiv.org/abs/2605.13852v1)**  
  Authors: Ido Sobol, Kihyuk Sohn, Yoav Blum, Egor Zakharov, Max Bluvstein, Andrea Vedaldi, Or Litany  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2605.13852v1.pdf)  
  <details><summary>Abstract</summary>

  We often aim to generate images that are both photorealistic and 3D-consistent, adhering to precise geometry, material, and viewpoint controls. Typically, this is achieved by fine-tuning an image generator, pre-trained on billions of real images, using renders of synthetic 3D assets, where annotations for control signals are available. While this approach can learn the desired controls, it often compromises the realism of the images due to domain gap between photographs and renders. We observe that this issue largely arises from the model learning an unintended association between the presence of control signals and the synthetic appearance of the images. To address this, we introduce Realiz3D, a lightweight framework for training diffusion models, that decouples controls and visual domain. The key idea is to explicitly learn visual domain, real or synthetic, separately from other control signals by introducing a co-variate that, fed into small residual adapters, shifts the domain. Then, the...
  </details>  
- **[SLAT-Phys: Fast Material Property Field Prediction from Structured 3D Latents](https://arxiv.org/abs/2603.23973v1)**  
  Authors: Rocktim Jyoti Das, Dinesh Manocha  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2603.23973v1.pdf)  
  <details><summary>Abstract</summary>

  Estimating the material property field of 3D assets is critical for physics-based simulation, robotics, and digital twin generation. Existing vision-based approaches are either too expensive and slow or rely on 3D information. We present SLAT-Phys, an end-to-end method that predicts spatially varying material property fields of 3D assets directly from a single RGB image without explicit 3D reconstruction. Our approach leverages spatially organised latent features from a pretrained 3D asset generation model that encodes rich geometry and semantic prior, and trains a lightweight neural decoder to estimate Young's modulus, density, and Poisson's ratio. The coarse volumetric layout and semantic cues of the latent representation about object geometry and appearance enable accurate material estimation. Our experiments demonstrate that our method provides competitive accuracy in predicting continuous material parameters when compared against prior approaches, while significantly reducing computation time. In particular, SLAT-Phys requires only 9.9 seconds per object on an NVIDIA RTXA5000...
  </details>  
  Keywords: 3d asset generation  
- **[One View Is Enough! Monocular Training for In-the-Wild Novel View Generation](https://arxiv.org/abs/2603.23488v2)**  
  Authors: Adrien Ramanana Rahary, Nicolas Dufour, Patrick Perez, David Picard  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2603.23488v2.pdf) | [![GitHub](https://img.shields.io/github/stars/AdrienRR/ovie?style=social)](https://github.com/AdrienRR/ovie)  
  <details><summary>Abstract</summary>

  Monocular novel-view synthesis has long required multi-view image pairs for supervision, limiting training data scale and diversity. We argue it is not necessary: one view is enough. We present OVIE, trained entirely on unpaired internet images. We leverage a monocular depth estimator as a geometric scaffold at training time: we lift a source image into 3D, apply a sampled camera transformation, and project to obtain a pseudo-target view. To handle disocclusions, we introduce a masked training formulation that restricts geometric, perceptual, and textural losses to valid regions, enabling training on 30 million uncurated images. At inference, OVIE is geometry-free, requiring no depth estimator or 3D representation. Trained exclusively on in-the-wild images, OVIE outperforms prior methods in a zero-shot setting, while being 600x faster than the second-best baseline. Code and models are publicly available at https://github.com/AdrienRR/ovie.
  </details>  
- **[SIMART: Decomposing Monolithic Meshes into Sim-ready Articulated Assets via MLLM](https://arxiv.org/abs/2603.23386v1)**  
  Authors: Chuanrui Zhang, Minghan Qin, Yuang Wang, Baifeng Xie, Hang Li, Ziwei Wang  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2603.23386v1.pdf)  
  <details><summary>Abstract</summary>

  High-quality articulated 3D assets are indispensable for embodied AI and physical simulation, yet 3D generation still focuses on static meshes, leaving a gap in "sim-ready" interactive objects. Most recent articulated object creation methods rely on multi-stage pipelines that accumulate errors across decoupled modules. Alternatively, unified MLLMs offer a single-stage path to joint static asset understanding and sim-ready asset generation. However dense voxel-based 3D tokenization yields long 3D token sequences and high memory overhead, limiting scalability to complex articulated objects. To address this, we propose SIMART, a unified MLLM framework that jointly performs part-level decomposition and kinematic prediction. By introducing a Sparse 3D VQ-VAE, SIMART reduces token counts by 70% vs. dense voxel tokens, enabling high-fidelity multi-part assemblies. SIMART achieves state-of-the-art performance on PartNet-Mobility and in-the-wild AIGC datasets, and enables physics-based robotic simulation.
  </details>  
- **[Patchwork: A compact representation for 3D polygonal shapes](https://arxiv.org/abs/2605.16266v1)**  
  Authors: Ruichen Zheng, Biao Zhang, Michael Birsak, Mikhail Skopenkov, Peter Wonka  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2605.16266v1.pdf) | [![GitHub](https://img.shields.io/github/stars/Ankbzpx/patchwork-experiment?style=social)](https://github.com/Ankbzpx/patchwork-experiment)  
  <details><summary>Abstract</summary>

  We introduce Patchwork, a new general-purpose shape representation capable of modeling 2D and 3D geometry with a small number of parameters. Patchwork is grounded in a rigorous mathematical framework, providing provable complexity bounds and the ability to approximate arbitrary shapes with arbitrary precision in any dimension. We propose an efficient gradient-based optimization scheme to fit Patchwork representations to 2D and 3D data, along with a novel regularization loss that progressively prunes redundant elements, yielding high compactness after convergence. Our approach offers fast fitting performance, a fraction of the required parameters compared to existing alternatives, and native support for inside-outside classification, making it a versatile and compact representation for geometric learning and reconstruction tasks, with future potential for 3D generation. Our implementation is available at: https://github.com/Ankbzpx/patchwork-experiment.
  </details>  
- **[Know3D: Prompting 3D Generation with Knowledge from Vision-Language Models](https://arxiv.org/abs/2603.22782v1)**  
  Authors: Wenyue Chen, Wenjue Chen, Peng Li, Qinghe Wang, Xu Jia, Heliang Zheng, Rongfei Jia, Yuan Liu, Ronggang Wang  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2603.22782v1.pdf)  
  <details><summary>Abstract</summary>

  Recent advances in 3D generation have improved the fidelity and geometric details of synthesized 3D assets. However, due to the inherent ambiguity of single-view observations and the lack of robust global structural priors caused by limited 3D training data, the unseen regions generated by existing models are often stochastic and difficult to control, which may sometimes fail to align with user intentions or produce implausible geometries. In this paper, we propose Know3D, a novel framework that incorporates rich knowledge from multimodal large language models into 3D generative processes via latent hidden-state injection, enabling language-controllable generation of the back-view for 3D assets. We utilize a VLM-diffusion-based model, where the VLM is responsible for semantic understanding and guidance. The diffusion model acts as a bridge that transfers semantic knowledge from the VLM to the 3D generation model. In this way, we successfully bridge the gap between abstract textual instructions and the geometric...
  </details>  
- **[FullCircle: Effortless 3D Reconstruction from Casual 360$^\circ$ Captures](https://arxiv.org/abs/2603.22572v1)**  
  Authors: Yalda Foroutan, Ipek Oztas, Daniel Rebain, Aysegul Dundar, Kwang Moo Yi, Lily Goli, Andrea Tagliasacchi  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2603.22572v1.pdf) | [![Project](https://img.shields.io/badge/-Project-blue)](https://theialab.github.io/fullcircle)  
  <details><summary>Abstract</summary>

  Radiance fields have emerged as powerful tools for 3D scene reconstruction. However, casual capture remains challenging due to the narrow field of view of perspective cameras, which limits viewpoint coverage and feature correspondences necessary for reliable camera calibration and reconstruction. While commercially available 360$^\circ$ cameras offer significantly broader coverage than perspective cameras for the same capture effort, existing 360$^\circ$ reconstruction methods require special capture protocols and pre-processing steps that undermine the promise of radiance fields: effortless workflows to capture and reconstruct 3D scenes. We propose a practical pipeline for reconstructing 3D scenes directly from raw 360$^\circ$ camera captures. We require no special capture protocols or pre-processing, and exhibit robustness to a prevalent source of reconstruction errors: the human operator that is visible in all 360$^\circ$ imagery. To facilitate evaluation, we introduce a multi-tiered dataset of scenes captured as raw dual-fisheye images, establishing a benchmark for robust casual 360$^\circ$ reconstruction. Our...
  </details>  
  Keywords: 3d scene reconstruction  
- **[From Part to Whole: 3D Generative World Model with an Adaptive Structural Hierarchy](https://arxiv.org/abs/2603.21557v1)**  
  Authors: Bi'an Du, Daizong Liu, Pufan Li, Wei Hu  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2603.21557v1.pdf)  
  <details><summary>Abstract</summary>

  Single-image 3D generation lies at the core of vision-to-graphics models in the real world. However, it remains a fundamental challenge to achieve reliable generalization across diverse semantic categories and highly variable structural complexity under sparse supervision. Existing approaches typically model objects in a monolithic manner or rely on a fixed number of parts, including recent part-aware models such as PartCrafter, which still require a labor-intensive user-specified part count. Such designs easily lead to overfitting, fragmented or missing structural components, and limited compositional generalization when encountering novel object layouts. To this end, this paper rethinks single-image 3D generation as learning an adaptive part-whole hierarchy in the flexible 3D latent space. We present a novel part-to-whole 3D generative world model that autonomously discovers latent structural slots by inferring soft and compositional masks directly from image tokens. Specifically, an adaptive slot-gating mechanism dynamically determines the slot-wise activation probabilities and smoothly consolidates redundant slots...
  </details>  
- **[HyReach: Vision-Guided Hybrid Manipulator Reaching in Unseen Cluttered Environments](https://arxiv.org/abs/2603.21421v1)**  
  Authors: Shivani Kamtikar, Kendall Koe, Justin Wasserman, Samhita Marri, Benjamin Walt, Naveen Kumar Uppalapati, Girish Krishnan, Girish Chowdhary  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2603.21421v1.pdf)  
  <details><summary>Abstract</summary>

  As robotic systems increasingly operate in unstructured, cluttered, and previously unseen environments, there is a growing need for manipulators that combine compliance, adaptability, and precise control. This work presents a real-time hybrid rigid-soft continuum manipulator system designed for robust open-world object reaching in such challenging environments. The system integrates vision-based perception and 3D scene reconstruction with shape-aware motion planning to generate safe trajectories. A learning-based controller drives the hybrid arm to arbitrary target poses, leveraging the flexibility of the soft segment while maintaining the precision of the rigid segment. The system operates without environment-specific retraining, enabling direct generalization to new scenes. Extensive real-world experiments demonstrate consistent reaching performance with errors below 2 cm across diverse cluttered setups, highlighting the potential of hybrid manipulators for adaptive and reliable operation in unstructured environments.
  </details>  
  Keywords: 3d scene reconstruction  
- **[Text-Image Conditioned 3D Generation](https://arxiv.org/abs/2603.21295v1)**  
  Authors: Jiazhong Cen, Jiemin Fang, Sikuang Li, Guanjun Wu, Chen Yang, Taoran Yi, Zanwei Zhou, Zhikuan Bao, Lingxi Xie, Wei Shen, Qi Tian  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2603.21295v1.pdf) | [![Project](https://img.shields.io/badge/-Project-blue)](https://jumpat.github.io/tigon-page)  
  <details><summary>Abstract</summary>

  High-quality 3D assets are essential for VR/AR, industrial design, and entertainment, motivating growing interest in generative models that create 3D content from user prompts. Most existing 3D generators, however, rely on a single conditioning modality: image-conditioned models achieve high visual fidelity by exploiting pixel-aligned cues but suffer from viewpoint bias when the input view is limited or ambiguous, while text-conditioned models provide broad semantic guidance yet lack low-level visual detail. This limits how users can express intent and raises a natural question: can these two modalities be combined for more flexible and faithful 3D generation? Our diagnostic study shows that even simple late fusion of text- and image-conditioned predictions outperforms single-modality models, revealing strong cross-modal complementarity. We therefore formalize Text-Image Conditioned 3D Generation, which requires joint reasoning over a visual exemplar and a textual specification. To address this task, we introduce TIGON, a minimalist dual-branch baseline with separate image- and...
  </details>  
  Keywords: 3d generator  
- **[Training-Free Instance-Aware 3D Scene Reconstruction and Diffusion-Based View Synthesis from Sparse Images](https://arxiv.org/abs/2603.21166v1)**  
  Authors: Jiatong Xia, Lingqiao Liu  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2603.21166v1.pdf) | [![Project](https://img.shields.io/badge/-Project-blue)](https://jiatongxia.github.io/TID3R)  
  <details><summary>Abstract</summary>

  We introduce a novel, training-free system for reconstructing, understanding, and rendering 3D indoor scenes from a sparse set of unposed RGB images. Unlike traditional radiance field approaches that require dense views and per-scene optimization, our pipeline achieves high-fidelity results without any training or pose preprocessing. The system integrates three key innovations: (1) A robust point cloud reconstruction module that filters unreliable geometry using a warping-based anomaly removal strategy; (2) A warping-guided 2D-to-3D instance lifting mechanism that propagates 2D segmentation masks into a consistent, instance-aware 3D representation; and (3) A novel rendering approach that projects the point cloud into new views and refines the renderings with a 3D-aware diffusion model. Our method leverages the generative power of diffusion to compensate for missing geometry and enhances realism, especially under sparse input conditions. We further demonstrate that object-level scene editing such as instance removal can be naturally supported in our pipeline by modifying...
  </details>  
  Keywords: 3d scene reconstruction  
- **[Memory Over Maps: 3D Object Localization Without Reconstruction](https://arxiv.org/abs/2603.20530v2)**  
  Authors: Rui Zhou, Xander Yap, Jianwen Cao, Allison Lau, Boyang Sun, Marc Pollefeys  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2603.20530v2.pdf) | [![Project](https://img.shields.io/badge/-Project-blue)](https://ruizhou-cn.github.io/memory-over-maps)  
  <details><summary>Abstract</summary>

  Target localization is a prerequisite for embodied tasks such as navigation and manipulation. Conventional approaches rely on constructing explicit 3D scene representations to enable target localization, such as point clouds, voxel grids, or scene graphs. While effective, these pipelines incur substantial mapping time, storage overhead, and scalability limitations. Recent advances in vision-language models suggest that rich semantic reasoning can be performed directly on 2D observations, raising a fundamental question: is a complete 3D scene reconstruction necessary for object localization? In this work, we revisit object localization and propose a map-free pipeline that stores only posed RGB-D keyframes as a lightweight visual memory--without constructing any global 3D representation of the scene. At query time, our method retrieves candidate views, re-ranks them with a vision-language model, and constructs a sparse, on-demand 3D estimate of the queried target through depth backprojection and multi-view fusion. Compared to reconstruction-based pipelines, this design drastically reduces preprocessing...
  </details>  
  Keywords: 3d scene reconstruction  
- **[Uni-Classifier: Leveraging Video Diffusion Priors for Universal Guidance Classifier](https://arxiv.org/abs/2603.20382v1)**  
  Authors: Yujie Zhou, Pengyang Ling, Jiazi Bu, Bingjie Gao, Li Niu  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2603.20382v1.pdf)  
  <details><summary>Abstract</summary>

  In practical AI workflows, complex tasks often involve chaining multiple generative models, such as using a video or 3D generation model after a 2D image generator. However, distributional mismatches between the output of upstream models and the expected input of downstream models frequently degrade overall generation quality. To address this issue, we propose Uni-Classifier (Uni-C), a simple yet effective plug-and-play module that leverages video diffusion priors to guide the denoising process of preceding models, thereby aligning their outputs with downstream requirements. Uni-C can also be applied independently to enhance the output quality of individual generative models. Extensive experiments across video and 3D generation tasks demonstrate that Uni-C consistently improves generation quality in both workflow-based and standalone settings, highlighting its versatility and strong generalization capability.
  </details>  
- **[WorldAgents: Can Foundation Image Models be Agents for 3D World Models?](https://arxiv.org/abs/2603.19708v1)**  
  Authors: Ziya Erkoç, Angela Dai, Matthias Nießner  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2603.19708v1.pdf)  
  <details><summary>Abstract</summary>

  Given the remarkable ability of 2D foundation image models to generate high-fidelity outputs, we investigate a fundamental question: do 2D foundation image models inherently possess 3D world model capabilities? To answer this, we systematically evaluate multiple state-of-the-art image generation models and Vision-Language Models (VLMs) on the task of 3D world synthesis. To harness and benchmark their potential implicit 3D capability, we propose an agentic framing to facilitate 3D world generation. Our approach employs a multi-agent architecture: a VLM-based director that formulates prompts to guide image synthesis, a generator that synthesizes new image views, and a VLM-backed two-step verifier that evaluates and selectively curates generated frames from both 2D image and 3D reconstruction space. Crucially, we demonstrate that our agentic approach provides coherent and robust 3D reconstruction, producing output scenes that can be explored by rendering novel views. Through extensive experiments across various foundation models, we demonstrate that 2D models do...
  </details>  
- **[DreamPartGen: Semantically Grounded Part-Level 3D Generation via Collaborative Latent Denoising](https://arxiv.org/abs/2603.19216v1)**  
  Authors: Tianjiao Yu, Xinzhuo Li, Muntasir Wahed, Jerry Xiong, Yifan Shen, Ying Shen, Ismini Lourentzou  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2603.19216v1.pdf)  
  <details><summary>Abstract</summary>

  Understanding and generating 3D objects as compositions of meaningful parts is fundamental to human perception and reasoning. However, most text-to-3D methods overlook the semantic and functional structure of parts. While recent part-aware approaches introduce decomposition, they remain largely geometry-focused, lacking semantic grounding and failing to model how parts align with textual descriptions or their inter-part relations. We propose DreamPartGen, a framework for semantically grounded, part-aware text-to-3D generation. DreamPartGen introduces Duplex Part Latents (DPLs) that jointly model each part's geometry and appearance, and Relational Semantic Latents (RSLs) that capture inter-part dependencies derived from language. A synchronized co-denoising process enforces mutual geometric and semantic consistency, enabling coherent, interpretable, and text-aligned 3D synthesis. Across multiple benchmarks, DreamPartGen delivers state-of-the-art performance in geometric fidelity and text-shape alignment.
  </details>  
  Keywords: text-to-3d  
- **[V-Dreamer: Automating Robotic Simulation and Trajectory Synthesis via Video Generation Priors](https://arxiv.org/abs/2603.18811v1)**  
  Authors: Songjia He, Zixuan Chen, Hongyu Ding, Dian Shao, Jieqi Shi, Chenxu Li, Jing Huo, Yang Gao  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2603.18811v1.pdf)  
  <details><summary>Abstract</summary>

  Training generalist robots demands large-scale, diverse manipulation data, yet real-world collection is prohibitively expensive, and existing simulators are often constrained by fixed asset libraries and manual heuristics. To bridge this gap, we present V-Dreamer, a fully automated framework that generates open-vocabulary, simulation-ready manipulation environments and executable expert trajectories directly from natural language instructions. V-Dreamer employs a novel generative pipeline that constructs physically grounded 3D scenes using large language models and 3D generative models, validated by geometric constraints to ensure stable, collision-free layouts. Crucially, for behavior synthesis, we leverage video generation models as rich motion priors. These visual predictions are then mapped into executable robot trajectories via a robust Sim-to-Gen visual-kinematic alignment module utilizing CoTracker3 and VGGT. This pipeline supports high visual diversity and physical fidelity without manual intervention. To evaluate the generated data, we train imitation learning policies on synthesized trajectories encompassing diverse object and environment variations. Extensive evaluations on...
  </details>  
- **[Points-to-3D: Structure-Aware 3D Generation with Point Cloud Priors](https://arxiv.org/abs/2603.18782v3)**  
  Authors: Jiatong Xia, Zicheng Duan, Anton van den Hengel, Lingqiao Liu  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2603.18782v3.pdf) | [![Project](https://img.shields.io/badge/-Project-blue)](https://jiatongxia.github.io/points2-3D)  
  <details><summary>Abstract</summary>

  Recent progress in 3D generation has been driven largely by models conditioned on images or text, while readily available 3D priors are still underused. In many real-world scenarios, the visible-region point cloud are easy to obtain from active sensors such as LiDAR or from feed-forward predictors like VGGT, offering explicit geometric constraints that current methods fail to exploit. In this work, we introduce Points-to-3D, a diffusion-based framework that leverages point cloud priors for geometry-controllable 3D asset and scene generation. Built on a latent 3D diffusion model TRELLIS, Points-to-3D first replaces pure-noise sparse structure latent initialization with a point cloud priors tailored input formulation.A structure inpainting network, trained within the TRELLIS framework on task-specific data designed to learn global structural inpainting, is then used for inference with a staged sampling strategy (structural inpainting followed by boundary refinement), completing the global geometry while preserving the visible regions of the input priors. In...
  </details>  
  Keywords: controllable 3d generation  
- **[SEAR: Simple and Efficient Adaptation of Visual Geometric Transformers for RGB+Thermal 3D Reconstruction](https://arxiv.org/abs/2603.18774v1)**  
  Authors: Vsevolod Skorokhodov, Chenghao Xu, Shuo Sun, Olga Fink, Malcolm Mielle  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2603.18774v1.pdf) | [![GitHub](https://img.shields.io/github/stars/Schindler-EPFL-Lab/SEAR?style=social)](https://www.github.com/Schindler-EPFL-Lab/SEAR)  
  <details><summary>Abstract</summary>

  Foundational feed-forward visual geometry models enable accurate and efficient camera pose estimation and scene reconstruction by learning strong scene priors from massive RGB datasets. However, their effectiveness drops when applied to mixed sensing modalities, such as RGB-thermal (RGB-T) images. We observe that while a visual geometry grounded transformer pretrained on RGB data generalizes well to thermal-only reconstruction, it struggles to align RGB and thermal modalities when processed jointly. To address this, we propose SEAR, a simple yet efficient fine-tuning strategy that adapts a pretrained geometry transformer to multimodal RGB-T inputs. Despite being trained on a relatively small RGB-T dataset, our approach significantly outperforms state-of-the-art methods for 3D reconstruction and camera pose estimation, achieving significant improvements over all metrics (e.g., over 29\% in AUC@30) and delivering higher detail and consistency between modalities with negligible overhead in inference time compared to the original pretrained model. Notably, SEAR enables reliable multimodal pose estimation...
  </details>  
  Keywords: 3d scene reconstruction  
- **[LoST: Level of Semantics Tokenization for 3D Shapes](https://arxiv.org/abs/2603.17995v1)**  
  Authors: Niladri Shekhar Dutt, Zifan Shi, Paul Guerrero, Chun-Hao Paul Huang, Duygu Ceylan, Niloy J. Mitra, Xuelin Chen  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2603.17995v1.pdf)  
  <details><summary>Abstract</summary>

  Tokenization is a fundamental technique in the generative modeling of various modalities. In particular, it plays a critical role in autoregressive (AR) models, which have recently emerged as a compelling option for 3D generation. However, optimal tokenization of 3D shapes remains an open question. State-of-the-art (SOTA) methods primarily rely on geometric level-of-detail (LoD) hierarchies, originally designed for rendering and compression. These spatial hierarchies are often token-inefficient and lack semantic coherence for AR modeling. We propose Level-of-Semantics Tokenization (LoST), which orders tokens by semantic salience, such that early prefixes decode into complete, plausible shapes that possess principal semantics, while subsequent tokens refine instance-specific geometric and semantic details. To train LoST, we introduce Relational Inter-Distance Alignment (RIDA), a novel 3D semantic alignment loss that aligns the relational structure of the 3D shape latent space with that of the semantic DINO feature space. Experiments show that LoST achieves SOTA reconstruction, surpassing previous LoD-based...
  </details>  
- **[TAPESTRY: From Geometry to Appearance via Consistent Turntable Videos](https://arxiv.org/abs/2603.17735v1)**  
  Authors: Yan Zeng, Haoran Jiang, Kaixin Yao, Qixuan Zhang, Longwen Zhang, Lan Xu, Jingyi Yu  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2603.17735v1.pdf)  
  <details><summary>Abstract</summary>

  Automatically generating photorealistic and self-consistent appearances for untextured 3D models is a critical challenge in digital content creation. The advancement of large-scale video generation models offers a natural approach: directly synthesizing 360-degree turntable videos (TTVs), which can serve not only as high-quality dynamic previews but also as an intermediate representation to drive texture synthesis and neural rendering. However, existing general-purpose video diffusion models struggle to maintain strict geometric consistency and appearance stability across the full range of views, making their outputs ill-suited for high-quality 3D reconstruction. To this end, we introduce TAPESTRY, a framework for generating high-fidelity TTVs conditioned on explicit 3D geometry. We reframe the 3D appearance generation task as a geometry-conditioned video diffusion problem: given a 3D mesh, we first render and encode multi-modal geometric features to constrain the video generation process with pixel-level precision, thereby enabling the creation of high-quality and consistent TTVs. Building upon this, we...
  </details>  
  Keywords: texture synthesis  
- **[SegviGen: Repurposing 3D Generative Model for Part Segmentation](https://arxiv.org/abs/2603.16869v3)**  
  Authors: Lin Li, Haoran Feng, Zehuan Huang, Haohua Chen, Wenbo Nie, Shaohua Hou, Keqing Fan, Pan Hu, Sheng Wang, Buyu Li, Lu Sheng  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2603.16869v3.pdf) | [![Project](https://img.shields.io/badge/-Project-blue)](https://fenghora.github.io/SegviGen-Page)  
  <details><summary>Abstract</summary>

  We introduce SegviGen, a framework that repurposes native 3D generative models for 3D part segmentation. Existing pipelines either lift strong 2D priors into 3D via distillation or multi-view mask aggregation, often suffering from cross-view inconsistency and blurred boundaries, or explore native 3D discriminative segmentation, which typically requires large-scale annotated 3D data and substantial training resources. In contrast, SegviGen leverages the structured priors encoded in pretrained 3D generative model to induce segmentation through distinctive part colorization, establishing a novel and efficient framework for part segmentation. Specifically, SegviGen encodes a 3D asset and predicts part-indicative colors on active voxels of a geometry-aligned reconstruction. It supports interactive part segmentation, full segmentation, and full segmentation with 2D guidance in a unified framework. Extensive experiments show that SegviGen improves over the prior state of the art by 40% on interactive part segmentation and by 15% on full segmentation, while using only 0.32% of the labeled...
  </details>  
- **[MessyKitchens: Contact-rich object-level 3D scene reconstruction](https://arxiv.org/abs/2603.16868v1)**  
  Authors: Junaid Ahmed Ansari, Ran Ding, Fabio Pizzati, Ivan Laptev  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2603.16868v1.pdf) | [![Project](https://img.shields.io/badge/-Project-blue)](https://messykitchens.github.io)  
  <details><summary>Abstract</summary>

  Monocular 3D scene reconstruction has recently seen significant progress. Powered by the modern neural architectures and large-scale data, recent methods achieve high performance in depth estimation from a single image. Meanwhile, reconstructing and decomposing common scenes into individual 3D objects remains a hard challenge due to the large variety of objects, frequent occlusions and complex object relations. Notably, beyond shape and pose estimation of individual objects, applications in robotics and animation require physically-plausible scene reconstruction where objects obey physical principles of non-penetration and realistic contacts. In this work we advance object-level scene reconstruction along two directions. First, we introduceMessyKitchens, a new dataset with real-world scenes featuring cluttered environments and providing high-fidelity object-level ground truth in terms of 3D object shapes, poses and accurate object contacts. Second, we build on the recent SAM 3D approach for single-object reconstruction and extend it with Multi-Object Decoder (MOD) for joint object-level scene reconstruction. To...
  </details>  
  Keywords: 3d scene reconstruction  
- **[Retrieval-Augmented Sketch-Guided 3D Building Generation](https://arxiv.org/abs/2603.16612v1)**  
  Authors: Zhengyang Wang, Nuttapong Rochanavibhata, Yuxiao Ren, Xusheng Du, Ye Zhang, Haoran Xie  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2603.16612v1.pdf)  
  <details><summary>Abstract</summary>

  In the early design stage of Japanese detached houses, the lack of a unified design representation among clients, sales representatives, and designers leads to design drift and inefficient feedback. Usually, sketches handed off by sales representatives may lose details for quick drawing, which reduces the fidelity of subsequent 3D generation using generative AI models. The generated 3D model typically takes the form of a single unified mesh, preventing component-level editing. To solve these issues, we propose a multi-stage 3D generative design framework capable of producing architectural models from rough design sketches. The framework combines generative and retrieval-based methods to enable component-level editing and personalized customization. It adopts a multimodal representation for 3D model generation and applies component segmentation to localize architectural components such as windows and doors and uses retrieval to support targeted replacement of components. Experiments show that the work enables modular customization which is thought to be suitable...
  </details>  
- **[Interact3D: Compositional 3D Generation of Interactive Objects](https://arxiv.org/abs/2603.16085v1)**  
  Authors: Hui Shan, Keyang Luo, Ming Li, Sizhe Zheng, Yanwei Fu, Zhen Chen, Xiangru Huang  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2603.16085v1.pdf)  
  <details><summary>Abstract</summary>

  Recent breakthroughs in 3D generation have enabled the synthesis of high-fidelity individual assets. However, generating 3D compositional objects from single images--particularly under occlusions--remains challenging. Existing methods often degrade geometric details in hidden regions and fail to preserve the underlying object-object spatial relationships (OOR). We present a novel framework Interact3D designed to generate physically plausible interacting 3D compositional objects. Our approach first leverages advanced generative priors to curate high-quality individual assets with a unified 3D guidance scene. To physically compose these assets, we then introduce a robust two-stage composition pipeline. Based on the 3D guidance scene, the primary object is anchored through precise global-to-local geometric alignment (registration), while subsequent geometries are integrated using a differentiable Signed Distance Field (SDF)-based optimization that explicitly penalizes geometry intersections. To reduce challenging collisions, we further deploy a closed-loop, agentic refinement strategy. A Vision-Language Model (VLM) autonomously analyzes multi-view renderings of the composed scene, formulates targeted...
  </details>  
- **[PiGRAND: Physics-informed Graph Neural Diffusion for Intelligent Additive Manufacturing](https://arxiv.org/abs/2603.15194v1)**  
  Authors: Benjamin Uhrich, Tim Häntschel, Erhard Rahm  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2603.15194v1.pdf) | [![GitHub](https://img.shields.io/github/stars/bu32loxa/PiGRAND?style=social)](https://github.com/bu32loxa/PiGRAND)  
  <details><summary>Abstract</summary>

  A comprehensive understanding of heat transport is essential for optimizing various mechanical and engineering applications, including 3D printing. Recent advances in machine learning, combined with physics-based models, have enabled a powerful fusion of numerical methods and data-driven algorithms. This progress is driven by the availability of limited sensor data in various engineering and scientific domains, where the cost of data collection and the inaccessibility of certain measurements are high. To this end, we present PiGRAND, a Physics-informed graph neural diffusion framework. In order to reduce the computational complexity of graph learning, an efficient graph construction procedure was developed. Our approach is inspired by the explicit Euler and implicit Crank-Nicolson methods for modeling continuous heat transport, leveraging sub-learning models to secure the accurate diffusion across graph nodes. To enhance computational performance, our approach is combined with efficient transfer learning. We evaluate PiGRAND on thermal images from 3D printing, demonstrating significant improvements...
  </details>  
- **[OCRA: Object-Centric Learning with 3D and Tactile Priors for Human-to-Robot Action Transfer](https://arxiv.org/abs/2603.14401v1)**  
  Authors: Kuanning Wang, Ke Fan, Yuqian Fu, Siyu Lin, Hu Luo, Daniel Seita, Yanwei Fu, Yu-Gang Jiang, Xiangyang Xue  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2603.14401v1.pdf)  
  <details><summary>Abstract</summary>

  We present OCRA, an Object-Centric framework for video-based human-to-Robot Action transfer that learns directly from human demonstration videos to enable robust manipulation. Object-centric learning emphasizes task-relevant objects and their interactions while filtering out irrelevant background, providing a natural and scalable way to teach robots. OCRA leverages multi-view RGB videos, the state-of-the-art 3D foundation model VGGT, and advanced detection and segmentation models to reconstruct object-centric 3D point clouds, capturing rich interactions between objects. To handle properties not easily perceived by vision alone, we incorporate tactile priors via a large-scale dataset of over one million tactile images. These 3D and tactile priors are fused through a multimodal module (ResFiLM) and fed into a Diffusion Policy to generate robust manipulation actions. Extensive experiments on both vision-only and visuo-tactile tasks show that OCRA significantly outperforms existing baselines and ablations, demonstrating its effectiveness for learning from human demonstration videos.
  </details>  
- **[In-Field 3D Wheat Head Instance Segmentation From TLS Point Clouds Using Deep Learning Without Manual Labels](https://arxiv.org/abs/2603.14309v1)**  
  Authors: Tomislav Medic, Liangliang Nan  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2603.14309v1.pdf)  
  <details><summary>Abstract</summary>

  3D instance segmentation for laser scanning (LiDAR) point clouds remains a challenge in many remote sensing-related domains. Successful solutions typically rely on supervised deep learning and manual annotations, and consequently focus on objects that can be well delineated through visual inspection and manual labeling of point clouds. However, for tasks with more complex and cluttered scenes, such as in-field plant phenotyping in agriculture, such approaches are often infeasible. In this study, we tackle the task of in-field wheat head instance segmentation directly from terrestrial laser scanning (TLS) point clouds. To address the problem and circumvent the need for manual annotations, we propose a novel two-stage pipeline. To obtain the initial 3D instance proposals, the first stage uses 3D-to-2D multi-view projections, the Grounded SAM pipeline for zero-shot 2D object-centric segmentation, and multi-view label fusion. The second stage uses these initial proposals as noisy pseudo-labels to train a supervised 3D panoptic-style segmentation...
  </details>  
- **[OAHuman: Occlusion-Aware 3D Human Reconstruction from Monocular Images](https://arxiv.org/abs/2603.14249v1)**  
  Authors: Yuanwang Yang, Hongliang Liu, Muxin Zhang, Nan Ma, Jingyu Yang, Yu-Kun Lai, Kun Li  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2603.14249v1.pdf)  
  <details><summary>Abstract</summary>

  Monocular 3D human reconstruction in real-world scenarios remains highly challenging due to frequent occlusions from surrounding objects, people, or image truncation. Such occlusions lead to missing geometry and unreliable appearance cues, severely degrading the completeness and realism of reconstructed human models. Although recent neural implicit methods achieve impressive results on clean inputs, they struggle under occlusion due to entangled modeling of shape and texture. In this paper, we propose OAHuman, an occlusion-aware framework that explicitly decouples geometry reconstruction and texture synthesis for robust 3D human modeling from a single RGB image. The core innovation lies in the decoupling-perception paradigm, which addresses the fundamental issue of geometry-texture cross-contamination in occluded regions. Our framework ensures that geometry reconstruction is perceptually reinforced even in occluded areas, isolating it from texture interference. In parallel, texture synthesis is learned exclusively from visible regions, preventing texture errors from being transferred to the occluded areas. This decoupling...
  </details>  
  Keywords: texture synthesis  
- **[SK-Adapter: Skeleton-Based Structural Control for Native 3D Generation](https://arxiv.org/abs/2603.14152v1)**  
  Authors: Anbang Wang, Yuzhuo Ao, Shangzhe Wu, Chi-Keung Tang  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2603.14152v1.pdf) | [![Project](https://img.shields.io/badge/-Project-blue)](https://sk-adapter.github.io)  
  <details><summary>Abstract</summary>

  Native 3D generative models have achieved remarkable fidelity and speed, yet they suffer from a critical limitation: inability to prescribe precise structural articulations, where precise structural control within the native 3D space remains underexplored. This paper proposes SK-Adapter, a simple and yet highly efficient and effective framework that unlocks precise skeletal manipulation for native 3D generation. Moving beyond text or image prompts, which can be ambiguous for precise structure, we treat the 3D skeleton as a first-class control signal. SK-Adapter is a lightweight structural adapter network that encodes joint coordinates and topology into learnable tokens, which are injected into the frozen 3D generation backbone via cross-attention. This smart design allows the model to not only effectively "attend" to specific 3D structural constraints but also preserve its original generative priors. To bridge the data gap, we contribute Objaverse-TMS dataset, a large-scale dataset of 24k text-mesh-skeleton pairs. Extensive experiments confirm that our...
  </details>  
- **[EI-Part: Explode for Completion and Implode for Refinement](https://arxiv.org/abs/2603.14021v1)**  
  Authors: Wanhu Sun, Zhongjin Luo, Heliang Zheng, Jiahao Chang, Chongjie Ye, Huiang He, Shengchu Zhao, Rongfei Jia, Xiaoguang Han  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2603.14021v1.pdf) | [![Project](https://img.shields.io/badge/-Project-blue)](https://cvhadessun.github.io/EI-Part)  
  <details><summary>Abstract</summary>

  Part-level 3D generation is crucial for various downstream applications, including gaming, film production, and industrial design. However, decomposing a 3D shape into geometrically plausible and meaningful components remains a significant challenge. Previous part-based generation methods often struggle to produce well-constructed parts, exhibiting poor structural coherence, geometric implausibility, inaccuracy, or inefficiency. To address these challenges, we introduce EI-Part, a novel framework specifically designed to generate high-quality 3D shapes with components, characterized by strong structural coherence, geometric plausibility, geometric fidelity, and generation efficiency. We propose utilizing distinct representations at different stages: an Explode state for part completion and an Implode state for geometry refinement. This strategy fully leverages spatial resolution, enabling flexible part completion and fine geometric detail generation. To maintain structural coherence between parts, a self-attention mechanism is incorporated in both exploded and imploded states, facilitating effective information perception and feature fusion among components during generation. Extensive experiments on multiple benchmarks...
  </details>  
- **[Scene Generation at Absolute Scale: Utilizing Semantic and Geometric Guidance From Text for Accurate and Interpretable 3D Indoor Scene Generation](https://arxiv.org/abs/2603.13910v1)**  
  Authors: Stefan Ainetter, Thomas Deixelberger, Edoardo A. Dominici, Philipp Drescher, Konstantinos Vardis, Markus Steinberger  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2603.13910v1.pdf)  
  <details><summary>Abstract</summary>

  We present GuidedSceneGen, a text-to-3D generation framework that produces metrically accurate, globally consistent, and semantically interpretable indoor scenes. Unlike prior text-driven methods that often suffer from geometric drift or scale ambiguity, our approach maintains an absolute world coordinate frame throughout the entire generation process. Starting from a textual scene description, we predict a global 3D layout encoding both semantic and geometric structure, which serves as a guiding proxy for downstream stages. A semantics- and depth-conditioned panoramic diffusion model then synthesizes 360° imagery aligned with the global layout, substantially improving spatial coherence. To explore unobserved regions, we employ a video diffusion model guided by optimized camera trajectories that balances coverage and collision avoidance, achieving up to 10x faster sampling compared to exhaustive path exploration. The generated views are fused using 3D Gaussian Splatting, yielding a consistent and fully navigable 3D scene in absolute scale. GuidedSceneGen enables accurate transfer of object poses...
  </details>  
  Keywords: text-to-3d  
- **[SldprtNet: A Large-Scale Multimodal Dataset for CAD Generation in Language-Driven 3D Design](https://arxiv.org/abs/2603.13098v1)**  
  Authors: Ruogu Li, Sikai Li, Yao Mu, Mingyu Ding  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2603.13098v1.pdf)  
  <details><summary>Abstract</summary>

  We introduce SldprtNet, a large-scale dataset comprising over 242,000 industrial parts, designed for semantic-driven CAD modeling, geometric deep learning, and the training and fine-tuning of multimodal models for 3D design. The dataset provides 3D models in both .step and .sldprt formats to support diverse training and testing. To enable parametric modeling and facilitate dataset scalability, we developed supporting tools, an encoder and a decoder, which support 13 types of CAD commands and enable lossless transformation between 3D models and a structured text representation. Additionally, each sample is paired with a composite image created by merging seven rendered views from different viewpoints of the 3D model, effectively reducing input token length and accelerating inference. By combining this image with the parameterized text output from the encoder, we employ the lightweight multimodal language model Qwen2.5-VL-7B to generate a natural language description of each part's appearance and functionality. To ensure accuracy, we manually...
  </details>  
- **[InterEdit: Navigating Text-Guided Multi-Human 3D Motion Editing](https://arxiv.org/abs/2603.13082v1)**  
  Authors: Yebin Yang, Di Wen, Lei Qi, Weitong Kong, Junwei Zheng, Ruiping Liu, Yufan Chen, Chengzhi Wu, Kailun Yang, Yuqian Fu, Danda Pani Paudel, Luc Van Gool, Kunyu Peng  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2603.13082v1.pdf) | [![GitHub](https://img.shields.io/github/stars/YNG916/InterEdit?style=social)](https://github.com/YNG916/InterEdit)  
  <details><summary>Abstract</summary>

  Text-guided 3D motion editing has seen success in single-person scenarios, but its extension to multi-person settings is less explored due to limited paired data and the complexity of inter-person interactions. We introduce the task of multi-person 3D motion editing, where a target motion is generated from a source and a text instruction. To support this, we propose InterEdit3D, a new dataset with manual two-person motion change annotations, and a Text-guided Multi-human Motion Editing (TMME) benchmark. We present InterEdit, a synchronized classifier-free conditional diffusion model for TMME. It introduces Semantic-Aware Plan Token Alignment with learnable tokens to capture high-level interaction cues and an Interaction-Aware Frequency Token Alignment strategy using DCT and energy pooling to model periodic motion dynamics. Experiments show that InterEdit improves text-to-motion consistency and edit fidelity, achieving state-of-the-art TMME performance. The dataset and code will be released at https://github.com/YNG916/InterEdit.
  </details>  
- **[SceneAssistant: A Visual Feedback Agent for Open-Vocabulary 3D Scene Generation](https://arxiv.org/abs/2603.12238v1)**  
  Authors: Jun Luo, Jiaxiang Tang, Ruijie Lu, Gang Zeng  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2603.12238v1.pdf) | [![GitHub](https://img.shields.io/github/stars/ROUJINN/SceneAssistant?style=social)](https://github.com/ROUJINN/SceneAssistant)  
  <details><summary>Abstract</summary>

  Text-to-3D scene generation from natural language is highly desirable for digital content creation. However, existing methods are largely domain-restricted or reliant on predefined spatial relationships, limiting their capacity for unconstrained, open-vocabulary 3D scene synthesis. In this paper, we introduce SceneAssistant, a visual-feedback-driven agent designed for open-vocabulary 3D scene generation. Our framework leverages modern 3D object generation model along with the spatial reasoning and planning capabilities of Vision-Language Models (VLMs). To enable open-vocabulary scene composition, we provide the VLMs with a comprehensive set of atomic operations (e.g., Scale, Rotate, FocusOn). At each interaction step, the VLM receives rendered visual feedback and takes actions accordingly, iteratively refining the scene to achieve more coherent spatial arrangements and better alignment with the input text. Experimental results demonstrate that our method can generate diverse, open-vocabulary, and high-quality 3D scenes. Both qualitative analysis and quantitative human evaluations demonstrate the superiority of our approach over existing methods....
  </details>  
  Keywords: text-to-3d  
- **[Hoi3DGen: Generating High-Quality Human-Object-Interactions in 3D](https://arxiv.org/abs/2603.12126v1)**  
  Authors: Agniv Sharma, Xianghui Xie, Tom Fischer, Eddy Ilg, Gerard Pons-Moll  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2603.12126v1.pdf)  
  <details><summary>Abstract</summary>

  Modeling and generating 3D human-object interactions from text is crucial for applications in AR, XR, and gaming. Existing approaches often rely on score distillation from text-to-image models, but their results suffer from the Janus problem and do not follow text prompts faithfully due to the scarcity of high-quality interaction data. We introduce Hoi3DGen, a framework that generates high-quality textured meshes of human-object interaction that follow the input interaction descriptions precisely. We first curate realistic and high-quality interaction data leveraging multimodal large language models, and then create a full text-to-3D pipeline, which achieves orders-of-magnitude improvements in interaction fidelity. Our method surpasses baselines by 4-15x in text consistency and 3-7x in 3D model quality, exhibiting strong generalization to diverse categories and interaction types, while maintaining high-quality 3D generation.
  </details>  
  Keywords: text-to-3d  
- **[MV-SAM3D: Adaptive Multi-View Fusion for Layout-Aware 3D Generation](https://arxiv.org/abs/2603.11633v2)**  
  Authors: Baicheng Li, Dong Wu, Jun Li, Shunkai Zhou, Zecui Zeng, Lusong Li, Hongbin Zha  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2603.11633v2.pdf) | [![GitHub](https://img.shields.io/github/stars/devinli123/MV-SAM3D?style=social)](https://github.com/devinli123/MV-SAM3D)  
  <details><summary>Abstract</summary>

  Recent unified 3D generation models have made remarkable progress in producing high-quality 3D assets from a single image. Notably, layout-aware approaches such as SAM3D can reconstruct multiple objects while preserving their spatial arrangement, opening the door to practical scene-level 3D generation. However, current methods are limited to single-view input and cannot leverage complementary multi-view observations, while independently estimated object poses often lead to physically implausible layouts such as interpenetration and floating artifacts. We present MV-SAM3D, a training-free framework that extends layout-aware 3D generation with multi-view consistency and physical plausibility. We formulate multi-view fusion as a Multi-Diffusion process in 3D latent space and propose two adaptive weighting strategies -- attention-entropy weighting and visibility weighting -- that enable confidence-aware fusion, ensuring each viewpoint contributes according to its local observation reliability. For multi-object composition, we introduce physics-aware optimization that injects collision and contact constraints both during and after generation, yielding physically plausible object...
  </details>  
- **[Seeing Isn't Orienting: A Cognitively Grounded Benchmark Reveals Systematic Orientation Failures in MLLMs Supplementary](https://arxiv.org/abs/2603.11410v2)**  
  Authors: Nazia Tasnim, Keanu Nichols, Yuting Yang, Nicholas Ikechukwu, Elva Zou, Deepti Ghadiyaram, Bryan A. Plummer  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2603.11410v2.pdf)  
  <details><summary>Abstract</summary>

  Humans learn object orientation progressively, from recognizing which way an object faces, to mentally rotating it, to reasoning about orientations between objects. Current vision-language benchmarks largely conflate orientation with position and general scene understanding. We introduce Discriminative Orientation Reasoning Intelligence (DORI), a cognitively grounded hierarchical benchmark that makes object orientation the primary target. Inspired by stages of human orientation cognition, DORI decomposes orientation into four dimensions, each evaluated at coarse (categorical) and granular (metric) levels. Composed from 13,652 images across 14 sources, DORI provides 33,656 multiple-choice questions covering 67 object categories in real-world and synthetic settings. Its coarse-to-granular design isolates orientation from confounds such as object recognition difficulty, scene clutter, and linguistic ambiguity via bounding-box isolation, standardized spatial reference frames, and structured prompts. Evaluating 24 state-of-the-art vision-language models shows a clear pattern: models that perform well on general spatial benchmarks are near-random on object-centric orientation tasks. The best models reach...
  </details>  
  Keywords: 3d scene reconstruction  
- **[DenoiseSplat: Feed-Forward Gaussian Splatting for Noisy 3D Scene Reconstruction](https://arxiv.org/abs/2603.09291v1)**  
  Authors: Fuzhen Jiang, Zhuoran Li, Yinlin Zhang  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2603.09291v1.pdf)  
  <details><summary>Abstract</summary>

  3D scene reconstruction and novel-view synthesis are fundamental for VR, robotics, and content creation. However, most NeRF and 3D Gaussian Splatting pipelines assume clean inputs and degrade under real noise and artifacts. We therefore propose DenoiseSplat, a feed-forward 3D Gaussian splatting method for noisy multi-view images. We build a large-scale, scene-consistent noisy--clean benchmark on RE10K by injecting Gaussian, Poisson, speckle, and salt-and-pepper noise with controlled intensities. With a lightweight MVSplat-style feed-forward backbone, we train end-to-end using only clean 2D renderings as supervision and no 3D ground truth. On noisy RE10K, DenoiseSplat outperforms vanilla MVSplat and a strong two-stage baseline (IDF + MVSplat) in PSNR/SSIM and LPIPS across noise types and levels.
  </details>  
  Keywords: 3d scene reconstruction  
- **[ForgeDreamer: Industrial Text-to-3D Generation with Multi-Expert LoRA and Cross-View Hypergraph](https://arxiv.org/abs/2603.09266v4)**  
  Authors: Junhao Cai, Deyu Zeng, Junhao Pang, Lini Li, Zongze Wu, Xiaopin Zhong  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2603.09266v4.pdf) | [![GitHub](https://img.shields.io/github/stars/Junhaocai27/ForgeDreamer?style=social)](https://github.com/Junhaocai27/ForgeDreamer)  
  <details><summary>Abstract</summary>

  Current text-to-3D generation methods excel in natural scenes but struggle with industrial applications due to two critical limitations: domain adaptation challenges where conventional LoRA fusion causes knowledge interference across categories, and geometric reasoning deficiencies where pairwise consistency constraints fail to capture higher-order structural dependencies essential for precision manufacturing. We propose a novel framework named ForgeDreamer addressing both challenges through two key innovations. First, we introduce a Multi-Expert LoRA Ensemble mechanism that consolidates multiple category-specific LoRA models into a unified representation, achieving superior cross-category generalization while eliminating knowledge interference. Second, building on enhanced semantic understanding, we develop a Cross-View Hypergraph Geometric Enhancement approach that captures structural dependencies spanning multiple viewpoints simultaneously. These components work synergistically improved semantic understanding, enables more effective geometric reasoning, while hypergraph modeling ensures manufacturing-level consistency. Extensive experiments on a custom industrial dataset demonstrate superior semantic generalization and enhanced geometric fidelity compared to state-of-the-art approaches. Code is available...
  </details>  
  Keywords: text-to-3d  
- **[Spherical-GOF: Geometry-Aware Panoramic Gaussian Opacity Fields for 3D Scene Reconstruction](https://arxiv.org/abs/2603.08503v1)**  
  Authors: Zhe Yang, Guoqiang Zhao, Sheng Wu, Kai Luo, Kailun Yang  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2603.08503v1.pdf) | [![GitHub](https://img.shields.io/github/stars/1170632760/Spherical-GOF?style=social)](https://github.com/1170632760/Spherical-GOF)  
  <details><summary>Abstract</summary>

  Omnidirectional images are increasingly used in robotics and vision due to their wide field of view. However, extending 3D Gaussian Splatting (3DGS) to panoramic camera models remains challenging, as existing formulations are designed for perspective projections and naive adaptations often introduce distortion and geometric inconsistencies. We present Spherical-GOF, an omnidirectional Gaussian rendering framework built upon Gaussian Opacity Fields (GOF). Unlike projection-based rasterization, Spherical-GOF performs GOF ray sampling directly on the unit sphere in spherical ray space, enabling consistent ray-Gaussian interactions for panoramic rendering. To make the spherical ray casting efficient and robust, we derive a conservative spherical bounding rule for fast ray-Gaussian culling and introduce a spherical filtering scheme that adapts Gaussian footprints to distortion-varying panoramic pixel sampling. Extensive experiments on standard panoramic benchmarks (OmniBlender and OmniPhotos) demonstrate competitive photometric quality and substantially improved geometric consistency. Compared with the strongest baseline, Spherical-GOF reduces depth reprojection error by 57% and improves...
  </details>  
  Keywords: 3d scene reconstruction  
- **[GarmentPainter: Efficient 3D Garment Texture Synthesis with Character-Guided Diffusion Model](https://arxiv.org/abs/2603.08228v1)**  
  Authors: Jinbo Wu, Xiaobo Gao, Xing Liu, Chen Zhao, Jialun Liu  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2603.08228v1.pdf)  
  <details><summary>Abstract</summary>

  Generating high-fidelity, 3D-consistent garment textures remains a challenging problem due to the inherent complexities of garment structures and the stringent requirement for detailed, globally consistent texture synthesis. Existing approaches either rely on 2D-based diffusion models, which inherently struggle with 3D consistency, require expensive multi-step optimization or depend on strict spatial alignment between 2D reference images and 3D meshes, which limits their flexibility and scalability. In this work, we introduce GarmentPainter, a simple yet efficient framework for synthesizing high-quality, 3D-aware garment textures in UV space. Our method leverages a UV position map as the 3D structural guidance, ensuring texture consistency across the garment surface during texture generation. To enhance control and adaptability, we introduce a type selection module, enabling fine-grained texture generation for specific garment components based on a character reference image, without requiring alignment between the reference image and the 3D mesh. GarmentPainter efficiently integrates all guidance signals into the...
  </details>  
  Keywords: texture synthesis  
- **[Skill-Evolving Grounded Reasoning for Free-Text Promptable 3D Medical Image Segmentation](https://arxiv.org/abs/2603.08215v1)**  
  Authors: Tongrui Zhang, Chenhui Wang, Yongming Li, Zhihao Chen, Xufeng Zhan, Hongming Shan  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2603.08215v1.pdf)  
  <details><summary>Abstract</summary>

  Free-text promptable 3D medical image segmentation offers an intuitive and clinically flexible interaction paradigm. However, current methods are highly sensitive to linguistic variability: minor changes in phrasing can cause substantial performance degradation despite identical clinical intent. Existing approaches attempt to improve robustness through stronger vision-language fusion or larger vocabularies, yet they lack mechanisms to consistently align ambiguous free-form expressions with anatomically grounded representations. We propose Skill-Evolving grounded Reasoning (SEER), a novel framework for free-text promptable 3D medical image segmentation that explicitly bridges linguistic variability and anatomical precision through a reasoning-driven design. First, we curate the SEER-Trace dataset, which pairs raw clinical requests with image-grounded, skill-tagged reasoning traces, establishing a reproducible benchmark. Second, SEER constructs an evidence-aligned target representation via a vision-language reasoning chain that verifies clinical intent against image-derived anatomical evidence, thereby enforcing semantic consistency before voxel-level decoding. Third, we introduce SEER-Loop, a dynamic skill-evolving strategy that distills high-reward reasoning...
  </details>  
- **[4DRC-OCC: Robust Semantic Occupancy Prediction Through Fusion of 4D Radar and Camera](https://arxiv.org/abs/2603.07794v1)**  
  Authors: David Ninfa, Andras Palffy, Holger Caesar  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2603.07794v1.pdf)  
  <details><summary>Abstract</summary>

  Autonomous driving requires robust perception across diverse environmental conditions, yet 3D semantic occupancy prediction remains challenging under adverse weather and lighting. In this work, we present the first study combining 4D radar and camera data for 3D semantic occupancy prediction. Our fusion leverages the complementary strengths of both modalities: 4D radar provides reliable range, velocity, and angle measurements in challenging conditions, while cameras contribute rich semantic and texture information. We further show that integrating depth cues from camera pixels enables lifting 2D images to 3D, improving scene reconstruction accuracy. Additionally, we introduce a fully automatically labeled dataset for training semantic occupancy models, substantially reducing reliance on costly manual annotation. Experiments demonstrate the robustness of 4D radar across diverse scenarios, highlighting its potential to advance autonomous vehicle perception.
  </details>  
- **[PARSE: Part-Aware Relational Spatial Modeling](https://arxiv.org/abs/2603.07704v2)**  
  Authors: Yinuo Bai, Peijun Xu, Kuixiang Shao, Yuyang Jiao, Jingxuan Zhang, Kaixin Yao, Jiayuan Gu, Jingyi Yu  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2603.07704v2.pdf)  
  <details><summary>Abstract</summary>

  Inter-object relations underpin spatial intelligence, yet existing representations -- linguistic prepositions or object-level scene graphs -- are too coarse to specify which regions actually support, contain, or contact one another, leading to ambiguous and physically inconsistent layouts. To address these ambiguities, a part-level formulation is needed; therefore, we introduce PARSE, a framework that explicitly models how object parts interact to determine feasible and spatially grounded scene configurations. PARSE centers on the Part-centric Assembly Graph (PAG), which encodes geometric relations between specific object parts, and a Part-Aware Spatial Configuration Solver that converts these relations into geometric constraints to assemble collision-free, physically valid scenes. Using PARSE, we build PARSE-10K, a dataset of 10,000 3D indoor scenes constructed from real-image layout priors and a curated part-annotated shape database, each with dense contact structures and a part-level contact graph. With this structured, spatially grounded supervision, fine-tuning Qwen3-VL on PARSE-10K yields stronger object-level layout reasoning...
  </details>  
- **[3DGS-HPC: Distractor-free 3D Gaussian Splatting with Hybrid Patch-wise Classification](https://arxiv.org/abs/2603.07587v1)**  
  Authors: Jiahao Chen, Yipeng Qin, Ganlong Zhao, Xin Li, Wenping Wang, Guanbin Li  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2603.07587v1.pdf)  
  <details><summary>Abstract</summary>

  3D Gaussian Splatting (3DGS) has demonstrated remarkable performance in novel view synthesis and 3D scene reconstruction, yet its quality often degrades in real-world environments due to transient distractors, such as moving objects and varying shadows. Existing methods commonly rely on semantic cues extracted from pre-trained vision models to identify and suppress these distractors, but such semantics are misaligned with the binary distinction between static and transient regions and remain fragile under the appearance perturbations introduced during 3DGS optimization. We propose 3DGS-HPC, a framework that circumvents these limitations by combining two complementary principles: a patch-wise classification strategy that leverages local spatial consistency for robust region-level decisions, and a hybrid classification metric that adaptively integrates photometric and perceptual cues for more reliable separation. Extensive experiments demonstrate the superiority and robustness of our method in mitigating distractors to improve 3DGS-based novel view synthesis.
  </details>  
  Keywords: 3d scene reconstruction, novel view synthesis  
- **[DogWeave: High-Fidelity 3D Canine Reconstruction from a Single Image via Normal Fusion and Conditional Inpainting](https://arxiv.org/abs/2603.07441v1)**  
  Authors: Shufan Sun, Chenchen Wang, Zongfu Yu  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2603.07441v1.pdf)  
  <details><summary>Abstract</summary>

  Monocular 3D animal reconstruction is challenging due to complex articulation, self-occlusion, and fine-scale details such as fur. Existing methods often produce distorted geometry and inconsistent textures due to the lack of articulated 3D supervision and limited availability of back-view images in 2D datasets, which makes reconstructing unobserved regions particularly difficult. To address these limitations, we propose DogWeave, a model-based framework for reconstructing high-fidelity 3D canine models from a single RGB image. DogWeave improves geometry by refining a coarsely-initiated parametric mesh into a detailed SDF representation through multi-view normal field optimization using diffusion-enhanced normals. It then generates view-consistent textures through conditional partial inpainting guided by structure and style cues, enabling realistic reconstruction of unobserved regions. Using only about 7,000 dog images processed via our 2D pipeline for training, DogWeave produces complete, realistic 3D models and outperforms state-of-the-art single image to 3d reconstruction methods in both shape accuracy and texture realism for...
  </details>  
- **[CanoVerse: 3D Object Scalable Canonicalization and Dataset for Generation and Pose](https://arxiv.org/abs/2603.07144v1)**  
  Authors: Li Jin, Yuchen Yang, Weikai Chen, Yujie Wang, Dehao Hao, Tanghui Jia, Yingda Yin, Zeyu Hu, Runze Zhang, Keyang Luo, Li Yuan, Long Quan, Xin Wang, Xueying Qin  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2603.07144v1.pdf) | [![GitHub](https://img.shields.io/github/stars/123321456-gif/Canoverse?style=social)](https://github.com/123321456-gif/Canoverse)  
  <details><summary>Abstract</summary>

  3D learning systems implicitly assume that objects occupy a coherent reference frame. Nonetheless, in practice, every asset arrives with an arbitrary global rotation, and models are left to resolve directional ambiguity on their own. This persistent misalignment suppresses pose-consistent generation, and blocks the emergence of stable directional semantics. To address this issue, we construct \methodName{}, a massive canonical 3D dataset of 320K objects over 1,156 categories -- an order-of-magnitude increase over prior work. At this scale, directional semantics become statistically learnable: Canoverse improves 3D generation stability, enables precise cross-modal 3D shape retrieval, and unlocks zero-shot point-cloud orientation estimation even for out-of-distribution data. This is achieved by a new canonicalization framework that reduces alignment from minutes to seconds per object via compact hypothesis generation and lightweight human discrimination, transforming canonicalization from manual curation into a high-throughput data generation pipeline. The Canoverse dataset will be publicly released upon acceptance. Project page: https://github.com/123321456-gif/Canoverse
  </details>  
- **[Optimizing Multi-Modal Models for Image-Based Shape Retrieval: The Role of Pre-Alignment and Hard Contrastive Learning](https://arxiv.org/abs/2603.06982v1)**  
  Authors: Paul Julius Kühn, Cedric Spengler, Michael Weinmann, Arjan Kuijper, Saptarshi Neil Sinha  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2603.06982v1.pdf)  
  <details><summary>Abstract</summary>

  Image-based shape retrieval (IBSR) aims to retrieve 3D models from a database given a query image, hence addressing a classical task in computer vision, computer graphics, and robotics. Recent approaches typically rely on bridging the domain gap between 2D images and 3D shapes based on the use of multi-view renderings as well as task-specific metric learning to embed shapes and images into a common latent space. In contrast, we address IBSR through large-scale multi-modal pretraining and show that explicit view-based supervision is not required. Inspired by pre-aligned image--point-cloud encoders from ULIP and OpenShape that have been used for tasks such as 3D shape classification, we propose the use of pre-aligned image and shape encoders for zero-shot and standard IBSR by embedding images and point clouds into a shared representation space and performing retrieval via similarity search over compact single-embedding shape descriptors. This formulation allows skipping view synthesis and naturally enables...
  </details>  
- **[Rewis3d: Reconstruction Improves Weakly-Supervised Semantic Segmentation](https://arxiv.org/abs/2603.06374v1)**  
  Authors: Jonas Ernst, Wolfgang Boettcher, Lukas Hoyer, Jan Eric Lenssen, Bernt Schiele  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2603.06374v1.pdf)  
  <details><summary>Abstract</summary>

  We present Rewis3d, a framework that leverages recent advances in feed-forward 3D reconstruction to significantly improve weakly supervised semantic segmentation on 2D images. Obtaining dense, pixel-level annotations remains a costly bottleneck for training segmentation models. Alleviating this issue, sparse annotations offer an efficient weakly-supervised alternative. However, they still incur a performance gap. To address this, we introduce a novel approach that leverages 3D scene reconstruction as an auxiliary supervisory signal. Our key insight is that 3D geometric structure recovered from 2D videos provides strong cues that can propagate sparse annotations across entire scenes. Specifically, a dual student-teacher architecture enforces semantic consistency between 2D images and reconstructed 3D point clouds, using state-of-the-art feed-forward reconstruction to generate reliable geometric supervision. Extensive experiments demonstrate that Rewis3d achieves state-of-the-art performance in sparse supervision, outperforming existing approaches by 2-7% without requiring additional labels or inference overhead.
  </details>  
  Keywords: 3d scene reconstruction  
- **[JOPP-3D: Joint Open Vocabulary Semantic Segmentation on Point Clouds and Panoramas](https://arxiv.org/abs/2603.06168v2)**  
  Authors: Sandeep Inuganti, Hideaki Kanayama, Kanta Shimizu, Mahdi Chamseddine, Soichiro Yokota, Didier Stricker, Jason Rambach  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2603.06168v2.pdf)  
  <details><summary>Abstract</summary>

  Semantic segmentation across visual modalities such as 3D point clouds and panoramic images remains a challenging task, primarily due to the scarcity of annotated data and the limited adaptability of fixed-label models. In this paper, we present JOPP-3D, an open-vocabulary semantic segmentation framework that jointly leverages panoramic and point cloud data to enable language-driven scene understanding. We convert RGB-D panoramic images into their corresponding tangential perspective images and 3D point clouds, then use these modalities to extract and align foundational vision-language features. This allows natural language querying to generate semantic masks on both input modalities. Experimental evaluation on the Stanford-2D-3D-s and ToF-360 datasets demonstrates the capability of JOPP-3D to produce coherent and semantically meaningful segmentations across panoramic and 3D domains. Our proposed method achieves a significant improvement compared to the SOTA in open and closed vocabulary 2D and 3D semantic segmentation.
  </details>  
- **[Pano3DComposer: Feed-Forward Compositional 3D Scene Generation from Single Panoramic Image](https://arxiv.org/abs/2603.05908v1)**  
  Authors: Zidian Qiu, Ancong Wu  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2603.05908v1.pdf) | [![Project](https://img.shields.io/badge/-Project-blue)](https://qiuzidian.github.io/pano3dcomposer-page)  
  <details><summary>Abstract</summary>

  Current compositional image-to-3D scene generation approaches construct 3D scenes by time-consuming iterative layout optimization or inflexible joint object-layout generation. Moreover, most methods rely on limited field-of-view perspective images, hindering the creation of complete 360-degree environments. To address these limitations, we design Pano3DComposer, an efficient feed-forward framework for panoramic images. To decouple object generation from layout estimation, we propose a plug-and-play Object-World Transformation Predictor. This module converts the 3D objects generated by off-the-shelf image-to-3D models from local to world coordinates. To achieve this, we adapt the VGGT architecture to Alignment-VGGT by using target object crop, multi-view object renderings and camera parameters to predict the transformation. The predictor is trained using pseudo-geometric supervision to address the shape discrepancy between generated and ground-truth objects. For input images from unseen domains, we further introduce a Coarse-to-Fine (C2F) alignment mechanism for Pano3DComposer that iteratively refines geometric consistency with feedback of scene rendering. Our method achieves...
  </details>  
  Keywords: text-to-3d, image-to-3d  
- **[Cog2Gen3D: Sculpturing 3D Semantic-Geometric Cognition for 3D Generation](https://arxiv.org/abs/2603.05845v1)**  
  Authors: Haonan Wang, Hanyu Zhou, Haoyue Liu, Tao Gu, Luxin Yan  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2603.05845v1.pdf)  
  <details><summary>Abstract</summary>

  Generative models have achieved success in producing semantically plausible 2D images, but it remains challenging in 3D generation due to the absence of spatial geometry constraints. Typically, existing methods utilize geometric features as conditions to enhance spatial awareness. However, these methods can only model relative relationships and are prone to scale inconsistency of absolute geometry. Thus, we argue that semantic information and absolute geometry empower 3D cognition, thereby enabling controllable 3D generation for the physical world. In this work, we propose Cog2Gen3D, a 3D cognition-guided diffusion framework for 3D generation. Our model is guided by three key designs: 1) Cognitive Feature Embeddings. We encode different modalities into semantic and geometric representations and further extract logical representations. 2) 3D Latent Cognition Graph. We structure different representations into dual-stream semantic-geometric graphs and fuse them via common-based cross-attention to obtain a 3D cognition graph. 3) Cognition-Guided Latent Diffusion. We leverage the fused 3D...
  </details>  
  Keywords: controllable 3d generation  
- **[Spectral Probing of Feature Upsamplers in 2D-to-3D Scene Reconstruction](https://arxiv.org/abs/2603.05787v1)**  
  Authors: Ling Xiao, Yuliang Xiu, Yue Chen, Guoming Wang, Toshihiko Yamasaki  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2603.05787v1.pdf)  
  <details><summary>Abstract</summary>

  A typical 2D-to-3D pipeline takes multi-view images as input, where a Vision Foundation Model (VFM) extracts features that are spatially upsampled to dense representations for 3D reconstruction. If dense features across views preserve geometric consistency, differentiable rendering can recover an accurate 3D representation, making the feature upsampler a critical component. Recent learnable upsampling methods mainly aim to enhance spatial details, such as sharper geometry or richer textures, yet their impact on 3D awareness remains underexplored. To address this gap, we introduce a spectral diagnostic framework with six complementary metrics that characterize amplitude redistribution, structural spectral alignment, and directional stability. Across classical interpolation and learnable upsampling methods on CLIP and DINO backbones, we observe three key findings. First, structural spectral consistency (SSC/CSC) is the strongest predictor of NVS quality, whereas High-Frequency Spectral Slope Drift (HFSS) often correlates negatively with reconstruction performance, indicating that emphasizing high-frequency details alone does not necessarily improve...
  </details>  
  Keywords: 3d scene reconstruction  
- **[OWL: A Novel Approach to Machine Perception During Motion](https://arxiv.org/abs/2603.05686v1)**  
  Authors: Daniel Raviv, Juan D. Yepes  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2603.05686v1.pdf)  
  <details><summary>Abstract</summary>

  We introduce a perception-related function, OWL, designed to address the complex challenges of 3D perception during motion. It derives its values directly from two fundamental visual motion cues, with one set of cue values per point per time instant. During motion, two visual motion cues relative to a fixation point emerge: 1) perceived local visual looming of points near the fixation point, and 2) perceived rotation of the rigid object relative to the fixation point. It also expresses the relation between two well-known physical quantities, the relative instantaneous directional range and directional translation in 3D between the camera and any visible 3D point, without explicitly requiring their measurement or prior knowledge of their individual values. OWL offers a unified, analytical time-based approach that enhances and simplifies key perception capabilities, including scaled 3D mapping and camera heading. Simulations demonstrate that OWL achieves geometric constancy of 3D objects over time and enables...
  </details>  
  Keywords: 3d scene reconstruction  
- **[Towards 3D Scene Understanding of Gas Plumes in LWIR Hyperspectral Images Using Neural Radiance Fields](https://arxiv.org/abs/2603.05473v1)**  
  Authors: Scout Jarman, Zigfried Hampel-Arias, Adra Carr, Kevin R. Moon  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2603.05473v1.pdf)  
  <details><summary>Abstract</summary>

  Hyperspectral images (HSI) have many applications, ranging from environmental monitoring to national security, and can be used for material detection and identification. Longwave infrared (LWIR) HSI can be used for gas plume detection and analysis. Oftentimes, only a few images of a scene of interest are available and are analyzed individually. The ability to combine information from multiple images into a single, cohesive representation could enhance analysis by providing more context on the scene's geometry and spectral properties. Neural radiance fields (NeRFs) create a latent neural representation of volumetric scene properties that enable novel-view rendering and geometry reconstruction, offering a promising avenue for hyperspectral 3D scene reconstruction. We explore the possibility of using NeRFs to create 3D scene reconstructions from LWIR HSI and demonstrate that the model can be used for the basic downstream analysis task of gas plume detection. The physics-based DIRSIG software suite was used to generate a...
  </details>  
  Keywords: 3d scene reconstruction  
- **[RelaxFlow: Text-Driven Amodal 3D Generation](https://arxiv.org/abs/2603.05425v2)**  
  Authors: Jiayin Zhu, Guoji Fu, Xiaolu Liu, Qiyuan He, Yicong Li, Angela Yao  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2603.05425v2.pdf)  
  <details><summary>Abstract</summary>

  Image-to-3D generation faces inherent semantic ambiguity under occlusion, where partial observation alone is often insufficient to determine object category. In this work, we formalize text-driven amodal 3D generation, where text prompts steer the completion of unseen regions while strictly preserving input observation. Crucially, we identify that these objectives demand distinct control granularities: rigid control for the observation versus relaxed structural control for the prompt. To this end, we propose RelaxFlow, a training-free dual-branch framework that decouples control granularity via a Multi-Prior Consensus Module and a Relaxation Mechanism. Theoretically, we prove that our relaxation is equivalent to applying a low-pass filter on the generative vector field, which suppresses high-frequency instance details to isolate geometric structure that accommodates the observation. To facilitate evaluation, we introduce two diagnostic benchmarks, ExtremeOcc-3D and AmbiSem-3D. Extensive experiments demonstrate that RelaxFlow successfully steers the generation of unseen regions to match the prompt intent without compromising visual fidelity.
  </details>  
  Keywords: image-to-3d  
- **[MultiGO++: Monocular 3D Clothed Human Reconstruction via Geometry-Texture Collaboration](https://arxiv.org/abs/2603.04993v1)**  
  Authors: Nanjie Yao, Gangjian Zhang, Wenhao Shen, Jian Shu, Yu Feng, Hao Wang  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2603.04993v1.pdf)  
  <details><summary>Abstract</summary>

  Monocular 3D clothed human reconstruction aims to generate a complete and realistic textured 3D avatar from a single image. Existing methods are commonly trained under multi-view supervision with annotated geometric priors, and during inference, these priors are estimated by the pre-trained network from the monocular input. These methods are constrained by three key limitations: texturally by unavailability of training data, geometrically by inaccurate external priors, and systematically by biased single-modality supervision, all leading to suboptimal reconstruction. To address these issues, we propose a novel reconstruction framework, named MultiGO++, which achieves effective systematic geometry-texture collaboration. It consists of three core parts: (1) A multi-source texture synthesis strategy that constructs 15,000+ 3D textured human scans to improve the performance on texture quality estimation in challenge scenarios; (2) A region-aware shape extraction module that extracts and interacts features of each body region to obtain geometry information and a Fourier geometry encoder that mitigates...
  </details>  
  Keywords: texture synthesis  
- **[Revisiting an Old Perspective Projection for Monocular 3D Morphable Models Regression](https://arxiv.org/abs/2603.04958v1)**  
  Authors: Toby Chong, Ryota Nakajima  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2603.04958v1.pdf)  
  <details><summary>Abstract</summary>

  We introduce a novel camera model for monocular 3D Morphable Model (3DMM) regression methods that effectively captures the perspective distortion effect commonly seen in close-up facial images.   Fitting 3D morphable models to video is a key technique in content creation. In particular, regression-based approaches have produced fast and accurate results by matching the rendered output of the morphable model to the target image. These methods typically achieve stable performance with orthographic projection, which eliminates the ambiguity between focal length and object distance. However, this simplification makes them unsuitable for close-up footage, such as that captured with head-mounted cameras.   We extend orthographic projection with a new shrinkage parameter, incorporating a pseudo-perspective effect while preserving the stability of the original projection. We present several techniques that allow finetuning of existing models, and demonstrate the effectiveness of our modification through both quantitative and qualitative comparisons using a custom dataset recorded with head-mounted cameras.
  </details>  
- **[DM-CFO: A Diffusion Model for Compositional 3D Tooth Generation with Collision-Free Optimization](https://arxiv.org/abs/2603.03602v1)**  
  Authors: Yan Tian, Pengcheng Xue, Weiping Ding, Mahmoud Hassaballah, Karen Egiazarian, Aura Conci, Abdulkadir Sengur, Leszek Rutkowski  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2603.03602v1.pdf) | [![Project](https://img.shields.io/badge/-Project-blue)](https://amateurc.github.io/CF-3DTeeth)  
  <details><summary>Abstract</summary>

  The automatic design of a 3D tooth model plays a crucial role in dental digitization. However, current approaches face challenges in compositional 3D tooth generation because both the layouts and shapes of missing teeth need to be optimized.In addition, collision conflicts are often omitted in 3D Gaussian-based compositional 3D generation, where objects may intersect with each other due to the absence of explicit geometric information on the object surfaces. Motivated by graph generation through diffusion models and collision detection using 3D Gaussians, we propose an approach named DM-CFO for compositional tooth generation, where the layout of missing teeth is progressively restored during the denoising phase under both text and graph constraints. Then, the Gaussian parameters of each layout-guided tooth and the entire jaw are alternately updated using score distillation sampling (SDS). Furthermore, a regularization term based on the distances between the 3D Gaussians of neighboring teeth and the anchor tooth...
  </details>  
- **[Direct Reward Fine-Tuning on Poses for Single Image to 3D Human in the Wild](https://arxiv.org/abs/2603.02619v1)**  
  Authors: Seunguk Do, Minwoo Huh, Joonghyuk Shin, Jaesik Park  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2603.02619v1.pdf) | [![Project](https://img.shields.io/badge/-Project-blue)](https://seunguk-do.github.io/drpose)  
  <details><summary>Abstract</summary>

  Single-view 3D human reconstruction has achieved remarkable progress through the adoption of multi-view diffusion models, yet the recovered 3D humans often exhibit unnatural poses. This phenomenon becomes pronounced when reconstructing 3D humans with dynamic or challenging poses, which we attribute to the limited scale of available 3D human datasets with diverse poses. To address this limitation, we introduce DrPose, Direct Reward fine-tuning algorithm on Poses, which enables post-training of a multi-view diffusion model on diverse poses without requiring expensive 3D human assets. DrPose trains a model using only human poses paired with single-view images, employing a direct reward fine-tuning to maximize PoseScore, which is our proposed differentiable reward that quantifies consistency between a generated multi-view latent image and a ground-truth human pose. This optimization is conducted on DrPose15K, a novel dataset that was constructed from an existing human motion dataset and a pose-conditioned video generative model. Constructed from abundant human...
  </details>  
- **[3D Field of Junctions: A Noise-Robust, Training-Free Structural Prior for Volumetric Inverse Problems](https://arxiv.org/abs/2603.02149v1)**  
  Authors: Namhoon Kim, Narges Moeini, Justin Romberg, Sara Fridovich-Keil  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2603.02149v1.pdf)  
  <details><summary>Abstract</summary>

  Volume denoising is a foundational problem in computational imaging, as many 3D imaging inverse problems face high levels of measurement noise. Inspired by the strong 2D image denoising properties of Field of Junctions (ICCV 2021), we propose a novel, fully volumetric 3D Field of Junctions (3D FoJ) representation that optimizes a junction of 3D wedges that best explain each 3D patch of a full volume, while encouraging consistency between overlapping patches. In addition to direct volume denoising, we leverage our 3D FoJ representation as a structural prior that: (i) requires no training data, and thus precludes the risk of hallucination, (ii) preserves and enhances sharp edge and corner structures in 3D, even under low signal to noise ratio (SNR), and (iii) can be used as a drop-in denoising representation via projected or proximal gradient descent for any volumetric inverse problem with low SNR. We demonstrate successful volume reconstruction and denoising...
  </details>  
- **[OnlineX: Unified Online 3D Reconstruction and Understanding with Active-to-Stable State Evolution](https://arxiv.org/abs/2603.02134v2)**  
  Authors: Chong Xia, Fangfu Liu, Yule Wang, Yize Pang, Yueqi Duan  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2603.02134v2.pdf)  
  <details><summary>Abstract</summary>

  Recent advances in generalizable 3D Gaussian Splatting (3DGS) have enabled rapid 3D scene reconstruction within seconds, eliminating the need for per-scene optimization. However, existing methods primarily follow an offline reconstruction paradigm, lacking the capacity for continuous reconstruction, which limits their applicability to online scenarios such as robotics and VR/AR. In this paper, we introduce OnlineX, a feed-forward framework that reconstructs both 3D visual appearance and language fields in an online manner using only streaming images. A key challenge in online formulation is the cumulative drift issue, which is rooted in the fundamental conflict between two opposing roles of the memory state: an active role that constantly refreshes to capture high-frequency local geometry, and a stable role that conservatively accumulates and preserves the long-term global structure. To address this, we introduce a decoupled active-to-stable state evolution paradigm. Our framework decouples the memory state into a dedicated active state and a persistent...
  </details>  
  Keywords: 3d scene reconstruction, novel view synthesis  
- **[Tiny-DroNeRF: Tiny Neural Radiance Fields aboard Federated Learning-enabled Nano-drones](https://arxiv.org/abs/2603.01850v1)**  
  Authors: Ilenia Carboni, Elia Cereda, Lorenzo Lamberti, Daniele Malpetti, Francesco Conti, Daniele Palossi  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2603.01850v1.pdf)  
  <details><summary>Abstract</summary>

  Sub-30g nano-sized aerial robots can leverage their agility and form factor to autonomously explore cluttered and narrow environments, like in industrial inspection and search and rescue missions. However, the price for their tiny size is a strong limit in their resources, i.e., sub-100 mW microcontroller units (MCUs) delivering $\sim$100 GOps/s at best, and memory budgets well below 100 MB. Despite these strict constraints, we aim to enable complex vision-based tasks aboard nano-drones, such as dense 3D scene reconstruction: a key robotic task underlying fundamental capabilities like spatial awareness and motion planning. Top-performing 3D reconstruction methods leverage neural radiance fields (NeRF) models, which require GBs of memory and massive computation, usually delivered by high-end GPUs consuming 100s of Watts. Our work introduces Tiny-DroNeRF, a lightweight NeRF model, based on Instant-NGP, and optimized for running on a GAP9 ultra-low-power (ULP) MCU aboard our nano-drones. Then, we further empower our Tiny-DroNeRF by leveraging...
  </details>  
  Keywords: 3d scene reconstruction  
- **[Dehallu3D: Hallucination-Mitigated 3D Generation from Single Image via Cyclic View Consistency Refinement](https://arxiv.org/abs/2603.01601v2)**  
  Authors: Xiwen Wang, Shichao Zhang, Hailun Zhang, Ruowei Wang, Mao Li, Chenyu Zhou, Qijun Zhao, Ji-Zhe Zhou  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2603.01601v2.pdf)  
  <details><summary>Abstract</summary>

  Large 3D reconstruction models have revolutionized the 3D content generation field, enabling broad applications in virtual reality and gaming. Just like other large models, large 3D reconstruction models suffer from hallucinations as well, introducing structural outliers (e.g., odd holes or protrusions) that deviate from the input data. However, unlike other large models, hallucinations in large 3D reconstruction models remain severely underexplored, leading to malformed 3D-printed objects or insufficient immersion in virtual scenes. Such hallucinations majorly originate from that existing methods reconstruct 3D content from sparsely generated multi-view images which suffer from large viewpoint gaps and discontinuities. To mitigate hallucinations by eliminating the outliers, we propose Dehallu3D for 3D mesh generation. Our key idea is to design a balanced multi-view continuity constraint to enforce smooth transitions across dense intermediate viewpoints, while avoiding over-smoothing that could erase sharp geometric features. Therefore, Dehallu3D employs a plug-and-play optimization module with two key constraints: (i)...
  </details>  
- **[Preference Score Distillation: Leveraging 2D Rewards to Align Text-to-3D Generation with Human Preference](https://arxiv.org/abs/2603.01594v1)**  
  Authors: Jiaqi Leng, Shuyuan Tu, Haidong Cao, Sicheng Xie, Daoguo Dong, Zuxuan Wu, Yu-Gang Jiang  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2603.01594v1.pdf)  
  <details><summary>Abstract</summary>

  Human preference alignment presents a critical yet underexplored challenge for diffusion models in text-to-3D generation. Existing solutions typically require task-specific fine-tuning, posing significant hurdles in data-scarce 3D domains. To address this, we propose Preference Score Distillation (PSD), an optimization-based framework that leverages pretrained 2D reward models for human-aligned text-to-3D synthesis without 3D training data. Our key insight stems from the incompatibility of pixel-level gradients: due to the absence of noisy samples during reward model training, direct application of 2D reward gradients disturbs the denoising process. Noticing that similar issue occurs in the naive classifier guidance in conditioned diffusion models, we fundamentally rethink preference alignment as a classifier-free guidance (CFG)-style mechanism through our implicit reward model. Furthermore, recognizing that frozen pretrained diffusion models constrain performance, we introduce an adaptive strategy to co-optimize preference scores and negative text embeddings. By incorporating CFG during optimization, online refinement of negative text embeddings dynamically enhances...
  </details>  
  Keywords: text-to-3d  
- **[TIMI: Training-Free Image-to-3D Multi-Instance Generation with Spatial Fidelity](https://arxiv.org/abs/2603.01371v1)**  
  Authors: Xiao Cai, Lianli Gao, Pengpeng Zeng, Ji Zhang, Heng Tao Shen, Jingkuan Song  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2603.01371v1.pdf)  
  <details><summary>Abstract</summary>

  Precise spatial fidelity in Image-to-3D multi-instance generation is critical for downstream real-world applications. Recent work attempts to address this by fine-tuning pre-trained Image-to-3D (I23D) models on multi-instance datasets, which incurs substantial training overhead and struggles to guarantee spatial fidelity. In fact, we observe that pre-trained I23D models already possess meaningful spatial priors, which remain underutilized as evidenced by instance entanglement issues. Motivated by this, we propose TIMI, a novel Training-free framework for Image-to-3D Multi-Instance generation that achieves high spatial fidelity. Specifically, we first introduce an Instance-aware Separation Guidance (ISG) module, which facilitates instance disentanglement during the early denoising stage. Next, to stabilize the guidance introduced by ISG, we devise a Spatial-stabilized Geometry-adaptive Update (SGU) module that promotes the preservation of the geometric characteristics of instances while maintaining their relative relationships. Extensive experiments demonstrate that our method yields better performance in terms of both global layout and distinct local instances compared...
  </details>  
  Keywords: image-to-3d  
- **[ArtLLM: Generating Articulated Assets via 3D LLM](https://arxiv.org/abs/2603.01142v2)**  
  Authors: Penghao Wang, Siyuan Xie, Hongyu Yan, Xianghui Yang, Jingwei Huang, Chunchao Guo, Jiayuan Gu  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2603.01142v2.pdf)  
  <details><summary>Abstract</summary>

  Creating interactive digital environments for gaming, robotics, and simulation relies on articulated 3D objects whose functionality emerges from their part geometry and kinematic structure. However, existing approaches remain fundamentally limited: optimization-based reconstruction methods require slow, per-object joint fitting and typically handle only simple, single-joint objects, while retrieval-based methods assemble parts from a fixed library, leading to repetitive geometry and poor generalization. To address these challenges, we introduce ArtLLM, a novel framework for generating high-quality articulated assets directly from complete 3D meshes. At its core is a 3D multimodal large language model trained on a large-scale articulation dataset curated from both existing articulation datasets and procedurally generated objects. Unlike prior work, ArtLLM autoregressively predicts a variable number of parts and joints, inferring their kinematic structure in a unified manner from the object's point cloud. This articulation-aware layout then conditions a 3D generative model to synthesize high-fidelity part geometries. Experiments on the...
  </details>  
- **[StegoNGP: 3D Cryptographic Steganography using Instant-NGP](https://arxiv.org/abs/2603.00949v1)**  
  Authors: Wenxiang Jiang, Yujun Lan, Shuo Zhao, Yuanshan Liu, Mingzhu Zhou, Jinxin Wang  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2603.00949v1.pdf) | [![GitHub](https://img.shields.io/github/stars/jiang-wenxiang/StegoNGP?style=social)](https://github.com/jiang-wenxiang/StegoNGP)  
  <details><summary>Abstract</summary>

  Recently, Instant Neural Graphics Primitives (Instant-NGP) has achieved significant success in rapid 3D scene reconstruction, but securely embedding high-capacity hidden data, such as an entire 3D scene, remains a challenge. Existing methods rely on external decoders, require architectural modifications, and suffer from limited capacity, which makes them easily detectable. We propose a novel parameter-free 3D Cryptographic Steganography using Instant-NGP (StegoNGP), which leverages the Instant-NGP hash encoding function as a key-controlled scene switcher. By associating a default key with a cover scene and a secret key with a hidden scene, our method trains a single model to interweave both representations within the same network weights. The resulting model is indistinguishable from a standard Instant-NGP in architecture and parameter count. We also introduce an enhanced Multi-Key scheme, which assigns multiple independent keys across hash levels, dramatically expanding the key space and providing high robustness against partial key disclosure attacks. Experimental results demonstrated...
  </details>  
  Keywords: 3d scene reconstruction  

### June 2026
- **[OASIS: From Simulation Data Collection to Real-World Humanoid Loco-Manipulation](https://arxiv.org/abs/2606.08548v1)**  
  Authors: Zehao Yu, Jiakun Zheng, Weiji Xie, Jiyuan Shi, Chenyun Zhang, Chenjia Bai, Xuelong Li  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2606.08548v1.pdf) | [![Project](https://img.shields.io/badge/-Project-blue)](https://oasis-humanoid.github.io)  
  <details><summary>Abstract</summary>

  Recent progress in robot manipulation has been largely driven by learning from large-scale demonstrations. For humanoid robot loco-manipulation tasks, however, existing data sources force an unsatisfying tradeoff between trajectory quality and scalability. Real-world teleoperation provides the highest-quality trajectories but requires dedicated physical space and time-consuming scene resets. Simulation offers an alternative way out of this dilemma: it can produce clean, embodiment-aligned data at scale without any physical hardware. In this paper, we propose OASIS, a simulation-data-driven framework for humanoid loco-manipulation. OASIS automatically reconstructs realistic object assets from real-world images using a 3D generative model. Based on these assets, trajectories are first collected through teleoperation in simulation, and then augmented under diverse domain randomizations in a post-processing stage. With the resulting simulation data, we further design a hierarchical visuomotor policy for humanoid loco-manipulation. Extensive experiments on the real humanoid robot show that, under zero-shot deployment, the policy trained on our simulation...
  </details>  
- **[3DMorph: Single-Image-Guided Local 3D Shape Editing and Morphing](https://arxiv.org/abs/2606.07115v1)**  
  Authors: Tobias Preintner, Yunfei Deng, Phillip Müller, Sebastian Illing, Adrian König, Thomas Bäck, Elena Raponi, Niki van Stein  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2606.07115v1.pdf)  
  <details><summary>Abstract</summary>

  Despite recent progress in 3D generation, intuitive editing of existing shapes remains limited. Unlike images, which benefit from well-established inpainting tools, general 3D objects such as meshes still lack simple and effective methods for local shape editing. Existing approaches are often global, domain-specific, require complex user interaction, or focus on appearance (color and texture) rather than geometry. We introduce 3DMorph, a training-free framework for single-image-guided local 3D shape editing and morphing. Given an edited image showing a desired shape modification, our method automatically localizes the relevant 3D region and transfers 2D modifications to 3D while preserving unmodified areas. 3DMorph also enables intermediate shape generation between the original and edited objects, facilitating design exploration. To benchmark editing quality, we introduce Delta3D, an image-guided local 3D editing benchmark with paired ground-truth edits. Experimental results show that 3DMorph translates intuitive 2D edits into 3D, outperforming state-of-the-art generative and editing methods.
  </details>  
- **[HomeWorld: A Unified Floorplan-to-Furnished Framework for Generating Controllable, Densely Interactive Whole-Home Scenes](https://arxiv.org/abs/2606.06390v1)**  
  Authors: Wenbo Li, Xiaoliang Ju, Zipeng Qin, Rongyao Fang, Hongsheng Li  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2606.06390v1.pdf) | [![Project](https://img.shields.io/badge/-Project-blue)](https://kairos-homeworld.github.io)  
  <details><summary>Abstract</summary>

  Indoor scene generation is crucial for robot simulation and modern interior design. However, complex layouts together with scarce 3D scene data make learning-based generation challenging. Existing methods often rely on hand-crafted rules or focus on isolated sub-tasks (e.g., floorplan synthesis or single-room furnishing), producing whole-home scenes that lack global coherence, realism, and simulation readiness. To mitigate these limitations, we propose a unified hierarchical framework that decomposes indoor scene synthesis into controllable stages. First, we curate a large-scale dataset of 300K real residential floorplans to train a large language model for whole-home floorplan generation. With detailed descriptions and a K-D tree-based representation, our method enables fine-grained, controllable whole-home floorplan generation. Building upon the generated whole-home floorplan, we leverage image generation models to draft furniture layouts from multi-level roaming viewpoints, and then generate the layouts of small manipulable objects on different supporting surfaces (e.g., cabinets, desks, and dining tables) for embodied AI...
  </details>  
- **[Global-Local Monte Carlo Tree Search in Vision-Language Models for Text-to-3D Indoor Scene Generation](https://arxiv.org/abs/2606.06002v2)**  
  Authors: Mengshi Qi, Wei Deng, Xianlin Zhang, Huadong Ma  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2606.06002v2.pdf)  
  <details><summary>Abstract</summary>

  Large Vision-Language Models have achieved significant reasoning performance in various tasks. However, there are few studies on text-to-3D indoor scene generation with LVLMs. The main challenge is that prevailing LVLM-based methods employ chain-of-thought sequential decision mechanisms that cannot revise earlier decisions, causing error propagation. In this paper, we consider the task as a planning problem constrained by spatial and layout commonsense. To solve this problem, we model it as a tree search problem with global and local trees, which differs from existing sequential decision-making approaches. In the global tree, we place each object iteratively and explore multiple attempts like humans furnishing a room, where the problem space is represented as a tree. To effectively search the tree, we propose a hierarchical scene representation and a PRM-guided MCTS method. This representation abstracts a scene into room level, region level, floor object level, and supported object level. The PRM-guided MCTS method uses...
  </details>  
  Keywords: text-to-3d  
- **[KV-Control: Parameter-Efficient K/V Injection for Trajectory-Controlled Text-to-Motion](https://arxiv.org/abs/2606.05624v1)**  
  Authors: Tengjiao Sun, Pengcheng Fang, Xiaoyu Zhan, Yanwen Guo, Dongjie Fu, Xiaohao Cai, Hansung Kim  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2606.05624v1.pdf)  
  <details><summary>Abstract</summary>

  Text-conditioned 3D human motion models now synthesize plausible motions from prompts, but practical animation and embodied-agent workflows rarely stop at text: a character may need to follow a sketched root path, hit an end-effector target, or satisfy a multi-joint trajectory while still preserving the gait, style, and intent described by language. This exposes a control trade-off. A trajectory controller should be precise without overwriting the pretrained text-conditioned motion prior, yet existing solutions either duplicate large portions of the generator to regain per-layer control access or move much of the cost to test-time optimization. We introduce KV-Control, a compact attention-side control interface for frozen masked text-to-motion transformers. The key idea is to make geometric constraints available as memory inside self-attention rather than injecting them through a global pose token or enforcing them only at the output side. To support this interface, we co-design a part-tokenized motion substrate and controller: \textbf{PartVQ} learns...
  </details>  
- **[Unpaired RGB-Thermal Gaussian-Splatting Using Visual Geometric Transformers](https://arxiv.org/abs/2606.05491v1)**  
  Authors: Jean Cordonnier, Chenghao Xu, Olga Fink, Malcolm Mielle  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2606.05491v1.pdf)  
  <details><summary>Abstract</summary>

  Multi-modal novel view synthesis (NVS) combining RGB and thermal imagery enables precise 3D scene reconstruction with visual and thermal information. However, existing methods typically rely on precisely calibrated RGB-thermal image pairs or stereo setups, limiting scalability and practical deployment. To address this, we introduce a framework for unpaired RGB-thermal NVS that leverages VGGT, a 3D feed-forward transformer architecture, to independently estimate camera poses for each modality. The pose sets are then aligned using the Procrustes algorithm with a cross-modal feature matcher, enabling joint registration without paired calibration. Building on this alignment, we further propose a multi-modal 3D Gaussian Splatting approach that learns directly from unpaired RGB and thermal images. Experiments on diverse scenes demonstrate that our method achieves competitive performance in thermal view synthesis while maintaining RGB fidelity. Moreover, we show that existing reconstruction approaches can produce modality-specific reconstructions that lack cross-modal consistency. We thus introduce a benchmarking framework to...
  </details>  
  Keywords: 3d scene reconstruction, novel view synthesis  
- **[MeshFlow: Efficient Artistic Mesh Generation via MeshVAE and Flow-based Diffusion Transformer](https://arxiv.org/abs/2606.04621v1)**  
  Authors: Weiyu Li, Antoine Toisoul, Tom Monnier, Roman Shapovalov, Rakesh Ranjan, Ping Tan, Andrea Vedaldi  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2606.04621v1.pdf) | [![GitHub](https://img.shields.io/github/stars/facebookresearch/meshflow?style=social)](https://github.com/facebookresearch/meshflow) | [![Project](https://img.shields.io/badge/-Project-blue)](https://mesh-flow.github.io)  
  <details><summary>Abstract</summary>

  We present MeshFlow, a new method for generating artist-like 3D meshes. Current mesh generators often adopt Auto-Regressive (AR) next-token prediction, a natural choice given the discrete nature of mesh topology. However, AR methods scale poorly because the inference cost is quadratic in mesh size. They also require discretizing the vertex coordinates, which introduces quantization errors. To address these challenges, we introduce a Variational Autoencoder (VAE) that, supervised with a contrastive loss, represents both continuous vertex positions and discrete connectivity in a continuous latent space. This latent space is significantly more compact than prior token-based mesh representations. We then build a 3D generator based on a Rectified Flow transformer, generating all mesh vertices and edges in parallel. Our model generates meshes 18x faster than the fastest AR generator while also achieving excellent accuracy across standard mesh-generation metrics. Homepage: https://mesh-flow.github.io/, Code: https://github.com/facebookresearch/meshflow
  </details>  
  Keywords: 3d generator  
- **[3DThinkVLA: Endowing Vision-Language-Action Models with Latent 3D Priors via 3D-Thinking-Guided Co-training](https://arxiv.org/abs/2606.04436v1)**  
  Authors: Jiaxin Shi, Xidong Zhang, Fucai Zhu, Zhe Li, Siyu Zhu, Weihao Yuan  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2606.04436v1.pdf)  
  <details><summary>Abstract</summary>

  We propose a 3D-thinking-guided co-training framework that enables vision-language-action (VLA) models to perform 3D spatial reasoning implicitly during action prediction. Our core insight is that 3D geometry perception and 3D spatial reasoning are distinct capabilities that can be disentangled and injected at different feature hierarchies. During training, three tightly coupled components work in concert primarily within the latent space: (1) To gain geometric priors, a latent 3D geometry perception module aligns intermediate visual features with a 3D foundation model, acquiring low-level geometric cues without architectural modifications to the VLM backbone. (2) Complementing this, an online 3D reasoning distillation module mitigates the prompt-induced reasoning gap via a shared reasoning anchor token. During 3D VLM co-training, this anchor is emitted as the first output token to robustly encode spatial priors. During VLA training, it serves as an input token inserted between the task and action instructions, transferring high-level spatial thinking from explicit...
  </details>  
- **[PerceptTwin: Semantic Scene Reconstruction for Iterative LLM Planning and Verification](https://arxiv.org/abs/2606.04226v1)**  
  Authors: Charlie Gauthier, Sacha Morin, Liam Paull  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2606.04226v1.pdf)  
  <details><summary>Abstract</summary>

  Simulation environments are useful for both robot policy learning and planning verification and validation. Traditionally, the process of creating a simulation was onerous. Creating a bespoke simulation environment for each individual environment that a robot would operate in was simply infeasible. In this work, we introduce PerceptTwin, a fully automatic pipeline that constructs interactive simulations directly from semantic scene representations produced by a robot's perception stack. PerceptTwin combines open-vocabulary object maps with 3D asset generation, affordance prediction, and commonsense condition checking. These interactive simulations can be used to validate and refine plans before they are executed on the robot hardware. Borrowing from the AI alignment literature, we also introduce an LLM judge that verifies plan correctness and alignment with human preferences. Experiments show that PerceptTwin feedback allows LLM planners to refine plans, enhance safety, and resist harmful black-box prompting attacks. In our suite of tasks, PerceptTwin improves plan success by...
  </details>  
  Keywords: 3d asset generation  
- **[SymTRELLIS: Symmetry-Enforced Voxel Latents for 3D Generation](https://arxiv.org/abs/2606.04108v1)**  
  Authors: Guangda Ji, Qimin Chen, Qinchan Li, Mingrui Zhao, Kai Wang, Hao Zhang  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2606.04108v1.pdf)  
  <details><summary>Abstract</summary>

  Single-view 3D generative models have achieved impressive visual quality, yet they are not designed to satisfy structural or functional requirements, and in practice, often fall short. Symmetry is one such requirement: violations, even subtle ones, on symmetry can render a model physically unusable. We present SymTRELLIS, a method that enforces arbitrary finite point group symmetries (rotational, reflectional, and polyhedral) during the flow-based 3D generation of TRELLIS.2, without retraining the underlying VAE or flow model. Our key idea is to approximate the latent-space action of spatial transformations as a learned linear operator on voxel latents, implemented as a lightweight spatial-transform latent mapper trained on generic, non-symmetric 3D data. At generation time, we enforce symmetry by averaging predicted flow velocities across all symmetry-equivalent transformations at each ODE step, a process we call velocity symmetrization. The symmetry specification can be estimated automatically from an initial TRELLIS.2 generation or supplied by the user, enabling...
  </details>  
- **[SimuScene: Simulation-Ready Compositional 3D Scene Reconstruction from a Single Image](https://arxiv.org/abs/2606.03994v1)**  
  Authors: Inhee Lee, Sangwon Baik, Sungjoo Kim, Hyeonwoo Kim, Hyunsoo Cha, Hanbyul Joo  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2606.03994v1.pdf)  
  <details><summary>Abstract</summary>

  Reconstructing interactive, simulation-ready 3D scenes from a single image is a critical bottleneck for robotic manipulation. While recent single-image lifters recover plausible per-object shapes, composing them yields scenes that collapse under physical simulation due to interpenetrating, hovering, or sinking objects. Existing physics-aware methods address this strictly as a post-hoc layout correction, leaving the underlying geometric errors unresolved. To address this, we introduce SimuScene, a compositional 3D reconstruction pipeline that puts physics in the loop of shape and layout estimation. Rather than using physics merely for layout cleanup, we utilize the physics engine as a diagnostic measurement tool during the generative process itself. By diagnostically simulating reconstructed objects under gravity, we convert penetration and support failures into quantitative correction signals that drive gravity-axis stretching and amodal shape resampling. This physics-informed feedback loop mitigates accumulated reconstruction errors and produces a stable, simulation-ready compositional 3D scene. Extensive experiments demonstrate state-of-the-art performance on physical...
  </details>  
  Keywords: 3d scene reconstruction  
- **[PerBite: A Curated Diagnostic Workflow for Bite-Aware Food Volume Estimation](https://arxiv.org/abs/2606.02021v1)**  
  Authors: Ahmad AlMughrabi, Farid Al-Areqi, David Fernández Gómez, Umair Haroon, Marc Bolaños, Ricardo Marques, Petia Radeva  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2606.02021v1.pdf) | [![GitHub](https://img.shields.io/github/stars/GCVCG/PerBite-CVPR-MetaFood-2026?style=social)](https://github.com/GCVCG/PerBite-CVPR-MetaFood-2026)  
  <details><summary>Abstract</summary>

  Can a visually plausible food mesh be trusted to estimate the volume of consumed food? \method investigates this question using selected paired before- and after-consumption states from the MetaFood CVPR 2026 Continuous 3D Reconstruction While Eating Challenge. The submitted workflow follows a curated reconstruction protocol: SAM~3 segments the food and plate regions; Hunyuan3D/SAM~3D generates a dimensionless food mesh; the plate diameter provides the metric scale; the plate geometry is removed in Blender; and the remaining mesh is hole-filled, made watertight, and integrated to estimate volume. MoGe-2 is used only as an auxiliary cue for initial dish-diameter estimation when direct plate measurement is uncertain; it is not the primary scale source for the reported challenge result. \method ranks first, with an average Chamfer distance of 8.31 across 34 meshes using rigid ICP without scale correction. On 17 before- and after-pairs, it achieves 33.87\% state-level volume MAPE and zero monotonicity violations, while...
  </details>  

### January 2026
- **[Gaussian-Constrained LeJEPA Representations for Unsupervised Scene Discovery and Pose Consistency](https://arxiv.org/abs/2602.07016v1)**  
  Authors: Mohsen Mostafa  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2602.07016v1.pdf)  
  <details><summary>Abstract</summary>

  Unsupervised 3D scene reconstruction from unstructured image collections remains a fundamental challenge in computer vision, particularly when images originate from multiple unrelated scenes and contain significant visual ambiguity. The Image Matching Challenge 2025 (IMC2025) highlights these difficulties by requiring both scene discovery and camera pose estimation under real-world conditions, including outliers and mixed content. This paper investigates the application of Gaussian-constrained representations inspired by LeJEPA (Joint Embedding Predictive Architecture) to address these challenges. We present three progressively refined pipelines, culminating in a LeJEPA-inspired approach that enforces isotropic Gaussian constraints on learned image embeddings. Rather than introducing new theoretical guarantees, our work empirically evaluates how these constraints influence clustering consistency and pose estimation robustness in practice. Experimental results on IMC2025 demonstrate that Gaussian-constrained embeddings can improve scene separation and pose plausibility compared to heuristic-driven baselines, particularly in visually ambiguous settings. These findings suggest that theoretically motivated representation constraints offer a promising...
  </details>  
  Keywords: 3d scene reconstruction  
- **[Visual Personalization Turing Test](https://arxiv.org/abs/2601.22680v1)**  
  Authors: Rameen Abdal, James Burgess, Sergey Tulyakov, Kuan-Chieh Jackson Wang  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2601.22680v1.pdf)  
  <details><summary>Abstract</summary>

  We introduce the Visual Personalization Turing Test (VPTT), a new paradigm for evaluating contextual visual personalization based on perceptual indistinguishability, rather than identity replication. A model passes the VPTT if its output (image, video, 3D asset, etc.) is indistinguishable to a human or calibrated VLM judge from content a given person might plausibly create or share. To operationalize VPTT, we present the VPTT Framework, integrating a 10k-persona benchmark (VPTT-Bench), a visual retrieval-augmented generator (VPRAG), and the VPTT Score, a text-only metric calibrated against human and VLM judgments. We show high correlation across human, VLM, and VPTT evaluations, validating the VPTT Score as a reliable perceptual proxy. Experiments demonstrate that VPRAG achieves the best alignment-originality balance, offering a scalable and privacy-safe foundation for personalized generative AI.
  </details>  
- **[EUGens: Efficient, Unified, and General Dense Layers](https://arxiv.org/abs/2601.22563v2)**  
  Authors: Sang Min Kim, Byeongchan Kim, Arijit Sehanobish, Somnath Basu Roy Chowdhury, Rahul Kidambi, Dongseok Shim, Avinava Dubey, Snigdha Chaturvedi, Min-hwan Oh, Krzysztof Choromanski  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2601.22563v2.pdf)  
  <details><summary>Abstract</summary>

  Efficient neural networks are essential for scaling machine learning models to real-time applications and resource-constrained environments. Fully-connected feedforward layers (FFLs) introduce computation and parameter count bottlenecks within neural network architectures. To address this challenge, in this work, we propose a new class of dense layers that generalize standard fully-connected feedforward layers, \textbf{E}fficient, \textbf{U}nified and \textbf{Gen}eral dense layers (EUGens). EUGens leverage random features to approximate standard FFLs and go beyond them by incorporating a direct dependence on the input norms in their computations. The proposed layers unify existing efficient FFL extensions and improve efficiency by reducing inference complexity from quadratic to linear time. They also lead to \textbf{the first} unbiased algorithms approximating FFLs with arbitrary polynomial activation functions. Furthermore, EuGens reduce the parameter count and computational overhead while preserving the expressive power and adaptability of FFLs. We also present a layer-wise knowledge transfer technique that bypasses backpropagation, enabling efficient adaptation of...
  </details>  
  Keywords: 3d scene reconstruction  
- **[CG-MLLM: Captioning and Generating 3D content via Multi-modal Large Language Models](https://arxiv.org/abs/2601.21798v2)**  
  Authors: Junming Huang, Chi Wang, Letian Li, Guangkai Xu, Donglin Huang, Hao Chen, Qiang Dai, Weiwei Xu  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2601.21798v2.pdf)  
  <details><summary>Abstract</summary>

  Large Language Models(LLMs) have revolutionized text generation and multimodal perception,but their capabilities in 3D content generation remain underexplored. Existing methods compromise by producing either low-resolution meshes or coarse structural proxies, failing to capture finegrained geometry natively. In this paper, we propose CG-MLLM, a novel Multi-modal Large Language Model (MLLM) capable of 3D captioning and high-resolution 3D generation in a single framework. Leveraging the Mixture-ofTransformer architecture, CG-MLLM decouples disparate modeling needs, where the Token-level Autoregressive (TokenAR) Transformer handles token-level content, and the Block-level Autoregressive (BlockAR) Transformer handles blocklevel content. By integrating a pre-trained visionlanguage backbone with a specialized 3D VAE latent space, CG-MLLM facilitates long-context interactions between standard tokens and spatial blocks within a single integrated architecture. Experimental results show that CG-MLLM significantly outperforms existing MLLMs in generating high-fidelity 3D objects, effectively bringing high-resolution 3D content creation into the mainstream LLM paradigm. Beyond generation, we further observe that learning to produce...
  </details>  
- **[GeoDiff3D: Self-Supervised 3D Scene Generation with Geometry-Constrained 2D Diffusion Guidance](https://arxiv.org/abs/2601.19785v2)**  
  Authors: Haozhi Zhu, Miaomiao Zhao, Dingyao Liu, Runze Tian, Yan Zhang, Jie Guo, Fenggen Yu  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2601.19785v2.pdf)  
  <details><summary>Abstract</summary>

  3D scene generation is a core technology for gaming, film/VFX, and VR/AR. Growing demand for rapid iteration, high-fidelity detail, and accessible content creation has further increased interest in this area. Existing methods broadly follow two paradigms - indirect 2D-to-3D reconstruction and direct 3D generation - but both are limited by weak structural modeling and heavy reliance on large-scale ground-truth supervision, often producing structural artifacts, geometric inconsistencies, and degraded high-frequency details in complex scenes. We propose GeoDiff3D, an efficient self-supervised framework that uses coarse geometry as a structural anchor and a geometry-constrained 2D diffusion model to provide texture-rich reference images. Importantly, GeoDiff3D does not require strict multi-view consistency of the diffusion-generated references and remains robust to the resulting noisy, inconsistent guidance. We further introduce voxel-aligned 3D feature aggregation and dual self-supervision to maintain scene coherence and fine details while substantially reducing dependence on labeled data. GeoDiff3D also trains with low computational...
  </details>  
- **[RoamScene3D: Immersive Text-to-3D Scene Generation via Adaptive Object-aware Roaming](https://arxiv.org/abs/2601.19433v1)**  
  Authors: Jisheng Chu, Wenrui Li, Rui Zhao, Wangmeng Zuo, Shifeng Chen, Xiaopeng Fan  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2601.19433v1.pdf) | [![GitHub](https://img.shields.io/github/stars/JS-CHU/RoamScene3D?style=social)](https://github.com/JS-CHU/RoamScene3D)  
  <details><summary>Abstract</summary>

  Generating immersive 3D scenes from texts is a core task in computer vision, crucial for applications in virtual reality and game development. Despite the promise of leveraging 2D diffusion priors, existing methods suffer from spatial blindness and rely on predefined trajectories that fail to exploit the inner relationships among salient objects. Consequently, these approaches are unable to comprehend the semantic layout, preventing them from exploring the scene adaptively to infer occluded content. Moreover, current inpainting models operate in 2D image space, struggling to plausibly fill holes caused by camera motion. To address these limitations, we propose RoamScene3D, a novel framework that bridges the gap between semantic guidance and spatial generation. Our method reasons about the semantic relations among objects and produces consistent and photorealistic scenes. Specifically, we employ a vision-language model (VLM) to construct a scene graph that encodes object relations, guiding the camera to perceive salient object boundaries and...
  </details>  
  Keywords: text-to-3d  
- **[TIGaussian: Disentangle Gaussians for Spatial-Awared Text-Image-3D Alignment](https://arxiv.org/abs/2601.19247v1)**  
  Authors: Jiarun Liu, Qifeng Chen, Yiru Zhao, Minghua Liu, Baorui Ma, Sheng Yang  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2601.19247v1.pdf)  
  <details><summary>Abstract</summary>

  While visual-language models have profoundly linked features between texts and images, the incorporation of 3D modality data, such as point clouds and 3D Gaussians, further enables pretraining for 3D-related tasks, e.g., cross-modal retrieval, zero-shot classification, and scene recognition. As challenges remain in extracting 3D modal features and bridging the gap between different modalities, we propose TIGaussian, a framework that harnesses 3D Gaussian Splatting (3DGS) characteristics to strengthen cross-modality alignment through multi-branch 3DGS tokenizer and modality-specific 3D feature alignment strategies. Specifically, our multi-branch 3DGS tokenizer decouples the intrinsic properties of 3DGS structures into compact latent representations, enabling more generalizable feature extraction. To further bridge the modality gap, we develop a bidirectional cross-modal alignment strategies: a multi-view feature fusion mechanism that leverages diffusion priors to resolve perspective ambiguity in image-3D alignment, while a text-3D projection module adaptively maps 3D features to text embedding space for better text-3D alignment. Extensive experiments on various...
  </details>  
- **[AI-Driven Three-Dimensional Reconstruction and Quantitative Analysis for Burn Injury Assessment](https://arxiv.org/abs/2602.00113v1)**  
  Authors: S. Kalaycioglu, C. Hong, K. Zhai, H. Xie, J. N. Wong  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2602.00113v1.pdf)  
  <details><summary>Abstract</summary>

  Accurate, reproducible burn assessment is critical for treatment planning, healing monitoring, and medico-legal documentation, yet conventional visual inspection and 2D photography are subjective and limited for longitudinal comparison. This paper presents an AI-enabled burn assessment and management platform that integrates multi-view photogrammetry, 3D surface reconstruction, and deep learning-based segmentation within a structured clinical workflow. Using standard multi-angle images from consumer-grade cameras, the system reconstructs patient-specific 3D burn surfaces and maps burn regions onto anatomy to compute objective metrics in real-world units, including surface area, TBSA, depth-related geometric proxies, and volumetric change. Successive reconstructions are spatially aligned to quantify healing progression over time, enabling objective tracking of wound contraction and depth reduction. The platform also supports structured patient intake, guided image capture, 3D analysis and visualization, treatment recommendations, and automated report generation. Simulation-based evaluation demonstrates stable reconstructions, consistent metric computation, and clinically plausible longitudinal trends, supporting a scalable, non-invasive approach to...
  </details>  
- **[Splat-Portrait: Generalizing Talking Heads with Gaussian Splatting](https://arxiv.org/abs/2601.18633v1)**  
  Authors: Tong Shi, Melonie de Almeida, Daniela Ivanova, Nicolas Pugeault, Paul Henderson  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2601.18633v1.pdf) | [![GitHub](https://img.shields.io/github/stars/stonewalking/Splat-portrait?style=social)](https://github.com/stonewalking/Splat-portrait)  
  <details><summary>Abstract</summary>

  Talking Head Generation aims at synthesizing natural-looking talking videos from speech and a single portrait image. Previous 3D talking head generation methods have relied on domain-specific heuristics such as warping-based facial motion representation priors to animate talking motions, yet still produce inaccurate 3D avatar reconstructions, thus undermining the realism of generated animations. We introduce Splat-Portrait, a Gaussian-splatting-based method that addresses the challenges of 3D head reconstruction and lip motion synthesis. Our approach automatically learns to disentangle a single portrait image into a static 3D reconstruction represented as static Gaussian Splatting, and a predicted whole-image 2D background. It then generates natural lip motion conditioned on input audio, without any motion driven priors. Training is driven purely by 2D reconstruction and score-distillation losses, without 3D supervision nor landmarks. Experimental results demonstrate that Splat-Portrait exhibits superior performance on talking head generation and novel view synthesis, achieving better visual quality compared to previous works....
  </details>  
  Keywords: novel view synthesis  
- **[Agreement-Driven Multi-View 3D Reconstruction for Live Cattle Weight Estimation](https://arxiv.org/abs/2601.17791v1)**  
  Authors: Rabin Dulal, Wenfeng Jia, Lihong Zheng, Jane Quinn  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2601.17791v1.pdf)  
  <details><summary>Abstract</summary>

  Accurate cattle live weight estimation is vital for livestock management, welfare, and productivity. Traditional methods, such as manual weighing using a walk-over weighing system or proximate measurements using body condition scoring, involve manual handling of stock and can impact productivity from both a stock and economic perspective. To address these issues, this study investigated a cost-effective, non-contact method for live weight calculation in cattle using 3D reconstruction. The proposed pipeline utilized multi-view RGB images with SAM 3D-based agreement-guided fusion, followed by ensemble regression. Our approach generates a single 3D point cloud per animal and compares classical ensemble models with deep learning models under low-data conditions. Results show that SAM 3D with multi-view agreement fusion outperforms other 3D generation methods, while classical ensemble models provide the most consistent performance for practical farm scenarios (R$^2$ = 0.69 $\pm$ 0.10, MAPE = 2.22 $\pm$ 0.56 \%), making this practical for on-farm implementation. These...
  </details>  
- **[From Prompts to Worlds: How Users Iterate, Explore, and Make Sense of AI-Generated 3D Environments](https://arxiv.org/abs/2603.13233v1)**  
  Authors: Aung Pyae  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2603.13233v1.pdf)  
  <details><summary>Abstract</summary>

  Text-to-3D generative AI systems create navigable environments from natural language prompts, but unlike text-to-image generation, evaluation requires embodied exploration of spatial coherence, scale, and navigability. We present the first empirical study of a commercial text-to-3D platform, combining think-aloud protocols, behavioral observation, and validated measures of usability, presence, and engagement. We report three findings. First, asymmetric expressibility: users readily convey semantic intent (themes, atmosphere) but struggle to specify spatial structure (layout, scale), reflecting a language-to-space limitation rather than a skill deficit. Second, episodic presence: immersion arises when expectations align with outputs but does not accumulate into sustained place illusion. Third, structural iteration breakdowns: refinement fails due to interaction barriers - poor discoverability, opaque feedback, and high temporal costs - rather than user limitations. Together, these dynamics form a reinforcing cycle in which spatial mismatches persist, producing episodic presence and ongoing sensemaking. We reframe text-to-3D interaction as negotiated meaning-making rather than linear...
  </details>  
  Keywords: text-to-3d  
- **[Neural Particle Automata: Learning Self-Organizing Particle Dynamics](https://arxiv.org/abs/2601.16096v1)**  
  Authors: Hyunsoo Kim, Ehsan Pajouheshgar, Sabine Süsstrunk, Wenzel Jakob, Jinah Park  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2601.16096v1.pdf)  
  <details><summary>Abstract</summary>

  We introduce Neural Particle Automata (NPA), a Lagrangian generalization of Neural Cellular Automata (NCA) from static lattices to dynamic particle systems. Unlike classical Eulerian NCA where cells are pinned to pixels or voxels, NPA model each cell as a particle with a continuous position and internal state, both updated by a shared, learnable neural rule. This particle-based formulation yields clear individuation of cells, allows heterogeneous dynamics, and concentrates computation only on regions where activity is present. At the same time, particle systems pose challenges: neighborhoods are dynamic, and a naive implementation of local interactions scale quadratically with the number of particles. We address these challenges by replacing grid-based neighborhood perception with differentiable Smoothed Particle Hydrodynamics (SPH) operators backed by memory-efficient, CUDA-accelerated kernels, enabling scalable end-to-end training. Across tasks including morphogenesis, point-cloud classification, and particle-based texture synthesis, we show that NPA retain key NCA behaviors such as robustness and self-regeneration, while...
  </details>  
  Keywords: texture synthesis  
- **[ThermoSplat: Cross-Modal 3D Gaussian Splatting with Feature Modulation and Geometry Decoupling](https://arxiv.org/abs/2601.15897v2)**  
  Authors: Zhaoqi Su, Shihai Chen, Xinyan Lin, Liqin Huang, Zhipeng Su, Xiaoqiang Lu  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2601.15897v2.pdf)  
  <details><summary>Abstract</summary>

  Multi-modal scene reconstruction integrating RGB and thermal infrared data is essential for robust environmental perception across diverse lighting and weather conditions. However, extending 3D Gaussian Splatting (3DGS) to multi-spectral scenarios remains challenging. Current approaches often struggle to fully leverage the complementary information of multi-modal data, typically relying on mechanisms that either tend to neglect cross-modal correlations or leverage shared representations that fail to adaptively handle the complex structural correlations and physical discrepancies between spectrums. To address these limitations, we propose ThermoSplat, a novel framework that enables deep spectral-aware reconstruction through active feature modulation and adaptive geometry decoupling. First, we introduce a Spectrum-Aware Adaptive Modulation that dynamically conditions shared latent features on thermal structural priors, effectively guiding visible texture synthesis with reliable cross-modal geometric cues. Second, to accommodate modality-specific geometric inconsistencies, we propose a Modality-Adaptive Geometric Decoupling scheme that learns independent opacity offsets and executes an independent rasterization pass for the...
  </details>  
  Keywords: texture synthesis  
- **[POTR: Post-Training 3DGS Compression](https://arxiv.org/abs/2601.14821v1)**  
  Authors: Bert Ramlot, Martijn Courteaux, Peter Lambert, Glenn Van Wallendael  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2601.14821v1.pdf)  
  <details><summary>Abstract</summary>

  3D Gaussian Splatting (3DGS) has recently emerged as a promising contender to Neural Radiance Fields (NeRF) in 3D scene reconstruction and real-time novel view synthesis. 3DGS outperforms NeRF in training and inference speed but has substantially higher storage requirements. To remedy this downside, we propose POTR, a post-training 3DGS codec built on two novel techniques. First, POTR introduces a novel pruning approach that uses a modified 3DGS rasterizer to efficiently calculate every splat's individual removal effect simultaneously. This technique results in 2-4x fewer splats than other post-training pruning techniques and as a result also significantly accelerates inference with experiments demonstrating 1.5-2x faster inference than other compressed models. Second, we propose a novel method to recompute lighting coefficients, significantly reducing their entropy without using any form of training. Our fast and highly parallel approach especially increases AC lighting coefficient sparsity, with experiments demonstrating increases from 70% to 97%, with minimal loss...
  </details>  
  Keywords: 3d scene reconstruction, novel view synthesis  
- **[Interp3D: Correspondence-aware Interpolation for Generative Textured 3D Morphing](https://arxiv.org/abs/2601.14103v1)**  
  Authors: Xiaolu Liu, Yicong Li, Qiyuan He, Jiayin Zhu, Wei Ji, Angela Yao, Jianke Zhu  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2601.14103v1.pdf) | [![GitHub](https://img.shields.io/github/stars/xiaolul2/Interp3D?style=social)](https://github.com/xiaolul2/Interp3D)  
  <details><summary>Abstract</summary>

  Textured 3D morphing seeks to generate smooth and plausible transitions between two 3D assets, preserving both structural coherence and fine-grained appearance. This ability is crucial not only for advancing 3D generation research but also for practical applications in animation, editing, and digital content creation. Existing approaches either operate directly on geometry, limiting them to shape-only morphing while neglecting textures, or extend 2D interpolation strategies into 3D, which often causes semantic ambiguity, structural misalignment, and texture blurring. These challenges underscore the necessity to jointly preserve geometric consistency, texture alignment, and robustness throughout the transition process. To address this, we propose Interp3D, a novel training-free framework for textured 3D morphing. It harnesses generative priors and adopts a progressive alignment principle to ensure both geometric fidelity and texture coherence. Starting from semantically aligned interpolation in condition space, Interp3D enforces structural consistency via SLAT (Structured Latent)-guided structure interpolation, and finally transfers appearance details through...
  </details>  
- **[Spherical Geometry Diffusion: Generating High-quality 3D Face Geometry via Sphere-anchored Representations](https://arxiv.org/abs/2601.13371v1)**  
  Authors: Junyi Zhang, Yiming Wang, Yunhong Lu, Qichao Wang, Wenzhe Qian, Xiaoyin Xu, David Gu, Min Zhang  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2601.13371v1.pdf)  
  <details><summary>Abstract</summary>

  A fundamental challenge in text-to-3D face generation is achieving high-quality geometry. The core difficulty lies in the arbitrary and intricate distribution of vertices in 3D space, making it challenging for existing models to establish clean connectivity and resulting in suboptimal geometry. To address this, our core insight is to simplify the underlying geometric structure by constraining the distribution onto a simple and regular manifold, a topological sphere. Building on this, we first propose the Spherical Geometry Representation, a novel face representation that anchors geometric signals to uniform spherical coordinates. This guarantees a regular point distribution, from which the mesh connectivity can be robustly reconstructed. Critically, this canonical sphere can be seamlessly unwrapped into a 2D map, creating a perfect synergy with powerful 2D generative models. We then introduce Spherical Geometry Diffusion, a conditional diffusion framework built upon this 2D map. It enables diverse and controllable generation by jointly modeling geometry...
  </details>  
  Keywords: text-to-3d, texture synthesis  
- **[Generalizable and Animatable 3D Full-Head Gaussian Avatar from a Single Image](https://arxiv.org/abs/2601.12770v1)**  
  Authors: Shuling Zhao, Dan Xu  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2601.12770v1.pdf)  
  <details><summary>Abstract</summary>

  Building 3D animatable head avatars from a single image is an important yet challenging problem. Existing methods generally collapse under large camera pose variations, compromising the realism of 3D avatars. In this work, we propose a new framework to tackle the novel setting of one-shot 3D full-head animatable avatar reconstruction in a single feed-forward pass, enabling real-time animation and simultaneous 360$^\circ$ rendering views. To facilitate efficient animation control, we model 3D head avatars with Gaussian primitives embedded on the surface of a parametric face model within the UV space. To obtain knowledge of full-head geometry and textures, we leverage rich 3D full-head priors within a pretrained 3D generative adversarial network (GAN) for global full-head feature extraction and multi-view supervision. To increase the fidelity of the 3D reconstruction of the input image, we take advantage of the symmetric nature of the UV space and human faces to fuse local fine-grained input...
  </details>  
- **[KaoLRM: Repurposing Pre-trained Large Reconstruction Models for Parametric 3D Face Reconstruction](https://arxiv.org/abs/2601.12736v1)**  
  Authors: Qingtian Zhu, Xu Cao, Zhixiang Wang, Yinqiang Zheng, Takafumi Taketomi  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2601.12736v1.pdf) | [![GitHub](https://img.shields.io/github/stars/CyberAgentAILab/KaoLRM?style=social)](https://github.com/CyberAgentAILab/KaoLRM)  
  <details><summary>Abstract</summary>

  We propose KaoLRM to re-target the learned prior of the Large Reconstruction Model (LRM) for parametric 3D face reconstruction from single-view images. Parametric 3D Morphable Models (3DMMs) have been widely used for facial reconstruction due to their compact and interpretable parameterization, yet existing 3DMM regressors often exhibit poor consistency across varying viewpoints. To address this, we harness the pre-trained 3D prior of LRM and incorporate FLAME-based 2D Gaussian Splatting into LRM's rendering pipeline. Specifically, KaoLRM projects LRM's pre-trained triplane features into the FLAME parameter space to recover geometry, and models appearance via 2D Gaussian primitives that are tightly coupled to the FLAME mesh. The rich prior enables the FLAME regressor to be aware of the 3D structure, leading to accurate and robust reconstructions under self-occlusions and diverse viewpoints. Experiments on both controlled and in-the-wild benchmarks demonstrate that KaoLRM achieves superior reconstruction accuracy and cross-view consistency, while existing methods remain sensitive...
  </details>  
- **[Proc3D: Procedural 3D Generation and Parametric Editing of 3D Shapes with Large Language Models](https://arxiv.org/abs/2601.12234v1)**  
  Authors: Fadlullah Raji, Stefano Petrangeli, Matheus Gadelha, Yu Shen, Uttaran Bhattacharya, Gang Wu  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2601.12234v1.pdf)  
  <details><summary>Abstract</summary>

  Generating 3D models has traditionally been a complex task requiring specialized expertise. While recent advances in generative AI have sought to automate this process, existing methods produce non-editable representation, such as meshes or point clouds, limiting their adaptability for iterative design. In this paper, we introduce Proc3D, a system designed to generate editable 3D models while enabling real-time modifications. At its core, Proc3D introduces procedural compact graph (PCG), a graph representation of 3D models, that encodes the algorithmic rules and structures necessary for generating the model. This representation exposes key parameters, allowing intuitive manual adjustments via sliders and checkboxes, as well as real-time, automated modifications through natural language prompts using Large Language Models (LLMs). We demonstrate Proc3D's capabilities using two generative approaches: GPT-4o with in-context learning (ICL) and a fine-tuned LLAMA-3 model. Experimental results show that Proc3D outperforms existing methods in editing efficiency, achieving more than 400x speedup over conventional...
  </details>  
- **[studentSplat: Your Student Model Learns Single-view 3D Gaussian Splatting](https://arxiv.org/abs/2601.11772v1)**  
  Authors: Yimu Pan, Hongda Mao, Qingshuang Chen, Yelin Kim  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2601.11772v1.pdf)  
  <details><summary>Abstract</summary>

  Recent advance in feed-forward 3D Gaussian splatting has enable remarkable multi-view 3D scene reconstruction or single-view 3D object reconstruction but single-view 3D scene reconstruction remain under-explored due to inherited ambiguity in single-view. We present \textbf{studentSplat}, a single-view 3D Gaussian splatting method for scene reconstruction. To overcome the scale ambiguity and extrapolation problems inherent in novel-view supervision from a single input, we introduce two techniques: 1) a teacher-student architecture where a multi-view teacher model provides geometric supervision to the single-view student during training, addressing scale ambiguity and encourage geometric validity; and 2) an extrapolation network that completes missing scene context, enabling high-quality extrapolation. Extensive experiments show studentSplat achieves state-of-the-art single-view novel-view reconstruction quality and comparable performance to multi-view methods at the scene level. Furthermore, studentSplat demonstrates competitive performance as a self-supervised single-view depth estimation method, highlighting its potential for general single-view 3D understanding tasks.
  </details>  
  Keywords: 3d scene reconstruction  
- **[ATATA: One Algorithm to Align Them All](https://arxiv.org/abs/2601.11194v2)**  
  Authors: Boyi Pang, Savva Ignatyev, Vladimir Ippolitov, Ramil Khafizov, Yurii Melnik, Oleg Voynov, Maksim Nakhodnov, Aibek Alanov, Xiaopeng Fan, Peter Wonka, Evgeny Burnaev  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2601.11194v2.pdf)  
  <details><summary>Abstract</summary>

  We suggest a new multi-modal algorithm for joint inference of paired structurally aligned samples with Rectified Flow models. While some existing methods propose a codependent generation process, they do not view the problem of joint generation from a structural alignment perspective. Recent work uses Score Distillation Sampling to generate aligned 3D models, but SDS is known to be time-consuming, prone to mode collapse, and often provides cartoonish results. By contrast, our suggested approach relies on the joint transport of a segment in the sample space, yielding faster computation at inference time. Our approach can be built on top of an arbitrary Rectified Flow model operating on the structured latent space. We show the applicability of our method to the domains of image, video, and 3D shape generation using state-of-the-art baselines and evaluate it against both editing-based and joint inference-based competing approaches. We demonstrate a high degree of structural alignment for...
  </details>  
- **[SyncTwin: Fast Digital Twin Construction and Synchronization for Safe Robotic Manipulation](https://arxiv.org/abs/2601.09920v2)**  
  Authors: Ruopeng Huang, Boyu Yang, Wenlong Gui, Jeremy Morgan, Erdem Biyik, Jiachen Li  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2601.09920v2.pdf) | [![Project](https://img.shields.io/badge/-Project-blue)](https://sync-twin.github.io)  
  <details><summary>Abstract</summary>

  Accurate and safe robotic manipulation under dynamic and visually occluded conditions remains a core challenge in real-world deployment. We introduce SyncTwin, a novel digital twin framework that unifies fast 3D scene reconstruction and real-to-sim synchronization for robust and safety-aware robotic manipulation in such environments. In the offline stage, we employ VGGT to rapidly reconstruct object-level 3D assets from RGB images, forming a reusable geometry library. During execution, SyncTwin continuously synchronizes the digital twin by tracking real-world object states via point cloud segmentation updates and aligning them through colored-ICP registration. The synchronized twin enables motion planners to compute collision-free and dynamically feasible trajectories in simulation, which are safely executed on the real robot through a closed real-to-sim-to-real loop. Experiments in dynamic and occluded scenes show that SyncTwin improves manipulation performance and motion safety, demonstrating the effectiveness of digital twin synchronization for real-world robotic execution. The video demos and code can be...
  </details>  
  Keywords: 3d scene reconstruction  
- **[StdGEN++: A Comprehensive System for Semantic-Decomposed 3D Character Generation](https://arxiv.org/abs/2601.07660v1)**  
  Authors: Yuze He, Yanning Zhou, Wang Zhao, Jingwen Ye, Zhongkai Wu, Ran Yi, Yong-Jin Liu  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2601.07660v1.pdf)  
  <details><summary>Abstract</summary>

  We present StdGEN++, a novel and comprehensive system for generating high-fidelity, semantically decomposed 3D characters from diverse inputs. Existing 3D generative methods often produce monolithic meshes that lack the structural flexibility required by industrial pipelines in gaming and animation. Addressing this gap, StdGEN++ is built upon a Dual-branch Semantic-aware Large Reconstruction Model (Dual-Branch S-LRM), which jointly reconstructs geometry, color, and per-component semantics in a feed-forward manner. To achieve production-level fidelity, we introduce a novel semantic surface extraction formalism compatible with hybrid implicit fields. This mechanism is accelerated by a coarse-to-fine proposal scheme, which significantly reduces memory footprint and enables high-resolution mesh generation. Furthermore, we propose a video-diffusion-based texture decomposition module that disentangles appearance into editable layers (e.g., separated iris and skin), resolving semantic confusion in facial regions. Experiments demonstrate that StdGEN++ achieves state-of-the-art performance, significantly outperforming existing methods in geometric accuracy and semantic disentanglement. Crucially, the resulting structural independence unlocks...
  </details>  
- **[HERE: Hierarchical Active Exploration of Radiance Field with Epistemic Uncertainty Minimization](https://arxiv.org/abs/2601.07242v2)**  
  Authors: Taekbeom Lee, Dabin Kim, Youngseok Jang, H. Jin Kim  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2601.07242v2.pdf) | [![Project](https://img.shields.io/badge/-Project-blue)](https://taekbum.github.io/here)  
  <details><summary>Abstract</summary>

  We present HERE, an active 3D scene reconstruction framework based on neural radiance fields, enabling high-fidelity implicit mapping. Our approach centers around an active learning strategy for camera trajectory generation, driven by accurate identification of unseen regions, which supports efficient data acquisition and precise scene reconstruction. The key to our approach is epistemic uncertainty quantification based on evidential deep learning, which directly captures data insufficiency and exhibits a strong correlation with reconstruction errors. This allows our framework to more reliably identify unexplored or poorly reconstructed regions compared to existing methods, leading to more informed and targeted exploration. Additionally, we design a hierarchical exploration strategy that leverages learned epistemic uncertainty, where local planning extracts target viewpoints from high-uncertainty voxels based on visibility for trajectory generation, and global planning uses uncertainty to guide large-scale coverage for efficient and comprehensive reconstruction. The effectiveness of the proposed method in active 3D reconstruction is demonstrated...
  </details>  
  Keywords: 3d scene reconstruction  
- **[OSCAR: Optical-aware Semantic Control for Aleatoric Refinement in Sar-to-Optical Translation](https://arxiv.org/abs/2601.06835v1)**  
  Authors: Hyunseo Lee, Sang Min Kim, Ho Kyung Shin, Taeheon Kim, Woo-Jeoung Nam  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2601.06835v1.pdf)  
  <details><summary>Abstract</summary>

  Synthetic Aperture Radar (SAR) provides robust all-weather imaging capabilities; however, translating SAR observations into photo-realistic optical images remains a fundamentally ill-posed problem. Current approaches are often hindered by the inherent speckle noise and geometric distortions of SAR data, which frequently result in semantic misinterpretation, ambiguous texture synthesis, and structural hallucinations. To address these limitations, a novel SAR-to-Optical (S2O) translation framework is proposed, integrating three core technical contributions: (i) Cross-Modal Semantic Alignment, which establishes an Optical-Aware SAR Encoder by distilling robust semantic priors from an Optical Teacher into a SAR Student (ii) Semantically-Grounded Generative Guidance, realized by a Semantically-Grounded ControlNet that integrates class-aware text prompts for global context with hierarchical visual prompts for local spatial guidance; and (iii) an Uncertainty-Aware Objective, which explicitly models aleatoric uncertainty to dynamically modulate the reconstruction focus, effectively mitigating artifacts caused by speckle-induced ambiguity. Extensive experiments demonstrate that the proposed method achieves superior perceptual quality and...
  </details>  
  Keywords: texture synthesis  
- **[Rotate Your Character: Revisiting Video Diffusion Models for High-Quality 3D Character Generation](https://arxiv.org/abs/2601.05722v1)**  
  Authors: Jin Wang, Jianxiang Lu, Comi Chen, Guangzheng Xu, Haoyu Yang, Peng Chen, Na Zhang, Yifan Xu, Longhuang Wu, Shuai Shao, Qinglin Lu, Ping Luo  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2601.05722v1.pdf)  
  <details><summary>Abstract</summary>

  Generating high-quality 3D characters from single images remains a significant challenge in digital content creation, particularly due to complex body poses and self-occlusion. In this paper, we present RCM (Rotate your Character Model), an advanced image-to-video diffusion framework tailored for high-quality novel view synthesis (NVS) and 3D character generation. Compared to existing diffusion-based approaches, RCM offers several key advantages: (1) transferring characters with any complex poses into a canonical pose, enabling consistent novel view synthesis across the entire viewing orbit, (2) high-resolution orbital video generation at 1024x1024 resolution, (3) controllable observation positions given different initial camera poses, and (4) multi-view conditioning supporting up to 4 input images, accommodating diverse user scenarios. Extensive experiments demonstrate that RCM outperforms state-of-the-art methods in both novel view synthesis and 3D generation quality.
  </details>  
  Keywords: novel view synthesis  
- **[Gen3R: 3D Scene Generation Meets Feed-Forward Reconstruction](https://arxiv.org/abs/2601.04090v2)**  
  Authors: Jiaxin Huang, Yuanbo Yang, Bangbang Yang, Lin Ma, Yuewen Ma, Yiyi Liao  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2601.04090v2.pdf)  
  <details><summary>Abstract</summary>

  We present Gen3R, a method that bridges the strong priors of foundational reconstruction models and video diffusion models for scene-level 3D generation. We repurpose the VGGT reconstruction model to produce geometric latents by training an adapter on its tokens, which are regularized to align with the appearance latents of pre-trained video diffusion models. By jointly generating these disentangled yet aligned latents, Gen3R produces both RGB videos and corresponding 3D geometry, including camera poses, depth maps, and global point clouds. Experiments demonstrate that our approach achieves state-of-the-art results in single- and multi-image conditioned 3D scene generation. Additionally, our method can enhance the robustness of reconstruction by leveraging generative priors, demonstrating the mutual benefit of tightly coupling reconstruction and generative models.
  </details>  
- **[A Comparative Study of 3D Model Acquisition Methods for Synthetic Data Generation of Agricultural Products](https://arxiv.org/abs/2601.03784v1)**  
  Authors: Steven Moonen, Rob Salaets, Kenneth Batstone, Abdellatif Bey-Temsamani, Nick Michiels  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2601.03784v1.pdf)  
  <details><summary>Abstract</summary>

  In the manufacturing industry, computer vision systems based on artificial intelligence (AI) are widely used to reduce costs and increase production. Training these AI models requires a large amount of training data that is costly to acquire and annotate, especially in high-variance, low-volume manufacturing environments. A popular approach to reduce the need for real data is the use of synthetic data that is generated by leveraging computer-aided design (CAD) models available in the industry. However, in the agricultural industry these models are not readily available, increasing the difficulty in leveraging synthetic data. In this paper, we present different techniques for substituting CAD files to create synthetic datasets. We measure their relative performance when used to train an AI object detection model to separate stones and potatoes in a bin picking environment. We demonstrate that using highly representative 3D models acquired by scanning or using image-to-3D approaches can be used to...
  </details>  
  Keywords: image-to-3d  
- **[360-GeoGS: Geometrically Consistent Feed-Forward 3D Gaussian Splatting Reconstruction for 360 Images](https://arxiv.org/abs/2601.02102v1)**  
  Authors: Jiaqi Yao, Zhongmiao Yan, Jingyi Xu, Songpengcheng Xia, Yan Xiang, Ling Pei  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2601.02102v1.pdf)  
  <details><summary>Abstract</summary>

  3D scene reconstruction is fundamental for spatial intelligence applications such as AR, robotics, and digital twins. Traditional multi-view stereo struggles with sparse viewpoints or low-texture regions, while neural rendering approaches, though capable of producing high-quality results, require per-scene optimization and lack real-time efficiency. Explicit 3D Gaussian Splatting (3DGS) enables efficient rendering, but most feed-forward variants focus on visual quality rather than geometric consistency, limiting accurate surface reconstruction and overall reliability in spatial perception tasks. This paper presents a novel feed-forward 3DGS framework for 360 images, capable of generating geometrically consistent Gaussian primitives while maintaining high rendering quality. A Depth-Normal geometric regularization is introduced to couple rendered depth gradients with normal information, supervising Gaussian rotation, scale, and position to improve point cloud and surface accuracy. Experimental results show that the proposed method maintains high rendering quality while significantly improving geometric consistency, providing an effective solution for 3D reconstruction in spatial perception...
  </details>  
  Keywords: 3d scene reconstruction  
- **[UnrealPose: Leveraging Game Engine Kinematics for Large-Scale Synthetic Human Pose Data](https://arxiv.org/abs/2601.00991v1)**  
  Authors: Joshua Kawaguchi, Saad Manzur, Emily Gao Wang, Maitreyi Sinha, Bryan Vela, Yunxi Wang, Brandon Vela, Wayne B. Hayes  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2601.00991v1.pdf)  
  <details><summary>Abstract</summary>

  Diverse, accurately labeled 3D human pose data is expensive and studio-bound, while in-the-wild datasets lack known ground truth. We introduce UnrealPose-Gen, an Unreal Engine 5 pipeline built on Movie Render Queue for high-quality offline rendering. Our generated frames include: (i) 3D joints in world and camera coordinates, (ii) 2D projections and COCO-style keypoints with occlusion and joint-visibility flags, (iii) person bounding boxes, and (iv) camera intrinsics and extrinsics. We use UnrealPose-Gen to present UnrealPose-1M, an approximately one million frame corpus comprising eight sequences: five scripted "coherent" sequences spanning five scenes, approximately 40 actions, and five subjects; and three randomized sequences across three scenes, approximately 100 actions, and five subjects, all captured from diverse camera trajectories for broad viewpoint coverage. As a fidelity check, we report real-to-synthetic results on four tasks: image-to-3D pose, 2D keypoint detection, 2D-to-3D lifting, and person detection/segmentation. Though time and resources constrain us from an unlimited dataset,...
  </details>  
  Keywords: image-to-3d  
- **[MorphAny3D: Unleashing the Power of Structured Latent in 3D Morphing](https://arxiv.org/abs/2601.00204v3)**  
  Authors: Xiaokun Sun, Zeyu Cai, Hao Tang, Ying Tai, Jian Yang, Zhenyu Zhang  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2601.00204v3.pdf) | [![Project](https://img.shields.io/badge/-Project-blue)](https://xiaokunsun.github.io/MorphAny3D.github.io)  
  <details><summary>Abstract</summary>

  3D morphing remains challenging due to the difficulty of generating semantically consistent and temporally smooth deformations, especially across categories. We present MorphAny3D, a training-free framework that leverages Structured Latent (SLAT) representations for high-quality 3D morphing. Our key insight is that intelligently blending source and target SLAT features within the attention mechanisms of 3D generators naturally produces plausible morphing sequences. To this end, we introduce Morphing Cross-Attention (MCA), which fuses source and target information for structural coherence, and Temporal-Fused Self-Attention (TFSA), which enhances temporal consistency by incorporating features from preceding frames. An orientation correction strategy further mitigates the pose ambiguity within the morphing steps. Extensive experiments show that our method generates state-of-the-art morphing sequences, even for challenging cross-category cases. MorphAny3D further supports advanced applications such as decoupled morphing and 3D style transfer, and can be generalized to other SLAT-based generative models. Project page: https://xiaokunsun.github.io/MorphAny3D.github.io/.
  </details>  
  Keywords: 3d generator  

### February 2026
- **[Mesh-Pro: Asynchronous Advantage-guided Ranking Preference Optimization for Artist-style Quadrilateral Mesh Generation](https://arxiv.org/abs/2603.00526v1)**  
  Authors: Zhen Zhou, Jian Liu, Biwen Lei, Jing Xu, Haohan Weng, Yiling Zhu, Zhuo Chen, Junfeng Fan, Yunkai Ma, Dazhao Du, Song Guo, Fengshui Jing, Chunchao Guo  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2603.00526v1.pdf)  
  <details><summary>Abstract</summary>

  Reinforcement learning (RL) has demonstrated remarkable success in text and image generation, yet its potential in 3D generation remains largely unexplored. Existing attempts typically rely on offline direct preference optimization (DPO) method, which suffers from low training efficiency and limited generalization. In this work, we aim to enhance both the training efficiency and generation quality of RL in 3D mesh generation. Specifically, (1) we design the first asynchronous online RL framework tailored for 3D mesh generation post-training efficiency improvement, which is 3.75$\times$ faster than synchronous RL. (2) We propose Advantage-guided Ranking Preference Optimization (ARPO), a novel RL algorithm that achieves a better trade-off between training efficiency and generalization than current RL algorithms designed for 3D mesh generation, such as DPO and group relative policy optimization (GRPO). (3) Based on asynchronous ARPO, we propose Mesh-Pro, which additionally introduces a novel diagonal-aware mixed triangular-quadrilateral tokenization for mesh representation and a ray-based reward...
  </details>  
- **[HumanOrbit: 3D Human Reconstruction as 360° Orbit Generation](https://arxiv.org/abs/2602.24148v1)**  
  Authors: Keito Suzuki, Kunyao Chen, Lei Wang, Bang Du, Runfa Blark Li, Peng Liu, Ning Bi, Truong Nguyen  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2602.24148v1.pdf)  
  <details><summary>Abstract</summary>

  We present a method for generating a full 360° orbit video around a person from a single input image. Existing methods typically adapt image-based diffusion models for multi-view synthesis, but yield inconsistent results across views and with the original identity. In contrast, recent video diffusion models have demonstrated their ability in generating photorealistic results that align well with the given prompts. Inspired by these results, we propose HumanOrbit, a video diffusion model for multi-view human image generation. Our approach enables the model to synthesize continuous camera rotations around the subject, producing geometrically consistent novel views while preserving the appearance and identity of the person. Using the generated multi-view frames, we further propose a reconstruction pipeline that recovers a textured mesh of the subject. Experimental results validate the effectiveness of HumanOrbit for multi-view image generation and that the reconstructed 3D models exhibit superior completeness and fidelity compared to those from state-of-the-art...
  </details>  
  Keywords: multi-view synthesis  
- **[BuildAnyPoint: 3D Building Structured Abstraction from Diverse Point Clouds](https://arxiv.org/abs/2602.23645v1)**  
  Authors: Tongyan Hua, Haoran Gong, Yuan Liu, Di Wang, Ying-Cong Chen, Wufan Zhao  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2602.23645v1.pdf)  
  <details><summary>Abstract</summary>

  We introduce BuildAnyPoint, a novel generative framework for structured 3D building reconstruction from point clouds with diverse distributions, such as those captured by airborne LiDAR and Structure-from-Motion. To recover artist-created building abstraction in this highly underconstrained setting, we capitalize on the role of explicit 3D generative priors in autoregressive mesh generation. Specifically, we design a Loosely Cascaded Diffusion Transformer (Loca-DiT) that initially recovers the underlying distribution from noisy or sparse points, followed by autoregressively encapsulating them into compact meshes. We first formulate distribution recovery as a conditional generation task by training latent diffusion models conditioned on input point clouds, and then tailor a decoder-only transformer for conditional autoregressive mesh generation based on the recovered point clouds. Our method delivers substantial qualitative and quantitative improvements over prior building abstraction methods. Furthermore, the effectiveness of our approach is evidenced by the strong performance of its recovered point clouds on building point cloud...
  </details>  
- **[SceneTransporter: Optimal Transport-Guided Compositional Latent Diffusion for Single-Image Structured 3D Scene Generation](https://arxiv.org/abs/2602.22785v1)**  
  Authors: Ling Wang, Hao-Xiang Guo, Xinzhou Wang, Fuchun Sun, Kai Sun, Pengkun Liu, Hang Xiao, Zhong Wang, Guangyuan Fu, Eric Li, Yang Liu, Yikai Wang  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2602.22785v1.pdf) | [![Project](https://img.shields.io/badge/-Project-blue)](https://2019epwl.github.io/SceneTransporter)  
  <details><summary>Abstract</summary>

  We introduce SceneTransporter, an end-to-end framework for structured 3D scene generation from a single image. While existing methods generate part-level 3D objects, they often fail to organize these parts into distinct instances in open-world scenes. Through a debiased clustering probe, we reveal a critical insight: this failure stems from the lack of structural constraints within the model's internal assignment mechanism. Based on this finding, we reframe the task of structured 3D scene generation as a global correlation assignment problem. To solve this, SceneTransporter formulates and solves an entropic Optimal Transport (OT) objective within the denoising loop of the compositional DiT model. This formulation imposes two powerful structural constraints. First, the resulting transport plan gates cross-attention to enforce an exclusive, one-to-one routing of image patches to part-level 3D latents, preventing entanglement. Second, the competitive nature of the transport encourages the grouping of similar patches, a process that is further regularized by...
  </details>  
- **[CRAG: Can 3D Generative Models Help 3D Assembly?](https://arxiv.org/abs/2602.22629v1)**  
  Authors: Zeyu Jiang, Sihang Li, Siqi Tan, Chenyang Xu, Juexiao Zhang, Julia Galway-Witham, Xue Wang, Scott A. Williams, Radu Iovita, Chen Feng, Jing Zhang  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2602.22629v1.pdf)  
  <details><summary>Abstract</summary>

  Most existing 3D assembly methods treat the problem as pure pose estimation, rearranging observed parts via rigid transformations. In contrast, human assembly naturally couples structural reasoning with holistic shape inference. Inspired by this intuition, we reformulate 3D assembly as a joint problem of assembly and generation. We show that these two processes are mutually reinforcing: assembly provides part-level structural priors for generation, while generation injects holistic shape context that resolves ambiguities in assembly. Unlike prior methods that cannot synthesize missing geometry, we propose CRAG, which simultaneously generates plausible complete shapes and predicts poses for input parts. Extensive experiments demonstrate state-of-the-art performance across in-the-wild objects with diverse geometries, varying part counts, and missing pieces. Our code and models will be released.
  </details>  
- **[SF3D-RGB: Scene Flow Estimation from Monocular Camera and Sparse LiDAR](https://arxiv.org/abs/2602.21699v1)**  
  Authors: Rajai Alhimdiat, Ramy Battrawy, René Schuster, Didier Stricker, Wesam Ashour  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2602.21699v1.pdf)  
  <details><summary>Abstract</summary>

  Scene flow estimation is an extremely important task in computer vision to support the perception of dynamic changes in the scene. For robust scene flow, learning-based approaches have recently achieved impressive results using either image-based or LiDAR-based modalities. However, these methods have tended to focus on the use of a single modality. To tackle these problems, we present a deep learning architecture, SF3D-RGB, that enables sparse scene flow estimation using 2D monocular images and 3D point clouds (e.g., acquired by LiDAR) as inputs. Our architecture is an end-to-end model that first encodes information from each modality into features and fuses them together. Then, the fused features enhance a graph matching module for better and more robust mapping matrix computation to generate an initial scene flow. Finally, a residual scene flow module further refines the initial scene flow. Our model is designed to strike a balance between accuracy and efficiency. Furthermore,...
  </details>  
- **[Pseudo-View Enhancement via Confidence Fusion for Unposed Sparse-View Reconstruction](https://arxiv.org/abs/2602.21535v1)**  
  Authors: Beizhen Zhao, Sicheng Yu, Guanzhi Ding, Yu Hu, Hao Wang  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2602.21535v1.pdf)  
  <details><summary>Abstract</summary>

  3D scene reconstruction under unposed sparse viewpoints is a highly challenging yet practically important problem, especially in outdoor scenes due to complex lighting and scale variation. With extremely limited input views, directly utilizing diffusion model to synthesize pseudo frames will introduce unreasonable geometry, which will harm the final reconstruction quality. To address these issues, we propose a novel framework for sparse-view outdoor reconstruction that achieves high-quality results through bidirectional pseudo frame restoration and scene perception Gaussian management. Specifically, we introduce a bidirectional pseudo frame restoration method that restores missing content by diffusion-based synthesis guided by adjacent frames with a lightweight pseudo-view deblur model and confidence mask inference algorithm. Then we propose a scene perception Gaussian management strategy that optimize Gaussians based on joint depth-density information. These designs significantly enhance reconstruction completeness, suppress floating artifacts and improve overall geometric consistency under extreme view sparsity. Experiments on outdoor benchmarks demonstrate substantial gains...
  </details>  
  Keywords: 3d scene reconstruction  
- **[Human Video Generation from a Single Image with 3D Pose and View Control](https://arxiv.org/abs/2602.21188v1)**  
  Authors: Tiantian Wang, Chun-Han Yao, Tao Hu, Mallikarjun Byrasandra Ramalinga Reddy, Ming-Hsuan Yang, Varun Jampani  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2602.21188v1.pdf)  
  <details><summary>Abstract</summary>

  Recent diffusion methods have made significant progress in generating videos from single images due to their powerful visual generation capabilities. However, challenges persist in image-to-video synthesis, particularly in human video generation, where inferring view-consistent, motion-dependent clothing wrinkles from a single image remains a formidable problem. In this paper, we present Human Video Generation in 4D (HVG), a latent video diffusion model capable of generating high-quality, multi-view, spatiotemporally coherent human videos from a single image with 3D pose and view control. HVG achieves this through three key designs: (i) Articulated Pose Modulation, which captures the anatomical relationships of 3D joints via a novel dual-dimensional bone map and resolves self-occlusions across views by introducing 3D information; (ii) View and Temporal Alignment, which ensures multi-view consistency and alignment between a reference image and pose sequences for frame-to-frame stability; and (iii) Progressive Spatio-Temporal Sampling with temporal alignment to maintain smooth transitions in long multi-view...
  </details>  
- **[BoxSplitGen: A Generative Model for 3D Part Bounding Boxes in Varying Granularity](https://arxiv.org/abs/2602.20666v1)**  
  Authors: Juil Koo, Wei-Tung Lin, Chanho Park, Chanhyeok Park, Minhyuk Sung  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2602.20666v1.pdf)  
  <details><summary>Abstract</summary>

  Human creativity follows a perceptual process, moving from abstract ideas to finer details during creation. While 3D generative models have advanced dramatically, models specifically designed to assist human imagination in 3D creation -- particularly for detailing abstractions from coarse to fine -- have not been explored. We propose a framework that enables intuitive and interactive 3D shape generation by iteratively splitting bounding boxes to refine the set of bounding boxes. The main technical components of our framework are two generative models: the box-splitting generative model and the box-to-shape generative model. The first model, named BoxSplitGen, generates a collection of 3D part bounding boxes with varying granularity by iteratively splitting coarse bounding boxes. It utilizes part bounding boxes created through agglomerative merging and learns the reverse of the merging process -- the splitting sequences. The model consists of two main components: the first learns the categorical distribution of the box to...
  </details>  
- **[SceMoS: Scene-Aware 3D Human Motion Synthesis by Planning with Geometry-Grounded Tokens](https://arxiv.org/abs/2602.20476v1)**  
  Authors: Anindita Ghosh, Vladislav Golyanik, Taku Komura, Philipp Slusallek, Christian Theobalt, Rishabh Dabral  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2602.20476v1.pdf)  
  <details><summary>Abstract</summary>

  Synthesizing text-driven 3D human motion within realistic scenes requires learning both semantic intent ("walk to the couch") and physical feasibility (e.g., avoiding collisions). Current methods use generative frameworks that simultaneously learn high-level planning and low-level contact reasoning, and rely on computationally expensive 3D scene data such as point clouds or voxel occupancy grids. We propose SceMoS, a scene-aware motion synthesis framework that shows that structured 2D scene representations can serve as a powerful alternative to full 3D supervision in physically grounded motion synthesis. SceMoS disentangles global planning from local execution using lightweight 2D cues and relying on (1) a text-conditioned autoregressive global motion planner that operates on a bird's-eye-view (BEV) image rendered from an elevated corner of the scene, encoded with DINOv2 features, as the scene representation, and (2) a geometry-grounded motion tokenizer trained via a conditional VQ-VAE, that uses 2D local scene heightmap, thus embedding surface physics directly into...
  </details>  
- **[CLIPoint3D: Language-Grounded Few-Shot Unsupervised 3D Point Cloud Domain Adaptation](https://arxiv.org/abs/2602.20409v2)**  
  Authors: Mainak Singha, Sarthak Mehrotra, Paolo Casari, Subhasis Chaudhuri, Elisa Ricci, Biplab Banerjee  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2602.20409v2.pdf) | [![Project](https://img.shields.io/badge/-Project-blue)](https://sarthakm320.github.io/CLIPoint3D)  
  <details><summary>Abstract</summary>

  Recent vision-language models (VLMs) such as CLIP demonstrate impressive cross-modal reasoning, extending beyond images to 3D perception. Yet, these models remain fragile under domain shifts, especially when adapting from synthetic to real-world point clouds. Conventional 3D domain adaptation approaches rely on heavy trainable encoders, yielding strong accuracy but at the cost of efficiency. We introduce CLIPoint3D, the first framework for few-shot unsupervised 3D point cloud domain adaptation built upon CLIP. Our approach projects 3D samples into multiple depth maps and exploits the frozen CLIP backbone, refined through a knowledge-driven prompt tuning scheme that integrates high-level language priors with geometric cues from a lightweight 3D encoder. To adapt task-specific features effectively, we apply parameter-efficient fine-tuning to CLIP's encoders and design an entropy-guided view sampling strategy for selecting confident projections. Furthermore, an optimal transport-based alignment loss and an uncertainty-aware prototype alignment loss collaboratively bridge source-target distribution gaps while maintaining class separability. Extensive...
  </details>  
- **[Large-scale Photorealistic Outdoor 3D Scene Reconstruction from UAV Imagery Using Gaussian Splatting Techniques](https://arxiv.org/abs/2602.20342v1)**  
  Authors: Christos Maikos, Georgios Angelidis, Georgios Th. Papadopoulos  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2602.20342v1.pdf)  
  <details><summary>Abstract</summary>

  In this study, we present an end-to-end pipeline capable of converting drone-captured video streams into high-fidelity 3D reconstructions with minimal latency. Unmanned aerial vehicles (UAVs) are extensively used in aerial real-time perception applications. Moreover, recent advances in 3D Gaussian Splatting (3DGS) have demonstrated significant potential for real-time neural rendering. However, their integration into end-to-end UAV-based reconstruction and visualization systems remains underexplored. Our goal is to propose an efficient architecture that combines live video acquisition via RTMP streaming, synchronized sensor fusion, camera pose estimation, and 3DGS optimization, achieving continuous model updates and low-latency deployment within interactive visualization environments that supports immersive augmented and virtual reality (AR/VR) applications. Experimental results demonstrate that the proposed method achieves competitive visual fidelity, while delivering significantly higher rendering performance and substantially reduced end-to-end latency, compared to NeRF-based approaches. Reconstruction quality remains within 4-7\% of high-fidelity offline references, confirming the suitability of the proposed system for real-time,...
  </details>  
  Keywords: 3d scene reconstruction  
- **[BigMaQ: A Big Macaque Motion and Animation Dataset Bridging Image and 3D Pose Representations](https://arxiv.org/abs/2602.19874v1)**  
  Authors: Lucas Martini, Alexander Lappe, Anna Bognár, Rufin Vogels, Martin A. Giese  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2602.19874v1.pdf) | [![Project](https://img.shields.io/badge/-Project-blue)](https://martinivis.github.io/BigMaQ)  
  <details><summary>Abstract</summary>

  The recognition of dynamic and social behavior in animals is fundamental for advancing ethology, ecology, medicine and neuroscience. Recent progress in deep learning has enabled automated behavior recognition from video, yet an accurate reconstruction of the three-dimensional (3D) pose and shape has not been integrated into this process. Especially for non-human primates, mesh-based tracking efforts lag behind those for other species, leaving pose descriptions restricted to sparse keypoints that are unable to fully capture the richness of action dynamics. To address this gap, we introduce the $\textbf{Big Ma}$ca$\textbf{Q}$ue 3D Motion and Animation Dataset ($\texttt{BigMaQ}$), a large-scale dataset comprising more than 750 scenes of interacting rhesus macaques with detailed 3D pose descriptions. Extending previous surface-based animal tracking methods, we construct subject-specific textured avatars by adapting a high-quality macaque template mesh to individual monkeys. This allows us to provide pose descriptions that are more accurate than previous state-of-the-art surface-based animal tracking methods....
  </details>  
- **[RAP: Fast Feedforward Rendering-Free Attribute-Guided Primitive Importance Score Prediction for Efficient 3D Gaussian Splatting Processing](https://arxiv.org/abs/2602.19753v1)**  
  Authors: Kaifa Yang, Qi Yang, Yiling Xu, Zhu Li  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2602.19753v1.pdf) | [![GitHub](https://img.shields.io/github/stars/yyyykf/RAP?style=social)](https://github.com/yyyykf/RAP)  
  <details><summary>Abstract</summary>

  3D Gaussian Splatting (3DGS) has emerged as a leading technology for high-quality 3D scene reconstruction. However, the iterative refinement and densification process leads to the generation of a large number of primitives, each contributing to the reconstruction to a substantially different extent. Estimating primitive importance is thus crucial, both for removing redundancy during reconstruction and for enabling efficient compression and transmission. Existing methods typically rely on rendering-based analyses, where each primitive is evaluated through its contribution across multiple camera viewpoints. However, such methods are sensitive to the number and selection of views, rely on specialized differentiable rasterizers, and have long calculation times that grow linearly with view count, making them difficult to integrate as plug-and-play modules and limiting scalability and generalization. To address these issues, we propose RAP, a fast feedforward rendering-free attribute-guided method for efficient importance score prediction in 3DGS. RAP infers primitive significance directly from intrinsic Gaussian attributes...
  </details>  
  Keywords: 3d scene reconstruction  
- **[TeHOR: Text-Guided 3D Human and Object Reconstruction with Textures](https://arxiv.org/abs/2602.19679v1)**  
  Authors: Hyeongjin Nam, Daniel Sungho Jung, Kyoung Mu Lee  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2602.19679v1.pdf)  
  <details><summary>Abstract</summary>

  Joint reconstruction of 3D human and object from a single image is an active research area, with pivotal applications in robotics and digital content creation. Despite recent advances, existing approaches suffer from two fundamental limitations. First, their reconstructions rely heavily on physical contact information, which inherently cannot capture non-contact human-object interactions, such as gazing at or pointing toward an object. Second, the reconstruction process is primarily driven by local geometric proximity, neglecting the human and object appearances that provide global context crucial for understanding holistic interactions. To address these issues, we introduce TeHOR, a framework built upon two core designs. First, beyond contact information, our framework leverages text descriptions of human-object interactions to enforce semantic alignment between the 3D reconstruction and its textual cues, enabling reasoning over a wider spectrum of interactions, including non-contact cases. Second, we incorporate appearance cues of the 3D human and object into the alignment process...
  </details>  
- **[Vinedresser3D: Agentic Text-guided 3D Editing](https://arxiv.org/abs/2602.19542v1)**  
  Authors: Yankuan Chi, Xiang Li, Zixuan Huang, James M. Rehg  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2602.19542v1.pdf)  
  <details><summary>Abstract</summary>

  Text-guided 3D editing aims to modify existing 3D assets using natural-language instructions. Current methods struggle to jointly understand complex prompts, automatically localize edits in 3D, and preserve unedited content. We introduce Vinedresser3D, an agentic framework for high-quality text-guided 3D editing that operates directly in the latent space of a native 3D generative model. Given a 3D asset and an editing prompt, Vinedresser3D uses a multimodal large language model to infer rich descriptions of the original asset, identify the edit region and edit type (addition, modification, deletion), and generate decomposed structural and appearance-level text guidance. The agent then selects an informative view and applies an image editing model to obtain visual guidance. Finally, an inversion-based rectified-flow inpainting pipeline with an interleaved sampling module performs editing in the 3D latent space, enforcing prompt alignment while maintaining 3D coherence and unedited regions. Experiments on diverse 3D edits demonstrate that Vinedresser3D outperforms prior baselines...
  </details>  
- **[Advances in GRPO for Generation Models: A Survey](https://arxiv.org/abs/2603.06623v1)**  
  Authors: Zexiang Liu, Xianglong He, Yangguang Li  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2603.06623v1.pdf)  
  <details><summary>Abstract</summary>

  Large-scale flow matching models have achieved strong performance across generative tasks such as text-to-image, video, 3D, and speech synthesis. However, aligning their outputs with human preferences and task-specific objectives remains challenging. Flow-GRPO extends Group Relative Policy Optimization (GRPO) to generation models, enabling stable reinforcement learning alignment for generative systems. Since its introduction, Flow-GRPO has triggered rapid research growth, spanning methodological refinements and diverse application domains. This survey provides a comprehensive review of Flow-GRPO and its subsequent developments. We organize existing work along two primary dimensions. First, we analyze methodological advances beyond the original framework, including reward signal design, credit assignment, sampling efficiency, diversity preservation, reward hacking mitigation, and reward model construction. Second, we examine extensions of GRPO-based alignment across generative paradigms and modalities, including text-to-image, video generation, image editing, speech and audio, 3D modeling, embodied vision-language-action systems, unified multimodal models, autoregressive and masked diffusion models, and restoration tasks. By synthesizing...
  </details>  
- **[3D Scene Rendering with Multimodal Gaussian Splatting](https://arxiv.org/abs/2602.17124v1)**  
  Authors: Chi-Shiang Gau, Konstantinos D. Polyzos, Athanasios Bacharis, Saketh Madhuvarasu, Tara Javidi  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2602.17124v1.pdf)  
  <details><summary>Abstract</summary>

  3D scene reconstruction and rendering are core tasks in computer vision, with applications spanning industrial monitoring, robotics, and autonomous driving. Recent advances in 3D Gaussian Splatting (GS) and its variants have achieved impressive rendering fidelity while maintaining high computational and memory efficiency. However, conventional vision-based GS pipelines typically rely on a sufficient number of camera views to initialize the Gaussian primitives and train their parameters, typically incurring additional processing cost during initialization while falling short in conditions where visual cues are unreliable, such as adverse weather, low illumination, or partial occlusions. To cope with these challenges, and motivated by the robustness of radio-frequency (RF) signals to weather, lighting, and occlusions, we introduce a multimodal framework that integrates RF sensing, such as automotive radar, with GS-based rendering as a more efficient and robust alternative to vision-only GS rendering. The proposed approach enables efficient depth prediction from only sparse RF-based depth measurements,...
  </details>  
  Keywords: 3d scene reconstruction  
- **[Cholec80-port: A Geometrically Consistent Trocar Port Segmentation Dataset for Robust Surgical Scene Understanding](https://arxiv.org/abs/2602.17060v1)**  
  Authors: Shunsuke Kikuchi, Atsushi Kouno, Hiroki Matsuzaki  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2602.17060v1.pdf)  
  <details><summary>Abstract</summary>

  Trocar ports are camera-fixed, pseudo-static structures that can persistently occlude laparoscopic views and attract disproportionate feature points due to specular, textured surfaces. This makes ports particularly detrimental to geometry-based downstream pipelines such as image stitching, 3D reconstruction, and visual SLAM, where dynamic or non-anatomical outliers degrade alignment and tracking stability. Despite this practical importance, explicit port labels are rare in public surgical datasets, and existing annotations often violate geometric consistency by masking the central lumen (opening), even when anatomical regions are visible through it. We present Cholec80-port, a high-fidelity trocar port segmentation dataset derived from Cholec80, together with a rigorous standard operating procedure (SOP) that defines a port-sleeve mask excluding the central opening. We additionally cleanse and unify existing public datasets under the same SOP. Experiments demonstrate that geometrically consistent annotations substantially improve cross-dataset robustness beyond what dataset size alone provides.
  </details>  
- **[PartRAG: Retrieval-Augmented Part-Level 3D Generation and Editing](https://arxiv.org/abs/2602.17033v1)**  
  Authors: Peize Li, Zeyu Zhang, Hao Tang  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2602.17033v1.pdf) | [![GitHub](https://img.shields.io/github/stars/AIGeeksGroup/PartRAG?style=social)](https://github.com/AIGeeksGroup/PartRAG) | [![Project](https://img.shields.io/badge/-Project-blue)](https://aigeeksgroup.github.io/PartRAG)  
  <details><summary>Abstract</summary>

  Single-image 3D generation with part-level structure remains challenging: learned priors struggle to cover the long tail of part geometries and maintain multi-view consistency, and existing systems provide limited support for precise, localized edits. We present PartRAG, a retrieval-augmented framework that integrates an external part database with a diffusion transformer to couple generation with an editable representation. To overcome the first challenge, we introduce a Hierarchical Contrastive Retrieval module that aligns dense image patches with 3D part latents at both part and object granularity, retrieving from a curated bank of 1,236 part-annotated assets to inject diverse, physically plausible exemplars into denoising. To overcome the second challenge, we add a masked, part-level editor that operates in a shared canonical space, enabling swaps, attribute refinements, and compositional updates without regenerating the whole object while preserving non-target parts and multi-view consistency. PartRAG achieves competitive results on Objaverse, ShapeNet, and ABO-reducing Chamfer Distance from 0.1726...
  </details>  
- **[DressWild: Feed-Forward Pose-Agnostic Garment Sewing Pattern Generation from In-the-Wild Images](https://arxiv.org/abs/2602.16502v1)**  
  Authors: Zeng Tao, Ying Jiang, Yunuo Chen, Tianyi Xie, Huamin Wang, Yingnian Wu, Yin Yang, Abishek Sampath Kumar, Kenji Tashiro, Chenfanfu Jiang  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2602.16502v1.pdf)  
  <details><summary>Abstract</summary>

  Recent advances in garment pattern generation have shown promising progress. However, existing feed-forward methods struggle with diverse poses and viewpoints, while optimization-based approaches are computationally expensive and difficult to scale. This paper focuses on sewing pattern generation for garment modeling and fabrication applications that demand editable, separable, and simulation-ready garments. We propose DressWild, a novel feed-forward pipeline that reconstructs physics-consistent 2D sewing patterns and the corresponding 3D garments from a single in-the-wild image. Given an input image, our method leverages vision-language models (VLMs) to normalize pose variations at the image level, then extract pose-aware, 3D-informed garment features. These features are fused through a transformer-based encoder and subsequently used to predict sewing pattern parameters, which can be directly applied to physical simulation, texture synthesis, and multi-layer virtual try-on. Extensive experiments demonstrate that our approach robustly recovers diverse sewing patterns and the corresponding 3D garments from in-the-wild images without requiring multi-view inputs...
  </details>  
  Keywords: texture synthesis  
- **[MeshMimic: Geometry-Aware Humanoid Motion Learning through 3D Scene Reconstruction](https://arxiv.org/abs/2602.15733v1)**  
  Authors: Qiang Zhang, Jiahao Ma, Peiran Liu, Shuai Shi, Zeran Su, Zifan Wang, Jingkai Sun, Wei Cui, Jialin Yu, Gang Han, Wen Zhao, Pihai Sun, Kangning Yin, Jiaxu Wang, Jiahang Cao, Lingfeng Zhang, Hao Cheng, Xiaoshuai Hao, Yiding Ji, Junwei Liang, Jian Tang, Renjing Xu, Yijie Guo  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2602.15733v1.pdf)  
  <details><summary>Abstract</summary>

  Humanoid motion control has witnessed significant breakthroughs in recent years, with deep reinforcement learning (RL) emerging as a primary catalyst for achieving complex, human-like behaviors. However, the high dimensionality and intricate dynamics of humanoid robots make manual motion design impractical, leading to a heavy reliance on expensive motion capture (MoCap) data. These datasets are not only costly to acquire but also frequently lack the necessary geometric context of the surrounding physical environment. Consequently, existing motion synthesis frameworks often suffer from a decoupling of motion and scene, resulting in physical inconsistencies such as contact slippage or mesh penetration during terrain-aware tasks. In this work, we present MeshMimic, an innovative framework that bridges 3D scene reconstruction and embodied intelligence to enable humanoid robots to learn coupled "motion-terrain" interactions directly from video. By leveraging state-of-the-art 3D vision models, our framework precisely segments and reconstructs both human trajectories and the underlying 3D geometry of...
  </details>  
  Keywords: 3d scene reconstruction  
- **[PAct: Part-Decomposed Single-View Articulated Object Generation](https://arxiv.org/abs/2602.14965v1)**  
  Authors: Qingming Liu, Xinyue Yao, Shuyuan Zhang, Yueci Deng, Guiliang Liu, Zhen Liu, Kui Jia  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2602.14965v1.pdf)  
  <details><summary>Abstract</summary>

  Articulated objects are central to interactive 3D applications, including embodied AI, robotics, and VR/AR, where functional part decomposition and kinematic motion are essential. Yet producing high-fidelity articulated assets remains difficult to scale because it requires reliable part decomposition and kinematic rigging. Existing approaches largely fall into two paradigms: optimization-based reconstruction or distillation, which can be accurate but often takes tens of minutes to hours per instance, and inference-time methods that rely on template or part retrieval, producing plausible results that may not match the specific structure and appearance in the input observation. We introduce a part-centric generative framework for articulated object creation that synthesizes part geometry, composition, and articulation under explicit part-aware conditioning. Our representation models an object as a set of movable parts, each encoded by latent tokens augmented with part identity and articulation cues. Conditioned on a single image, the model generates articulated 3D assets that preserve instance-level...
  </details>  
  Keywords: articulated object generation, rigging  
- **[VAR-3D: View-aware Auto-Regressive Model for Text-to-3D Generation via a 3D Tokenizer](https://arxiv.org/abs/2602.13818v1)**  
  Authors: Zongcheng Han, Dongyan Cao, Haoran Sun, Yu Hong  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2602.13818v1.pdf)  
  <details><summary>Abstract</summary>

  Recent advances in auto-regressive transformers have achieved remarkable success in generative modeling. However, text-to-3D generation remains challenging, primarily due to bottlenecks in learning discrete 3D representations. Specifically, existing approaches often suffer from information loss during encoding, causing representational distortion before the quantization process. This effect is further amplified by vector quantization, ultimately degrading the geometric coherence of text-conditioned 3D shapes. Moreover, the conventional two-stage training paradigm induces an objective mismatch between reconstruction and text-conditioned auto-regressive generation. To address these issues, we propose View-aware Auto-Regressive 3D (VAR-3D), which intergrates a view-aware 3D Vector Quantized-Variational AutoEncoder (VQ-VAE) to convert the complex geometric structure of 3D models into discrete tokens. Additionally, we introduce a rendering-supervised training strategy that couples discrete token prediction with visual reconstruction, encouraging the generative process to better preserve visual fidelity and structural consistency relative to the input text. Experiments demonstrate that VAR-3D significantly outperforms existing methods in both generation...
  </details>  
  Keywords: text-to-3d  
- **[AssetFormer: Modular 3D Assets Generation with Autoregressive Transformer](https://arxiv.org/abs/2602.12100v1)**  
  Authors: Lingting Zhu, Shengju Qian, Haidi Fan, Jiayu Dong, Zhenchao Jin, Siwei Zhou, Gen Dong, Xin Wang, Lequan Yu  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2602.12100v1.pdf) | [![GitHub](https://img.shields.io/github/stars/Advocate99/AssetFormer?style=social)](https://github.com/Advocate99/AssetFormer)  
  <details><summary>Abstract</summary>

  The digital industry demands high-quality, diverse modular 3D assets, especially for user-generated content~(UGC). In this work, we introduce AssetFormer, an autoregressive Transformer-based model designed to generate modular 3D assets from textual descriptions. Our pilot study leverages real-world modular assets collected from online platforms. AssetFormer tackles the challenge of creating assets composed of primitives that adhere to constrained design parameters for various applications. By innovatively adapting module sequencing and decoding techniques inspired by language models, our approach enhances asset generation quality through autoregressive modeling. Initial results indicate the effectiveness of AssetFormer in streamlining asset creation for professional development and UGC scenarios. This work presents a flexible framework extendable to various types of modular 3D assets, contributing to the broader field of 3D content generation. The code is available at https://github.com/Advocate99/AssetFormer.
  </details>  
- **[Bioinspired123D: Generative 3D Modeling System for Bioinspired Structures](https://arxiv.org/abs/2603.29592v1)**  
  Authors: Rachel K. Luu, Markus J. Buehler  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2603.29592v1.pdf)  
  <details><summary>Abstract</summary>

  Generative AI has made rapid progress in text, image, and video synthesis, yet text-to-3D modeling for scientific design remains particularly challenging due to limited controllability and high computational cost. Most existing 3D generative methods rely on meshes, voxels, or point clouds which can be costly to train and difficult to control. We introduce Bioinspired123D, a lightweight and modular code-as-geometry pipeline that generates fabricable 3D structures directly through parametric programs rather than dense visual representations. At the core of Bioinspired123D is Bioinspired3D, a compact language model finetuned to translate natural language design cues into Blender Python scripts encoding smooth, biologically inspired geometries. We curate a domain-specific dataset of over 4,000 bioinspired and geometric design scripts spanning helical, cellular, and tubular motifs with parametric variability. The dataset is expanded and validated through an automated LLM-driven, Blender-based quality control pipeline. Bioinspired3D is then embedded in a graph-based agentic framework that integrates multimodal retrieval-augmented...
  </details>  
  Keywords: text-to-3d  
- **[Multimodal Priors-Augmented Text-Driven 3D Human-Object Interaction Generation](https://arxiv.org/abs/2602.10659v1)**  
  Authors: Yin Wang, Ziyao Zhang, Zhiying Leng, Haitian Liu, Frederick W. B. Li, Mu Li, Xiaohui Liang  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2602.10659v1.pdf)  
  <details><summary>Abstract</summary>

  We address the challenging task of text-driven 3D human-object interaction (HOI) motion generation. Existing methods primarily rely on a direct text-to-HOI mapping, which suffers from three key limitations due to the significant cross-modality gap: (Q1) sub-optimal human motion, (Q2) unnatural object motion, and (Q3) weak interaction between humans and objects. To address these challenges, we propose MP-HOI, a novel framework grounded in four core insights: (1) Multimodal Data Priors: We leverage multimodal data (text, image, pose/object) from large multimodal models as priors to guide HOI generation, which tackles Q1 and Q2 in data modeling. (2) Enhanced Object Representation: We improve existing object representations by incorporating geometric keypoints, contact features, and dynamic properties, enabling expressive object representations, which tackles Q2 in data representation. (3) Multimodal-Aware Mixture-of-Experts (MoE) Model: We propose a modality-aware MoE model for effective multimodal feature fusion paradigm, which tackles Q1 and Q2 in feature fusion. (4) Cascaded Diffusion...
  </details>  
- **[Stroke3D: Lifting 2D strokes into rigged 3D model via latent diffusion models](https://arxiv.org/abs/2602.09713v2)**  
  Authors: Ruisi Zhao, Haoren Zheng, Zongxin Yang, Hehe Fan, Yi Yang  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2602.09713v2.pdf)  
  <details><summary>Abstract</summary>

  Rigged 3D assets are fundamental to 3D deformation and animation. However, existing 3D generation methods face challenges in generating animatable geometry, while rigging techniques lack fine-grained structural control over skeleton creation. To address these limitations, we introduce Stroke3D, a novel framework that directly generates rigged meshes from user inputs: 2D drawn strokes and a descriptive text prompt. Our approach pioneers a two-stage pipeline that separates the generation into: 1) Controllable Skeleton Generation, we employ the Skeletal Graph VAE (Sk-VAE) to encode the skeleton's graph structure into a latent space, where the Skeletal Graph DiT (Sk-DiT) generates a skeletal embedding. The generation process is conditioned on both the text for semantics and the 2D strokes for explicit structural control, with the VAE's decoder reconstructing the final high-quality 3D skeleton; and 2) Enhanced Mesh Synthesis via TextuRig and SKA-DPO, where we then synthesize a textured mesh conditioned on the generated skeleton. For...
  </details>  
  Keywords: rigging  
- **[Single-Slice-to-3D Reconstruction in Medical Imaging and Natural Objects: A Comparative Benchmark with SAM 3D](https://arxiv.org/abs/2602.09407v2)**  
  Authors: Yan Luo, Advaith Ravishankar, Serena Liu, Yutong Yang, Mengyu Wang  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2602.09407v2.pdf)  
  <details><summary>Abstract</summary>

  While three-dimensional imaging is essential for clinical diagnosis, its high cost and long wait times have motivated the use of image-to-3D foundation models to infer volume from two-dimensional modalities. However, because these models are trained on natural images, their learned geometric priors struggle to transfer to inherently planar medical data. A benchmark of five state-of-the-art models (SAM3D, Hunyuan3D-2.1, Direct3D, Hi3DGen, and TripoSG) across six medical and two natural datasets revealed that voxel-based overlap remains uniformly low across all methods due to severe depth ambiguity from single-slice inputs. Despite this fundamental volumetric failure, global distance metrics indicate that SAM3D best captures topological similarity to ground-truth medical shapes, whereas alternative models are prone to oversimplification. Ultimately, these findings quantify the limits of zero-shot single-slice 3D inference, highlighting that reliable medical 3D reconstruction requires domain-specific adaptation and anatomical constraints to overcome complex medical geometries.
  </details>  
  Keywords: image-to-3d  
- **[SceneSmith: Agentic Generation of Simulation-Ready Indoor Scenes](https://arxiv.org/abs/2602.09153v2)**  
  Authors: Nicholas Pfaff, Thomas Cohn, Sergey Zakharov, Rick Cory, Russ Tedrake  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2602.09153v2.pdf)  
  <details><summary>Abstract</summary>

  Simulation has become a key tool for training and evaluating home robots at scale, yet existing environments fail to capture the diversity and physical complexity of real indoor spaces. Current scene synthesis methods produce sparsely furnished rooms that lack the dense clutter, articulated furniture, and physical properties essential for robotic manipulation. We introduce SceneSmith, a hierarchical agentic framework that generates simulation-ready indoor environments from natural language prompts. SceneSmith constructs scenes through successive stages$\unicode{x2013}$from architectural layout to furniture placement to small object population$\unicode{x2013}$each implemented as an interaction among VLM agents: designer, critic, and orchestrator. The framework tightly integrates asset generation through text-to-3D synthesis for static objects, dataset retrieval for articulated objects, and physical property estimation. SceneSmith generates 3-6x more objects than prior methods, with <2% inter-object collisions and 96% of objects remaining stable under physics simulation. In a user study with 205 participants, it achieves 92% average realism and 91% average...
  </details>  
  Keywords: text-to-3d  
- **[Recovering 3D Shapes from Ultra-Fast Motion-Blurred Images](https://arxiv.org/abs/2602.07860v1)**  
  Authors: Fei Yu, Shudan Guo, Shiqing Xin, Beibei Wang, Haisen Zhao, Wenzheng Chen  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2602.07860v1.pdf) | [![Project](https://img.shields.io/badge/-Project-blue)](https://maxmilite.github.io/rec-from-ultrafast-blur)  
  <details><summary>Abstract</summary>

  We consider the problem of 3D shape recovery from ultra-fast motion-blurred images. While 3D reconstruction from static images has been extensively studied, recovering geometry from extreme motion-blurred images remains challenging. Such scenarios frequently occur in both natural and industrial settings, such as fast-moving objects in sports (e.g., balls) or rotating machinery, where rapid motion distorts object appearance and makes traditional 3D reconstruction techniques like Multi-View Stereo (MVS) ineffective. In this paper, we propose a novel inverse rendering approach for shape recovery from ultra-fast motion-blurred images. While conventional rendering techniques typically synthesize blur by averaging across multiple frames, we identify a major computational bottleneck in the repeated computation of barycentric weights. To address this, we propose a fast barycentric coordinate solver, which significantly reduces computational overhead and achieves a speedup of up to 4.57x, enabling efficient and photorealistic simulation of high-speed motion. Crucially, our method is fully differentiable, allowing gradients to...
  </details>  
- **[From Blurry to Believable: Enhancing Low-quality Talking Heads with 3D Generative Priors](https://arxiv.org/abs/2602.06122v1)**  
  Authors: Ding-Jiun Huang, Yuanhao Wang, Shao-Ji Yuan, Albert Mosella-Montoro, Francisco Vicente Carrasco, Cheng Zhang, Fernando De la Torre  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2602.06122v1.pdf)  
  <details><summary>Abstract</summary>

  Creating high-fidelity, animatable 3D talking heads is crucial for immersive applications, yet often hindered by the prevalence of low-quality image or video sources, which yield poor 3D reconstructions. In this paper, we introduce SuperHead, a novel framework for enhancing low-resolution, animatable 3D head avatars. The core challenge lies in synthesizing high-quality geometry and textures, while ensuring both 3D and temporal consistency during animation and preserving subject identity. Despite recent progress in image, video and 3D-based super-resolution (SR), existing SR techniques are ill-equipped to handle dynamic 3D inputs. To address this, SuperHead leverages the rich priors from pre-trained 3D generative models via a novel dynamics-aware 3D inversion scheme. This process optimizes the latent representation of the generative model to produce a super-resolved 3D Gaussian Splatting (3DGS) head model, which is subsequently rigged to an underlying parametric head model (e.g., FLAME) for animation. The inversion is jointly supervised using a sparse collection...
  </details>  
- **[ShapeUP: Scalable Image-Conditioned 3D Editing](https://arxiv.org/abs/2602.05676v2)**  
  Authors: Inbar Gat, Dana Cohen-Bar, Guy Levy, Elad Richardson, Daniel Cohen-Or  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2602.05676v2.pdf)  
  <details><summary>Abstract</summary>

  Recent advancements in 3D foundation models have enabled the generation of high-fidelity assets, yet precise 3D manipulation remains a significant challenge. Existing 3D editing frameworks often face a difficult trade-off between visual controllability, geometric consistency, and scalability. Specifically, optimization-based methods are prohibitively slow, multi-view 2D propagation techniques suffer from visual drift, and training-free latent manipulation methods are inherently bound by frozen priors and cannot directly benefit from scaling. In this work, we present ShapeUP, a scalable, image-conditioned 3D editing framework that formulates editing as a supervised latent-to-latent translation within a native 3D representation. This formulation allows ShapeUP to build on a pretrained 3D foundation model, leveraging its strong generative prior while adapting it to editing through supervised training. In practice, ShapeUP is trained on triplets consisting of a source 3D shape, an edited 2D image, and the corresponding edited 3D shape, and learns a direct mapping using a 3D Diffusion...
  </details>  
- **[Monte Carlo Rendering to Diffusion Curves with Differential BEM](https://arxiv.org/abs/2602.05492v1)**  
  Authors: Ryusuke Sugimoto, Christopher Batty, Siddhartha Chaudhuri, Iliyan Georgiev, Toshiya Hachisuka, Kevin Wampler, Michal Lukáč  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2602.05492v1.pdf)  
  <details><summary>Abstract</summary>

  We present a method for generating vector graphics, in the form of diffusion curves, directly from noisy samples produced by a Monte Carlo renderer. While generating raster images from 3D geometry via Monte Carlo raytracing is commonplace, there is no corresponding practical approach for robustly and directly extracting editable vector images with shading information from 3D geometry. To fill this gap, we formulate the problem as a stochastic optimization problem over the space of geometries and colors of diffusion curve handles, and solve it with the Levenberg-Marquardt algorithm. At the core of our method is a novel differential boundary element method (BEM) framework that reconstructs colors from diffusion curve handles and computes gradients with respect to their parameters, requiring the expensive matrix factorization only once at the beginning of the optimization. Unlike triangulation-based techniques that require a clean domain decomposition, our method is robust to geometrically challenging scenarios, such as...
  </details>  
- **[Fast-SAM3D: 3Dfy Anything in Images but Faster](https://arxiv.org/abs/2602.05293v2)**  
  Authors: Weilun Feng, Mingqiang Wu, Zhiliang Chen, Chuanguang Yang, Haotong Qin, Yuqi Li, Xiaokun Liu, Guoxin Fan, Libo Huang, Yulun Zhang, Michele Magno, Yongjun Xu, Zhulin An  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2602.05293v2.pdf) | [![GitHub](https://img.shields.io/github/stars/wlfeng0509/Fast-SAM3D?style=social)](https://github.com/wlfeng0509/Fast-SAM3D)  
  <details><summary>Abstract</summary>

  SAM3D enables scalable, open-world 3D reconstruction from complex scenes, yet its deployment is hindered by prohibitive inference latency. In this work, we conduct the \textbf{first systematic investigation} into its inference dynamics, revealing that generic acceleration strategies are brittle in this context. We demonstrate that these failures stem from neglecting the pipeline's inherent multi-level \textbf{heterogeneity}: the kinematic distinctiveness between shape and layout, the intrinsic sparsity of texture refinement, and the spectral variance across geometries. To address this, we present \textbf{Fast-SAM3D}, a training-free framework that dynamically aligns computation with instantaneous generation complexity. Our approach integrates three heterogeneity-aware mechanisms: (1) \textit{Modality-Aware Step Caching} to decouple structural evolution from sensitive layout updates; (2) \textit{Joint Spatiotemporal Token Carving} to concentrate refinement on high-entropy regions; and (3) \textit{Spectral-Aware Token Aggregation} to adapt decoding resolution. Extensive experiments demonstrate that Fast-SAM3D delivers up to \textbf{2.67$\times$} end-to-end speedup with negligible fidelity loss, establishing a new Pareto frontier for efficient...
  </details>  
- **[Seeing Through Clutter: Structured 3D Scene Reconstruction via Iterative Object Removal](https://arxiv.org/abs/2602.04053v2)**  
  Authors: Rio Aguina-Kang, Kevin James Blackburn-Matzen, Thibault Groueix, Vladimir Kim, Matheus Gadelha  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2602.04053v2.pdf) | [![Project](https://img.shields.io/badge/-Project-blue)](https://rioak.github.io/seeingthroughclutter)  
  <details><summary>Abstract</summary>

  We present SeeingThroughClutter, a method for reconstructing structured 3D representations from single images by segmenting and modeling objects individually. Prior approaches rely on intermediate tasks such as semantic segmentation and depth estimation, which often underperform in complex scenes, particularly in the presence of occlusion and clutter. We address this by introducing an iterative object removal and reconstruction pipeline that decomposes complex scenes into a sequence of simpler subtasks. Using VLMs as orchestrators, foreground objects are removed one at a time via detection, segmentation, object removal, and 3D fitting. We show that removing objects allows for cleaner segmentations of subsequent objects, even in highly occluded scenes. Our method requires no task-specific training and benefits directly from ongoing advances in foundation models. We demonstrate stateof-the-art robustness on 3D-Front and ADE20K datasets. Project Page: https://rioak.github.io/seeingthroughclutter/
  </details>  
  Keywords: 3d scene reconstruction  
- **[HY3D-Bench: Generation of 3D Assets](https://arxiv.org/abs/2602.03907v1)**  
  Authors: Team Hunyuan3D, :, Bowen Zhang, Chunchao Guo, Dongyuan Guo, Haolin Liu, Hongyu Yan, Huiwen Shi, Jiaao Yu, Jiachen Xu, Jingwei Huang, Kunhong Li, Lifu Wang, Linus, Penghao Wang, Qingxiang Lin, Ruining Tang, Xianghui Yang, Yang Li, Yirui Guan, Yunfei Zhao, Yunhan Yang, Zeqiang Lai, Zhihao Liang, Zibo Zhao  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2602.03907v1.pdf)  
  <details><summary>Abstract</summary>

  While recent advances in neural representations and generative models have revolutionized 3D content creation, the field remains constrained by significant data processing bottlenecks. To address this, we introduce HY3D-Bench, an open-source ecosystem designed to establish a unified, high-quality foundation for 3D generation. Our contributions are threefold: (1) We curate a library of 250k high-fidelity 3D objects distilled from large-scale repositories, employing a rigorous pipeline to deliver training-ready artifacts, including watertight meshes and multi-view renderings; (2) We introduce structured part-level decomposition, providing the granularity essential for fine-grained perception and controllable editing; and (3) We bridge real-world distribution gaps via a scalable AIGC synthesis pipeline, contributing 125k synthetic assets to enhance diversity in long-tail categories. Validated empirically through the training of Hunyuan3D-2.1-Small, HY3D-Bench democratizes access to robust data resources, aiming to catalyze innovation across 3D perception, robotics, and digital content creation.
  </details>  
- **[PnP-U3D: Plug-and-Play 3D Framework Bridging Autoregression and Diffusion for Unified Understanding and Generation](https://arxiv.org/abs/2602.03533v1)**  
  Authors: Yongwei Chen, Tianyi Wei, Yushi Lan, Zhaoyang Lyu, Shangchen Zhou, Xudong Xu, Xingang Pan  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2602.03533v1.pdf)  
  <details><summary>Abstract</summary>

  The rapid progress of large multimodal models has inspired efforts toward unified frameworks that couple understanding and generation. While such paradigms have shown remarkable success in 2D, extending them to 3D remains largely underexplored. Existing attempts to unify 3D tasks under a single autoregressive (AR) paradigm lead to significant performance degradation due to forced signal quantization and prohibitive training cost. Our key insight is that the essential challenge lies not in enforcing a unified autoregressive paradigm, but in enabling effective information interaction between generation and understanding while minimally compromising their inherent capabilities and leveraging pretrained models to reduce training cost. Guided by this perspective, we present the first unified framework for 3D understanding and generation that combines autoregression with diffusion. Specifically, we adopt an autoregressive next-token prediction paradigm for 3D understanding, and a continuous diffusion paradigm for 3D generation. A lightweight transformer bridges the feature space of large language models...
  </details>  
- **[FastPhysGS: Accelerating Physics-based Dynamic 3DGS Simulation via Interior Completion and Adaptive Optimization](https://arxiv.org/abs/2602.01723v1)**  
  Authors: Yikun Ma, Yiqing Li, Jingwen Ye, Zhongkai Wu, Weidong Zhang, Lin Gao, Zhi Jin  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2602.01723v1.pdf)  
  <details><summary>Abstract</summary>

  Extending 3D Gaussian Splatting (3DGS) to 4D physical simulation remains challenging. Based on the Material Point Method (MPM), existing methods either rely on manual parameter tuning or distill dynamics from video diffusion models, limiting the generalization and optimization efficiency. Recent attempts using LLMs/VLMs suffer from a text/image-to-3D perceptual gap, yielding unstable physics behavior. In addition, they often ignore the surface structure of 3DGS, leading to implausible motion. We propose FastPhysGS, a fast and robust framework for physics-based dynamic 3DGS simulation:(1) Instance-aware Particle Filling (IPF) with Monte Carlo Importance Sampling (MCIS) to efficiently populate interior particles while preserving geometric fidelity; (2) Bidirectional Graph Decoupling Optimization (BGDO), an adaptive strategy that rapidly optimizes material parameters predicted from a VLM. Experiments show FastPhysGS achieves high-fidelity physical simulation in 1 minute using only 7 GB runtime memory, outperforming prior works with broad potential applications.
  </details>  
  Keywords: image-to-3d  

### December 2025
- **[PartMotionEdit: Fine-Grained Text-Driven 3D Human Motion Editing via Part-Level Modulation](https://arxiv.org/abs/2512.24200v1)**  
  Authors: Yujie Yang, Zhichao Zhang, Jiazhou Chen, Zichao Wu  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2512.24200v1.pdf)  
  <details><summary>Abstract</summary>

  Existing text-driven 3D human motion editing methods have demonstrated significant progress, but are still difficult to precisely control over detailed, part-specific motions due to their global modeling nature. In this paper, we propose PartMotionEdit, a novel fine-grained motion editing framework that operates via part-level semantic modulation. The core of PartMotionEdit is a Part-aware Motion Modulation (PMM) module, which builds upon a predefined five-part body decomposition. PMM dynamically predicts time-varying modulation weights for each body part, enabling precise and interpretable editing of local motions. To guide the training of PMM, we also introduce a part-level similarity curve supervision mechanism enhanced with dual-layer normalization. This mechanism assists PMM in learning semantically consistent and editable distributions across all body parts. Furthermore, we design a Bidirectional Motion Interaction (BMI) module. It leverages bidirectional cross-modal attention to achieve more accurate semantic alignment between textual instructions and motion semantics. Extensive quantitative and qualitative evaluations on a...
  </details>  
- **[Memorization in 3D Shape Generation: An Empirical Study](https://arxiv.org/abs/2512.23628v1)**  
  Authors: Shu Pu, Boya Zeng, Kaichen Zhou, Mengyu Wang, Zhuang Liu  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2512.23628v1.pdf) | [![GitHub](https://img.shields.io/github/stars/zlab-princeton/3d_mem?style=social)](https://github.com/zlab-princeton/3d_mem)  
  <details><summary>Abstract</summary>

  Generative models are increasingly used in 3D vision to synthesize novel shapes, yet it remains unclear whether their generation relies on memorizing training shapes. Understanding their memorization could help prevent training data leakage and improve the diversity of generated results. In this paper, we design an evaluation framework to quantify memorization in 3D generative models and study the influence of different data and modeling designs on memorization. We first apply our framework to quantify memorization in existing methods. Next, through controlled experiments with a latent vector-set (Vecset) diffusion model, we find that, on the data side, memorization depends on data modality, and increases with data diversity and finer-grained conditioning; on the modeling side, it peaks at a moderate guidance scale and can be mitigated by longer Vecsets and simple rotation augmentation. Together, our framework and analysis provide an empirical understanding of memorization in 3D generative models and suggest simple yet...
  </details>  
- **[GVSynergy-Det: Synergistic Gaussian-Voxel Representations for Multi-View 3D Object Detection](https://arxiv.org/abs/2512.23176v1)**  
  Authors: Yi Zhang, Yi Wang, Lei Yao, Lap-Pui Chau  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2512.23176v1.pdf)  
  <details><summary>Abstract</summary>

  Image-based 3D object detection aims to identify and localize objects in 3D space using only RGB images, eliminating the need for expensive depth sensors required by point cloud-based methods. Existing image-based approaches face two critical challenges: methods achieving high accuracy typically require dense 3D supervision, while those operating without such supervision struggle to extract accurate geometry from images alone. In this paper, we present GVSynergy-Det, a novel framework that enhances 3D detection through synergistic Gaussian-Voxel representation learning. Our key insight is that continuous Gaussian and discrete voxel representations capture complementary geometric information: Gaussians excel at modeling fine-grained surface details while voxels provide structured spatial context. We introduce a dual-representation architecture that: 1) adapts generalizable Gaussian Splatting to extract complementary geometric features for detection tasks, and 2) develops a cross-representation enhancement mechanism that enriches voxel features with geometric details from Gaussian fields. Unlike previous methods that either rely on time-consuming per-scene...
  </details>  
- **[SwinTF3D: A Lightweight Multimodal Fusion Approach for Text-Guided 3D Medical Image Segmentation](https://arxiv.org/abs/2512.22878v1)**  
  Authors: Hasan Faraz Khan, Noor Fatima, Muzammil Behzad  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2512.22878v1.pdf)  
  <details><summary>Abstract</summary>

  The recent integration of artificial intelligence into medical imaging has driven remarkable advances in automated organ segmentation. However, most existing 3D segmentation frameworks rely exclusively on visual learning from large annotated datasets restricting their adaptability to new domains and clinical tasks. The lack of semantic understanding in these models makes them ineffective in addressing flexible, user-defined segmentation objectives. To overcome these limitations, we propose SwinTF3D, a lightweight multimodal fusion approach that unifies visual and linguistic representations for text-guided 3D medical image segmentation. The model employs a transformer-based visual encoder to extract volumetric features and integrates them with a compact text encoder via an efficient fusion mechanism. This design allows the system to understand natural-language prompts and correctly align semantic cues with their corresponding spatial structures in medical volumes, while producing accurate, context-aware segmentation results with low computational overhead. Extensive experiments on the BTCV dataset demonstrate that SwinTF3D achieves competitive Dice...
  </details>  
- **[Pose-Guided Residual Refinement for Interpretable Text-to-Motion Generation and Editing](https://arxiv.org/abs/2512.22464v1)**  
  Authors: Sukhyun Jeong, Yong-Hoon Choi  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2512.22464v1.pdf)  
  <details><summary>Abstract</summary>

  Text-based 3D motion generation aims to automatically synthesize diverse motions from natural-language descriptions to extend user creativity, whereas motion editing modifies an existing motion sequence in response to text while preserving its overall structure. Pose-code-based frameworks such as CoMo map quantifiable pose attributes into discrete pose codes that support interpretable motion control, but their frame-wise representation struggles to capture subtle temporal dynamics and high-frequency details, often degrading reconstruction fidelity and local controllability. To address this limitation, we introduce pose-guided residual refinement for motion (PGR$^2$M), a hybrid representation that augments interpretable pose codes with residual codes learned via residual vector quantization (RVQ). A pose-guided RVQ tokenizer decomposes motion into pose latents that encode coarse global structure and residual latents that model fine-grained temporal variations. Residual dropout further discourages over-reliance on residuals, preserving the semantic alignment and editability of the pose codes. On top of this tokenizer, a base Transformer autoregressively predicts...
  </details>  
- **[SAM 3D for 3D Object Reconstruction from Remote Sensing Images](https://arxiv.org/abs/2512.22452v1)**  
  Authors: Junsheng Yao, Lichao Mou, Qingyu Li  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2512.22452v1.pdf)  
  <details><summary>Abstract</summary>

  Monocular 3D building reconstruction from remote sensing imagery is essential for scalable urban modeling, yet existing methods often require task-specific architectures and intensive supervision. This paper presents the first systematic evaluation of SAM 3D, a general-purpose image-to-3D foundation model, for monocular remote sensing building reconstruction. We benchmark SAM 3D against TRELLIS on samples from the NYC Urban Dataset, employing Frechet Inception Distance (FID) and CLIP-based Maximum Mean Discrepancy (CMMD) as evaluation metrics. Experimental results demonstrate that SAM 3D produces more coherent roof geometry and sharper boundaries compared to TRELLIS. We further extend SAM 3D to urban scene reconstruction through a segment-reconstruct-compose pipeline, demonstrating its potential for urban scene modeling. We also analyze practical limitations and discuss future research directions. These findings provide practical guidance for deploying foundation models in urban 3D reconstruction and motivate future integration of scene-level structural priors.
  </details>  
  Keywords: image-to-3d  
- **[Learning Dynamic Scene Reconstruction with Sinusoidal Geometric Priors](https://arxiv.org/abs/2512.22295v1)**  
  Authors: Tian Guo, Hui Yuan, Philip Xu, David Elizondo  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2512.22295v1.pdf)  
  <details><summary>Abstract</summary>

  We propose SirenPose, a novel loss function that combines the periodic activation properties of sinusoidal representation networks with geometric priors derived from keypoint structures to improve the accuracy of dynamic 3D scene reconstruction. Existing approaches often struggle to maintain motion modeling accuracy and spatiotemporal consistency in fast moving and multi target scenes. By introducing physics inspired constraint mechanisms, SirenPose enforces coherent keypoint predictions across both spatial and temporal dimensions. We further expand the training dataset to 600,000 annotated instances to support robust learning. Experimental results demonstrate that models trained with SirenPose achieve significant improvements in spatiotemporal consistency metrics compared to prior methods, showing superior performance in handling rapid motion and complex scene changes.
  </details>  
  Keywords: 3d scene reconstruction  
- **[A Three-Level Alignment Framework for Large-Scale 3D Retrieval and Controlled 4D Generation](https://arxiv.org/abs/2512.22294v1)**  
  Authors: Philip Xu  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2512.22294v1.pdf)  
  <details><summary>Abstract</summary>

  We introduce Uni4D, a unified framework for large scale open vocabulary 3D retrieval and controlled 4D generation based on structured three level alignment across text, 3D models, and image modalities. Built upon the Align3D 130 dataset, Uni4D employs a 3D text multi head attention and search model to optimize text to 3D retrieval through improved semantic alignment. The framework further strengthens cross modal alignment through three components: precise text to 3D retrieval, multi view 3D to image alignment, and image to text alignment for generating temporally consistent 4D assets. Experimental results demonstrate that Uni4D achieves high quality 3D retrieval and controllable 4D generation, advancing dynamic multimodal understanding and practical applications.
  </details>  
- **[SegMo: Segment-aligned Text to 3D Human Motion Generation](https://arxiv.org/abs/2512.21237v1)**  
  Authors: Bowen Dang, Lin Wu, Xiaohang Yang, Zheng Yuan, Zhixiang Chen  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2512.21237v1.pdf)  
  <details><summary>Abstract</summary>

  Generating 3D human motions from textual descriptions is an important research problem with broad applications in video games, virtual reality, and augmented reality. Recent methods align the textual description with human motion at the sequence level, neglecting the internal semantic structure of modalities. However, both motion descriptions and motion sequences can be naturally decomposed into smaller and semantically coherent segments, which can serve as atomic alignment units to achieve finer-grained correspondence. Motivated by this, we propose SegMo, a novel Segment-aligned text-conditioned human Motion generation framework to achieve fine-grained text-motion alignment. Our framework consists of three modules: (1) Text Segment Extraction, which decomposes complex textual descriptions into temporally ordered phrases, each representing a simple atomic action; (2) Motion Segment Extraction, which partitions complete motion sequences into corresponding motion segments; and (3) Fine-grained Text-Motion Alignment, which aligns text and motion segments with contrastive learning. Extensive experiments demonstrate that SegMo improves the strong...
  </details>  
- **[UltraShape 1.0: High-Fidelity 3D Shape Generation via Scalable Geometric Refinement](https://arxiv.org/abs/2512.21185v2)**  
  Authors: Tanghui Jia, Dongyu Yan, Dehao Hao, Yang Li, Kaiyi Zhang, Xianyi He, Lanjiong Li, Yuhan Wang, Jinnan Chen, Lutao Jiang, Qishen Yin, Long Quan, Ying-Cong Chen, Li Yuan  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2512.21185v2.pdf)  
  <details><summary>Abstract</summary>

  In this report, we introduce UltraShape 1.0, a scalable 3D diffusion framework for high-fidelity 3D geometry generation. The proposed approach adopts a two-stage generation pipeline: a coarse global structure is first synthesized and then refined to produce detailed, high-quality geometry. To support reliable 3D generation, we develop a comprehensive data processing pipeline that includes a novel watertight processing method and high-quality data filtering. This pipeline improves the geometric quality of publicly available 3D datasets by removing low-quality samples, filling holes, and thickening thin structures, while preserving fine-grained geometric details. To enable fine-grained geometry refinement, we decouple spatial localization from geometric detail synthesis in the diffusion process. We achieve this by performing voxel-based refinement at fixed spatial locations, where voxel queries derived from coarse geometry provide explicit positional anchors encoded via RoPE, allowing the diffusion model to focus on synthesizing local geometric details within a reduced, structured solution space. Our model...
  </details>  
- **[Enhancing annotations for 5D apple pose estimation through 3D Gaussian Splatting (3DGS)](https://arxiv.org/abs/2512.20148v1)**  
  Authors: Robert van de Ven, Trim Bresilla, Bram Nelissen, Ard Nieuwenhuizen, Eldert J. van Henten, Gert Kootstra  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2512.20148v1.pdf)  
  <details><summary>Abstract</summary>

  Automating tasks in orchards is challenging because of the large amount of variation in the environment and occlusions. One of the challenges is apple pose estimation, where key points, such as the calyx, are often occluded. Recently developed pose estimation methods no longer rely on these key points, but still require them for annotations, making annotating challenging and time-consuming. Due to the abovementioned occlusions, there can be conflicting and missing annotations of the same fruit between different images. Novel 3D reconstruction methods can be used to simplify annotating and enlarge datasets. We propose a novel pipeline consisting of 3D Gaussian Splatting to reconstruct an orchard scene, simplified annotations, automated projection of the annotations to images, and the training and evaluation of a pose estimation method. Using our pipeline, 105 manual annotations were required to obtain 28,191 training labels, a reduction of 99.6%. Experimental results indicated that training with labels of...
  </details>  
- **[Scaling Point-based Differentiable Rendering for Large-scale Reconstruction](https://arxiv.org/abs/2512.20017v1)**  
  Authors: Hexu Zhao, Xiaoteng Liu, Xiwen Min, Jianhao Huang, Youming Deng, Yanfei Li, Ang Li, Jinyang Li, Aurojit Panda  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2512.20017v1.pdf)  
  <details><summary>Abstract</summary>

  Point-based Differentiable Rendering (PBDR) enables high-fidelity 3D scene reconstruction, but scaling PBDR to high-resolution and large scenes requires efficient distributed training systems. Existing systems are tightly coupled to a specific PBDR method. And they suffer from severe communication overhead due to poor data locality. In this paper, we present Gaian, a general distributed training system for PBDR. Gaian provides a unified API expressive enough to support existing PBDR methods, while exposing rich data-access information, which Gaian leverages to optimize locality and reduce communication. We evaluated Gaian by implementing 4 PBDR algorithms. Our implementations achieve high performance and resource efficiency: across six datasets and up to 128 GPUs, it reduces communication by up to 91% and improves training throughput by 1.50x-3.71x.
  </details>  
  Keywords: 3d scene reconstruction  
- **[Symmetry Matters: Auditing and Symmetrizing 3D Generative Models](https://arxiv.org/abs/2512.18953v2)**  
  Authors: Nicolas Caytuiro, Ivan Sipiran  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2512.18953v2.pdf)  
  <details><summary>Abstract</summary>

  Symmetry is a strong prior present in many object categories, yet standard benchmarks for 3D generative models rarely report whether this prior is preserved. We study symmetry preservation in unconditional point cloud generation. We first audit the symmetry of generated shapes by several 3D generative models and compute a normalized symmetry score based on the Chamfer Distance (CD). We show that although current 3D generative models achieve competitive results under standard evaluation, they reveal a persistent symmetry gap when a symmetry-aware evaluation protocol is applied. To test whether this gap is merely inherited from the training data, we evaluate these models over a mirrored-objects dataset derived from ShapeNet and analyze symmetry dynamics during training. Mechanistic interpretability techniques were employed at the sampling and latent levels to further show that reflection symmetry is not reliably encoded in the learned generative process. Finally, to address this gap, we propose a data-centric symmetry-aware...
  </details>  
- **[Local Patches Meet Global Context: Scalable 3D Diffusion Priors for Computed Tomography Reconstruction](https://arxiv.org/abs/2512.18161v1)**  
  Authors: Taewon Yang, Jason Hu, Jeffrey A. Fessler, Liyue Shen  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2512.18161v1.pdf)  
  <details><summary>Abstract</summary>

  Diffusion models learn strong image priors that can be leveraged to solve inverse problems like medical image reconstruction. However, for real-world applications such as 3D Computed Tomography (CT) imaging, directly training diffusion models on 3D data presents significant challenges due to the high computational demands of extensive GPU resources and large-scale datasets. Existing works mostly reuse 2D diffusion priors to address 3D inverse problems, but fail to fully realize and leverage the generative capacity of diffusion models for high-dimensional data. In this study, we propose a novel 3D patch-based diffusion model that can learn a fully 3D diffusion prior from limited data, enabling scalable generation of high-resolution 3D images. Our core idea is to learn the prior of 3D patches to achieve scalable efficiency, while coupling local and global information to guarantee high-quality 3D image generation, by modeling the joint distribution of position-aware 3D local patches and downsampled 3D volume...
  </details>  
- **[Towards Autonomous Navigation in Endovascular Interventions](https://arxiv.org/abs/2512.18081v2)**  
  Authors: Tudor Jianu  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2512.18081v2.pdf)  
  <details><summary>Abstract</summary>

  Cardiovascular diseases remain the leading cause of global mortality, with minimally invasive treatment options offered through endovascular interventions. However, the precision and adaptability of current robotic systems for endovascular navigation are limited by heuristic control, low autonomy, and the absence of haptic feedback. This thesis presents an integrated AI-driven framework for autonomous guidewire navigation in complex vascular environments, addressing key challenges in data availability, simulation fidelity, and navigational accuracy. A high-fidelity, real-time simulation platform, CathSim, is introduced for reinforcement learning based catheter navigation, featuring anatomically accurate vascular models and contact dynamics. Building on CathSim, the Expert Navigation Network is developed, a policy that fuses visual, kinematic, and force feedback for autonomous tool control. To mitigate data scarcity, the open-source, bi-planar fluoroscopic dataset Guide3D is proposed, comprising more than 8,700 annotated images for 3D guidewire reconstruction. Finally, SplineFormer, a transformer-based model, is introduced to directly predict guidewire geometry as continuous B-spline...
  </details>  
- **[3D-RE-GEN: 3D Reconstruction of Indoor Scenes with a Generative Framework](https://arxiv.org/abs/2512.17459v1)**  
  Authors: Tobias Sautter, Jan-Niklas Dihlmann, Hendrik P. A. Lensch  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2512.17459v1.pdf)  
  <details><summary>Abstract</summary>

  Recent advances in 3D scene generation produce visually appealing output, but current representations hinder artists' workflows that require modifiable 3D textured mesh scenes for visual effects and game development. Despite significant advances, current textured mesh scene reconstruction methods are far from artist ready, suffering from incorrect object decomposition, inaccurate spatial relationships, and missing backgrounds. We present 3D-RE-GEN, a compositional framework that reconstructs a single image into textured 3D objects and a background. We show that combining state of the art models from specific domains achieves state of the art scene reconstruction performance, addressing artists' requirements. Our reconstruction pipeline integrates models for asset detection, reconstruction, and placement, pushing certain models beyond their originally intended domains. Obtaining occluded objects is treated as an image editing task with generative models to infer and reconstruct with scene level reasoning under consistent lighting and geometry. Unlike current methods, 3D-RE-GEN generates a comprehensive background that spatially...
  </details>  
  Keywords: 3d scene reconstruction  
- **[SynergyWarpNet: Attention-Guided Cooperative Warping for Neural Portrait Animation](https://arxiv.org/abs/2512.17331v1)**  
  Authors: Shihang Li, Zhiqiang Gong, Minming Ye, Yue Gao, Wen Yao  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2512.17331v1.pdf)  
  <details><summary>Abstract</summary>

  Recent advances in neural portrait animation have demonstrated remarked potential for applications in virtual avatars, telepresence, and digital content creation. However, traditional explicit warping approaches often struggle with accurate motion transfer or recovering missing regions, while recent attention-based warping methods, though effective, frequently suffer from high complexity and weak geometric grounding. To address these issues, we propose SynergyWarpNet, an attention-guided cooperative warping framework designed for high-fidelity talking head synthesis. Given a source portrait, a driving image, and a set of reference images, our model progressively refines the animation in three stages. First, an explicit warping module performs coarse spatial alignment between the source and driving image using 3D dense optical flow. Next, a reference-augmented correction module leverages cross-attention across 3D keypoints and texture features from multiple reference images to semantically complete occluded or distorted regions. Finally, a confidence-guided fusion module integrates the warped outputs with spatially-adaptive fusing, using a learned...
  </details>  
- **[SNOW: Spatio-Temporal Scene Understanding with World Knowledge for Open-World Embodied Reasoning](https://arxiv.org/abs/2512.16461v1)**  
  Authors: Tin Stribor Sohn, Maximilian Dillitzer, Jason J. Corso, Eric Sax  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2512.16461v1.pdf)  
  <details><summary>Abstract</summary>

  Autonomous robotic systems require spatio-temporal understanding of dynamic environments to ensure reliable navigation and interaction. While Vision-Language Models (VLMs) provide open-world semantic priors, they lack grounding in 3D geometry and temporal dynamics. Conversely, geometric perception captures structure and motion but remains semantically sparse. We propose SNOW (Scene Understanding with Open-World Knowledge), a training-free and backbone-agnostic framework for unified 4D scene understanding that integrates VLM-derived semantics with point cloud geometry and temporal consistency. SNOW processes synchronized RGB images and 3D point clouds, using HDBSCAN clustering to generate object-level proposals that guide SAM2-based segmentation. Each segmented region is encoded through our proposed Spatio-Temporal Tokenized Patch Encoding (STEP), producing multimodal tokens that capture localized semantic, geometric, and temporal attributes. These tokens are incrementally integrated into a 4D Scene Graph (4DSG), which serves as 4D prior for downstream reasoning. A lightweight SLAM backend anchors all STEP tokens spatially in the environment, providing the global...
  </details>  
- **[Single-View Shape Completion for Robotic Grasping in Clutter](https://arxiv.org/abs/2512.16449v1)**  
  Authors: Abhishek Kashyap, Yuxuan Yang, Henrik Andreasson, Todor Stoyanov  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2512.16449v1.pdf) | [![Project](https://img.shields.io/badge/-Project-blue)](https://amm.aass.oru.se/shape-completion-grasping)  
  <details><summary>Abstract</summary>

  In vision-based robot manipulation, a single camera view can only capture one side of objects of interest, with additional occlusions in cluttered scenes further restricting visibility. As a result, the observed geometry is incomplete, and grasp estimation algorithms perform suboptimally. To address this limitation, we leverage diffusion models to perform category-level 3D shape completion from partial depth observations obtained from a single view, reconstructing complete object geometries to provide richer context for grasp planning. Our method focuses on common household items with diverse geometries, generating full 3D shapes that serve as input to downstream grasp inference networks. Unlike prior work, which primarily considers isolated objects or minimal clutter, we evaluate shape completion and grasping in realistic clutter scenarios with household objects. In preliminary evaluations on a cluttered scene, our approach consistently results in better grasp success rates than a naive baseline without shape completion by 23% and over a recent...
  </details>  
- **[Hierarchical Neural Surfaces for 3D Mesh Compression](https://arxiv.org/abs/2512.15985v1)**  
  Authors: Sai Karthikey Pentapati, Gregoire Phillips, Alan Bovik  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2512.15985v1.pdf)  
  <details><summary>Abstract</summary>

  Implicit Neural Representations (INRs) have been demonstrated to achieve state-of-the-art compression of a broad range of modalities such as images, videos, 3D surfaces, and audio. Most studies have focused on building neural counterparts of traditional implicit representations of 3D geometries, such as signed distance functions. However, the triangle mesh-based representation of geometry remains the most widely used representation in the industry, while building INRs capable of generating them has been sparsely studied. In this paper, we present a method for building compact INRs of zero-genus 3D manifolds. Our method relies on creating a spherical parameterization of a given 3D mesh - mapping the surface of a mesh to that of a unit sphere - then constructing an INR that encodes the displacement vector field defined continuously on its surface that regenerates the original shape. The compactness of our representation can be attributed to its hierarchical structure, wherein it first recovers...
  </details>  
- **[Native and Compact Structured Latents for 3D Generation](https://arxiv.org/abs/2512.14692v1)**  
  Authors: Jianfeng Xiang, Xiaoxue Chen, Sicheng Xu, Ruicheng Wang, Zelong Lv, Yu Deng, Hongyuan Zhu, Yue Dong, Hao Zhao, Nicholas Jing Yuan, Jiaolong Yang  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2512.14692v1.pdf)  
  <details><summary>Abstract</summary>

  Recent advancements in 3D generative modeling have significantly improved the generation realism, yet the field is still hampered by existing representations, which struggle to capture assets with complex topologies and detailed appearance. This paper present an approach for learning a structured latent representation from native 3D data to address this challenge. At its core is a new sparse voxel structure called O-Voxel, an omni-voxel representation that encodes both geometry and appearance. O-Voxel can robustly model arbitrary topology, including open, non-manifold, and fully-enclosed surfaces, while capturing comprehensive surface attributes beyond texture color, such as physically-based rendering parameters. Based on O-Voxel, we design a Sparse Compression VAE which provides a high spatial compression rate and a compact latent space. We train large-scale flow-matching models comprising 4B parameters for 3D generation using diverse public 3D asset datasets. Despite their scale, inference remains highly efficient. Meanwhile, the geometry and material quality of our generated...
  </details>  
- **[SS4D: Native 4D Generative Model via Structured Spacetime Latents](https://arxiv.org/abs/2512.14284v1)**  
  Authors: Zhibing Li, Mengchen Zhang, Tong Wu, Jing Tan, Jiaqi Wang, Dahua Lin  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2512.14284v1.pdf)  
  <details><summary>Abstract</summary>

  We present SS4D, a native 4D generative model that synthesizes dynamic 3D objects directly from monocular video. Unlike prior approaches that construct 4D representations by optimizing over 3D or video generative models, we train a generator directly on 4D data, achieving high fidelity, temporal coherence, and structural consistency. At the core of our method is a compressed set of structured spacetime latents. Specifically, (1) To address the scarcity of 4D training data, we build on a pre-trained single-image-to-3D model, preserving strong spatial consistency. (2) Temporal consistency is enforced by introducing dedicated temporal layers that reason across frames. (3) To support efficient training and inference over long video sequences, we compress the latent sequence along the temporal axis using factorized 4D convolutions and temporal downsampling blocks. In addition, we employ a carefully designed training strategy to enhance robustness against occlusion
  </details>  
  Keywords: image-to-3d  
- **[Feedforward 3D Editing via Text-Steerable Image-to-3D](https://arxiv.org/abs/2512.13678v1)**  
  Authors: Ziqi Ma, Hongqiao Chen, Yisong Yue, Georgia Gkioxari  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2512.13678v1.pdf) | [![Project](https://img.shields.io/badge/-Project-blue)](https://glab-caltech.github.io/steer3d)  
  <details><summary>Abstract</summary>

  Recent progress in image-to-3D has opened up immense possibilities for design, AR/VR, and robotics. However, to use AI-generated 3D assets in real applications, a critical requirement is the capability to edit them easily. We present a feedforward method, Steer3D, to add text steerability to image-to-3D models, which enables editing of generated 3D assets with language. Our approach is inspired by ControlNet, which we adapt to image-to-3D generation to enable text steering directly in a forward pass. We build a scalable data engine for automatic data generation, and develop a two-stage training recipe based on flow-matching training and Direct Preference Optimization (DPO). Compared to competing methods, Steer3D more faithfully follows the language instruction and maintains better consistency with the original 3D asset, while being 2.4x to 28.5x faster. Steer3D demonstrates that it is possible to add a new modality (text) to steer the generation of pretrained image-to-3D generative models with 100k...
  </details>  
  Keywords: image-to-3d  
- **[Cross-Level Sensor Fusion with Object Lists via Transformer for 3D Object Detection](https://arxiv.org/abs/2512.12884v2)**  
  Authors: Xiangzhong Liu, Jiajie Zhang, Hao Shen  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2512.12884v2.pdf)  
  <details><summary>Abstract</summary>

  In automotive sensor fusion systems, smart sensors and Vehicle-to-Everything (V2X) modules are commonly utilized. Sensor data from these systems are typically available only as processed object lists rather than raw sensor data from traditional sensors. Instead of processing other raw data separately and then fusing them at the object level, we propose an end-to-end cross-level fusion concept with Transformer, which integrates highly abstract object list information with raw camera images for 3D object detection. Object lists are fed into a Transformer as denoising queries and propagated together with learnable queries through the latter feature aggregation process. Additionally, a deformable Gaussian mask, derived from the positional and size dimensional priors from the object lists, is explicitly integrated into the Transformer decoder. This directs attention toward the target area of interest and accelerates model training convergence. Furthermore, as there is no public dataset containing object lists as a standalone modality, we propose...
  </details>  
- **[Quantum Implicit Neural Representations for 3D Scene Reconstruction and Novel View Synthesis](https://arxiv.org/abs/2512.12683v1)**  
  Authors: Yeray Cordero, Paula García-Molina, Fernando Vilariño  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2512.12683v1.pdf)  
  <details><summary>Abstract</summary>

  Implicit neural representations (INRs) have become a powerful paradigm for continuous signal modeling and 3D scene reconstruction, yet classical networks suffer from a well-known spectral bias that limits their ability to capture high-frequency details. Quantum Implicit Representation Networks (QIREN) mitigate this limitation by employing parameterized quantum circuits with inherent Fourier structures, enabling compact and expressive frequency modeling beyond classical MLPs. In this paper, we present Quantum Neural Radiance Fields (Q-NeRF), the first hybrid quantum-classical framework for neural radiance field rendering. Q-NeRF integrates QIREN modules into the Nerfacto backbone, preserving its efficient sampling, pose refinement, and volumetric rendering strategies while replacing selected density and radiance prediction components with quantum-enhanced counterparts. We systematically evaluate three hybrid configurations on standard multi-view indoor datasets, comparing them to classical baselines using PSNR, SSIM, and LPIPS metrics. Results show that hybrid quantum-classical models achieve competitive reconstruction quality under limited computational resources, with quantum modules particularly effective...
  </details>  
  Keywords: 3d scene reconstruction, novel view synthesis  
- **[Animus3D: Text-driven 3D Animation via Motion Score Distillation](https://arxiv.org/abs/2512.12534v1)**  
  Authors: Qi Sun, Can Wang, Jiaxiang Shang, Wensen Feng, Jing Liao  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2512.12534v1.pdf) | [![Project](https://img.shields.io/badge/-Project-blue)](https://qiisun.github.io/animus3d_page)  
  <details><summary>Abstract</summary>

  We present Animus3D, a text-driven 3D animation framework that generates motion field given a static 3D asset and text prompt. Previous methods mostly leverage the vanilla Score Distillation Sampling (SDS) objective to distill motion from pretrained text-to-video diffusion, leading to animations with minimal movement or noticeable jitter. To address this, our approach introduces a novel SDS alternative, Motion Score Distillation (MSD). Specifically, we introduce a LoRA-enhanced video diffusion model that defines a static source distribution rather than pure noise as in SDS, while another inversion-based noise estimation technique ensures appearance preservation when guiding motion. To further improve motion fidelity, we incorporate explicit temporal and spatial regularization terms that mitigate geometric distortions across time and space. Additionally, we propose a motion refinement module to upscale the temporal resolution and enhance fine-grained details, overcoming the fixed-resolution constraints of the underlying video model. Extensive experiments demonstrate that Animus3D successfully animates static 3D assets...
  </details>  
- **[Particulate: Feed-Forward 3D Object Articulation](https://arxiv.org/abs/2512.11798v2)**  
  Authors: Ruining Li, Yuxin Yao, Chuanxia Zheng, Christian Rupprecht, Joan Lasenby, Shangzhe Wu, Andrea Vedaldi  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2512.11798v2.pdf)  
  <details><summary>Abstract</summary>

  We introduce Particulate, a feed-forward model that, given a 3D mesh of an object, infers its articulations, including its 3D parts, their kinematic structure, and the motion constraints. The model is based on a transformer network, the Part Articulation Transformer, which predicts all these parameters for all joints. We train the network end-to-end on a diverse collection of articulated 3D assets from public datasets. During inference, Particulate maps the output of the network back to the input mesh, yielding a fully articulated 3D model in seconds, much faster than prior approaches that require per-object optimization. Particulate also works on AI-generated 3D assets, enabling the generation of articulated 3D objects from a single (real or synthetic) image when combined with an off-the-shelf image-to-3D model. We further introduce a new challenging benchmark for 3D articulation estimation curated from high-quality public 3D assets, and redesign the evaluation protocol to be more consistent with...
  </details>  
  Keywords: image-to-3d  
- **[3DTeethSAM: Taming SAM2 for 3D Teeth Segmentation](https://arxiv.org/abs/2512.11557v1)**  
  Authors: Zhiguo Lu, Jianwen Lou, Mingjun Ma, Hairong Jin, Youyi Zheng, Kun Zhou  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2512.11557v1.pdf)  
  <details><summary>Abstract</summary>

  3D teeth segmentation, involving the localization of tooth instances and their semantic categorization in 3D dental models, is a critical yet challenging task in digital dentistry due to the complexity of real-world dentition. In this paper, we propose 3DTeethSAM, an adaptation of the Segment Anything Model 2 (SAM2) for 3D teeth segmentation. SAM2 is a pretrained foundation model for image and video segmentation, demonstrating a strong backbone in various downstream scenarios. To adapt SAM2 for 3D teeth data, we render images of 3D teeth models from predefined views, apply SAM2 for 2D segmentation, and reconstruct 3D results using 2D-3D projections. Since SAM2's performance depends on input prompts and its initial outputs often have deficiencies, and given its class-agnostic nature, we introduce three light-weight learnable modules: (1) a prompt embedding generator to derive prompt embeddings from image embeddings for accurate mask decoding, (2) a mask refiner to enhance SAM2's initial segmentation...
  </details>  
- **[CADKnitter: Compositional CAD Generation from Text and Geometry Guidance](https://arxiv.org/abs/2512.11199v1)**  
  Authors: Tri Le, Khang Nguyen, Baoru Huang, Tung D. Ta, Anh Nguyen  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2512.11199v1.pdf)  
  <details><summary>Abstract</summary>

  Crafting computer-aided design (CAD) models has long been a painstaking and time-intensive task, demanding both precision and expertise from designers. With the emergence of 3D generation, this task has undergone a transformative impact, shifting not only from visual fidelity to functional utility but also enabling editable CAD designs. Prior works have achieved early success in single-part CAD generation, which is not well-suited for real-world applications, as multiple parts need to be assembled under semantic and geometric constraints. In this paper, we propose CADKnitter, a compositional CAD generation framework with a geometry-guided diffusion sampling strategy. CADKnitter is able to generate a complementary CAD part that follows both the geometric constraints of the given CAD model and the semantic constraints of the desired design text prompt. We also curate a dataset, so-called KnitCAD, containing over 310,000 samples of CAD models, along with textual prompts and assembly metadata that provide semantic and geometric...
  </details>  
- **[Are We Ready for RL in Text-to-3D Generation? A Progressive Investigation](https://arxiv.org/abs/2512.10949v1)**  
  Authors: Yiwen Tang, Zoey Guo, Kaixin Zhu, Ray Zhang, Qizhi Chen, Dongzhi Jiang, Junli Liu, Bohan Zeng, Haoming Song, Delin Qu, Tianyi Bai, Dan Xu, Wentao Zhang, Bin Zhao  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2512.10949v1.pdf) | [![GitHub](https://img.shields.io/github/stars/Ivan-Tang-3D/3DGen-R1?style=social)](https://github.com/Ivan-Tang-3D/3DGen-R1)  
  <details><summary>Abstract</summary>

  Reinforcement learning (RL), earlier proven to be effective in large language and multi-modal models, has been successfully extended to enhance 2D image generation recently. However, applying RL to 3D generation remains largely unexplored due to the higher spatial complexity of 3D objects, which require globally consistent geometry and fine-grained local textures. This makes 3D generation significantly sensitive to reward designs and RL algorithms. To address these challenges, we conduct the first systematic study of RL for text-to-3D autoregressive generation across several dimensions. (1) Reward designs: We evaluate reward dimensions and model choices, showing that alignment with human preference is crucial, and that general multi-modal models provide robust signal for 3D attributes. (2) RL algorithms: We study GRPO variants, highlighting the effectiveness of token-level optimization, and further investigate the scaling of training data and iterations. (3) Text-to-3D Benchmarks: Since existing benchmarks fail to measure implicit reasoning abilities in 3D generation models,...
  </details>  
  Keywords: text-to-3d  
- **[SWiT-4D: Sliding-Window Transformer for Lossless and Parameter-Free Temporal 4D Generation](https://arxiv.org/abs/2512.10860v1)**  
  Authors: Kehong Gong, Zhengyu Wen, Mingxi Xu, Weixia He, Qi Wang, Ning Zhang, Zhengyu Li, Chenbin Li, Dongze Lian, Wei Zhao, Xiaoyu He, Mingyuan Zhang  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2512.10860v1.pdf) | [![Project](https://img.shields.io/badge/-Project-blue)](https://animotionlab.github.io/SWIT4D)  
  <details><summary>Abstract</summary>

  Despite significant progress in 4D content generation, the conversion of monocular videos into high-quality animated 3D assets with explicit 4D meshes remains considerably challenging. The scarcity of large-scale, naturally captured 4D mesh datasets further limits the ability to train generalizable video-to-4D models from scratch in a purely data-driven manner. Meanwhile, advances in image-to-3D generation, supported by extensive datasets, offer powerful prior models that can be leveraged. To better utilize these priors while minimizing reliance on 4D supervision, we introduce SWiT-4D, a Sliding-Window Transformer for lossless, parameter-free temporal 4D mesh generation. SWiT-4D integrates seamlessly with any Diffusion Transformer (DiT)-based image-to-3D generator, adding spatial-temporal modeling across video frames while preserving the original single-image forward process, enabling 4D mesh reconstruction from videos of arbitrary length. To recover global translation, we further introduce an optimization-based trajectory module tailored for static-camera monocular videos. SWiT-4D demonstrates strong data efficiency: with only a single short (<10s) video...
  </details>  
  Keywords: 3d generator, image-to-3d  
- **[UniPart: Part-Level 3D Generation with Unified 3D Geom-Seg Latents](https://arxiv.org/abs/2512.09435v3)**  
  Authors: Xufan He, Yushuang Wu, Xiaoyang Guo, Chongjie Ye, Jiaqing Zhou, Tianlei Hu, Xiaoguang Han, Dong Du  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2512.09435v3.pdf)  
  <details><summary>Abstract</summary>

  Part-level 3D generation is essential for applications requiring decomposable and structured 3D synthesis. However, existing methods either rely on implicit part segmentation with limited granularity control or depend on strong external segmenters trained on large annotated datasets. In this work, we observe that part awareness emerges naturally during whole-object geometry learning and propose Geom-Seg VecSet, a unified geometry-segmentation latent representation that jointly encodes object geometry and part-level structure. Building on this representation, we introduce UniPart, a two-stage latent diffusion framework for image-guided part-level 3D generation. The first stage performs joint geometry generation and latent part segmentation, while the second stage conditions part-level diffusion on both whole-object and part-specific latents. A dual-space generation scheme further enhances geometric fidelity by predicting part latents in both global and canonical spaces. Extensive experiments demonstrate that UniPart achieves superior segmentation controllability and part-level geometric quality compared with existing approaches.
  </details>  
- **[ASSIST-3D: Adapted Scene Synthesis for Class-Agnostic 3D Instance Segmentation](https://arxiv.org/abs/2512.09364v1)**  
  Authors: Shengchao Zhou, Jiehong Lin, Jiahui Liu, Shizhen Zhao, Chirui Chang, Xiaojuan Qi  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2512.09364v1.pdf)  
  <details><summary>Abstract</summary>

  Class-agnostic 3D instance segmentation tackles the challenging task of segmenting all object instances, including previously unseen ones, without semantic class reliance. Current methods struggle with generalization due to the scarce annotated 3D scene data or noisy 2D segmentations. While synthetic data generation offers a promising solution, existing 3D scene synthesis methods fail to simultaneously satisfy geometry diversity, context complexity, and layout reasonability, each essential for this task. To address these needs, we propose an Adapted 3D Scene Synthesis pipeline for class-agnostic 3D Instance SegmenTation, termed as ASSIST-3D, to synthesize proper data for model generalization enhancement. Specifically, ASSIST-3D features three key innovations, including 1) Heterogeneous Object Selection from extensive 3D CAD asset collections, incorporating randomness in object sampling to maximize geometric and contextual diversity; 2) Scene Layout Generation through LLM-guided spatial reasoning combined with depth-first search for reasonable object placements; and 3) Realistic Point Cloud Construction via multi-view RGB-D image rendering...
  </details>  
- **[Prompt-to-Parts: Generative AI for Physical Assembly and Scalable Instructions](https://arxiv.org/abs/2512.15743v1)**  
  Authors: David Noever  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2512.15743v1.pdf)  
  <details><summary>Abstract</summary>

  We present a framework for generating physically realizable assembly instructions from natural language descriptions. Unlike unconstrained text-to-3D approaches, our method operates within a discrete parts vocabulary, enforcing geometric validity, connection constraints, and buildability ordering. Using LDraw as a text-rich intermediate representation, we demonstrate that large language models can be guided with tools to produce valid step-by-step construction sequences and assembly instructions for brick-based prototypes of more than 3000 assembly parts. We introduce a Python library for programmatic model generation and evaluate buildable outputs on complex satellites, aircraft, and architectural domains. The approach aims for demonstrable scalability, modularity, and fidelity that bridges the gap between semantic design intent and manufacturable output. Physical prototyping follows from natural language specifications. The work proposes a novel elemental lingua franca as a key missing piece from the previous pixel-based diffusion methods or computer-aided design (CAD) models that fail to support complex assembly instructions or component...
  </details>  
  Keywords: text-to-3d  
- **[WonderZoom: Multi-Scale 3D World Generation](https://arxiv.org/abs/2512.09164v1)**  
  Authors: Jin Cao, Hong-Xing Yu, Jiajun Wu  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2512.09164v1.pdf) | [![Project](https://img.shields.io/badge/-Project-blue)](https://wonderzoom.github.io)  
  <details><summary>Abstract</summary>

  We present WonderZoom, a novel approach to generating 3D scenes with contents across multiple spatial scales from a single image. Existing 3D world generation models remain limited to single-scale synthesis and cannot produce coherent scene contents at varying granularities. The fundamental challenge is the lack of a scale-aware 3D representation capable of generating and rendering content with largely different spatial sizes. WonderZoom addresses this through two key innovations: (1) scale-adaptive Gaussian surfels for generating and real-time rendering of multi-scale 3D scenes, and (2) a progressive detail synthesizer that iteratively generates finer-scale 3D contents. Our approach enables users to "zoom into" a 3D region and auto-regressively synthesize previously non-existent fine details from landscapes to microscopic features. Experiments demonstrate that WonderZoom significantly outperforms state-of-the-art video and 3D models in both quality and alignment, enabling multi-scale 3D world creation from a single image. We show video results and an interactive viewer of generated...
  </details>  
- **[Self-Evolving 3D Scene Generation from a Single Image](https://arxiv.org/abs/2512.08905v1)**  
  Authors: Kaizhi Zheng, Yue Fan, Jing Gu, Zishuo Xu, Xuehai He, Xin Eric Wang  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2512.08905v1.pdf)  
  <details><summary>Abstract</summary>

  Generating high-quality, textured 3D scenes from a single image remains a fundamental challenge in vision and graphics. Recent image-to-3D generators recover reasonable geometry from single views, but their object-centric training limits generalization to complex, large-scale scenes with faithful structure and texture. We present EvoScene, a self-evolving, training-free framework that progressively reconstructs complete 3D scenes from single images. The key idea is combining the complementary strengths of existing models: geometric reasoning from 3D generation models and visual knowledge from video generation models. Through three iterative stages--Spatial Prior Initialization, Visual-guided 3D Scene Mesh Generation, and Spatial-guided Novel View Generation--EvoScene alternates between 2D and 3D domains, gradually improving both structure and appearance. Experiments on diverse scenes demonstrate that EvoScene achieves superior geometric stability, view-consistent textures, and unseen-region completion compared to strong baselines, producing ready-to-use 3D meshes for practical applications.
  </details>  
  Keywords: 3d generator, image-to-3d  
- **[Photo3D: Advancing Photorealistic 3D Generation through Structure-Aligned Detail Enhancement](https://arxiv.org/abs/2512.08535v3)**  
  Authors: Xinyue Liang, Zhinyuan Ma, Lingchen Sun, Yanjun Guo, Lei Zhang  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2512.08535v3.pdf)  
  <details><summary>Abstract</summary>

  Although recent 3D-native generators have made great progress in synthesizing reliable geometry, they still fall short in achieving realistic appearances. A key obstacle lies in the lack of diverse and high-quality real-world 3D assets with rich texture details, since capturing such data is intrinsically difficult due to the diverse scales of scenes, non-rigid motions of objects, and the limited precision of 3D scanners. We introduce Photo3D, a framework for advancing photorealistic 3D generation, which is driven by the image data generated by the GPT-4o-Image model. Considering that the generated images can distort 3D structures due to their lack of multi-view consistency, we design a structure-aligned multi-view synthesis pipeline and construct a detail-enhanced multi-view dataset paired with 3D geometry. Building on it, we present a realistic detail enhancement scheme that leverages perceptual feature adaptation and semantic structure matching to enforce appearance consistency with realistic details while preserving the structural consistency with...
  </details>  
  Keywords: multi-view synthesis  
- **[On-the-fly Large-scale 3D Reconstruction from Multi-Camera Rigs](https://arxiv.org/abs/2512.08498v1)**  
  Authors: Yijia Guo, Tong Hu, Zhiwei Li, Liwen Hu, Keming Qian, Xitong Lin, Shengbo Chen, Tiejun Huang, Lei Ma  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2512.08498v1.pdf)  
  <details><summary>Abstract</summary>

  Recent advances in 3D Gaussian Splatting (3DGS) have enabled efficient free-viewpoint rendering and photorealistic scene reconstruction. While on-the-fly extensions of 3DGS have shown promise for real-time reconstruction from monocular RGB streams, they often fail to achieve complete 3D coverage due to the limited field of view (FOV). Employing a multi-camera rig fundamentally addresses this limitation. In this paper, we present the first on-the-fly 3D reconstruction framework for multi-camera rigs. Our method incrementally fuses dense RGB streams from multiple overlapping cameras into a unified Gaussian representation, achieving drift-free trajectory estimation and efficient online reconstruction. We propose a hierarchical camera initialization scheme that enables coarse inter-camera alignment without calibration, followed by a lightweight multi-camera bundle adjustment that stabilizes trajectories while maintaining real-time performance. Furthermore, we introduce a redundancy-free Gaussian sampling strategy and a frequency-aware optimization scheduler to reduce the number of Gaussian primitives and the required optimization iterations, thereby maintaining both efficiency...
  </details>  
  Keywords: 3d scene reconstruction  
- **[Conditional Morphogenesis: Emergent Generation of Structural Digits via Neural Cellular Automata](https://arxiv.org/abs/2512.08360v1)**  
  Authors: Ali Sakour  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2512.08360v1.pdf)  
  <details><summary>Abstract</summary>

  Biological systems exhibit remarkable morphogenetic plasticity, where a single genome can encode various specialized cellular structures triggered by local chemical signals. In the domain of Deep Learning, Differentiable Neural Cellular Automata (NCA) have emerged as a paradigm to mimic this self-organization. However, existing NCA research has predominantly focused on continuous texture synthesis or single-target object recovery, leaving the challenge of class-conditional structural generation largely unexplored. In this work, we propose a novel Conditional Neural Cellular Automata (c-NCA) architecture capable of growing distinct topological structures - specifically MNIST digits - from a single generic seed, guided solely by a spatially broadcasted class vector. Unlike traditional generative models (e.g., GANs, VAEs) that rely on global reception fields, our model enforces strict locality and translation equivariance. We demonstrate that by injecting a one-hot condition into the cellular perception field, a single set of local rules can learn to break symmetry and self-assemble into...
  </details>  
  Keywords: texture synthesis  
- **[MolSculpt: Sculpting 3D Molecular Geometries from Chemical Syntax](https://arxiv.org/abs/2512.10991v1)**  
  Authors: Zhanpeng Chen, Weihao Gao, Shunyu Wang, Yanan Zhu, Hong Meng, Yuexian Zou  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2512.10991v1.pdf) | [![GitHub](https://img.shields.io/github/stars/SakuraTroyChen/MolSculpt?style=social)](https://github.com/SakuraTroyChen/MolSculpt)  
  <details><summary>Abstract</summary>

  Generating precise 3D molecular geometries is crucial for drug discovery and material science. While prior efforts leverage 1D representations like SELFIES to ensure molecular validity, they fail to fully exploit the rich chemical knowledge entangled within 1D models, leading to a disconnect between 1D syntactic generation and 3D geometric realization. To bridge this gap, we propose MolSculpt, a novel framework that "sculpts" 3D molecular geometries from chemical syntax. MolSculpt is built upon a frozen 1D molecular foundation model and a 3D molecular diffusion model. We introduce a set of learnable queries to extract inherent chemical knowledge from the foundation model, and a trainable projector then injects this cross-modal information into the conditioning space of the diffusion model to guide the 3D geometry generation. In this way, our model deeply integrates 1D latent chemical knowledge into the 3D generation process through end-to-end optimization. Experiments demonstrate that MolSculpt achieves state-of-the-art (SOTA) performance...
  </details>  
- **[MoCA: Mixture-of-Components Attention for Scalable Compositional 3D Generation](https://arxiv.org/abs/2512.07628v1)**  
  Authors: Zhiqi Li, Wenhuan Li, Tengfei Wang, Zhenwei Wang, Junta Wu, Haoyuan Wang, Yunhan Yang, Zehuan Huang, Yang Li, Peidong Liu, Chunchao Guo  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2512.07628v1.pdf) | [![Project](https://img.shields.io/badge/-Project-blue)](https://lizhiqi49.github.io/MoCA)  
  <details><summary>Abstract</summary>

  Compositionality is critical for 3D object and scene generation, but existing part-aware 3D generation methods suffer from poor scalability due to quadratic global attention costs when increasing the number of components. In this work, we present MoCA, a compositional 3D generative model with two key designs: (1) importance-based component routing that selects top-k relevant components for sparse global attention, and (2) unimportant components compression that preserve contextual priors of unselected components while reducing computational complexity of global attention. With these designs, MoCA enables efficient, fine-grained compositional 3D asset creation with scalable number of components. Extensive experiments show MoCA outperforms baselines on both compositional object and scene generation tasks. Project page: https://lizhiqi49.github.io/MoCA
  </details>  
- **[ControlVP: Interactive Geometric Refinement of AI-Generated Images with Consistent Vanishing Points](https://arxiv.org/abs/2512.07504v1)**  
  Authors: Ryota Okumura, Kaede Shiohara, Toshihiko Yamasaki  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2512.07504v1.pdf) | [![GitHub](https://img.shields.io/github/stars/RyotaOkumura/ControlVP?style=social)](https://github.com/RyotaOkumura/ControlVP)  
  <details><summary>Abstract</summary>

  Recent text-to-image models, such as Stable Diffusion, have achieved impressive visual quality, yet they often suffer from geometric inconsistencies that undermine the structural realism of generated scenes. One prominent issue is vanishing point inconsistency, where projections of parallel lines fail to converge correctly in 2D space. This leads to structurally implausible geometry that degrades spatial realism, especially in architectural scenes. We propose ControlVP, a user-guided framework for correcting vanishing point inconsistencies in generated images. Our approach extends a pre-trained diffusion model by incorporating structural guidance derived from building contours. We also introduce geometric constraints that explicitly encourage alignment between image edges and perspective cues. Our method enhances global geometric consistency while maintaining visual fidelity comparable to the baselines. This capability is particularly valuable for applications that require accurate spatial structure, such as image-to-3D reconstruction. The dataset and source code are available at https://github.com/RyotaOkumura/ControlVP .
  </details>  
  Keywords: image-to-3d  
- **[Hierarchical Image-Guided 3D Point Cloud Segmentation in Industrial Scenes via Multi-View Bayesian Fusion](https://arxiv.org/abs/2512.06882v1)**  
  Authors: Yu Zhu, Naoya Chiba, Koichi Hashimoto  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2512.06882v1.pdf)  
  <details><summary>Abstract</summary>

  Reliable 3D segmentation is critical for understanding complex scenes with dense layouts and multi-scale objects, as commonly seen in industrial environments. In such scenarios, heavy occlusion weakens geometric boundaries between objects, and large differences in object scale will cause end-to-end models fail to capture both coarse and fine details accurately. Existing 3D point-based methods require costly annotations, while image-guided methods often suffer from semantic inconsistencies across views. To address these challenges, we propose a hierarchical image-guided 3D segmentation framework that progressively refines segmentation from instance-level to part-level. Instance segmentation involves rendering a top-view image and projecting SAM-generated masks prompted by YOLO-World back onto the 3D point cloud. Part-level segmentation is subsequently performed by rendering multi-view images of each instance obtained from the previous stage and applying the same 2D segmentation and back-projection process at each view, followed by Bayesian updating fusion to ensure semantic consistency across views. Experiments on real-world...
  </details>  
- **[DragMesh: Interactive 3D Generation Made Easy](https://arxiv.org/abs/2512.06424v1)**  
  Authors: Tianshan Zhang, Zeyu Zhang, Hao Tang  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2512.06424v1.pdf) | [![GitHub](https://img.shields.io/github/stars/AIGeeksGroup/DragMesh?style=social)](https://github.com/AIGeeksGroup/DragMesh) | [![Project](https://img.shields.io/badge/-Project-blue)](https://aigeeksgroup.github.io/DragMesh)  
  <details><summary>Abstract</summary>

  While generative models have excelled at creating static 3D content, the pursuit of systems that understand how objects move and respond to interactions remains a fundamental challenge. Current methods for articulated motion lie at a crossroads: they are either physically consistent but too slow for real-time use, or generative but violate basic kinematic constraints. We present DragMesh, a robust framework for real-time interactive 3D articulation built around a lightweight motion generation core. Our core contribution is a novel decoupled kinematic reasoning and motion generation framework. First, we infer the latent joint parameters by decoupling semantic intent reasoning (which determines the joint type) from geometric regression (which determines the axis and origin using our Kinematics Prediction Network (KPP-Net)). Second, to leverage the compact, continuous, and singularity-free properties of dual quaternions for representing rigid body motion, we develop a novel Dual Quaternion VAE (DQ-VAE). This DQ-VAE receives these predicted priors, along with...
  </details>  
- **[Shoot-Bounce-3D: Single-Shot Occlusion-Aware 3D from Lidar by Decomposing Two-Bounce Light](https://arxiv.org/abs/2512.06080v1)**  
  Authors: Tzofi Klinghoffer, Siddharth Somasundaram, Xiaoyu Xiang, Yuchen Fan, Christian Richardt, Akshat Dave, Ramesh Raskar, Rakesh Ranjan  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2512.06080v1.pdf) | [![Project](https://img.shields.io/badge/-Project-blue)](https://shoot-bounce-3d.github.io)  
  <details><summary>Abstract</summary>

  3D scene reconstruction from a single measurement is challenging, especially in the presence of occluded regions and specular materials, such as mirrors. We address these challenges by leveraging single-photon lidars. These lidars estimate depth from light that is emitted into the scene and reflected directly back to the sensor. However, they can also measure light that bounces multiple times in the scene before reaching the sensor. This multi-bounce light contains additional information that can be used to recover dense depth, occluded geometry, and material properties. Prior work with single-photon lidar, however, has only demonstrated these use cases when a laser sequentially illuminates one scene point at a time. We instead focus on the more practical - and challenging - scenario of illuminating multiple scene points simultaneously. The complexity of light transport due to the combined effects of multiplexed illumination, two-bounce light, shadows, and specular reflections is challenging to invert analytically....
  </details>  
  Keywords: 3d scene reconstruction  
- **[Curvature-Regularized Variational Autoencoder for 3D Scene Reconstruction from Sparse Depth](https://arxiv.org/abs/2512.05783v1)**  
  Authors: Maryam Yousefi, Soodeh Bakhshandeh  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2512.05783v1.pdf) | [![GitHub](https://img.shields.io/github/stars/Maryousefi/GeoVAE-3D?style=social)](https://github.com/Maryousefi/GeoVAE-3D)  
  <details><summary>Abstract</summary>

  When depth sensors provide only 5% of needed measurements, reconstructing complete 3D scenes becomes difficult. Autonomous vehicles and robots cannot tolerate the geometric errors that sparse reconstruction introduces. We propose curvature regularization through a discrete Laplacian operator, achieving 18.1% better reconstruction accuracy than standard variational autoencoders. Our contribution challenges an implicit assumption in geometric deep learning: that combining multiple geometric constraints improves performance. A single well-designed regularization term not only matches but exceeds the effectiveness of complex multi-term formulations. The discrete Laplacian offers stable gradients and noise suppression with just 15% training overhead and zero inference cost. Code and models are available at https://github.com/Maryousefi/GeoVAE-3D.
  </details>  
  Keywords: 3d scene reconstruction  
- **[Hyperspectral Unmixing with 3D Convolutional Sparse Coding and Projected Simplex Volume Maximization](https://arxiv.org/abs/2512.05674v1)**  
  Authors: Gargi Panda, Soumitra Kundu, Saumik Bhattacharya, Aurobinda Routray  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2512.05674v1.pdf)  
  <details><summary>Abstract</summary>

  Hyperspectral unmixing (HSU) aims to separate each pixel into its constituent endmembers and estimate their corresponding abundance fractions. This work presents an algorithm-unrolling-based network for the HSU task, named the 3D Convolutional Sparse Coding Network (3D-CSCNet), built upon a 3D CSC model. Unlike existing unrolling-based networks, our 3D-CSCNet is designed within the powerful autoencoder (AE) framework. Specifically, to solve the 3D CSC problem, we propose a 3D CSC block (3D-CSCB) derived through deep algorithm unrolling. Given a hyperspectral image (HSI), 3D-CSCNet employs the 3D-CSCB to estimate the abundance matrix. The use of 3D CSC enables joint learning of spectral and spatial relationships in the 3D HSI data cube. The estimated abundance matrix is then passed to the AE decoder to reconstruct the HSI, and the decoder weights are extracted as the endmember matrix. Additionally, we propose a projected simplex volume maximization (PSVM) algorithm for endmember estimation, and the resulting endmembers...
  </details>  
- **[SpaceControl: Introducing Test-Time Spatial Control to 3D Generative Modeling](https://arxiv.org/abs/2512.05343v2)**  
  Authors: Elisabetta Fedele, Francis Engelmann, Ian Huang, Or Litany, Marc Pollefeys, Leonidas Guibas  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2512.05343v2.pdf) | [![Project](https://img.shields.io/badge/-Project-blue)](https://spacecontrol3d.github.io)  
  <details><summary>Abstract</summary>

  Generative methods for 3D assets have recently achieved remarkable progress, yet providing intuitive and precise control over the object geometry remains a key challenge. Existing approaches predominantly rely on text or image prompts, which often fall short in geometric specificity: language can be ambiguous, and images are difficult to manipulate. In this work, we introduce SpaceControl, a training-free test-time method for explicit spatial control of 3D asset generation. Our approach accepts a wide range of geometric inputs, from coarse primitives to detailed meshes, and integrates seamlessly with modern generative models without requiring any additional training. A control parameter lets users trade off between geometric fidelity and output realism. Extensive quantitative evaluation and user studies demonstrate that SpaceControl outperforms both training-based and optimization-based baselines in geometric faithfulness while preserving high visual quality. Finally, we present an interactive interface for real-time superquadric editing and direct 3D asset generation, enabling seamless use in...
  </details>  
  Keywords: 3d asset generation  
- **[Object Reconstruction under Occlusion with Generative Priors and Contact-induced Constraints](https://arxiv.org/abs/2512.05079v1)**  
  Authors: Minghan Zhu, Zhiyi Wang, Qihang Sun, Maani Ghaffari, Michael Posa  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2512.05079v1.pdf)  
  <details><summary>Abstract</summary>

  Object geometry is key information for robot manipulation. Yet, object reconstruction is a challenging task because cameras only capture partial observations of objects, especially when occlusion occurs. In this paper, we leverage two extra sources of information to reduce the ambiguity of vision signals. First, generative models learn priors of the shapes of commonly seen objects, allowing us to make reasonable guesses of the unseen part of geometry. Second, contact information, which can be obtained from videos and physical interactions, provides sparse constraints on the boundary of the geometry. We combine the two sources of information through contact-guided 3D generation. The guidance formulation is inspired by drag-based editing in generative models. Experiments on synthetic and real-world data show that our approach improves the reconstruction compared to pure 3D generation and contact-based optimization.
  </details>  
- **[LaFiTe: A Generative Latent Field for 3D Native Texturing](https://arxiv.org/abs/2512.04786v1)**  
  Authors: Chia-Hao Chen, Zi-Xin Zou, Yan-Pei Cao, Ze Yuan, Guan Luo, Xiaojuan Qi, Ding Liang, Song-Hai Zhang, Yuan-Chen Guo  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2512.04786v1.pdf)  
  <details><summary>Abstract</summary>

  Generating high-fidelity, seamless textures directly on 3D surfaces, what we term 3D-native texturing, remains a fundamental open challenge, with the potential to overcome long-standing limitations of UV-based and multi-view projection methods. However, existing native approaches are constrained by the absence of a powerful and versatile latent representation, which severely limits the fidelity and generality of their generated textures. We identify this representation gap as the principal barrier to further progress. We introduce LaFiTe, a framework that addresses this challenge by learning to generate textures as a 3D generative sparse latent color field. At its core, LaFiTe employs a variational autoencoder (VAE) to encode complex surface appearance into a sparse, structured latent space, which is subsequently decoded into a continuous color field. This representation achieves unprecedented fidelity, exceeding state-of-the-art methods by >10 dB PSNR in reconstruction, by effectively disentangling texture appearance from mesh topology and UV parameterization. Building upon this strong...
  </details>  
- **[Order Matters: 3D Shape Generation from Sequential VR Sketches](https://arxiv.org/abs/2512.04761v3)**  
  Authors: Yizi Chen, Sidi Wu, Tianyi Xiao, Nina Wiedemann, Loic Landrieu  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2512.04761v3.pdf) | [![Project](https://img.shields.io/badge/-Project-blue)](https://chenyizi086.github.io/VRSketch2Shape_website)  
  <details><summary>Abstract</summary>

  VR sketching lets users explore and iterate on ideas directly in 3D, offering a faster and more intuitive alternative to conventional CAD tools. However, existing sketch-to-shape models ignore the temporal ordering of strokes, discarding crucial cues about structure and design intent. We introduce VRSketch2Shape, the first framework and multi-category dataset for generating 3D shapes from sequential VR sketches. Our contributions are threefold: (i) an automated pipeline that generates sequential VR sketches from arbitrary shapes, (ii) a dataset of over 20k synthetic and 900 hand-drawn sketch-shape pairs across four categories, and (iii) an order-aware sketch encoder coupled with a diffusion-based 3D generator. Our approach yields higher geometric fidelity than prior work, generalizes effectively from synthetic to real sketches with minimal supervision, and performs well even on partial sketches. All data and models will be released open-source at https://chenyizi086.github.io/VRSketch2Shape_website.
  </details>  
  Keywords: 3d generator  
- **[GaussianBlender: Instant Stylization of 3D Gaussians with Disentangled Latent Spaces](https://arxiv.org/abs/2512.03683v1)**  
  Authors: Melis Ocal, Xiaoyan Xing, Yue Li, Ngo Anh Vien, Sezer Karaoglu, Theo Gevers  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2512.03683v1.pdf)  
  <details><summary>Abstract</summary>

  3D stylization is central to game development, virtual reality, and digital arts, where the demand for diverse assets calls for scalable methods that support fast, high-fidelity manipulation. Existing text-to-3D stylization methods typically distill from 2D image editors, requiring time-intensive per-asset optimization and exhibiting multi-view inconsistency due to the limitations of current text-to-image models, which makes them impractical for large-scale production. In this paper, we introduce GaussianBlender, a pioneering feed-forward framework for text-driven 3D stylization that performs edits instantly at inference. Our method learns structured, disentangled latent spaces with controlled information sharing for geometry and appearance from spatially-grouped 3D Gaussians. A latent diffusion model then applies text-conditioned edits on these learned representations. Comprehensive evaluations show that GaussianBlender not only delivers instant, high-fidelity, geometry-preserving, multi-view consistent stylization, but also surpasses methods that require per-instance test-time optimization - unlocking practical, democratized 3D stylization at scale.
  </details>  
  Keywords: text-to-3d  
- **[GAOT: Generating Articulated Objects Through Text-Guided Diffusion Models](https://arxiv.org/abs/2512.03566v1)**  
  Authors: Hao Sun, Lei Fan, Donglin Di, Shaohui Liu  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2512.03566v1.pdf)  
  <details><summary>Abstract</summary>

  Articulated object generation has seen increasing advancements, yet existing models often lack the ability to be conditioned on text prompts. To address the significant gap between textual descriptions and 3D articulated object representations, we propose GAOT, a three-phase framework that generates articulated objects from text prompts, leveraging diffusion models and hypergraph learning in a three-step process. First, we fine-tune a point cloud generation model to produce a coarse representation of objects from text prompts. Given the inherent connection between articulated objects and graph structures, we design a hypergraph-based learning method to refine these coarse representations, representing object parts as graph vertices. Finally, leveraging a diffusion model, the joints of articulated objects-represented as graph edges-are generated based on the object parts. Extensive qualitative and quantitative experiments on the PartNet-Mobility dataset demonstrate the effectiveness of our approach, achieving superior performance over previous methods.
  </details>  
  Keywords: articulated object generation  
- **[KeyPointDiffuser: Unsupervised 3D Keypoint Learning via Latent Diffusion Models](https://arxiv.org/abs/2512.03450v1)**  
  Authors: Rhys Newbury, Juyan Zhang, Tin Tran, Hanna Kurniawati, Dana Kulić  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2512.03450v1.pdf)  
  <details><summary>Abstract</summary>

  Understanding and representing the structure of 3D objects in an unsupervised manner remains a core challenge in computer vision and graphics. Most existing unsupervised keypoint methods are not designed for unconditional generative settings, restricting their use in modern 3D generative pipelines; our formulation explicitly bridges this gap. We present an unsupervised framework for learning spatially structured 3D keypoints from point cloud data. These keypoints serve as a compact and interpretable representation that conditions an Elucidated Diffusion Model (EDM) to reconstruct the full shape. The learned keypoints exhibit repeatable spatial structure across object instances and support smooth interpolation in keypoint space, indicating that they capture geometric variation. Our method achieves strong performance across diverse object categories, yielding a 6 percentage-point improvement in keypoint consistency compared to prior approaches.
  </details>  
- **[ShelfGaussian: Shelf-Supervised Open-Vocabulary Gaussian-based 3D Scene Understanding](https://arxiv.org/abs/2512.03370v3)**  
  Authors: Lingjun Zhao, Yandong Luo, James Hays, Lu Gan  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2512.03370v3.pdf) | [![Project](https://img.shields.io/badge/-Project-blue)](https://lunarlab-gatech.github.io/ShelfGaussian)  
  <details><summary>Abstract</summary>

  We introduce ShelfGaussian, an open-vocabulary multi-modal Gaussian-based 3D scene understanding framework supervised by off-the-shelf vision foundation models (VFMs). Gaussian-based methods have demonstrated superior performance and computational efficiency across a wide range of scene understanding tasks. However, existing methods either model objects as closed-set semantic Gaussians supervised by annotated 3D labels, neglecting their rendering ability, or learn open-set Gaussian representations via purely 2D self-supervision, leading to degraded geometry and limited to camera-only settings. To fully exploit the potential of Gaussians, we propose a Multi-Modal Gaussian Transformer that enables Gaussians to query features from diverse sensor modalities, and a Shelf-Supervised Learning Paradigm that efficiently optimizes Gaussians with VFM features jointly at 2D image and 3D scene levels. We evaluate ShelfGaussian on various perception and planning tasks. Experiments on Occ3D-nuScenes demonstrate its state-of-the-art zero-shot semantic occupancy prediction performance. ShelfGaussian is further evaluated on an unmanned ground vehicle (UGV) to assess its in the-wild...
  </details>  
- **[TEXTRIX: Latent Attribute Grid for Native Texture Generation and Beyond](https://arxiv.org/abs/2512.02993v1)**  
  Authors: Yifei Zeng, Yajie Bao, Jiachen Qian, Shuang Wu, Youtian Lin, Hao Zhu, Buyu Li, Feihu Zhang, Xun Cao, Yao Yao  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2512.02993v1.pdf)  
  <details><summary>Abstract</summary>

  Prevailing 3D texture generation methods, which often rely on multi-view fusion, are frequently hindered by inter-view inconsistencies and incomplete coverage of complex surfaces, limiting the fidelity and completeness of the generated content. To overcome these challenges, we introduce TEXTRIX, a native 3D attribute generation framework for high-fidelity texture synthesis and downstream applications such as precise 3D part segmentation. Our approach constructs a latent 3D attribute grid and leverages a Diffusion Transformer equipped with sparse attention, enabling direct coloring of 3D models in volumetric space and fundamentally avoiding the limitations of multi-view fusion. Built upon this native representation, the framework naturally extends to high-precision 3D segmentation by training the same architecture to predict semantic attributes on the grid. Extensive experiments demonstrate state-of-the-art performance on both tasks, producing seamless, high-fidelity textures and accurate 3D part segmentation with precise boundaries.
  </details>  
  Keywords: texture synthesis  
- **[Layout Anything: One Transformer for Universal Room Layout Estimation](https://arxiv.org/abs/2512.02952v1)**  
  Authors: Md Sohag Mia, Muhammad Abdullah Adnan  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2512.02952v1.pdf)  
  <details><summary>Abstract</summary>

  We present Layout Anything, a transformer-based framework for indoor layout estimation that adapts the OneFormer's universal segmentation architecture to geometric structure prediction. Our approach integrates OneFormer's task-conditioned queries and contrastive learning with two key modules: (1) a layout degeneration strategy that augments training data while preserving Manhattan-world constraints through topology-aware transformations, and (2) differentiable geometric losses that directly enforce planar consistency and sharp boundary predictions during training. By unifying these components in an end-to-end framework, the model eliminates complex post-processing pipelines while achieving high-speed inference at 114ms. Extensive experiments demonstrate state-of-the-art performance across standard benchmarks, with pixel error (PE) of 5.43% and corner error (CE) of 4.02% on the LSUN, PE of 7.04% (CE 5.17%) on the Hedau and PE of 4.03% (CE 3.15%) on the Matterport3D-Layout datasets. The framework's combination of geometric awareness and computational efficiency makes it particularly suitable for augmented reality applications and large-scale 3D scene reconstruction...
  </details>  
  Keywords: 3d scene reconstruction  
- **[DepthScape: Authoring 2.5D Designs via Depth Estimation, Semantic Understanding, and Geometry Extraction](https://arxiv.org/abs/2512.02263v1)**  
  Authors: Xia Su, Cuong Nguyen, Matheus A. Gadelha, Jon E. Froehlich  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2512.02263v1.pdf)  
  <details><summary>Abstract</summary>

  2.5D effects, such as occlusion and perspective foreshortening, enhance visual dynamics and realism by incorporating 3D depth cues into 2D designs. However, creating such effects remains challenging and labor-intensive due to the complexity of depth perception. We introduce DepthScape, a human-AI collaborative system that facilitates 2.5D effect creation by directly placing design elements into 3D reconstructions. Using monocular depth reconstruction, DepthScape transforms images into 3D reconstructions where visual contents are placed to automatically achieve realistic occlusion and perspective foreshortening. To further simplify 3D placement through a 2D viewport, DepthScape uses a vision-language model to analyze source images and extract key visual components as content anchors for direct manipulation editing. We evaluate DepthScape with nine participants of varying design backgrounds, confirming the effectiveness of our creation pipeline. We also test on 100 professional stock images to assess robustness, and conduct an expert evaluation that confirms the quality of DepthScape's results.
  </details>  
- **[CoatFusion: Controllable Material Coating in Images](https://arxiv.org/abs/2512.02143v1)**  
  Authors: Sagie Levy, Elad Aharoni, Matan Levy, Ariel Shamir, Dani Lischinski  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2512.02143v1.pdf)  
  <details><summary>Abstract</summary>

  We introduce Material Coating, a novel image editing task that simulates applying a thin material layer onto an object while preserving its underlying coarse and fine geometry. Material coating is fundamentally different from existing "material transfer" methods, which are designed to replace an object's intrinsic material, often overwriting fine details. To address this new task, we construct a large-scale synthetic dataset (110K images) of 3D objects with varied, physically-based coatings, named DataCoat110K. We then propose CoatFusion, a novel architecture that enables this task by conditioning a diffusion model on both a 2D albedo texture and granular, PBR-style parametric controls, including roughness, metalness, transmission, and a key thickness parameter. Experiments and user studies show CoatFusion produces realistic, controllable coatings and significantly outperforms existing material editing and transfer methods on this new task.
  </details>  
- **[SRAM: Shape-Realism Alignment Metric for No Reference 3D Shape Evaluation](https://arxiv.org/abs/2512.01373v1)**  
  Authors: Sheng Liu, Tianyu Luan, Phani Nuney, Xuelu Feng, Junsong Yuan  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2512.01373v1.pdf)  
  <details><summary>Abstract</summary>

  3D generation and reconstruction techniques have been widely used in computer games, film, and other content creation areas. As the application grows, there is a growing demand for 3D shapes that look truly realistic. Traditional evaluation methods rely on a ground truth to measure mesh fidelity. However, in many practical cases, a shape's realism does not depend on having a ground truth reference. In this work, we propose a Shape-Realism Alignment Metric that leverages a large language model (LLM) as a bridge between mesh shape information and realism evaluation. To achieve this, we adopt a mesh encoding approach that converts 3D shapes into the language token space. A dedicated realism decoder is designed to align the language model's output with human perception of realism. Additionally, we introduce a new dataset, RealismGrading, which provides human-annotated realism scores without the need for ground truth shapes. Our dataset includes shapes generated by 16...
  </details>  
- **[TabletopGen: Instance-Level Interactive 3D Tabletop Scene Generation from Text or Single Image](https://arxiv.org/abs/2512.01204v3)**  
  Authors: Ziqian Wang, Yonghao He, Licheng Yang, Wei Zou, Hongxuan Ma, Liu Liu, Wei Sui, Yuxin Guo, Hu Su  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2512.01204v3.pdf)  
  <details><summary>Abstract</summary>

  Generating high-fidelity, physically interactive 3D simulated tabletop scenes is essential for embodied AI -- especially for robotic manipulation policy learning and data synthesis. However, current text- or image-driven 3D scene generation methods mainly focus on large-scale scenes, struggling to capture the high-density layouts and complex spatial relations that characterize tabletop scenes. To address these challenges, we propose TabletopGen, a training-free, fully automatic framework that generates diverse, instance-level interactive 3D tabletop scenes. TabletopGen accepts a reference image as input, which can be synthesized by a text-to-image model to enhance scene diversity. We then perform instance segmentation and completion on the reference to obtain per-instance images. Each instance is reconstructed into a 3D model followed by canonical coordinate alignment. The aligned 3D models then undergo pose and scale estimation before being assembled into a collision-free, simulation-ready tabletop scene. A key component of our framework is a novel pose and scale alignment approach...
  </details>  

### April 2026
- **[MeshReGen: A Unified 3D Geometry Regeneration Framework](https://arxiv.org/abs/2604.28134v2)**  
  Authors: Geon Yeong Park, Roman Shapovalov, Rakesh Ranjan, Jong Chul Ye, Andrea Vedaldi, Thu Nguyen-Phuoc  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2604.28134v2.pdf)  
  <details><summary>Abstract</summary>

  We consider the problem of regenerating 3D objects from 2D images and initial 3D shapes. Most 3D generators operate in a one-shot fashion, converting text or images to a 3D object with limited controllability. We introduce instead MeshReGen, a 3D regenerator that is conditioned on an initial 3D shape. This conceptually simple formulation allows us to support numerous useful tasks, including 3D enhancement, reconstruction, and editing. MeshReGen uses a new conditioning mechanism based on VecSet, which allows the regenerator to update or improve the input geometry with consistent fine-grained details. MeshReGen learns a widely applicable regeneration prior from off-the-shelf 3D datasets via self-supervised pretext tasks and augmentations, without additional annotations. We evaluate both the geometric consistency and fine-grained quality of MeshReGen, achieving state-of-the-art performance in controllable 3D generation across several tasks.
  </details>  
  Keywords: 3d generator, controllable 3d generation  
- **[REVIVE 3D: Refinement via Encoded Voluminous Inflated prior for Volume Enhancement](https://arxiv.org/abs/2604.27504v1)**  
  Authors: Hankyeol Lee, Wooyeol Baek, Seongdo Kim, Jongyoo Kim  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2604.27504v1.pdf)  
  <details><summary>Abstract</summary>

  Recent generative models have shown strong performance in generating diverse 3D assets from 2D images, a fundamental research topic in computer vision and graphics. However, these models still struggle to generate voluminous 3D assets when the input is a flat image that provides limited 3D cues. We introduce REVIVE 3D, a two-stage, plug-and-play pipeline for generating voluminous 3D assets from flat images. In Stage 1, we construct an Inflated Prior by inflating the foreground silhouette to recover global volume and superimposing part-aware details to capture local structure. In Stage 2, 3D Latent Refinement injects Gaussian noise into the Inflated Prior's latent and then denoises it, using the prior's geometric cues to leverage the backbone's pretrained 3D knowledge. Furthermore, our framework supports image-conditioned 3D editing. To quantify volume and surface flatness, we propose Compactness and Normal Anisotropy. We validate Compactness and Normal Anisotropy through a user study, showing that these metrics...
  </details>  
- **[ProcFunc: Function-Oriented Abstractions for Procedural 3D Generation in Python](https://arxiv.org/abs/2604.26943v1)**  
  Authors: Alexander Raistrick, Karhan Kayan, Jack Nugent, David Yan, Lingjie Mei, Meenal Parakh, Hongyu Wen, Dylan Li, Yiming Zuo, Erich Liang, Jia Deng  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2604.26943v1.pdf) | [![GitHub](https://img.shields.io/github/stars/princeton-vl/procfunc?style=social)](https://github.com/princeton-vl/procfunc)  
  <details><summary>Abstract</summary>

  We introduce ProcFunc, a library for Blender-based procedural 3D generation in Python. ProcFunc provides a library of easy-to-use Python functions, which streamline creating, combining, analyzing, and executing procedural generation code. ProcFunc makes it easy to create large-scale diverse training data, by combinatorial compositions of semantic components. VLMs can use ProcFunc to edit procedural material and geometry code and can create new procedural code with significantly fewer coding errors. Finally, as an example use case, we use ProcFunc to develop a new procedural generator of indoor rooms, which includes a collection of new compositional procedural materials. We demonstrate the detail, runtime efficiency, and diversity of this room generator, as well as its use for 3D synthetic data generation. Please visit https://github.com/princeton-vl/procfunc for source code.
  </details>  
- **[3D Generation for Embodied AI and Robotic Simulation: A Survey](https://arxiv.org/abs/2604.26509v3)**  
  Authors: Tianwei Ye, Yifan Mao, Minwen Liao, Jian Liu, Chunchao Guo, Dazhao Du, Quanxin Shou, Fangqi Zhu, Song Guo  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2604.26509v3.pdf) | [![Project](https://img.shields.io/badge/-Project-blue)](https://3dgen4robot.github.io)  
  <details><summary>Abstract</summary>

  Embodied AI and robotic systems increasingly depend on scalable, diverse, and physically grounded 3D content for simulation-based training and real-world deployment. While 3D generative modeling has advanced rapidly, embodied applications impose requirements far beyond visual realism: generated objects must carry kinematic structure and material properties, scenes must support interaction and task execution, and the resulting content must bridge the gap between simulation and reality. This survey reviews 3D generation for embodied AI and organizes the literature around three roles that 3D generation plays in embodied systems. In Data Generator, 3D generation produces simulation-ready objects and assets, including articulated, physically grounded, and deformable content for downstream interaction; in Simulation Environments, it constructs interactive and task-oriented worlds, spanning structure-aware, controllable, and agentic scene generation; and in Sim2Real Bridge, it supports digital twin reconstruction, data augmentation, and synthetic demonstrations for downstream robot learning and real-world transfer. We also show that the field is...
  </details>  
- **[ESICA: A Scalable Framework for Text-Guided 3D Medical Image Segmentation](https://arxiv.org/abs/2604.24876v1)**  
  Authors: Yu Xin, Gorkem Can Ates, Jun Ma, Sumin Kim, Ying Zhang, Kaleb E Smith, Kuang Gong, Wei Shao  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2604.24876v1.pdf) | [![GitHub](https://img.shields.io/github/stars/mirthAI/ESICA?style=social)](https://github.com/mirthAI/ESICA)  
  <details><summary>Abstract</summary>

  Text guided 3D medical image segmentation offers a flexible alternative to class based and spatial prompt based models by allowing users to specify regions of interest directly in natural language. This paradigm avoids reliance on predefined label sets, reduces ambiguous outputs, and aligns more naturally with clinical workflows. However, existing text guided frameworks are often computationally expensive, exhibit weak text volume feature alignment, and fail to capture fine anatomical details. We propose ESICA, a lightweight and scalable framework that addresses these challenges through three innovations: (1) a similarity matrix based mask prediction formulation that enhances semantic alignment, (2) an efficient decomposed decoder with adapter modules for accurate volumetric decoding, and (3) a two pass refinement strategy that sharpens boundaries and resolves uncertain regions. To improve training stability and generalization, ESICA adopts a two stage scheme consisting of positive only pretraining followed by balanced fine tuning. On the CVPR BiomedSegFM benchmark...
  </details>  
- **[Prox-E: Fine-Grained 3D Shape Editing via Primitive-Based Abstractions](https://arxiv.org/abs/2604.23774v2)**  
  Authors: Etai Sella, Hao Phung, Nitay Amiel, Or Litany, Or Patashnik, Hadar Averbuch-Elor  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2604.23774v2.pdf)  
  <details><summary>Abstract</summary>

  Text-based 2D image editing models have recently reached an impressive level of maturity, motivating a growing body of work that heavily depends on these models to drive 3D edits. While effective for appearance-based modifications, such 2D-centric 3D editing pipelines often struggle with fine-grained 3D editing, where localized structural changes must be applied while strictly preserving an object's overall identity. To address this limitation, we propose Prox-E, a training-free framework that enables fine-grained 3D control through an explicit, primitive-based geometric abstraction. Our framework first abstracts an input 3D shape into a compact set of geometric primitives. A pretrained vision-language model (VLM) then edits this abstraction to specify primitive-level changes. These structural edits are subsequently used to guide a 3D generative model, enabling fine-grained, localized modifications while preserving unchanged regions of the original shape. Through extensive experiments, we demonstrate that our method consistently balances identity preservation, shape quality, and instruction fidelity more...
  </details>  
- **[A Pose-only Geometric Constraint for Multi-Camera Pose Adjustment](https://arxiv.org/abs/2604.23704v1)**  
  Authors: Shunkun Liang, Banglei Guan, Bin Li, Qifeng Yu, Yang Shang  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2604.23704v1.pdf)  
  <details><summary>Abstract</summary>

  Multi-camera systems offer rich observation capabilities for visual navigation and 3D scene reconstruction; however, the resulting feature redundancy often compromises computational efficiency. This challenge is particularly pronounced during bundle adjustment, where the non-linear optimization of both system poses and scene points incurs substantial computational overhead. To address this challenge, this paper introduces a pose-only geometric constraint for multi-camera systems and proposes a corresponding pose adjustment algorithm. Specifically, we use generalized camera model to establish a unified representation of the multi-camera system. Building upon this model, we formulate the multi-camera pose-only constraint, which implicitly represents a 3D scene point using two base observations and their associated poses, thereby achieving a pose-only representation of the projection geometry. Subsequently, we introduce a multi-camera pose adjustment algorithm that eliminates 3D points from the parameter space, thereby achieving efficient and focused pose optimization. Experimental results on both synthetic and real-world datasets demonstrate that the proposed...
  </details>  
  Keywords: 3d scene reconstruction  
- **[From Visual Synthesis to Interactive Worlds: Toward Production-Ready 3D Asset Generation](https://arxiv.org/abs/2604.23629v2)**  
  Authors: Jiafeng Wu, Zhuofan Lou, Jian Liu, Dazhao Du, Chunchao Guo, Song Guo  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2604.23629v2.pdf)  
  <details><summary>Abstract</summary>

  Three-dimensional content generation has progressed from producing isolated, visually plausible shapes to constructing structured assets that can be deployed in real-time interactive environments. This trajectory is driven by converging demands from game development, embodied AI, world simulation, digital twins, and spatial computing, all of which require 3D content that goes beyond surface appearance to satisfy engine-level constraints on topology, UV parameterization, physically based materials, skeletal rigging, and physics-aware scene layout. Despite rapid advances in generative modeling, a persistent gap separates the outputs of current methods from the production-ready standard expected by interactive applications. This survey addresses that gap by organizing the literature around the asset production pipeline rather than algorithmic families. Along the horizontal axis we distinguish three asset tiers, namely general objects, characters, and scenes, while the vertical axis traces each tier through the full production lifecycle from data foundations and geometry synthesis through topology optimization, UV unwrapping, PBR...
  </details>  
  Keywords: 3d asset generation, rigging  
- **[PILOT: One Physics-Integrated Generation Framework to Unify 2D and 3D Radio Map Construction](https://arxiv.org/abs/2604.23533v1)**  
  Authors: Weiming Huang, Hao Sun, Junting Chen  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2604.23533v1.pdf)  
  <details><summary>Abstract</summary>

  Unified 2D and 3D radio map construction supports network planning, wireless digital twins, and unmanned aerial vehicle (UAV) applications. In urban environments, blockage, reflection, and diffraction make accurate construction expensive for physics-based solvers. Autoregressive next-token prediction offers a single sequential formulation that can cover both 2D and 3D generation, but standard raster ordering ignores the spatial structure of radio propagation. When generation follows propagation, each token is predicted from propagation-relevant history rather than spatially arbitrary context, which provides more causally informative conditioning and lowers conditional uncertainty. We propose PILOT, a pretrained autoregressive framework that replaces raster scan with a wavefront sequence expanding outward from the transmitter. Each prediction step is guided by an environment-aware instruction that spatially aligns environment features with the queried radio map region. The same framework extends to 3D radio maps through height-slice stacking while a gradient loss enforces vertical continuity. On standard 2D benchmarks, PILOT achieves...
  </details>  
- **[PASR: Pose-Aware 3D Shape Retrieval from Occluded Single Views](https://arxiv.org/abs/2604.22658v2)**  
  Authors: Jiaxin Shi, Guofeng Zhang, Wufei Ma, Naifu Liang, Adam Kortylewski, Alan Yuille  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2604.22658v2.pdf)  
  <details><summary>Abstract</summary>

  Single-view 3D shape retrieval is a fundamental yet challenging task that is increasingly important with the growth of available 3D data. Existing approaches largely fall into two categories: those using contrastive learning to map point cloud features into existing vision-language spaces and those that learn a common embedding space for 2D images and 3D shapes. However, these feed-forward, holistic alignments are often difficult to interpret, which in turn limits their robustness and generalization to real-world applications. To address this problem, we propose Pose-Aware 3D Shape Retrieval (PASR), a framework that formulates retrieval as a feature-level analysis-by-synthesis problem by distilling knowledge from a 2D foundation model (DINOv3) into a 3D encoder. By aligning pose-conditioned 3D projections with 2D feature maps, our method bridges the gap between real-world images and synthetic meshes. During inference, PASR performs a test-time optimization via analysis-by-synthesis, jointly searching for the shape and pose that best reconstruct the...
  </details>  
- **[Context Unrolling in Omni Models](https://arxiv.org/abs/2604.21921v1)**  
  Authors: Ceyuan Yang, Zhijie Lin, Yang Zhao, Fei Xiao, Hao He, Qi Zhao, Chaorui Deng, Kunchang Li, Zihan Ding, Yuwei Guo, Fuyun Wang, Fangqi Zhu, Xiaonan Nie, Shenhan Zhu, Shanchuan Lin, Hongsheng Li, Weilin Huang, Guang Shi, Haoqi Fan  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2604.21921v1.pdf)  
  <details><summary>Abstract</summary>

  We present Omni, a unified multimodal model natively trained on diverse modalities, including text, images, videos, 3D geometry, and hidden representations. We find that such training enables Context Unrolling, where the model explicitly reasons across multiple modal representations before producing predictions. This process enables the model to aggregate complementary information across heterogeneous modalities, facilitating a more faithful approximation of the shared multimodal knowledge manifold and improving downstream reasoning fidelity. As a result, Omni achieves strong performance on both multimodal generation and understanding benchmarks, while demonstrating advanced multimodal reasoning capabilities, including in-context generation of text, image, video, and 3D geometry.
  </details>  
- **[Sculpt4D: Generating 4D Shapes via Sparse-Attention Diffusion Transformers](https://arxiv.org/abs/2604.21592v1)**  
  Authors: Minghao Yin, Wenbo Hu, Jiale Xu, Ying Shan, Kai Han  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2604.21592v1.pdf)  
  <details><summary>Abstract</summary>

  Recent breakthroughs in 3D generative modeling have yielded remarkable progress in static shape synthesis, yet high-fidelity dynamic 4D generation remains elusive, hindered by temporal artifacts and prohibitive computational demand. We present Sculpt4D, a native 4D generative framework that seamlessly integrates efficient temporal modeling into a pretrained 3D Diffusion Transformer (Hunyuan3D 2.1), thereby mitigating the scarcity of 4D training data. At its core lies a Block Sparse Attention mechanism that preserves object identity by anchoring to the initial frame while capturing rich motion dynamics via a time-decaying sparse mask. This design faithfully models complex spatiotemporal dependencies with high fidelity, while sidestepping the quadratic overhead of full attention and reducing network total computation by 56%. Consequently, Sculpt4D establishes a new state-of-the-art in temporally coherent 4D synthesis and charts a path toward efficient and scalable 4D generation.
  </details>  
- **[Seed3D 2.0: Advancing High-Fidelity Simulation-Ready 3D Content Generation](https://arxiv.org/abs/2605.13862v1)**  
  Authors: Diandian Gu, Jing Lin, Gaohong Liu, Jiahang Liu, Su Ma, Guang Shi, Jun Wang, Qinlong Wang, Qianyi Wu, Zhongcong Xu, Xuanyu Yi, Zihao Yu, Jianfeng Zhang, Zhuolin Zheng, Yifan Zhu, Rui Chen, Hengkai Guo, Xiaoyang Guo, Mingcong Han, Xu Han, Xiu Li, Yixun Liang, Weiqiang Lou, Junzhe Lu, Guan Luo, Minghan Qin, Shuguang Wang, Yuang Wang  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2605.13862v1.pdf) | [![Project](https://img.shields.io/badge/-Project-blue)](https://exp.volcengine.com/ark/vision?_vtm_=0.0.c70961.d701978.0&mode=vision&modelId=doubao-seed3d-2-0-260328&tab=Gen3D)  
  <details><summary>Abstract</summary>

  We present Seed3D 2.0, an advanced 3D content generation system built on Seed3D 1.0, with substantial improvements across generation fidelity, simulation-ready capabilities, and application coverage. For geometry, a coarse-to-fine two-stage pipeline decouples global structure learning from high-frequency detail recovery, while a locality-aware VAE achieves higher spatial compression and more efficient decoding. For texture and material generation, we replace the cascaded pipeline of Seed3D 1.0 with a unified PBR model that directly generates multi-view albedo and metallic-roughness maps, enhanced by Mixture-of-Experts scaling and VLM-based semantic conditioning for improved material precision and visual fidelity. Beyond single-object generation, Seed3D 2.0 introduces a simulation-ready model suite comprising scene layout planning, part-aware decomposition, and training-free articulation generation, enabling coherent scene construction and part-level physical interaction across physics and graphics engines. A large-scale human preference study against five recent commercial models shows that Seed3D 2.0 achieves consistent win rates of 69.0% to 89.9% in textured 3D...
  </details>  
  Keywords: 3d asset generation  
- **[FurnSet: Exploiting Repeats for 3D Scene Reconstruction](https://arxiv.org/abs/2604.20093v1)**  
  Authors: Paul Dobre, Xin Wang, Hongzhou Yang  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2604.20093v1.pdf)  
  <details><summary>Abstract</summary>

  Single-view 3D scene reconstruction involves inferring both object geometry and spatial layout. Existing methods typically reconstruct objects independently or rely on implicit scene context, failing to exploit the repeated instances commonly present in realworld scenes. We propose FurnSet, a framework that explicitly identifies and leverages repeated object instances to improve reconstruction. Our method introduces per-object CLS tokens and a set-aware self-attention mechanism that groups identical instances and aggregates complementary observations across them, enabling joint reconstruction. We further combine scene-level and object-level conditioning to guide object reconstruction, followed by layout optimization using object point clouds with 3D and 2D projection losses for scene alignment. Experiments on 3D-Future and 3D-Front demonstrate improved scene reconstruction quality, highlighting the effectiveness of exploiting repetition for robust 3D scene reconstruction.
  </details>  
  Keywords: 3d scene reconstruction  
- **[Unposed-to-3D: Learning Simulation-Ready Vehicles from Real-World Images](https://arxiv.org/abs/2604.19257v1)**  
  Authors: Hongyuan Liu, Bochao Zou, Qiankun Liu, Haochen Yu, Qi Mei, Jianfei Jiang, Chen Liu, Cheng Bi, Zhao Wang, Xueyang Zhang, Yifei Zhan, Jiansheng Chen, Huimin Ma  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2604.19257v1.pdf)  
  <details><summary>Abstract</summary>

  Creating realistic and simulation-ready 3D assets is crucial for autonomous driving research and virtual environment construction. However, existing 3D vehicle generation methods are often trained on synthetic data with significant domain gaps from real-world distributions. The generated models often exhibit arbitrary poses and undefined scales, resulting in poor visual consistency when integrated into driving scenes. In this paper, we present Unposed-to-3D, a novel framework that learns to reconstruct 3D vehicles from real-world driving images using image-only supervision. Our approach consists of two stages. In the first stage, we train an image-to-3D reconstruction network using posed images with known camera parameters. In the second stage, we remove camera supervision and use a camera prediction head that directly estimates the camera parameters from unposed images. The predicted pose is then used for differentiable rendering to provide self-supervised photometric feedback, enabling the model to learn 3D geometry purely from unposed images. To ensure...
  </details>  
  Keywords: image-to-3d  
- **[Align then Refine: Text-Guided 3D Prostate Lesion Segmentation](https://arxiv.org/abs/2604.18713v1)**  
  Authors: Cuiling Sun, Linkai Peng, Adam Murphy, Elif Keles, Hiten D. Patel, Ashley Ross, Frank Miller, Baris Turkbey, Andrea Mia Bejar, Halil Ertugrul Aktas, Gorkem Durak, Ulas Bagci  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2604.18713v1.pdf) | [![GitHub](https://img.shields.io/github/stars/NUBagciLab/Prostate-Lesion-Segmentation?style=social)](https://github.com/NUBagciLab/Prostate-Lesion-Segmentation)  
  <details><summary>Abstract</summary>

  Automated 3D segmentation of prostate lesions from biparametric MRI (bp-MRI) is essential for reliable algorithmic analysis, but achieving high precision remains challenging. Volumetric methods must combine multiple modalities while ensuring anatomical consistency, but current models struggle to integrate cross-modal information reliably. While vision-language models (VLMs) are replacing the currently used architectural designs, they still lack the fine-grained, lesion-level semantics required for effective localized guidance. To address these limitations, we propose a new multi-encoder U-Net architecture incorporating three key innovations: (1) an alignment loss that enhances foreground text-image similarity to inject lesion semantics; (2) a heatmap loss that calibrates the similarity map and suppresses spurious background activations; and (3) a final-stage, confidence-gated multi-head cross-attention refiner that performs localized boundary edits in high-confidence regions. A phase-scheduled training regime stabilizes the optimization of these components. Our method consistently outperforms prior approaches, establishing a new state-of-the-art on the PI-CAI dataset through enhanced multi-modal fusion...
  </details>  
- **[Asset Harvester: Extracting 3D Assets from Autonomous Driving Logs for Simulation](https://arxiv.org/abs/2604.18468v1)**  
  Authors: Tianshi Cao, Jiawei Ren, Yuxuan Zhang, Jaewoo Seo, Jiahui Huang, Shikhar Solanki, Haotian Zhang, Mingfei Guo, Haithem Turki, Muxingzi Li, Yue Zhu, Sipeng Zhang, Zan Gojcic, Sanja Fidler, Kangxue Yin  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2604.18468v1.pdf)  
  <details><summary>Abstract</summary>

  Closed-loop simulation is a core component of autonomous vehicle (AV) development, enabling scalable testing, training, and safety validation before real-world deployment. Neural scene reconstruction converts driving logs into interactive 3D environments for simulation, but it does not produce complete 3D object assets required for agent manipulation and large-viewpoint novel-view synthesis. To address this challenge, we present Asset Harvester, an image-to-3D model and end-to-end pipeline that converts sparse, in-the-wild object observations from real driving logs into complete, simulation-ready assets. Rather than relying on a single model component, we developed a system-level design for real-world AV data that combines large-scale curation of object-centric training tuples, geometry-aware preprocessing across heterogeneous sensors, and a robust training recipe that couples sparse-view-conditioned multiview generation with 3D Gaussian lifting. Within this system, SparseViewDiT is explicitly designed to address limited-angle views and other real-world data challenges. Together with hybrid data curation, augmentation, and self-distillation, this system enables scalable...
  </details>  
  Keywords: image-to-3d  
- **[E3VS-Bench: A Benchmark for Viewpoint-Dependent Active Perception in 3D Gaussian Splatting Scenes](https://arxiv.org/abs/2604.17969v2)**  
  Authors: Koya Sakamoto, Taiki Miyanishi, Daichi Azuma, Shuhei Kurita, Shu Morikuni, Naoya Chiba, Motoaki Kawanabe, Yusuke Iwasawa, Yutaka Matsuo  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2604.17969v2.pdf)  
  <details><summary>Abstract</summary>

  Visual search in 3D environments requires embodied agents to actively explore their surroundings and acquire task-relevant evidence. However, existing visual search and embodied AI benchmarks, including EQA, typically rely on static observations or constrained egocentric motion, and thus do not explicitly evaluate fine-grained viewpoint-dependent phenomena that arise under unrestricted 5-DoF viewpoint control in real-world 3D environments, such as visibility changes caused by vertical viewpoint shifts, revealing contents inside containers, and disambiguating object attributes that are only observable from specific angles. To address this limitation, we introduce {E3VS-Bench}, a benchmark for embodied 3D visual search where agents must control their viewpoints in 5-DoF to gather viewpoint-dependent evidence for question answering. E3VS-Bench consists of 99 high-fidelity 3D scenes reconstructed using 3D Gaussian Splatting and 2,014 question-driven episodes. 3D Gaussian Splatting enables photorealistic free-viewpoint rendering that preserves fine-grained visual details (e.g., small text and subtle attributes) often degraded in mesh-based simulators, thereby allowing...
  </details>  
- **[ZSG-IAD: A Multimodal Framework for Zero-Shot Grounded Industrial Anomaly Detection](https://arxiv.org/abs/2604.17949v1)**  
  Authors: Qiuhui Chen, Jiaxiang Song, Shuai Tan, Weimin Zhong  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2604.17949v1.pdf)  
  <details><summary>Abstract</summary>

  Deep learning-based industrial anomaly detectors often behave as black boxes, making it hard to justify decisions with physically meaningful defect evidence. We propose ZSG-IAD, a multimodal vision-language framework for zero-shot grounded industrial anomaly detection. Given RGB images, sensor images, and 3D point clouds, ZSG-IAD generates structured anomaly reports and pixel-level anomaly masks. ZSG-IAD introduces a language-guided two-hop grounding module: (1) anomaly-related sentences select evidence-like latent slots distilled from multimodal features, yielding coarse spatial support; (2) selected slots modulate feature maps via channel-spatial gating and a lightweight decoder to produce fine-grained masks. To improve reliability, we further apply Executable-Rule GRPO with verifiable rewards to promote structured outputs, anomaly-region consistency, and reasoning-conclusion coherence. Experiments across multiple industrial anomaly benchmarks show strong zero-shot performance and more transparent, physically grounded explanations than prior methods. We will release code and annotations to support future research on trustworthy industrial anomaly detection systems.
  </details>  
- **[View-Consistent 3D Scene Editing via Dual-Path Structural Correspondense and Semantic Continuity](https://arxiv.org/abs/2604.17801v2)**  
  Authors: Pufan Li, Bi'an Du, Shenghe Zheng, Junyi Yao, Wei Hu  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2604.17801v2.pdf)  
  <details><summary>Abstract</summary>

  Text-driven 3D scene editing has recently attracted increasing attention. Most existing methods follow a render-edit-optimize pipeline, where multi-view images are rendered from a 3D scene, edited with 2D image editors, and then used to optimize the underlying 3D representation. However, cross-view inconsistency remains a major bottleneck. Although recent methods introduce geometric cues, cross-view interactions, or video priors to mitigate this issue, they still largely rely on inference-time synchronization and thus remain limited in robustness and generalization.In this work, we recast multi-view consistent 3D editing from a distributional perspective: 3D scene editing essentially requires a joint distribution modeling across viewpoints.Based on this insight, we propose a view-consistent 3D editing framework that explicitly introduces cross-view dependencies into the editing process. Furthermore, motivated by the observation that structural correspondence and semantic continuity rely on different cross-view cues, we introduce a dual-path consistency mechanism consisting of projection-guided structural guidance and patch-level semantic propagation for...
  </details>  
- **[MetaEarth3D: Unlocking World-scale 3D Generation with Spatially Scalable Generative Modeling](https://arxiv.org/abs/2604.22828v1)**  
  Authors: Jinqi Cao, Zhiping Yu, Baihong Lin, Chenyang Liu, Zhenwei Shi, Zhengxia Zou  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2604.22828v1.pdf)  
  <details><summary>Abstract</summary>

  Recent generative AI models have achieved remarkable breakthroughs in language and visual understanding. However, although these models can generate realistic visual content, their spatial scale remains confined to bounded environments, preventing them from capturing how geographic environments evolve across thousands of kilometers or from modeling the spatial structure of the large-scale physical world. This limitation poses a critical challenge for ultra-wide-area spatial intelligence in Earth observation and simulation, revealing a deeper gap in generative AI: progress has relied primarily on scaling model parameters and training data, while overlooking spatial scale as a core dimension of intelligence. Here, motivated by this missing dimension, we investigate spatial scale as a new scaling axis in foundation models and present MetaEarth3D, the first generative foundation model capable of spatially consistent generation at the planetary scale. Taking optical Earth observation simulation as a testbed, MetaEarth3D enables the generation of multi-level, unbounded, and diverse 3D scenes...
  </details>  
- **[UniMesh: Unifying 3D Mesh Understanding and Generation](https://arxiv.org/abs/2604.17472v1)**  
  Authors: Peng Huang, Yifeng Chen, Zeyu Zhang, Hao Tang  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2604.17472v1.pdf) | [![GitHub](https://img.shields.io/github/stars/AIGeeksGroup/UniMesh?style=social)](https://github.com/AIGeeksGroup/UniMesh) | [![Project](https://img.shields.io/badge/-Project-blue)](https://aigeeksgroup.github.io/UniMesh)  
  <details><summary>Abstract</summary>

  Recent advances in 3D vision have led to specialized models for either 3D understanding (e.g., shape classification, segmentation, reconstruction) or 3D generation (e.g., synthesis, completion, and editing). However, these tasks are often tackled in isolation, resulting in fragmented architectures and representations that hinder knowledge transfer and holistic scene modeling. To address these challenges, we propose UniMesh, a unified framework that jointly learns 3D generation and understanding within a single architecture. First, we introduce a novel Mesh Head that acts as a cross model interface, bridging diffusion based image generation with implicit shape decoders. Second, we develop Chain of Mesh (CoM), a geometric instantiation of iterative reasoning that enables user driven semantic mesh editing through a closed loop latent, prompting, and re generation cycle. Third, we incorporate a self reflection mechanism based on an Actor Evaluator Self reflection triad to diagnose and correct failures in high level tasks like 3D captioning....
  </details>  
- **[A Rapid Deployment Pipeline for Autonomous Humanoid Grasping Based on Foundation Models](https://arxiv.org/abs/2604.17258v1)**  
  Authors: Yifei Yan, Yankai Liao, Linqi Ye  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2604.17258v1.pdf)  
  <details><summary>Abstract</summary>

  Deploying a humanoid robot to manipulate a new object has traditionally required one to two days of effort: data collection, manual annotation, 3D model acquisition, and model training. This paper presents an end-to-end rapid deployment pipeline that integrates three foundation-model components to shorten the onboarding cycle for a new object to approximately 30 minutes: (i) Roboflow-based automatic annotation to assist in training a YOLOv8 object detector; (ii) 3D reconstruction based on Meta SAM 3D, which eliminates the need for a dedicated laser scanner; and (iii) zero-shot 6-DoF pose tracking based on FoundationPose, using the SAM~3D-generated mesh directly as the template. The estimated pose drives a Unity-based inverse kinematics planner, whose joint commands are streamed via UDP to a Unitree~G1 humanoid and executed through the Unitree SDK. We demonstrate detection accuracy of mAP@0.5 = 0.995, pose tracking precision of $σ< 1.05$ mm, and successful grasping on a real robot at five...
  </details>  
- **[Instant Colorization of Gaussian Splats](https://arxiv.org/abs/2604.17155v1)**  
  Authors: Daniel Lieber, Alexander Mock, Nils Wandel  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2604.17155v1.pdf)  
  <details><summary>Abstract</summary>

  Gaussian Splatting has recently become one of the most popular frameworks for photorealistic 3D scene reconstruction and rendering. While current rasterizers allow for efficient mappings of 3D Gaussian splats onto 2D camera views, this work focuses on mapping 2D image information (e.g. color, neural features or segmentation masks) efficiently back onto an existing scene of Gaussian splats. This 'opposite' direction enables applications ranging from scene relighting and stylization to 3D semantic segmentation, but also introduces challenges, such as view-dependent colorization and occlusion handling.   Our approach tackles these challenges using the normal equation to solve a visibility-weighted least squares problem for every Gaussian and can be implemented efficiently with existing differentiable rasterizers. We demonstrate the effectiveness of our approach on scene relighting, feature enrichment and 3D semantic segmentation tasks, achieving up to an order of magnitude speedup compared to gradient descent-based baselines.
  </details>  
  Keywords: 3d scene reconstruction  
- **[LAGS: Low-Altitude Gaussian Splatting with Groupwise Heterogeneous Graph Learning](https://arxiv.org/abs/2604.16910v1)**  
  Authors: Yikun Wang, Yujie Wan, Wei Zuo, Shuai Wang, Yik-Chung Wu, Chengzhong Xu, Huseyin Arslan  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2604.16910v1.pdf)  
  <details><summary>Abstract</summary>

  Low-altitude Gaussian splatting (LAGS) facilitates 3D scene reconstruction by aggregating aerial images from distributed drones. However, as LAGS prioritizes maximizing reconstruction quality over communication throughput, existing low-altitude resource allocation schemes become inefficient. This inefficiency stems from their failure to account for image diversity introduced by varying viewpoints. To fill this gap, we propose a groupwise heterogeneous graph neural network (GW-HGNN) for LAGS resource allocation. GW-HGNN explicitly models the non-uniform contribution of different image groups to the reconstruction process, thus automatically balancing data fidelity and transmission cost. The key insight of GW-HGNN is to transform LAGS losses and communication constraints into graph learning costs for dual-level message passing. Experiments on real-world LAGS datasets demonstrate that GW-HGNN significantly outperforms state-of-the-art benchmarks across key rendering metrics, including PSNR, SSIM, and LPIPS. Furthermore, GW-HGNN reduces computational latency by approximately 100x compared to the widely-used MOSEK solver, achieving millisecond-level inference suitable for real-time deployment.
  </details>  
  Keywords: 3d scene reconstruction  
- **[Repurposing 3D Generative Model for Autoregressive Layout Generation](https://arxiv.org/abs/2604.16299v1)**  
  Authors: Haoran Feng, Yifan Niu, Zehuan Huang, Yang-Tian Sun, Chunchao Guo, Yuxin Peng, Lu Sheng  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2604.16299v1.pdf) | [![GitHub](https://img.shields.io/github/stars/fenghora/LaviGen?style=social)](https://github.com/fenghora/LaviGen)  
  <details><summary>Abstract</summary>

  We introduce LaviGen, a framework that repurposes 3D generative models for 3D layout generation. Unlike previous methods that infer object layouts from textual descriptions, LaviGen operates directly in the native 3D space, formulating layout generation as an autoregressive process that explicitly models geometric relations and physical constraints among objects, producing coherent and physically plausible 3D scenes. To further enhance this process, we propose an adapted 3D diffusion model that integrates scene, object, and instruction information and employs a dual-guidance self-rollout distillation mechanism to improve efficiency and spatial accuracy. Extensive experiments on the LayoutVLM benchmark show LaviGen achieves superior 3D layout generation performance, with 19% higher physical plausibility than the state of the art and 65% faster computation. Our code is publicly available at https://github.com/fenghora/LaviGen.
  </details>  
- **[Beyond Prompts: Unconditional 3D Inversion for Out-of-Distribution Shapes](https://arxiv.org/abs/2604.14914v1)**  
  Authors: Victoria Yue Chen, Emery Pierson, Léopold Maillard, Maks Ovsjanikov  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2604.14914v1.pdf) | [![Project](https://img.shields.io/badge/-Project-blue)](https://daidedou.sorpi.fr/publication/beyondprompts)  
  <details><summary>Abstract</summary>

  Text-driven inversion of generative models is a core paradigm for manipulating 2D or 3D content, unlocking numerous applications such as text-based editing, style transfer, or inverse problems. However, it relies on the assumption that generative models remain sensitive to natural language prompts. We demonstrate that for state-of-the-art native text-to-3D generative models, this assumption often collapses. We identify a critical failure mode where generation trajectories are drawn into latent ``sink traps'': regions where the model becomes insensitive to prompt modifications. In these regimes, changes to the input text fail to alter internal representations in a way that alters the output geometry. Crucially, we observe that this is not a limitation of the model's \textit{geometric} expressivity; the same generative models possess the ability to produce a vast diversity of shapes but, as we demonstrate, become insensitive to out-of-distribution \textit{text} guidance. We investigate this behavior by analyzing the sampling trajectories of the generative...
  </details>  
  Keywords: text-to-3d  
- **[One-shot Compositional 3D Head Avatars with Deformable Hair](https://arxiv.org/abs/2604.14782v1)**  
  Authors: Yuan Sun, Xuan Wang, WeiLi Zhang, Wenxuan Zhang, Yu Guo, Fei Wang  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2604.14782v1.pdf)  
  <details><summary>Abstract</summary>

  We propose a compositional method for constructing a complete 3D head avatar from a single image. Prior one-shot holistic approaches frequently fail to produce realistic hair dynamics during animation, largely due to inadequate decoupling of hair from the facial region, resulting in entangled geometry and unnatural deformations. Our method explicitly decouples hair from the face, modeling these components using distinct deformation paradigms while integrating them into a unified rendering pipeline. Furthermore, by leveraging image-to-3D lifting techniques, we preserve fine-grained textures from the input image to the greatest extent possible, effectively mitigating the common issue of high-frequency information loss in generalized models. Specifically, given a frontal portrait image, we first perform hair removal to obtain a bald image. Both the original image and the bald image are then lifted to dense, detail-rich 3D Gaussian Splatting (3DGS) representations. For the bald 3DGS, we rig it to a FLAME mesh via non-rigid registration...
  </details>  
  Keywords: image-to-3d  
- **[Rethinking Image-to-3D Generation with Sparse Queries: Efficiency, Capacity, and Input-View Bias](https://arxiv.org/abs/2604.13905v1)**  
  Authors: Zhiyuan Xu, Jiuming Liu, Yuxin Chen, Masayoshi Tomizuka, Chenfeng Xu, Chensheng Peng  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2604.13905v1.pdf)  
  <details><summary>Abstract</summary>

  We present SparseGen, a novel framework for efficient image-to-3D generation, which exhibits low input-view bias while being significantly faster. Unlike traditional approaches that rely on dense volumetric grids, triplanes, or pixel-aligned primitives, we model scenes with a compact sparse set of learned 3D anchor queries and a learned expansion operator that decodes each transformed query into a small local set of 3D Gaussian primitives. Trained under a rectified-flow reconstruction objective without 3D supervision, our model learns to allocate representation capacity where geometry and appearance matter, achieving significant reductions in memory and inference time while preserving multi-view fidelity. We introduce quantitative measures of input-view bias and utilization to show that sparse queries reduce overfitting to conditioning views while being representationally efficient. Our results argue that sparse set-latent expansion is a principled, practical alternative for efficient 3D generative modeling.
  </details>  
  Keywords: image-to-3d  
- **[ClipGStream: Clip-Stream Gaussian Splatting for Any Length and Any Motion Multi-View Dynamic Scene Reconstruction](https://arxiv.org/abs/2604.13746v1)**  
  Authors: Jie Liang, Jiahao Wu, Chao Wang, Jiayu Yang, Xiaoyun Zheng, Kaiqiang Xiong, Zhanke Wang, Jinbo Yan, Feng Gao, Ronggang Wang  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2604.13746v1.pdf) | [![Project](https://img.shields.io/badge/-Project-blue)](https://liangjie1999.github.io/ClipGStreamWeb)  
  <details><summary>Abstract</summary>

  Dynamic 3D scene reconstruction is essential for immersive media such as VR, MR, and XR, yet remains challenging for long multi-view sequences with large-scale motion. Existing dynamic Gaussian approaches are either Frame-Stream, offering scalability but poor temporal stability, or Clip, achieving local consistency at the cost of high memory and limited sequence length. We propose ClipGStream, a hybrid reconstruction framework that performs stream optimization at the clip level rather than the frame level. The sequence is divided into short clips, where dynamic motion is modeled using clip-independent spatio-temporal fields and residual anchor compensation to capture local variations efficiently, while inter-clip inherited anchors and decoders maintain structural consistency across clips. This Clip-Stream design enables scalable, flicker-free reconstruction of long dynamic videos with high temporal coherence and reduced memory overhead. Extensive experiments demonstrate that ClipGStream achieves state-of-the-art reconstruction quality and efficiency. The project page is available at: https://liangjie1999.github.io/ClipGStreamWeb/
  </details>  
  Keywords: 3d scene reconstruction  
- **[ReConText3D: Replay-based Continual Text-to-3D Generation](https://arxiv.org/abs/2604.13730v1)**  
  Authors: Muhammad Ahmed Ullah Khan, Muhammad Haris Bin Amir, Didier Stricker, Muhammad Zeshan Afzal  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2604.13730v1.pdf) | [![Project](https://img.shields.io/badge/-Project-blue)](https://mauk95.github.io/ReConText3D)  
  <details><summary>Abstract</summary>

  Continual learning enables models to acquire new knowledge over time while retaining previously learned capabilities. However, its application to text-to-3D generation remains unexplored. We present ReConText3D, the first framework for continual text-to-3D generation. We first demonstrate that existing text-to-3D models suffer from catastrophic forgetting under incremental training. ReConText3D enables generative models to incrementally learn new 3D categories from textual descriptions while preserving the ability to synthesize previously seen assets. Our method constructs a compact and diverse replay memory through text-embedding k-Center selection, allowing representative rehearsal of prior knowledge without modifying the underlying architecture. To systematically evaluate continual text-to-3D learning, we introduce Toys4K-CL, a benchmark derived from the Toys4K dataset that provides balanced and semantically diverse class-incremental splits. Extensive experiments on the Toys4K-CL benchmark show that ReConText3D consistently outperforms all baselines across different generative backbones, maintaining high-quality generation for both old and new classes. To the best of our knowledge, this...
  </details>  
  Keywords: text-to-3d  
- **[Beyond Voxel 3D Editing: Learning from 3D Masks and Self-Constructed Data](https://arxiv.org/abs/2604.13688v1)**  
  Authors: Yizhao Xu, Hongyuan Zhu, Caiyun Liu, Tianfu Wang, Keyu Chen, Sicheng Xu, Jiaolong Yang, Nicholas Jing Yuan, Qi Zhang  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2604.13688v1.pdf)  
  <details><summary>Abstract</summary>

  3D editing refers to the ability to apply local or global modifications to 3D assets. Effective 3D editing requires maintaining semantic consistency by performing localized changes according to prompts, while also preserving local invariance so that unchanged regions remain consistent with the original. However, existing approaches have significant limitations: multi-view editing methods incur losses when projecting back to 3D, while voxel-based editing is constrained in both the regions that can be modified and the scale of modifications. Moreover, the lack of sufficiently large editing datasets for training and evaluation remains a challenge. To address these challenges, we propose a Beyond Voxel 3D Editing (BVE) framework with a self-constructed large-scale dataset specifically tailored for 3D editing. Building upon this dataset, our model enhances a foundational image-to-3D generative architecture with lightweight, trainable modules, enabling efficient injection of textual semantics without the need for expensive full-model retraining. Furthermore, we introduce an annotation-free 3D...
  </details>  
  Keywords: image-to-3d  
- **[Cross-Attentive Multiview Fusion of Vision-Language Embeddings](https://arxiv.org/abs/2604.12551v1)**  
  Authors: Tomas Berriel Martins, Martin R. Oswald, Javier Civera  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2604.12551v1.pdf)  
  <details><summary>Abstract</summary>

  Vision-language models have been key to the development of open-vocabulary 2D semantic segmentation. Lifting these models from 2D images to 3D scenes, however, remains a challenging problem. Existing approaches typically back-project and average 2D descriptors across views, or heuristically select a single representative one, often resulting in suboptimal 3D representations. In this work, we introduce a novel multiview transformer architecture that cross-attends across vision-language descriptors from multiple viewpoints and fuses them into a unified per-3D-instance embedding. As a second contribution, we leverage multiview consistency as a self-supervision signal for this fusion, which significantly improves performance when added to a standard supervised target-class loss. Our Cross-Attentive Multiview Fusion, which we denote with its acronym CAMFusion, not only consistently outperforms naive averaging or single-view descriptor selection, but also achieves state-of-the-art results on 3D semantic and instance classification benchmarks, including zero-shot evaluations on out-of-domain datasets.
  </details>  
- **[ReefMapGS: Enabling Large-Scale Underwater Reconstruction by Closing the Loop Between Multimodal SLAM and Gaussian Splatting](https://arxiv.org/abs/2604.11992v1)**  
  Authors: Daniel Yang, Jungseok Hong, John J. Leonard, Yogesh Girdhar  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2604.11992v1.pdf)  
  <details><summary>Abstract</summary>

  3D Gaussian Splatting is a powerful visual representation, providing high-quality and efficient 3D scene reconstruction, but it is crucially dependent on accurate camera poses typically obtained from computationally intensive processes like structure-from-motion that are unsuitable for field robot applications. However, in these domains, multimodal sensor data from acoustic, inertial, pressure, and visual sensors are available and suitable for pose-graph optimization-based SLAM methods that can estimate the vehicle's trajectory and thus our needed camera poses while providing uncertainty. We propose a 3DGS-based incremental reconstruction framework, ReefMapGS, that builds an initial model from a high certainty region and progressively expands to incorporate the whole scene. We reconstruct the scene incrementally by interleaving local tracking of new image observations with optimization of the underlying 3DGS scene. These refined poses are integrated back into the pose-graph to globally optimize the whole trajectory. We show COLMAP-free 3D reconstruction of two underwater reef sites with complex...
  </details>  
  Keywords: 3d scene reconstruction  
- **[Efficient Transceiver Design for Aerial Image Transmission and Large-scale Scene Reconstruction](https://arxiv.org/abs/2604.11098v2)**  
  Authors: Zeyi Ren, Jialin Dong, Wei Zuo, Yikun Wang, Bingyang Cheng, Sheng Zhou, Zhisheng Niu  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2604.11098v2.pdf)  
  <details><summary>Abstract</summary>

  Large-scale three-dimensional (3D) scene reconstruction in low-altitude intelligent networks (LAIN) demands highly efficient wireless image transmission. However, existing schemes struggle to balance severe pilot overhead with the transmission accuracy required to maintain reconstruction fidelity. To strike a balance between efficiency and reliability, this paper proposes a novel deep learning-based end-to-end (E2E) transceiver design that integrates 3D Gaussian Splatting (3DGS) directly into the training process. By jointly optimizing the communication modules via the combined 3DGS rendering loss, our approach explicitly improves scene recovery quality. Furthermore, this task-driven framework enables the use of a sparse pilot scheme, significantly reducing transmission overhead while maintaining robust image recovery under low-altitude channel conditions. Extensive experiments on real-world aerial image datasets demonstrate that the proposed E2E design significantly outperforms existing baselines, delivering superior transmission performance and accurate 3D scene reconstructions.
  </details>  
  Keywords: 3d scene reconstruction  
- **[AffordGen: Generating Diverse Demonstrations for Generalizable Object Manipulation with Afford Correspondence](https://arxiv.org/abs/2604.10579v2)**  
  Authors: Jiawei Zhang, Kaizhe Hu, Yingqian Huang, Yuanchen Ju, Zhengrong Xue, Huazhe Xu  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2604.10579v2.pdf) | [![Project](https://img.shields.io/badge/-Project-blue)](https://jiaweiz9.github.io/AffordGen-release)  
  <details><summary>Abstract</summary>

  Despite the recent success of modern imitation learning methods in robot manipulation, their performance is often constrained by geometric variations due to limited data diversity. Leveraging powerful 3D generative models and vision foundation models (VFMs), the proposed AffordGen framework overcomes this limitation by utilizing the semantic correspondence of meaningful keypoints across large-scale 3D meshes to generate new robot manipulation trajectories. This large-scale, affordance-aware dataset is then used to train a robust, closed-loop visuomotor policy, combining the semantic generalizability of affordances with the reactive robustness of end-to-end learning. Experiments in simulation and the real world show that policies trained with AffordGen achieve high success rates and enable zero-shot generalization to truly unseen objects, significantly improving data efficiency in robot learning. Project Page: https://jiaweiz9.github.io/AffordGen-release/
  </details>  
- **[Adapting 2D Multi-Modal Large Language Model for 3D CT Image Analysis](https://arxiv.org/abs/2604.10233v1)**  
  Authors: Yang Yu, Dunyuan Xu, Yaoqian Li, Xiaomeng Li, Jinpeng Li, Pheng-Ann Heng  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2604.10233v1.pdf)  
  <details><summary>Abstract</summary>

  3D medical image analysis is of great importance in disease diagnosis and treatment. Recently, multimodal large language models (MLLMs) have exhibited robust perceptual capacity, strong cross-modal alignment, and promising generalizability. Therefore, they have great potential to improve the performance of medical report generation (MRG) and medical visual question answering (MVQA), which serve as two important tasks in clinical scenarios. However, due to the scarcity of 3D medical images, existing 3D medical MLLMs suffer from insufficiently pretrained vision encoder and inability to extract customized image features for different kinds of tasks. In this paper, we propose to first transfer a 2D MLLM, which is well trained with 2D natural images, to support 3D medical volumetric inputs while reusing all of its pre-trained parameters. To enable the vision encoder to extract tailored image features for various tasks, we then design a Text-Guided Hierarchical MoE (TGH-MoE) framework, which can distinguish tasks under the...
  </details>  
- **[Hitem3D 2.0: Multi-View Guided Native 3D Texture Generation](https://arxiv.org/abs/2604.09231v1)**  
  Authors: Huiang He, Shengchu Zhao, Jianwen Huang, Jie Li, Jiaqi Wu, Hu Zhang, Pei Tang, Heliang Zheng, Yukun Li, Rongfei Jia  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2604.09231v1.pdf)  
  <details><summary>Abstract</summary>

  Although recent advances have improved the quality of 3D texture generation, existing methods still struggle with incomplete texture coverage, cross-view inconsistency, and misalignment between geometry and texture. To address these limitations, we propose Hitem3D 2.0, a multi-view guided native 3D texture generation framework that enhances texture quality through the integration of 2D multi-view generation priors and native 3D texture representations. Hitem3D 2.0 comprises two key components: a multi-view synthesis framework and a native 3D texture generation model. The multi-view generation is built upon a pre-trained image editing backbone and incorporates plug-and-play modules that explicitly promote geometric alignment, cross-view consistency, and illumination uniformity, thereby enabling the synthesis of high-fidelity multi-view images. Conditioned on the generated views and 3D geometry, the native 3D texture generation model projects multi-view textures onto 3D surfaces while plausibly completing textures in unseen regions. Through the integration of multi-view consistency constraints with native 3D texture modeling, Hitem3D...
  </details>  
  Keywords: multi-view synthesis  
- **[Physically Grounded 3D Generative Reconstruction under Hand Occlusion using Proprioception and Multi-Contact Touch](https://arxiv.org/abs/2604.09100v1)**  
  Authors: Gabriele Mario Caddeo, Pasquale Marra, Lorenzo Natale  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2604.09100v1.pdf)  
  <details><summary>Abstract</summary>

  We propose a multimodal, physically grounded approach for metric-scale amodal object reconstruction and pose estimation under severe hand occlusion. Unlike prior occlusion-aware 3D generation methods that rely only on vision, we leverage physical interaction signals: proprioception provides the posed hand geometry, and multi-contact touch constrains where the object surface must lie, reducing ambiguity in occluded regions. We represent object structure as a pose-aware, camera-aligned signed distance field (SDF) and learn a compact latent space with a Structure-VAE. In this latent space, we train a conditional flow-matching diffusion model, pretraining on vision-only images and finetuning on occluded manipulation scenes while conditioning on visible RGB evidence, occluder/visibility masks, the hand latent representation, and tactile information. Crucially, we incorporate physics-based objectives and differentiable decoder-guidance during finetuning and inference to reduce hand--object interpenetration and to align the reconstructed surface with contact observations. Because our method produces a metric, physically consistent structure estimate, it integrates...
  </details>  
- **[SIC3D: Style Image Conditioned Text-to-3D Gaussian Splatting Generation](https://arxiv.org/abs/2604.08760v1)**  
  Authors: Ming He, Zhixiang Chen, Steve Maddock  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2604.08760v1.pdf)  
  <details><summary>Abstract</summary>

  Recent progress in text-to-3D object generation enables the synthesis of detailed geometry from text input by leveraging 2D diffusion models and differentiable 3D representations. However, the approaches often suffer from limited controllability and texture ambiguity due to the limitation of the text modality. To address this, we present SIC3D, a controllable image-conditioned text-to-3D generation pipeline with 3D Gaussian Splatting (3DGS). There are two stages in SIC3D. The first stage generates the 3D object content from text with a text-to-3DGS generation model. The second stage transfers style from a reference image to the 3DGS. Within this stylization stage, we introduce a novel Variational Stylized Score Distillation (VSSD) loss to effectively capture both global and local texture patterns while mitigating conflicts between geometry and appearance. A scaling regularization is further applied to prevent the emergence of artifacts and preserve the pattern from the style image. Extensive experiments demonstrate that SIC3D enhances geometric...
  </details>  
  Keywords: text-to-3d  
- **[AniGen: Unified $S^3$ Fields for Animatable 3D Asset Generation](https://arxiv.org/abs/2604.08746v2)**  
  Authors: Yi-Hua Huang, Zi-Xin Zou, Yuting He, Chirui Chang, Cheng-Feng Pu, Ziyi Yang, Yuan-Chen Guo, Yan-Pei Cao, Xiaojuan Qi  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2604.08746v2.pdf) | [![Project](https://img.shields.io/badge/-Project-blue)](https://yihua7.github.io/AniGen-web)  
  <details><summary>Abstract</summary>

  Animatable 3D assets, defined as geometry equipped with an articulated skeleton and skinning weights, are fundamental to interactive graphics, embodied agents, and animation production. While recent 3D generative models can synthesize visually plausible shapes from images, the results are typically static. Obtaining usable rigs via post-hoc auto-rigging is brittle and often produces skeletons that are topologically inconsistent with the generated geometry. We present AniGen, a unified framework that directly generates animate-ready 3D assets conditioned on a single image. Our key insight is to represent shape, skeleton, and skinning as mutually consistent $S^3$ Fields (Shape, Skeleton, Skin) defined over a shared spatial domain. To enable the robust learning of these fields, we introduce two technical innovations: (i) a confidence-decaying skeleton field that explicitly handles the geometric ambiguity of bone prediction at Voronoi boundaries, and (ii) a dual skin feature field that decouples skinning weights from specific joint counts, allowing a fixed-architecture...
  </details>  
  Keywords: 3d asset generation, rigging  
- **[Scal3R: Scalable Test-Time Training for Large-Scale 3D Reconstruction](https://arxiv.org/abs/2604.08542v1)**  
  Authors: Tao Xie, Peishan Yang, Yudong Jin, Yingfeng Cai, Wei Yin, Weiqiang Ren, Qian Zhang, Wei Hua, Sida Peng, Xiaoyang Guo, Xiaowei Zhou  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2604.08542v1.pdf) | [![Project](https://img.shields.io/badge/-Project-blue)](https://zju3dv.github.io/scal3r)  
  <details><summary>Abstract</summary>

  This paper addresses the task of large-scale 3D scene reconstruction from long video sequences. Recent feed-forward reconstruction models have shown promising results by directly regressing 3D geometry from RGB images without explicit 3D priors or geometric constraints. However, these methods often struggle to maintain reconstruction accuracy and consistency over long sequences due to limited memory capacity and the inability to effectively capture global contextual cues. In contrast, humans can naturally exploit the global understanding of the scene to inform local perception. Motivated by this, we propose a novel neural global context representation that efficiently compresses and retains long-range scene information, enabling the model to leverage extensive contextual cues for enhanced reconstruction accuracy and consistency. The context representation is realized through a set of lightweight neural sub-networks that are rapidly adapted during test time via self-supervised objectives, which substantially increases memory capacity without incurring significant computational overhead. The experiments on multiple...
  </details>  
  Keywords: 3d scene reconstruction  
- **[SurfelSplat: Learning Efficient and Generalizable Gaussian Surfel Representations for Sparse-View Surface Reconstruction](https://arxiv.org/abs/2604.08370v1)**  
  Authors: Chensheng Dai, Shengjun Zhang, Min Chen, Yueqi Duan  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2604.08370v1.pdf)  
  <details><summary>Abstract</summary>

  3D Gaussian Splatting (3DGS) has demonstrated impressive performance in 3D scene reconstruction. Beyond novel view synthesis, it shows great potential for multi-view surface reconstruction. Existing methods employ optimization-based reconstruction pipelines that achieve precise and complete surface extractions. However, these approaches typically require dense input views and high time consumption for per-scene optimization. To address these limitations, we propose SurfelSplat, a feed-forward framework that generates efficient and generalizable pixel-aligned Gaussian surfel representations from sparse-view images. We observe that conventional feed-forward structures struggle to recover accurate geometric attributes of Gaussian surfels because the spatial frequency of pixel-aligned primitives exceeds Nyquist sampling rates. Therefore, we propose a cross-view feature aggregation module based on the Nyquist sampling theorem. Specifically, we first adapt the geometric forms of Gaussian surfels with spatial sampling rate-guided low-pass filters. We then project the filtered surfels across all input views to obtain cross-view feature correlations. By processing these correlations through...
  </details>  
  Keywords: 3d scene reconstruction, novel view synthesis  
- **[Brain3D: EEG-to-3D Decoding of Visual Representations via Multimodal Reasoning](https://arxiv.org/abs/2604.08068v1)**  
  Authors: Emanuele Balloni, Emanuele Frontoni, Chiara Matti, Marina Paolanti, Roberto Pierdicca, Emiliano Santarnecchi  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2604.08068v1.pdf)  
  <details><summary>Abstract</summary>

  Decoding visual information from electroencephalography (EEG) has recently achieved promising results, primarily focusing on reconstructing two-dimensional (2D) images from brain activity. However, the reconstruction of three-dimensional (3D) representations remains largely unexplored. This limits the geometric understanding and reduces the applicability of neural decoding in different contexts. To address this gap, we propose Brain3D, a multimodal architecture for EEG-to-3D reconstruction based on EEG-to-image decoding. It progressively transforms neural representations into the 3D domain using geometry-aware generative reasoning. Our pipeline first produces visually grounded images from EEG signals, then employs a multimodal large language model to extract structured 3D-aware descriptions, which guide a diffusion-based generation stage whose outputs are finally converted into coherent 3D meshes via a single-image-to-3D model. By decomposing the problem into structured stages, the proposed approach avoids direct EEG-to-3D mappings and enables scalable brain-driven 3D generation. We conduct a comprehensive evaluation comparing the reconstructed 3D outputs against the original...
  </details>  
  Keywords: image-to-3d  
- **[A Semi-Automated Framework for 3D Reconstruction of Medieval Manuscript Miniatures](https://arxiv.org/abs/2604.08610v1)**  
  Authors: Riccardo Pallotto, Pierluigi Feliciati, Tiberio Uricchio  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2604.08610v1.pdf)  
  <details><summary>Abstract</summary>

  This paper presents a semi-automated framework for transforming two-dimensional miniatures from medieval manuscripts into three-dimensional digital models suitable for extended reality (XR), tactile 3D~printing, and web-based visualization. We evaluate seven image-to-3D methods (TripoSR, SF3D, SPAR3D, TRELLIS, Wonder3D, SAM~3D, Hi3DGen) on 69~manuscript figures from two collections using rendering-based metrics (Silhouette IoU, LPIPS, CLIP~Score) and volumetric measures (Depth Range Ratio, watertight percentage), revealing a trade-off between volumetric expansion and geometric fidelity. Hi3DGen balances topological quality with rich surface detail through its normal bridging approach, making it a good starting point for expert refinement. Our pipeline combines SAM segmentation, Hi3DGen mesh generation, expert refinement in ZBrush, and AI-assisted texturing. Two case studies on Gothic illuminations from the Decretum Gratiani (Vatican Library) and Renaissance miniatures by Giulio Clovio demonstrate applicability across artistic traditions. The resulting models can support WebXR visualization, AR overlay on physical manuscripts, and tactile 3D~prints for visually impaired users.
  </details>  
  Keywords: image-to-3d  
- **[Synthetic Dataset Generation for Partially Observed Indoor Objects](https://arxiv.org/abs/2604.07010v1)**  
  Authors: Jelle Vermandere, Maarten Bassier, Maarten Vergauwen  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2604.07010v1.pdf)  
  <details><summary>Abstract</summary>

  Learning-based methods for 3D scene reconstruction and object completion require large datasets containing partial scans paired with complete ground-truth geometry. However, acquiring such datasets using real-world scanning systems is costly and time-consuming, particularly when accurate ground truth for occluded regions is required. In this work, we present a virtual scanning framework implemented in Unity for generating realistic synthetic 3D scan datasets. The proposed system simulates the behaviour of real-world scanners using configurable parameters such as scan resolution, measurement range, and distance-dependent noise. Instead of directly sampling mesh surfaces, the framework performs ray-based scanning from virtual viewpoints, enabling realistic modelling of sensor visibility and occlusion effects. In addition, panoramic images captured at the scanner location are used to assign colours to the resulting point clouds. To support scalable dataset creation, the scanner is integrated with a procedural indoor scene generation pipeline that automatically produces diverse room layouts and furniture arrangements. Using...
  </details>  
  Keywords: 3d scene reconstruction  
- **[FORGE: Fine-grained Multimodal Evaluation for Manufacturing Scenarios](https://arxiv.org/abs/2604.07413v2)**  
  Authors: Xiangru Jian, Hao Xu, Wei Pang, Xinjian Zhao, Chengyu Tao, Qixin Zhang, Xikun Zhang, Chao Zhang, Guanzhi Deng, Alex Xue, Juan Du, Tianshu Yu, Garth Tarr, Linqi Song, Qiuzhuang Sun, Dacheng Tao  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2604.07413v2.pdf) | [![Project](https://img.shields.io/badge/-Project-blue)](https://ai4manufacturing.github.io/forge-web)  
  <details><summary>Abstract</summary>

  The manufacturing sector is increasingly adopting Multimodal Large Language Models (MLLMs) to transition from simple perception to autonomous execution, yet current evaluations fail to reflect the rigorous demands of real-world manufacturing environments. Progress is hindered by data scarcity and a lack of fine-grained domain semantics in existing datasets. To bridge this gap, we introduce FORGE. Wefirst construct a high-quality multimodal dataset that combines real-world 2D images and 3D point clouds, annotated with fine-grained domain semantics (e.g., exact model numbers). We then evaluate 18 state-of-the-art MLLMs across three manufacturing tasks, namely workpiece verification, structural surface inspection, and assembly verification, revealing significant performance gaps. Counter to conventional understanding, the bottleneck analysis shows that visual grounding is not the primary limiting factor. Instead, insufficient domain-specific knowledge is the key bottleneck, setting a clear direction for future research. Beyond evaluation, we show that our structured annotations can serve as an actionable training resource: supervised...
  </details>  
- **[LiveStre4m: Feed-Forward Live Streaming of Novel Views from Unposed Multi-View Video](https://arxiv.org/abs/2604.06740v1)**  
  Authors: Pedro Quesado, Erkut Akdag, Yasaman Kashefbahrami, Willem Menu, Egor Bondarev  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2604.06740v1.pdf) | [![GitHub](https://img.shields.io/github/stars/pedro-quesado/LiveStre4m?style=social)](https://github.com/pedro-quesado/LiveStre4m)  
  <details><summary>Abstract</summary>

  Live-streaming Novel View Synthesis (NVS) from unposed multi-view video remains an open challenge in a wide range of applications. Existing methods for dynamic scene representation typically require ground-truth camera parameters and involve lengthy optimizations ($\approx 2.67$s), which makes them unsuitable for live streaming scenarios. To address this issue, we propose a novel viewpoint video live-streaming method (LiveStre4m), a feed-forward model for real-time NVS from unposed sparse multi-view inputs. LiveStre4m introduces a multi-view vision transformer for keyframe 3D scene reconstruction coupled with a diffusion-transformer interpolation module that ensures temporal consistency and stable streaming. In addition, a Camera Pose Predictor module is proposed to efficiently estimate both poses and intrinsics directly from RGB images, removing the reliance on known camera calibration information. Our approach enables temporally consistent novel-view video streaming in real-time using as few as two synchronized unposed input streams. LiveStre4m attains an average reconstruction time of $ 0.07$s per-frame at...
  </details>  
  Keywords: 3d scene reconstruction, novel view synthesis  
- **[PoM: A Linear-Time Replacement for Attention with the Polynomial Mixer](https://arxiv.org/abs/2604.06129v1)**  
  Authors: David Picard, Nicolas Dufour, Lucas Degeorge, Arijit Ghosh, Davide Allegro, Tom Ravaud, Yohann Perron, Corentin Sautier, Zeynep Sonat Baltaci, Fei Meng, Syrine Kalleli, Marta López-Rauhut, Thibaut Loiseau, Ségolène Albouy, Raphael Baena, Elliot Vincent, Loic Landrieu  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2604.06129v1.pdf) | [![GitHub](https://img.shields.io/github/stars/davidpicard/pom?style=social)](https://github.com/davidpicard/pom)  
  <details><summary>Abstract</summary>

  This paper introduces the Polynomial Mixer (PoM), a novel token mixing mechanism with linear complexity that serves as a drop-in replacement for self-attention. PoM aggregates input tokens into a compact representation through a learned polynomial function, from which each token retrieves contextual information. We prove that PoM satisfies the contextual mapping property, ensuring that transformers equipped with PoM remain universal sequence-to-sequence approximators. We replace standard self-attention with PoM across five diverse domains: text generation, handwritten text recognition, image generation, 3D modeling, and Earth observation. PoM matches the performance of attention-based models while drastically reducing computational cost when working with long sequences. The code is available at https://github.com/davidpicard/pom.
  </details>  
- **[SEM-ROVER: Semantic Voxel-Guided Diffusion for Large-Scale Driving Scene Generation](https://arxiv.org/abs/2604.06113v1)**  
  Authors: Hiba Dahmani, Nathan Piasco, Moussab Bennehar, Luis Roldão, Dzmitry Tsishkou, Laurent Caraffa, Jean-Philippe Tarel, Roland Brémond  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2604.06113v1.pdf)  
  <details><summary>Abstract</summary>

  Scalable generation of outdoor driving scenes requires 3D representations that remain consistent across multiple viewpoints and scale to large areas. Existing solutions either rely on image or video generative models distilled to 3D space, harming the geometric coherence and restricting the rendering to training views, or are limited to small-scale 3D scene or object-centric generation. In this work, we propose a 3D generative framework based on $Σ$-Voxfield grid, a discrete representation where each occupied voxel stores a fixed number of colorized surface samples. To generate this representation, we train a semantic-conditioned diffusion model that operates on local voxel neighborhoods and uses 3D positional encodings to capture spatial structure. We scale to large scenes via progressive spatial outpainting over overlapping regions. Finally, we render the generated $Σ$-Voxfield grid with a deferred rendering module to obtain photorealistic images, enabling large-scale multiview-consistent 3D scene generation without per-scene optimization. Extensive experiments show that our...
  </details>  
- **[HiPolicy: Hierarchical Multi-Frequency Action Chunking for Policy Learning](https://arxiv.org/abs/2604.06067v1)**  
  Authors: Jiyao Zhang, Zimu Han, Junhan Wang, Xionghao Wu, Shihong Lin, Jinzhou Li, Hongwei Fan, Ruihai Wu, Dongjiang Li, Hao Dong  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2604.06067v1.pdf)  
  <details><summary>Abstract</summary>

  Robotic imitation learning faces a fundamental trade-off between modeling long-horizon dependencies and enabling fine-grained closed-loop control. Existing fixed-frequency action chunking approaches struggle to achieve both. Building on this insight, we propose HiPolicy, a hierarchical multi-frequency action chunking framework that jointly predicts action sequences at different frequencies to capture both coarse high-level plans and precise reactive motions. We extract and fuse hierarchical features from history observations aligned to each frequency for multi-frequency chunk generation, and introduce an entropy-guided execution mechanism that adaptively balances long-horizon planning with fine-grained control based on action uncertainty. Experiments on diverse simulated benchmarks and real-world manipulation tasks show that HiPolicy can be seamlessly integrated into existing 2D and 3D generative policies, delivering consistent improvements in performance while significantly enhancing execution efficiency.
  </details>  
- **[Part-Level 3D Gaussian Vehicle Generation with Joint and Hinge Axis Estimation](https://arxiv.org/abs/2604.05070v1)**  
  Authors: Shiyao Qian, Yuan Ren, Dongfeng Bai, Bingbing Liu  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2604.05070v1.pdf)  
  <details><summary>Abstract</summary>

  Simulation is essential for autonomous driving, yet current frameworks often model vehicles as rigid assets and fail to capture part-level articulation. With perception algorithms increasingly leveraging dynamics such as wheel steering or door opening, realistic simulation requires animatable vehicle representations. Existing CAD-based pipelines are limited by library coverage and fixed templates, preventing faithful reconstruction of in-the-wild instances. We propose a generative framework that, from a single image or sparse multi-view input, synthesizes an animatable 3D Gaussian vehicle. Our method addresses two challenges: (i) large 3D asset generators are optimized for static quality but not articulation, leading to distortions at part boundaries when animated; and (ii) segmentation alone cannot provide the kinematic parameters required for motion. To overcome this, we introduce a part-edge refinement module that enforces exclusive Gaussian ownership and a kinematic reasoning head that predicts joint positions and hinge axes of movable parts. Together, these components enable faithful part-aware...
  </details>  
- **[Immunizing 3D Gaussian Generative Models Against Unauthorized Fine-Tuning via Attribute-Space Traps](https://arxiv.org/abs/2604.09688v1)**  
  Authors: Jianwei Zhang, Sihan Cao, Chaoning Zhang, Ziming Hong, Jiaxin Huang, Pengcheng Zheng, Caiyan Qin, Wei Dong, Yang Yang, Tongliang Liu  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2604.09688v1.pdf)  
  <details><summary>Abstract</summary>

  Recent large-scale generative models enable high-quality 3D synthesis. However, the public accessibility of pre-trained weights introduces a critical vulnerability. Adversaries can fine-tune these models to steal specialized knowledge acquired during pre-training, leading to intellectual property infringement. Unlike defenses for 2D images and language models, 3D generators require specialized protection due to their explicit Gaussian representations, which expose fundamental structural parameters directly to gradient-based optimization. We propose GaussLock, the first approach designed to defend 3D generative models against fine-tuning attacks. GaussLock is a lightweight parameter-space immunization framework that integrates authorized distillation with attribute-aware trap losses targeting position, scale, rotation, opacity, and color. Specifically, these traps systematically collapse spatial distributions, distort geometric shapes, align rotational axes, and suppress primitive visibility to fundamentally destroy structural integrity. By jointly optimizing these dual objectives, the distillation process preserves fidelity on authorized tasks while the embedded traps actively disrupt unauthorized reconstructions. Experiments on large-scale Gaussian models...
  </details>  
  Keywords: 3d generator  
- **[HandDreamer: Zero-Shot Text to 3D Hand Model Generation using Corrective Hand Shape Guidance](https://arxiv.org/abs/2604.04425v1)**  
  Authors: Green Rosh, Prateek Kukreja, Vishakha SR, Pawan Prasad B H  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2604.04425v1.pdf)  
  <details><summary>Abstract</summary>

  The emergence of virtual reality has necessitated the generation of detailed and customizable 3D hand models for interaction in the virtual world. However, the current methods for 3D hand model generation are both expensive and cumbersome, offering very little customizability to the users. While recent advancements in zero-shot text-to-3D synthesis have enabled the generation of diverse and customizable 3D models using Score Distillation Sampling (SDS), they do not generalize very well to 3D hand model generation, resulting in unnatural hand structures, view-inconsistencies and loss of details. To address these limitations, we introduce HandDreamer, the first method for zero-shot 3D hand model generation from text prompts. Our findings suggest that view-inconsistencies in SDS is primarily caused due to the ambiguity in the probability landscape described by the text prompt, resulting in similar views converging to different modes of the distribution. This is particularly aggravated for hands due to the large variations...
  </details>  
  Keywords: text-to-3d  
- **[CRAFT: Video Diffusion for Bimanual Robot Data Generation](https://arxiv.org/abs/2604.03552v1)**  
  Authors: Jason Chen, I-Chun Arthur Liu, Gaurav Sukhatme, Daniel Seita  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2604.03552v1.pdf) | [![Project](https://img.shields.io/badge/-Project-blue)](https://craftaug.github.io)  
  <details><summary>Abstract</summary>

  Bimanual robot learning from demonstrations is fundamentally limited by the cost and narrow visual diversity of real-world data, which constrains policy robustness across viewpoints, object configurations, and embodiments. We present Canny-guided Robot Data Generation using Video Diffusion Transformers (CRAFT), a video diffusion-based framework for scalable bimanual demonstration generation that synthesizes temporally coherent manipulation videos while producing action labels. By conditioning video diffusion on edge-based structural cues extracted from simulator-generated trajectories, CRAFT produces physically plausible trajectory variations and supports a unified augmentation pipeline spanning object pose changes, camera viewpoints, lighting and background variations, cross-embodiment transfer, and multi-view synthesis. We leverage a pre-trained video diffusion model to convert simulated videos, along with action labels from the simulation trajectories, into action-consistent demonstrations. Starting from only a few real-world demonstrations, CRAFT generates a large, visually diverse set of photorealistic training data, bypassing the need to replay demonstrations on the real robot (Sim2Real). Across simulated...
  </details>  
  Keywords: multi-view synthesis  
- **[Adaptive Local Frequency Filtering for Fourier-Encoded Implicit Neural Representations](https://arxiv.org/abs/2604.02846v2)**  
  Authors: Ligen Shi, Jun Qiu, Yuhang Zheng, Zengyu Pang, Chang Liu  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2604.02846v2.pdf)  
  <details><summary>Abstract</summary>

  Fourier-encoded implicit neural representations (INRs) have shown strong capability in modeling continuous signals from discrete samples. However, conventional Fourier feature mappings use a fixed set of frequencies over the entire spatial domain, making them poorly suited to signals with spatially varying local spectra and often leading to slow convergence of high-frequency details. To address this issue, we propose an adaptive local frequency filtering method for Fourier-encoded INRs. The proposed method introduces a spatially varying parameter $α(\mathbf{x})$ to modulate encoded Fourier components, enabling a smooth transition among low-pass, band-pass, and high-pass behaviors at different spatial locations. We further analyze the effect of the proposed filter from the neural tangent kernel (NTK) perspective and provide an NTK-inspired interpretation of how it reshapes the effective kernel spectrum. Experiments on 2D image fitting, 3D shape representation, and sparse data reconstruction demonstrate that the proposed method consistently improves reconstruction quality and leads to faster optimization...
  </details>  
- **[NavCrafter: Exploring 3D Scenes from a Single Image](https://arxiv.org/abs/2604.02828v1)**  
  Authors: Hongbo Duan, Peiyu Zhuang, Yi Liu, Zhengyang Zhang, Yuxin Zhang, Pengting Luo, Fangming Liu, Xueqian Wang  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2604.02828v1.pdf)  
  <details><summary>Abstract</summary>

  Creating flexible 3D scenes from a single image is vital when direct 3D data acquisition is costly or impractical. We introduce NavCrafter, a novel framework that explores 3D scenes from a single image by synthesizing novel-view video sequences with camera controllability and temporal-spatial consistency. NavCrafter leverages video diffusion models to capture rich 3D priors and adopts a geometry-aware expansion strategy to progressively extend scene coverage. To enable controllable multi-view synthesis, we introduce a multi-stage camera control mechanism that conditions diffusion models with diverse trajectories via dual-branch camera injection and attention modulation. We further propose a collision-aware camera trajectory planner and an enhanced 3D Gaussian Splatting (3DGS) pipeline with depth-aligned supervision, structural regularization and refinement. Extensive experiments demonstrate that NavCrafter achieves state-of-the-art novel-view synthesis under large viewpoint shifts and substantially improves 3D reconstruction fidelity.
  </details>  
  Keywords: multi-view synthesis  
- **[Omni123: Exploring 3D Native Foundation Models with Limited 3D Data by Unifying Text to 2D and 3D Generation](https://arxiv.org/abs/2604.02289v1)**  
  Authors: Chongjie Ye, Cheng Cao, Chuanyu Pan, Yiming Hao, Yihao Zhi, Yuanming Hu, Xiaoguang Han  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2604.02289v1.pdf)  
  <details><summary>Abstract</summary>

  Recent multimodal large language models have achieved strong performance in unified text and image understanding and generation, yet extending such native capability to 3D remains challenging due to limited data. Compared to abundant 2D imagery, high-quality 3D assets are scarce, making 3D synthesis under-constrained. Existing methods often rely on indirect pipelines that edit in 2D and lift results into 3D via optimization, sacrificing geometric consistency. We present Omni123, a 3D-native foundation model that unifies text-to-2D and text-to-3D generation within a single autoregressive framework. Our key insight is that cross-modal consistency between images and 3D can serve as an implicit structural constraint. By representing text, images, and 3D as discrete tokens in a shared sequence space, the model leverages abundant 2D data as a geometric prior to improve 3D representations. We introduce an interleaved X-to-X training paradigm that coordinates diverse cross-modal tasks over heterogeneous paired datasets without requiring fully aligned text-image-3D...
  </details>  
  Keywords: text-to-3d  
- **[GEMM-GS: Accelerating 3D Gaussian Splatting on Tensor Cores with GEMM-Compatible Blending](https://arxiv.org/abs/2604.02120v2)**  
  Authors: Haomin Li, Bowen Zhu, Fangxin Liu, Zongwu Wang, Xinran Liang, Li Jiang, Haibing Guan  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2604.02120v2.pdf) | [![GitHub](https://img.shields.io/github/stars/shieldforever/GEMM-GS?style=social)](https://github.com/shieldforever/GEMM-GS)  
  <details><summary>Abstract</summary>

  Neural Radiance Fields (NeRF) enables 3D scene reconstruction from several 2D images but incurs high rendering latency via its point-sampling design. 3D Gaussian Splatting (3DGS) improves on NeRF with explicit scene representation and an optimized pipeline yet still fails to meet practical real-time demands. Existing acceleration works overlook the evolving Tensor Cores of modern GPUs because 3DGS pipeline lacks General Matrix Multiplication (GEMM) operations. This paper proposes GEMM-GS, an acceleration approach utilizing tensor cores on GPUs via GEMM-friendly blending transformation. It equivalently reformulates the 3DGS blending process into a GEMM-compatible form to utilize Tensor Cores. A high-performance CUDA kernel is designed, integrating a three-stage double-buffered pipeline that overlaps computation and memory access. Extensive experiments show that GEMM-GS achieves $1.42\times$ speedup over vanilla 3DGS and provides an additional $1.47\times$ speedup on average when combining with existing acceleration approaches. Code is released at https://github.com/shieldforever/GEMM-GS.
  </details>  
  Keywords: 3d scene reconstruction  
- **[SDesc3D: Towards Layout-Aware 3D Indoor Scene Generation from Short Descriptions](https://arxiv.org/abs/2604.01972v3)**  
  Authors: Jie Feng, Jiawei Shen, Junjia Huang, Junpeng Zhang, Mingtao Feng, Weisheng Dong, Guanbin Li  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2604.01972v3.pdf)  
  <details><summary>Abstract</summary>

  3D indoor scene generation conditioned on short textual descriptions provides a promising avenue for interactive 3D environment construction without the need for labor-intensive layout specification. Despite recent progress in text-conditioned 3D scene generation, existing works suffer from poor physical plausibility and insufficient detail richness in such semantic condensation cases, largely due to their reliance on explicit semantic cues about compositional objects and their spatial relationships. This limitation highlights the need for enhanced 3D reasoning capabilities, particularly in terms of prior integration and spatial anchoring. Motivated by this, we propose SDesc3D, a short-text conditioned 3D indoor scene generation framework, that leverages multi-view structural priors and regional functionality implications to enable 3D layout reasoning under sparse textual guidance. Specifically, we introduce a Multi-view scene prior augmentation that enriches underspecified textual inputs with aggregated multi-view structural knowledge, shifting from inaccessible semantic relation cues to multi-view relational prior aggregation. Building on this, we design...
  </details>  
- **[Control-DINO: Feature Space Conditioning for Controllable Image-to-Video Diffusion](https://arxiv.org/abs/2604.01761v1)**  
  Authors: Edoardo A. Dominici, Thomas Deixelberger, Konstantinos Vardis, Markus Steinberger  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2604.01761v1.pdf)  
  <details><summary>Abstract</summary>

  Video models have recently been applied with success to problems in content generation, novel view synthesis, and, more broadly, world simulation. Many applications in generation and transfer rely on conditioning these models, typically through perceptual, geometric, or simple semantic signals, fundamentally using them as generative renderers. At the same time, high-dimensional features obtained from large-scale self-supervised learning on images or point clouds are increasingly used as a general-purpose interface for vision models. The connection between the two has been explored for subject specific editing, aligning and training video diffusion models, but not in the role of a more general conditioning signal for pretrained video diffusion models. Features obtained through self-supervised learning like DINO, contain a lot of entangled information about style, lighting and semantics of the scene. This makes them great at reconstruction tasks but limits their generative capabilities. In this paper, we show how we can use the features...
  </details>  
  Keywords: novel view synthesis  
- **[Satellite-Free Training for Drone-View Geo-Localization](https://arxiv.org/abs/2604.01581v2)**  
  Authors: Tao Liu, Yingzhi Zhang, Kan Ren, Xiaoqi Zhao  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2604.01581v2.pdf)  
  <details><summary>Abstract</summary>

  Drone-view geo-localization (DVGL) aims to determine the location of drones in GPS-denied environments by retrieving the corresponding geotagged satellite tile from a reference gallery given UAV observations of a location. In many existing formulations, these observations are represented by a single oblique UAV image. In contrast, our satellite-free setting is designed for multi-view UAV sequences, which are used to construct a geometry-normalized UAV-side location representation before cross-view retrieval. Existing approaches rely on satellite imagery during training, either through paired supervision or unsupervised alignment, which limits practical deployment when satellite data are unavailable or restricted. In this paper, we propose a satellite-free training (SFT) framework that converts drone imagery into cross-view compatible representations through three main stages: drone-side 3D scene reconstruction, geometry-based pseudo-orthophoto generation, and satellite-free feature aggregation for retrieval. Specifically, we first reconstruct dense 3D scenes from multi-view drone images using 3D Gaussian splatting and project the reconstructed geometry into...
  </details>  
  Keywords: 3d scene reconstruction  
- **[DirectFisheye-GS: Enabling Native Fisheye Input in Gaussian Splatting with Cross-View Joint Optimization](https://arxiv.org/abs/2604.00648v2)**  
  Authors: Zhengxian Yang, Fei Xie, Xutao Xue, Rui Zhang, Taicheng Huang, Yang Liu, Mengqi Ji, Tao Yu  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2604.00648v2.pdf) | [![Project](https://img.shields.io/badge/-Project-blue)](https://yzxqh.github.io/DirectFisheye-GS)  
  <details><summary>Abstract</summary>

  3D Gaussian Splatting (3DGS) has enabled efficient 3D scene reconstruction from everyday images with real-time, high-fidelity rendering, greatly advancing VR/AR applications. Fisheye cameras, with their wider field of view (FOV), promise high-quality reconstructions from fewer inputs and have recently attracted much attention. However, since 3DGS relies on rasterization, most subsequent works involving fisheye camera inputs first undistort images before training, which introduces two problems: 1) Black borders at image edges cause information loss and negate the fisheye's large FOV advantage; 2) Undistortion's stretch-and-interpolate resampling spreads each pixel's value over a larger area, diluting detail density -- causes 3DGS overfitting these low-frequency zones, producing blur and floating artifacts. In this work, we integrate fisheye camera model into the original 3DGS framework, enabling native fisheye image input for training without preprocessing. Despite correct modeling, we observed that the reconstructed scenes still exhibit floaters at image edges: Distortion increases toward the periphery, and...
  </details>  
  Keywords: 3d scene reconstruction  



## Categorized Papers

### Cross-Modal Generation

*Showing the latest 50 out of 96 papers*

- **[Global-Local Monte Carlo Tree Search in Vision-Language Models for Text-to-3D Indoor Scene Generation](https://arxiv.org/abs/2606.06002v2)**  
  Authors: Mengshi Qi, Wei Deng, Xianlin Zhang, Huadong Ma  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2606.06002v2.pdf)  
  <details><summary>Abstract</summary>

  Large Vision-Language Models have achieved significant reasoning performance in various tasks. However, there are few studies on text-to-3D indoor scene generation with LVLMs. The main challenge is that prevailing LVLM-based methods employ chain-of-thought sequential decision mechanisms that cannot revise earlier decisions, causing error propagation. In this paper, we consider the task as a planning problem constrained by spatial and layout commonsense. To solve this problem, we model it as a tree search problem with global and local trees, which differs from existing sequential decision-making approaches. In the global tree, we place each object iteratively and explore multiple attempts like humans furnishing a room, where the problem space is represented as a tree. To effectively search the tree, we propose a hierarchical scene representation and a PRM-guided MCTS method. This representation abstracts a scene into room level, region level, floor object level, and supported object level. The PRM-guided MCTS method uses...
  </details>  
  Keywords: text-to-3d  
- **[REST3D: Reconstructing Physically Stable 3D Scenes from a Single Image](https://arxiv.org/abs/2605.30338v1)**  
  Authors: Xiaoxuan Ma, Jiashun Wang, Nicolas Ugrinovic, Yehonathan Litman, Kris Kitani  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2605.30338v1.pdf)  
  <details><summary>Abstract</summary>

  Reconstructing physically stable 3D scenes from a single RGB image enables casual images to be converted into simulation-ready digital assets for applications such as immersive interaction and content creation. However, existing single-image reconstruction methods fall short in capturing the physical structure of a scene. As a result, they often produce geometrically plausible but physically inconsistent results, including object floating and penetration, which lead to unstable behavior in physics simulations. Image-conditioned scene generation methods improve physical plausibility but often rely on strong scene priors, yielding plausible yet inaccurate object arrangements that fail to match the input image. We propose REST3D, a single-image reconstruction framework that can reconstruct physically stable 3D scenes by integrating physical scene understanding with physics-constrained refinement. We first introduce an agentic physical scene understanding technique that constructs a scene-tree representation capturing object physical states and inter-object relationships from a gravity-support perspective, providing a structural prior for reconstruction. Leveraging...
  </details>  
  Keywords: image-to-3d  
- **[Helix4D: Complex 4D Mesh Generation](https://arxiv.org/abs/2605.26109v1)**  
  Authors: Jiraphon Yenphraphai, Jianqi Chen, Jian Wang, Gordon Qian, Sergey Tulyakov, Rameen Abdal, Raymond A. Yeh, Peter Wonka, Chaoyang Wang  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2605.26109v1.pdf)  
  <details><summary>Abstract</summary>

  Current video-to-4D methods struggle with complex topology changes, transparent materials, thin structures, and inner surfaces. We present Helix4D, a dynamic mesh generation framework by inheriting the expressive representation of Trellis2, adapting it from image-to-3D to video-conditioned 4D generation. Our design arises from two key questions: (a) how to enable Trellis2's frame-local attention to share information across frames while preserving its pretrained quality on rare cases such as transparent objects and inner surfaces, and (b) how to inject temporal information into a purely 3D positional encoding without breaking pretrained capabilities. We address (a) with a sliding-window cross-frame attention and anchor on the first frame. The first frame is generated by the base Trellis2 model and injected into our model, letting it inherit Trellis2's quality in rare cases through cross-frame attention. We address (b) with a 4D temporal encoding that repurposes redundant low-frequency spatial RoPE bands for time, extending the encoding from...
  </details>  
  Keywords: image-to-3d  
- **[Variance Reduction for Expectations with Diffusion Teachers](https://arxiv.org/abs/2605.21489v2)**  
  Authors: Jesse Bettencourt, Xindi Wu, Matan Atzmon, James Lucas, Jonathan Lorraine  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2605.21489v2.pdf)  
  <details><summary>Abstract</summary>

  Pretrained diffusion models serve as frozen teachers feeding downstream pipelines such as text-to-3D, single-step distillation, and data attribution. The teacher gradients these pipelines consume are Monte Carlo (MC) expectations over noise levels and Gaussian noise samples; their estimator variance dominates compute cost because each draw requires expensive upstream work (rendering, simulation, encoding). We introduce CARV, a compute-aware variance-accounting framework that motivates a hierarchical MC estimator: amortize the expensive upstream computation over cheap diffusion-noise resamples, sharpened by timestep importance sampling and a stratified-inverse-CDF construction. In our text-to-3D distillation and attribution experiments, CARV delivers 2-3x effective compute multipliers (most from amortized reuse; ~25% additional from IS+stratification) without changing the objective; in single-step distillation, the same techniques cut gradient variance by an order of magnitude but do not improve downstream FID, marking the regime where MC variance is no longer the bottleneck.
  </details>  
  Keywords: text-to-3d  
- **[ROAR-3D: Routing Arbitrary Views for High-Fidelity 3D Generation](https://arxiv.org/abs/2605.21121v1)**  
  Authors: Hanxiao Sun, Mingxin Yang, Shuhui Yang, Zebin He, Xintong Han, Hongbo Fu, Chunchao Guo, Wenhan Luo  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2605.21121v1.pdf)  
  <details><summary>Abstract</summary>

  Single-image-to-3D generative models can now produce high-quality geometry, yet conditioning on a single view inevitably introduces ambiguity about unseen regions. Multi-view conditioning can reduce this ambiguity, but existing methods either require fixed canonical viewpoints or rely on external reconstruction modules that impose heavy training costs and limit generation quality. We observe that pretrained single-view models already possess strong 2D-to-3D grounding that can be reused for multi-view conditioning. However, a closer analysis reveals that their conditioning mechanism entangles orientation control with geometry transfer, two functions that conflict when images from different viewpoints are naively combined. Based on this analysis, we propose ROAR-3D, a lightweight method that upgrades a pretrained single-view model to accept an arbitrary number of unposed images. A token-wise view router assigns each 3D latent token to its most relevant view, implicitly establishing 2D-to-3D correspondences without explicit pose input. A dual-stream attention design preserves the pretrained primary-view behavior while...
  </details>  
  Keywords: image-to-3d  
- **[Structural Energy Guidance for View-Consistent Text-to-3D Generation](https://arxiv.org/abs/2605.19876v1)**  
  Authors: Qing Zhang, Jinguang Tong, Jing Zhang, Jie Hong, Xuesong Li  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2605.19876v1.pdf)  
  <details><summary>Abstract</summary>

  Text-to-3D generation based on diffusion models often suffers from the Janus problem, leading to inconsistent geometry across viewpoints. This work identifies viewpoint bias in 2D diffusion priors as the main cause and proposes Structural Energy-Guided Sampling (SEGS), a training-free and plug-and-play framework to improve multi-view consistency. SEGS constructs a structural energy in the PCA subspace of U-Net features and injects its gradient into the denoising process. It can be easily integrated into SDS/VSD pipelines without retraining. Experiments show that SEGS reduces the Janus Rate by about 10% on average and improves View-CS scores across multiple baselines, including DreamFusion, Magic3D, and LucidDreamer. This method effectively alleviates viewpoint artifacts while preserving appearance fidelity, providing a flexible solution for high-quality text-to-3D content generation.
  </details>  
  Keywords: text-to-3d  
- **[TelePhysics: Physics-Grounded Multi-Object Scene Generation from a Single Image with Real-Time Interaction](https://arxiv.org/abs/2605.20290v1)**  
  Authors: Xin Zhang, Yabo Chen, Yijie Fang, Wanying Qu, Haibin Huang, Chi Zhang, Feng Xu, Xuelong Li  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2605.20290v1.pdf) | [![GitHub](https://img.shields.io/github/stars/xinzhang007/TelePhysics?style=social)](https://github.com/xinzhang007/TelePhysics)  
  <details><summary>Abstract</summary>

  Recent generative video models achieve impressive visual quality but remain constrained by limited physical consistency and controllability. Existing video generation methods provide minimal physical control, and single-image-to-3D conversion approaches often suffer from object interpenetration. Furthermore, physics-based scene-level 3D generation methods exhibit spatial misalignment, stylized artifacts, and inconsistencies with the input data, restricting their use in realistic interactive video synthesis. We propose TelePhysics, a training-free framework that converts a single image into a physically consistent and controllable video through holistic scene-level 3D reconstruction. By representing the full scene geometry in a unified spatial coordinate system, TelePhysics resolves object penetration and alignment ambiguity. Unlike prior methods, this formulation enables accurate scenelevel multi-object interactions and introduces richer, complex control types for advanced mechanicsbased manipulation. By decoupling simulation from rendering, TelePhysics bypasses latency-heavy priors, achieving real-time physical interaction previews paired while preserving photorealistic visual fidelity. Experimental results demonstrate that TelePhysics substantially outperforms prior methods...
  </details>  
  Keywords: image-to-3d  
- **[Efficient 3D Content Reconstruction and Generation](https://arxiv.org/abs/2605.18052v1)**  
  Authors: Jiahao Li  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2605.18052v1.pdf)  
  <details><summary>Abstract</summary>

  Automatic 3D content creation seeks to replace labor-intensive modeling and scanning pipelines with systems that can synthesize or recover 3D assets directly from text or images. Its applications span video games, virtual reality, robotics, and simulation, enabling rapid asset prototyping, diverse interactive world generation, and efficient 3D data collection for training foundation models. Contemporary solutions largely follow two complementary paradigms: (i) text- or image-to-3D generation, which learns priors over 3D geometry and appearance to create novel assets from natural language or a single view image; and (ii) 3D reconstruction, which estimates camera poses and geometry from RGB images. This thesis advances both directions. On the generation side, I introduce Instant3D, which combines multi-view diffusion with feed-forward sparse-view 3D reconstruction to produce high-quality assets in 5-20 seconds. On the reconstruction side, I develop FastMap, a structure-from-motion pipeline that achieves up to 10x speedup over prior state-of-the-art by using first-order optimization with...
  </details>  
  Keywords: novel view synthesis, image-to-3d  
- **[3DPhysVideo: Consistency-Guided Flow SDE for Video Generation via 3D Scene Reconstruction and Physical Simulation](https://arxiv.org/abs/2605.16795v1)**  
  Authors: Hwidong Kim, Yunho Kim, Tae-Kyun Kim  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2605.16795v1.pdf)  
  <details><summary>Abstract</summary>

  Video generative models have made remarkable progress, yet they often yield visual artifacts that violate grounding in physical dynamics. Recent works such as PhysGen3D tackle single image-to-3D physics through mesh reconstruction and Physically-Based Rendering, but challenges remain in modeling fluid dynamics, multi-object interactions and photorealism. This work introduces 3DPhysVideo, a novel training-free pipeline that generates physically realistic videos from a single image. We repurpose an off-the-shelf video model for two stages. First, we use it as a novel view synthesizer to reconstruct complete 360-degree 3D scene geometry by guiding the image-to-video (I2V) flow model with rendered point clouds. Second, after applying physics solvers to this geometry, the physically simulated point cloud is used to guide the same I2V flow model to synthesize final, high-quality videos. Consistency-Guided Flow SDE, which decomposes the predicted velocity of the I2V flow model into denoising and consistency bias, enforces consistency to the conditional inputs, allowing...
  </details>  
  Keywords: 3d scene reconstruction, image-to-3d  
- **[EVA01: Unified Native 3D Understanding and Generation via Mixture-of-Transformers](https://arxiv.org/abs/2605.16745v1)**  
  Authors: Zongyuan Yang, Mingjing Yi, Wanli Ma, Chenzhuo Fan, Bocheng Li, Baolin Liu, Yuke Lou, Yingde Song, Yongping Xiong, Zhengdong Guo, Shimu Wang  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2605.16745v1.pdf) | [![Project](https://img.shields.io/badge/-Project-blue)](https://www.seeles.ai/research/pages/EVA01)  
  <details><summary>Abstract</summary>

  This paper addresses the challenge of integrating 3D meshes as a native modality within Multimodal Large Language Models (MLLMs). Diffusion-based large reconstruction models decouple semantic understanding from geometric reasoning, operating as stateless reconstructors conditioned on dense 2D pixel priors. Recent MLLM-based methods treat the 3D modality as an external output rather than a native component of the multimodal sequence, making incremental adaptations without a systematic analysis of how geometric manifolds align with MLLM feature spaces. We introduce EVA01, a unified framework that extends the modality boundary of MLLMs to natively incorporate 3D mesh understanding, generation, and context-aware editing. Built upon a Mixture-of-Transformers (MoT) architecture, EVA01 decouples the model into a pre-trained Understanding Expert ($E_{\mathrm{und}}$) and a structurally mirrored Generation Expert ($E_{\mathrm{gen}}$), coupled through shared global self-attention with hard modality routing. This design aligns the semantic latent space of the MLLM backbone with the geometric manifold, enabling direct transfer of multimodal...
  </details>  
  Keywords: text-to-3d  

### Dynamic & Articulated Modeling

- **[Fishbone: From One 3D Asset to a Million Controllable Edits](https://arxiv.org/abs/2605.24805v1)**  
  Authors: Yumeng He, Xiaoying Wang, Peihao Li, Yanjia Huang, Joe Masterjohn, Jiajun Wu, Leonidas Guibas, Yin Yang, Ying Jiang, Chenfanfu Jiang  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2605.24805v1.pdf)  
  <details><summary>Abstract</summary>

  Large-scale controllable 3D assets are critical for computer graphics, embodied AI, robotics, and interactive content creation, yet creating diverse 3D assets remains challenging due to the high cost of manual modeling and rigging. Shape deformation offers a natural way to generate variations from existing meshes, but existing data-driven methods often rely on sparse user inputs, while parametric editing frameworks require manually designed control structures and category-specific configurations. Inspired by natural creatures, where a central spine governs global shape and cross-sectional ribs control local variation, we introduce Fishbone, a unified rib-spine representation for general shapes that supports controllable parametric mesh deformation, reduced-space dynamics, and animation. Given an input mesh, Fishbone computes a geodesic scalar field with an adaptive heat method, extracts iso-contours as cross-sectional ribs, constructs a smooth geometry-aware spine through rib centers, and associates surface vertices with nearby rib and spine structures using Gaussian-weighted skinning. The resulting representation enables real-time...
  </details>  
  Keywords: controllable 3d generation, rigging  
- **[Artiverse: A Diverse and Physically Grounded Dataset for Articulated Objects](https://arxiv.org/abs/2605.24403v1)**  
  Authors: Denys Iliash, Jiayi Liu, Egor Fokin, Qirui Wu, Ali Mahdavi-Amiri, Manolis Savva, Angel X. Chang  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2605.24403v1.pdf)  
  <details><summary>Abstract</summary>

  We present Artiverse, a diverse and physically grounded dataset of high-quality articulated 3D objects designed for realistic functional modeling and simulation. Artiverse contains 5.4K human-authored objects across a broad range of 88 categories, aggregated from multiple 3D static repositories. Objects are annotated with functional parts, interior structures, realistic kinematic relationships and articulated joints including multi-DoF joints, and physical attributes such as metric scale, material, and mass. We develop a semi-automated annotation pipeline that combines few-shot segmentation, geometric reasoning, and multi-stage human verification to achieve high-quality and efficient annotation, reducing manual annotation time by over 30%. We demonstrate the value of Artiverse on tasks of part mobility analysis, articulated object generation, and physics-based interaction. Artiverse provides a data resource to advance functional understanding for articulated objects.
  </details>  
  Keywords: articulated object generation  
- **[TOPOS: High-Fidelity and Efficient Industry-Grade 3D Head Generation](https://arxiv.org/abs/2605.14594v1)**  
  Authors: Bojun Xiong, Zoubin Bi, Xinghui Peng, Yunmu Wang, Junchen Deng, Jun Liang, Jing Li, Bowen Cai, Huan Fu  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2605.14594v1.pdf)  
  <details><summary>Abstract</summary>

  High-fidelity 3D head generation plays a crucial role in the film, animation and video game industries. In industrial pipelines, studios typically enforce a fixed reference topology across all head assets, as such a clean and uniform topology is a prerequisite for production-level rigging, skinning and animation. In this paper, we present TOPOS, a framework tailored for single image conditioned 3D head generation that jointly recovers geometry and appearance under such an industry-standard topology. In contrast to general 3D generative models which produce triangle meshes with inconsistent topology and numerous vertices, hindering semantic correspondence and asset-level reuse, TOPOS generates head meshes with a fixed, studio-style topology, enabling consistent vertex-level correspondence across all generated heads. To model heads under this unified topology, we proposed a novel variational autoencoder structure, termed TOPOS-VAE. Inspired by multi-model large language models (MLLMs), our TOPOS-VAE leverages the Perceiver Resampler to convert input pointclouds sampled from head meshes...
  </details>  
  Keywords: rigging  
- **[Rigel3D: Rig-aware Latents for Animation-Ready 3D Asset Generation](https://arxiv.org/abs/2605.13129v1)**  
  Authors: Nikitas Chatzis, Marios Loizou, Evangelos Kalogerakis  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2605.13129v1.pdf)  
  <details><summary>Abstract</summary>

  Recent 3D generative models can synthesize high-quality assets, but their outputs are typically static: they lack the skeletal rigs, joint hierarchies, and skinning weights required for animation. This limits their use in games, film, simulation, virtual agents, and embodied AI, where assets must not only look plausible but also move plausibly. We introduce Rigel3D, a generative method for animation-ready 3D assets represented as rigged meshes. Unlike post-hoc auto-rigging methods that attach rigs to completed shapes, our method jointly models geometry and rig structure through coupled surface and skeleton structured latent representations. A rig-aware autoencoder decodes these representations into mesh geometry, skeleton topology, joint coordinates, and skinning weights, while a two-stage latent generative model synthesizes both surface and skeleton representations for image-conditioned generation. To support downstream animation workflows, we further introduce an open-vocabulary joint labeling module that embeds generated joints into a shared vision-language space, enabling correspondence to arbitrary retargeting templates....
  </details>  
  Keywords: 3d asset generation, rigging  
- **[MeshReGen: A Unified 3D Geometry Regeneration Framework](https://arxiv.org/abs/2604.28134v2)**  
  Authors: Geon Yeong Park, Roman Shapovalov, Rakesh Ranjan, Jong Chul Ye, Andrea Vedaldi, Thu Nguyen-Phuoc  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2604.28134v2.pdf)  
  <details><summary>Abstract</summary>

  We consider the problem of regenerating 3D objects from 2D images and initial 3D shapes. Most 3D generators operate in a one-shot fashion, converting text or images to a 3D object with limited controllability. We introduce instead MeshReGen, a 3D regenerator that is conditioned on an initial 3D shape. This conceptually simple formulation allows us to support numerous useful tasks, including 3D enhancement, reconstruction, and editing. MeshReGen uses a new conditioning mechanism based on VecSet, which allows the regenerator to update or improve the input geometry with consistent fine-grained details. MeshReGen learns a widely applicable regeneration prior from off-the-shelf 3D datasets via self-supervised pretext tasks and augmentations, without additional annotations. We evaluate both the geometric consistency and fine-grained quality of MeshReGen, achieving state-of-the-art performance in controllable 3D generation across several tasks.
  </details>  
  Keywords: 3d generator, controllable 3d generation  
- **[From Visual Synthesis to Interactive Worlds: Toward Production-Ready 3D Asset Generation](https://arxiv.org/abs/2604.23629v2)**  
  Authors: Jiafeng Wu, Zhuofan Lou, Jian Liu, Dazhao Du, Chunchao Guo, Song Guo  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2604.23629v2.pdf)  
  <details><summary>Abstract</summary>

  Three-dimensional content generation has progressed from producing isolated, visually plausible shapes to constructing structured assets that can be deployed in real-time interactive environments. This trajectory is driven by converging demands from game development, embodied AI, world simulation, digital twins, and spatial computing, all of which require 3D content that goes beyond surface appearance to satisfy engine-level constraints on topology, UV parameterization, physically based materials, skeletal rigging, and physics-aware scene layout. Despite rapid advances in generative modeling, a persistent gap separates the outputs of current methods from the production-ready standard expected by interactive applications. This survey addresses that gap by organizing the literature around the asset production pipeline rather than algorithmic families. Along the horizontal axis we distinguish three asset tiers, namely general objects, characters, and scenes, while the vertical axis traces each tier through the full production lifecycle from data foundations and geometry synthesis through topology optimization, UV unwrapping, PBR...
  </details>  
  Keywords: 3d asset generation, rigging  
- **[AniGen: Unified $S^3$ Fields for Animatable 3D Asset Generation](https://arxiv.org/abs/2604.08746v2)**  
  Authors: Yi-Hua Huang, Zi-Xin Zou, Yuting He, Chirui Chang, Cheng-Feng Pu, Ziyi Yang, Yuan-Chen Guo, Yan-Pei Cao, Xiaojuan Qi  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2604.08746v2.pdf) | [![Project](https://img.shields.io/badge/-Project-blue)](https://yihua7.github.io/AniGen-web)  
  <details><summary>Abstract</summary>

  Animatable 3D assets, defined as geometry equipped with an articulated skeleton and skinning weights, are fundamental to interactive graphics, embodied agents, and animation production. While recent 3D generative models can synthesize visually plausible shapes from images, the results are typically static. Obtaining usable rigs via post-hoc auto-rigging is brittle and often produces skeletons that are topologically inconsistent with the generated geometry. We present AniGen, a unified framework that directly generates animate-ready 3D assets conditioned on a single image. Our key insight is to represent shape, skeleton, and skinning as mutually consistent $S^3$ Fields (Shape, Skeleton, Skin) defined over a shared spatial domain. To enable the robust learning of these fields, we introduce two technical innovations: (i) a confidence-decaying skeleton field that explicitly handles the geometric ambiguity of bone prediction at Voronoi boundaries, and (ii) a dual skin feature field that decouples skinning weights from specific joint counts, allowing a fixed-architecture...
  </details>  
  Keywords: 3d asset generation, rigging  
- **[Points-to-3D: Structure-Aware 3D Generation with Point Cloud Priors](https://arxiv.org/abs/2603.18782v3)**  
  Authors: Jiatong Xia, Zicheng Duan, Anton van den Hengel, Lingqiao Liu  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2603.18782v3.pdf) | [![Project](https://img.shields.io/badge/-Project-blue)](https://jiatongxia.github.io/points2-3D)  
  <details><summary>Abstract</summary>

  Recent progress in 3D generation has been driven largely by models conditioned on images or text, while readily available 3D priors are still underused. In many real-world scenarios, the visible-region point cloud are easy to obtain from active sensors such as LiDAR or from feed-forward predictors like VGGT, offering explicit geometric constraints that current methods fail to exploit. In this work, we introduce Points-to-3D, a diffusion-based framework that leverages point cloud priors for geometry-controllable 3D asset and scene generation. Built on a latent 3D diffusion model TRELLIS, Points-to-3D first replaces pure-noise sparse structure latent initialization with a point cloud priors tailored input formulation.A structure inpainting network, trained within the TRELLIS framework on task-specific data designed to learn global structural inpainting, is then used for inference with a staged sampling strategy (structural inpainting followed by boundary refinement), completing the global geometry while preserving the visible regions of the input priors. In...
  </details>  
  Keywords: controllable 3d generation  
- **[Cog2Gen3D: Sculpturing 3D Semantic-Geometric Cognition for 3D Generation](https://arxiv.org/abs/2603.05845v1)**  
  Authors: Haonan Wang, Hanyu Zhou, Haoyue Liu, Tao Gu, Luxin Yan  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2603.05845v1.pdf)  
  <details><summary>Abstract</summary>

  Generative models have achieved success in producing semantically plausible 2D images, but it remains challenging in 3D generation due to the absence of spatial geometry constraints. Typically, existing methods utilize geometric features as conditions to enhance spatial awareness. However, these methods can only model relative relationships and are prone to scale inconsistency of absolute geometry. Thus, we argue that semantic information and absolute geometry empower 3D cognition, thereby enabling controllable 3D generation for the physical world. In this work, we propose Cog2Gen3D, a 3D cognition-guided diffusion framework for 3D generation. Our model is guided by three key designs: 1) Cognitive Feature Embeddings. We encode different modalities into semantic and geometric representations and further extract logical representations. 2) 3D Latent Cognition Graph. We structure different representations into dual-stream semantic-geometric graphs and fuse them via common-based cross-attention to obtain a 3D cognition graph. 3) Cognition-Guided Latent Diffusion. We leverage the fused 3D...
  </details>  
  Keywords: controllable 3d generation  
- **[PAct: Part-Decomposed Single-View Articulated Object Generation](https://arxiv.org/abs/2602.14965v1)**  
  Authors: Qingming Liu, Xinyue Yao, Shuyuan Zhang, Yueci Deng, Guiliang Liu, Zhen Liu, Kui Jia  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2602.14965v1.pdf)  
  <details><summary>Abstract</summary>

  Articulated objects are central to interactive 3D applications, including embodied AI, robotics, and VR/AR, where functional part decomposition and kinematic motion are essential. Yet producing high-fidelity articulated assets remains difficult to scale because it requires reliable part decomposition and kinematic rigging. Existing approaches largely fall into two paradigms: optimization-based reconstruction or distillation, which can be accurate but often takes tens of minutes to hours per instance, and inference-time methods that rely on template or part retrieval, producing plausible results that may not match the specific structure and appearance in the input observation. We introduce a part-centric generative framework for articulated object creation that synthesizes part geometry, composition, and articulation under explicit part-aware conditioning. Our representation models an object as a set of movable parts, each encoded by latent tokens augmented with part identity and articulation cues. Conditioned on a single image, the model generates articulated 3D assets that preserve instance-level...
  </details>  
  Keywords: articulated object generation, rigging  

### Scene & View Synthesis

*Showing the latest 50 out of 107 papers*

- **[Unpaired RGB-Thermal Gaussian-Splatting Using Visual Geometric Transformers](https://arxiv.org/abs/2606.05491v1)**  
  Authors: Jean Cordonnier, Chenghao Xu, Olga Fink, Malcolm Mielle  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2606.05491v1.pdf)  
  <details><summary>Abstract</summary>

  Multi-modal novel view synthesis (NVS) combining RGB and thermal imagery enables precise 3D scene reconstruction with visual and thermal information. However, existing methods typically rely on precisely calibrated RGB-thermal image pairs or stereo setups, limiting scalability and practical deployment. To address this, we introduce a framework for unpaired RGB-thermal NVS that leverages VGGT, a 3D feed-forward transformer architecture, to independently estimate camera poses for each modality. The pose sets are then aligned using the Procrustes algorithm with a cross-modal feature matcher, enabling joint registration without paired calibration. Building on this alignment, we further propose a multi-modal 3D Gaussian Splatting approach that learns directly from unpaired RGB and thermal images. Experiments on diverse scenes demonstrate that our method achieves competitive performance in thermal view synthesis while maintaining RGB fidelity. Moreover, we show that existing reconstruction approaches can produce modality-specific reconstructions that lack cross-modal consistency. We thus introduce a benchmarking framework to...
  </details>  
  Keywords: 3d scene reconstruction, novel view synthesis  
- **[SimuScene: Simulation-Ready Compositional 3D Scene Reconstruction from a Single Image](https://arxiv.org/abs/2606.03994v1)**  
  Authors: Inhee Lee, Sangwon Baik, Sungjoo Kim, Hyeonwoo Kim, Hyunsoo Cha, Hanbyul Joo  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2606.03994v1.pdf)  
  <details><summary>Abstract</summary>

  Reconstructing interactive, simulation-ready 3D scenes from a single image is a critical bottleneck for robotic manipulation. While recent single-image lifters recover plausible per-object shapes, composing them yields scenes that collapse under physical simulation due to interpenetrating, hovering, or sinking objects. Existing physics-aware methods address this strictly as a post-hoc layout correction, leaving the underlying geometric errors unresolved. To address this, we introduce SimuScene, a compositional 3D reconstruction pipeline that puts physics in the loop of shape and layout estimation. Rather than using physics merely for layout cleanup, we utilize the physics engine as a diagnostic measurement tool during the generative process itself. By diagnostically simulating reconstructed objects under gravity, we convert penetration and support failures into quantitative correction signals that drive gravity-axis stretching and amodal shape resampling. This physics-informed feedback loop mitigates accumulated reconstruction errors and produces a stable, simulation-ready compositional 3D scene. Extensive experiments demonstrate state-of-the-art performance on physical...
  </details>  
  Keywords: 3d scene reconstruction  
- **[Beyond Static Gaussians: An Empirical Investigation of Architectural Paradigms for Dynamic 3D Scene Reconstruction](https://arxiv.org/abs/2606.00452v1)**  
  Authors: Adrian Ramlal, John S. Zelek  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2606.00452v1.pdf)  
  <details><summary>Abstract</summary>

  Dynamic scene reconstruction via 3D Gaussian Splatting (3DGS) has emerged as a compelling approach for representing evolving environments, yet understanding trade-offs between methodologies remains crucial. This paper presents a comprehensive analysis of dynamic 3DGS methods, categorizing them into two paradigms: structure-guided methods employing auxiliary representations (deformation fields, canonical spaces, grids) to model temporal changes, and gaussian-centric methods encoding dynamics directly into primitives via continuous functions or 4D representations. We evaluate representative methods from both paradigms on the D-NeRF benchmark. Our findings reveal that structure-guided methods achieve superior reconstruction fidelity and compact model sizes, while gaussian-centric approaches demonstrate significantly higher rendering speeds enabling real-time performance, though with greater quality variability and potentially substantial storage overhead. This analysis highlights a fundamental trade-off between reconstruction quality/compactness versus rendering speed, providing insights to guide future research and application development in dynamic scene reconstruction.
  </details>  
  Keywords: 3d scene reconstruction  
- **[Feature-Optimized Vision for Adaptive 3D Scene Reconstruction](https://arxiv.org/abs/2605.31534v1)**  
  Authors: Eric Liang  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2605.31534v1.pdf)  
  <details><summary>Abstract</summary>

  Three-dimensional scene reconstruction depends on local image evidence that is both visually discriminative and geometrically useful. Fixed feature thresholds and uniform feature budgets are easy to deploy, but they can waste computation on repeated texture, low-parallax regions, or unstable points. This paper proposes an adaptive feature-optimized vision front end for 3D reconstruction. The method scores candidate features by texture, repeatability, distinctiveness, expected triangulation angle, and spatial coverage, then allocates a per-view feature budget to maximize useful tracks under a fixed reconstruction pipeline. A small synthetic multi-view prototype evaluates four selection policies across corridor, facade, object-table, and cluttered scenes. Compared with random, texture-only, and uniform-grid baselines, the adaptive policy obtains the best quality-aware completeness and the lowest aggregate reconstruction RMSE while preserving broad image coverage. The result is not a replacement for modern learned matching or neural reconstruction systems; it is a modular front-end policy that can make classical and learned...
  </details>  
  Keywords: 3d scene reconstruction  
- **[VolFill: Single-View Amodal 3D Scene Reconstruction with Volumetric Flow Matching](https://arxiv.org/abs/2605.31466v1)**  
  Authors: Tuan Duc Ngo, Chuang Gan, Evangelos Kalogerakis  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2605.31466v1.pdf)  
  <details><summary>Abstract</summary>

  Reconstructing the complete geometry of a scene from a single RGB image remains challenging - especially when inferring hidden structures where visual evidence is incomplete. We introduce VolFill, a generative framework that predicts the 3D structure of the complete scene rather than relying on traditional pixel-aligned regression. Our method utilizes a hybrid 3D VAE to compress sparse truncated unsigned distance function grids into a compact latent space, paired with a latent Diffusion Transformer that denoises this representation to recover the complete scene. We condition the generation on geometry foundation models, leveraging rich spatial priors for robust reasoning. Unlike existing methods limited by per-ray constraints or unstructured point-cloud queries, VolFill provides a structured representation that supports direct surface extraction and occupancy queries at scale. Extensive experiments on the SCRREAM and NRGB-D datasets demonstrate that our approach significantly outperforms current baselines, providing a robust foundation for holistic spatial understanding.
  </details>  
  Keywords: 3d scene reconstruction  
- **[DelowlightSplat: Feed-Forward Gaussian Splatting for Lowlight 3D Scene Reconstruction](https://arxiv.org/abs/2605.26629v1)**  
  Authors: Fuzhen Jiang, Zengtian Xie, Zhuoran Li  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2605.26629v1.pdf)  
  <details><summary>Abstract</summary>

  Novel-view synthesis and 3D reconstruction from sparse posed images are central to robotics and AR/VR. Yet, feed-forward 3D Gaussian reconstruction fails under lowlight due to noise, color shifts, and unreliable correspondence. We propose DelowlightSplat, a lowlight-aware feed-forward Gaussian splatting framework for clean novel-view rendering. We build a controllable multi-view lowlight benchmark by degrading only context views while keeping target views clean. We introduce a lightweight Lowlight Adapter for residual enhancement to improve matchability, and couple it with cost-volume-based multi-view inference to directly predict clean 3D Gaussians. Experiments show that DelowlightSplat significantly outperforms previous feed-forward method and two-stage pipeline under lowlight conditions.
  </details>  
  Keywords: 3d scene reconstruction  
- **[Underwater360: Reconstructing Underwater Scenes from Panoramic Images with Omnidirectional Gaussian Splatting](https://arxiv.org/abs/2605.26447v1)**  
  Authors: Jiangbei Hu, Weichao Song, Shibo Yu, Mohan Wang, Zihan Yi, Rui Wu, Mingkang Xiang, Na Lei, Shengfa Wang, Zhongxuan Luo, Ying He  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2605.26447v1.pdf) | [![GitHub](https://img.shields.io/github/stars/SwcK423/Underwater360?style=social)](https://github.com/SwcK423/Underwater360)  
  <details><summary>Abstract</summary>

  Underwater scene reconstruction is essential for immersive exploration of aquatic environments, yet remains challenging due to complex participating-media effects such as absorption and scattering, as well as the limited field of view (FoV) of conventional cameras. Although combining panoramic imaging with 3D Gaussian Splatting (3DGS) offers a promising direction for photorealistic underwater rendering, traditional 3DGS struggles with both spherical projection distortion and underwater medium degradation. In this paper, we propose \textbf{Underwater360}, a physics-informed omnidirectional 3DGS framework for underwater panoramic scene reconstruction. First, we introduce an Omnidirectional Gaussian Splatting module that performs ray casting directly in spherical camera space instead of relying on 2D projection approximations, thereby reducing geometric distortions under 360$^\circ$ FoV. Second, we design a physics-based appearance-medium modeling architecture with pose-conditioned appearance embeddings to explicitly decouple intrinsic scene radiance from depth-dependent backscatter and attenuation, enabling physically grounded scene appearance restoration. Finally, we establish a new panoramic underwater benchmark dataset...
  </details>  
  Keywords: novel view synthesis  
- **[TriSplat: Simulation-Ready Feed-Forward 3D Scene Reconstruction](https://arxiv.org/abs/2605.26115v1)**  
  Authors: Weijie Wang, Zimu Li, Jinchuan Shi, Zeyu Zhang, Botao Ye, Marc Pollefeys, Donny Y. Chen, Bohan Zhuang  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2605.26115v1.pdf)  
  <details><summary>Abstract</summary>

  Sparse-view 3D reconstruction is increasingly addressed with feed-forward splatting networks that predict explicit primitives directly from images. Yet most existing methods remain centered on Gaussian primitives and expose surfaces only indirectly: extracting a usable mesh for downstream simulation, physics reasoning, or embodied interaction still requires expensive post-hoc steps that break the feed-forward promise. This limitation is especially pronounced in pose-free settings, where scene structure and camera parameters must be estimated jointly from sparse observations. We present TriSplat, a feed-forward reconstruction network that represents scenes with oriented triangle primitives and directly exports simulation-ready mesh scenes from a single forward pass. Given input images, the network predicts local 3D point maps, triangle attributes, camera poses, and optional intrinsics. Rather than regressing triangle orientation as an unconstrained latent variable, our approach constructs geometry normals from the predicted point maps, refines them with an image-conditioned normal head, and converts them into stable local frames...
  </details>  
  Keywords: 3d scene reconstruction  
- **[GenRecon: Bridging Generative Priors for Multi-View 3D Scene Reconstruction](https://arxiv.org/abs/2605.23888v1)**  
  Authors: Katharina Schmid, Nicolas von Lützow, Jozef Hladký, Angela Dai, Matthias Nießner  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2605.23888v1.pdf)  
  <details><summary>Abstract</summary>

  We introduce a new approach to high-fidelity 3D scene reconstruction from multi-view RGB images that tightly couples reconstruction with a strong generative 3D prior. We cast scene reconstruction as conditional 3D generation over a set of spatially-localized, overlapping chunks that together tile the scene, scaling generation to large scene extents. Crucially, we inherit the fidelity and completeness of state-of-the-art generative shape models -- we use Trellis.2 as an example -- which we generalize to the scene level. To this end, we propose a projection-based conditioning mechanism that lifts posed multi-view image features into a coherent 3D representation aligned with the generative model, independent of view ordering and spatially anchored to the scene, yielding high-fidelity, multi-view consistent generated geometry. This enables lifting the strong object-level prior of Trellis.2 to multi-view, scene-scale generation, producing faithful, editable PBR mesh reconstructions of indoor environments. As a result, we obtain high-fidelity results that outperform cutting-edge...
  </details>  
  Keywords: 3d scene reconstruction  
- **[LangFlash: Feed-forward 3D Language Gaussian Splatting from Sparse Unposed Images](https://arxiv.org/abs/2605.23287v1)**  
  Authors: Yilong Liu, Wanhua Li, Chen Zhu-Tian, Hanspeter Pfister  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2605.23287v1.pdf) | [![Project](https://img.shields.io/badge/-Project-blue)](https://liylo.github.io/langflash.github.io)  
  <details><summary>Abstract</summary>

  We present LangFlash, a feed-forward framework for 3D Language Gaussian Splatting that reconstructs 3D scenes parameterized by Gaussian primitives enriched with language-aligned semantic features from sparse unposed multi-view images. Unlike optimization-based 3D methods, LangFlash directly predicts the geometry and semantics in a single forward pass, enabling low-latency 3D reconstruction and language-consistent scene understanding. To support large-scale training, we enriched the RealEstate10k dataset with coherent and dense semantic information for 3D semantic supervision. Furthermore, we propose a sparse semantic encoding scheme that combines a global semantic dictionary with locally varying per-primitive weights, preserving high-level linguistic information, while reducing representation complexity. Experimental results show that LangFlash achieves superior novel view synthesis and semantic consistency compared with previous methods. This study establishes a new paradigm for pose-free, language-grounded 3D scene reconstruction, advancing generalizable 3D vision and multimodal scene understanding. Demo is available at https://liylo.github.io/langflash.github.io/.
  </details>  
  Keywords: 3d scene reconstruction, novel view synthesis  

### Surface & Appearance Modeling

- **[Advances in Neural 3D Mesh Texturing: A Survey](https://arxiv.org/abs/2606.00137v1)**  
  Authors: Sai Raj Kishore Perla, Hao Zhang, Ali Mahdavi-Amiri  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2606.00137v1.pdf)  
  <details><summary>Abstract</summary>

  Texturing 3D meshes plays a vital role in determining the visual realism of digital objects and scenes. Although recent generative 3D approaches based on Neural Radiance Fields and Gaussian Splatting can produce textured assets directly, polygonal meshes remain the core representation across modeling, animation, visual effects, and gaming pipelines. Neural 3D mesh texturing therefore continues to be an essential and active area of research. In this survey, we present a comprehensive review of recent advances in neural 3D mesh texturing, covering methods for texture synthesis, transfer, and completion. We first summarize key foundations in mesh geometry, texture mapping, differentiable rendering, and neural generative models, and then organize the literature into a unified taxonomy spanning early GAN-based methods to modern diffusion-based pipelines. We further analyze common architectures and supervision strategies, review datasets and evaluation protocols, and discuss emerging applications, practical/commercial systems, and open challenges. Together, these insights provide a structured perspective...
  </details>  
  Keywords: texture synthesis  
- **[Perceptually Lossless Tactile Texture Synthesis with Compact Spectral Envelope Models](https://arxiv.org/abs/2605.23804v1)**  
  Authors: Jagan K. Balasubramanian, Yasemin Vardar  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2605.23804v1.pdf)  
  <details><summary>Abstract</summary>

  Modern audio-visual media rely on compact representations for efficient storage and transmission, whereas realistic digital touch still depends on high-resolution tactile recordings. Existing approaches for representing tactile signals constrain manipulation and limit the generation of new content. Here, we introduce two compact representations, spectral beta and spectral slope, that capture the temporal spectral structure of finger-surface friction signals while preserving perceptually relevant information. Spectral beta models spectral skewness using a two-parameter beta distribution, whereas spectral slope approximates the spectrum with an asymmetric bandpass filter defined by low- and high-pass orders. We evaluated these representations in a perceptual study with 14 participants using five virtual textures rendered on a friction-modulation display and compared them with physical textures and high-fidelity reproductions of recorded signals. Spectral beta achieved perceptual similarity ratings comparable to those of the original high-fidelity reproductions. Regression analysis further showed that matching spectral energy across nine critical frequency bands was...
  </details>  
  Keywords: texture synthesis  
- **[Generative Texture Diversification of 3D Pedestrians for Robust Autonomous Driving Perception](https://arxiv.org/abs/2605.13755v1)**  
  Authors: Arka Bhowmick, Enes Ozeren, Ahmed Abdullah, Oliver Wasenmuller  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2605.13755v1.pdf)  
  <details><summary>Abstract</summary>

  In recent years, autonomous driving has significantly in creased the demand for high-quality data to train 2D and 3D perception models for safety-critical scenarios. Real world datasets struggle to meet this demand as require ments continuously evolve and large-scale annotated data collection remains costly and time-consuming making syn thetic data a scalable, practical and controllable alterna tive. Pedestrian detection is among the most safety-critical tasks in autonomous driving. In this paper, we propose a simple yet effective method for scaling variability in 3D pedestrian assets for synthetic scene generation. Starting from a single 3D base asset, we generate multiple distinct pedestrian instances by synthesizing diverse facial textures and identity-level appearance variations using StyleGAN2 and automatically mapping them onto 3D meshes. This ap proach enables scalable appearance-level asset diversifica tion without requiring the design of new geometries for each instance. Using the assets, we construct synthetic datasets and study the impact...
  </details>  
  Keywords: texture synthesis  
- **[Texture Regenerating and Grafting Using Genome-Driven Neural Cellular Automata](https://arxiv.org/abs/2605.13630v1)**  
  Authors: Mirela-Magdalena Catrina, Ioana Cristina Plajer, Alexandra Băicoianu  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2605.13630v1.pdf)  
  <details><summary>Abstract</summary>

  This study significantly advances multi-texture synthesis using Neural Cellular Automata (NCAs) by introducing a novel training methodology that enables robust self-regeneration of textures in damaged regions. This inherent healing mechanism, essential for dynamic and adaptive systems, extends beyond traditional computer graphics applications, highlighting the fundamental self-organizing properties of NCAs. Furthermore, we present a versatile grafting technique, enabling the seamless combination of distinct textures. This is achieved efficiently during the inference phase, without requiring specialized retraining, through precise initialization of the NCA's genome channels. Our findings demonstrate the generation of high-quality, complex textures with fluid transitions, showcasing a powerful and efficient paradigm for dynamic texture composition and self-repair in autonomous systems.
  </details>  
  Keywords: texture synthesis  
- **[Noise-Started One-Step Real-World Super-Resolution via LR-Conditioned SplitMeanFlow and GAN Refinement](https://arxiv.org/abs/2605.09328v1)**  
  Authors: Wei Zhu, Kai Zhang, Yu Zheng, Lei Luo, Yong Guo, Jian Yang  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2605.09328v1.pdf)  
  <details><summary>Abstract</summary>

  Pre-trained text-to-image (T2I) diffusion models have shown strong potential for real-world image super-resolution (Real-ISR), owing to their noise-started generation process that enables realistic texture synthesis and captures the one-to-many nature of super-resolution. However, diffusion-based Real-ISR methods still face a fundamental efficiency-quality trade-off. Multi-step methods generate high-quality results by iteratively denoising random Gaussian noise under LR conditioning, but suffer from slow sampling. Recent one-step methods greatly improve efficiency, yet they typically replace noise-started generation with direct LR-to-HR restoration, which weakens stochasticity and limits realistic detail synthesis. To address this issue, we propose SMFSR, a noise-started one-step Real-ISR framework via LR-conditioned SplitMeanFlow and GAN refinement. SMFSR preserves the random-noise starting point of diffusion models and learns a direct noise-to-HR mapping conditioned on the LR image. To this end, Interval Splitting Consistency distills the multi-step generative trajectory into a single average-velocity prediction, enabling efficient one-step generation. To compensate for the reduced opportunity for...
  </details>  
  Keywords: texture synthesis  
- **[HairGPT: Strand-as-Language Autoregressive Modeling for Realistic 3D Hairstyle Synthesis](https://arxiv.org/abs/2605.08824v1)**  
  Authors: Haimin Luo, Min Ouyang, Lan Xu, Jingyi Yu  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2605.08824v1.pdf) | [![Project](https://img.shields.io/badge/-Project-blue)](https://haiminluo.github.io/hairgpt)  
  <details><summary>Abstract</summary>

  Hair is a rich medium of visual and cultural expression, yet its digital modeling remains challenging due to the duality of fluidity and structure. Many existing generative approaches rely primarily on continuous diffusion fields, which entangle global topology with local texture and obscure the semantic and structural organization of hairstyles. To address this, we propose HairGPT, a strand-centric framework that treats strands as generative primitives and formulates realistic 3D hairstyle synthesis as a dual-decoupled autoregressive sequence modeling problem. Our method applies spatial decoupling across semantic scalp regions and structural decoupling along a hierarchical strand representation, progressing from global layout to fine-grained style. We further introduce a geometric tokenizer and region-aware semantic annotations to guide strand-level generation, enabling compositional editing, synthesis of rare and complex hairstyles, and adaptation to stylized domains. By aligning generative modeling with the workflow of digital grooming, HairGPT turns hair generation from opaque texture synthesis into a...
  </details>  
  Keywords: texture synthesis  
- **[TAPESTRY: From Geometry to Appearance via Consistent Turntable Videos](https://arxiv.org/abs/2603.17735v1)**  
  Authors: Yan Zeng, Haoran Jiang, Kaixin Yao, Qixuan Zhang, Longwen Zhang, Lan Xu, Jingyi Yu  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2603.17735v1.pdf)  
  <details><summary>Abstract</summary>

  Automatically generating photorealistic and self-consistent appearances for untextured 3D models is a critical challenge in digital content creation. The advancement of large-scale video generation models offers a natural approach: directly synthesizing 360-degree turntable videos (TTVs), which can serve not only as high-quality dynamic previews but also as an intermediate representation to drive texture synthesis and neural rendering. However, existing general-purpose video diffusion models struggle to maintain strict geometric consistency and appearance stability across the full range of views, making their outputs ill-suited for high-quality 3D reconstruction. To this end, we introduce TAPESTRY, a framework for generating high-fidelity TTVs conditioned on explicit 3D geometry. We reframe the 3D appearance generation task as a geometry-conditioned video diffusion problem: given a 3D mesh, we first render and encode multi-modal geometric features to constrain the video generation process with pixel-level precision, thereby enabling the creation of high-quality and consistent TTVs. Building upon this, we...
  </details>  
  Keywords: texture synthesis  
- **[OAHuman: Occlusion-Aware 3D Human Reconstruction from Monocular Images](https://arxiv.org/abs/2603.14249v1)**  
  Authors: Yuanwang Yang, Hongliang Liu, Muxin Zhang, Nan Ma, Jingyu Yang, Yu-Kun Lai, Kun Li  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2603.14249v1.pdf)  
  <details><summary>Abstract</summary>

  Monocular 3D human reconstruction in real-world scenarios remains highly challenging due to frequent occlusions from surrounding objects, people, or image truncation. Such occlusions lead to missing geometry and unreliable appearance cues, severely degrading the completeness and realism of reconstructed human models. Although recent neural implicit methods achieve impressive results on clean inputs, they struggle under occlusion due to entangled modeling of shape and texture. In this paper, we propose OAHuman, an occlusion-aware framework that explicitly decouples geometry reconstruction and texture synthesis for robust 3D human modeling from a single RGB image. The core innovation lies in the decoupling-perception paradigm, which addresses the fundamental issue of geometry-texture cross-contamination in occluded regions. Our framework ensures that geometry reconstruction is perceptually reinforced even in occluded areas, isolating it from texture interference. In parallel, texture synthesis is learned exclusively from visible regions, preventing texture errors from being transferred to the occluded areas. This decoupling...
  </details>  
  Keywords: texture synthesis  
- **[GarmentPainter: Efficient 3D Garment Texture Synthesis with Character-Guided Diffusion Model](https://arxiv.org/abs/2603.08228v1)**  
  Authors: Jinbo Wu, Xiaobo Gao, Xing Liu, Chen Zhao, Jialun Liu  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2603.08228v1.pdf)  
  <details><summary>Abstract</summary>

  Generating high-fidelity, 3D-consistent garment textures remains a challenging problem due to the inherent complexities of garment structures and the stringent requirement for detailed, globally consistent texture synthesis. Existing approaches either rely on 2D-based diffusion models, which inherently struggle with 3D consistency, require expensive multi-step optimization or depend on strict spatial alignment between 2D reference images and 3D meshes, which limits their flexibility and scalability. In this work, we introduce GarmentPainter, a simple yet efficient framework for synthesizing high-quality, 3D-aware garment textures in UV space. Our method leverages a UV position map as the 3D structural guidance, ensuring texture consistency across the garment surface during texture generation. To enhance control and adaptability, we introduce a type selection module, enabling fine-grained texture generation for specific garment components based on a character reference image, without requiring alignment between the reference image and the 3D mesh. GarmentPainter efficiently integrates all guidance signals into the...
  </details>  
  Keywords: texture synthesis  
- **[MultiGO++: Monocular 3D Clothed Human Reconstruction via Geometry-Texture Collaboration](https://arxiv.org/abs/2603.04993v1)**  
  Authors: Nanjie Yao, Gangjian Zhang, Wenhao Shen, Jian Shu, Yu Feng, Hao Wang  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2603.04993v1.pdf)  
  <details><summary>Abstract</summary>

  Monocular 3D clothed human reconstruction aims to generate a complete and realistic textured 3D avatar from a single image. Existing methods are commonly trained under multi-view supervision with annotated geometric priors, and during inference, these priors are estimated by the pre-trained network from the monocular input. These methods are constrained by three key limitations: texturally by unavailability of training data, geometrically by inaccurate external priors, and systematically by biased single-modality supervision, all leading to suboptimal reconstruction. To address these issues, we propose a novel reconstruction framework, named MultiGO++, which achieves effective systematic geometry-texture collaboration. It consists of three core parts: (1) A multi-source texture synthesis strategy that constructs 15,000+ 3D textured human scans to improve the performance on texture quality estimation in challenge scenarios; (2) A region-aware shape extraction module that extracts and interacts features of each body region to obtain geometry information and a Fourier geometry encoder that mitigates...
  </details>  
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