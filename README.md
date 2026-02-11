# Awesome Gaussian Splatting [![Awesome](https://awesome.re/badge.svg)](https://awesome.re)

A curated list of latest research papers, projects and resources related to Gaussian Splatting. Content is automatically updated daily.

> Last Update: 2026-02-11 01:08:25

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
- [Scene & View Synthesis](#scene-&-view-synthesis) (115 papers) - Focus on reconstructing complex scenes or synthesizing new perspectives from limited data.
- [Surface & Appearance Modeling](#surface-&-appearance-modeling) (25 papers) - Techniques for generating realistic textures, materials, and surface details.



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
- **[Unsupervised Representation Learning for 3D Mesh Parameterization with Semantic and Visibility Objectives](https://arxiv.org/abs/2509.25094v2)**  
  Authors: AmirHossein Zamani, Bruno Roy, Arianna Rampini  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2509.25094v2.pdf) | [![GitHub](https://img.shields.io/github/stars/AHHHZ975/Semantic-Visibility-UV-Param?style=social)](https://github.com/AHHHZ975/Semantic-Visibility-UV-Param)  
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
- **[UP2You: Fast Reconstruction of Yourself from Unconstrained Photo Collections](https://arxiv.org/abs/2509.24817v1)**  
  Authors: Zeyu Cai, Ziyang Li, Xiaoben Li, Boqian Li, Zeyu Wang, Zhenyu Zhang, Yuliang Xiu  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2509.24817v1.pdf) | [![Project](https://img.shields.io/badge/-Project-blue)](https://zcai0612.github.io/UP2You)  
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
- **[SIE3D: Single-image Expressive 3D Avatar generation via Semantic Embedding and Perceptual Expression Loss](https://arxiv.org/abs/2509.24004v1)**  
  Authors: Zhiqi Huang, Dulongkai Cui, Jinglu Hu  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2509.24004v1.pdf) | [![Project](https://img.shields.io/badge/-Project-blue)](https://blazingcrystal1747.github.io/SIE3D)  
  <details><summary>Abstract</summary>

  Generating high-fidelity 3D head avatars from a single image is challenging, as current methods lack fine-grained, intuitive control over expressions via text. This paper proposes SIE3D, a framework that generates expressive 3D avatars from a single image and descriptive text. SIE3D fuses identity features from the image with semantic embedding from text through a novel conditioning scheme, enabling detailed control. To ensure generated expressions accurately match the text, it introduces an innovative perceptual expression loss function. This loss uses a pre-trained expression classifier to regularize the generation process, guaranteeing expression accuracy. Extensive experiments show SIE3D significantly improves controllability and realism, outperforming competitive methods in identity preservation and expression fidelity on a single consumer-grade GPU. Project page: https://blazingcrystal1747.github.io/SIE3D/
  </details>  
- **[Towards Fine-Grained Text-to-3D Quality Assessment: A Benchmark and A Two-Stage Rank-Learning Metric](https://arxiv.org/abs/2509.23841v2)**  
  Authors: Bingyang Cui, Yujie Zhang, Qi Yang, Zhu Li, Yiling Xu  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2509.23841v2.pdf) | [![Project](https://img.shields.io/badge/-Project-blue)](https://cbysjtu.github.io/Rank2Score)  
  <details><summary>Abstract</summary>

  Recent advances in Text-to-3D (T23D) generative models have enabled the synthesis of diverse, high-fidelity 3D assets from textual prompts. However, existing challenges restrict the development of reliable T23D quality assessment (T23DQA). First, existing benchmarks are outdated, fragmented, and coarse-grained, making fine-grained metric training infeasible. Moreover, current objective metrics exhibit inherent design limitations, resulting in non-representative feature extraction and diminished metric robustness. To address these limitations, we introduce T23D-CompBench, a comprehensive benchmark for compositional T23D generation. We define five components with twelve sub-components for compositional prompts, which are used to generate 3,600 textured meshes from ten state-of-the-art generative models. A large-scale subjective experiment is conducted to collect 129,600 reliable human ratings across different perspectives. Based on T23D-CompBench, we further propose Rank2Score, an effective evaluator with two-stage training for T23DQA. Rank2Score enhances pairwise training via supervised contrastive regression and curriculum learning in the first stage, and subsequently refines predictions using mean opinion...
  </details>  
  Keywords: text-to-3d  
- **[M3DLayout: A Multi-Source Dataset of 3D Indoor Layouts and Structured Descriptions for 3D Generation](https://arxiv.org/abs/2509.23728v2)**  
  Authors: Yiheng Zhang, Zhuojiang Cai, Mingdao Wang, Meitong Guo, Tianxiao Li, Li Lin, Yuwang Wang  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2509.23728v2.pdf)  
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
- **[ZeroScene: A Zero-Shot Framework for 3D Scene Generation from a Single Image and Controllable Texture Editing](https://arxiv.org/abs/2509.23607v1)**  
  Authors: Xiang Tang, Ruotong Li, Xiaopeng Fan  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2509.23607v1.pdf)  
  <details><summary>Abstract</summary>

  In the field of 3D content generation, single image scene reconstruction methods still struggle to simultaneously ensure the quality of individual assets and the coherence of the overall scene in complex environments, while texture editing techniques often fail to maintain both local continuity and multi-view consistency. In this paper, we propose a novel system ZeroScene, which leverages the prior knowledge of large vision models to accomplish both single image-to-3D scene reconstruction and texture editing in a zero-shot manner. ZeroScene extracts object-level 2D segmentation and depth information from input images to infer spatial relationships within the scene. It then jointly optimizes 3D and 2D projection losses of the point cloud to update object poses for precise scene alignment, ultimately constructing a coherent and complete 3D scene that encompasses both foreground and background. Moreover, ZeroScene supports texture editing of objects in the scene. By imposing constraints on the diffusion model and introducing...
  </details>  
  Keywords: 3d scene reconstruction, image-to-3d  
- **[FM-SIREN & FM-FINER: Nyquist-Informed Frequency Multiplier for Implicit Neural Representation with Periodic Activation](https://arxiv.org/abs/2509.23438v2)**  
  Authors: Mohammed Alsakabi, Wael Mobeirek, John M. Dolan, Ozan K. Tonguz  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2509.23438v2.pdf)  
  <details><summary>Abstract</summary>

  Existing periodic activation-based implicit neural representation (INR) networks, such as SIREN and FINER, suffer from hidden feature redundancy, where neurons within a layer capture overlapping frequency components due to the use of a fixed frequency multiplier. This redundancy limits the expressive capacity of multilayer perceptrons (MLPs). Drawing inspiration from classical signal processing methods such as the Discrete Sine Transform (DST), we propose FM-SIREN and FM-FINER, which assign Nyquist-informed, neuron-specific frequency multipliers to periodic activations. Unlike existing approaches, our design introduces frequency diversity without requiring hyperparameter tuning or additional network depth. This simple yet principled modification reduces the redundancy of features by nearly 50% and consistently improves signal reconstruction across diverse INR tasks, including fitting 1D audio, 2D image and 3D shape, and synthesis of neural radiance fields (NeRF), outperforming their baseline counterparts while maintaining efficiency.
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
- **[Finding 3D Positions of Distant Objects from Noisy Camera Movement and Semantic Segmentation Sequences](https://arxiv.org/abs/2509.20906v1)**  
  Authors: Julius Pesonen, Arno Solin, Eija Honkavaara  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2509.20906v1.pdf)  
  <details><summary>Abstract</summary>

  3D object localisation based on a sequence of camera measurements is essential for safety-critical surveillance tasks, such as drone-based wildfire monitoring. Localisation of objects detected with a camera can typically be solved with dense depth estimation or 3D scene reconstruction. However, in the context of distant objects or tasks limited by the amount of available computational resources, neither solution is feasible. In this paper, we show that the task can be solved using particle filters for both single and multiple target scenarios. The method was studied using a 3D simulation and a drone-based image segmentation sequence with global navigation satellite system (GNSS)-based camera pose estimates. The results showed that a particle filter can be used to solve practical localisation tasks based on camera poses and image segments in these situations where other solutions fail. The particle filter is independent of the detection method, making it flexible for new tasks. The...
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
- **[Seeing Through Reflections: Advancing 3D Scene Reconstruction in Mirror-Containing Environments with Gaussian Splatting](https://arxiv.org/abs/2509.18956v1)**  
  Authors: Zijing Guo, Yunyang Zhao, Lin Wang  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2509.18956v1.pdf)  
  <details><summary>Abstract</summary>

  Mirror-containing environments pose unique challenges for 3D reconstruction and novel view synthesis (NVS), as reflective surfaces introduce view-dependent distortions and inconsistencies. While cutting-edge methods such as Neural Radiance Fields (NeRF) and 3D Gaussian Splatting (3DGS) excel in typical scenes, their performance deteriorates in the presence of mirrors. Existing solutions mainly focus on handling mirror surfaces through symmetry mapping but often overlook the rich information carried by mirror reflections. These reflections offer complementary perspectives that can fill in absent details and significantly enhance reconstruction quality. To advance 3D reconstruction in mirror-rich environments, we present MirrorScene3D, a comprehensive dataset featuring diverse indoor scenes, 1256 high-quality images, and annotated mirror masks, providing a benchmark for evaluating reconstruction methods in reflective settings. Building on this, we propose ReflectiveGS, an extension of 3D Gaussian Splatting that utilizes mirror reflections as complementary viewpoints rather than simple symmetry artifacts, enhancing scene geometry and recovering absent details. Experiments...
  </details>  
  Keywords: novel view synthesis, 3d scene reconstruction  
- **[Hierarchical Neural Semantic Representation for 3D Semantic Correspondence](https://arxiv.org/abs/2509.17431v2)**  
  Authors: Keyu Du, Jingyu Hu, Haipeng Li, Hao Xu, Haibing Huang, Chi-Wing Fu, Shuaicheng Liu  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2509.17431v2.pdf)  
  <details><summary>Abstract</summary>

  This paper presents a new approach to estimate accurate and robust 3D semantic correspondence with the hierarchical neural semantic representation. Our work has three key contributions. First, we design the hierarchical neural semantic representation (HNSR), which consists of a global semantic feature to capture high-level structure and multi-resolution local geometric features to preserve fine details, by carefully harnessing 3D priors from pre-trained 3D generative models. Second, we design a progressive global-to-local matching strategy, which establishes coarse semantic correspondence using the global semantic feature, then iteratively refines it with local geometric features, yielding accurate and semantically-consistent mappings. Third, our framework is training-free and broadly compatible with various pre-trained 3D generative backbones, demonstrating strong generalization across diverse shape categories. Our method also supports various applications, such as shape co-segmentation, keypoint matching, and texture transfer, and generalizes well to structurally diverse shapes, with promising results even in cross-category scenarios. Both qualitative and quantitative...
  </details>  
- **[SmokeSeer: 3D Gaussian Splatting for Smoke Removal and Scene Reconstruction](https://arxiv.org/abs/2509.17329v3)**  
  Authors: Neham Jain, Andrew Jong, Sebastian Scherer, Ioannis Gkioulekas  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2509.17329v3.pdf)  
  <details><summary>Abstract</summary>

  Smoke in real-world scenes can severely degrade image quality and hamper visibility. Recent image restoration methods either rely on data-driven priors that are susceptible to hallucinations, or are limited to static low-density smoke. We introduce SmokeSeer, a method for simultaneous 3D scene reconstruction and smoke removal from multi-view video sequences. Our method uses thermal and RGB images, leveraging the reduced scattering in thermal images to see through smoke. We build upon 3D Gaussian splatting to fuse information from the two image modalities, and decompose the scene into smoke and non-smoke components. Unlike prior work, SmokeSeer handles a broad range of smoke densities and adapts to temporally varying smoke. We validate our method on synthetic data and a new real-world smoke dataset with RGB and thermal images. We provide an open-source implementation and data on the project website.
  </details>  
  Keywords: 3d scene reconstruction  
- **[DT-NeRF: A Diffusion and Transformer-Based Optimization Approach for Neural Radiance Fields in 3D Reconstruction](https://arxiv.org/abs/2509.17232v1)**  
  Authors: Bo Liu, Runlong Li, Li Zhou, Yan Zhou  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2509.17232v1.pdf)  
  <details><summary>Abstract</summary>

  This paper proposes a Diffusion Model-Optimized Neural Radiance Field (DT-NeRF) method, aimed at enhancing detail recovery and multi-view consistency in 3D scene reconstruction. By combining diffusion models with Transformers, DT-NeRF effectively restores details under sparse viewpoints and maintains high accuracy in complex geometric scenes. Experimental results demonstrate that DT-NeRF significantly outperforms traditional NeRF and other state-of-the-art methods on the Matterport3D and ShapeNet datasets, particularly in metrics such as PSNR, SSIM, Chamfer Distance, and Fidelity. Ablation experiments further confirm the critical role of the diffusion and Transformer modules in the model's performance, with the removal of either module leading to a decline in performance. The design of DT-NeRF showcases the synergistic effect between modules, providing an efficient and accurate solution for 3D scene reconstruction. Future research may focus on further optimizing the model, exploring more advanced generative models and network architectures to enhance its performance in large-scale dynamic scenes.
  </details>  
  Keywords: 3d scene reconstruction  
- **[Efficient 3D Scene Reconstruction and Simulation from Sparse Endoscopic Views](https://arxiv.org/abs/2509.17027v1)**  
  Authors: Zhenya Yang  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2509.17027v1.pdf)  
  <details><summary>Abstract</summary>

  Surgical simulation is essential for medical training, enabling practitioners to develop crucial skills in a risk-free environment while improving patient safety and surgical outcomes. However, conventional methods for building simulation environments are cumbersome, time-consuming, and difficult to scale, often resulting in poor details and unrealistic simulations. In this paper, we propose a Gaussian Splatting-based framework to directly reconstruct interactive surgical scenes from endoscopic data while ensuring efficiency, rendering quality, and realism. A key challenge in this data-driven simulation paradigm is the restricted movement of endoscopic cameras, which limits viewpoint diversity. As a result, the Gaussian Splatting representation overfits specific perspectives, leading to reduced geometric accuracy. To address this issue, we introduce a novel virtual camera-based regularization method that adaptively samples virtual viewpoints around the scene and incorporates them into the optimization process to mitigate overfitting. An effective depth-based regularization is applied to both real and virtual views to further refine...
  </details>  
  Keywords: 3d scene reconstruction  
- **[MMPart: Harnessing Multi-Modal Large Language Models for Part-Aware 3D Generation](https://arxiv.org/abs/2509.16768v1)**  
  Authors: Omid Bonakdar, Nasser Mozayani  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2509.16768v1.pdf)  
  <details><summary>Abstract</summary>

  Generative 3D modeling has advanced rapidly, driven by applications in VR/AR, metaverse, and robotics. However, most methods represent the target object as a closed mesh devoid of any structural information, limiting editing, animation, and semantic understanding. Part-aware 3D generation addresses this problem by decomposing objects into meaningful components, but existing pipelines face challenges: in existing methods, the user has no control over which objects are separated and how model imagine the occluded parts in isolation phase. In this paper, we introduce MMPart, an innovative framework for generating part-aware 3D models from a single image. We first use a VLM to generate a set of prompts based on the input image and user descriptions. In the next step, a generative model generates isolated images of each object based on the initial image and the previous step's prompts as supervisor (which control the pose and guide model how imagine previously occluded areas)....
  </details>  
- **[Pain in 3D: Generating Controllable Synthetic Faces for Automated Pain Assessment](https://arxiv.org/abs/2509.16727v4)**  
  Authors: Xin Lei Lin, Soroush Mehraban, Abhishek Moturu, Babak Taati  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2509.16727v4.pdf)  
  <details><summary>Abstract</summary>

  Automated pain assessment from facial expressions is crucial for non-communicative patients, such as those with dementia. Progress has been limited by two challenges: (i) existing datasets exhibit severe demographic and label imbalance due to ethical constraints, and (ii) current generative models cannot precisely control facial action units (AUs), facial structure, or clinically validated pain levels. We present 3DPain, a large-scale synthetic dataset specifically designed for automated pain assessment, featuring unprecedented annotation richness and demographic diversity. Our three-stage framework generates diverse 3D meshes, textures them with diffusion models, and applies AU-driven face rigging to synthesize multi-view faces with paired neutral and pain images, AU configurations, PSPI scores, and the first dataset-level annotations of pain-region heatmaps. The dataset comprises 82,500 samples across 25,000 pain expression heatmaps and 2,500 synthetic identities balanced by age, gender, and ethnicity. We further introduce ViTPain, a Vision Transformer based cross-modal distillation framework in which a heatmap-trained teacher...
  </details>  
  Keywords: rigging  
- **[Vision-Language Models as Differentiable Semantic and Spatial Rewards for Text-to-3D Generation](https://arxiv.org/abs/2509.15772v1)**  
  Authors: Weimin Bai, Yubo Li, Weijian Luo, Wenzheng Chen, He Sun  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2509.15772v1.pdf)  
  <details><summary>Abstract</summary>

  Score Distillation Sampling (SDS) enables high-quality text-to-3D generation by supervising 3D models through the denoising of multi-view 2D renderings, using a pretrained text-to-image diffusion model to align with the input prompt and ensure 3D consistency. However, existing SDS-based methods face two fundamental limitations: (1) their reliance on CLIP-style text encoders leads to coarse semantic alignment and struggles with fine-grained prompts; and (2) 2D diffusion priors lack explicit 3D spatial constraints, resulting in geometric inconsistencies and inaccurate object relationships in multi-object scenes. To address these challenges, we propose VLM3D, a novel text-to-3D generation framework that integrates large vision-language models (VLMs) into the SDS pipeline as differentiable semantic and spatial priors. Unlike standard text-to-image diffusion priors, VLMs leverage rich language-grounded supervision that enables fine-grained prompt alignment. Moreover, their inherent vision language modeling provides strong spatial understanding, which significantly enhances 3D consistency for single-object generation and improves relational reasoning in multi-object scenes. We...
  </details>  
  Keywords: text-to-3d  
- **[AToken: A Unified Tokenizer for Vision](https://arxiv.org/abs/2509.14476v2)**  
  Authors: Jiasen Lu, Liangchen Song, Mingze Xu, Byeongjoo Ahn, Yanjun Wang, Chen Chen, Afshin Dehghan, Yinfei Yang  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2509.14476v2.pdf)  
  <details><summary>Abstract</summary>

  We present AToken, the first unified visual tokenizer that achieves both high-fidelity reconstruction and semantic understanding across images, videos, and 3D assets. Unlike existing tokenizers that specialize in either reconstruction or understanding for single modalities, AToken encodes these diverse visual inputs into a shared 4D latent space, unifying both tasks and modalities in a single framework. Specifically, we introduce a pure transformer architecture with 4D rotary position embeddings to process visual inputs of arbitrary resolutions and temporal durations. To ensure stable training, we introduce an adversarial-free training objective that combines perceptual and Gram matrix losses, achieving state-of-the-art reconstruction quality. By employing a progressive training curriculum, AToken gradually expands from single images, videos, and 3D, and supports both continuous and discrete latent tokens. AToken achieves 0.21 rFID with 82.2% ImageNet accuracy for images, 3.01 rFVD with 40.2% MSRVTT retrieval for videos, and 28.28 PSNR with 90.9% classification accuracy for 3D.. In...
  </details>  
  Keywords: image-to-3d  
- **[CraftMesh: High-Fidelity Generative Mesh Manipulation via Poisson Seamless Fusion](https://arxiv.org/abs/2509.13688v2)**  
  Authors: James Jincheng, Yuxiao Wu, Youcheng Cai, Ligang Liu  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2509.13688v2.pdf)  
  <details><summary>Abstract</summary>

  Controllable, high-fidelity mesh editing remains a significant challenge in 3D content creation. Existing generative methods often struggle with complex geometries and fail to produce detailed results. We propose CraftMesh, a novel framework for high-fidelity generative mesh manipulation via Poisson Seamless Fusion. Our key insight is to decompose mesh editing into a pipeline that leverages the strengths of 2D and 3D generative models: we edit a 2D reference image, then generate a region-specific 3D mesh, and seamlessly fuse it into the original model. We introduce two core techniques: Poisson Geometric Fusion, which utilizes a hybrid SDF/Mesh representation with normal blending to achieve harmonious geometric integration, and Poisson Texture Harmonization for visually consistent texture blending. Experimental results demonstrate that CraftMesh outperforms state-of-the-art methods, delivering superior global consistency and local detail in complex editing tasks.
  </details>  
- **[Gaussian Alignment for Relative Camera Pose Estimation via Single-View Reconstruction](https://arxiv.org/abs/2509.13652v1)**  
  Authors: Yumin Li, Dylan Campbell  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2509.13652v1.pdf)  
  <details><summary>Abstract</summary>

  Estimating metric relative camera pose from a pair of images is of great importance for 3D reconstruction and localisation. However, conventional two-view pose estimation methods are not metric, with camera translation known only up to a scale, and struggle with wide baselines and textureless or reflective surfaces. This paper introduces GARPS, a training-free framework that casts this problem as the direct alignment of two independently reconstructed 3D scenes. GARPS leverages a metric monocular depth estimator and a Gaussian scene reconstructor to obtain a metric 3D Gaussian Mixture Model (GMM) for each image. It then refines an initial pose from a feed-forward two-view pose estimator by optimising a differentiable GMM alignment objective. This objective jointly considers geometric structure, view-independent colour, anisotropic covariance, and semantic feature consistency, and is robust to occlusions and texture-poor regions without requiring explicit 2D correspondences. Extensive experiments on the Real\-Estate10K dataset demonstrate that GARPS outperforms both classical...
  </details>  
  Keywords: single-view reconstruction  
- **[Semantic-Enhanced Cross-Modal Place Recognition for Robust Robot Localization](https://arxiv.org/abs/2509.13474v1)**  
  Authors: Yujia Lin, Nicholas Evans  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2509.13474v1.pdf)  
  <details><summary>Abstract</summary>

  Ensuring accurate localization of robots in environments without GPS capability is a challenging task. Visual Place Recognition (VPR) techniques can potentially achieve this goal, but existing RGB-based methods are sensitive to changes in illumination, weather, and other seasonal changes. Existing cross-modal localization methods leverage the geometric properties of RGB images and 3D LiDAR maps to reduce the sensitivity issues highlighted above. Currently, state-of-the-art methods struggle in complex scenes, fine-grained or high-resolution matching, and situations where changes can occur in viewpoint. In this work, we introduce a framework we call Semantic-Enhanced Cross-Modal Place Recognition (SCM-PR) that combines high-level semantics utilizing RGB images for robust localization in LiDAR maps. Our proposed method introduces: a VMamba backbone for feature extraction of RGB images; a Semantic-Aware Feature Fusion (SAFF) module for using both place descriptors and segmentation masks; LiDAR descriptors that incorporate both semantics and geometry; and a cross-modal semantic attention mechanism in NetVLAD...
  </details>  
- **[StyleSculptor: Zero-Shot Style-Controllable 3D Asset Generation with Texture-Geometry Dual Guidance](https://arxiv.org/abs/2509.13301v2)**  
  Authors: Zefan Qu, Zhenwei Wang, Haoyuan Wang, Ke Xu, Gerhard Hancke, Rynson W. H. Lau  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2509.13301v2.pdf)  
  <details><summary>Abstract</summary>

  Creating 3D assets that follow the texture and geometry style of existing ones is often desirable or even inevitable in practical applications like video gaming and virtual reality. While impressive progress has been made in generating 3D objects from text or images, creating style-controllable 3D assets remains a complex and challenging problem. In this work, we propose StyleSculptor, a novel training-free approach for generating style-guided 3D assets from a content image and one or more style images. Unlike previous works, StyleSculptor achieves style-guided 3D generation in a zero-shot manner, enabling fine-grained 3D style control that captures the texture, geometry, or both styles of user-provided style images. At the core of StyleSculptor is a novel Style Disentangled Attention (SD-Attn) module, which establishes a dynamic interaction between the input content image and style image for style-guided 3D asset generation via a cross-3D attention mechanism, enabling stable feature fusion and effective style-guided generation....
  </details>  
  Keywords: 3d asset generation  
- **[Dream3DAvatar: Text-Controlled 3D Avatar Reconstruction from a Single Image](https://arxiv.org/abs/2509.13013v1)**  
  Authors: Gaofeng Liu, Hengsen Li, Ruoyu Gao, Xuetong Li, Zhiyuan Ma, Tao Fang  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2509.13013v1.pdf)  
  <details><summary>Abstract</summary>

  With the rapid advancement of 3D representation techniques and generative models, substantial progress has been made in reconstructing full-body 3D avatars from a single image. However, this task remains fundamentally ill-posedness due to the limited information available from monocular input, making it difficult to control the geometry and texture of occluded regions during generation. To address these challenges, we redesign the reconstruction pipeline and propose Dream3DAvatar, an efficient and text-controllable two-stage framework for 3D avatar generation. In the first stage, we develop a lightweight, adapter-enhanced multi-view generation model. Specifically, we introduce the Pose-Adapter to inject SMPL-X renderings and skeletal information into SDXL, enforcing geometric and pose consistency across views. To preserve facial identity, we incorporate ID-Adapter-G, which injects high-resolution facial features into the generation process. Additionally, we leverage BLIP2 to generate high-quality textual descriptions of the multi-view images, enhancing text-driven controllability in occluded regions. In the second stage, we design...
  </details>  
- **[Improving Accuracy and Efficiency of Implicit Neural Representations: Making SIREN a WINNER](https://arxiv.org/abs/2509.12980v1)**  
  Authors: Hemanth Chandravamsi, Dhanush V. Shenoy, Steven H. Frankel  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2509.12980v1.pdf)  
  <details><summary>Abstract</summary>

  We identify and address a fundamental limitation of sinusoidal representation networks (SIRENs), a class of implicit neural representations. SIRENs Sitzmann et al. (2020), when not initialized appropriately, can struggle at fitting signals that fall outside their frequency support. In extreme cases, when the network's frequency support misaligns with the target spectrum, a 'spectral bottleneck' phenomenon is observed, where the model yields to a near-zero output and fails to recover even the frequency components that are within its representational capacity. To overcome this, we propose WINNER - Weight Initialization with Noise for Neural Representations. WINNER perturbs uniformly initialized weights of base SIREN with Gaussian noise - whose noise scales are adaptively determined by the spectral centroid of the target signal. Similar to random Fourier embeddings, this mitigates 'spectral bias' but without introducing additional trainable parameters. Our method achieves state-of-the-art audio fitting and significant gains in image and 3D shape fitting tasks...
  </details>  
- **[Hunyuan3D Studio: End-to-End AI Pipeline for Game-Ready 3D Asset Generation](https://arxiv.org/abs/2509.12815v1)**  
  Authors: Biwen Lei, Yang Li, Xinhai Liu, Shuhui Yang, Lixin Xu, Jingwei Huang, Ruining Tang, Haohan Weng, Jian Liu, Jing Xu, Zhen Zhou, Yiling Zhu, Jiankai Xing, Jiachen Xu, Changfeng Ma, Xinhao Yan, Yunhan Yang, Chunshi Wang, Duoteng Xu, Xueqi Ma, Yuguang Chen, Jing Li, Mingxin Yang, Sheng Zhang, Yifei Feng, Xin Huang, Di Luo, Zebin He, Puhua Jiang, Changrong Hu, Zihan Qin, Shiwei Miao, Haolin Liu, Yunfei Zhao, Zeqiang Lai, Qingxiang Lin, Zibo Zhao, Kunhong Li, Xianghui Yang, Huiwen Shi, Xin Yang, Yuxuan Wang, Zebin Yao, Yihang Lian, Sicong Liu, Xintong Han, Wangchen Qin, Caisheng Ouyang, Jianyin Liu, Tianwen Yuan, Shuai Jiang, Hong Duan, Yanqi Niu, Wencong Lin, Yifu Sun, Shirui Huang, Lin Niu, Gu Gong, Guojian Xiao, Bojian Zheng, Xiang Yuan, Qi Chen, Jie Xiao, Dongyang Zheng, Xiaofeng Yang, Kai Liu, Jianchen Zhu, Lifu Wang, Qinglin Lu, Jie Liu, Liang Dong, Fan Jiang, Ruibin Chen, Lei Wang, Chao Zhang, Jiaxin Lin, Hao Zhang, Zheng Ye, Peng He, Runzhou Wu, Yinhe Wu, Jiayao Du, Jupeng Chen, Xinyue Mao, Dongyuan Guo, Yixuan Tang, Yulin Tsai, Yonghao Tan, Jiaao Yu, Junlin Yu, Keren Zhang, Yifan Li, Peng Chen, Tian Liu, Di Wang, Yuhong Liu, Linus, Jie Jiang, Zhuo Chen, Chunchao Guo  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2509.12815v1.pdf)  
  <details><summary>Abstract</summary>

  The creation of high-quality 3D assets, a cornerstone of modern game development, has long been characterized by labor-intensive and specialized workflows. This paper presents Hunyuan3D Studio, an end-to-end AI-powered content creation platform designed to revolutionize the game production pipeline by automating and streamlining the generation of game-ready 3D assets. At its core, Hunyuan3D Studio integrates a suite of advanced neural modules (such as Part-level 3D Generation, Polygon Generation, Semantic UV, etc.) into a cohesive and user-friendly system. This unified framework allows for the rapid transformation of a single concept image or textual description into a fully-realized, production-quality 3D model complete with optimized geometry and high-fidelity PBR textures. We demonstrate that assets generated by Hunyuan3D Studio are not only visually compelling but also adhere to the stringent technical requirements of contemporary game engines, significantly reducing iteration time and lowering the barrier to entry for 3D content creation. By providing a seamless...
  </details>  
  Keywords: 3d asset generation  
- **[SPGen: Spherical Projection as Consistent and Flexible Representation for Single Image 3D Shape Generation](https://arxiv.org/abs/2509.12721v1)**  
  Authors: Jingdong Zhang, Weikai Chen, Yuan Liu, Jionghao Wang, Zhengming Yu, Zhuowen Shen, Bo Yang, Wenping Wang, Xin Li  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2509.12721v1.pdf)  
  <details><summary>Abstract</summary>

  Existing single-view 3D generative models typically adopt multiview diffusion priors to reconstruct object surfaces, yet they remain prone to inter-view inconsistencies and are unable to faithfully represent complex internal structure or nontrivial topologies. In particular, we encode geometry information by projecting it onto a bounding sphere and unwrapping it into a compact and structural multi-layer 2D Spherical Projection (SP) representation. Operating solely in the image domain, SPGen offers three key advantages simultaneously: (1) Consistency. The injective SP mapping encodes surface geometry with a single viewpoint which naturally eliminates view inconsistency and ambiguity; (2) Flexibility. Multi-layer SP maps represent nested internal structures and support direct lifting to watertight or open 3D surfaces; (3) Efficiency. The image-domain formulation allows the direct inheritance of powerful 2D diffusion priors and enables efficient finetuning with limited computational resources. Extensive experiments demonstrate that SPGen significantly outperforms existing baselines in geometric quality and computational efficiency.
  </details>  
- **[Generative AI Pipeline for Interactive Prompt-driven 2D-to-3D Vascular Reconstruction for Fontan Geometries from Contrast-Enhanced X-Ray Fluoroscopy Imaging](https://arxiv.org/abs/2509.13372v1)**  
  Authors: Prahlad G Menon  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2509.13372v1.pdf)  
  <details><summary>Abstract</summary>

  Fontan palliation for univentricular congenital heart disease progresses to hemodynamic failure with complex flow patterns poorly characterized by conventional 2D imaging. Current assessment relies on fluoroscopic angiography, providing limited 3D geometric information essential for computational fluid dynamics (CFD) analysis and surgical planning. A multi-step AI pipeline was developed utilizing Google's Gemini 2.5 Flash (2.5B parameters) for systematic, iterative processing of fluoroscopic angiograms through transformer-based neural architecture. The pipeline encompasses medical image preprocessing, vascular segmentation, contrast enhancement, artifact removal, and virtual hemodynamic flow visualization within 2D projections. Final views were processed through Tencent's Hunyuan3D-2mini (384M parameters) for stereolithography file generation. The pipeline successfully generated geometrically optimized 2D projections from single-view angiograms after 16 processing steps using a custom web interface. Initial iterations contained hallucinated vascular features requiring iterative refinement to achieve anatomically faithful representations. Final projections demonstrated accurate preservation of complex Fontan geometry with enhanced contrast suitable for 3D conversion. AI-generated...
  </details>  
- **[T2Bs: Text-to-Character Blendshapes via Video Generation](https://arxiv.org/abs/2509.10678v2)**  
  Authors: Jiahao Luo, Chaoyang Wang, Michael Vasilkovsky, Vladislav Shakhrai, Di Liu, Peiye Zhuang, Sergey Tulyakov, Peter Wonka, Hsin-Ying Lee, James Davis, Jian Wang  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2509.10678v2.pdf)  
  <details><summary>Abstract</summary>

  We present T2Bs, a framework for generating high-quality, animatable character head morphable models from text by combining static text-to-3D generation with video diffusion. Text-to-3D models produce detailed static geometry but lack motion synthesis, while video diffusion models generate motion with temporal and multi-view geometric inconsistencies. T2Bs bridges this gap by leveraging deformable 3D Gaussian splatting to align static 3D assets with video outputs. By constraining motion with static geometry and employing a view-dependent deformation MLP, T2Bs (i) outperforms existing 4D generation methods in accuracy and expressiveness while reducing video artifacts and view inconsistencies, and (ii) reconstructs smooth, coherent, fully registered 3D geometries designed to scale for building morphable models with diverse, realistic facial motions. This enables synthesizing expressive, animatable character heads that surpass current 4D generation techniques.
  </details>  
  Keywords: text-to-3d  
- **[Geometric Neural Distance Fields for Learning Human Motion Priors](https://arxiv.org/abs/2509.09667v1)**  
  Authors: Zhengdi Yu, Simone Foti, Linguang Zhang, Amy Zhao, Cem Keskin, Stefanos Zafeiriou, Tolga Birdal  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2509.09667v1.pdf)  
  <details><summary>Abstract</summary>

  We introduce Neural Riemannian Motion Fields (NRMF), a novel 3D generative human motion prior that enables robust, temporally consistent, and physically plausible 3D motion recovery. Unlike existing VAE or diffusion-based methods, our higher-order motion prior explicitly models the human motion in the zero level set of a collection of neural distance fields (NDFs) corresponding to pose, transition (velocity), and acceleration dynamics. Our framework is rigorous in the sense that our NDFs are constructed on the product space of joint rotations, their angular velocities, and angular accelerations, respecting the geometry of the underlying articulations. We further introduce: (i) a novel adaptive-step hybrid algorithm for projecting onto the set of plausible motions, and (ii) a novel geometric integrator to "roll out" realistic motion trajectories during test-time-optimization and generation. Our experiments show significant and consistent gains: trained on the AMASS dataset, NRMF remarkably generalizes across multiple input modalities and to diverse tasks ranging...
  </details>  
- **[One View, Many Worlds: Single-Image to 3D Object Meets Generative Domain Randomization for One-Shot 6D Pose Estimation](https://arxiv.org/abs/2509.07978v1)**  
  Authors: Zheng Geng, Nan Wang, Shaocong Xu, Chongjie Ye, Bohan Li, Zhaoxi Chen, Sida Peng, Hao Zhao  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2509.07978v1.pdf) | [![Project](https://img.shields.io/badge/-Project-blue)](https://gzwsama.github.io/OnePoseviaGen.github.io)  
  <details><summary>Abstract</summary>

  Estimating the 6D pose of arbitrary unseen objects from a single reference image is critical for robotics operating in the long-tail of real-world instances. However, this setting is notoriously challenging: 3D models are rarely available, single-view reconstructions lack metric scale, and domain gaps between generated models and real-world images undermine robustness. We propose OnePoseViaGen, a pipeline that tackles these challenges through two key components. First, a coarse-to-fine alignment module jointly refines scale and pose by combining multi-view feature matching with render-and-compare refinement. Second, a text-guided generative domain randomization strategy diversifies textures, enabling effective fine-tuning of pose estimators with synthetic data. Together, these steps allow high-fidelity single-view 3D generation to support reliable one-shot 6D pose estimation. On challenging benchmarks (YCBInEOAT, Toyota-Light, LM-O), OnePoseViaGen achieves state-of-the-art performance far surpassing prior approaches. We further demonstrate robust dexterous grasping with a real robot hand, validating the practicality of our method in real-world manipulation. Project...
  </details>  
  Keywords: single-view reconstruction  
- **[Point Linguist Model: Segment Any Object via Bridged Large 3D-Language Model](https://arxiv.org/abs/2509.07825v1)**  
  Authors: Zhuoxu Huang, Mingqi Gao, Jungong Han  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2509.07825v1.pdf)  
  <details><summary>Abstract</summary>

  3D object segmentation with Large Language Models (LLMs) has become a prevailing paradigm due to its broad semantics, task flexibility, and strong generalization. However, this paradigm is hindered by representation misalignment: LLMs process high-level semantic tokens, whereas 3D point clouds convey only dense geometric structures. In prior methods, misalignment limits both input and output. At the input stage, dense point patches require heavy pre-alignment, weakening object-level semantics and confusing similar distractors. At the output stage, predictions depend only on dense features without explicit geometric cues, leading to a loss of fine-grained accuracy. To address these limitations, we present the Point Linguist Model (PLM), a general framework that bridges the representation gap between LLMs and dense 3D point clouds without requiring large-scale pre-alignment between 3D-text or 3D-images. Specifically, we introduce Object-centric Discriminative Representation (OcDR), which learns object-centric tokens that capture target semantics and scene relations under a hard negative-aware training objective....
  </details>  
- **[DreamLifting: A Plug-in Module Lifting MV Diffusion Models for 3D Asset Generation](https://arxiv.org/abs/2509.07435v1)**  
  Authors: Ze-Xin Yin, Jiaxiong Qiu, Liu Liu, Xinjie Wang, Wei Sui, Zhizhong Su, Jian Yang, Jin Xie  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2509.07435v1.pdf) | [![Project](https://img.shields.io/badge/-Project-blue)](https://zx-yin.github.io/dreamlifting)  
  <details><summary>Abstract</summary>

  The labor- and experience-intensive creation of 3D assets with physically based rendering (PBR) materials demands an autonomous 3D asset creation pipeline. However, most existing 3D generation methods focus on geometry modeling, either baking textures into simple vertex colors or leaving texture synthesis to post-processing with image diffusion models. To achieve end-to-end PBR-ready 3D asset generation, we present Lightweight Gaussian Asset Adapter (LGAA), a novel framework that unifies the modeling of geometry and PBR materials by exploiting multi-view (MV) diffusion priors from a novel perspective. The LGAA features a modular design with three components. Specifically, the LGAA Wrapper reuses and adapts network layers from MV diffusion models, which encapsulate knowledge acquired from billions of images, enabling better convergence in a data-efficient manner. To incorporate multiple diffusion priors for geometry and PBR synthesis, the LGAA Switcher aligns multiple LGAA Wrapper layers encapsulating different knowledge. Then, a tamed variational autoencoder (VAE), termed LGAA...
  </details>  
  Keywords: pbr materials, texture synthesis, 3d asset generation  
- **[SynthDrive: Scalable Real2Sim2Real Sensor Simulation Pipeline for High-Fidelity Asset Generation and Driving Data Synthesis](https://arxiv.org/abs/2509.06798v1)**  
  Authors: Zhengqing Chen, Ruohong Mei, Xiaoyang Guo, Qingjie Wang, Yubin Hu, Wei Yin, Weiqiang Ren, Qian Zhang  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2509.06798v1.pdf)  
  <details><summary>Abstract</summary>

  In the field of autonomous driving, sensor simulation is essential for generating rare and diverse scenarios that are difficult to capture in real-world environments. Current solutions fall into two categories: 1) CG-based methods, such as CARLA, which lack diversity and struggle to scale to the vast array of rare cases required for robust perception training; and 2) learning-based approaches, such as NeuSim, which are limited to specific object categories (vehicles) and require extensive multi-sensor data, hindering their applicability to generic objects. To address these limitations, we propose a scalable real2sim2real system that leverages 3D generation to automate asset mining, generation, and rare-case data synthesis.
  </details>  
- **[3DOF+Quantization: 3DGS quantization for large scenes with limited Degrees of Freedom](https://arxiv.org/abs/2509.06400v1)**  
  Authors: Matthieu Gendrin, Stéphane Pateux, Théo Ladune  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2509.06400v1.pdf)  
  <details><summary>Abstract</summary>

  3D Gaussian Splatting (3DGS) is a major breakthrough in 3D scene reconstruction. With a number of views of a given object or scene, the algorithm trains a model composed of 3D gaussians, which enables the production of novel views from arbitrary points of view. This freedom of movement is referred to as 6DoF for 6 degrees of freedom: a view is produced for any position (3 degrees), orientation of camera (3 other degrees). On large scenes, though, the input views are acquired from a limited zone in space, and the reconstruction is valuable for novel views from the same zone, even if the scene itself is almost unlimited in size. We refer to this particular case as 3DoF+, meaning that the 3 degrees of freedom of camera position are limited to small offsets around the central position. Considering the problem of coordinate quantization, the impact of position error on the...
  </details>  
  Keywords: 3d scene reconstruction  
- **[Learning in ImaginationLand: Omnidirectional Policies through 3D Generative Models (OP-Gen)](https://arxiv.org/abs/2509.06191v1)**  
  Authors: Yifei Ren, Edward Johns  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2509.06191v1.pdf)  
  <details><summary>Abstract</summary>

  Recent 3D generative models, which are capable of generating full object shapes from just a few images, now open up new opportunities in robotics. In this work, we show that 3D generative models can be used to augment a dataset from a single real-world demonstration, after which an omnidirectional policy can be learned within this imagined dataset. We found that this enables a robot to perform a task when initialised from states very far from those observed during the demonstration, including starting from the opposite side of the object relative to the real-world demonstration, significantly reducing the number of demonstrations required for policy learning. Through several real-world experiments across tasks such as grasping objects, opening a drawer, and placing trash into a bin, we study these omnidirectional policies by investigating the effect of various design choices on policy behaviour, and we show superior performance to recent baselines which use alternative...
  </details>  
- **[3D-Image Reconstruction using MIMO-SAR FMCW Radar](https://arxiv.org/abs/2509.05977v1)**  
  Authors: Ayush Jha, Dhanireddy Chandrika, Chandra Sekhar Seelamantula, Chetan Singh Thakur  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2509.05977v1.pdf)  
  <details><summary>Abstract</summary>

  With the advancement of millimeter-wave radar technology, Synthetic Aperture Radar (SAR) imaging at millimeter-wave frequencies has gained significant attention in both academic research and industrial applications. However, traditional SAR imaging algorithms primarily focus on extracting two-dimensional information from detected targets, which limits their potential for 3D scene reconstruction. In this work, we demonstrated a fast time-domain reconstruction algorithm for achieving high-resolution 3D radar imaging at millimeter-wave (mmWave) frequencies. This approach leverages a combination of virtual Multiple Input Multiple Output (MIMO) Frequency Modulated Continuous Wave (FMCW) radar with the precision of Synthetic Aperture Radar (SAR) technique, setting the stage for a new era of advanced radar imaging applications.
  </details>  
  Keywords: 3d scene reconstruction  
- **[Visibility-Aware Language Aggregation for Open-Vocabulary Segmentation in 3D Gaussian Splatting](https://arxiv.org/abs/2509.05515v1)**  
  Authors: Sen Wang, Kunyi Li, Siyun Liang, Elena Alegret, Jing Ma, Nassir Navab, Stefano Gasperini  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2509.05515v1.pdf)  
  <details><summary>Abstract</summary>

  Recently, distilling open-vocabulary language features from 2D images into 3D Gaussians has attracted significant attention. Although existing methods achieve impressive language-based interactions of 3D scenes, we observe two fundamental issues: background Gaussians contributing negligibly to a rendered pixel get the same feature as the dominant foreground ones, and multi-view inconsistencies due to view-specific noise in language embeddings. We introduce Visibility-Aware Language Aggregation (VALA), a lightweight yet effective method that computes marginal contributions for each ray and applies a visibility-aware gate to retain only visible Gaussians. Moreover, we propose a streaming weighted geometric median in cosine space to merge noisy multi-view features. Our method yields a robust, view-consistent language feature embedding in a fast and memory-efficient manner. VALA improves open-vocabulary localization and segmentation across reference datasets, consistently surpassing existing works.
  </details>  
- **[A Scalable Attention-Based Approach for Image-to-3D Texture Mapping](https://arxiv.org/abs/2509.05131v1)**  
  Authors: Arianna Rampini, Kanika Madan, Bruno Roy, AmirHossein Zamani, Derek Cheung  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2509.05131v1.pdf)  
  <details><summary>Abstract</summary>

  High-quality textures are critical for realistic 3D content creation, yet existing generative methods are slow, rely on UV maps, and often fail to remain faithful to a reference image. To address these challenges, we propose a transformer-based framework that predicts a 3D texture field directly from a single image and a mesh, eliminating the need for UV mapping and differentiable rendering, and enabling faster texture generation. Our method integrates a triplane representation with depth-based backprojection losses, enabling efficient training and faster inference. Once trained, it generates high-fidelity textures in a single forward pass, requiring only 0.2s per shape. Extensive qualitative, quantitative, and user preference evaluations demonstrate that our method outperforms state-of-the-art baselines on single-image texture reconstruction in terms of both fidelity to the input image and perceptual quality, highlighting its practicality for scalable, high-quality, and controllable 3D content creation.
  </details>  
  Keywords: uv mapping, image-to-3d  
- **[Few-step Flow for 3D Generation via Marginal-Data Transport Distillation](https://arxiv.org/abs/2509.04406v1)**  
  Authors: Zanwei Zhou, Taoran Yi, Jiemin Fang, Chen Yang, Lingxi Xie, Xinggang Wang, Wei Shen, Qi Tian  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2509.04406v1.pdf)  
  <details><summary>Abstract</summary>

  Flow-based 3D generation models typically require dozens of sampling steps during inference. Though few-step distillation methods, particularly Consistency Models (CMs), have achieved substantial advancements in accelerating 2D diffusion models, they remain under-explored for more complex 3D generation tasks. In this study, we propose a novel framework, MDT-dist, for few-step 3D flow distillation. Our approach is built upon a primary objective: distilling the pretrained model to learn the Marginal-Data Transport. Directly learning this objective needs to integrate the velocity fields, while this integral is intractable to be implemented. Therefore, we propose two optimizable objectives, Velocity Matching (VM) and Velocity Distillation (VD), to equivalently convert the optimization target from the transport level to the velocity and the distribution level respectively. Velocity Matching (VM) learns to stably match the velocity fields between the student and the teacher, but inevitably provides biased gradient estimates. Velocity Distillation (VD) further enhances the optimization process by leveraging...
  </details>  
- **[Improved 3D Scene Stylization via Text-Guided Generative Image Editing with Region-Based Control](https://arxiv.org/abs/2509.05285v1)**  
  Authors: Haruo Fujiwara, Yusuke Mukuta, Tatsuya Harada  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2509.05285v1.pdf)  
  <details><summary>Abstract</summary>

  Recent advances in text-driven 3D scene editing and stylization, which leverage the powerful capabilities of 2D generative models, have demonstrated promising outcomes. However, challenges remain in ensuring high-quality stylization and view consistency simultaneously. Moreover, applying style consistently to different regions or objects in the scene with semantic correspondence is a challenging task. To address these limitations, we introduce techniques that enhance the quality of 3D stylization while maintaining view consistency and providing optional region-controlled style transfer. Our method achieves stylization by re-training an initial 3D representation using stylized multi-view 2D images of the source views. Therefore, ensuring both style consistency and view consistency of stylized multi-view images is crucial. We achieve this by extending the style-aligned depth-conditioned view generation framework, replacing the fully shared attention mechanism with a single reference-based attention-sharing mechanism, which effectively aligns style across different viewpoints. Additionally, inspired by recent 3D inpainting methods, we utilize a grid...
  </details>  
- **[TauGenNet: Plasma-Driven Tau PET Image Synthesis via Text-Guided 3D Diffusion Models](https://arxiv.org/abs/2509.04269v1)**  
  Authors: Yuxin Gong, Se-in Jang, Wei Shao, Yi Su, Kuang Gong  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2509.04269v1.pdf)  
  <details><summary>Abstract</summary>

  Accurate quantification of tau pathology via tau positron emission tomography (PET) scan is crucial for diagnosing and monitoring Alzheimer's disease (AD). However, the high cost and limited availability of tau PET restrict its widespread use. In contrast, structural magnetic resonance imaging (MRI) and plasma-based biomarkers provide non-invasive and widely available complementary information related to brain anatomy and disease progression. In this work, we propose a text-guided 3D diffusion model for 3D tau PET image synthesis, leveraging multimodal conditions from both structural MRI and plasma measurement. Specifically, the textual prompt is from the plasma p-tau217 measurement, which is a key indicator of AD progression, while MRI provides anatomical structure constraints. The proposed framework is trained and evaluated using clinical AV1451 tau PET data from the Alzheimer's Disease Neuroimaging Initiative (ADNI) database. Experimental results demonstrate that our approach can generate realistic, clinically meaningful 3D tau PET across a range of disease stages....
  </details>  
- **[LMVC: An End-to-End Learned Multiview Video Coding Framework](https://arxiv.org/abs/2509.03922v1)**  
  Authors: Xihua Sheng, Yingwen Zhang, Long Xu, Shiqi Wang  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2509.03922v1.pdf)  
  <details><summary>Abstract</summary>

  Multiview video is a key data source for volumetric video, enabling immersive 3D scene reconstruction but posing significant challenges in storage and transmission due to its massive data volume. Recently, deep learning-based end-to-end video coding has achieved great success, yet most focus on single-view or stereo videos, leaving general multiview scenarios underexplored. This paper proposes an end-to-end learned multiview video coding (LMVC) framework that ensures random access and backward compatibility while enhancing compression efficiency. Our key innovation lies in effectively leveraging independent-view motion and content information to enhance dependent-view compression. Specifically, to exploit the inter-view motion correlation, we propose a feature-based inter-view motion vector prediction method that conditions dependent-view motion encoding on decoded independent-view motion features, along with an inter-view motion entropy model that learns inter-view motion priors. To exploit the inter-view content correlation, we propose a disparity-free inter-view context prediction module that predicts inter-view contexts from decoded independent-view content...
  </details>  
  Keywords: 3d scene reconstruction  
- **[Unifi3D: A Study on 3D Representations for Generation and Reconstruction in a Common Framework](https://arxiv.org/abs/2509.02474v1)**  
  Authors: Nina Wiedemann, Sainan Liu, Quentin Leboutet, Katelyn Gao, Benjamin Ummenhofer, Michael Paulitsch, Kai Yuan  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2509.02474v1.pdf) | [![GitHub](https://img.shields.io/github/stars/isl-org/unifi3d?style=social)](https://github.com/isl-org/unifi3d)  
  <details><summary>Abstract</summary>

  Following rapid advancements in text and image generation, research has increasingly shifted towards 3D generation. Unlike the well-established pixel-based representation in images, 3D representations remain diverse and fragmented, encompassing a wide variety of approaches such as voxel grids, neural radiance fields, signed distance functions, point clouds, or octrees, each offering distinct advantages and limitations. In this work, we present a unified evaluation framework designed to assess the performance of 3D representations in reconstruction and generation. We compare these representations based on multiple criteria: quality, computational efficiency, and generalization performance. Beyond standard model benchmarking, our experiments aim to derive best practices over all steps involved in the 3D generation pipeline, including preprocessing, mesh reconstruction, compression with autoencoders, and generation. Our findings highlight that reconstruction errors significantly impact overall performance, underscoring the need to evaluate generation and reconstruction jointly. We provide insights that can inform the selection of suitable 3D models for...
  </details>  
- **[TeRA: Rethinking Text-guided Realistic 3D Avatar Generation](https://arxiv.org/abs/2509.02466v1)**  
  Authors: Yanwen Wang, Yiyu Zhuang, Jiawei Zhang, Li Wang, Yifei Zeng, Xun Cao, Xinxin Zuo, Hao Zhu  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2509.02466v1.pdf)  
  <details><summary>Abstract</summary>

  In this paper, we rethink text-to-avatar generative models by proposing TeRA, a more efficient and effective framework than the previous SDS-based models and general large 3D generative models. Our approach employs a two-stage training strategy for learning a native 3D avatar generative model. Initially, we distill a decoder to derive a structured latent space from a large human reconstruction model. Subsequently, a text-controlled latent diffusion model is trained to generate photorealistic 3D human avatars within this latent space. TeRA enhances the model performance by eliminating slow iterative optimization and enables text-based partial customization through a structured 3D human representation. Experiments have proven our approach's superiority over previous text-to-avatar generative models in subjective and objective evaluation.
  </details>  
- **[Category-Aware 3D Object Composition with Disentangled Texture and Shape Multi-view Diffusion](https://arxiv.org/abs/2509.02357v1)**  
  Authors: Zeren Xiong, Zikun Chen, Zedong Zhang, Xiang Li, Ying Tai, Jian Yang, Jun Li  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2509.02357v1.pdf) | [![Project](https://img.shields.io/badge/-Project-blue)](https://xzr52.github.io/C33D)  
  <details><summary>Abstract</summary>

  In this paper, we tackle a new task of 3D object synthesis, where a 3D model is composited with another object category to create a novel 3D model. However, most existing text/image/3D-to-3D methods struggle to effectively integrate multiple content sources, often resulting in inconsistent textures and inaccurate shapes. To overcome these challenges, we propose a straightforward yet powerful approach, category+3D-to-3D (C33D), for generating novel and structurally coherent 3D models. Our method begins by rendering multi-view images and normal maps from the input 3D model, then generating a novel 2D object using adaptive text-image harmony (ATIH) with the front-view image and a text description from another object category as inputs. To ensure texture consistency, we introduce texture multi-view diffusion, which refines the textures of the remaining multi-view RGB images based on the novel 2D object. For enhanced shape accuracy, we propose shape multi-view diffusion to improve the 2D shapes of both...
  </details>  
- **[Differentiable Expectation-Maximisation and Applications to Gaussian Mixture Model Optimal Transport](https://arxiv.org/abs/2509.02109v2)**  
  Authors: Samuel Boïté, Eloi Tanguy, Julie Delon, Agnès Desolneux, Rémi Flamary  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2509.02109v2.pdf)  
  <details><summary>Abstract</summary>

  The Expectation-Maximisation (EM) algorithm is a central tool in statistics and machine learning, widely used for latent-variable models such as Gaussian Mixture Models (GMMs). Despite its ubiquity, EM is typically treated as a non-differentiable black box, preventing its integration into modern learning pipelines where end-to-end gradient propagation is essential. In this work, we present and compare several differentiation strategies for EM, from full automatic differentiation to approximate methods, assessing their accuracy and computational efficiency. As a key application, we leverage this differentiable EM in the computation of the Mixture Wasserstein distance $\mathrm{MW}_2$ between GMMs, allowing $\mathrm{MW}_2$ to be used as a differentiable loss in imaging and machine learning tasks. To complement our practical use of $\mathrm{MW}_2$, we contribute a novel stability result which provides theoretical justification for the use of $\mathrm{MW}_2$ with EM, and also introduce a novel unbalanced variant of $\mathrm{MW}_2$. Numerical experiments on barycentre computation, colour and style...
  </details>  
  Keywords: texture synthesis  
- **[RibPull: Implicit Occupancy Fields and Medial Axis Extraction for CT Ribcage Scans](https://arxiv.org/abs/2509.01402v1)**  
  Authors: Emmanouil Nikolakakis, Amine Ouasfi, Julie Digne, Razvan Marinescu  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2509.01402v1.pdf)  
  <details><summary>Abstract</summary>

  We present RibPull, a methodology that utilizes implicit occupancy fields to bridge computational geometry and medical imaging. Implicit 3D representations use continuous functions that handle sparse and noisy data more effectively than discrete methods. While voxel grids are standard for medical imaging, they suffer from resolution limitations, topological information loss, and inefficient handling of sparsity. Coordinate functions preserve complex geometrical information and represent a better solution for sparse data representation, while allowing for further morphological operations. Implicit scene representations enable neural networks to encode entire 3D scenes within their weights. The result is a continuous function that can implicitly compesate for sparse signals and infer further information about the 3D scene by passing any combination of 3D coordinates as input to the model. In this work, we use neural occupancy fields that predict whether a 3D point lies inside or outside an object to represent CT-scanned ribcages. We also apply...
  </details>  

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
- **[RobotArena $\infty$: Scalable Robot Benchmarking via Real-to-Sim Translation](https://arxiv.org/abs/2510.23571v1)**  
  Authors: Yash Jangir, Yidi Zhang, Kashu Yamazaki, Chenyu Zhang, Kuan-Hsun Tu, Tsung-Wei Ke, Lei Ke, Yonatan Bisk, Katerina Fragkiadaki  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2510.23571v1.pdf)  
  <details><summary>Abstract</summary>

  The pursuit of robot generalists - instructable agents capable of performing diverse tasks across diverse environments - demands rigorous and scalable evaluation. Yet real-world testing of robot policies remains fundamentally constrained: it is labor-intensive, slow, unsafe at scale, and difficult to reproduce. Existing simulation benchmarks are similarly limited, as they train and test policies within the same synthetic domains and cannot assess models trained from real-world demonstrations or alternative simulation environments. As policies expand in scope and complexity, these barriers only intensify, since defining "success" in robotics often hinges on nuanced human judgments of execution quality. In this paper, we introduce a new benchmarking framework that overcomes these challenges by shifting VLA evaluation into large-scale simulated environments augmented with online human feedback. Leveraging advances in vision-language models, 2D-to-3D generative modeling, and differentiable rendering, our approach automatically converts video demonstrations from widely used robot datasets into simulated counterparts. Within these digital...
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
- **[LAMP: Data-Efficient Linear Affine Weight-Space Models for Parameter-Controlled 3D Shape Generation and Extrapolation](https://arxiv.org/abs/2510.22491v2)**  
  Authors: Ghadi Nehme, Yanxia Zhang, Dule Shu, Matt Klenk, Faez Ahmed  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2510.22491v2.pdf)  
  <details><summary>Abstract</summary>

  Generating high-fidelity 3D geometries that satisfy specific parameter constraints has broad applications in design and engineering. However, current methods typically rely on large training datasets and struggle with controllability and generalization beyond the training distributions. To overcome these limitations, we introduce LAMP (Linear Affine Mixing of Parametric shapes), a data-efficient framework for controllable and interpretable 3D generation. LAMP first aligns signed distance function (SDF) decoders by overfitting each exemplar from a shared initialization, then synthesizes new geometries by solving a parameter-constrained mixing problem in the aligned weight space. To ensure robustness, we further propose a safety metric that detects geometry validity via linearity mismatch. We evaluate LAMP on two 3D parametric benchmarks: DrivAerNet++ and BlendedNet. We found that LAMP enables (i) controlled interpolation within bounds with as few as 100 samples, (ii) safe extrapolation by up to 100% parameter difference beyond training ranges, (iii) physics performance-guided optimization under fixed parameters....
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
- **[Advances in 4D Representation: Geometry, Motion, and Interaction](https://arxiv.org/abs/2510.19255v1)**  
  Authors: Mingrui Zhao, Sauradip Nag, Kai Wang, Aditya Vora, Guangda Ji, Peter Chun, Ali Mahdavi-Amiri, Hao Zhang  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2510.19255v1.pdf) | [![Project](https://img.shields.io/badge/-Project-blue)](https://mingrui-zhao.github.io/4DRep-GMI)  
  <details><summary>Abstract</summary>

  We present a survey on 4D generation and reconstruction, a fast-evolving subfield of computer graphics whose developments have been propelled by recent advances in neural fields, geometric and motion deep learning, as well 3D generative artificial intelligence (GenAI). While our survey is not the first of its kind, we build our coverage of the domain from a unique and distinctive perspective of 4D representations\/}, to model 3D geometry evolving over time while exhibiting motion and interaction. Specifically, instead of offering an exhaustive enumeration of many works, we take a more selective approach by focusing on representative works to highlight both the desirable properties and ensuing challenges of each representation under different computation, application, and data scenarios. The main take-away message we aim to convey to the readers is on how to select and then customize the appropriate 4D representations for their tasks. Organizationally, we separate the 4D representations based on...
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
- **[Scale-DiT: Ultra-High-Resolution Image Generation with Hierarchical Local Attention](https://arxiv.org/abs/2510.16325v1)**  
  Authors: Yuyao Zhang, Yu-Wing Tai  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2510.16325v1.pdf)  
  <details><summary>Abstract</summary>

  Ultra-high-resolution text-to-image generation demands both fine-grained texture synthesis and globally coherent structure, yet current diffusion models remain constrained to sub-$1K \times 1K$ resolutions due to the prohibitive quadratic complexity of attention and the scarcity of native $4K$ training data. We present \textbf{Scale-DiT}, a new diffusion framework that introduces hierarchical local attention with low-resolution global guidance, enabling efficient, scalable, and semantically coherent image synthesis at ultra-high resolutions. Specifically, high-resolution latents are divided into fixed-size local windows to reduce attention complexity from quadratic to near-linear, while a low-resolution latent equipped with scaled positional anchors injects global semantics. A lightweight LoRA adaptation bridges global and local pathways during denoising, ensuring consistency across structure and detail. To maximize inference efficiency, we repermute token sequence in Hilbert curve order and implement a fused-kernel for skipping masked operations, resulting in a GPU-friendly design. Extensive experiments demonstrate that Scale-DiT achieves more than $2\times$ faster inference and lower...
  </details>  
  Keywords: texture synthesis  
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
- **[H2OFlow: Grounding Human-Object Affordances with 3D Generative Models and Dense Diffused Flows](https://arxiv.org/abs/2510.21769v1)**  
  Authors: Harry Zhang, Luca Carlone  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2510.21769v1.pdf)  
  <details><summary>Abstract</summary>

  Understanding how humans interact with the surrounding environment, and specifically reasoning about object interactions and affordances, is a critical challenge in computer vision, robotics, and AI. Current approaches often depend on labor-intensive, hand-labeled datasets capturing real-world or simulated human-object interaction (HOI) tasks, which are costly and time-consuming to produce. Furthermore, most existing methods for 3D affordance understanding are limited to contact-based analysis, neglecting other essential aspects of human-object interactions, such as orientation (\eg, humans might have a preferential orientation with respect certain objects, such as a TV) and spatial occupancy (\eg, humans are more likely to occupy certain regions around an object, like the front of a microwave rather than its back). To address these limitations, we introduce \emph{H2OFlow}, a novel framework that comprehensively learns 3D HOI affordances -- encompassing contact, orientation, and spatial occupancy -- using only synthetic data generated from 3D generative models. H2OFlow employs a dense 3D-flow-based...
  </details>  
- **[Comprehensive language-image pre-training for 3D medical image understanding](https://arxiv.org/abs/2510.15042v2)**  
  Authors: Tassilo Wald, Ibrahim Ethem Hamamci, Yuan Gao, Sam Bond-Taylor, Harshita Sharma, Maximilian Ilse, Cynthia Lo, Olesya Melnichenko, Anton Schwaighofer, Noel C. F. Codella, Maria Teodora Wetscherek, Klaus H. Maier-Hein, Panagiotis Korfiatis, Valentina Salvatelli, Javier Alvarez-Valle, Fernando Pérez-García  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2510.15042v2.pdf) | [![Project](https://img.shields.io/badge/-Project-blue)](https://huggingface.co/microsoft/colipri.) | [![HuggingFace](https://img.shields.io/badge/-HuggingFace-yellow)](https://huggingface.co/microsoft/colipri)  
  <details><summary>Abstract</summary>

  Vision-language pre-training, i.e., aligning images with paired text, is a powerful paradigm to create encoders that can be directly used for tasks such as classification, retrieval, and segmentation. In the 3D medical image domain, these capabilities allow vision-language encoders (VLEs) to support radiologists by retrieving patients with similar abnormalities, predicting likelihoods of abnormality, or, with downstream adaptation, generating radiological reports. While the methodology holds promise, data availability and domain-specific hurdles limit the capabilities of current 3D VLEs. In this paper, we overcome these challenges by injecting additional supervision via a report generation objective and combining vision-language with vision-only pre-training. This allows us to leverage both image-only and paired image-text 3D datasets, increasing the total amount of data to which our model is exposed. Through these additional objectives, paired with best practices of the 3D medical imaging domain, we develop the Comprehensive Language-Image Pre-training (COLIPRI) encoder family. Our COLIPRI encoders achieve...
  </details>  
- **[VIST3A: Text-to-3D by Stitching a Multi-view Reconstruction Network to a Video Generator](https://arxiv.org/abs/2510.13454v1)**  
  Authors: Hyojun Go, Dominik Narnhofer, Goutam Bhat, Prune Truong, Federico Tombari, Konrad Schindler  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2510.13454v1.pdf)  
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
- **[G4Splat: Geometry-Guided Gaussian Splatting with Generative Prior](https://arxiv.org/abs/2510.12099v1)**  
  Authors: Junfeng Ni, Yixin Chen, Zhifei Yang, Yu Liu, Ruijie Lu, Song-Chun Zhu, Siyuan Huang  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2510.12099v1.pdf) | [![Project](https://img.shields.io/badge/-Project-blue)](https://dali-jack.github.io/g4splat-web)  
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
- **[LOBE-GS: Load-Balanced and Efficient 3D Gaussian Splatting for Large-Scale Scene Reconstruction](https://arxiv.org/abs/2510.01767v1)**  
  Authors: Sheng-Hsiang Hung, Ting-Yu Yen, Wei-Fang Sun, Simon See, Shih-Hsuan Hung, Hung-Kuo Chu  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2510.01767v1.pdf)  
  <details><summary>Abstract</summary>

  3D Gaussian Splatting (3DGS) has established itself as an efficient representation for real-time, high-fidelity 3D scene reconstruction. However, scaling 3DGS to large and unbounded scenes such as city blocks remains difficult. Existing divide-and-conquer methods alleviate memory pressure by partitioning the scene into blocks, but introduce new bottlenecks: (i) partitions suffer from severe load imbalance since uniform or heuristic splits do not reflect actual computational demands, and (ii) coarse-to-fine pipelines fail to exploit the coarse stage efficiently, often reloading the entire model and incurring high overhead. In this work, we introduce LoBE-GS, a novel Load-Balanced and Efficient 3D Gaussian Splatting framework, that re-engineers the large-scale 3DGS pipeline. LoBE-GS introduces a depth-aware partitioning method that reduces preprocessing from hours to minutes, an optimization-based strategy that balances visible Gaussians -- a strong proxy for computational load -- across blocks, and two lightweight techniques, visibility cropping and selective densification, to further reduce training cost....
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
- **[PAT3D: Physics-Augmented Text-to-3D Scene Generation](https://arxiv.org/abs/2511.21978v1)**  
  Authors: Guying Lin, Kemeng Huang, Michael Liu, Ruihan Gao, Hanke Chen, Lyuhao Chen, Beijia Lu, Taku Komura, Yuan Liu, Jun-Yan Zhu, Minchen Li  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2511.21978v1.pdf)  
  <details><summary>Abstract</summary>

  We introduce PAT3D, the first physics-augmented text-to-3D scene generation framework that integrates vision-language models with physics-based simulation to produce physically plausible, simulation-ready, and intersection-free 3D scenes. Given a text prompt, PAT3D generates 3D objects, infers their spatial relations, and organizes them into a hierarchical scene tree, which is then converted into initial conditions for simulation. A differentiable rigid-body simulator ensures realistic object interactions under gravity, driving the scene toward static equilibrium without interpenetrations. To further enhance scene quality, we introduce a simulation-in-the-loop optimization procedure that guarantees physical stability and non-intersection, while improving semantic consistency with the input prompt. Experiments demonstrate that PAT3D substantially outperforms prior approaches in physical plausibility, semantic consistency, and visual quality. Beyond high-quality generation, PAT3D uniquely enables simulation-ready 3D scenes for downstream tasks such as scene editing and robotic manipulation. Code and data will be released upon acceptance.
  </details>  
  Keywords: text-to-3d  
- **[AmodalGen3D: Generative Amodal 3D Object Reconstruction from Sparse Unposed Views](https://arxiv.org/abs/2511.21945v1)**  
  Authors: Junwei Zhou, Yu-Wing Tai  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2511.21945v1.pdf)  
  <details><summary>Abstract</summary>

  Reconstructing 3D objects from a few unposed and partially occluded views is a common yet challenging problem in real-world scenarios, where many object surfaces are never directly observed. Traditional multi-view or inpainting-based approaches struggle under such conditions, often yielding incomplete or geometrically inconsistent reconstructions. We introduce AmodalGen3D, a generative framework for amodal 3D object reconstruction that infers complete, occlusion-free geometry and appearance from arbitrary sparse inputs. The model integrates 2D amodal completion priors with multi-view stereo geometry conditioning, supported by a View-Wise Cross Attention mechanism for sparse-view feature fusion and a Stereo-Conditioned Cross Attention module for unobserved structure inference. By jointly modeling visible and hidden regions, AmodalGen3D faithfully reconstructs 3D objects that are consistent with sparse-view constraints while plausibly hallucinating unseen parts. Experiments on both synthetic and real-world datasets demonstrate that AmodalGen3D achieves superior fidelity and completeness under occlusion-heavy sparse-view settings, addressing a pressing need for object-level 3D scene reconstruction...
  </details>  
  Keywords: 3d scene reconstruction  
- **[HTTM: Head-wise Temporal Token Merging for Faster VGGT](https://arxiv.org/abs/2511.21317v1)**  
  Authors: Weitian Wang, Lukas Meiner, Rai Shubham, Cecilia De La Parra, Akash Kumar  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2511.21317v1.pdf)  
  <details><summary>Abstract</summary>

  The Visual Geometry Grounded Transformer (VGGT) marks a significant leap forward in 3D scene reconstruction, as it is the first model that directly infers all key 3D attributes (camera poses, depths, and dense geometry) jointly in one pass. However, this joint inference mechanism requires global attention layers that perform all-to-all attention computation on tokens from all views. For reconstruction of large scenes with long-sequence inputs, this causes a significant latency bottleneck. In this paper, we propose head-wise temporal merging (HTTM), a training-free 3D token merging method for accelerating VGGT. Existing merging techniques merge tokens uniformly across different attention heads, resulting in identical tokens in the layers' output, which hinders the model's representational ability. HTTM tackles this problem by merging tokens in multi-head granularity, which preserves the uniqueness of feature tokens after head concatenation. Additionally, this enables HTTM to leverage the spatial locality and temporal correspondence observed at the head level...
  </details>  
  Keywords: 3d scene reconstruction  
- **[MODEST: Multi-Optics Depth-of-Field Stereo Dataset](https://arxiv.org/abs/2511.20853v2)**  
  Authors: Nisarg K. Trivedi, Vinayak A. Belludi, Li-Yun Wang, Pardis Taghavi, Dante Lok  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2511.20853v2.pdf)  
  <details><summary>Abstract</summary>

  Reliable depth estimation under real optical conditions remains a core challenge for camera vision in systems such as autonomous robotics and augmented reality. Despite recent progress in depth estimation and depth-of-field rendering, research remains constrained by the lack of large-scale, high-fidelity, real stereo DSLR datasets, limiting real-world generalization and evaluation of models trained on synthetic data as shown extensively in literature. We present the first high-resolution (5472$\times$3648px) stereo DSLR dataset with 18000 images, systematically varying focal length and aperture across complex real scenes and capturing the optical realism and complexity of professional camera systems. For 9 scenes with varying scene complexity, lighting and background, images are captured with two identical camera assemblies at 10 focal lengths (28-70mm) and 5 apertures (f/2.8-f/22), spanning 50 optical configurations in 2000 images per scene. This full-range optics coverage enables controlled analysis of geometric and optical effects for monocular and stereo depth estimation, shallow depth-of-field...
  </details>  
  Keywords: novel view synthesis, 3d scene reconstruction  
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
- **[Yo'City: Personalized and Boundless 3D Realistic City Scene Generation via Self-Critic Expansion](https://arxiv.org/abs/2511.18734v2)**  
  Authors: Keyang Lu, Sifan Zhou, Hongbin Xu, Gang Xu, Zhifei Yang, Yikai Wang, Zhen Xiao, Jieyi Long, Ming Li  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2511.18734v2.pdf)  
  <details><summary>Abstract</summary>

  Realistic 3D city generation is fundamental to a wide range of applications, including virtual reality and digital twins. However, most existing methods rely on training a single diffusion model, which limits their ability to generate personalized and boundless city-scale scenes. In this paper, we present Yo'City, a novel agentic framework that enables user-customized and infinitely expandable 3D city generation by leveraging the reasoning and compositional capabilities of off-the-shelf large models. Specifically, Yo'City first conceptualize the city through a top-down planning strategy that defines a hierarchical "City-District-Grid" structure. The Global Planner determines the overall layout and potential functional districts, while the Local Designer further refines each district with detailed grid-level descriptions. Subsequently, the grid-level 3D generation is achieved through a "produce-refine-evaluate" isometric image synthesis loop, followed by image-to-3D generation. To simulate continuous city evolution, Yo'City further introduces a user-interactive, relationship-guided expansion mechanism, which performs scene graph-based distance- and semantics-aware layout optimization,...
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
- **[SAM 3D: 3Dfy Anything in Images](https://arxiv.org/abs/2511.16624v1)**  
  Authors: SAM 3D Team, Xingyu Chen, Fu-Jen Chu, Pierre Gleize, Kevin J Liang, Alexander Sax, Hao Tang, Weiyao Wang, Michelle Guo, Thibaut Hardin, Xiang Li, Aohan Lin, Jiawei Liu, Ziqi Ma, Anushka Sagar, Bowen Song, Xiaodong Wang, Jianing Yang, Bowen Zhang, Piotr Dollár, Georgia Gkioxari, Matt Feiszli, Jitendra Malik  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2511.16624v1.pdf)  
  <details><summary>Abstract</summary>

  We present SAM 3D, a generative model for visually grounded 3D object reconstruction, predicting geometry, texture, and layout from a single image. SAM 3D excels in natural images, where occlusion and scene clutter are common and visual recognition cues from context play a larger role. We achieve this with a human- and model-in-the-loop pipeline for annotating object shape, texture, and pose, providing visually grounded 3D reconstruction data at unprecedented scale. We learn from this data in a modern, multi-stage training framework that combines synthetic pretraining with real-world alignment, breaking the 3D "data barrier". We obtain significant gains over recent work, with at least a 5:1 win rate in human preference tests on real-world objects and scenes. We will release our code and model weights, an online demo, and a new challenging benchmark for in-the-wild 3D object reconstruction.
  </details>  
- **[From Prompts to Printable Models: Support-Effective 3D Generation via Offset Direct Preference Optimization](https://arxiv.org/abs/2511.16434v1)**  
  Authors: Chenming Wu, Xiaofan Li, Chengkai Dai  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2511.16434v1.pdf)  
  <details><summary>Abstract</summary>

  The transition from digital 3D models to physical objects via 3D printing often requires support structures to prevent overhanging features from collapsing during the fabrication process. While current slicing technologies offer advanced support strategies, they focus on post-processing optimizations rather than addressing the underlying need for support-efficient design during the model generation phase. This paper introduces SEG (\textit{\underline{S}upport-\underline{E}ffective \underline{G}eneration}), a novel framework that integrates Direct Preference Optimization with an Offset (ODPO) into the 3D generation pipeline to directly optimize models for minimal support material usage. By incorporating support structure simulation into the training process, SEG encourages the generation of geometries that inherently require fewer supports, thus reducing material waste and production time. We demonstrate SEG's effectiveness through extensive experiments on two benchmark datasets, Thingi10k-Val and GPT-3DP-Val, showing that SEG significantly outperforms baseline models such as TRELLIS, DPO, and DRO in terms of support volume reduction and printability. Qualitative results further...
  </details>  
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
- **[3DAlign-DAER: Dynamic Attention Policy and Efficient Retrieval Strategy for Fine-grained 3D-Text Alignment at Scale](https://arxiv.org/abs/2511.13211v1)**  
  Authors: Yijia Fan, Jusheng Zhang, Kaitong Cai, Jing Yang, Jian Wang, Keze Wang  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2511.13211v1.pdf)  
  <details><summary>Abstract</summary>

  Despite recent advancements in 3D-text cross-modal alignment, existing state-of-the-art methods still struggle to align fine-grained textual semantics with detailed geometric structures, and their alignment performance degrades significantly when scaling to large-scale 3D databases. To overcome this limitation, we introduce 3DAlign-DAER, a unified framework designed to align text and 3D geometry via the proposed dynamic attention policy and the efficient retrieval strategy, capturing subtle correspondences for diverse cross-modal retrieval and classification tasks. Specifically, during the training, our proposed dynamic attention policy (DAP) employs the Hierarchical Attention Fusion (HAF) module to represent the alignment as learnable fine-grained token-to-point attentions. To optimize these attentions across different tasks and geometric hierarchies, our DAP further exploits the Monte Carlo tree search to dynamically calibrate HAF attention weights via a hybrid reward signal and further enhances the alignment between textual descriptions and local 3D geometry. During the inference, our 3DAlign-DAER introduces an Efficient Retrieval Strategy (ERS)...
  </details>  
- **[Reconstructing 3D Scenes in Native High Dynamic Range](https://arxiv.org/abs/2511.12895v1)**  
  Authors: Kaixuan Zhang, Minxian Li, Mingwu Ren, Jiankang Deng, Xiatian Zhu  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2511.12895v1.pdf)  
  <details><summary>Abstract</summary>

  High Dynamic Range (HDR) imaging is essential for professional digital media creation, e.g., filmmaking, virtual production, and photorealistic rendering. However, 3D scene reconstruction has primarily focused on Low Dynamic Range (LDR) data, limiting its applicability to professional workflows. Existing approaches that reconstruct HDR scenes from LDR observations rely on multi-exposure fusion or inverse tone-mapping, which increase capture complexity and depend on synthetic supervision. With the recent emergence of cameras that directly capture native HDR data in a single exposure, we present the first method for 3D scene reconstruction that directly models native HDR observations. We propose {\bf Native High dynamic range 3D Gaussian Splatting (NH-3DGS)}, which preserves the full dynamic range throughout the reconstruction pipeline. Our key technical contribution is a novel luminance-chromaticity decomposition of the color representation that enables direct optimization from native HDR camera data. We demonstrate on both synthetic and real multi-view HDR datasets that NH-3DGS significantly...
  </details>  
  Keywords: 3d scene reconstruction  
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
- **[VLAD-Grasp: Zero-shot Grasp Detection via Vision-Language Models](https://arxiv.org/abs/2511.05791v1)**  
  Authors: Manav Kulshrestha, S. Talha Bukhari, Damon Conover, Aniket Bera  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2511.05791v1.pdf)  
  <details><summary>Abstract</summary>

  Robotic grasping is a fundamental capability for autonomous manipulation; however, most existing methods rely on large-scale expert annotations and necessitate retraining to handle new objects. We present VLAD-Grasp, a Vision-Language model Assisted zero-shot approach for Detecting grasps. From a single RGB-D image, our method (1) prompts a large vision-language model to generate a goal image where a straight rod "impales" the object, representing an antipodal grasp, (2) predicts depth and segmentation to lift this generated image into 3D, and (3) aligns generated and observed object point clouds via principal component analysis and correspondence-free optimization to recover an executable grasp pose. Unlike prior work, our approach is training-free and does not rely on curated grasp datasets. Despite this, VLAD-Grasp achieves performance that is competitive with or superior to that of state-of-the-art supervised models on the Cornell and Jacquard datasets. We further demonstrate zero-shot generalization to novel real-world objects on a Franka...
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

### May 2025
- **[ArtiScene: Language-Driven Artistic 3D Scene Generation Through Image Intermediary](https://arxiv.org/abs/2506.00742v1)**  
  Authors: Zeqi Gu, Yin Cui, Zhaoshuo Li, Fangyin Wei, Yunhao Ge, Jinwei Gu, Ming-Yu Liu, Abe Davis, Yifan Ding  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2506.00742v1.pdf) | [![Project](https://img.shields.io/badge/-Project-blue)](https://artiscene-cvpr.github.io)  
  <details><summary>Abstract</summary>

  Designing 3D scenes is traditionally a challenging task that demands both artistic expertise and proficiency with complex software. Recent advances in text-to-3D generation have greatly simplified this process by letting users create scenes based on simple text descriptions. However, as these methods generally require extra training or in-context learning, their performance is often hindered by the limited availability of high-quality 3D data. In contrast, modern text-to-image models learned from web-scale images can generate scenes with diverse, reliable spatial layouts and consistent, visually appealing styles. Our key insight is that instead of learning directly from 3D scenes, we can leverage generated 2D images as an intermediary to guide 3D synthesis. In light of this, we introduce ArtiScene, a training-free automated pipeline for scene design that integrates the flexibility of free-form text-to-image generation with the diversity and reliability of 2D intermediary layouts. First, we generate 2D images from a scene description, then...
  </details>  
  Keywords: text-to-3d  
- **[Pro3D-Editor : A Progressive-Views Perspective for Consistent and Precise 3D Editing](https://arxiv.org/abs/2506.00512v2)**  
  Authors: Yang Zheng, Mengqi Huang, Nan Chen, Zhendong Mao  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2506.00512v2.pdf)  
  <details><summary>Abstract</summary>

  Text-guided 3D editing aims to precisely edit semantically relevant local 3D regions, which has significant potential for various practical applications ranging from 3D games to film production. Existing methods typically follow a view-indiscriminate paradigm: editing 2D views indiscriminately and projecting them back into 3D space. However, they overlook the different cross-view interdependencies, resulting in inconsistent multi-view editing. In this study, we argue that ideal consistent 3D editing can be achieved through a \textit{progressive-views paradigm}, which propagates editing semantics from the editing-salient view to other editing-sparse views. Specifically, we propose \textit{Pro3D-Editor}, a novel framework, which mainly includes Primary-view Sampler, Key-view Render, and Full-view Refiner. Primary-view Sampler dynamically samples and edits the most editing-salient view as the primary view. Key-view Render accurately propagates editing semantics from the primary view to other key views through its Mixture-of-View-Experts Low-Rank Adaption (MoVE-LoRA). Full-view Refiner edits and refines the 3D object based on the edited multi-views....
  </details>  
- **[AdaHuman: Animatable Detailed 3D Human Generation with Compositional Multiview Diffusion](https://arxiv.org/abs/2505.24877v1)**  
  Authors: Yangyi Huang, Ye Yuan, Xueting Li, Jan Kautz, Umar Iqbal  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2505.24877v1.pdf)  
  <details><summary>Abstract</summary>

  Existing methods for image-to-3D avatar generation struggle to produce highly detailed, animation-ready avatars suitable for real-world applications. We introduce AdaHuman, a novel framework that generates high-fidelity animatable 3D avatars from a single in-the-wild image. AdaHuman incorporates two key innovations: (1) A pose-conditioned 3D joint diffusion model that synthesizes consistent multi-view images in arbitrary poses alongside corresponding 3D Gaussian Splats (3DGS) reconstruction at each diffusion step; (2) A compositional 3DGS refinement module that enhances the details of local body parts through image-to-image refinement and seamlessly integrates them using a novel crop-aware camera ray map, producing a cohesive detailed 3D avatar. These components allow AdaHuman to generate highly realistic standardized A-pose avatars with minimal self-occlusion, enabling rigging and animation with any input motion. Extensive evaluation on public benchmarks and in-the-wild images demonstrates that AdaHuman significantly outperforms state-of-the-art methods in both avatar reconstruction and reposing. Code and models will be publicly available for...
  </details>  
  Keywords: rigging, image-to-3d  
- **[Tackling View-Dependent Semantics in 3D Language Gaussian Splatting](https://arxiv.org/abs/2505.24746v1)**  
  Authors: Jiazhong Cen, Xudong Zhou, Jiemin Fang, Changsong Wen, Lingxi Xie, Xiaopeng Zhang, Wei Shen, Qi Tian  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2505.24746v1.pdf) | [![GitHub](https://img.shields.io/github/stars/SJTU-DeepVisionLab/LaGa?style=social)](https://github.com/SJTU-DeepVisionLab/LaGa)  
  <details><summary>Abstract</summary>

  Recent advancements in 3D Gaussian Splatting (3D-GS) enable high-quality 3D scene reconstruction from RGB images. Many studies extend this paradigm for language-driven open-vocabulary scene understanding. However, most of them simply project 2D semantic features onto 3D Gaussians and overlook a fundamental gap between 2D and 3D understanding: a 3D object may exhibit various semantics from different viewpoints--a phenomenon we term view-dependent semantics. To address this challenge, we propose LaGa (Language Gaussians), which establishes cross-view semantic connections by decomposing the 3D scene into objects. Then, it constructs view-aggregated semantic representations by clustering semantic descriptors and reweighting them based on multi-view semantics. Extensive experiments demonstrate that LaGa effectively captures key information from view-dependent semantics, enabling a more comprehensive understanding of 3D scenes. Notably, under the same settings, LaGa achieves a significant improvement of +18.7% mIoU over the previous SOTA on the LERF-OVS dataset. Our code is available at: https://github.com/SJTU-DeepVisionLab/LaGa.
  </details>  
  Keywords: 3d scene reconstruction  
- **[pyMEAL: A Multi-Encoder Augmentation-Aware Learning for Robust and Generalizable Medical Image Translation](https://arxiv.org/abs/2505.24421v1)**  
  Authors: Abdul-mojeed Olabisi Ilyas, Adeleke Maradesa, Jamal Banzi, Jianpan Huang, Henry K. F. Mak, Kannie W. Y. Chan  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2505.24421v1.pdf)  
  <details><summary>Abstract</summary>

  Medical imaging is critical for diagnostics, but clinical adoption of advanced AI-driven imaging faces challenges due to patient variability, image artifacts, and limited model generalization. While deep learning has transformed image analysis, 3D medical imaging still suffers from data scarcity and inconsistencies due to acquisition protocols, scanner differences, and patient motion. Traditional augmentation uses a single pipeline for all transformations, disregarding the unique traits of each augmentation and struggling with large data volumes. To address these challenges, we propose a Multi-encoder Augmentation-Aware Learning (MEAL) framework that leverages four distinct augmentation variants processed through dedicated encoders. Three fusion strategies such as concatenation (CC), fusion layer (FL), and adaptive controller block (BD) are integrated to build multi-encoder models that combine augmentation-specific features before decoding. MEAL-BD uniquely preserves augmentation-aware representations, enabling robust, protocol-invariant feature learning. As demonstrated in a Computed Tomography (CT)-to-T1-weighted Magnetic Resonance Imaging (MRI) translation study, MEAL-BD consistently achieved the best...
  </details>  
- **[LTM3D: Bridging Token Spaces for Conditional 3D Generation with Auto-Regressive Diffusion Framework](https://arxiv.org/abs/2505.24245v1)**  
  Authors: Xin Kang, Zihan Zheng, Lei Chu, Yue Gao, Jiahao Li, Hao Pan, Xuejin Chen, Yan Lu  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2505.24245v1.pdf)  
  <details><summary>Abstract</summary>

  We present LTM3D, a Latent Token space Modeling framework for conditional 3D shape generation that integrates the strengths of diffusion and auto-regressive (AR) models. While diffusion-based methods effectively model continuous latent spaces and AR models excel at capturing inter-token dependencies, combining these paradigms for 3D shape generation remains a challenge. To address this, LTM3D features a Conditional Distribution Modeling backbone, leveraging a masked autoencoder and a diffusion model to enhance token dependency learning. Additionally, we introduce Prefix Learning, which aligns condition tokens with shape latent tokens during generation, improving flexibility across modalities. We further propose a Latent Token Reconstruction module with Reconstruction-Guided Sampling to reduce uncertainty and enhance structural fidelity in generated shapes. Our approach operates in token space, enabling support for multiple 3D representations, including signed distance fields, point clouds, meshes, and 3D Gaussian Splatting. Extensive experiments on image- and text-conditioned shape generation tasks demonstrate that LTM3D outperforms existing...
  </details>  
- **[Rooms from Motion: Un-posed Indoor 3D Object Detection as Localization and Mapping](https://arxiv.org/abs/2505.23756v1)**  
  Authors: Justin Lazarow, Kai Kang, Afshin Dehghan  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2505.23756v1.pdf)  
  <details><summary>Abstract</summary>

  We revisit scene-level 3D object detection as the output of an object-centric framework capable of both localization and mapping using 3D oriented boxes as the underlying geometric primitive. While existing 3D object detection approaches operate globally and implicitly rely on the a priori existence of metric camera poses, our method, Rooms from Motion (RfM) operates on a collection of un-posed images. By replacing the standard 2D keypoint-based matcher of structure-from-motion with an object-centric matcher based on image-derived 3D boxes, we estimate metric camera poses, object tracks, and finally produce a global, semantic 3D object map. When a priori pose is available, we can significantly improve map quality through optimization of global 3D boxes against individual observations. RfM shows strong localization performance and subsequently produces maps of higher quality than leading point-based and multi-view 3D object detection methods on CA-1M and ScanNet++, despite these global methods relying on overparameterization through point...
  </details>  
- **[Holistic Large-Scale Scene Reconstruction via Mixed Gaussian Splatting](https://arxiv.org/abs/2505.23280v1)**  
  Authors: Chuandong Liu, Huijiao Wang, Lei Yu, Gui-Song Xia  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2505.23280v1.pdf) | [![GitHub](https://img.shields.io/github/stars/azhuantou/MixGS?style=social)](https://github.com/azhuantou/MixGS)  
  <details><summary>Abstract</summary>

  Recent advances in 3D Gaussian Splatting have shown remarkable potential for novel view synthesis. However, most existing large-scale scene reconstruction methods rely on the divide-and-conquer paradigm, which often leads to the loss of global scene information and requires complex parameter tuning due to scene partitioning and local optimization. To address these limitations, we propose MixGS, a novel holistic optimization framework for large-scale 3D scene reconstruction. MixGS models the entire scene holistically by integrating camera pose and Gaussian attributes into a view-aware representation, which is decoded into fine-detailed Gaussians. Furthermore, a novel mixing operation combines decoded and original Gaussians to jointly preserve global coherence and local fidelity. Extensive experiments on large-scale scenes demonstrate that MixGS achieves state-of-the-art rendering quality and competitive speed, while significantly reducing computational requirements, enabling large-scale scene reconstruction training on a single 24GB VRAM GPU. The code will be released at https://github.com/azhuantou/MixGS.
  </details>  
  Keywords: novel view synthesis, 3d scene reconstruction  
- **[UniTEX: Universal High Fidelity Generative Texturing for 3D Shapes](https://arxiv.org/abs/2505.23253v1)**  
  Authors: Yixun Liang, Kunming Luo, Xiao Chen, Rui Chen, Hongyu Yan, Weiyu Li, Jiarui Liu, Ping Tan  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2505.23253v1.pdf) | [![GitHub](https://img.shields.io/github/stars/YixunLiang/UniTEX?style=social)](https://github.com/YixunLiang/UniTEX)  
  <details><summary>Abstract</summary>

  We present UniTEX, a novel two-stage 3D texture generation framework to create high-quality, consistent textures for 3D assets. Existing approaches predominantly rely on UV-based inpainting to refine textures after reprojecting the generated multi-view images onto the 3D shapes, which introduces challenges related to topological ambiguity. To address this, we propose to bypass the limitations of UV mapping by operating directly in a unified 3D functional space. Specifically, we first propose that lifts texture generation into 3D space via Texture Functions (TFs)--a continuous, volumetric representation that maps any 3D point to a texture value based solely on surface proximity, independent of mesh topology. Then, we propose to predict these TFs directly from images and geometry inputs using a transformer-based Large Texturing Model (LTM). To further enhance texture quality and leverage powerful 2D priors, we develop an advanced LoRA-based strategy for efficiently adapting large-scale Diffusion Transformers (DiTs) for high-quality multi-view texture synthesis...
  </details>  
  Keywords: texture synthesis, uv mapping  
- **[Zero-P-to-3: Zero-Shot Partial-View Images to 3D Object](https://arxiv.org/abs/2505.23054v1)**  
  Authors: Yuxuan Lin, Ruihang Chu, Zhenyu Chen, Xiao Tang, Lei Ke, Haoling Li, Yingji Zhong, Zhihao Li, Shiyong Liu, Xiaofei Wu, Jianzhuang Liu, Yujiu Yang  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2505.23054v1.pdf)  
  <details><summary>Abstract</summary>

  Generative 3D reconstruction shows strong potential in incomplete observations. While sparse-view and single-image reconstruction are well-researched, partial observation remains underexplored. In this context, dense views are accessible only from a specific angular range, with other perspectives remaining inaccessible. This task presents two main challenges: (i) limited View Range: observations confined to a narrow angular scope prevent effective traditional interpolation techniques that require evenly distributed perspectives. (ii) inconsistent Generation: views created for invisible regions often lack coherence with both visible regions and each other, compromising reconstruction consistency. To address these challenges, we propose \method, a novel training-free approach that integrates the local dense observations and multi-source priors for reconstruction. Our method introduces a fusion-based strategy to effectively align these priors in DDIM sampling, thereby generating multi-view consistent images to supervise invisible views. We further design an iterative refinement strategy, which uses the geometric structures of the object to enhance reconstruction quality....
  </details>  
- **[CLIPGaussian: Universal and Multimodal Style Transfer Based on Gaussian Splatting](https://arxiv.org/abs/2505.22854v2)**  
  Authors: Kornel Howil, Joanna Waczyńska, Piotr Borycki, Tadeusz Dziarmaga, Marcin Mazur, Przemysław Spurek  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2505.22854v2.pdf)  
  <details><summary>Abstract</summary>

  Gaussian Splatting (GS) has recently emerged as an efficient representation for rendering 3D scenes from 2D images and has been extended to images, videos, and dynamic 4D content. However, applying style transfer to GS-based representations, especially beyond simple color changes, remains challenging. In this work, we introduce CLIPGaussian, the first unified style transfer framework that supports text- and image-guided stylization across multiple modalities: 2D images, videos, 3D objects, and 4D scenes. Our method operates directly on Gaussian primitives and integrates into existing GS pipelines as a plug-in module, without requiring large generative models or retraining from scratch. The CLIPGaussian approach enables joint optimization of color and geometry in 3D and 4D settings, and achieves temporal coherence in videos, while preserving the model size. We demonstrate superior style fidelity and consistency across all tasks, validating CLIPGaussian as a universal and efficient solution for multimodal style transfer.
  </details>  
- **[Right Side Up? Disentangling Orientation Understanding in MLLMs with Fine-grained Multi-axis Perception Tasks](https://arxiv.org/abs/2505.21649v4)**  
  Authors: Keanu Nichols, Nazia Tasnim, Yuting Yan, Nicholas Ikechukwu, Elva Zou, Deepti Ghadiyaram, Bryan A. Plummer  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2505.21649v4.pdf) | [![Project](https://img.shields.io/badge/-Project-blue)](https://huggingface.co/datasets/appledora/DORI-Benchmark) | [![Dataset](https://img.shields.io/badge/-Dataset-orange)](https://huggingface.co/datasets/appledora/DORI-Benchmark)  
  <details><summary>Abstract</summary>

  Object orientation understanding represents a fundamental challenge in visual perception critical for applications like robotic manipulation and augmented reality. Current vision-language benchmarks fail to isolate this capability, often conflating it with positional relationships and general scene understanding. We introduce DORI (Discriminative Orientation Reasoning Intelligence), a comprehensive benchmark establishing object orientation perception as a primary evaluation target. DORI assesses four dimensions of orientation comprehension: frontal alignment, rotational transformations, relative directional relationships, and canonical orientation understanding. Through carefully curated tasks from 11 datasets spanning 67 object categories across synthetic and real-world scenarios, DORI provides insights on how multi-modal systems understand object orientations. Our evaluation of 15 state-of-the-art vision-language models reveals critical limitations: even the best models achieve only 54.2% accuracy on coarse tasks and 33.0% on granular orientation judgments, with performance deteriorating for tasks requiring reference frame shifts or compound rotations. These findings demonstrate the need for dedicated orientation representation mechanisms, as...
  </details>  
  Keywords: 3d scene reconstruction  
- **[MV-CoLight: Efficient Object Compositing with Consistent Lighting and Shadow Generation](https://arxiv.org/abs/2505.21483v1)**  
  Authors: Kerui Ren, Jiayang Bai, Linning Xu, Lihan Jiang, Jiangmiao Pang, Mulin Yu, Bo Dai  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2505.21483v1.pdf)  
  <details><summary>Abstract</summary>

  Object compositing offers significant promise for augmented reality (AR) and embodied intelligence applications. Existing approaches predominantly focus on single-image scenarios or intrinsic decomposition techniques, facing challenges with multi-view consistency, complex scenes, and diverse lighting conditions. Recent inverse rendering advancements, such as 3D Gaussian and diffusion-based methods, have enhanced consistency but are limited by scalability, heavy data requirements, or prolonged reconstruction time per scene. To broaden its applicability, we introduce MV-CoLight, a two-stage framework for illumination-consistent object compositing in both 2D images and 3D scenes. Our novel feed-forward architecture models lighting and shadows directly, avoiding the iterative biases of diffusion-based methods. We employ a Hilbert curve-based mapping to align 2D image inputs with 3D Gaussian scene representations seamlessly. To facilitate training and evaluation, we further introduce a large-scale 3D compositing dataset. Experiments demonstrate state-of-the-art harmonized results across standard benchmarks and our dataset, as well as casually captured real-world scenes demonstrate the...
  </details>  
- **[Plenodium: UnderWater 3D Scene Reconstruction with Plenoptic Medium Representation](https://arxiv.org/abs/2505.21258v1)**  
  Authors: Changguanng Wu, Jiangxin Dong, Chengjian Li, Jinhui Tang  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2505.21258v1.pdf) | [![Project](https://img.shields.io/badge/-Project-blue)](https://plenodium.github.io)  
  <details><summary>Abstract</summary>

  We present Plenodium (plenoptic medium), an effective and efficient 3D representation framework capable of jointly modeling both objects and participating media. In contrast to existing medium representations that rely solely on view-dependent modeling, our novel plenoptic medium representation incorporates both directional and positional information through spherical harmonics encoding, enabling highly accurate underwater scene reconstruction. To address the initialization challenge in degraded underwater environments, we propose the pseudo-depth Gaussian complementation to augment COLMAP-derived point clouds with robust depth priors. In addition, a depth ranking regularized loss is developed to optimize the geometry of the scene and improve the ordinal consistency of the depth maps. Extensive experiments on real-world underwater datasets demonstrate that our method achieves significant improvements in 3D reconstruction. Furthermore, we conduct a simulated dataset with ground truth and the controllable scattering medium to demonstrate the restoration capability of our method in underwater scenarios. Our code and dataset are available...
  </details>  
  Keywords: 3d scene reconstruction  
- **[3D-UIR: 3D Gaussian for Underwater 3D Scene Reconstruction via Physics Based Appearance-Medium Decoupling](https://arxiv.org/abs/2505.21238v2)**  
  Authors: Jieyu Yuan, Yujun Li, Yuanlin Zhang, Chunle Guo, Xiongxin Tang, Ruixing Wang, Chongyi Li  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2505.21238v2.pdf) | [![Project](https://img.shields.io/badge/-Project-blue)](https://bilityniu.github.io/3D-UIR)  
  <details><summary>Abstract</summary>

  Novel view synthesis for underwater scene reconstruction presents unique challenges due to complex light-media interactions. Optical scattering and absorption in water body bring inhomogeneous medium attenuation interference that disrupts conventional volume rendering assumptions of uniform propagation medium. While 3D Gaussian Splatting (3DGS) offers real-time rendering capabilities, it struggles with underwater inhomogeneous environments where scattering media introduce artifacts and inconsistent appearance. In this study, we propose a physics-based framework that disentangles object appearance from water medium effects through tailored Gaussian modeling. Our approach introduces appearance embeddings, which are explicit medium representations for backscatter and attenuation, enhancing scene consistency. In addition, we propose a distance-guided optimization strategy that leverages pseudo-depth maps as supervision with depth regularization and scale penalty terms to improve geometric fidelity. By integrating the proposed appearance and medium modeling components via an underwater imaging model, our approach achieves both high-quality novel view synthesis and physically accurate scene restoration. Experiments...
  </details>  
  Keywords: novel view synthesis, 3d scene reconstruction  
- **[Advancing high-fidelity 3D and Texture Generation with 2.5D latents](https://arxiv.org/abs/2505.21050v2)**  
  Authors: Xin Yang, Jiantao Lin, Yingjie Xu, Haodong Li, Yingcong Chen  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2505.21050v2.pdf)  
  <details><summary>Abstract</summary>

  Despite the availability of large-scale 3D datasets and advancements in 3D generative models, the complexity and uneven quality of 3D geometry and texture data continue to hinder the performance of 3D generation techniques. In most existing approaches, 3D geometry and texture are generated in separate stages using different models and non-unified representations, frequently leading to unsatisfactory coherence between geometry and texture. To address these challenges, we propose a novel framework for joint generation of 3D geometry and texture. Specifically, we focus in generate a versatile 2.5D representations that can be seamlessly transformed between 2D and 3D. Our approach begins by integrating multiview RGB, normal, and coordinate images into a unified representation, termed as 2.5D latents. Next, we adapt pre-trained 2D foundation models for high-fidelity 2.5D generation, utilizing both text and image conditions. Finally, we introduce a lightweight 2.5D-to-3D refiner-decoder framework that efficiently generates detailed 3D representations from 2.5D images. Extensive...
  </details>  
- **[Uni-Instruct: One-step Diffusion Model through Unified Diffusion Divergence Instruction](https://arxiv.org/abs/2505.20755v4)**  
  Authors: Yifei Wang, Weimin Bai, Colin Zhang, Debing Zhang, Weijian Luo, He Sun  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2505.20755v4.pdf)  
  <details><summary>Abstract</summary>

  In this paper, we unify more than 10 existing one-step diffusion distillation approaches, such as Diff-Instruct, DMD, SIM, SiD, $f$-distill, etc, inside a theory-driven framework which we name the \textbf{\emph{Uni-Instruct}}. Uni-Instruct is motivated by our proposed diffusion expansion theory of the $f$-divergence family. Then we introduce key theories that overcome the intractability issue of the original expanded $f$-divergence, resulting in an equivalent yet tractable loss that effectively trains one-step diffusion models by minimizing the expanded $f$-divergence family. The novel unification introduced by Uni-Instruct not only offers new theoretical contributions that help understand existing approaches from a high-level perspective but also leads to state-of-the-art one-step diffusion generation performances. On the CIFAR10 generation benchmark, Uni-Instruct achieves record-breaking Frechet Inception Distance (FID) values of \textbf{\emph{1.46}} for unconditional generation and \textbf{\emph{1.38}} for conditional generation. On the ImageNet-$64\times 64$ generation benchmark, Uni-Instruct achieves a new SoTA one-step generation FID of \textbf{\emph{1.02}}, which outperforms its 79-step...
  </details>  
  Keywords: text-to-3d  
- **[DIPO: Dual-State Images Controlled Articulated Object Generation Powered by Diverse Data](https://arxiv.org/abs/2505.20460v2)**  
  Authors: Ruiqi Wu, Xinjie Wang, Liu Liu, Chunle Guo, Jiaxiong Qiu, Chongyi Li, Lichao Huang, Zhizhong Su, Ming-Ming Cheng  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2505.20460v2.pdf)  
  <details><summary>Abstract</summary>

  We present DIPO, a novel framework for the controllable generation of articulated 3D objects from a pair of images: one depicting the object in a resting state and the other in an articulated state. Compared to the single-image approach, our dual-image input imposes only a modest overhead for data collection, but at the same time provides important motion information, which is a reliable guide for predicting kinematic relationships between parts. Specifically, we propose a dual-image diffusion model that captures relationships between the image pair to generate part layouts and joint parameters. In addition, we introduce a Chain-of-Thought (CoT) based graph reasoner that explicitly infers part connectivity relationships. To further improve robustness and generalization on complex articulated objects, we develop a fully automated dataset expansion pipeline, name LEGO-Art, that enriches the diversity and complexity of PartNet-Mobility dataset. We propose PM-X, a large-scale dataset of complex articulated 3D objects, accompanied by rendered...
  </details>  
  Keywords: articulated object generation  
- **[ART-DECO: Arbitrary Text Guidance for 3D Detailizer Construction](https://arxiv.org/abs/2505.20431v3)**  
  Authors: Qimin Chen, Yuezhi Yang, Wang Yifan, Vladimir G. Kim, Siddhartha Chaudhuri, Hao Zhang, Zhiqin Chen  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2505.20431v3.pdf)  
  <details><summary>Abstract</summary>

  We introduce a 3D detailizer, a neural model which can instantaneously (in <1s) transform a coarse 3D shape proxy into a high-quality asset with detailed geometry and texture as guided by an input text prompt. Our model is trained using the text prompt, which defines the shape class and characterizes the appearance and fine-grained style of the generated details. The coarse 3D proxy, which can be easily varied and adjusted (e.g., via user editing), provides structure control over the final shape. Importantly, our detailizer is not optimized for a single shape; it is the result of distilling a generative model, so that it can be reused, without retraining, to generate any number of shapes, with varied structures, whose local details all share a consistent style and appearance. Our detailizer training utilizes a pretrained multi-view image diffusion model, with text conditioning, to distill the foundational knowledge therein into our detailizer via...
  </details>  
  Keywords: text-to-3d, 3d asset generation  
- **[Harnessing the Power of Training-Free Techniques in Text-to-2D Generation for Text-to-3D Generation via Score Distillation Sampling](https://arxiv.org/abs/2505.19868v1)**  
  Authors: Junhong Lee, Seungwook Kim, Minsu Cho  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2505.19868v1.pdf)  
  <details><summary>Abstract</summary>

  Recent studies show that simple training-free techniques can dramatically improve the quality of text-to-2D generation outputs, e.g. Classifier-Free Guidance (CFG) or FreeU. However, these training-free techniques have been underexplored in the lens of Score Distillation Sampling (SDS), which is a popular and effective technique to leverage the power of pretrained text-to-2D diffusion models for various tasks. In this paper, we aim to shed light on the effect such training-free techniques have on SDS, via a particular application of text-to-3D generation via 2D lifting. We present our findings, which show that varying the scales of CFG presents a trade-off between object size and surface smoothness, while varying the scales of FreeU presents a trade-off between texture details and geometric errors. Based on these findings, we provide insights into how we can effectively harness training-free techniques for SDS, via a strategic scaling of such techniques in a dynamic manner with respect to...
  </details>  
  Keywords: text-to-3d  
- **[CAD-Coder: Text-to-CAD Generation with Chain-of-Thought and Geometric Reward](https://arxiv.org/abs/2505.19713v3)**  
  Authors: Yandong Guan, Xilin Wang, Ximing Xing, Jing Zhang, Dong Xu, Qian Yu  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2505.19713v3.pdf)  
  <details><summary>Abstract</summary>

  In this work, we introduce CAD-Coder, a novel framework that reformulates text-to-CAD as the generation of CadQuery scripts - a Python-based, parametric CAD language. This representation enables direct geometric validation, a richer modeling vocabulary, and seamless integration with existing LLMs. To further enhance code validity and geometric fidelity, we propose a two-stage learning pipeline: (1) supervised fine-tuning on paired text-CadQuery data, and (2) reinforcement learning with Group Reward Policy Optimization (GRPO), guided by a CAD-specific reward comprising both a geometric reward (Chamfer Distance) and a format reward. We also introduce a chain-of-thought (CoT) planning process to improve model reasoning, and construct a large-scale, high-quality dataset of 110K text-CadQuery-3D model triplets and 1.5K CoT samples via an automated pipeline. Extensive experiments demonstrate that CAD-Coder enables LLMs to generate diverse, valid, and complex CAD models directly from natural language, advancing the state of the art of text-to-CAD generation and geometric reasoning.
  </details>  
- **[From Single Images to Motion Policies via Video-Generation Environment Representations](https://arxiv.org/abs/2505.19306v1)**  
  Authors: Weiming Zhi, Ziyong Ma, Tianyi Zhang, Matthew Johnson-Roberson  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2505.19306v1.pdf)  
  <details><summary>Abstract</summary>

  Autonomous robots typically need to construct representations of their surroundings and adapt their motions to the geometry of their environment. Here, we tackle the problem of constructing a policy model for collision-free motion generation, consistent with the environment, from a single input RGB image. Extracting 3D structures from a single image often involves monocular depth estimation. Developments in depth estimation have given rise to large pre-trained models such as DepthAnything. However, using outputs of these models for downstream motion generation is challenging due to frustum-shaped errors that arise. Instead, we propose a framework known as Video-Generation Environment Representation (VGER), which leverages the advances of large-scale video generation models to generate a moving camera video conditioned on the input image. Frames of this video, which form a multiview dataset, are then input into a pre-trained 3D foundation model to produce a dense point cloud. We then introduce a multi-scale noise approach...
  </details>  
- **[TK-Mamba: Marrying KAN With Mamba for Text-Driven 3D Medical Image Segmentation](https://arxiv.org/abs/2505.18525v2)**  
  Authors: Haoyu Yang, Yutong Guan, Meixing Shi, Yuxiang Cai, Jintao Chen, Sun Bing, Wenhui Lei, Mianxin Liu, Xiaoming Shi, Yankai Jiang, Jianwei Yin  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2505.18525v2.pdf) | [![GitHub](https://img.shields.io/github/stars/yhy-whu/TK-Mamba?style=social)](https://github.com/yhy-whu/TK-Mamba)  
  <details><summary>Abstract</summary>

  3D medical image segmentation is important for clinical diagnosis and treatment but faces challenges from high-dimensional data and complex spatial dependencies. Traditional single-modality networks, such as CNNs and Transformers, are often limited by computational inefficiency and constrained contextual modeling in 3D settings. To alleviate these limitations, we propose TK-Mamba, a multimodal framework that fuses the linear-time Mamba with Kolmogorov-Arnold Networks (KAN) to form an efficient hybrid backbone. Our approach is characterized by two primary technical contributions. Firstly, we introduce the novel 3D-Group-Rational KAN (3D-GR-KAN), which marks the first application of KAN in 3D medical imaging, providing a superior and computationally efficient nonlinear feature transformation crucial for complex volumetric structures. Secondly, we devise a dual-branch text-driven strategy using Pubmedclip's embeddings. This strategy significantly enhances segmentation robustness and accuracy by simultaneously capturing inter-organ semantic relationships to mitigate label inconsistencies and aligning image features with anatomical texts. By combining this advanced backbone and...
  </details>  
- **[Canonical Pose Reconstruction from Single Depth Image for 3D Non-rigid Pose Recovery on Limited Datasets](https://arxiv.org/abs/2505.17992v1)**  
  Authors: Fahd Alhamazani, Yu-Kun Lai, Paul L. Rosin  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2505.17992v1.pdf)  
  <details><summary>Abstract</summary>

  3D reconstruction from 2D inputs, especially for non-rigid objects like humans, presents unique challenges due to the significant range of possible deformations. Traditional methods often struggle with non-rigid shapes, which require extensive training data to cover the entire deformation space. This study addresses these limitations by proposing a canonical pose reconstruction model that transforms single-view depth images of deformable shapes into a canonical form. This alignment facilitates shape reconstruction by enabling the application of rigid object reconstruction techniques, and supports recovering the input pose in voxel representation as part of the reconstruction task, utilizing both the original and deformed depth images. Notably, our model achieves effective results with only a small dataset of approximately 300 samples. Experimental results on animal and human datasets demonstrate that our model outperforms other state-of-the-art methods.
  </details>  
- **[Is Single-View Mesh Reconstruction Ready for Robotics?](https://arxiv.org/abs/2505.17966v2)**  
  Authors: Frederik Nolte, Andreas Geiger, Bernhard Schölkopf, Ingmar Posner  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2505.17966v2.pdf)  
  <details><summary>Abstract</summary>

  This paper evaluates single-view mesh reconstruction models for their potential in enabling instant digital twin creation for real-time planning and dynamics prediction using physics simulators for robotic manipulation. Recent single-view 3D reconstruction advances offer a promising avenue toward an automated real-to-sim pipeline: directly mapping a single observation of a scene into a simulation instance by reconstructing scene objects as individual, complete, and physically plausible 3D meshes. However, their suitability for physics simulations and robotics applications under immediacy, physical fidelity, and simulation readiness remains underexplored. We establish robotics-specific benchmarking criteria for 3D reconstruction, including handling typical inputs, collision-free and stable geometry, occlusions robustness, and meeting computational constraints. Our empirical evaluation using realistic robotics datasets shows that despite success on computer vision benchmarks, existing approaches fail to meet robotics-specific requirements. We quantitively examine limitations of single-view reconstruction for practical robotics implementation, in contrast to prior work that focuses on multi-view approaches. Our...
  </details>  
  Keywords: single-view reconstruction  
- **[SeaLion: Semantic Part-Aware Latent Point Diffusion Models for 3D Generation](https://arxiv.org/abs/2505.17721v2)**  
  Authors: Dekai Zhu, Yan Di, Stefan Gavranovic, Slobodan Ilic  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2505.17721v2.pdf)  
  <details><summary>Abstract</summary>

  Denoising diffusion probabilistic models have achieved significant success in point cloud generation, enabling numerous downstream applications, such as generative data augmentation and 3D model editing. However, little attention has been given to generating point clouds with point-wise segmentation labels, as well as to developing evaluation metrics for this task. Therefore, in this paper, we present SeaLion, a novel diffusion model designed to generate high-quality and diverse point clouds with fine-grained segmentation labels. Specifically, we introduce the semantic part-aware latent point diffusion technique, which leverages the intermediate features of the generative models to jointly predict the noise for perturbed latent points and associated part segmentation labels during the denoising process, and subsequently decodes the latent points to point clouds conditioned on part segmentation labels. To effectively evaluate the quality of generated point clouds, we introduce a novel point cloud pairwise distance calculation method named part-aware Chamfer distance (p-CD). This method enables...
  </details>  
- **[Direct3D-S2: Gigascale 3D Generation Made Easy with Spatial Sparse Attention](https://arxiv.org/abs/2505.17412v2)**  
  Authors: Shuang Wu, Youtian Lin, Feihu Zhang, Yifei Zeng, Yikang Yang, Yajie Bao, Jiachen Qian, Siyu Zhu, Xun Cao, Philip Torr, Yao Yao  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2505.17412v2.pdf) | [![Project](https://img.shields.io/badge/-Project-blue)](https://www.neural4d.com/research/direct3d-s2)  
  <details><summary>Abstract</summary>

  Generating high-resolution 3D shapes using volumetric representations such as Signed Distance Functions (SDFs) presents substantial computational and memory challenges. We introduce Direct3D-S2, a scalable 3D generation framework based on sparse volumes that achieves superior output quality with dramatically reduced training costs. Our key innovation is the Spatial Sparse Attention (SSA) mechanism, which greatly enhances the efficiency of Diffusion Transformer (DiT) computations on sparse volumetric data. SSA allows the model to effectively process large token sets within sparse volumes, substantially reducing computational overhead and achieving a 3.9x speedup in the forward pass and a 9.6x speedup in the backward pass. Our framework also includes a variational autoencoder (VAE) that maintains a consistent sparse volumetric format across input, latent, and output stages. Compared to previous methods with heterogeneous representations in 3D VAE, this unified design significantly improves training efficiency and stability. Our model is trained on public available datasets, and experiments demonstrate...
  </details>  
- **[RealEngine: Simulating Autonomous Driving in Realistic Context](https://arxiv.org/abs/2505.16902v2)**  
  Authors: Junzhe Jiang, Nan Song, Jingyu Li, Xiatian Zhu, Li Zhang  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2505.16902v2.pdf)  
  <details><summary>Abstract</summary>

  Driving simulation plays a crucial role in developing reliable driving agents by providing controlled, evaluative environments. To enable meaningful assessments, a high-quality driving simulator must satisfy several key requirements: multi-modal sensing capabilities (e.g., camera and LiDAR) with realistic scene rendering to minimize observational discrepancies; closed-loop evaluation to support free-form trajectory behaviors; highly diverse traffic scenarios for thorough evaluation; multi-agent cooperation to capture interaction dynamics; and high computational efficiency to ensure affordability and scalability. However, existing simulators and benchmarks fail to comprehensively meet these fundamental criteria. To bridge this gap, this paper introduces RealEngine, a novel driving simulation framework that holistically integrates 3D scene reconstruction and novel view synthesis techniques to achieve realistic and flexible closed-loop simulation in the driving context. By leveraging real-world multi-modal sensor data, RealEngine reconstructs background scenes and foreground traffic participants separately, allowing for highly diverse and realistic traffic scenarios through flexible scene composition. This synergistic fusion...
  </details>  
  Keywords: novel view synthesis, 3d scene reconstruction  
- **[SHaDe: Compact and Consistent Dynamic 3D Reconstruction via Tri-Plane Deformation and Latent Diffusion](https://arxiv.org/abs/2505.16535v1)**  
  Authors: Asrar Alruwayqi  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2505.16535v1.pdf)  
  <details><summary>Abstract</summary>

  We present a novel framework for dynamic 3D scene reconstruction that integrates three key components: an explicit tri-plane deformation field, a view-conditioned canonical radiance field with spherical harmonics (SH) attention, and a temporally-aware latent diffusion prior. Our method encodes 4D scenes using three orthogonal 2D feature planes that evolve over time, enabling efficient and compact spatiotemporal representation. These features are explicitly warped into a canonical space via a deformation offset field, eliminating the need for MLP-based motion modeling. In canonical space, we replace traditional MLP decoders with a structured SH-based rendering head that synthesizes view-dependent color via attention over learned frequency bands improving both interpretability and rendering efficiency. To further enhance fidelity and temporal consistency, we introduce a transformer-guided latent diffusion module that refines the tri-plane and deformation features in a compressed latent space. This generative module denoises scene representations under ambiguous or out-of-distribution (OOD) motion, improving generalization. Our model...
  </details>  
  Keywords: 3d scene reconstruction  
- **[AdvReal: Physical Adversarial Patch Generation Framework for Security Evaluation of Object Detection Systems](https://arxiv.org/abs/2505.16402v2)**  
  Authors: Yuanhao Huang, Yilong Ren, Jinlei Wang, Lujia Huo, Xuesong Bai, Jinchuan Zhang, Haiyan Yu  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2505.16402v2.pdf) | [![GitHub](https://img.shields.io/github/stars/Huangyh98/AdvReal.git?style=social)](https://github.com/Huangyh98/AdvReal.git)  
  <details><summary>Abstract</summary>

  Autonomous vehicles are typical complex intelligent systems with artificial intelligence at their core. However, perception methods based on deep learning are extremely vulnerable to adversarial samples, resulting in security accidents. How to generate effective adversarial examples in the physical world and evaluate object detection systems is a huge challenge. In this study, we propose a unified joint adversarial training framework for both 2D and 3D domains, which simultaneously optimizes texture maps in 2D image and 3D mesh spaces to better address intra-class diversity and real-world environmental variations. The framework includes a novel realistic enhanced adversarial module, with time-space and relighting mapping pipeline that adjusts illumination consistency between adversarial patches and target garments under varied viewpoints. Building upon this, we develop a realism enhancement mechanism that incorporates non-rigid deformation modeling and texture remapping to ensure alignment with the human body's non-rigid surfaces in 3D scenes. Extensive experiment results in digital and...
  </details>  
- **[Constructing a 3D Scene from a Single Image](https://arxiv.org/abs/2505.15765v2)**  
  Authors: Kaizhi Zheng, Ruijian Zha, Zishuo Xu, Jing Gu, Jie Yang, Xin Eric Wang  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2505.15765v2.pdf)  
  <details><summary>Abstract</summary>

  Acquiring detailed 3D scenes typically demands costly equipment, multi-view data, or labor-intensive modeling. Therefore, a lightweight alternative, generating complex 3D scenes from a single top-down image, plays an essential role in real-world applications. While recent 3D generative models have achieved remarkable results at the object level, their extension to full-scene generation often leads to inconsistent geometry, layout hallucinations, and low-quality meshes. In this work, we introduce SceneFuse-3D, a training-free framework designed to synthesize coherent 3D scenes from a single top-down view. Our method is grounded in two principles: region-based generation to improve image-to-3D alignment and resolution, and spatial-aware 3D inpainting to ensure global scene coherence and high-quality geometry generation. Specifically, we decompose the input image into overlapping regions and generate each using a pretrained 3D object generator, followed by a masked rectified flow inpainting process that fills in missing geometry while maintaining structural continuity. This modular design allows us to...
  </details>  
  Keywords: image-to-3d  
- **[PlantDreamer: Achieving Realistic 3D Plant Models with Diffusion-Guided Gaussian Splatting](https://arxiv.org/abs/2505.15528v1)**  
  Authors: Zane K J Hartley, Lewis A G Stuart, Andrew P French, Michael P Pound  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2505.15528v1.pdf)  
  <details><summary>Abstract</summary>

  Recent years have seen substantial improvements in the ability to generate synthetic 3D objects using AI. However, generating complex 3D objects, such as plants, remains a considerable challenge. Current generative 3D models struggle with plant generation compared to general objects, limiting their usability in plant analysis tools, which require fine detail and accurate geometry. We introduce PlantDreamer, a novel approach to 3D synthetic plant generation, which can achieve greater levels of realism for complex plant geometry and textures than available text-to-3D models. To achieve this, our new generation pipeline leverages a depth ControlNet, fine-tuned Low-Rank Adaptation and an adaptable Gaussian culling algorithm, which directly improve textural realism and geometric integrity of generated 3D plant models. Additionally, PlantDreamer enables both purely synthetic plant generation, by leveraging L-System-generated meshes, and the enhancement of real-world plant point clouds by converting them into 3D Gaussian Splats. We evaluate our approach by comparing its outputs...
  </details>  
  Keywords: text-to-3d  
- **[Personalize Your Gaussian: Consistent 3D Scene Personalization from a Single Image](https://arxiv.org/abs/2505.14537v3)**  
  Authors: Yuxuan Wang, Xuanyu Yi, Qingshan Xu, Yuan Zhou, Long Chen, Hanwang Zhang  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2505.14537v3.pdf)  
  <details><summary>Abstract</summary>

  Personalizing 3D scenes from a single reference image enables intuitive user-guided editing, which requires achieving both multi-view consistency across perspectives and referential consistency with the input image. However, these goals are particularly challenging due to the viewpoint bias caused by the limited perspective provided in a single image. Lacking the mechanisms to effectively expand reference information beyond the original view, existing methods of image-conditioned 3DGS personalization often suffer from this viewpoint bias and struggle to produce consistent results. Therefore, in this paper, we present Consistent Personalization for 3D Gaussian Splatting (CP-GS), a framework that progressively propagates the single-view reference appearance to novel perspectives. In particular, CP-GS integrates pre-trained image-to-3D generation and iterative LoRA fine-tuning to extract and extend the reference appearance, and finally produces faithful multi-view guidance images and the personalized 3DGS outputs through a view-consistent generation process guided by geometric cues. Extensive experiments on real-world scenes show that our...
  </details>  
  Keywords: image-to-3d  
- **[Sparc3D: Sparse Representation and Construction for High-Resolution 3D Shapes Modeling](https://arxiv.org/abs/2505.14521v3)**  
  Authors: Zhihao Li, Yufei Wang, Heliang Zheng, Yihao Luo, Bihan Wen  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2505.14521v3.pdf)  
  <details><summary>Abstract</summary>

  High-fidelity 3D object synthesis remains significantly more challenging than 2D image generation due to the unstructured nature of mesh data and the cubic complexity of dense volumetric grids. Existing two-stage pipelines-compressing meshes with a VAE (using either 2D or 3D supervision), followed by latent diffusion sampling-often suffer from severe detail loss caused by inefficient representations and modality mismatches introduced in VAE. We introduce Sparc3D, a unified framework that combines a sparse deformable marching cubes representation Sparcubes with a novel encoder Sparconv-VAE. Sparcubes converts raw meshes into high-resolution ($1024^3$) surfaces with arbitrary topology by scattering signed distance and deformation fields onto a sparse cube, allowing differentiable optimization. Sparconv-VAE is the first modality-consistent variational autoencoder built entirely upon sparse convolutional networks, enabling efficient and near-lossless 3D reconstruction suitable for high-resolution generative modeling through latent diffusion. Sparc3D achieves state-of-the-art reconstruction fidelity on challenging inputs, including open surfaces, disconnected components, and intricate geometry. It...
  </details>  
- **[Hybrid 3D-4D Gaussian Splatting for Fast Dynamic Scene Representation](https://arxiv.org/abs/2505.13215v1)**  
  Authors: Seungjun Oh, Younggeun Lee, Hyejin Jeon, Eunbyung Park  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2505.13215v1.pdf)  
  <details><summary>Abstract</summary>

  Recent advancements in dynamic 3D scene reconstruction have shown promising results, enabling high-fidelity 3D novel view synthesis with improved temporal consistency. Among these, 4D Gaussian Splatting (4DGS) has emerged as an appealing approach due to its ability to model high-fidelity spatial and temporal variations. However, existing methods suffer from substantial computational and memory overhead due to the redundant allocation of 4D Gaussians to static regions, which can also degrade image quality. In this work, we introduce hybrid 3D-4D Gaussian Splatting (3D-4DGS), a novel framework that adaptively represents static regions with 3D Gaussians while reserving 4D Gaussians for dynamic elements. Our method begins with a fully 4D Gaussian representation and iteratively converts temporally invariant Gaussians into 3D, significantly reducing the number of parameters and improving computational efficiency. Meanwhile, dynamic Gaussians retain their full 4D representation, capturing complex motions with high fidelity. Our approach achieves significantly faster training times compared to baseline...
  </details>  
  Keywords: novel view synthesis, 3d scene reconstruction  
- **[Touch2Shape: Touch-Conditioned 3D Diffusion for Shape Exploration and Reconstruction](https://arxiv.org/abs/2505.13091v1)**  
  Authors: Yuanbo Wang, Zhaoxuan Zhang, Jiajin Qiu, Dilong Sun, Zhengyu Meng, Xiaopeng Wei, Xin Yang  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2505.13091v1.pdf)  
  <details><summary>Abstract</summary>

  Diffusion models have made breakthroughs in 3D generation tasks. Current 3D diffusion models focus on reconstructing target shape from images or a set of partial observations. While excelling in global context understanding, they struggle to capture the local details of complex shapes and limited to the occlusion and lighting conditions. To overcome these limitations, we utilize tactile images to capture the local 3D information and propose a Touch2Shape model, which leverages a touch-conditioned diffusion model to explore and reconstruct the target shape from touch. For shape reconstruction, we have developed a touch embedding module to condition the diffusion model in creating a compact representation and a touch shape fusion module to refine the reconstructed shape. For shape exploration, we combine the diffusion model with reinforcement learning to train a policy. This involves using the generated latent vector from the diffusion model to guide the touch exploration policy training through a...
  </details>  
- **[Real-Time Spatial Reasoning by Mobile Robots for Reconstruction and Navigation in Dynamic LiDAR Scenes](https://arxiv.org/abs/2505.12267v1)**  
  Authors: Pengdi Huang, Mingyang Wang, Huan Tian, Minglun Gong, Hao Zhang, Hui Huang  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2505.12267v1.pdf)  
  <details><summary>Abstract</summary>

  Our brain has an inner global positioning system which enables us to sense and navigate 3D spaces in real time. Can mobile robots replicate such a biological feat in a dynamic environment? We introduce the first spatial reasoning framework for real-time surface reconstruction and navigation that is designed for outdoor LiDAR scanning data captured by ground mobile robots and capable of handling moving objects such as pedestrians. Our reconstruction-based approach is well aligned with the critical cellular functions performed by the border vector cells (BVCs) over all layers of the medial entorhinal cortex (MEC) for surface sensing and tracking. To address the challenges arising from blurred boundaries resulting from sparse single-frame LiDAR points and outdated data due to object movements, we integrate real-time single-frame mesh reconstruction, via visibility reasoning, with robot navigation assistance through on-the-fly 3D free space determination. This enables continuous and incremental updates of the scene and free...
  </details>  
  Keywords: 3d scene reconstruction  
- **[Generating Digital Models Using Text-to-3D and Image-to-3D Prompts: Critical Case Study](https://arxiv.org/abs/2505.11799v1)**  
  Authors: Rushan Ziatdinov, Rifkat Nabiyev  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2505.11799v1.pdf)  
  <details><summary>Abstract</summary>

  In the world of technology and AI, digital models play an important role in our lives and are an essential part of the digital twins of real-world objects. They can be created by designers, artists, or game developers using spline curves and surfaces, meshes, and voxels, but making such models is too time-consuming. With the growth of AI tools, there is interest in the automated generation of 3D models, such as generative design approaches, which can save creators valuable time. This paper reviews several online 3D model generators and critically analyses the results, hoping to see higher-quality results from different prompts.
  </details>  
  Keywords: text-to-3d, image-to-3d  

### June 2025
- **[SCORP: Scene-Consistent Object Refinement via Proxy Generation and Tuning](https://arxiv.org/abs/2506.23835v2)**  
  Authors: Ziwei Chen, Ziling Liu, Zitong Huang, Mingqi Gao, Feng Zheng  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2506.23835v2.pdf) | [![GitHub](https://img.shields.io/github/stars/PolySummit/SCORP?style=social)](https://github.com/PolySummit/SCORP)  
  <details><summary>Abstract</summary>

  Viewpoint missing of objects is common in scene reconstruction, as camera paths typically prioritize capturing the overall scene structure rather than individual objects. This makes it highly challenging to achieve high-fidelity object-level modeling while maintaining accurate scene-level representation. Addressing this issue is critical for advancing downstream tasks requiring high-fidelity object reconstruction. In this paper, we introduce Scene-Consistent Object Refinement via Proxy Generation and Tuning (SCORP), a novel 3D enhancement framework that leverages 3D generative priors to recover fine-grained object geometry and appearance under missing views. Starting with proxy generation by substituting degraded objects using a 3D generation model, SCORP then progressively refines geometry and texture by aligning each proxy to its degraded counterpart in 7-DoF pose, followed by correcting spatial and appearance inconsistencies through registration-constrained enhancement. This two-stage proxy tuning ensures the high-fidelity geometry and appearance of the original object in unseen views while maintaining consistency in spatial positioning, observed...
  </details>  
  Keywords: novel view synthesis  
- **[PGOV3D: Open-Vocabulary 3D Semantic Segmentation with Partial-to-Global Curriculum](https://arxiv.org/abs/2506.23607v1)**  
  Authors: Shiqi Zhang, Sha Zhang, Jiajun Deng, Yedong Shen, Mingxiao MA, Yanyong Zhang  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2506.23607v1.pdf)  
  <details><summary>Abstract</summary>

  Existing open-vocabulary 3D semantic segmentation methods typically supervise 3D segmentation models by merging text-aligned features (e.g., CLIP) extracted from multi-view images onto 3D points. However, such approaches treat multi-view images merely as intermediaries for transferring open-vocabulary information, overlooking their rich semantic content and cross-view correspondences, which limits model effectiveness. To address this, we propose PGOV3D, a novel framework that introduces a Partial-to-Global curriculum for improving open-vocabulary 3D semantic segmentation. The key innovation lies in a two-stage training strategy. In the first stage, we pre-train the model on partial scenes that provide dense semantic information but relatively simple geometry. These partial point clouds are derived from multi-view RGB-D inputs via pixel-wise depth projection. To enable open-vocabulary learning, we leverage a multi-modal large language model (MLLM) and a 2D segmentation foundation model to generate open-vocabulary labels for each viewpoint, offering rich and aligned supervision. An auxiliary inter-frame consistency module is introduced to...
  </details>  
- **[SurgTPGS: Semantic 3D Surgical Scene Understanding with Text Promptable Gaussian Splatting](https://arxiv.org/abs/2506.23309v2)**  
  Authors: Yiming Huang, Long Bai, Beilei Cui, Kun Yuan, Guankun Wang, Mobarak I. Hoque, Nicolas Padoy, Nassir Navab, Hongliang Ren  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2506.23309v2.pdf) | [![GitHub](https://img.shields.io/github/stars/lastbasket/SurgTPGS?style=social)](https://github.com/lastbasket/SurgTPGS)  
  <details><summary>Abstract</summary>

  In contemporary surgical research and practice, accurately comprehending 3D surgical scenes with text-promptable capabilities is particularly crucial for surgical planning and real-time intra-operative guidance, where precisely identifying and interacting with surgical tools and anatomical structures is paramount. However, existing works focus on surgical vision-language model (VLM), 3D reconstruction, and segmentation separately, lacking support for real-time text-promptable 3D queries. In this paper, we present SurgTPGS, a novel text-promptable Gaussian Splatting method to fill this gap. We introduce a 3D semantics feature learning strategy incorporating the Segment Anything model and state-of-the-art vision-language models. We extract the segmented language features for 3D surgical scene reconstruction, enabling a more in-depth understanding of the complex surgical environment. We also propose semantic-aware deformation tracking to capture the seamless deformation of semantic features, providing a more precise reconstruction for both texture and semantic features. Furthermore, we present semantic region-aware optimization, which utilizes regional-based semantic information to supervise...
  </details>  
- **[AlignCVC: Aligning Cross-View Consistency for Single-Image-to-3D Generation](https://arxiv.org/abs/2506.23150v2)**  
  Authors: Xinyue Liang, Zhiyuan Ma, Lingchen Sun, Yanjun Guo, Lei Zhang  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2506.23150v2.pdf)  
  <details><summary>Abstract</summary>

  Single-image-to-3D models typically follow a sequential generation and reconstruction workflow. However, intermediate multi-view images synthesized by pre-trained generation models often lack cross-view consistency (CVC), significantly degrading 3D reconstruction performance. While recent methods attempt to refine CVC by feeding reconstruction results back into the multi-view generator, these approaches struggle with noisy and unstable reconstruction outputs that limit effective CVC improvement. We introduce AlignCVC, a novel framework that fundamentally re-frames single-image-to-3D generation through distribution alignment rather than relying on strict regression losses. Our key insight is to align both generated and reconstructed multi-view distributions toward the ground-truth multi-view distribution, establishing a principled foundation for improved CVC. Observing that generated images exhibit weak CVC while reconstructed images display strong CVC due to explicit rendering, we propose a soft-hard alignment strategy with distinct objectives for generation and reconstruction models. This approach not only enhances generation quality but also dramatically accelerates inference to as few...
  </details>  
  Keywords: image-to-3d  
- **[ParticleFormer: A 3D Point Cloud World Model for Multi-Object, Multi-Material Robotic Manipulation](https://arxiv.org/abs/2506.23126v4)**  
  Authors: Suning Huang, Qianzhong Chen, Xiaohan Zhang, Jiankai Sun, Mac Schwager  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2506.23126v4.pdf) | [![Project](https://img.shields.io/badge/-Project-blue)](https://suninghuang19.github.io/particleformer_page)  
  <details><summary>Abstract</summary>

  3D world models (i.e., learning-based 3D dynamics models) offer a promising approach to generalizable robotic manipulation by capturing the underlying physics of environment evolution conditioned on robot actions. However, existing 3D world models are primarily limited to single-material dynamics using a particle-based Graph Neural Network model, and often require time-consuming 3D scene reconstruction to obtain 3D particle tracks for training. In this work, we present ParticleFormer, a Transformer-based point cloud world model trained with a hybrid point cloud reconstruction loss, supervising both global and local dynamics features in multi-material, multi-object robot interactions. ParticleFormer captures fine-grained multi-object interactions between rigid, deformable, and flexible materials, trained directly from real-world robot perception data without an elaborate scene reconstruction. We demonstrate the model's effectiveness both in 3D scene forecasting tasks, and in downstream manipulation tasks using a Model Predictive Control (MPC) policy. In addition, we extend existing dynamics learning benchmarks to include diverse multi-material,...
  </details>  
  Keywords: 3d scene reconstruction  
- **[Neural Cellular Automata: From Cells to Pixels](https://arxiv.org/abs/2506.22899v2)**  
  Authors: Ehsan Pajouheshgar, Yitao Xu, Ali Abbasi, Alexander Mordvintsev, Wenzel Jakob, Sabine Süsstrunk  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2506.22899v2.pdf)  
  <details><summary>Abstract</summary>

  Neural Cellular Automata (NCAs) are bio-inspired dynamical systems in which identical cells iteratively apply a learned local update rule to self-organize into complex patterns, exhibiting regeneration, robustness, and spontaneous dynamics. Despite their success in texture synthesis and morphogenesis, NCAs remain largely confined to low-resolution outputs. This limitation stems from (1) training time and memory requirements that grow quadratically with grid size, (2) the strictly local propagation of information that impedes long-range cell communication, and (3) the heavy compute demands of real-time inference at high resolution. In this work, we overcome this limitation by pairing an NCA that evolves on a coarse grid with a lightweight implicit decoder that maps cell states and local coordinates to appearance attributes, enabling the same model to render outputs at arbitrary resolution. Moreover, because both the decoder and NCA updates are local, inference remains highly parallelizable. To supervise high-resolution outputs efficiently, we introduce task-specific losses...
  </details>  
  Keywords: texture synthesis  
- **[PhotonSplat: 3D Scene Reconstruction and Colorization from SPAD Sensors](https://arxiv.org/abs/2506.21680v1)**  
  Authors: Sai Sri Teja, Sreevidya Chintalapati, Vinayak Gupta, Mukund Varma T, Haejoon Lee, Aswin Sankaranarayanan, Kaushik Mitra  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2506.21680v1.pdf)  
  <details><summary>Abstract</summary>

  Advances in 3D reconstruction using neural rendering have enabled high-quality 3D capture. However, they often fail when the input imagery is corrupted by motion blur, due to fast motion of the camera or the objects in the scene. This work advances neural rendering techniques in such scenarios by using single-photon avalanche diode (SPAD) arrays, an emerging sensing technology capable of sensing images at extremely high speeds. However, the use of SPADs presents its own set of unique challenges in the form of binary images, that are driven by stochastic photon arrivals. To address this, we introduce PhotonSplat, a framework designed to reconstruct 3D scenes directly from SPAD binary images, effectively navigating the noise vs. blur trade-off. Our approach incorporates a novel 3D spatial filtering technique to reduce noise in the renderings. The framework also supports both no-reference using generative priors and reference-based colorization from a single blurry image, enabling downstream...
  </details>  
  Keywords: 3d scene reconstruction  
- **[FairyGen: Storied Cartoon Video from a Single Child-Drawn Character](https://arxiv.org/abs/2506.21272v2)**  
  Authors: Jiayi Zheng, Xiaodong Cun  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2506.21272v2.pdf) | [![GitHub](https://img.shields.io/github/stars/GVCLab/FairyGen?style=social)](https://github.com/GVCLab/FairyGen)  
  <details><summary>Abstract</summary>

  We propose FairyGen, an automatic system for generating story-driven cartoon videos from a single child's drawing, while faithfully preserving its unique artistic style. Unlike previous storytelling methods that primarily focus on character consistency and basic motion, FairyGen explicitly disentangles character modeling from stylized background generation and incorporates cinematic shot design to support expressive and coherent storytelling. Given a single character sketch, we first employ an MLLM to generate a structured storyboard with shot-level descriptions that specify environment settings, character actions, and camera perspectives. To ensure visual consistency, we introduce a style propagation adapter that captures the character's visual style and applies it to the background, faithfully retaining the character's full visual identity while synthesizing style-consistent scenes. A shot design module further enhances visual diversity and cinematic quality through frame cropping and multi-view synthesis based on the storyboard. To animate the story, we reconstruct a 3D proxy of the character to...
  </details>  
  Keywords: multi-view synthesis  
- **[Geometry and Perception Guided Gaussians for Multiview-consistent 3D Generation from a Single Image](https://arxiv.org/abs/2506.21152v3)**  
  Authors: Pufan Li, Bi'an Du, Wei Hu  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2506.21152v3.pdf)  
  <details><summary>Abstract</summary>

  Generating realistic 3D objects from single-view images requires natural appearance, 3D consistency, and the ability to capture multiple plausible interpretations of unseen regions. Existing approaches often rely on fine-tuning pretrained 2D diffusion models or directly generating 3D information through fast network inference or 3D Gaussian Splatting, but their results generally suffer from poor multiview consistency and lack geometric detail. To tackle these issues, we present a novel method that seamlessly integrates geometry and perception information without requiring additional model training to reconstruct detailed 3D objects from a single image. Specifically, we incorporate geometry and perception priors to initialize the Gaussian branches and guide their parameter optimization. The geometry prior captures the rough 3D shapes, while the perception prior utilizes the 2D pretrained diffusion model to enhance multiview information. Subsequently, we introduce a stable Score Distillation Sampling for fine-grained prior distillation to ensure effective knowledge transfer. The model is further enhanced...
  </details>  
  Keywords: novel view synthesis  
- **[CL-Splats: Continual Learning of Gaussian Splatting with Local Optimization](https://arxiv.org/abs/2506.21117v2)**  
  Authors: Jan Ackermann, Jonas Kulhanek, Shengqu Cai, Haofei Xu, Marc Pollefeys, Gordon Wetzstein, Leonidas Guibas, Songyou Peng  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2506.21117v2.pdf)  
  <details><summary>Abstract</summary>

  In dynamic 3D environments, accurately updating scene representations over time is crucial for applications in robotics, mixed reality, and embodied AI. As scenes evolve, efficient methods to incorporate changes are needed to maintain up-to-date, high-quality reconstructions without the computational overhead of re-optimizing the entire scene. This paper introduces CL-Splats, which incrementally updates Gaussian splatting-based 3D representations from sparse scene captures. CL-Splats integrates a robust change-detection module that segments updated and static components within the scene, enabling focused, local optimization that avoids unnecessary re-computation. Moreover, CL-Splats supports storing and recovering previous scene states, facilitating temporal segmentation and new scene-analysis applications. Our extensive experiments demonstrate that CL-Splats achieves efficient updates with improved reconstruction quality over the state-of-the-art. This establishes a robust foundation for future real-time adaptation in 3D scene reconstruction tasks.
  </details>  
  Keywords: 3d scene reconstruction  
- **[Consistent Zero-shot 3D Texture Synthesis Using Geometry-aware Diffusion and Temporal Video Models](https://arxiv.org/abs/2506.20946v1)**  
  Authors: Donggoo Kang, Jangyeong Kim, Dasol Jeong, Junyoung Choi, Jeonga Wi, Hyunmin Lee, Joonho Gwon, Joonki Paik  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2506.20946v1.pdf)  
  <details><summary>Abstract</summary>

  Current texture synthesis methods, which generate textures from fixed viewpoints, suffer from inconsistencies due to the lack of global context and geometric understanding. Meanwhile, recent advancements in video generation models have demonstrated remarkable success in achieving temporally consistent videos. In this paper, we introduce VideoTex, a novel framework for seamless texture synthesis that leverages video generation models to address both spatial and temporal inconsistencies in 3D textures. Our approach incorporates geometry-aware conditions, enabling precise utilization of 3D mesh structures. Additionally, we propose a structure-wise UV diffusion strategy, which enhances the generation of occluded areas by preserving semantic information, resulting in smoother and more coherent textures. VideoTex not only achieves smoother transitions across UV boundaries but also ensures high-quality, temporally stable textures across video frames. Extensive experiments demonstrate that VideoTex outperforms existing methods in texture fidelity, seam blending, and stability, paving the way for dynamic real-time applications that demand both visual...
  </details>  
  Keywords: texture synthesis  
- **[WonderFree: Enhancing Novel View Quality and Cross-View Consistency for 3D Scene Exploration](https://arxiv.org/abs/2506.20590v1)**  
  Authors: Chaojun Ni, Jie Li, Haoyun Li, Hengyu Liu, Xiaofeng Wang, Zheng Zhu, Guosheng Zhao, Boyuan Wang, Chenxin Li, Guan Huang, Wenjun Mei  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2506.20590v1.pdf)  
  <details><summary>Abstract</summary>

  Interactive 3D scene generation from a single image has gained significant attention due to its potential to create immersive virtual worlds. However, a key challenge in current 3D generation methods is the limited explorability, which cannot render high-quality images during larger maneuvers beyond the original viewpoint, particularly when attempting to move forward into unseen areas. To address this challenge, we propose WonderFree, the first model that enables users to interactively generate 3D worlds with the freedom to explore from arbitrary angles and directions. Specifically, we decouple this challenge into two key subproblems: novel view quality, which addresses visual artifacts and floating issues in novel views, and cross-view consistency, which ensures spatial consistency across different viewpoints. To enhance rendering quality in novel views, we introduce WorldRestorer, a data-driven video restoration model designed to eliminate floaters and artifacts. In addition, a data collection pipeline is presented to automatically gather training data for...
  </details>  
- **[DreamAnywhere: Object-Centric Panoramic 3D Scene Generation](https://arxiv.org/abs/2506.20367v1)**  
  Authors: Edoardo Alberto Dominici, Jozef Hladky, Floor Verhoeven, Lukas Radl, Thomas Deixelberger, Stefan Ainetter, Philipp Drescher, Stefan Hauswiesner, Arno Coomans, Giacomo Nazzaro, Konstantinos Vardis, Markus Steinberger  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2506.20367v1.pdf)  
  <details><summary>Abstract</summary>

  Recent advances in text-to-3D scene generation have demonstrated significant potential to transform content creation across multiple industries. Although the research community has made impressive progress in addressing the challenges of this complex task, existing methods often generate environments that are only front-facing, lack visual fidelity, exhibit limited scene understanding, and are typically fine-tuned for either indoor or outdoor settings. In this work, we address these issues and propose DreamAnywhere, a modular system for the fast generation and prototyping of 3D scenes. Our system synthesizes a 360° panoramic image from text, decomposes it into background and objects, constructs a complete 3D representation through hybrid inpainting, and lifts object masks to detailed 3D objects that are placed in the virtual environment. DreamAnywhere supports immersive navigation and intuitive object-level editing, making it ideal for scene exploration, visual mock-ups, and rapid prototyping -- all with minimal manual modeling. These features make our system particularly...
  </details>  
  Keywords: text-to-3d, novel view synthesis  
- **[3D Arena: An Open Platform for Generative 3D Evaluation](https://arxiv.org/abs/2506.18787v1)**  
  Authors: Dylan Ebert  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2506.18787v1.pdf)  
  <details><summary>Abstract</summary>

  Evaluating Generative 3D models remains challenging due to misalignment between automated metrics and human perception of quality. Current benchmarks rely on image-based metrics that ignore 3D structure or geometric measures that fail to capture perceptual appeal and real-world utility. To address this gap, we present 3D Arena, an open platform for evaluating image-to-3D generation models through large-scale human preference collection using pairwise comparisons. Since launching in June 2024, the platform has collected 123,243 votes from 8,096 users across 19 state-of-the-art models, establishing the largest human preference evaluation for Generative 3D. We contribute the iso3d dataset of 100 evaluation prompts and demonstrate quality control achieving 99.75% user authenticity through statistical fraud detection. Our ELO-based ranking system provides reliable model assessment, with the platform becoming an established evaluation resource. Through analysis of this preference data, we present insights into human preference patterns. Our findings reveal preferences for visual presentation features, with Gaussian...
  </details>  
  Keywords: image-to-3d  
- **[Reconstructing Tornadoes in 3D with Gaussian Splatting](https://arxiv.org/abs/2506.18677v2)**  
  Authors: Adam Yang, Nadula Kadawedduwa, Tianfu Wang, Sunny Sharma, Emily F. Wisinski, Jhayron S. Pérez-Carrasquilla, Kyle J. C. Hall, Dean Calhoun, Jonathan Starfeldt, Timothy P. Canty, Maria Molina, Christopher Metzler  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2506.18677v2.pdf)  
  <details><summary>Abstract</summary>

  Accurately reconstructing the 3D structure of tornadoes is critically important for understanding and preparing for this highly destructive weather phenomenon. While modern 3D scene reconstruction techniques, such as 3D Gaussian splatting (3DGS), could provide a valuable tool for reconstructing the 3D structure of tornados, at present we are critically lacking a controlled tornado dataset with which to develop and validate these tools. In this work we capture and release a novel multiview dataset of a small lab-based tornado. We demonstrate one can effectively reconstruct and visualize the 3D structure of this tornado using 3DGS.
  </details>  
  Keywords: 3d scene reconstruction  
- **[Auto-Regressively Generating Multi-View Consistent Images](https://arxiv.org/abs/2506.18527v2)**  
  Authors: JiaKui Hu, Yuxiao Yang, Jialun Liu, Jinbo Wu, Chen Zhao, Yanye Lu  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2506.18527v2.pdf) | [![GitHub](https://img.shields.io/github/stars/MILab-PKU/MVAR?style=social)](https://github.com/MILab-PKU/MVAR)  
  <details><summary>Abstract</summary>

  Generating multi-view images from human instructions is crucial for 3D content creation. The primary challenges involve maintaining consistency across multiple views and effectively synthesizing shapes and textures under diverse conditions. In this paper, we propose the Multi-View Auto-Regressive (\textbf{MV-AR}) method, which leverages an auto-regressive model to progressively generate consistent multi-view images from arbitrary prompts. Firstly, the next-token-prediction capability of the AR model significantly enhances its effectiveness in facilitating progressive multi-view synthesis. When generating widely-separated views, MV-AR can utilize all its preceding views to extract effective reference information. Subsequently, we propose a unified model that accommodates various prompts via architecture designing and training strategies. To address multiple conditions, we introduce condition injection modules for text, camera pose, image, and shape. To manage multi-modal conditions simultaneously, a progressive training strategy is employed. This strategy initially adopts the text-to-multi-view (t2mv) model as a baseline to enhance the development of a comprehensive X-to-multi-view (X2mv)...
  </details>  
  Keywords: multi-view synthesis  
- **[End-to-End Fine-Tuning of 3D Texture Generation using Differentiable Rewards](https://arxiv.org/abs/2506.18331v4)**  
  Authors: AmirHossein Zamani, Tianhao Xie, Amir G. Aghdam, Tiberiu Popa, Eugene Belilovsky  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2506.18331v4.pdf) | [![GitHub](https://img.shields.io/github/stars/AHHHZ975/Differentiable-Texture-Learning?style=social)](https://github.com/AHHHZ975/Differentiable-Texture-Learning)  
  <details><summary>Abstract</summary>

  While recent 3D generative models can produce high-quality texture images, they often fail to capture human preferences or meet task-specific requirements. Moreover, a core challenge in the 3D texture generation domain is that most existing approaches rely on repeated calls to 2D text-to-image generative models, which lack an inherent understanding of the 3D structure of the input 3D mesh object. To alleviate these issues, we propose an end-to-end differentiable, reinforcement-learning-free framework that embeds human feedback, expressed as differentiable reward functions, directly into the 3D texture synthesis pipeline. By back-propagating preference signals through both geometric and appearance modules of the proposed framework, our method generates textures that respect the 3D geometry structure and align with desired criteria. To demonstrate its versatility, we introduce three novel geometry-aware reward functions, which offer a more controllable and interpretable pathway for creating high-quality 3D content from natural language. By conducting qualitative, quantitative, and user-preference evaluations...
  </details>  
  Keywords: texture synthesis  
- **[Limitations of NERF with pre-trained Vision Features for Few-Shot 3D Reconstruction](https://arxiv.org/abs/2506.18208v1)**  
  Authors: Ankit Sanjyal  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2506.18208v1.pdf)  
  <details><summary>Abstract</summary>

  Neural Radiance Fields (NeRF) have revolutionized 3D scene reconstruction from sparse image collections. Recent work has explored integrating pre-trained vision features, particularly from DINO, to enhance few-shot reconstruction capabilities. However, the effectiveness of such approaches remains unclear, especially in extreme few-shot scenarios. In this paper, we present a systematic evaluation of DINO-enhanced NeRF models, comparing baseline NeRF, frozen DINO features, LoRA fine-tuned features, and multi-scale feature fusion. Surprisingly, our experiments reveal that all DINO variants perform worse than the baseline NeRF, achieving PSNR values around 12.9 to 13.0 compared to the baseline's 14.71. This counterintuitive result suggests that pre-trained vision features may not be beneficial for few-shot 3D reconstruction and may even introduce harmful biases. We analyze potential causes including feature-task mismatch, overfitting to limited data, and integration challenges. Our findings challenge common assumptions in the field and suggest that simpler architectures focusing on geometric consistency may be more effective...
  </details>  
  Keywords: 3d scene reconstruction  
- **[DreamJourney: Perpetual View Generation with Video Diffusion Models](https://arxiv.org/abs/2506.17705v1)**  
  Authors: Bo Pan, Yang Chen, Yingwei Pan, Ting Yao, Wei Chen, Tao Mei  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2506.17705v1.pdf) | [![Project](https://img.shields.io/badge/-Project-blue)](https://dream-journey.vercel.app.) | [![Demo](https://img.shields.io/badge/-Demo-brightgreen)](https://dream-journey.vercel.app)  
  <details><summary>Abstract</summary>

  Perpetual view generation aims to synthesize a long-term video corresponding to an arbitrary camera trajectory solely from a single input image. Recent methods commonly utilize a pre-trained text-to-image diffusion model to synthesize new content of previously unseen regions along camera movement. However, the underlying 2D diffusion model lacks 3D awareness and results in distorted artifacts. Moreover, they are limited to generating views of static 3D scenes, neglecting to capture object movements within the dynamic 4D world. To alleviate these issues, we present DreamJourney, a two-stage framework that leverages the world simulation capacity of video diffusion models to trigger a new perpetual scene view generation task with both camera movements and object dynamics. Specifically, in stage I, DreamJourney first lifts the input image to 3D point cloud and renders a sequence of partial images from a specific camera trajectory. A video diffusion model is then utilized as generative prior to complete...
  </details>  
- **[Hunyuan3D 2.5: Towards High-Fidelity 3D Assets Generation with Ultimate Details](https://arxiv.org/abs/2506.16504v1)**  
  Authors: Zeqiang Lai, Yunfei Zhao, Haolin Liu, Zibo Zhao, Qingxiang Lin, Huiwen Shi, Xianghui Yang, Mingxin Yang, Shuhui Yang, Yifei Feng, Sheng Zhang, Xin Huang, Di Luo, Fan Yang, Fang Yang, Lifu Wang, Sicong Liu, Yixuan Tang, Yulin Cai, Zebin He, Tian Liu, Yuhong Liu, Jie Jiang, Linus, Jingwei Huang, Chunchao Guo  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2506.16504v1.pdf)  
  <details><summary>Abstract</summary>

  In this report, we present Hunyuan3D 2.5, a robust suite of 3D diffusion models aimed at generating high-fidelity and detailed textured 3D assets. Hunyuan3D 2.5 follows two-stages pipeline of its previous version Hunyuan3D 2.0, while demonstrating substantial advancements in both shape and texture generation. In terms of shape generation, we introduce a new shape foundation model -- LATTICE, which is trained with scaled high-quality datasets, model-size, and compute. Our largest model reaches 10B parameters and generates sharp and detailed 3D shape with precise image-3D following while keeping mesh surface clean and smooth, significantly closing the gap between generated and handcrafted 3D shapes. In terms of texture generation, it is upgraded with phyiscal-based rendering (PBR) via a novel multi-view architecture extended from Hunyuan3D 2.0 Paint model. Our extensive evaluation shows that Hunyuan3D 2.5 significantly outperforms previous methods in both shape and end-to-end texture generation.
  </details>  
- **[R3eVision: A Survey on Robust Rendering, Restoration, and Enhancement for 3D Low-Level Vision](https://arxiv.org/abs/2506.16262v2)**  
  Authors: Weeyoung Kwon, Jeahun Sung, Minkyu Jeon, Chanho Eom, Jihyong Oh  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2506.16262v2.pdf)  
  <details><summary>Abstract</summary>

  Neural rendering methods such as Neural Radiance Fields (NeRF) and 3D Gaussian Splatting (3DGS) have achieved significant progress in photorealistic 3D scene reconstruction and novel view synthesis. However, most existing models assume clean and high-resolution (HR) multi-view inputs, which limits their robustness under real-world degradations such as noise, blur, low-resolution (LR), and weather-induced artifacts. To address these limitations, the emerging field of 3D Low-Level Vision (3D LLV) extends classical 2D Low-Level Vision tasks including super-resolution (SR), deblurring, weather degradation removal, restoration, and enhancement into the 3D spatial domain. This survey, referred to as R\textsuperscript{3}eVision, provides a comprehensive overview of robust rendering, restoration, and enhancement for 3D LLV by formalizing the degradation-aware rendering problem and identifying key challenges related to spatio-temporal consistency and ill-posed optimization. Recent methods that integrate LLV into neural rendering frameworks are categorized to illustrate how they enable high-fidelity 3D reconstruction under adverse conditions. Application domains such as...
  </details>  
  Keywords: novel view synthesis, 3d scene reconstruction  
- **[Implicit 3D scene reconstruction using deep learning towards efficient collision understanding in autonomous driving](https://arxiv.org/abs/2506.15806v1)**  
  Authors: Akarshani Ramanayake, Nihal Kodikara  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2506.15806v1.pdf)  
  <details><summary>Abstract</summary>

  In crowded urban environments where traffic is dense, current technologies struggle to oversee tight navigation, but surface-level understanding allows autonomous vehicles to safely assess proximity to surrounding obstacles. 3D or 2D scene mapping of the surrounding objects is an essential task in addressing the above problem. Despite its importance in dense vehicle traffic conditions, 3D scene reconstruction of object shapes with higher boundary level accuracy is not yet entirely considered in current literature. The sign distance function represents any shape through parameters that calculate the distance from any point in space to the closest obstacle surface, making it more efficient in terms of storage. In recent studies, researchers have started to formulate problems with Implicit 3D reconstruction methods in the autonomous driving domain, highlighting the possibility of using sign distance function to map obstacles effectively. This research addresses this gap by developing a learning-based 3D scene reconstruction methodology that leverages...
  </details>  
  Keywords: 3d scene reconstruction  
- **[Nabla-R2D3: Effective and Efficient 3D Diffusion Alignment with 2D Rewards](https://arxiv.org/abs/2506.15684v1)**  
  Authors: Qingming Liu, Zhen Liu, Dinghuai Zhang, Kui Jia  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2506.15684v1.pdf)  
  <details><summary>Abstract</summary>

  Generating high-quality and photorealistic 3D assets remains a longstanding challenge in 3D vision and computer graphics. Although state-of-the-art generative models, such as diffusion models, have made significant progress in 3D generation, they often fall short of human-designed content due to limited ability to follow instructions, align with human preferences, or produce realistic textures, geometries, and physical attributes. In this paper, we introduce Nabla-R2D3, a highly effective and sample-efficient reinforcement learning alignment framework for 3D-native diffusion models using 2D rewards. Built upon the recently proposed Nabla-GFlowNet method, which matches the score function to reward gradients in a principled manner for reward finetuning, our Nabla-R2D3 enables effective adaptation of 3D diffusion models using only 2D reward signals. Extensive experiments show that, unlike vanilla finetuning baselines which either struggle to converge or suffer from reward hacking, Nabla-R2D3 consistently achieves higher rewards and reduced prior forgetting within a few finetuning steps.
  </details>  
- **[Hunyuan3D 2.1: From Images to High-Fidelity 3D Assets with Production-Ready PBR Material](https://arxiv.org/abs/2506.15442v1)**  
  Authors: Team Hunyuan3D, Shuhui Yang, Mingxin Yang, Yifei Feng, Xin Huang, Sheng Zhang, Zebin He, Di Luo, Haolin Liu, Yunfei Zhao, Qingxiang Lin, Zeqiang Lai, Xianghui Yang, Huiwen Shi, Zibo Zhao, Bowen Zhang, Hongyu Yan, Lifu Wang, Sicong Liu, Jihong Zhang, Meng Chen, Liang Dong, Yiwen Jia, Yulin Cai, Jiaao Yu, Yixuan Tang, Dongyuan Guo, Junlin Yu, Hao Zhang, Zheng Ye, Peng He, Runzhou Wu, Shida Wei, Chao Zhang, Yonghao Tan, Yifu Sun, Lin Niu, Shirui Huang, Bojian Zheng, Shu Liu, Shilin Chen, Xiang Yuan, Xiaofeng Yang, Kai Liu, Jianchen Zhu, Peng Chen, Tian Liu, Di Wang, Yuhong Liu, Linus, Jie Jiang, Jingwei Huang, Chunchao Guo  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2506.15442v1.pdf)  
  <details><summary>Abstract</summary>

  3D AI-generated content (AIGC) is a passionate field that has significantly accelerated the creation of 3D models in gaming, film, and design. Despite the development of several groundbreaking models that have revolutionized 3D generation, the field remains largely accessible only to researchers, developers, and designers due to the complexities involved in collecting, processing, and training 3D models. To address these challenges, we introduce Hunyuan3D 2.1 as a case study in this tutorial. This tutorial offers a comprehensive, step-by-step guide on processing 3D data, training a 3D generative model, and evaluating its performance using Hunyuan3D 2.1, an advanced system for producing high-resolution, textured 3D assets. The system comprises two core components: the Hunyuan3D-DiT for shape generation and the Hunyuan3D-Paint for texture synthesis. We will explore the entire workflow, including data preparation, model architecture, training strategies, evaluation metrics, and deployment. By the conclusion of this tutorial, you will have the knowledge to...
  </details>  
  Keywords: texture synthesis  
- **[RobotSmith: Generative Robotic Tool Design for Acquisition of Complex Manipulation Skills](https://arxiv.org/abs/2506.14763v1)**  
  Authors: Chunru Lin, Haotian Yuan, Yian Wang, Xiaowen Qiu, Tsun-Hsuan Wang, Minghao Guo, Bohan Wang, Yashraj Narang, Dieter Fox, Chuang Gan  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2506.14763v1.pdf)  
  <details><summary>Abstract</summary>

  Endowing robots with tool design abilities is critical for enabling them to solve complex manipulation tasks that would otherwise be intractable. While recent generative frameworks can automatically synthesize task settings, such as 3D scenes and reward functions, they have not yet addressed the challenge of tool-use scenarios. Simply retrieving human-designed tools might not be ideal since many tools (e.g., a rolling pin) are difficult for robotic manipulators to handle. Furthermore, existing tool design approaches either rely on predefined templates with limited parameter tuning or apply generic 3D generation methods that are not optimized for tool creation. To address these limitations, we propose RobotSmith, an automated pipeline that leverages the implicit physical knowledge embedded in vision-language models (VLMs) alongside the more accurate physics provided by physics simulations to design and use tools for robotic manipulation. Our system (1) iteratively proposes tool designs using collaborative VLM agents, (2) generates low-level robot trajectories...
  </details>  
- **[HRGS: Hierarchical Gaussian Splatting for Memory-Efficient High-Resolution 3D Reconstruction](https://arxiv.org/abs/2506.14229v1)**  
  Authors: Changbai Li, Haodong Zhu, Hanlin Chen, Juan Zhang, Tongfei Chen, Shuo Yang, Shuwei Shao, Wenhao Dong, Baochang Zhang  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2506.14229v1.pdf)  
  <details><summary>Abstract</summary>

  3D Gaussian Splatting (3DGS) has made significant strides in real-time 3D scene reconstruction, but faces memory scalability issues in high-resolution scenarios. To address this, we propose Hierarchical Gaussian Splatting (HRGS), a memory-efficient framework with hierarchical block-level optimization. First, we generate a global, coarse Gaussian representation from low-resolution data. Then, we partition the scene into multiple blocks, refining each block with high-resolution data. The partitioning involves two steps: Gaussian partitioning, where irregular scenes are normalized into a bounded cubic space with a uniform grid for task distribution, and training data partitioning, where only relevant observations are retained for each block. By guiding block refinement with the coarse Gaussian prior, we ensure seamless Gaussian fusion across adjacent blocks. To reduce computational demands, we introduce Importance-Driven Gaussian Pruning (IDGP), which computes importance scores for each Gaussian and removes those with minimal contribution, speeding up convergence and reducing memory usage. Additionally, we incorporate normal...
  </details>  
  Keywords: novel view synthesis, 3d scene reconstruction  
- **[Disentangling 3D from Large Vision-Language Models for Controlled Portrait Generation](https://arxiv.org/abs/2506.14015v1)**  
  Authors: Nick Yiwen Huang, Akin Caliskan, Berkay Kicanaoglu, James Tompkin, Hyeongwoo Kim  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2506.14015v1.pdf)  
  <details><summary>Abstract</summary>

  We consider the problem of disentangling 3D from large vision-language models, which we show on generative 3D portraits. This allows free-form text control of appearance attributes like age, hair style, and glasses, and 3D geometry control of face expression and camera pose. In this setting, we assume we use a pre-trained large vision-language model (LVLM; CLIP) to generate from a smaller 2D dataset with no additional paired labels and with a pre-defined 3D morphable model (FLAME). First, we disentangle using canonicalization to a 2D reference frame from a deformable neural 3D triplane representation. But another form of entanglement arises from the significant noise in the LVLM's embedding space that describes irrelevant features. This damages output quality and diversity, but we overcome this with a Jacobian regularization that can be computed efficiently with a stochastic approximator. Compared to existing methods, our approach produces portraits with added text and 3D control, where...
  </details>  
  Keywords: 3d generator  
- **[Dive3D: Diverse Distillation-based Text-to-3D Generation via Score Implicit Matching](https://arxiv.org/abs/2506.13594v1)**  
  Authors: Weimin Bai, Yubo Li, Wenzheng Chen, Weijian Luo, He Sun  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2506.13594v1.pdf)  
  <details><summary>Abstract</summary>

  Distilling pre-trained 2D diffusion models into 3D assets has driven remarkable advances in text-to-3D synthesis. However, existing methods typically rely on Score Distillation Sampling (SDS) loss, which involves asymmetric KL divergence--a formulation that inherently favors mode-seeking behavior and limits generation diversity. In this paper, we introduce Dive3D, a novel text-to-3D generation framework that replaces KL-based objectives with Score Implicit Matching (SIM) loss, a score-based objective that effectively mitigates mode collapse. Furthermore, Dive3D integrates both diffusion distillation and reward-guided optimization under a unified divergence perspective. Such reformulation, together with SIM loss, yields significantly more diverse 3D outputs while improving text alignment, human preference, and overall visual fidelity. We validate Dive3D across various 2D-to-3D prompts and find that it consistently outperforms prior methods in qualitative assessments, including diversity, photorealism, and aesthetic appeal. We further evaluate its performance on the GPTEval3D benchmark, comparing against nine state-of-the-art baselines. Dive3D also achieves strong results on...
  </details>  
  Keywords: text-to-3d  
- **[3D Hand Mesh-Guided AI-Generated Malformed Hand Refinement with Hand Pose Transformation via Diffusion Model](https://arxiv.org/abs/2506.12680v2)**  
  Authors: Chen-Bin Feng, Kangdao Liu, Jian Sun, Jiping Jin, Yiguo Jiang, Chi-Man Vong  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2506.12680v2.pdf)  
  <details><summary>Abstract</summary>

  The malformed hands in the AI-generated images seriously affect the authenticity of the images. To refine malformed hands, existing depth-based approaches use a hand depth estimator to guide the refinement of malformed hands. Due to the performance limitations of the hand depth estimator, many hand details cannot be represented, resulting in errors in the generated hands, such as confusing the palm and the back of the hand. To solve this problem, we propose a 3D mesh-guided refinement framework using a diffusion pipeline. We use a state-of-the-art 3D hand mesh estimator, which provides more details of the hands. For training, we collect and reannotate a dataset consisting of RGB images and 3D hand mesh. Then we design a diffusion inpainting model to generate refined outputs guided by 3D hand meshes. For inference, we propose a double check algorithm to facilitate the 3D hand mesh estimator to obtain robust hand mesh guidance...
  </details>  
- **[3D Skin Segmentation Methods in Medical Imaging: A Comparison](https://arxiv.org/abs/2506.11852v1)**  
  Authors: Martina Paccini, Giuseppe Patanè  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2506.11852v1.pdf)  
  <details><summary>Abstract</summary>

  Automatic segmentation of anatomical structures is critical in medical image analysis, aiding diagnostics and treatment planning. Skin segmentation plays a key role in registering and visualising multimodal imaging data. 3D skin segmentation enables applications in personalised medicine, surgical planning, and remote monitoring, offering realistic patient models for treatment simulation, procedural visualisation, and continuous condition tracking. This paper analyses and compares algorithmic and AI-driven skin segmentation approaches, emphasising key factors to consider when selecting a strategy based on data availability and application requirements. We evaluate an iterative region-growing algorithm and the TotalSegmentator, a deep learning-based approach, across different imaging modalities and anatomical regions. Our tests show that AI segmentation excels in automation but struggles with MRI due to its CT-based training, while the graphics-based method performs better for MRIs but introduces more noise. AI-driven segmentation also automates patient bed removal in CT, whereas the graphics-based method requires manual intervention.
  </details>  
- **[Vision-based Lifting of 2D Object Detections for Automated Driving](https://arxiv.org/abs/2506.11839v1)**  
  Authors: Hendrik Königshof, Kun Li, Christoph Stiller  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2506.11839v1.pdf)  
  <details><summary>Abstract</summary>

  Image-based 3D object detection is an inevitable part of autonomous driving because cheap onboard cameras are already available in most modern cars. Because of the accurate depth information, currently, most state-of-the-art 3D object detectors heavily rely on LiDAR data. In this paper, we propose a pipeline which lifts the results of existing vision-based 2D algorithms to 3D detections using only cameras as a cost-effective alternative to LiDAR. In contrast to existing approaches, we focus not only on cars but on all types of road users. To the best of our knowledge, we are the first using a 2D CNN to process the point cloud for each 2D detection to keep the computational effort as low as possible. Our evaluation on the challenging KITTI 3D object detection benchmark shows results comparable to state-of-the-art image-based approaches while having a runtime of only a third.
  </details>  
- **[VEIGAR: View-consistent Explicit Inpainting and Geometry Alignment for 3D object Removal](https://arxiv.org/abs/2506.15821v1)**  
  Authors: Pham Khai Nguyen Do, Bao Nguyen Tran, Nam Nguyen, Duc Dung Nguyen  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2506.15821v1.pdf)  
  <details><summary>Abstract</summary>

  Recent advances in Novel View Synthesis (NVS) and 3D generation have significantly improved editing tasks, with a primary emphasis on maintaining cross-view consistency throughout the generative process. Contemporary methods typically address this challenge using a dual-strategy framework: performing consistent 2D inpainting across all views guided by embedded priors either explicitly in pixel space or implicitly in latent space; and conducting 3D reconstruction with additional consistency guidance. Previous strategies, in particular, often require an initial 3D reconstruction phase to establish geometric structure, introducing considerable computational overhead. Even with the added cost, the resulting reconstruction quality often remains suboptimal. In this paper, we present VEIGAR, a computationally efficient framework that outperforms existing methods without relying on an initial reconstruction phase. VEIGAR leverages a lightweight foundation model to reliably align priors explicitly in the pixel space. In addition, we introduce a novel supervision strategy based on scale-invariant depth loss, which removes the need...
  </details>  
  Keywords: novel view synthesis  
- **[Hadamard Encoded Row Column Ultrasonic Expansive Scanning (HERCULES) with Bias-Switchable Row-Column Arrays](https://arxiv.org/abs/2506.11443v2)**  
  Authors: Darren Olufemi Dahunsi, Randy Palmar, Tyler Henry, Mohammad Rahim Sobhani, Negar Majidi, Joy Wang, Afshin Kashani Ilkhechi, Jeremy Brown, Roger Zemp  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2506.11443v2.pdf)  
  <details><summary>Abstract</summary>

  Top-Orthogonal-to-Bottom-Electrode (TOBE) arrays, also known as bias-switchable row-column arrays (RCAs), allow for imaging techniques otherwise impossible for non-bias-switachable RCAs. Hadamard Encoded Row Column Ultrasonic Expansive Scanning (HERCULES) is a novel imaging technique that allows for expansive 3D scanning by transmitting plane or cylindrical wavefronts and receiving using Hadamard-Encoded-Read-Out (HERO) to perform beamforming on what is effectively a full 2D synthetic receive aperture. This allows imaging beyond the shadow of the aperture of the RCA array, potentially allows for whole organ imaging and 3D visualization of tissue morphology. It additionally enables view large volumes through limited windows. In this work we demonstrated with simulation that we are able to image at comparable resolution to existing RCA imaging methods at tens to hundreds of volumes per second. We validated these simulations by demonstrating an experimental implementation of HERCULES using a custom fabricated TOBE array, custom biasing electronics, and a research ultrasound system....
  </details>  
- **[InstaInpaint: Instant 3D-Scene Inpainting with Masked Large Reconstruction Model](https://arxiv.org/abs/2506.10980v1)**  
  Authors: Junqi You, Chieh Hubert Lin, Weijie Lyu, Zhengbo Zhang, Ming-Hsuan Yang  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2506.10980v1.pdf) | [![Project](https://img.shields.io/badge/-Project-blue)](https://dhmbb2.github.io/InstaInpaint_page)  
  <details><summary>Abstract</summary>

  Recent advances in 3D scene reconstruction enable real-time viewing in virtual and augmented reality. To support interactive operations for better immersiveness, such as moving or editing objects, 3D scene inpainting methods are proposed to repair or complete the altered geometry. However, current approaches rely on lengthy and computationally intensive optimization, making them impractical for real-time or online applications. We propose InstaInpaint, a reference-based feed-forward framework that produces 3D-scene inpainting from a 2D inpainting proposal within 0.4 seconds. We develop a self-supervised masked-finetuning strategy to enable training of our custom large reconstruction model (LRM) on the large-scale dataset. Through extensive experiments, we analyze and identify several key designs that improve generalization, textural consistency, and geometric correctness. InstaInpaint achieves a 1000x speed-up from prior methods while maintaining a state-of-the-art performance across two standard benchmarks. Moreover, we show that InstaInpaint generalizes well to flexible downstream applications such as object insertion and multi-region inpainting....
  </details>  
  Keywords: 3d scene reconstruction  
- **[TexTailor: Customized Text-aligned Texturing via Effective Resampling](https://arxiv.org/abs/2506.10612v1)**  
  Authors: Suin Lee, Dae-Shik Kim  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2506.10612v1.pdf) | [![GitHub](https://img.shields.io/github/stars/Adios42/Textailor?style=social)](https://github.com/Adios42/Textailor)  
  <details><summary>Abstract</summary>

  We present TexTailor, a novel method for generating consistent object textures from textual descriptions. Existing text-to-texture synthesis approaches utilize depth-aware diffusion models to progressively generate images and synthesize textures across predefined multiple viewpoints. However, these approaches lead to a gradual shift in texture properties across viewpoints due to (1) insufficient integration of previously synthesized textures at each viewpoint during the diffusion process and (2) the autoregressive nature of the texture synthesis process. Moreover, the predefined selection of camera positions, which does not account for the object's geometry, limits the effective use of texture information synthesized from different viewpoints, ultimately degrading overall texture consistency. In TexTailor, we address these issues by (1) applying a resampling scheme that repeatedly integrates information from previously synthesized textures within the diffusion process, and (2) fine-tuning a depth-aware diffusion model on these resampled textures. During this process, we observed that using only a few training images...
  </details>  
  Keywords: texture synthesis  
- **[EmbodiedGen: Towards a Generative 3D World Engine for Embodied Intelligence](https://arxiv.org/abs/2506.10600v2)**  
  Authors: Xinjie Wang, Liu Liu, Yu Cao, Ruiqi Wu, Wenkang Qin, Dehui Wang, Wei Sui, Zhizhong Su  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2506.10600v2.pdf) | [![Project](https://img.shields.io/badge/-Project-blue)](https://horizonrobotics.github.io/robot_lab/embodied_gen/index.html)  
  <details><summary>Abstract</summary>

  Constructing a physically realistic and accurately scaled simulated 3D world is crucial for the training and evaluation of embodied intelligence tasks. The diversity, realism, low cost accessibility and affordability of 3D data assets are critical for achieving generalization and scalability in embodied AI. However, most current embodied intelligence tasks still rely heavily on traditional 3D computer graphics assets manually created and annotated, which suffer from high production costs and limited realism. These limitations significantly hinder the scalability of data driven approaches. We present EmbodiedGen, a foundational platform for interactive 3D world generation. It enables the scalable generation of high-quality, controllable and photorealistic 3D assets with accurate physical properties and real-world scale in the Unified Robotics Description Format (URDF) at low cost. These assets can be directly imported into various physics simulation engines for fine-grained physical control, supporting downstream tasks in training and evaluation. EmbodiedGen is an easy-to-use, full-featured toolkit composed...
  </details>  
  Keywords: text-to-3d, articulated object generation, image-to-3d  
- **[J-DDL: Surface Damage Detection and Localization System for Fighter Aircraft](https://arxiv.org/abs/2506.10505v1)**  
  Authors: Jin Huang, Mingqiang Wei, Zikuan Li, Hangyu Qu, Wei Zhao, Xinyu Bai  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2506.10505v1.pdf)  
  <details><summary>Abstract</summary>

  Ensuring the safety and extended operational life of fighter aircraft necessitates frequent and exhaustive inspections. While surface defect detection is feasible for human inspectors, manual methods face critical limitations in scalability, efficiency, and consistency due to the vast surface area, structural complexity, and operational demands of aircraft maintenance. We propose a smart surface damage detection and localization system for fighter aircraft, termed J-DDL. J-DDL integrates 2D images and 3D point clouds of the entire aircraft surface, captured using a combined system of laser scanners and cameras, to achieve precise damage detection and localization. Central to our system is a novel damage detection network built on the YOLO architecture, specifically optimized for identifying surface defects in 2D aircraft images. Key innovations include lightweight Fasternet blocks for efficient feature extraction, an optimized neck architecture incorporating Efficient Multiscale Attention (EMA) modules for superior feature aggregation, and the introduction of a novel loss function,...
  </details>  
- **[Hearing Hands: Generating Sounds from Physical Interactions in 3D Scenes](https://arxiv.org/abs/2506.09989v1)**  
  Authors: Yiming Dou, Wonseok Oh, Yuqing Luo, Antonio Loquercio, Andrew Owens  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2506.09989v1.pdf) | [![Project](https://img.shields.io/badge/-Project-blue)](https://www.yimingdou.com/hearing_hands)  
  <details><summary>Abstract</summary>

  We study the problem of making 3D scene reconstructions interactive by asking the following question: can we predict the sounds of human hands physically interacting with a scene? First, we record a video of a human manipulating objects within a 3D scene using their hands. We then use these action-sound pairs to train a rectified flow model to map 3D hand trajectories to their corresponding audio. At test time, a user can query the model for other actions, parameterized as sequences of hand poses, to estimate their corresponding sounds. In our experiments, we find that our generated sounds accurately convey material properties and actions, and that they are often indistinguishable to human observers from real sounds. Project page: https://www.yimingdou.com/hearing_hands/
  </details>  
  Keywords: 3d scene reconstruction  
- **[DreamCS: Geometry-Aware Text-to-3D Generation with Unpaired 3D Reward Supervision](https://arxiv.org/abs/2506.09814v2)**  
  Authors: Xiandong Zou, Ruihao Xia, Hongsong Wang, Pan Zhou  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2506.09814v2.pdf)  
  <details><summary>Abstract</summary>

  While text-to-3D generation has attracted growing interest, existing methods often struggle to produce 3D assets that align well with human preferences. Current preference alignment techniques for 3D content typically rely on hardly-collected preference-paired multi-view 2D images to train 2D reward models, when then guide 3D generation -- leading to geometric artifacts due to their inherent 2D bias. To address these limitations, we construct 3D-MeshPref, the first large-scale unpaired 3D preference dataset, featuring diverse 3D meshes annotated by a large language model and refined by human evaluators. We then develop RewardCS, the first reward model trained directly on unpaired 3D-MeshPref data using a novel Cauchy-Schwarz divergence objective, enabling effective learning of human-aligned 3D geometric preferences without requiring paired comparisons. Building on this, we propose DreamCS, a unified framework that integrates RewardCS into text-to-3D pipelines -- enhancing both implicit and explicit 3D generation with human preference feedback. Extensive experiments show DreamCS outperforms...
  </details>  
  Keywords: text-to-3d  
- **[3DGeoDet: General-purpose Geometry-aware Image-based 3D Object Detection](https://arxiv.org/abs/2506.09541v1)**  
  Authors: Yi Zhang, Yi Wang, Yawen Cui, Lap-Pui Chau  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2506.09541v1.pdf) | [![Project](https://img.shields.io/badge/-Project-blue)](https://cindy0725.github.io/3DGeoDet)  
  <details><summary>Abstract</summary>

  This paper proposes 3DGeoDet, a novel geometry-aware 3D object detection approach that effectively handles single- and multi-view RGB images in indoor and outdoor environments, showcasing its general-purpose applicability. The key challenge for image-based 3D object detection tasks is the lack of 3D geometric cues, which leads to ambiguity in establishing correspondences between images and 3D representations. To tackle this problem, 3DGeoDet generates efficient 3D geometric representations in both explicit and implicit manners based on predicted depth information. Specifically, we utilize the predicted depth to learn voxel occupancy and optimize the voxelized 3D feature volume explicitly through the proposed voxel occupancy attention. To further enhance 3D awareness, the feature volume is integrated with an implicit 3D representation, the truncated signed distance function (TSDF). Without requiring supervision from 3D signals, we significantly improve the model's comprehension of 3D geometry by leveraging intermediate 3D representations and achieve end-to-end training. Our approach surpasses the...
  </details>  
- **[Gaussian2Scene: 3D Scene Representation Learning via Self-supervised Learning with 3D Gaussian Splatting](https://arxiv.org/abs/2506.08777v2)**  
  Authors: Keyi Liu, Weidong Yang, Ben Fei, Ying He  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2506.08777v2.pdf)  
  <details><summary>Abstract</summary>

  Self-supervised learning (SSL) for point cloud pre-training has become a cornerstone for many 3D vision tasks, enabling effective learning from large-scale unannotated data. At the scene level, existing SSL methods often incorporate volume rendering into the pre-training framework, using RGB-D images as reconstruction signals to facilitate cross-modal learning. This strategy promotes alignment between 2D and 3D modalities and enables the model to benefit from rich visual cues in the RGB-D inputs. However, these approaches are limited by their reliance on implicit scene representations and high memory demands. Furthermore, since their reconstruction objectives are applied only in 2D space, they often fail to capture underlying 3D geometric structures. To address these challenges, we propose Gaussian2Scene, a novel scene-level SSL framework that leverages the efficiency and explicit nature of 3D Gaussian Splatting (3DGS) for pre-training. The use of 3DGS not only alleviates the computational burden associated with volume rendering but also supports...
  </details>  
  Keywords: 3d scene reconstruction  
- **[Orientation Matters: Making 3D Generative Models Orientation-Aligned](https://arxiv.org/abs/2506.08640v2)**  
  Authors: Yichong Lu, Yuzhuo Tian, Zijin Jiang, Yikun Zhao, Yuanbo Yang, Hao Ouyang, Haoji Hu, Huimin Yu, Yujun Shen, Yiyi Liao  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2506.08640v2.pdf)  
  <details><summary>Abstract</summary>

  Humans intuitively perceive object shape and orientation from a single image, guided by strong priors about canonical poses. However, existing 3D generative models often produce misaligned results due to inconsistent training data, limiting their usability in downstream tasks. To address this gap, we introduce the task of orientation-aligned 3D object generation: producing 3D objects from single images with consistent orientations across categories. To facilitate this, we construct Objaverse-OA, a dataset of 14,832 orientation-aligned 3D models spanning 1,008 categories. Leveraging Objaverse-OA, we fine-tune two representative 3D generative models based on multi-view diffusion and 3D variational autoencoder frameworks to produce aligned objects that generalize well to unseen objects across various categories. Experimental results demonstrate the superiority of our method over post-hoc alignment approaches. Furthermore, we showcase downstream applications enabled by our aligned object generation, including zero-shot object orientation estimation via analysis-by-synthesis and efficient arrow-based object rotation manipulation.
  </details>  
- **[Aligning Text, Images, and 3D Structure Token-by-Token](https://arxiv.org/abs/2506.08002v2)**  
  Authors: Aadarsh Sahoo, Vansh Tibrewal, Georgia Gkioxari  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2506.08002v2.pdf) | [![Project](https://img.shields.io/badge/-Project-blue)](https://glab-caltech.github.io/kyvo)  
  <details><summary>Abstract</summary>

  Creating machines capable of understanding the world in 3D is essential in assisting designers that build and edit 3D environments and robots navigating and interacting within a three-dimensional space. Inspired by advances in language and image modeling, we investigate the potential of autoregressive models for a new modality: structured 3D scenes. To this end, we propose a unified LLM framework that aligns language, images, and 3D scenes and provide a detailed ''cookbook'' outlining critical design choices for achieving optimal training and performance addressing key questions related to data representation, modality-specific objectives, and more. We show how to tokenize complex 3D objects to incorporate into our structured 3D scene modality. We evaluate performance across four core 3D tasks -- rendering, recognition, instruction-following, and question-answering -- and four 3D datasets, synthetic and real-world. We show our model's effectiveness on reconstructing complete 3D scenes consisting of complex objects from a single image and...
  </details>  
- **[UA-Pose: Uncertainty-Aware 6D Object Pose Estimation and Online Object Completion with Partial References](https://arxiv.org/abs/2506.07996v1)**  
  Authors: Ming-Feng Li, Xin Yang, Fu-En Wang, Hritam Basak, Yuyin Sun, Shreekant Gayaka, Min Sun, Cheng-Hao Kuo  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2506.07996v1.pdf) | [![Project](https://img.shields.io/badge/-Project-blue)](https://minfenli.github.io/UA-Pose)  
  <details><summary>Abstract</summary>

  6D object pose estimation has shown strong generalizability to novel objects. However, existing methods often require either a complete, well-reconstructed 3D model or numerous reference images that fully cover the object. Estimating 6D poses from partial references, which capture only fragments of an object's appearance and geometry, remains challenging. To address this, we propose UA-Pose, an uncertainty-aware approach for 6D object pose estimation and online object completion specifically designed for partial references. We assume access to either (1) a limited set of RGBD images with known poses or (2) a single 2D image. For the first case, we initialize a partial object 3D model based on the provided images and poses, while for the second, we use image-to-3D techniques to generate an initial object 3D model. Our method integrates uncertainty into the incomplete 3D model, distinguishing between seen and unseen regions. This uncertainty enables confidence assessment in pose estimation and...
  </details>  
  Keywords: image-to-3d  
- **[Squeeze3D: Your 3D Generation Model is Secretly an Extreme Neural Compressor](https://arxiv.org/abs/2506.07932v1)**  
  Authors: Rishit Dagli, Yushi Guan, Sankeerth Durvasula, Mohammadreza Mofayezi, Nandita Vijaykumar  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2506.07932v1.pdf)  
  <details><summary>Abstract</summary>

  We propose Squeeze3D, a novel framework that leverages implicit prior knowledge learnt by existing pre-trained 3D generative models to compress 3D data at extremely high compression ratios. Our approach bridges the latent spaces between a pre-trained encoder and a pre-trained generation model through trainable mapping networks. Any 3D model represented as a mesh, point cloud, or a radiance field is first encoded by the pre-trained encoder and then transformed (i.e. compressed) into a highly compact latent code. This latent code can effectively be used as an extremely compressed representation of the mesh or point cloud. A mapping network transforms the compressed latent code into the latent space of a powerful generative model, which is then conditioned to recreate the original 3D model (i.e. decompression). Squeeze3D is trained entirely on generated synthetic data and does not require any 3D datasets. The Squeeze3D architecture can be flexibly used with existing pre-trained 3D...
  </details>  
- **[R3D2: Realistic 3D Asset Insertion via Diffusion for Autonomous Driving Simulation](https://arxiv.org/abs/2506.07826v1)**  
  Authors: William Ljungbergh, Bernardo Taveira, Wenzhao Zheng, Adam Tonderski, Chensheng Peng, Fredrik Kahl, Christoffer Petersson, Michael Felsberg, Kurt Keutzer, Masayoshi Tomizuka, Wei Zhan  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2506.07826v1.pdf) | [![Project](https://img.shields.io/badge/-Project-blue)](https://research.zenseact.com/publications/R3D2)  
  <details><summary>Abstract</summary>

  Validating autonomous driving (AD) systems requires diverse and safety-critical testing, making photorealistic virtual environments essential. Traditional simulation platforms, while controllable, are resource-intensive to scale and often suffer from a domain gap with real-world data. In contrast, neural reconstruction methods like 3D Gaussian Splatting (3DGS) offer a scalable solution for creating photorealistic digital twins of real-world driving scenes. However, they struggle with dynamic object manipulation and reusability as their per-scene optimization-based methodology tends to result in incomplete object models with integrated illumination effects. This paper introduces R3D2, a lightweight, one-step diffusion model designed to overcome these limitations and enable realistic insertion of complete 3D assets into existing scenes by generating plausible rendering effects-such as shadows and consistent lighting-in real time. This is achieved by training R3D2 on a novel dataset: 3DGS object assets are generated from in-the-wild AD data using an image-conditioned 3D generative model, and then synthetically placed into neural...
  </details>  
  Keywords: text-to-3d  
- **[NOVA3D: Normal Aligned Video Diffusion Model for Single Image to 3D Generation](https://arxiv.org/abs/2506.07698v1)**  
  Authors: Yuxiao Yang, Peihao Li, Yuhong Zhang, Junzhe Lu, Xianglong He, Minghan Qin, Weitao Wang, Haoqian Wang  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2506.07698v1.pdf)  
  <details><summary>Abstract</summary>

  3D AI-generated content (AIGC) has made it increasingly accessible for anyone to become a 3D content creator. While recent methods leverage Score Distillation Sampling to distill 3D objects from pretrained image diffusion models, they often suffer from inadequate 3D priors, leading to insufficient multi-view consistency. In this work, we introduce NOVA3D, an innovative single-image-to-3D generation framework. Our key insight lies in leveraging strong 3D priors from a pretrained video diffusion model and integrating geometric information during multi-view video fine-tuning. To facilitate information exchange between color and geometric domains, we propose the Geometry-Temporal Alignment (GTA) attention mechanism, thereby improving generalization and multi-view consistency. Moreover, we introduce the de-conflict geometry fusion algorithm, which improves texture fidelity by addressing multi-view inaccuracies and resolving discrepancies in pose alignment. Extensive experiments validate the superiority of NOVA3D over existing baselines.
  </details>  
  Keywords: image-to-3d  
- **[HuSc3D: Human Sculpture dataset for 3D object reconstruction](https://arxiv.org/abs/2506.07628v1)**  
  Authors: Weronika Smolak-Dyżewska, Dawid Malarz, Grzegorz Wilczyński, Rafał Tobiasz, Joanna Waczyńska, Piotr Borycki, Przemysław Spurek  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2506.07628v1.pdf)  
  <details><summary>Abstract</summary>

  3D scene reconstruction from 2D images is one of the most important tasks in computer graphics. Unfortunately, existing datasets and benchmarks concentrate on idealized synthetic or meticulously captured realistic data. Such benchmarks fail to convey the inherent complexities encountered in newly acquired real-world scenes. In such scenes especially those acquired outside, the background is often dynamic, and by popular usage of cell phone cameras, there might be discrepancies in, e.g., white balance. To address this gap, we present HuSc3D, a novel dataset specifically designed for rigorous benchmarking of 3D reconstruction models under realistic acquisition challenges. Our dataset uniquely features six highly detailed, fully white sculptures characterized by intricate perforations and minimal textural and color variation. Furthermore, the number of images per scene varies significantly, introducing the additional challenge of limited training data for some instances alongside scenes with a standard number of views. By evaluating popular 3D reconstruction methods on...
  </details>  
  Keywords: 3d scene reconstruction  
- **[APTOS-2024 challenge report: Generation of synthetic 3D OCT images from fundus photographs](https://arxiv.org/abs/2506.07542v1)**  
  Authors: Bowen Liu, Weiyi Zhang, Peranut Chotcomwongse, Xiaolan Chen, Ruoyu Chen, Pawin Pakaymaskul, Niracha Arjkongharn, Nattaporn Vongsa, Xuelian Cheng, Zongyuan Ge, Kun Huang, Xiaohui Li, Yiru Duan, Zhenbang Wang, BaoYe Xie, Qiang Chen, Huazhu Fu, Michael A. Mahr, Jiaqi Qu, Wangyiyang Chen, Shiye Wang, Yubo Tan, Yongjie Li, Mingguang He, Danli Shi, Paisan Ruamviboonsuk  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2506.07542v1.pdf)  
  <details><summary>Abstract</summary>

  Optical Coherence Tomography (OCT) provides high-resolution, 3D, and non-invasive visualization of retinal layers in vivo, serving as a critical tool for lesion localization and disease diagnosis. However, its widespread adoption is limited by equipment costs and the need for specialized operators. In comparison, 2D color fundus photography offers faster acquisition and greater accessibility with less dependence on expensive devices. Although generative artificial intelligence has demonstrated promising results in medical image synthesis, translating 2D fundus images into 3D OCT images presents unique challenges due to inherent differences in data dimensionality and biological information between modalities. To advance generative models in the fundus-to-3D-OCT setting, the Asia Pacific Tele-Ophthalmology Society (APTOS-2024) organized a challenge titled Artificial Intelligence-based OCT Generation from Fundus Images. This paper details the challenge framework (referred to as APTOS-2024 Challenge), including: the benchmark dataset, evaluation methodology featuring two fidelity metrics-image-based distance (pixel-level OCT B-scan similarity) and video-based distance (semantic-level volumetric...
  </details>  
- **[Hybrid Mesh-Gaussian Representation for Efficient Indoor Scene Reconstruction](https://arxiv.org/abs/2506.06988v1)**  
  Authors: Binxiao Huang, Zhihao Li, Shiyong Liu, Xiao Tang, Jiajun Tang, Jiaqi Lin, Yuxin Cheng, Zhenyu Chen, Xiaofei Wu, Ngai Wong  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2506.06988v1.pdf)  
  <details><summary>Abstract</summary>

  3D Gaussian splatting (3DGS) has demonstrated exceptional performance in image-based 3D reconstruction and real-time rendering. However, regions with complex textures require numerous Gaussians to capture significant color variations accurately, leading to inefficiencies in rendering speed. To address this challenge, we introduce a hybrid representation for indoor scenes that combines 3DGS with textured meshes. Our approach uses textured meshes to handle texture-rich flat areas, while retaining Gaussians to model intricate geometries. The proposed method begins by pruning and refining the extracted mesh to eliminate geometrically complex regions. We then employ a joint optimization for 3DGS and mesh, incorporating a warm-up strategy and transmittance-aware supervision to balance their contributions seamlessly.Extensive experiments demonstrate that the hybrid representation maintains comparable rendering quality and achieves superior frames per second FPS with fewer Gaussian primitives.
  </details>  
- **[SPC to 3D: Novel View Synthesis from Binary SPC via I2I translation](https://arxiv.org/abs/2506.06890v1)**  
  Authors: Sumit Sharma, Gopi Raju Matta, Kaushik Mitra  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2506.06890v1.pdf)  
  <details><summary>Abstract</summary>

  Single Photon Avalanche Diodes (SPADs) represent a cutting-edge imaging technology, capable of detecting individual photons with remarkable timing precision. Building on this sensitivity, Single Photon Cameras (SPCs) enable image capture at exceptionally high speeds under both low and high illumination. Enabling 3D reconstruction and radiance field recovery from such SPC data holds significant promise. However, the binary nature of SPC images leads to severe information loss, particularly in texture and color, making traditional 3D synthesis techniques ineffective. To address this challenge, we propose a modular two-stage framework that converts binary SPC images into high-quality colorized novel views. The first stage performs image-to-image (I2I) translation using generative models such as Pix2PixHD, converting binary SPC inputs into plausible RGB representations. The second stage employs 3D scene reconstruction techniques like Neural Radiance Fields (NeRF) or Gaussian Splatting (3DGS) to generate novel views. We validate our two-stage pipeline (Pix2PixHD + Nerf/3DGS) through extensive qualitative...
  </details>  
  Keywords: novel view synthesis, 3d scene reconstruction  
- **[PartCrafter: Structured 3D Mesh Generation via Compositional Latent Diffusion Transformers](https://arxiv.org/abs/2506.05573v1)**  
  Authors: Yuchen Lin, Chenguo Lin, Panwang Pan, Honglei Yan, Yiqiang Feng, Yadong Mu, Katerina Fragkiadaki  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2506.05573v1.pdf)  
  <details><summary>Abstract</summary>

  We introduce PartCrafter, the first structured 3D generative model that jointly synthesizes multiple semantically meaningful and geometrically distinct 3D meshes from a single RGB image. Unlike existing methods that either produce monolithic 3D shapes or follow two-stage pipelines, i.e., first segmenting an image and then reconstructing each segment, PartCrafter adopts a unified, compositional generation architecture that does not rely on pre-segmented inputs. Conditioned on a single image, it simultaneously denoises multiple 3D parts, enabling end-to-end part-aware generation of both individual objects and complex multi-object scenes. PartCrafter builds upon a pretrained 3D mesh diffusion transformer (DiT) trained on whole objects, inheriting the pretrained weights, encoder, and decoder, and introduces two key innovations: (1) A compositional latent space, where each 3D part is represented by a set of disentangled latent tokens; (2) A hierarchical attention mechanism that enables structured information flow both within individual parts and across all parts, ensuring global coherence...
  </details>  
- **[Does Your 3D Encoder Really Work? When Pretrain-SFT from 2D VLMs Meets 3D VLMs](https://arxiv.org/abs/2506.05318v2)**  
  Authors: Haoyuan Li, Yanpeng Zhou, Yufei Gao, Tao Tang, Jianhua Han, Yujie Yuan, Dave Zhenyu Chen, Jiawang Bian, Hang Xu, Xiaodan Liang  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2506.05318v2.pdf)  
  <details><summary>Abstract</summary>

  Remarkable progress in 2D Vision-Language Models (VLMs) has spurred interest in extending them to 3D settings for tasks like 3D Question Answering, Dense Captioning, and Visual Grounding. Unlike 2D VLMs that typically process images through an image encoder, 3D scenes, with their intricate spatial structures, allow for diverse model architectures. Based on their encoder design, this paper categorizes recent 3D VLMs into 3D object-centric, 2D image-based, and 3D scene-centric approaches. Despite the architectural similarity of 3D scene-centric VLMs to their 2D counterparts, they have exhibited comparatively lower performance compared with the latest 3D object-centric and 2D image-based approaches. To understand this gap, we conduct an in-depth analysis, revealing that 3D scene-centric VLMs show limited reliance on the 3D scene encoder, and the pre-train stage appears less effective than in 2D VLMs. Furthermore, we observe that data scaling benefits are less pronounced on larger datasets. Our investigation suggests that while these...
  </details>  
- **[AI-powered Contextual 3D Environment Generation: A Systematic Review](https://arxiv.org/abs/2506.05449v1)**  
  Authors: Miguel Silva, Alexandre Valle de Carvalho  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2506.05449v1.pdf)  
  <details><summary>Abstract</summary>

  The generation of high-quality 3D environments is crucial for industries such as gaming, virtual reality, and cinema, yet remains resource-intensive due to the reliance on manual processes. This study performs a systematic review of existing generative AI techniques for 3D scene generation, analyzing their characteristics, strengths, limitations, and potential for improvement. By examining state-of-the-art approaches, it presents key challenges such as scene authenticity and the influence of textual inputs. Special attention is given to how AI can blend different stylistic domains while maintaining coherence, the impact of training data on output quality, and the limitations of current models. In addition, this review surveys existing evaluation metrics for assessing realism and explores how industry professionals incorporate AI into their workflows. The findings of this study aim to provide a comprehensive understanding of the current landscape and serve as a foundation for future research on AI-driven 3D content generation. Key findings include...
  </details>  
  Keywords: text-to-3d  
- **[ReSpace: Text-Driven 3D Indoor Scene Synthesis and Editing with Preference Alignment](https://arxiv.org/abs/2506.02459v4)**  
  Authors: Martin JJ. Bucher, Iro Armeni  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2506.02459v4.pdf)  
  <details><summary>Abstract</summary>

  Scene synthesis and editing has emerged as a promising direction in computer graphics. Current trained approaches for 3D indoor scenes either oversimplify object semantics through one-hot class encodings (e.g., 'chair' or 'table'), require masked diffusion for editing, ignore room boundaries, or rely on floor plan renderings that fail to capture complex layouts. LLM-based methods enable richer semantics via natural language (e.g., 'modern studio with light wood furniture'), but lack editing functionality, are limited to rectangular layouts, or rely on weak spatial reasoning from implicit world models. We introduce ReSpace, a generative framework for text-driven 3D indoor scene synthesis and editing using autoregressive language models. Our approach features a compact structured scene representation with explicit room boundaries that enables asset-agnostic deployment and frames scene editing as a next-token prediction task. We leverage a dual-stage training approach combining supervised fine-tuning and preference alignment, enabling a specially trained language model for object addition...
  </details>  
- **[Large Language Models for EEG: A Comprehensive Survey and Taxonomy](https://arxiv.org/abs/2506.06353v1)**  
  Authors: Naseem Babu, Jimson Mathew, A. P. Vinod  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2506.06353v1.pdf)  
  <details><summary>Abstract</summary>

  The growing convergence between Large Language Models (LLMs) and electroencephalography (EEG) research is enabling new directions in neural decoding, brain-computer interfaces (BCIs), and affective computing. This survey offers a systematic review and structured taxonomy of recent advancements that utilize LLMs for EEG-based analysis and applications. We organize the literature into four domains: (1) LLM-inspired foundation models for EEG representation learning, (2) EEG-to-language decoding, (3) cross-modal generation including image and 3D object synthesis, and (4) clinical applications and dataset management tools. The survey highlights how transformer-based architectures adapted through fine-tuning, few-shot, and zero-shot learning have enabled EEG-based models to perform complex tasks such as natural language generation, semantic interpretation, and diagnostic assistance. By offering a structured overview of modeling strategies, system designs, and application areas, this work serves as a foundational resource for future work to bridge natural language processing and neural signal analysis through language models.
  </details>  
- **[ShapeLLM-Omni: A Native Multimodal LLM for 3D Generation and Understanding](https://arxiv.org/abs/2506.01853v1)**  
  Authors: Junliang Ye, Zhengyi Wang, Ruowen Zhao, Shenghao Xie, Jun Zhu  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2506.01853v1.pdf) | [![GitHub](https://img.shields.io/github/stars/JAMESYJL/ShapeLLM-Omni?style=social)](https://github.com/JAMESYJL/ShapeLLM-Omni)  
  <details><summary>Abstract</summary>

  Recently, the powerful text-to-image capabilities of ChatGPT-4o have led to growing appreciation for native multimodal large language models. However, its multimodal capabilities remain confined to images and text. Yet beyond images, the ability to understand and generate 3D content is equally crucial. To address this gap, we propose ShapeLLM-Omni-a native 3D large language model capable of understanding and generating 3D assets and text in any sequence. First, we train a 3D vector-quantized variational autoencoder (VQVAE), which maps 3D objects into a discrete latent space to achieve efficient and accurate shape representation and reconstruction. Building upon the 3D-aware discrete tokens, we innovatively construct a large-scale continuous training dataset named 3D-Alpaca, encompassing generation, comprehension, and editing, thus providing rich resources for future research and training. Finally, by performing instruction-based training of the Qwen-2.5-vl-7B-Instruct model on the 3D-Alpaca dataset. Our work provides an effective attempt at extending multimodal models with basic 3D capabilities,...
  </details>  
- **[RadarSplat: Radar Gaussian Splatting for High-Fidelity Data Synthesis and 3D Reconstruction of Autonomous Driving Scenes](https://arxiv.org/abs/2506.01379v1)**  
  Authors: Pou-Chun Kung, Skanda Harisha, Ram Vasudevan, Aline Eid, Katherine A. Skinner  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2506.01379v1.pdf) | [![Project](https://img.shields.io/badge/-Project-blue)](https://umautobots.github.io/radarsplat)  
  <details><summary>Abstract</summary>

  High-Fidelity 3D scene reconstruction plays a crucial role in autonomous driving by enabling novel data generation from existing datasets. This allows simulating safety-critical scenarios and augmenting training datasets without incurring further data collection costs. While recent advances in radiance fields have demonstrated promising results in 3D reconstruction and sensor data synthesis using cameras and LiDAR, their potential for radar remains largely unexplored. Radar is crucial for autonomous driving due to its robustness in adverse weather conditions like rain, fog, and snow, where optical sensors often struggle. Although the state-of-the-art radar-based neural representation shows promise for 3D driving scene reconstruction, it performs poorly in scenarios with significant radar noise, including receiver saturation and multipath reflection. Moreover, it is limited to synthesizing preprocessed, noise-excluded radar images, failing to address realistic radar data synthesis. To address these limitations, this paper proposes RadarSplat, which integrates Gaussian Splatting with novel radar noise modeling to enable...
  </details>  
  Keywords: 3d scene reconstruction  
- **[Contra4: Evaluating Contrastive Cross-Modal Reasoning in Audio, Video, Image, and 3D](https://arxiv.org/abs/2506.01275v2)**  
  Authors: Artemis Panagopoulou, Le Xue, Honglu Zhou, silvio savarese, Ran Xu, Caiming Xiong, Chris Callison-Burch, Mark Yatskar, Juan Carlos Niebles  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2506.01275v2.pdf)  
  <details><summary>Abstract</summary>

  Real-world decision-making often begins with identifying which modality contains the most relevant information for a given query. While recent multimodal models have made impressive progress in processing diverse inputs, it remains unclear whether they can reason contrastively across multiple modalities to select the one that best satisfies a natural language prompt. We argue this capability is foundational, especially in retrieval-augmented and decision-time contexts, where systems must evaluate multiple signals and identify which one conveys the relevant information. To evaluate this skill, we introduce Contra4, a dataset for contrastive cross-modal reasoning across four modalities: image, audio, video, and 3D. Each example presents a natural language question alongside multiple candidate modality instances, and the model must select the one that semantically aligns with the prompt. Contra4 combines human-annotated captions with a mixture-of-models round-trip-consistency filter to ensure high-quality supervision, resulting in 174k training examples and a manually verified test set of 2.3k samples....
  </details>  

### July 2025
- **[MoGA: 3D Generative Avatar Prior for Monocular Gaussian Avatar Reconstruction](https://arxiv.org/abs/2507.23597v3)**  
  Authors: Zijian Dong, Longteng Duan, Jie Song, Michael J. Black, Andreas Geiger  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2507.23597v3.pdf) | [![Project](https://img.shields.io/badge/-Project-blue)](https://zj-dong.github.io/MoGA)  
  <details><summary>Abstract</summary>

  We present MoGA, a novel method to reconstruct high-fidelity 3D Gaussian avatars from a single-view image. The main challenge lies in inferring unseen appearance and geometric details while ensuring 3D consistency and realism. Most previous methods rely on 2D diffusion models to synthesize unseen views; however, these generated views are sparse and inconsistent, resulting in unrealistic 3D artifacts and blurred appearance. To address these limitations, we leverage a generative avatar model, that can generate diverse 3D avatars by sampling deformed Gaussians from a learned prior distribution. Due to limited 3D training data, such a 3D model alone cannot capture all image details of unseen identities. Consequently, we integrate it as a prior, ensuring 3D consistency by projecting input images into its latent space and enforcing additional 3D appearance and geometric constraints. Our novel approach formulates Gaussian avatar creation as model inversion by fitting the generative avatar to synthetic views from...
  </details>  
- **[BS-1-to-N: Diffusion-Based Environment-Aware Cross-BS Channel Knowledge Map Generation for Cell-Free Networks](https://arxiv.org/abs/2507.23236v1)**  
  Authors: Zhuoyin Dai, Di Wu, Yong Zeng, Xiaoli Xu, Xinyi Wang, Zesong Fei  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2507.23236v1.pdf)  
  <details><summary>Abstract</summary>

  Channel knowledge map (CKM) inference across base stations (BSs) is the key to achieving efficient environmentaware communications. This paper proposes an environmentaware cross-BS CKM inference method called BS-1-to-N based on the generative diffusion model. To this end, we first design the BS location embedding (BSLE) method tailored for cross-BS CKM inference to embed BS location information in the feature vector of CKM. Further, we utilize the cross- and self-attention mechanism for the proposed BS-1-to-N model to respectively learn the relationships between source and target BSs, as well as that among target BSs. Therefore, given the locations of the source and target BSs, together with the source CKMs as control conditions, cross-BS CKM inference can be performed for an arbitrary number of source and target BSs. Specifically, in architectures with massive distributed nodes like cell-free networks, traditional methods of sequentially traversing each BS for CKM construction are prohibitively costly. By contrast,...
  </details>  
  Keywords: multi-view synthesis  
- **[ReXGroundingCT: A 3D Chest CT Dataset for Segmentation of Findings from Free-Text Reports](https://arxiv.org/abs/2507.22030v2)**  
  Authors: Mohammed Baharoon, Luyang Luo, Michael Moritz, Abhinav Kumar, Sung Eun Kim, Xiaoman Zhang, Miao Zhu, Mahmoud Hussain Alabbad, Maha Sbayel Alhazmi, Neel P. Mistry, Lucas Bijnens, Kent Ryan Kleinschmidt, Brady Chrisler, Sathvik Suryadevara, Sri Sai Dinesh Jaliparthi, Noah Michael Prudlo, Mark David Marino, Jeremy Palacio, Rithvik Akula, Di Zhou, Hong-Yu Zhou, Ibrahim Ethem Hamamci, Scott J. Adams, Hassan Rayhan AlOmaish, Pranav Rajpurkar  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2507.22030v2.pdf) | [![Project](https://img.shields.io/badge/-Project-blue)](https://rexrank.ai/ReXGroundingCT) | [![Dataset](https://img.shields.io/badge/-Dataset-orange)](https://huggingface.co/datasets/rajpurkarlab/ReXGroundingCT)  
  <details><summary>Abstract</summary>

  We introduce ReXGroundingCT, the first publicly available dataset linking free-text findings to pixel-level 3D segmentations in chest CT scans. The dataset includes 3,142 non-contrast chest CT scans paired with standardized radiology reports from CT-RATE. Construction followed a structured three-stage pipeline. First, GPT-4 was used to extract and standardize findings, descriptors, and metadata from reports originally written in Turkish and machine-translated into English. Second, GPT-4o-mini categorized each finding into a hierarchical ontology of lung and pleural abnormalities. Third, 3D annotations were produced for all CT volumes: the training set was quality-assured by board-certified radiologists, and the validation and test sets were fully annotated by board-certified radiologists. Additionally, a complementary chain-of-thought dataset was created to provide step-by-step hierarchical anatomical reasoning for localizing findings within the CT volume, using GPT-4o and localization coordinates derived from organ segmentation models. ReXGroundingCT contains 16,301 annotated entities across 8,028 text-to-3D-segmentation pairs, covering diverse radiological patterns from 3,142...
  </details>  
  Keywords: text-to-3d  
- **[BANG: Dividing 3D Assets via Generative Exploded Dynamics](https://arxiv.org/abs/2507.21493v1)**  
  Authors: Longwen Zhang, Qixuan Zhang, Haoran Jiang, Yinuo Bai, Wei Yang, Lan Xu, Jingyi Yu  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2507.21493v1.pdf)  
  <details><summary>Abstract</summary>

  3D creation has always been a unique human strength, driven by our ability to deconstruct and reassemble objects using our eyes, mind and hand. However, current 3D design tools struggle to replicate this natural process, requiring considerable artistic expertise and manual labor. This paper introduces BANG, a novel generative approach that bridges 3D generation and reasoning, allowing for intuitive and flexible part-level decomposition of 3D objects. At the heart of BANG is "Generative Exploded Dynamics", which creates a smooth sequence of exploded states for an input geometry, progressively separating parts while preserving their geometric and semantic coherence. BANG utilizes a pre-trained large-scale latent diffusion model, fine-tuned for exploded dynamics with a lightweight exploded view adapter, allowing precise control over the decomposition process. It also incorporates a temporal attention module to ensure smooth transitions and consistency across time. BANG enhances control with spatial prompts, such as bounding boxes and surface regions,...
  </details>  
- **[Decomposing Densification in Gaussian Splatting for Faster 3D Scene Reconstruction](https://arxiv.org/abs/2507.20239v1)**  
  Authors: Binxiao Huang, Zhengwu Liu, Ngai Wong  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2507.20239v1.pdf)  
  <details><summary>Abstract</summary>

  3D Gaussian Splatting (GS) has emerged as a powerful representation for high-quality scene reconstruction, offering compelling rendering quality. However, the training process of GS often suffers from slow convergence due to inefficient densification and suboptimal spatial distribution of Gaussian primitives. In this work, we present a comprehensive analysis of the split and clone operations during the densification phase, revealing their distinct roles in balancing detail preservation and computational efficiency. Building upon this analysis, we propose a global-to-local densification strategy, which facilitates more efficient growth of Gaussians across the scene space, promoting both global coverage and local refinement. To cooperate with the proposed densification strategy and promote sufficient diffusion of Gaussian primitives in space, we introduce an energy-guided coarse-to-fine multi-resolution training framework, which gradually increases resolution based on energy density in 2D images. Additionally, we dynamically prune unnecessary Gaussian primitives to speed up the training. Extensive experiments on MipNeRF-360, Deep Blending,...
  </details>  
  Keywords: 3d scene reconstruction  
- **[LRR-Bench: Left, Right or Rotate? Vision-Language models Still Struggle With Spatial Understanding Tasks](https://arxiv.org/abs/2507.20174v2)**  
  Authors: Fei Kong  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2507.20174v2.pdf) | [![GitHub](https://img.shields.io/github/stars/kong13661/LRR-Bench?style=social)](https://github.com/kong13661/LRR-Bench)  
  <details><summary>Abstract</summary>

  Real-world applications, such as autonomous driving and humanoid robot manipulation, require precise spatial perception. However, it remains underexplored how Vision-Language Models (VLMs) recognize spatial relationships and perceive spatial movement. In this work, we introduce a spatial evaluation pipeline and construct a corresponding benchmark. Specifically, we categorize spatial understanding into two main types: absolute spatial understanding, which involves querying the absolute spatial position (e.g., left, right) of an object within an image, and 3D spatial understanding, which includes movement and rotation. Notably, our dataset is entirely synthetic, enabling the generation of test samples at a low cost while also preventing dataset contamination. We conduct experiments on multiple state-of-the-art VLMs and observe that there is significant room for improvement in their spatial understanding abilities. Explicitly, in our experiments, humans achieve near-perfect performance on all tasks, whereas current VLMs attain human-level performance only on the two simplest tasks. For the remaining tasks, the...
  </details>  
- **[ScenePainter: Semantically Consistent Perpetual 3D Scene Generation with Concept Relation Alignment](https://arxiv.org/abs/2507.19058v1)**  
  Authors: Chong Xia, Shengjun Zhang, Fangfu Liu, Chang Liu, Khodchaphun Hirunyaratsameewong, Yueqi Duan  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2507.19058v1.pdf) | [![Project](https://img.shields.io/badge/-Project-blue)](https://xiac20.github.io/ScenePainter)  
  <details><summary>Abstract</summary>

  Perpetual 3D scene generation aims to produce long-range and coherent 3D view sequences, which is applicable for long-term video synthesis and 3D scene reconstruction. Existing methods follow a "navigate-and-imagine" fashion and rely on outpainting for successive view expansion. However, the generated view sequences suffer from semantic drift issue derived from the accumulated deviation of the outpainting module. To tackle this challenge, we propose ScenePainter, a new framework for semantically consistent 3D scene generation, which aligns the outpainter's scene-specific prior with the comprehension of the current scene. To be specific, we introduce a hierarchical graph structure dubbed SceneConceptGraph to construct relations among multi-level scene concepts, which directs the outpainter for consistent novel views and can be dynamically refined to enhance diversity. Extensive experiments demonstrate that our framework overcomes the semantic drift issue and generates more consistent and immersive 3D view sequences. Project Page: https://xiac20.github.io/ScenePainter/.
  </details>  
  Keywords: 3d scene reconstruction  
- **[MVG4D: Image Matrix-Based Multi-View and Motion Generation for 4D Content Creation from a Single Image](https://arxiv.org/abs/2507.18371v2)**  
  Authors: DongFu Yin, Xiaotian Chen, Fei Richard Yu, Xuanchen Li, Xinhao Zhang  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2507.18371v2.pdf)  
  <details><summary>Abstract</summary>

  Advances in generative modeling have significantly enhanced digital content creation, extending from 2D images to complex 3D and 4D scenes. Despite substantial progress, producing high-fidelity and temporally consistent dynamic 4D content remains a challenge. In this paper, we propose MVG4D, a novel framework that generates dynamic 4D content from a single still image by combining multi-view synthesis with 4D Gaussian Splatting (4D GS). At its core, MVG4D employs an image matrix module that synthesizes temporally coherent and spatially diverse multi-view images, providing rich supervisory signals for downstream 3D and 4D reconstruction. These multi-view images are used to optimize a 3D Gaussian point cloud, which is further extended into the temporal domain via a lightweight deformation network. Our method effectively enhances temporal consistency, geometric fidelity, and visual realism, addressing key challenges in motion discontinuity and background degradation that affect prior 4D GS-based methods. Extensive experiments on the Objaverse dataset demonstrate that...
  </details>  
  Keywords: multi-view synthesis  
- **[LONG3R: Long Sequence Streaming 3D Reconstruction](https://arxiv.org/abs/2507.18255v1)**  
  Authors: Zhuoguang Chen, Minghui Qin, Tianyuan Yuan, Zhe Liu, Hang Zhao  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2507.18255v1.pdf) | [![Project](https://img.shields.io/badge/-Project-blue)](https://zgchen33.github.io/LONG3R)  
  <details><summary>Abstract</summary>

  Recent advancements in multi-view scene reconstruction have been significant, yet existing methods face limitations when processing streams of input images. These methods either rely on time-consuming offline optimization or are restricted to shorter sequences, hindering their applicability in real-time scenarios. In this work, we propose LONG3R (LOng sequence streaming 3D Reconstruction), a novel model designed for streaming multi-view 3D scene reconstruction over longer sequences. Our model achieves real-time processing by operating recurrently, maintaining and updating memory with each new observation. We first employ a memory gating mechanism to filter relevant memory, which, together with a new observation, is fed into a dual-source refined decoder for coarse-to-fine interaction. To effectively capture long-sequence memory, we propose a 3D spatio-temporal memory that dynamically prunes redundant spatial information while adaptively adjusting resolution along the scene. To enhance our model's performance on long sequences while maintaining training efficiency, we employ a two-stage curriculum training strategy,...
  </details>  
  Keywords: 3d scene reconstruction  
- **[Ultra3D: Efficient and High-Fidelity 3D Generation with Part Attention](https://arxiv.org/abs/2507.17745v3)**  
  Authors: Yiwen Chen, Zhihao Li, Yikai Wang, Hu Zhang, Qin Li, Chi Zhang, Guosheng Lin  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2507.17745v3.pdf)  
  <details><summary>Abstract</summary>

  Recent advances in sparse voxel representations have significantly improved the quality of 3D content generation, enabling high-resolution modeling with fine-grained geometry. However, existing frameworks suffer from severe computational inefficiencies due to the quadratic complexity of attention mechanisms in their two-stage diffusion pipelines. In this work, we propose Ultra3D, an efficient 3D generation framework that significantly accelerates sparse voxel modeling without compromising quality. Our method leverages the compact VecSet representation to efficiently generate a coarse object layout in the first stage, reducing token count and accelerating voxel coordinate prediction. To refine per-voxel latent features in the second stage, we introduce Part Attention, a geometry-aware localized attention mechanism that restricts attention computation within semantically consistent part regions. This design preserves structural continuity while avoiding unnecessary global attention, achieving up to 6.7x speed-up in latent generation. To support this mechanism, we construct a scalable part annotation pipeline that converts raw meshes into part-labeled...
  </details>  
- **[EarthCrafter: Scalable 3D Earth Generation via Dual-Sparse Latent Diffusion](https://arxiv.org/abs/2507.16535v2)**  
  Authors: Shang Liu, Chenjie Cao, Chaohui Yu, Wen Qian, Jing Wang, Fan Wang  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2507.16535v2.pdf) | [![Project](https://img.shields.io/badge/-Project-blue)](https://whiteinblue.github.io/earthcrafter)  
  <details><summary>Abstract</summary>

  Despite the remarkable developments achieved by recent 3D generation works, scaling these methods to geographic extents, such as modeling thousands of square kilometers of Earth's surface, remains an open challenge. We address this through a dual innovation in data infrastructure and model architecture. First, we introduce Aerial-Earth3D, the largest 3D aerial dataset to date, consisting of 50k curated scenes (each measuring 600m x 600m) captured across the U.S. mainland, comprising 45M multi-view Google Earth frames. Each scene provides pose-annotated multi-view images, depth maps, normals, semantic segmentation, and camera poses, with explicit quality control to ensure terrain diversity. Building on this foundation, we propose EarthCrafter, a tailored framework for large-scale 3D Earth generation via sparse-decoupled latent diffusion. Our architecture separates structural and textural generation: 1) Dual sparse 3D-VAEs compress high-resolution geometric voxels and textural 2D Gaussian Splats (2DGS) into compact latent spaces, largely alleviating the costly computation suffering from vast geographic...
  </details>  
- **[Scanning Bot: Efficient Scan Planning using Panoramic Cameras](https://arxiv.org/abs/2507.16175v2)**  
  Authors: Euijeong Lee, Kyung Min Han, Young J. Kim  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2507.16175v2.pdf)  
  <details><summary>Abstract</summary>

  Panoramic RGB-D cameras are known for their ability to produce high quality 3D scene reconstructions. However, operating these cameras involves manually selecting viewpoints and physically transporting the camera, making the generation of a 3D model time consuming and tedious. Additionally, the process can be challenging for novice users due to spatial constraints, such as ensuring sufficient feature overlap between viewpoint frames. To address these challenges, we propose a fully autonomous scan planning that generates an efficient tour plan for environment scanning, ensuring collision-free navigation and adequate overlap between viewpoints within the plan. Extensive experiments conducted in both synthetic and real-world environments validate the performance of our planner against state-of-the-art view planners. In particular, our method achieved an average scan coverage of 99 percent in the real-world experiment, with our approach being up to 3 times faster than state-of-the-art planners in total scan time.
  </details>  
  Keywords: 3d scene reconstruction  
- **[ObjectGS: Object-aware Scene Reconstruction and Scene Understanding via Gaussian Splatting](https://arxiv.org/abs/2507.15454v1)**  
  Authors: Ruijie Zhu, Mulin Yu, Linning Xu, Lihan Jiang, Yixuan Li, Tianzhu Zhang, Jiangmiao Pang, Bo Dai  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2507.15454v1.pdf) | [![Project](https://img.shields.io/badge/-Project-blue)](https://ruijiezhu94.github.io/ObjectGS_page)  
  <details><summary>Abstract</summary>

  3D Gaussian Splatting is renowned for its high-fidelity reconstructions and real-time novel view synthesis, yet its lack of semantic understanding limits object-level perception. In this work, we propose ObjectGS, an object-aware framework that unifies 3D scene reconstruction with semantic understanding. Instead of treating the scene as a unified whole, ObjectGS models individual objects as local anchors that generate neural Gaussians and share object IDs, enabling precise object-level reconstruction. During training, we dynamically grow or prune these anchors and optimize their features, while a one-hot ID encoding with a classification loss enforces clear semantic constraints. We show through extensive experiments that ObjectGS not only outperforms state-of-the-art methods on open-vocabulary and panoptic segmentation tasks, but also integrates seamlessly with applications like mesh extraction and scene editing. Project page: https://ruijiezhu94.github.io/ObjectGS_page
  </details>  
  Keywords: novel view synthesis, 3d scene reconstruction  
- **[BenchDepth: Are We on the Right Way to Evaluate Depth Foundation Models?](https://arxiv.org/abs/2507.15321v1)**  
  Authors: Zhenyu Li, Haotong Lin, Jiashi Feng, Peter Wonka, Bingyi Kang  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2507.15321v1.pdf)  
  <details><summary>Abstract</summary>

  Depth estimation is a fundamental task in computer vision with diverse applications. Recent advancements in deep learning have led to powerful depth foundation models (DFMs), yet their evaluation remains challenging due to inconsistencies in existing protocols. Traditional benchmarks rely on alignment-based metrics that introduce biases, favor certain depth representations, and complicate fair comparisons. In this work, we propose BenchDepth, a new benchmark that evaluates DFMs through five carefully selected downstream proxy tasks: depth completion, stereo matching, monocular feed-forward 3D scene reconstruction, SLAM, and vision-language spatial understanding. Unlike conventional evaluation protocols, our approach assesses DFMs based on their practical utility in real-world applications, bypassing problematic alignment procedures. We benchmark eight state-of-the-art DFMs and provide an in-depth analysis of key findings and observations. We hope our work sparks further discussion in the community on best practices for depth model evaluation and paves the way for future research and advancements in depth estimation.
  </details>  
  Keywords: 3d scene reconstruction  
- **[Stereo-GS: Multi-View Stereo Vision Model for Generalizable 3D Gaussian Splatting Reconstruction](https://arxiv.org/abs/2507.14921v2)**  
  Authors: Xiufeng Huang, Ka Chun Cheung, Runmin Cong, Simon See, Renjie Wan  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2507.14921v2.pdf)  
  <details><summary>Abstract</summary>

  Generalizable 3D Gaussian Splatting reconstruction showcases advanced Image-to-3D content creation but requires substantial computational resources and large datasets, posing challenges to training models from scratch. Current methods usually entangle the prediction of 3D Gaussian geometry and appearance, which rely heavily on data-driven priors and result in slow regression speeds. To address this, we propose \method, a disentangled framework for efficient 3D Gaussian prediction. Our method extracts features from local image pairs using a stereo vision backbone and fuses them via global attention blocks. Dedicated point and Gaussian prediction heads generate multi-view point-maps for geometry and Gaussian features for appearance, combined as GS-maps to represent the 3DGS object. A refinement network enhances these GS-maps for high-quality reconstruction. Unlike existing methods that depend on camera parameters, our approach achieves pose-free 3D reconstruction, improving robustness and practicality. By reducing resource demands while maintaining high-quality outputs, \method provides an efficient, scalable solution for real-world...
  </details>  
  Keywords: image-to-3d  
- **[TriCLIP-3D: A Unified Parameter-Efficient Framework for Tri-Modal 3D Visual Grounding based on CLIP](https://arxiv.org/abs/2507.14904v2)**  
  Authors: Fan Li, Zanyi Wang, Zeyi Huang, Guang Dai, Jingdong Wang, Mengmeng Wang  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2507.14904v2.pdf)  
  <details><summary>Abstract</summary>

  3D visual grounding allows an embodied agent to understand visual information in real-world 3D environments based on human instructions, which is crucial for embodied intelligence. Existing 3D visual grounding methods typically rely on separate encoders for different modalities (e.g., RGB images, text, and 3D point clouds), resulting in large and complex models that are inefficient to train. While some approaches use pre-trained 2D multi-modal models like CLIP for 3D tasks, they still struggle with aligning point cloud data to 2D encoders. As a result, these methods continue to depend on 3D encoders for feature extraction, further increasing model complexity and training inefficiency. In this paper, we propose a unified 2D pre-trained multi-modal network to process all three modalities (RGB images, text, and point clouds), significantly simplifying the architecture. By leveraging a 2D CLIP bi-modal model with adapter-based fine-tuning, this framework effectively adapts to the tri-modal setting, improving both adaptability and...
  </details>  
- **[Towards Geometric and Textural Consistency 3D Scene Generation via Single Image-guided Model Generation and Layout Optimization](https://arxiv.org/abs/2507.14841v1)**  
  Authors: Xiang Tang, Ruotong Li, Xiaopeng Fan  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2507.14841v1.pdf)  
  <details><summary>Abstract</summary>

  In recent years, 3D generation has made great strides in both academia and industry. However, generating 3D scenes from a single RGB image remains a significant challenge, as current approaches often struggle to ensure both object generation quality and scene coherence in multi-object scenarios. To overcome these limitations, we propose a novel three-stage framework for 3D scene generation with explicit geometric representations and high-quality textural details via single image-guided model generation and spatial layout optimization. Our method begins with an image instance segmentation and inpainting phase, which recovers missing details of occluded objects in the input images, thereby achieving complete generation of foreground 3D assets. Subsequently, our approach captures the spatial geometry of reference image by constructing pseudo-stereo viewpoint for camera parameter estimation and scene depth inference, while employing a model selection strategy to ensure optimal alignment between the 3D assets generated in the previous step and the input. Finally,...
  </details>  
- **[DreamScene: 3D Gaussian-based End-to-end Text-to-3D Scene Generation](https://arxiv.org/abs/2507.13985v2)**  
  Authors: Haoran Li, Yuli Tian, Kun Lan, Yong Liao, Lin Wang, Pan Hui, Peng Yuan Zhou  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2507.13985v2.pdf) | [![Project](https://img.shields.io/badge/-Project-blue)](https://jahnsonblack.github.io/DreamScene-Full)  
  <details><summary>Abstract</summary>

  Generating 3D scenes from natural language holds great promise for applications in gaming, film, and design. However, existing methods struggle with automation, 3D consistency, and fine-grained control. We present DreamScene, an end-to-end framework for high-quality and editable 3D scene generation from text or dialogue. DreamScene begins with a scene planning module, where a GPT-4 agent infers object semantics and spatial constraints to construct a hybrid graph. A graph-based placement algorithm then produces a structured, collision-free layout. Based on this layout, Formation Pattern Sampling (FPS) generates object geometry using multi-timestep sampling and reconstructive optimization, enabling fast and realistic synthesis. To ensure global consistent, DreamScene employs a progressive camera sampling strategy tailored to both indoor and outdoor settings. Finally, the system supports fine-grained scene editing, including object movement, appearance changes, and 4D dynamic motion. Experiments demonstrate that DreamScene surpasses prior methods in quality, consistency, and flexibility, offering a practical solution for open-domain...
  </details>  
  Keywords: text-to-3d  
- **[AutoPartGen: Autogressive 3D Part Generation and Discovery](https://arxiv.org/abs/2507.13346v2)**  
  Authors: Minghao Chen, Jianyuan Wang, Roman Shapovalov, Tom Monnier, Hyunyoung Jung, Dilin Wang, Rakesh Ranjan, Iro Laina, Andrea Vedaldi  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2507.13346v2.pdf)  
  <details><summary>Abstract</summary>

  We introduce AutoPartGen, a model that generates objects composed of 3D parts in an autoregressive manner. This model can take as input an image of an object, 2D masks of the object's parts, or an existing 3D object, and generate a corresponding compositional 3D reconstruction. Our approach builds upon 3DShape2VecSet, a recent latent 3D representation with powerful geometric expressiveness. We observe that this latent space exhibits strong compositional properties, making it particularly well-suited for part-based generation tasks. Specifically, AutoPartGen generates object parts autoregressively, predicting one part at a time while conditioning on previously generated parts and additional inputs, such as 2D images, masks, or 3D objects. This process continues until the model decides that all parts have been generated, thus determining automatically the type and number of parts. The resulting parts can be seamlessly assembled into coherent objects or scenes without requiring additional optimization. We evaluate both the overall 3D...
  </details>  
- **[MUPAX: Multidimensional Problem Agnostic eXplainable AI](https://arxiv.org/abs/2507.13090v1)**  
  Authors: Vincenzo Dentamaro, Felice Franchini, Giuseppe Pirlo, Irina Voiculescu  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2507.13090v1.pdf)  
  <details><summary>Abstract</summary>

  Robust XAI techniques should ideally be simultaneously deterministic, model agnostic, and guaranteed to converge. We propose MULTIDIMENSIONAL PROBLEM AGNOSTIC EXPLAINABLE AI (MUPAX), a deterministic, model agnostic explainability technique, with guaranteed convergency. MUPAX measure theoretic formulation gives principled feature importance attribution through structured perturbation analysis that discovers inherent input patterns and eliminates spurious relationships. We evaluate MUPAX on an extensive range of data modalities and tasks: audio classification (1D), image classification (2D), volumetric medical image analysis (3D), and anatomical landmark detection, demonstrating dimension agnostic effectiveness. The rigorous convergence guarantees extend to any loss function and arbitrary dimensions, making MUPAX applicable to virtually any problem context for AI. By contrast with other XAI methods that typically decrease performance when masking, MUPAX not only preserves but actually enhances model accuracy by capturing only the most important patterns of the original data. Extensive benchmarking against the state of the XAI art demonstrates MUPAX ability...
  </details>  
- **[Argus: Leveraging Multiview Images for Improved 3-D Scene Understanding With Large Language Models](https://arxiv.org/abs/2507.12916v1)**  
  Authors: Yifan Xu, Chao Zhang, Hanqi Jiang, Xiaoyan Wang, Ruifei Ma, Yiwei Li, Zihao Wu, Zeju Li, Xiangde Liu  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2507.12916v1.pdf)  
  <details><summary>Abstract</summary>

  Advancements in foundation models have made it possible to conduct applications in various downstream tasks. Especially, the new era has witnessed a remarkable capability to extend Large Language Models (LLMs) for tackling tasks of 3D scene understanding. Current methods rely heavily on 3D point clouds, but the 3D point cloud reconstruction of an indoor scene often results in information loss. Some textureless planes or repetitive patterns are prone to omission and manifest as voids within the reconstructed 3D point clouds. Besides, objects with complex structures tend to introduce distortion of details caused by misalignments between the captured images and the dense reconstructed point clouds. 2D multi-view images present visual consistency with 3D point clouds and provide more detailed representations of scene components, which can naturally compensate for these deficiencies. Based on these insights, we propose Argus, a novel 3D multimodal framework that leverages multi-view images for enhanced 3D scene understanding...
  </details>  
- **[PhysX-3D: Physical-Grounded 3D Asset Generation](https://arxiv.org/abs/2507.12465v4)**  
  Authors: Ziang Cao, Zhaoxi Chen, Liang Pan, Ziwei Liu  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2507.12465v4.pdf)  
  <details><summary>Abstract</summary>

  3D modeling is moving from virtual to physical. Existing 3D generation primarily emphasizes geometries and textures while neglecting physical-grounded modeling. Consequently, despite the rapid development of 3D generative models, the synthesized 3D assets often overlook rich and important physical properties, hampering their real-world application in physical domains like simulation and embodied AI. As an initial attempt to address this challenge, we propose \textbf{PhysX-3D}, an end-to-end paradigm for physical-grounded 3D asset generation. 1) To bridge the critical gap in physics-annotated 3D datasets, we present PhysXNet - the first physics-grounded 3D dataset systematically annotated across five foundational dimensions: absolute scale, material, affordance, kinematics, and function description. In particular, we devise a scalable human-in-the-loop annotation pipeline based on vision-language models, which enables efficient creation of physics-first assets from raw 3D assets.2) Furthermore, we propose \textbf{PhysXGen}, a feed-forward framework for physics-grounded image-to-3D asset generation, injecting physical knowledge into the pre-trained 3D structural space. Specifically,...
  </details>  
  Keywords: 3d asset generation, image-to-3d  
- **[SmokeSVD: Smoke Reconstruction from A Single View via Progressive Novel View Synthesis and Refinement with Diffusion Models](https://arxiv.org/abs/2507.12156v1)**  
  Authors: Chen Li, Shanshan Dong, Sheng Qiu, Jianmin Han, Zan Gao, Kemeng Huang, Taku Komura  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2507.12156v1.pdf)  
  <details><summary>Abstract</summary>

  Reconstructing dynamic fluids from sparse views is a long-standing and challenging problem, due to the severe lack of 3D information from insufficient view coverage. While several pioneering approaches have attempted to address this issue using differentiable rendering or novel view synthesis, they are often limited by time-consuming optimization and refinement processes under ill-posed conditions. To tackle above challenges, we propose SmokeSVD, an efficient and effective framework to progressively generate and reconstruct dynamic smoke from a single video by integrating both the powerful generative capabilities from diffusion models and physically guided consistency optimization towards realistic appearance and dynamic evolution. Specifically, we first propose a physically guided side-view synthesizer based on diffusion models, which explicitly incorporates divergence and gradient guidance of velocity fields to generate visually realistic and spatio-temporally consistent side-view images frame by frame, significantly alleviating the ill-posedness of single-view reconstruction without imposing additional constraints. Subsequently, we determine a rough estimation...
  </details>  
  Keywords: novel view synthesis, single-view reconstruction  
- **[SGLoc: Semantic Localization System for Camera Pose Estimation from 3D Gaussian Splatting Representation](https://arxiv.org/abs/2507.12027v1)**  
  Authors: Beining Xu, Siting Zhu, Hesheng Wang  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2507.12027v1.pdf) | [![GitHub](https://img.shields.io/github/stars/IRMVLab/SGLoc?style=social)](https://github.com/IRMVLab/SGLoc)  
  <details><summary>Abstract</summary>

  We propose SGLoc, a novel localization system that directly regresses camera poses from 3D Gaussian Splatting (3DGS) representation by leveraging semantic information. Our method utilizes the semantic relationship between 2D image and 3D scene representation to estimate the 6DoF pose without prior pose information. In this system, we introduce a multi-level pose regression strategy that progressively estimates and refines the pose of query image from the global 3DGS map, without requiring initial pose priors. Moreover, we introduce a semantic-based global retrieval algorithm that establishes correspondences between 2D (image) and 3D (3DGS map). By matching the extracted scene semantic descriptors of 2D query image and 3DGS semantic representation, we align the image with the local region of the global 3DGS map, thereby obtaining a coarse pose estimation. Subsequently, we refine the coarse pose by iteratively optimizing the difference between the query image and the rendered image from 3DGS. Our SGLoc demonstrates...
  </details>  
- **[Wavelet-GS: 3D Gaussian Splatting with Wavelet Decomposition](https://arxiv.org/abs/2507.12498v2)**  
  Authors: Beizhen Zhao, Yifan Zhou, Sicheng Yu, Zijian Wang, Hao Wang  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2507.12498v2.pdf)  
  <details><summary>Abstract</summary>

  3D Gaussian Splatting (3DGS) has revolutionized 3D scene reconstruction, which effectively balances rendering quality, efficiency, and speed. However, existing 3DGS approaches usually generate plausible outputs and face significant challenges in complex scene reconstruction, manifesting as incomplete holistic structural outlines and unclear local lighting effects. To address these issues simultaneously, we propose a novel decoupled optimization framework, which integrates wavelet decomposition into 3D Gaussian Splatting and 2D sampling. Technically, through 3D wavelet decomposition, our approach divides point clouds into high-frequency and low-frequency components, enabling targeted optimization for each. The low-frequency component captures global structural outlines and manages the distribution of Gaussians through voxelization. In contrast, the high-frequency component restores intricate geometric and textural details while incorporating a relight module to mitigate lighting artifacts and enhance photorealistic rendering. Additionally, a 2D wavelet decomposition is applied to the training images, simulating radiance variations. This provides critical guidance for high-frequency detail reconstruction, ensuring seamless...
  </details>  
  Keywords: 3d scene reconstruction  
- **[Physically Based Neural LiDAR Resimulation](https://arxiv.org/abs/2507.12489v1)**  
  Authors: Richard Marcus, Marc Stamminger  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2507.12489v1.pdf) | [![GitHub](https://img.shields.io/github/stars/richardmarcus/PBNLiDAR?style=social)](https://github.com/richardmarcus/PBNLiDAR)  
  <details><summary>Abstract</summary>

  Methods for Novel View Synthesis (NVS) have recently found traction in the field of LiDAR simulation and large-scale 3D scene reconstruction. While solutions for faster rendering or handling dynamic scenes have been proposed, LiDAR specific effects remain insufficiently addressed. By explicitly modeling sensor characteristics such as rolling shutter, laser power variations, and intensity falloff, our method achieves more accurate LiDAR simulation compared to existing techniques. We demonstrate the effectiveness of our approach through quantitative and qualitative comparisons with state-of-the-art methods, as well as ablation studies that highlight the importance of each sensor model component. Beyond that, we show that our approach exhibits advanced resimulation capabilities, such as generating high resolution LiDAR scans in the camera perspective.   Our code and the resulting dataset are available at https://github.com/richardmarcus/PBNLiDAR.
  </details>  
  Keywords: novel view synthesis, 3d scene reconstruction  
- **[4D-Animal: Freely Reconstructing Animatable 3D Animals from Videos](https://arxiv.org/abs/2507.10437v1)**  
  Authors: Shanshan Zhong, Jiawei Peng, Zehan Zheng, Zhongzhan Huang, Wufei Ma, Guofeng Zhang, Qihao Liu, Alan Yuille, Jieneng Chen  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2507.10437v1.pdf) | [![GitHub](https://img.shields.io/github/stars/zhongshsh/4D-Animal?style=social)](https://github.com/zhongshsh/4D-Animal)  
  <details><summary>Abstract</summary>

  Existing methods for reconstructing animatable 3D animals from videos typically rely on sparse semantic keypoints to fit parametric models. However, obtaining such keypoints is labor-intensive, and keypoint detectors trained on limited animal data are often unreliable. To address this, we propose 4D-Animal, a novel framework that reconstructs animatable 3D animals from videos without requiring sparse keypoint annotations. Our approach introduces a dense feature network that maps 2D representations to SMAL parameters, enhancing both the efficiency and stability of the fitting process. Furthermore, we develop a hierarchical alignment strategy that integrates silhouette, part-level, pixel-level, and temporal cues from pre-trained 2D visual models to produce accurate and temporally coherent reconstructions across frames. Extensive experiments demonstrate that 4D-Animal outperforms both model-based and model-free baselines. Moreover, the high-quality 3D assets generated by our method can benefit other 3D tasks, underscoring its potential for large-scale applications. The code is released at https://github.com/zhongshsh/4D-Animal.
  </details>  
- **[Advancing Text-to-3D Generation with Linearized Lookahead Variational Score Distillation](https://arxiv.org/abs/2507.09748v1)**  
  Authors: Yu Lei, Bingde Liu, Qingsong Xie, Haonan Lu, Zhijie Deng  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2507.09748v1.pdf)  
  <details><summary>Abstract</summary>

  Text-to-3D generation based on score distillation of pre-trained 2D diffusion models has gained increasing interest, with variational score distillation (VSD) as a remarkable example. VSD proves that vanilla score distillation can be improved by introducing an extra score-based model, which characterizes the distribution of images rendered from 3D models, to correct the distillation gradient. Despite the theoretical foundations, VSD, in practice, is likely to suffer from slow and sometimes ill-posed convergence. In this paper, we perform an in-depth investigation of the interplay between the introduced score model and the 3D model, and find that there exists a mismatching problem between LoRA and 3D distributions in practical implementation. We can simply adjust their optimization order to improve the generation quality. By doing so, the score model looks ahead to the current 3D state and hence yields more reasonable corrections. Nevertheless, naive lookahead VSD may suffer from unstable training in practice due...
  </details>  
  Keywords: text-to-3d  
- **[Stable Score Distillation](https://arxiv.org/abs/2507.09168v1)**  
  Authors: Haiming Zhu, Yangyang Xu, Chenshu Xu, Tingrui Shen, Wenxi Liu, Yong Du, Jun Yu, Shengfeng He  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2507.09168v1.pdf)  
  <details><summary>Abstract</summary>

  Text-guided image and 3D editing have advanced with diffusion-based models, yet methods like Delta Denoising Score often struggle with stability, spatial control, and editing strength. These limitations stem from reliance on complex auxiliary structures, which introduce conflicting optimization signals and restrict precise, localized edits. We introduce Stable Score Distillation (SSD), a streamlined framework that enhances stability and alignment in the editing process by anchoring a single classifier to the source prompt. Specifically, SSD utilizes Classifier-Free Guidance (CFG) equation to achieves cross-prompt alignment, and introduces a constant term null-text branch to stabilize the optimization process. This approach preserves the original content's structure and ensures that editing trajectories are closely aligned with the source prompt, enabling smooth, prompt-specific modifications while maintaining coherence in surrounding regions. Additionally, SSD incorporates a prompt enhancement branch to boost editing strength, particularly for style transformations. Our method achieves state-of-the-art results in 2D and 3D editing tasks, including...
  </details>  
- **[From One to More: Contextual Part Latents for 3D Generation](https://arxiv.org/abs/2507.08772v2)**  
  Authors: Shaocong Dong, Lihe Ding, Xiao Chen, Yaokun Li, Yuxin Wang, Yucheng Wang, Qi Wang, Jaehyeok Kim, Chenjian Gao, Zhanpeng Huang, Zibin Wang, Tianfan Xue, Dan Xu  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2507.08772v2.pdf)  
  <details><summary>Abstract</summary>

  Recent advances in 3D generation have transitioned from multi-view 2D rendering approaches to 3D-native latent diffusion frameworks that exploit geometric priors in ground truth data. Despite progress, three key limitations persist: (1) Single-latent representations fail to capture complex multi-part geometries, causing detail degradation; (2) Holistic latent coding neglects part independence and interrelationships critical for compositional design; (3) Global conditioning mechanisms lack fine-grained controllability. Inspired by human 3D design workflows, we propose CoPart - a part-aware diffusion framework that decomposes 3D objects into contextual part latents for coherent multi-part generation. This paradigm offers three advantages: i) Reduces encoding complexity through part decomposition; ii) Enables explicit part relationship modeling; iii) Supports part-level conditioning. We further develop a mutual guidance strategy to fine-tune pre-trained diffusion models for joint part latent denoising, ensuring both geometric coherence and foundation model priors. To enable large-scale training, we construct Partverse - a novel 3D part dataset derived...
  </details>  
  Keywords: articulated object generation  
- **[Learning human-to-robot handovers through 3D scene reconstruction](https://arxiv.org/abs/2507.08726v1)**  
  Authors: Yuekun Wu, Yik Lung Pang, Andrea Cavallaro, Changjae Oh  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2507.08726v1.pdf)  
  <details><summary>Abstract</summary>

  Learning robot manipulation policies from raw, real-world image data requires a large number of robot-action trials in the physical environment. Although training using simulations offers a cost-effective alternative, the visual domain gap between simulation and robot workspace remains a major limitation. Gaussian Splatting visual reconstruction methods have recently provided new directions for robot manipulation by generating realistic environments. In this paper, we propose the first method for learning supervised-based robot handovers solely from RGB images without the need of real-robot training or real-robot data collection. The proposed policy learner, Human-to-Robot Handover using Sparse-View Gaussian Splatting (H2RH-SGS), leverages sparse-view Gaussian Splatting reconstruction of human-to-robot handover scenes to generate robot demonstrations containing image-action pairs captured with a camera mounted on the robot gripper. As a result, the simulated camera pose changes in the reconstructed scene can be directly translated into gripper pose changes. We train a robot policy on demonstrations collected with...
  </details>  
  Keywords: 3d scene reconstruction  
- **[InstaScene: Towards Complete 3D Instance Decomposition and Reconstruction from Cluttered Scenes](https://arxiv.org/abs/2507.08416v2)**  
  Authors: Zesong Yang, Bangbang Yang, Wenqi Dong, Chenxuan Cao, Liyuan Cui, Yuewen Ma, Zhaopeng Cui, Hujun Bao  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2507.08416v2.pdf)  
  <details><summary>Abstract</summary>

  Humans can naturally identify and mentally complete occluded objects in cluttered environments. However, imparting similar cognitive ability to robotics remains challenging even with advanced reconstruction techniques, which models scenes as undifferentiated wholes and fails to recognize complete object from partial observations. In this paper, we propose InstaScene, a new paradigm towards holistic 3D perception of complex scenes with a primary goal: decomposing arbitrary instances while ensuring complete reconstruction. To achieve precise decomposition, we develop a novel spatial contrastive learning by tracing rasterization of each instance across views, significantly enhancing semantic supervision in cluttered scenes. To overcome incompleteness from limited observations, we introduce in-situ generation that harnesses valuable observations and geometric cues, effectively guiding 3D generative models to reconstruct complete instances that seamlessly align with the real world. Experiments on scene decomposition and object completion across complex real-world and synthetic scenes demonstrate that our method achieves superior decomposition accuracy while producing...
  </details>  
- **[DiffSpectra: Molecular Structure Elucidation from Spectra using Diffusion Models](https://arxiv.org/abs/2507.06853v2)**  
  Authors: Liang Wang, Yu Rong, Tingyang Xu, Zhenyi Zhong, Zhiyuan Liu, Pengju Wang, Deli Zhao, Qiang Liu, Shu Wu, Liang Wang, Yang Zhang  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2507.06853v2.pdf)  
  <details><summary>Abstract</summary>

  Molecular structure elucidation from spectra is a fundamental challenge in molecular science. Conventional approaches rely heavily on expert interpretation and lack scalability, while retrieval-based machine learning approaches remain constrained by limited reference libraries. Generative models offer a promising alternative, yet most adopt autoregressive architectures that overlook 3D geometry and struggle to integrate diverse spectral modalities. In this work, we present DiffSpectra, a generative framework that formulates molecular structure elucidation as a conditional generation process, directly inferring 2D and 3D molecular structures from multi-modal spectra using diffusion models. Its denoising network is parameterized by the Diffusion Molecule Transformer, an SE(3)-equivariant architecture for geometric modeling, conditioned by SpecFormer, a Transformer-based spectral encoder capturing multi-modal spectral dependencies. Extensive experiments demonstrate that DiffSpectra accurately elucidates molecular structures, achieving 40.76% top-1 and 99.49% top-10 accuracy. Its performance benefits substantially from 3D geometric modeling, SpecFormer pre-training, and multi-modal conditioning. To our knowledge, DiffSpectra is the first...
  </details>  
- **[OmniPart: Part-Aware 3D Generation with Semantic Decoupling and Structural Cohesion](https://arxiv.org/abs/2507.06165v1)**  
  Authors: Yunhan Yang, Yufan Zhou, Yuan-Chen Guo, Zi-Xin Zou, Yukun Huang, Ying-Tian Liu, Hao Xu, Ding Liang, Yan-Pei Cao, Xihui Liu  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2507.06165v1.pdf)  
  <details><summary>Abstract</summary>

  The creation of 3D assets with explicit, editable part structures is crucial for advancing interactive applications, yet most generative methods produce only monolithic shapes, limiting their utility. We introduce OmniPart, a novel framework for part-aware 3D object generation designed to achieve high semantic decoupling among components while maintaining robust structural cohesion. OmniPart uniquely decouples this complex task into two synergistic stages: (1) an autoregressive structure planning module generates a controllable, variable-length sequence of 3D part bounding boxes, critically guided by flexible 2D part masks that allow for intuitive control over part decomposition without requiring direct correspondences or semantic labels; and (2) a spatially-conditioned rectified flow model, efficiently adapted from a pre-trained holistic 3D generator, synthesizes all 3D parts simultaneously and consistently within the planned layout. Our approach supports user-defined part granularity, precise localization, and enables diverse downstream applications. Extensive experiments demonstrate that OmniPart achieves state-of-the-art performance, paving the way for...
  </details>  
  Keywords: 3d generator  
- **[DreamArt: Generating Interactable Articulated Objects from a Single Image](https://arxiv.org/abs/2507.05763v1)**  
  Authors: Ruijie Lu, Yu Liu, Jiaxiang Tang, Junfeng Ni, Yuxiang Wang, Diwen Wan, Gang Zeng, Yixin Chen, Siyuan Huang  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2507.05763v1.pdf) | [![Project](https://img.shields.io/badge/-Project-blue)](https://dream-art-0.github.io/DreamArt)  
  <details><summary>Abstract</summary>

  Generating articulated objects, such as laptops and microwaves, is a crucial yet challenging task with extensive applications in Embodied AI and AR/VR. Current image-to-3D methods primarily focus on surface geometry and texture, neglecting part decomposition and articulation modeling. Meanwhile, neural reconstruction approaches (e.g., NeRF or Gaussian Splatting) rely on dense multi-view or interaction data, limiting their scalability. In this paper, we introduce DreamArt, a novel framework for generating high-fidelity, interactable articulated assets from single-view images. DreamArt employs a three-stage pipeline: firstly, it reconstructs part-segmented and complete 3D object meshes through a combination of image-to-3D generation, mask-prompted 3D segmentation, and part amodal completion. Second, we fine-tune a video diffusion model to capture part-level articulation priors, leveraging movable part masks as prompt and amodal images to mitigate ambiguities caused by occlusion. Finally, DreamArt optimizes the articulation motion, represented by a dual quaternion, and conducts global texture refinement and repainting to ensure coherent,...
  </details>  
  Keywords: image-to-3d  
- **[Acquiring and Adapting Priors for Novel Tasks via Neural Meta-Architectures](https://arxiv.org/abs/2507.10446v2)**  
  Authors: Sudarshan Babu  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2507.10446v2.pdf)  
  <details><summary>Abstract</summary>

  The ability to transfer knowledge from prior experiences to novel tasks stands as a pivotal capability of intelligent agents, including both humans and computational models. This principle forms the basis of transfer learning, where large pre-trained neural networks are fine-tuned to adapt to downstream tasks. Transfer learning has demonstrated tremendous success, both in terms of task adaptation speed and performance. However there are several domains where, due to lack of data, training such large pre-trained models or foundational models is not a possibility - computational chemistry, computational immunology, and medical imaging are examples. To address these challenges, our work focuses on designing architectures to enable efficient acquisition of priors when large amounts of data are unavailable. In particular, we demonstrate that we can use neural memory to enable adaptation on non-stationary distributions with only a few samples. Then we demonstrate that our hypernetwork designs (a network that generates another network)...
  </details>  
  Keywords: text-to-3d  
- **[SegmentDreamer: Towards High-fidelity Text-to-3D Synthesis with Segmented Consistency Trajectory Distillation](https://arxiv.org/abs/2507.05256v2)**  
  Authors: Jiahao Zhu, Zixuan Chen, Guangcong Wang, Xiaohua Xie, Yi Zhou  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2507.05256v2.pdf)  
  <details><summary>Abstract</summary>

  Recent advancements in text-to-3D generation improve the visual quality of Score Distillation Sampling (SDS) and its variants by directly connecting Consistency Distillation (CD) to score distillation. However, due to the imbalance between self-consistency and cross-consistency, these CD-based methods inherently suffer from improper conditional guidance, leading to sub-optimal generation results. To address this issue, we present SegmentDreamer, a novel framework designed to fully unleash the potential of consistency models for high-fidelity text-to-3D generation. Specifically, we reformulate SDS through the proposed Segmented Consistency Trajectory Distillation (SCTD), effectively mitigating the imbalance issues by explicitly defining the relationship between self- and cross-consistency. Moreover, SCTD partitions the Probability Flow Ordinary Differential Equation (PF-ODE) trajectory into multiple sub-trajectories and ensures consistency within each segment, which can theoretically provide a significantly tighter upper bound on distillation error. Additionally, we propose a distillation pipeline for a more swift and stable generation. Extensive experiments demonstrate that our SegmentDreamer outperforms...
  </details>  
  Keywords: text-to-3d  
- **[SeqTex: Generate Mesh Textures in Video Sequence](https://arxiv.org/abs/2507.04285v1)**  
  Authors: Ze Yuan, Xin Yu, Yangtian Sun, Yuan-Chen Guo, Yan-Pei Cao, Ding Liang, Xiaojuan Qi  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2507.04285v1.pdf)  
  <details><summary>Abstract</summary>

  Training native 3D texture generative models remains a fundamental yet challenging problem, largely due to the limited availability of large-scale, high-quality 3D texture datasets. This scarcity hinders generalization to real-world scenarios. To address this, most existing methods finetune foundation image generative models to exploit their learned visual priors. However, these approaches typically generate only multi-view images and rely on post-processing to produce UV texture maps -- an essential representation in modern graphics pipelines. Such two-stage pipelines often suffer from error accumulation and spatial inconsistencies across the 3D surface. In this paper, we introduce SeqTex, a novel end-to-end framework that leverages the visual knowledge encoded in pretrained video foundation models to directly generate complete UV texture maps. Unlike previous methods that model the distribution of UV textures in isolation, SeqTex reformulates the task as a sequence generation problem, enabling the model to learn the joint distribution of multi-view renderings and UV...
  </details>  
- **[3D PixBrush: Image-Guided Local Texture Synthesis](https://arxiv.org/abs/2507.03731v1)**  
  Authors: Dale Decatur, Itai Lang, Kfir Aberman, Rana Hanocka  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2507.03731v1.pdf)  
  <details><summary>Abstract</summary>

  We present 3D PixBrush, a method for performing image-driven edits of local regions on 3D meshes. 3D PixBrush predicts a localization mask and a synthesized texture that faithfully portray the object in the reference image. Our predicted localizations are both globally coherent and locally precise. Globally - our method contextualizes the object in the reference image and automatically positions it onto the input mesh. Locally - our method produces masks that conform to the geometry of the reference image. Notably, our method does not require any user input (in the form of scribbles or bounding boxes) to achieve accurate localizations. Instead, our method predicts a localization mask on the 3D mesh from scratch. To achieve this, we propose a modification to the score distillation sampling technique which incorporates both the predicted localization and the reference image, referred to as localization-modulated image guidance. We demonstrate the effectiveness of our proposed technique...
  </details>  
  Keywords: texture synthesis  
- **[Point3R: Streaming 3D Reconstruction with Explicit Spatial Pointer Memory](https://arxiv.org/abs/2507.02863v2)**  
  Authors: Yuqi Wu, Wenzhao Zheng, Jie Zhou, Jiwen Lu  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2507.02863v2.pdf) | [![GitHub](https://img.shields.io/github/stars/YkiWu/Point3R?style=social)](https://github.com/YkiWu/Point3R)  
  <details><summary>Abstract</summary>

  Dense 3D scene reconstruction from an ordered sequence or unordered image collections is a critical step when bringing research in computer vision into practical scenarios. Following the paradigm introduced by DUSt3R, which unifies an image pair densely into a shared coordinate system, subsequent methods maintain an implicit memory to achieve dense 3D reconstruction from more images. However, such implicit memory is limited in capacity and may suffer from information loss of earlier frames. We propose Point3R, an online framework targeting dense streaming 3D reconstruction. To be specific, we maintain an explicit spatial pointer memory directly associated with the 3D structure of the current scene. Each pointer in this memory is assigned a specific 3D position and aggregates scene information nearby in the global coordinate system into a changing spatial feature. Information extracted from the latest frame interacts explicitly with this pointer memory, enabling dense integration of the current observation into...
  </details>  
  Keywords: 3d scene reconstruction  
- **[LiteReality: Graphics-Ready 3D Scene Reconstruction from RGB-D Scans](https://arxiv.org/abs/2507.02861v1)**  
  Authors: Zhening Huang, Xiaoyang Wu, Fangcheng Zhong, Hengshuang Zhao, Matthias Nießner, Joan Lasenby  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2507.02861v1.pdf) | [![Project](https://img.shields.io/badge/-Project-blue)](https://litereality.github.io) | [![Video](https://img.shields.io/badge/-Video-red)](https://www.youtube.com/watch?v=ecK9m3LXg2c)  
  <details><summary>Abstract</summary>

  We propose LiteReality, a novel pipeline that converts RGB-D scans of indoor environments into compact, realistic, and interactive 3D virtual replicas. LiteReality not only reconstructs scenes that visually resemble reality but also supports key features essential for graphics pipelines -- such as object individuality, articulation, high-quality physically based rendering materials, and physically based interaction. At its core, LiteReality first performs scene understanding and parses the results into a coherent 3D layout and objects with the help of a structured scene graph. It then reconstructs the scene by retrieving the most visually similar 3D artist-crafted models from a curated asset database. Next, the Material Painting module enhances realism by recovering high-quality, spatially varying materials. Finally, the reconstructed scene is integrated into a simulation engine with basic physical properties to enable interactive behavior. The resulting scenes are compact, editable, and fully compatible with standard graphics pipelines, making them suitable for applications in...
  </details>  
  Keywords: 3d scene reconstruction  
- **[AC-DiT: Adaptive Coordination Diffusion Transformer for Mobile Manipulation](https://arxiv.org/abs/2507.01961v3)**  
  Authors: Sixiang Chen, Jiaming Liu, Siyuan Qian, Han Jiang, Lily Li, Renrui Zhang, Zhuoyang Liu, Chenyang Gu, Chengkai Hou, Pengwei Wang, Zhongyuan Wang, Shanghang Zhang  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2507.01961v3.pdf)  
  <details><summary>Abstract</summary>

  Recently, mobile manipulation has attracted increasing attention for enabling language-conditioned robotic control in household tasks. However, existing methods still face challenges in coordinating mobile base and manipulator, primarily due to two limitations. On the one hand, they fail to explicitly model the influence of the mobile base on manipulator control, which easily leads to error accumulation under high degrees of freedom. On the other hand, they treat the entire mobile manipulation process with the same visual observation modality (e.g., either all 2D or all 3D), overlooking the distinct multimodal perception requirements at different stages during mobile manipulation. To address this, we propose the Adaptive Coordination Diffusion Transformer (AC-DiT), which enhances mobile base and manipulator coordination for end-to-end mobile manipulation. First, since the motion of the mobile base directly influences the manipulator's actions, we introduce a mobility-to-body conditioning mechanism that guides the model to first extract base motion representations, which are...
  </details>  
- **[Masks make discriminative models great again!](https://arxiv.org/abs/2507.00916v1)**  
  Authors: Tianshi Cao, Marie-Julie Rakotosaona, Ben Poole, Federico Tombari, Michael Niemeyer  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2507.00916v1.pdf)  
  <details><summary>Abstract</summary>

  We present Image2GS, a novel approach that addresses the challenging problem of reconstructing photorealistic 3D scenes from a single image by focusing specifically on the image-to-3D lifting component of the reconstruction process. By decoupling the lifting problem (converting an image to a 3D model representing what is visible) from the completion problem (hallucinating content not present in the input), we create a more deterministic task suitable for discriminative models. Our method employs visibility masks derived from optimized 3D Gaussian splats to exclude areas not visible from the source view during training. This masked training strategy significantly improves reconstruction quality in visible regions compared to strong baselines. Notably, despite being trained only on masked regions, Image2GS remains competitive with state-of-the-art discriminative models trained on full target images when evaluated on complete scenes. Our findings highlight the fundamental struggle discriminative models face when fitting unseen regions and demonstrate the advantages of addressing...
  </details>  
  Keywords: image-to-3d  
- **[DiGA3D: Coarse-to-Fine Diffusional Propagation of Geometry and Appearance for Versatile 3D Inpainting](https://arxiv.org/abs/2507.00429v1)**  
  Authors: Jingyi Pan, Dan Xu, Qiong Luo  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2507.00429v1.pdf) | [![Project](https://img.shields.io/badge/-Project-blue)](https://rorisis.github.io/DiGA3D)  
  <details><summary>Abstract</summary>

  Developing a unified pipeline that enables users to remove, re-texture, or replace objects in a versatile manner is crucial for text-guided 3D inpainting. However, there are still challenges in performing multiple 3D inpainting tasks within a unified framework: 1) Single reference inpainting methods lack robustness when dealing with views that are far from the reference view. 2) Appearance inconsistency arises when independently inpainting multi-view images with 2D diffusion priors; 3) Geometry inconsistency limits performance when there are significant geometric changes in the inpainting regions. To tackle these challenges, we introduce DiGA3D, a novel and versatile 3D inpainting pipeline that leverages diffusion models to propagate consistent appearance and geometry in a coarse-to-fine manner. First, DiGA3D develops a robust strategy for selecting multiple reference views to reduce errors during propagation. Next, DiGA3D designs an Attention Feature Propagation (AFP) mechanism that propagates attention features from the selected reference views to other views via...
  </details>  
- **[ViscoReg: Neural Signed Distance Functions via Viscosity Solutions](https://arxiv.org/abs/2507.00412v2)**  
  Authors: Meenakshi Krishnan, Ramani Duraiswami  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2507.00412v2.pdf)  
  <details><summary>Abstract</summary>

  Implicit Neural Representations (INRs) that learn Signed Distance Functions (SDFs) from point cloud data represent the state-of-the-art for geometrically accurate 3D scene reconstruction. However, training these Neural SDFs often requires enforcing the Eikonal equation, an ill-posed equation that also leads to unstable gradient flows. Numerical Eikonal solvers have relied on viscosity approaches for regularization and stability. Motivated by this well-established theory, we introduce ViscoReg, a novel regularizer that provably stabilizes Neural SDF training. Empirically, ViscoReg outperforms state-of-the-art approaches such as SIREN, DiGS, and StEik on ShapeNet, the Surface Reconstruction Benchmark, and 3D scene reconstruction datasets. Additionally, we establish novel generalization error estimates for Neural SDFs in terms of the training error, using the theory of viscosity solutions.
  </details>  
  Keywords: 3d scene reconstruction  
- **[Learning Dense Feature Matching via Lifting Single 2D Image to 3D Space](https://arxiv.org/abs/2507.00392v2)**  
  Authors: Yingping Liang, Yutao Hu, Wenqi Shao, Ying Fu  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2507.00392v2.pdf)  
  <details><summary>Abstract</summary>

  Feature matching plays a fundamental role in many computer vision tasks, yet existing methods heavily rely on scarce and clean multi-view image collections, which constrains their generalization to diverse and challenging scenarios. Moreover, conventional feature encoders are typically trained on single-view 2D images, limiting their capacity to capture 3D-aware correspondences. In this paper, we propose a novel two-stage framework that lifts 2D images to 3D space, named as \textbf{Lift to Match (L2M)}, taking full advantage of large-scale and diverse single-view images. To be specific, in the first stage, we learn a 3D-aware feature encoder using a combination of multi-view image synthesis and 3D feature Gaussian representation, which injects 3D geometry knowledge into the encoder. In the second stage, a novel-view rendering strategy, combined with large-scale synthetic data generation from single-view images, is employed to learn a feature decoder for robust feature matching, thus achieving generalization across diverse domains. Extensive experiments...
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
- **[CG-MLLM: Captioning and Generating 3D content via Multi-modal Large Language Models](https://arxiv.org/abs/2601.21798v1)**  
  Authors: Junming Huang, Weiwei Xu  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2601.21798v1.pdf)  
  <details><summary>Abstract</summary>

  Large Language Models(LLMs) have revolutionized text generation and multimodal perception, but their capabilities in 3D content generation remain underexplored. Existing methods compromise by producing either low-resolution meshes or coarse structural proxies, failing to capture fine-grained geometry natively. In this paper, we propose CG-MLLM, a novel Multi-modal Large Language Model (MLLM) capable of 3D captioning and high-resolution 3D generation in a single framework. Leveraging the Mixture-of-Transformer architecture, CG-MLLM decouples disparate modeling needs, where the Token-level Autoregressive (TokenAR) Transformer handles token-level content, and the Block-level Autoregressive (BlockAR) Transformer handles block-level content. By integrating a pre-trained vision-language backbone with a specialized 3D VAE latent space, CG-MLLM facilitates long-context interactions between standard tokens and spatial blocks within a single integrated architecture. Experimental results show that CG-MLLM significantly outperforms existing MLLMs in generating high-fidelity 3D objects, effectively bringing high-resolution 3D content creation into the mainstream LLM paradigm.
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
  Keywords: novel view synthesis, 3d scene reconstruction  
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
- **[ATATA: One Algorithm to Align Them All](https://arxiv.org/abs/2601.11194v1)**  
  Authors: Boyi Pang, Savva Ignatyev, Vladimir Ippolitov, Ramil Khafizov, Yurii Melnik, Oleg Voynov, Maksim Nakhodnov, Aibek Alanov, Xiaopeng Fan, Peter Wonka, Evgeny Burnaev  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2601.11194v1.pdf)  
  <details><summary>Abstract</summary>

  We suggest a new multi-modal algorithm for joint inference of paired structurally aligned samples with Rectified Flow models. While some existing methods propose a codependent generation process, they do not view the problem of joint generation from a structural alignment perspective. Recent work uses Score Distillation Sampling to generate aligned 3D models, but SDS is known to be time-consuming, prone to mode collapse, and often provides cartoonish results. By contrast, our suggested approach relies on the joint transport of a segment in the sample space, yielding faster computation at inference time. Our approach can be built on top of an arbitrary Rectified Flow model operating on the structured latent space. We show the applicability of our method to the domains of image, video, and 3D shape generation using state-of-the-art baselines and evaluate it against both editing-based and joint inference-based competing approaches. We demonstrate a high degree of structural alignment for...
  </details>  
- **[SyncTwin: Fast Digital Twin Construction and Synchronization for Safe Robotic Grasping](https://arxiv.org/abs/2601.09920v1)**  
  Authors: Ruopeng Huang, Boyu Yang, Wenlong Gui, Jeremy Morgan, Erdem Biyik, Jiachen Li  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2601.09920v1.pdf)  
  <details><summary>Abstract</summary>

  Accurate and safe grasping under dynamic and visually occluded conditions remains a core challenge in real-world robotic manipulation. We present SyncTwin, a digital twin framework that unifies fast 3D scene reconstruction and real-to-sim synchronization for robust and safety-aware grasping in such environments. In the offline stage, we employ VGGT to rapidly reconstruct object-level 3D assets from RGB images, forming a reusable geometry library for simulation. During execution, SyncTwin continuously synchronizes the digital twin by tracking real-world object states via point cloud segmentation updates and aligning them through colored-ICP registration. The updated twin enables motion planners to compute collision-free and dynamically feasible trajectories in simulation, which are safely executed on the real robot through a closed real-to-sim-to-real loop. Experiments in dynamic and occluded scenes show that SyncTwin improves grasp accuracy and motion safety, demonstrating the effectiveness of digital-twin synchronization for real-world robotic execution.
  </details>  
  Keywords: 3d scene reconstruction  
- **[StdGEN++: A Comprehensive System for Semantic-Decomposed 3D Character Generation](https://arxiv.org/abs/2601.07660v1)**  
  Authors: Yuze He, Yanning Zhou, Wang Zhao, Jingwen Ye, Zhongkai Wu, Ran Yi, Yong-Jin Liu  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2601.07660v1.pdf)  
  <details><summary>Abstract</summary>

  We present StdGEN++, a novel and comprehensive system for generating high-fidelity, semantically decomposed 3D characters from diverse inputs. Existing 3D generative methods often produce monolithic meshes that lack the structural flexibility required by industrial pipelines in gaming and animation. Addressing this gap, StdGEN++ is built upon a Dual-branch Semantic-aware Large Reconstruction Model (Dual-Branch S-LRM), which jointly reconstructs geometry, color, and per-component semantics in a feed-forward manner. To achieve production-level fidelity, we introduce a novel semantic surface extraction formalism compatible with hybrid implicit fields. This mechanism is accelerated by a coarse-to-fine proposal scheme, which significantly reduces memory footprint and enables high-resolution mesh generation. Furthermore, we propose a video-diffusion-based texture decomposition module that disentangles appearance into editable layers (e.g., separated iris and skin), resolving semantic confusion in facial regions. Experiments demonstrate that StdGEN++ achieves state-of-the-art performance, significantly outperforming existing methods in geometric accuracy and semantic disentanglement. Crucially, the resulting structural independence unlocks...
  </details>  
- **[HERE: Hierarchical Active Exploration of Radiance Field with Epistemic Uncertainty Minimization](https://arxiv.org/abs/2601.07242v1)**  
  Authors: Taekbeom Lee, Dabin Kim, Youngseok Jang, H. Jin Kim  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2601.07242v1.pdf)  
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
- **[Gen3R: 3D Scene Generation Meets Feed-Forward Reconstruction](https://arxiv.org/abs/2601.04090v1)**  
  Authors: Jiaxin Huang, Yuanbo Yang, Bangbang Yang, Lin Ma, Yuewen Ma, Yiyi Liao  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2601.04090v1.pdf)  
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
- **[MorphAny3D: Unleashing the Power of Structured Latent in 3D Morphing](https://arxiv.org/abs/2601.00204v1)**  
  Authors: Xiaokun Sun, Zeyu Cai, Hao Tang, Ying Tai, Jian Yang, Zhenyu Zhang  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2601.00204v1.pdf) | [![Project](https://img.shields.io/badge/-Project-blue)](https://xiaokunsun.github.io/MorphAny3D.github.io)  
  <details><summary>Abstract</summary>

  3D morphing remains challenging due to the difficulty of generating semantically consistent and temporally smooth deformations, especially across categories. We present MorphAny3D, a training-free framework that leverages Structured Latent (SLAT) representations for high-quality 3D morphing. Our key insight is that intelligently blending source and target SLAT features within the attention mechanisms of 3D generators naturally produces plausible morphing sequences. To this end, we introduce Morphing Cross-Attention (MCA), which fuses source and target information for structural coherence, and Temporal-Fused Self-Attention (TFSA), which enhances temporal consistency by incorporating features from preceding frames. An orientation correction strategy further mitigates the pose ambiguity within the morphing steps. Extensive experiments show that our method generates state-of-the-art morphing sequences, even for challenging cross-category cases. MorphAny3D further supports advanced applications such as decoupled morphing and 3D style transfer, and can be generalized to other SLAT-based generative models. Project page: https://xiaokunsun.github.io/MorphAny3D.github.io/.
  </details>  
  Keywords: 3d generator  

### February 2026
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
- **[ShapeUP: Scalable Image-Conditioned 3D Editing](https://arxiv.org/abs/2602.05676v1)**  
  Authors: Inbar Gat, Dana Cohen-Bar, Guy Levy, Elad Richardson, Daniel Cohen-Or  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2602.05676v1.pdf)  
  <details><summary>Abstract</summary>

  Recent advancements in 3D foundation models have enabled the generation of high-fidelity assets, yet precise 3D manipulation remains a significant challenge. Existing 3D editing frameworks often face a difficult trade-off between visual controllability, geometric consistency, and scalability. Specifically, optimization-based methods are prohibitively slow, multi-view 2D propagation techniques suffer from visual drift, and training-free latent manipulation methods are inherently bound by frozen priors and cannot directly benefit from scaling. In this work, we present ShapeUP, a scalable, image-conditioned 3D editing framework that formulates editing as a supervised latent-to-latent translation within a native 3D representation. This formulation allows ShapeUP to build on a pretrained 3D foundation model, leveraging its strong generative prior while adapting it to editing through supervised training. In practice, ShapeUP is trained on triplets consisting of a source 3D shape, an edited 2D image, and the corresponding edited 3D shape, and learns a direct mapping using a 3D Diffusion...
  </details>  
- **[Monte Carlo Rendering to Diffusion Curves with Differential BEM](https://arxiv.org/abs/2602.05492v1)**  
  Authors: Ryusuke Sugimoto, Christopher Batty, Siddhartha Chaudhuri, Iliyan Georgiev, Toshiya Hachisuka, Kevin Wampler, Michal Lukáč  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2602.05492v1.pdf)  
  <details><summary>Abstract</summary>

  We present a method for generating vector graphics, in the form of diffusion curves, directly from noisy samples produced by a Monte Carlo renderer. While generating raster images from 3D geometry via Monte Carlo raytracing is commonplace, there is no corresponding practical approach for robustly and directly extracting editable vector images with shading information from 3D geometry. To fill this gap, we formulate the problem as a stochastic optimization problem over the space of geometries and colors of diffusion curve handles, and solve it with the Levenberg-Marquardt algorithm. At the core of our method is a novel differential boundary element method (BEM) framework that reconstructs colors from diffusion curve handles and computes gradients with respect to their parameters, requiring the expensive matrix factorization only once at the beginning of the optimization. Unlike triangulation-based techniques that require a clean domain decomposition, our method is robust to geometrically challenging scenarios, such as...
  </details>  
- **[Fast-SAM3D: 3Dfy Anything in Images but Faster](https://arxiv.org/abs/2602.05293v1)**  
  Authors: Weilun Feng, Mingqiang Wu, Zhiliang Chen, Chuanguang Yang, Haotong Qin, Yuqi Li, Xiaokun Liu, Guoxin Fan, Zhulin An, Libo Huang, Yulun Zhang, Michele Magno, Yongjun Xu  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2602.05293v1.pdf) | [![GitHub](https://img.shields.io/github/stars/wlfeng0509/Fast-SAM3D?style=social)](https://github.com/wlfeng0509/Fast-SAM3D)  
  <details><summary>Abstract</summary>

  SAM3D enables scalable, open-world 3D reconstruction from complex scenes, yet its deployment is hindered by prohibitive inference latency. In this work, we conduct the \textbf{first systematic investigation} into its inference dynamics, revealing that generic acceleration strategies are brittle in this context. We demonstrate that these failures stem from neglecting the pipeline's inherent multi-level \textbf{heterogeneity}: the kinematic distinctiveness between shape and layout, the intrinsic sparsity of texture refinement, and the spectral variance across geometries. To address this, we present \textbf{Fast-SAM3D}, a training-free framework that dynamically aligns computation with instantaneous generation complexity. Our approach integrates three heterogeneity-aware mechanisms: (1) \textit{Modality-Aware Step Caching} to decouple structural evolution from sensitive layout updates; (2) \textit{Joint Spatiotemporal Token Carving} to concentrate refinement on high-entropy regions; and (3) \textit{Spectral-Aware Token Aggregation} to adapt decoding resolution. Extensive experiments demonstrate that Fast-SAM3D delivers up to \textbf{2.67$\times$} end-to-end speedup with negligible fidelity loss, establishing a new Pareto frontier for efficient...
  </details>  
- **[Seeing Through Clutter: Structured 3D Scene Reconstruction via Iterative Object Removal](https://arxiv.org/abs/2602.04053v1)**  
  Authors: Rio Aguina-Kang, Kevin James Blackburn-Matzen, Thibault Groueix, Vladimir Kim, Matheus Gadelha  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2602.04053v1.pdf) | [![Project](https://img.shields.io/badge/-Project-blue)](https://rioak.github.io/seeingthroughclutter)  
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
- **[Symmetrization of 3D Generative Models](https://arxiv.org/abs/2512.18953v1)**  
  Authors: Nicolas Caytuiro, Ivan Sipiran  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2512.18953v1.pdf)  
  <details><summary>Abstract</summary>

  We propose a novel data-centric approach to promote symmetry in 3D generative models by modifying the training data rather than the model architecture. Our method begins with an analysis of reflectional symmetry in both real-world 3D shapes and samples generated by state-of-the-art models. We hypothesize that training a generative model exclusively on half-objects, obtained by reflecting one half of the shapes along the x=0 plane, enables the model to learn a rich distribution of partial geometries which, when reflected during generation, yield complete shapes that are both visually plausible and geometrically symmetric. To test this, we construct a new dataset of half-objects from three ShapeNet classes (Airplane, Car, and Chair) and train two generative models. Experiments demonstrate that the generated shapes are symmetrical and consistent, compared with the generated objects from the original model and the original dataset objects.
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
- **[ViewMask-1-to-3: Multi-View Consistent Image Generation via Multimodal Diffusion Models](https://arxiv.org/abs/2512.14099v1)**  
  Authors: Ruishu Zhu, Zhihao Huang, Jiacheng Sun, Ping Luo, Hongyuan Zhang, Xuelong Li  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2512.14099v1.pdf)  
  <details><summary>Abstract</summary>

  Multi-view image generation from a single image and text description remains challenging due to the difficulty of maintaining geometric consistency across different viewpoints. Existing approaches typically rely on 3D-aware architectures or specialized diffusion models that require extensive multi-view training data and complex geometric priors. In this work, we introduce ViewMask-1-to-3, a pioneering approach to apply discrete diffusion models to multi-view image generation. Unlike continuous diffusion methods that operate in latent spaces, ViewMask-1-to-3 formulates multi-view synthesis as a discrete sequence modeling problem, where each viewpoint is represented as visual tokens obtained through MAGVIT-v2 tokenization. By unifying language and vision through masked token prediction, our approach enables progressive generation of multiple viewpoints through iterative token unmasking with text input. ViewMask-1-to-3 achieves cross-view consistency through simple random masking combined with self-attention, eliminating the requirement for complex 3D geometric constraints or specialized attention architectures. Our approach demonstrates that discrete diffusion provides a viable and...
  </details>  
  Keywords: multi-view synthesis  
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
  Keywords: novel view synthesis, 3d scene reconstruction  
- **[Animus3D: Text-driven 3D Animation via Motion Score Distillation](https://arxiv.org/abs/2512.12534v1)**  
  Authors: Qi Sun, Can Wang, Jiaxiang Shang, Wensen Feng, Jing Liao  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2512.12534v1.pdf) | [![Project](https://img.shields.io/badge/-Project-blue)](https://qiisun.github.io/animus3d_page)  
  <details><summary>Abstract</summary>

  We present Animus3D, a text-driven 3D animation framework that generates motion field given a static 3D asset and text prompt. Previous methods mostly leverage the vanilla Score Distillation Sampling (SDS) objective to distill motion from pretrained text-to-video diffusion, leading to animations with minimal movement or noticeable jitter. To address this, our approach introduces a novel SDS alternative, Motion Score Distillation (MSD). Specifically, we introduce a LoRA-enhanced video diffusion model that defines a static source distribution rather than pure noise as in SDS, while another inversion-based noise estimation technique ensures appearance preservation when guiding motion. To further improve motion fidelity, we incorporate explicit temporal and spatial regularization terms that mitigate geometric distortions across time and space. Additionally, we propose a motion refinement module to upscale the temporal resolution and enhance fine-grained details, overcoming the fixed-resolution constraints of the underlying video model. Extensive experiments demonstrate that Animus3D successfully animates static 3D assets...
  </details>  
- **[Particulate: Feed-Forward 3D Object Articulation](https://arxiv.org/abs/2512.11798v1)**  
  Authors: Ruining Li, Yuxin Yao, Chuanxia Zheng, Christian Rupprecht, Joan Lasenby, Shangzhe Wu, Andrea Vedaldi  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2512.11798v1.pdf)  
  <details><summary>Abstract</summary>

  We present Particulate, a feed-forward approach that, given a single static 3D mesh of an everyday object, directly infers all attributes of the underlying articulated structure, including its 3D parts, kinematic structure, and motion constraints. At its core is a transformer network, Part Articulation Transformer, which processes a point cloud of the input mesh using a flexible and scalable architecture to predict all the aforementioned attributes with native multi-joint support. We train the network end-to-end on a diverse collection of articulated 3D assets from public datasets. During inference, Particulate lifts the network's feed-forward prediction to the input mesh, yielding a fully articulated 3D model in seconds, much faster than prior approaches that require per-object optimization. Particulate can also accurately infer the articulated structure of AI-generated 3D assets, enabling full-fledged extraction of articulated 3D objects from a single (real or synthetic) image when combined with an off-the-shelf image-to-3D generator. We further...
  </details>  
  Keywords: 3d generator, image-to-3d  
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
- **[UniPart: Part-Level 3D Generation with Unified 3D Geom-Seg Latents](https://arxiv.org/abs/2512.09435v1)**  
  Authors: Xufan He, Yushuang Wu, Xiaoyang Guo, Chongjie Ye, Jiaqing Zhou, Tianlei Hu, Xiaoguang Han, Dong Du  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2512.09435v1.pdf)  
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
- **[Photo3D: Advancing Photorealistic 3D Generation through Structure-Aligned Detail Enhancement](https://arxiv.org/abs/2512.08535v1)**  
  Authors: Xinyue Liang, Zhinyuan Ma, Lingchen Sun, Yanjun Guo, Lei Zhang  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2512.08535v1.pdf)  
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
- **[SpaceControl: Introducing Test-Time Spatial Control to 3D Generative Modeling](https://arxiv.org/abs/2512.05343v1)**  
  Authors: Elisabetta Fedele, Francis Engelmann, Ian Huang, Or Litany, Marc Pollefeys, Leonidas Guibas  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2512.05343v1.pdf) | [![Project](https://img.shields.io/badge/-Project-blue)](https://spacecontrol3d.github.io)  
  <details><summary>Abstract</summary>

  Generative methods for 3D assets have recently achieved remarkable progress, yet providing intuitive and precise control over the object geometry remains a key challenge. Existing approaches predominantly rely on text or image prompts, which often fall short in geometric specificity: language can be ambiguous, and images are cumbersome to edit. In this work, we introduce SpaceControl, a training-free test-time method for explicit spatial control of 3D generation. Our approach accepts a wide range of geometric inputs, from coarse primitives to detailed meshes, and integrates seamlessly with modern pre-trained generative models without requiring any additional training. A controllable parameter lets users trade off between geometric fidelity and output realism. Extensive quantitative evaluation and user studies demonstrate that SpaceControl outperforms both training-based and optimization-based baselines in geometric faithfulness while preserving high visual quality. Finally, we present an interactive user interface that enables online editing of superquadrics for direct conversion into textured 3D...
  </details>  
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
- **[Order Matters: 3D Shape Generation from Sequential VR Sketches](https://arxiv.org/abs/2512.04761v2)**  
  Authors: Yizi Chen, Sidi Wu, Tianyi Xiao, Nina Wiedemann, Loic Landrieu  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2512.04761v2.pdf) | [![Project](https://img.shields.io/badge/-Project-blue)](https://chenyizi086.github.io/VRSketch2Shape_website)  
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
- **[ShelfGaussian: Shelf-Supervised Open-Vocabulary Gaussian-based 3D Scene Understanding](https://arxiv.org/abs/2512.03370v1)**  
  Authors: Lingjun Zhao, Yandong Luo, James Hay, Lu Gan  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2512.03370v1.pdf) | [![Project](https://img.shields.io/badge/-Project-blue)](https://lunarlab-gatech.github.io/ShelfGaussian)  
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

### August 2025
- **[3D-LATTE: Latent Space 3D Editing from Textual Instructions](https://arxiv.org/abs/2509.00269v3)**  
  Authors: Maria Parelli, Michael Oechsle, Michael Niemeyer, Federico Tombari, Andreas Geiger  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2509.00269v3.pdf) | [![Project](https://img.shields.io/badge/-Project-blue)](https://mparelli.github.io/3d-latte)  
  <details><summary>Abstract</summary>

  Despite the recent success of multi-view diffusion models for text/image-based 3D asset generation, instruction-based editing of 3D assets lacks surprisingly far behind the quality of generation models. The main reason is that recent approaches using 2D priors suffer from view-inconsistent editing signals. Going beyond 2D prior distillation methods and multi-view editing strategies, we propose a training-free editing method that operates within the latent space of a native 3D diffusion model, allowing us to directly manipulate 3D geometry. We guide the edit synthesis by blending 3D attention maps from the generation with the source object. Coupled with geometry-aware regularization guidance, a spectral modulation strategy in the Fourier domain and a refinement step for 3D enhancement, our method outperforms previous 3D editing methods enabling high-fidelity and precise edits across a wide range of shapes and semantic manipulations. Our project webpage is https://mparelli.github.io/3d-latte
  </details>  
  Keywords: 3d asset generation  
- **[Droplet3D: Commonsense Priors from Videos Facilitate 3D Generation](https://arxiv.org/abs/2508.20470v1)**  
  Authors: Xiaochuan Li, Guoguang Du, Runze Zhang, Liang Jin, Qi Jia, Lihua Lu, Zhenhua Guo, Yaqian Zhao, Haiyang Liu, Tianqi Wang, Changsheng Li, Xiaoli Gong, Rengang Li, Baoyu Fan  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2508.20470v1.pdf) | [![Project](https://img.shields.io/badge/-Project-blue)](https://dropletx.github.io)  
  <details><summary>Abstract</summary>

  Scaling laws have validated the success and promise of large-data-trained models in creative generation across text, image, and video domains. However, this paradigm faces data scarcity in the 3D domain, as there is far less of it available on the internet compared to the aforementioned modalities. Fortunately, there exist adequate videos that inherently contain commonsense priors, offering an alternative supervisory signal to mitigate the generalization bottleneck caused by limited native 3D data. On the one hand, videos capturing multiple views of an object or scene provide a spatial consistency prior for 3D generation. On the other hand, the rich semantic information contained within the videos enables the generated content to be more faithful to the text prompts and semantically plausible. This paper explores how to apply the video modality in 3D asset generation, spanning datasets to models. We introduce Droplet3D-4M, the first large-scale video dataset with multi-view level annotations, and...
  </details>  
  Keywords: 3d asset generation  
- **[Fast 3D Diffusion for Scalable Granular Media Synthesis](https://arxiv.org/abs/2508.19752v1)**  
  Authors: Muhammad Moeeze Hassan, Régis Cottereau, Filippo Gatti, Patryk Dec  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2508.19752v1.pdf)  
  <details><summary>Abstract</summary>

  Simulating granular media, using Discrete Element Method is a computationally intensive task. This is especially true during initialization phase, which dominates total simulation time because of large displacements involved and associated kinetic energy. We overcome this bottleneck with a novel generative pipeline based on 3D diffusion models that directly synthesizes arbitrarily large granular assemblies in their final and physically realistic configurations. The approach frames the problem as a 3D generative modeling task, consisting of a two-stage pipeline. First a diffusion model is trained to generate independent 3D voxel grids representing granular media. Second, a 3D inpainting model, adapted from 2D inpainting techniques using masked inputs, stitches these grids together seamlessly, enabling synthesis of large samples with physically realistic structure. The inpainting model explores several masking strategies for the inputs to the underlying UNets by training the network to infer missing portions of voxel grids from a concatenation of noised tensors,...
  </details>  
- **[DATR: Diffusion-based 3D Apple Tree Reconstruction Framework with Sparse-View](https://arxiv.org/abs/2508.19508v1)**  
  Authors: Tian Qiu, Alan Zoubi, Yiyuan Lin, Ruiming Du, Lailiang Cheng, Yu Jiang  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2508.19508v1.pdf)  
  <details><summary>Abstract</summary>

  Digital twin applications offered transformative potential by enabling real-time monitoring and robotic simulation through accurate virtual replicas of physical assets. The key to these systems is 3D reconstruction with high geometrical fidelity. However, existing methods struggled under field conditions, especially with sparse and occluded views. This study developed a two-stage framework (DATR) for the reconstruction of apple trees from sparse views. The first stage leverages onboard sensors and foundation models to semi-automatically generate tree masks from complex field images. Tree masks are used to filter out background information in multi-modal data for the single-image-to-3D reconstruction at the second stage. This stage consists of a diffusion model and a large reconstruction model for respective multi view and implicit neural field generation. The training of the diffusion model and LRM was achieved by using realistic synthetic apple trees generated by a Real2Sim data generator. The framework was evaluated on both field and...
  </details>  
  Keywords: image-to-3d  
- **[VoxHammer: Training-Free Precise and Coherent 3D Editing in Native 3D Space](https://arxiv.org/abs/2508.19247v1)**  
  Authors: Lin Li, Zehuan Huang, Haoran Feng, Gengxiong Zhuang, Rui Chen, Chunchao Guo, Lu Sheng  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2508.19247v1.pdf) | [![Project](https://img.shields.io/badge/-Project-blue)](https://huanngzh.github.io/VoxHammer-Page)  
  <details><summary>Abstract</summary>

  3D local editing of specified regions is crucial for game industry and robot interaction. Recent methods typically edit rendered multi-view images and then reconstruct 3D models, but they face challenges in precisely preserving unedited regions and overall coherence. Inspired by structured 3D generative models, we propose VoxHammer, a novel training-free approach that performs precise and coherent editing in 3D latent space. Given a 3D model, VoxHammer first predicts its inversion trajectory and obtains its inverted latents and key-value tokens at each timestep. Subsequently, in the denoising and editing phase, we replace the denoising features of preserved regions with the corresponding inverted latents and cached key-value tokens. By retaining these contextual features, this approach ensures consistent reconstruction of preserved areas and coherent integration of edited parts. To evaluate the consistency of preserved regions, we constructed Edit3D-Bench, a human-annotated dataset comprising hundreds of samples, each with carefully labeled 3D editing regions. Experiments...
  </details>  
- **[Articulate3D: Zero-Shot Text-Driven 3D Object Posing](https://arxiv.org/abs/2508.19244v1)**  
  Authors: Oishi Deb, Anjun Hu, Ashkan Khakzar, Philip Torr, Christian Rupprecht  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2508.19244v1.pdf) | [![Project](https://img.shields.io/badge/-Project-blue)](https://odeb1.github.io/articulate3d_page_deb)  
  <details><summary>Abstract</summary>

  We propose a training-free method, Articulate3D, to pose a 3D asset through language control. Despite advances in vision and language models, this task remains surprisingly challenging. To achieve this goal, we decompose the problem into two steps. We modify a powerful image-generator to create target images conditioned on the input image and a text instruction. We then align the mesh to the target images through a multi-view pose optimisation step. In detail, we introduce a self-attention rewiring mechanism (RSActrl) that decouples the source structure from pose within an image generative model, allowing it to maintain a consistent structure across varying poses. We observed that differentiable rendering is an unreliable signal for articulation optimisation; instead, we use keypoints to establish correspondences between input and target images. The effectiveness of Articulate3D is demonstrated across a diverse range of 3D objects and free-form text prompts, successfully manipulating poses while maintaining the original identity...
  </details>  
- **[LSD-3D: Large-Scale 3D Driving Scene Generation with Geometry Grounding](https://arxiv.org/abs/2508.19204v1)**  
  Authors: Julian Ost, Andrea Ramazzina, Amogh Joshi, Maximilian Bömer, Mario Bijelic, Felix Heide  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2508.19204v1.pdf)  
  <details><summary>Abstract</summary>

  Large-scale scene data is essential for training and testing in robot learning. Neural reconstruction methods have promised the capability of reconstructing large physically-grounded outdoor scenes from captured sensor data. However, these methods have baked-in static environments and only allow for limited scene control -- they are functionally constrained in scene and trajectory diversity by the captures from which they are reconstructed. In contrast, generating driving data with recent image or video diffusion models offers control, however, at the cost of geometry grounding and causality. In this work, we aim to bridge this gap and present a method that directly generates large-scale 3D driving scenes with accurate geometry, allowing for causal novel view synthesis with object permanence and explicit 3D geometry estimation. The proposed method combines the generation of a proxy geometry and environment representation with score distillation from learned 2D image priors. We find that this approach allows for high...
  </details>  
  Keywords: novel view synthesis  
- **[T-MLP: Tailed Multi-Layer Perceptron for Level-of-Detail Signal Representation](https://arxiv.org/abs/2509.00066v2)**  
  Authors: Chuanxiang Yang, Yuanfeng Zhou, Guangshun Wei, Siyu Ren, Yuan Liu, Junhui Hou, Wenping Wang  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2509.00066v2.pdf)  
  <details><summary>Abstract</summary>

  Level-of-detail (LoD) representation is critical for efficiently modeling and transmitting various types of signals, such as images and 3D shapes. In this work, we propose a novel network architecture that enables LoD signal representation. Our approach builds on a modified Multi-Layer Perceptron (MLP), which inherently operates at a single scale and thus lacks native LoD support. Specifically, we introduce the Tailed Multi-Layer Perceptron (T-MLP), which extends the MLP by attaching an output branch, also called tail, to each hidden layer. Each tail refines the residual between the current prediction and the ground-truth signal, so that the accumulated outputs across layers correspond to the target signals at different LoDs, enabling multi-scale modeling with supervision from only a single-resolution signal. Extensive experiments demonstrate that our T-MLP outperforms existing neural LoD baselines across diverse signal representation tasks.
  </details>  
- **[SAT-SKYLINES: 3D Building Generation from Satellite Imagery and Coarse Geometric Priors](https://arxiv.org/abs/2508.18531v1)**  
  Authors: Zhangyu Jin, Andrew Feng  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2508.18531v1.pdf)  
  <details><summary>Abstract</summary>

  We present SatSkylines, a 3D building generation approach that takes satellite imagery and coarse geometric priors. Without proper geometric guidance, existing image-based 3D generation methods struggle to recover accurate building structures from the top-down views of satellite images alone. On the other hand, 3D detailization methods tend to rely heavily on highly detailed voxel inputs and fail to produce satisfying results from simple priors such as cuboids. To address these issues, our key idea is to model the transformation from interpolated noisy coarse priors to detailed geometries, enabling flexible geometric control without additional computational cost. We have further developed Skylines-50K, a large-scale dataset of over 50,000 unique and stylized 3D building assets in order to support the generations of detailed building models. Extensive evaluations indicate the effectiveness of our model and strong generalization ability.
  </details>  
- **[IDU: Incremental Dynamic Update of Existing 3D Virtual Environments with New Imagery Data](https://arxiv.org/abs/2508.17579v1)**  
  Authors: Meida Chen, Luis Leal, Yue Hu, Rong Liu, Butian Xiong, Andrew Feng, Jiuyi Xu, Yangming Shi  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2508.17579v1.pdf)  
  <details><summary>Abstract</summary>

  For simulation and training purposes, military organizations have made substantial investments in developing high-resolution 3D virtual environments through extensive imaging and 3D scanning. However, the dynamic nature of battlefield conditions-where objects may appear or vanish over time-makes frequent full-scale updates both time-consuming and costly. In response, we introduce the Incremental Dynamic Update (IDU) pipeline, which efficiently updates existing 3D reconstructions, such as 3D Gaussian Splatting (3DGS), with only a small set of newly acquired images. Our approach starts with camera pose estimation to align new images with the existing 3D model, followed by change detection to pinpoint modifications in the scene. A 3D generative AI model is then used to create high-quality 3D assets of the new elements, which are seamlessly integrated into the existing 3D model. The IDU pipeline incorporates human guidance to ensure high accuracy in object identification and placement, with each update focusing on a single new...
  </details>  
- **[PersPose: 3D Human Pose Estimation with Perspective Encoding and Perspective Rotation](https://arxiv.org/abs/2508.17239v2)**  
  Authors: Xiaoyang Hao, Han Li  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2508.17239v2.pdf) | [![GitHub](https://img.shields.io/github/stars/KenAdamsJoseph/PersPose?style=social)](https://github.com/KenAdamsJoseph/PersPose)  
  <details><summary>Abstract</summary>

  Monocular 3D human pose estimation (HPE) methods estimate the 3D positions of joints from individual images. Existing 3D HPE approaches often use the cropped image alone as input for their models. However, the relative depths of joints cannot be accurately estimated from cropped images without the corresponding camera intrinsics, which determine the perspective relationship between 3D objects and the cropped images. In this work, we introduce Perspective Encoding (PE) to encode the camera intrinsics of the cropped images. Moreover, since the human subject can appear anywhere within the original image, the perspective relationship between the 3D scene and the cropped image differs significantly, which complicates model fitting. Additionally, the further the human subject deviates from the image center, the greater the perspective distortions in the cropped image. To address these issues, we propose Perspective Rotation (PR), a transformation applied to the original image that centers the human subject, thereby reducing...
  </details>  
- **[Structural Energy-Guided Sampling for View-Consistent Text-to-3D](https://arxiv.org/abs/2508.16917v1)**  
  Authors: Qing Zhang, Jinguang Tong, Jie Hong, Jing Zhang, Xuesong Li  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2508.16917v1.pdf)  
  <details><summary>Abstract</summary>

  Text-to-3D generation often suffers from the Janus problem, where objects look correct from the front but collapse into duplicated or distorted geometry from other angles. We attribute this failure to viewpoint bias in 2D diffusion priors, which propagates into 3D optimization. To address this, we propose Structural Energy-Guided Sampling (SEGS), a training-free, plug-and-play framework that enforces multi-view consistency entirely at sampling time. SEGS defines a structural energy in a PCA subspace of intermediate U-Net features and injects its gradients into the denoising trajectory, steering geometry toward the intended viewpoint while preserving appearance fidelity. Integrated seamlessly into SDS/VSD pipelines, SEGS significantly reduces Janus artifacts, achieving improved geometric alignment and viewpoint consistency without retraining or weight modification.
  </details>  
  Keywords: text-to-3d  
- **[MV-RAG: Retrieval Augmented Multiview Diffusion](https://arxiv.org/abs/2508.16577v1)**  
  Authors: Yosef Dayani, Omer Benishu, Sagie Benaim  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2508.16577v1.pdf)  
  <details><summary>Abstract</summary>

  Text-to-3D generation approaches have advanced significantly by leveraging pretrained 2D diffusion priors, producing high-quality and 3D-consistent outputs. However, they often fail to produce out-of-domain (OOD) or rare concepts, yielding inconsistent or inaccurate results. To this end, we propose MV-RAG, a novel text-to-3D pipeline that first retrieves relevant 2D images from a large in-the-wild 2D database and then conditions a multiview diffusion model on these images to synthesize consistent and accurate multiview outputs. Training such a retrieval-conditioned model is achieved via a novel hybrid strategy bridging structured multiview data and diverse 2D image collections. This involves training on multiview data using augmented conditioning views that simulate retrieval variance for view-specific reconstruction, alongside training on sets of retrieved real-world 2D images using a distinctive held-out view prediction objective: the model predicts the held-out view from the other views to infer 3D consistency from 2D data. To facilitate a rigorous OOD evaluation, we...
  </details>  
  Keywords: text-to-3d, image-to-3d  
- **[Text-Driven 3D Hand Motion Generation from Sign Language Data](https://arxiv.org/abs/2508.15902v1)**  
  Authors: Léore Bensabath, Mathis Petrovich, Gül Varol  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2508.15902v1.pdf)  
  <details><summary>Abstract</summary>

  Our goal is to train a generative model of 3D hand motions, conditioned on natural language descriptions specifying motion characteristics such as handshapes, locations, finger/hand/arm movements. To this end, we automatically build pairs of 3D hand motions and their associated textual labels with unprecedented scale. Specifically, we leverage a large-scale sign language video dataset, along with noisy pseudo-annotated sign categories, which we translate into hand motion descriptions via an LLM that utilizes a dictionary of sign attributes, as well as our complementary motion-script cues. This data enables training a text-conditioned hand motion diffusion model HandMDM, that is robust across domains such as unseen sign categories from the same sign language, but also signs from another sign language and non-sign hand movements. We contribute extensive experimental investigation of these scenarios and will make our trained models and data publicly available to support future research in this relatively new field.
  </details>  
- **[DriveSplat: Decoupled Driving Scene Reconstruction with Geometry-enhanced Partitioned Neural Gaussians](https://arxiv.org/abs/2508.15376v3)**  
  Authors: Cong Wang, Xianda Guo, Wenbo Xu, Wei Tian, Ruiqi Song, Chenming Zhang, Lingxi Li, Long Chen  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2508.15376v3.pdf)  
  <details><summary>Abstract</summary>

  In the realm of driving scenarios, the presence of rapidly moving vehicles, pedestrians in motion, and large-scale static backgrounds poses significant challenges for 3D scene reconstruction. Recent methods based on 3D Gaussian Splatting address the motion blur problem by decoupling dynamic and static components within the scene. However, these decoupling strategies overlook background optimization with adequate geometry relationships and rely solely on fitting each training view by adding Gaussians. Therefore, these models exhibit limited robustness in rendering novel views and lack an accurate geometric representation. To address the above issues, we introduce DriveSplat, a high-quality reconstruction method for driving scenarios based on neural Gaussian representations with dynamic-static decoupling. To better accommodate the predominantly linear motion patterns of driving viewpoints, a region-wise voxel initialization scheme is employed, which partitions the scene into near, middle, and far regions to enhance close-range detail representation. Deformable neural Gaussians are introduced to model non-rigid dynamic...
  </details>  
  Keywords: 3d scene reconstruction  
- **[Image-Conditioned 3D Gaussian Splat Quantization](https://arxiv.org/abs/2508.15372v2)**  
  Authors: Xinshuang Liu, Runfa Blark Li, Keito Suzuki, Truong Nguyen  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2508.15372v2.pdf)  
  <details><summary>Abstract</summary>

  3D Gaussian Splatting (3DGS) has attracted considerable attention for enabling high-quality real-time rendering. Although 3DGS compression methods have been proposed for deployment on storage-constrained devices, two limitations hinder archival use: (1) they compress medium-scale scenes only to the megabyte range, which remains impractical for large-scale scenes or extensive scene collections; and (2) they lack mechanisms to accommodate scene changes after long-term archival. To address these limitations, we propose an Image-Conditioned Gaussian Splat Quantizer (ICGS-Quantizer) that substantially enhances compression efficiency and provides adaptability to scene changes after archiving. ICGS-Quantizer improves quantization efficiency by jointly exploiting inter-Gaussian and inter-attribute correlations and by using shared codebooks across all training scenes, which are then fixed and applied to previously unseen test scenes, eliminating the overhead of per-scene codebooks. This approach effectively reduces the storage requirements for 3DGS to the kilobyte range while preserving visual fidelity. To enable adaptability to post-archival scene changes, ICGS-Quantizer conditions...
  </details>  
- **[Collaborative Multi-Modal Coding for High-Quality 3D Generation](https://arxiv.org/abs/2508.15228v1)**  
  Authors: Ziang Cao, Zhaoxi Chen, Liang Pan, Ziwei Liu  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2508.15228v1.pdf)  
  <details><summary>Abstract</summary>

  3D content inherently encompasses multi-modal characteristics and can be projected into different modalities (e.g., RGB images, RGBD, and point clouds). Each modality exhibits distinct advantages in 3D asset modeling: RGB images contain vivid 3D textures, whereas point clouds define fine-grained 3D geometries. However, most existing 3D-native generative architectures either operate predominantly within single-modality paradigms-thus overlooking the complementary benefits of multi-modality data-or restrict themselves to 3D structures, thereby limiting the scope of available training datasets. To holistically harness multi-modalities for 3D modeling, we present TriMM, the first feed-forward 3D-native generative model that learns from basic multi-modalities (e.g., RGB, RGBD, and point cloud). Specifically, 1) TriMM first introduces collaborative multi-modal coding, which integrates modality-specific features while preserving their unique representational strengths. 2) Furthermore, auxiliary 2D and 3D supervision are introduced to raise the robustness and performance of multi-modal coding. 3) Based on the embedded multi-modal code, TriMM employs a triplane latent diffusion...
  </details>  
- **[Fusing Monocular RGB Images with AIS Data to Create a 6D Pose Estimation Dataset for Marine Vessels](https://arxiv.org/abs/2508.14767v1)**  
  Authors: Fabian Holst, Emre Gülsoylu, Simone Frintrop  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2508.14767v1.pdf)  
  <details><summary>Abstract</summary>

  The paper presents a novel technique for creating a 6D pose estimation dataset for marine vessels by fusing monocular RGB images with Automatic Identification System (AIS) data. The proposed technique addresses the limitations of relying purely on AIS for location information, caused by issues like equipment reliability, data manipulation, and transmission delays. By combining vessel detections from monocular RGB images, obtained using an object detection network (YOLOX-X), with AIS messages, the technique generates 3D bounding boxes that represent the vessels' 6D poses, i.e. spatial and rotational dimensions. The paper evaluates different object detection models to locate vessels in image space. We also compare two transformation methods (homography and Perspective-n-Point) for aligning AIS data with image coordinates. The results of our work demonstrate that the Perspective-n-Point (PnP) method achieves a significantly lower projection error compared to homography-based approaches used before, and the YOLOX-X model achieves a mean Average Precision (mAP) of...
  </details>  
- **[GALA: Guided Attention with Language Alignment for Open Vocabulary Gaussian Splatting](https://arxiv.org/abs/2508.14278v2)**  
  Authors: Elena Alegret, Kunyi Li, Sen Wang, Siyun Liang, Michael Niemeyer, Stefano Gasperini, Nassir Navab, Federico Tombari  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2508.14278v2.pdf)  
  <details><summary>Abstract</summary>

  3D scene reconstruction and understanding have gained increasing popularity, yet existing methods still struggle to capture fine-grained, language-aware 3D representations from 2D images. In this paper, we present GALA, a novel framework for open-vocabulary 3D scene understanding with 3D Gaussian Splatting (3DGS). GALA distills a scene-specific 3D instance feature field via self-supervised contrastive learning. To extend to generalized language feature fields, we introduce the core contribution of GALA, a cross-attention module with two learnable codebooks that encode view-independent semantic embeddings. This design not only ensures intra-instance feature similarity but also supports seamless 2D and 3D open-vocabulary queries. It reduces memory consumption by avoiding per-Gaussian high-dimensional feature learning. Extensive experiments on real-world datasets demonstrate GALA's remarkable open-vocabulary performance on both 2D and 3D.
  </details>  
  Keywords: 3d scene reconstruction  
- **[OASIS: Real-Time Opti-Acoustic Sensing for Intervention Systems in Unstructured Environments](https://arxiv.org/abs/2508.12071v1)**  
  Authors: Amy Phung, Richard Camilli  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2508.12071v1.pdf)  
  <details><summary>Abstract</summary>

  High resolution underwater 3D scene reconstruction is crucial for various applications, including construction, infrastructure maintenance, monitoring, exploration, and scientific investigation. Prior work has leveraged the complementary sensing modalities of imaging sonars and optical cameras for opti-acoustic 3D scene reconstruction, demonstrating improved results over methods which rely solely on either sensor. However, while most existing approaches focus on offline reconstruction, real-time spatial awareness is essential for both autonomous and piloted underwater vehicle operations. This paper presents OASIS, an opti-acoustic fusion method that integrates data from optical images with voxel carving techniques to achieve real-time 3D reconstruction unstructured underwater workspaces. Our approach utilizes an "eye-in-hand" configuration, which leverages the dexterity of robotic manipulator arms to capture multiple workspace views across a short baseline. We validate OASIS through tank-based experiments and present qualitative and quantitative results that highlight its utility for underwater manipulation tasks.
  </details>  
  Keywords: 3d scene reconstruction  
- **[UniUGG: Unified 3D Understanding and Generation via Geometric-Semantic Encoding](https://arxiv.org/abs/2508.11952v2)**  
  Authors: Yueming Xu, Jiahui Zhang, Ze Huang, Yurui Chen, Yanpeng Zhou, Zhenyu Chen, Yu-Jie Yuan, Pengxiang Xia, Guowei Huang, Xinyue Cai, Zhongang Qi, Xingyue Quan, Jianye Hao, Hang Xu, Li Zhang  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2508.11952v2.pdf)  
  <details><summary>Abstract</summary>

  Despite the impressive progress on understanding and generating images shown by the recent unified architectures, the integration of 3D tasks remains challenging and largely unexplored. In this paper, we introduce UniUGG, the first unified understanding and generation framework for 3D modalities. Our unified framework employs an LLM to comprehend and decode sentences and 3D representations. At its core, we propose a spatial decoder leveraging a latent diffusion model to generate high-quality 3D representations. This allows for the generation and imagination of 3D scenes based on a reference image and an arbitrary view transformation, while remaining supports for spatial visual question answering (VQA) tasks. Additionally, we propose a geometric-semantic learning strategy to pretrain the vision encoder. This design jointly captures the input's semantic and geometric cues, enhancing both spatial understanding and generation. Extensive experimental results demonstrate the superiority of our method in visual representation, spatial understanding, and 3D generation. The source...
  </details>  
- **[CoreEditor: Consistent 3D Editing via Correspondence-constrained Diffusion](https://arxiv.org/abs/2508.11603v2)**  
  Authors: Zhe Zhu, Honghua Chen, Peng Li, Mingqiang Wei  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2508.11603v2.pdf)  
  <details><summary>Abstract</summary>

  Text-driven 3D editing seeks to modify 3D scenes according to textual descriptions, and most existing approaches tackle this by adapting pre-trained 2D image editors to multi-view inputs. However, without explicit control over multi-view information exchange, they often fail to maintain cross-view consistency, leading to insufficient edits and blurry details. We introduce CoreEditor, a novel framework for consistent text-to-3D editing. The key innovation is a correspondence-constrained attention mechanism that enforces precise interactions between pixels expected to remain consistent throughout the diffusion denoising process. Beyond relying solely on geometric alignment, we further incorporate semantic similarity estimated during denoising, enabling more reliable correspondence modeling and robust multi-view editing. In addition, we design a selective editing pipeline that allows users to choose preferred results from multiple candidates, offering greater flexibility and user control. Extensive experiments show that CoreEditor produces high-quality, 3D-consistent edits with sharper details, significantly outperforming prior methods.
  </details>  
  Keywords: text-to-3d  
- **[G-CUT3R: Guided 3D Reconstruction with Camera and Depth Prior Integration](https://arxiv.org/abs/2508.11379v2)**  
  Authors: Ramil Khafizov, Artem Komarichev, Ruslan Rakhimov, Peter Wonka, Evgeny Burnaev  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2508.11379v2.pdf)  
  <details><summary>Abstract</summary>

  We introduce G-CUT3R, a novel feed-forward approach for guided 3D scene reconstruction that enhances the CUT3R model by integrating prior information. Unlike existing feed-forward methods that rely solely on input images, our method leverages auxiliary data, such as depth, camera calibrations, or camera positions, commonly available in real-world scenarios. We propose a lightweight modification to CUT3R, incorporating a dedicated encoder for each modality to extract features, which are fused with RGB image tokens via zero convolution. This flexible design enables seamless integration of any combination of prior information during inference. Evaluated across multiple benchmarks, including 3D reconstruction and other multi-view tasks, our approach demonstrates significant performance improvements, showing its ability to effectively utilize available priors while maintaining compatibility with varying input modalities.
  </details>  
  Keywords: 3d scene reconstruction  
- **[Quantum Visual Fields with Neural Amplitude Encoding](https://arxiv.org/abs/2508.10900v1)**  
  Authors: Shuteng Wang, Christian Theobalt, Vladislav Golyanik  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2508.10900v1.pdf)  
  <details><summary>Abstract</summary>

  Quantum Implicit Neural Representations (QINRs) include components for learning and execution on gate-based quantum computers. While QINRs recently emerged as a promising new paradigm, many challenges concerning their architecture and ansatz design, the utility of quantum-mechanical properties, training efficiency and the interplay with classical modules remain. This paper advances the field by introducing a new type of QINR for 2D image and 3D geometric field learning, which we collectively refer to as Quantum Visual Field (QVF). QVF encodes classical data into quantum statevectors using neural amplitude encoding grounded in a learnable energy manifold, ensuring meaningful Hilbert space embeddings. Our ansatz follows a fully entangled design of learnable parametrised quantum circuits, with quantum (unitary) operations performed in the real Hilbert space, resulting in numerically stable training with fast convergence. QVF does not rely on classical post-processing -- in contrast to the previous QINR learning approach -- and directly employs projective measurement...
  </details>  
- **[TexVerse: A Universe of 3D Objects with High-Resolution Textures](https://arxiv.org/abs/2508.10868v2)**  
  Authors: Yibo Zhang, Li Zhang, Rui Ma, Nan Cao  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2508.10868v2.pdf)  
  <details><summary>Abstract</summary>

  We introduce TexVerse, a large-scale 3D dataset featuring high-resolution textures. While recent advances in large-scale 3D datasets have enhanced high-resolution geometry generation, creating high-resolution textures end-to-end remains underexplored due to the lack of suitable datasets. TexVerse fills this gap with a curated collection of over 858K unique high-resolution 3D models sourced from Sketchfab, including more than 158K models with physically based rendering (PBR) materials. Each model encompasses all of its high-resolution variants, bringing the total to 1.6M 3D instances. TexVerse also includes specialized subsets: TexVerse-Skeleton, with 69K rigged models, and TexVerse-Animation, with 54K animated models, both preserving original skeleton and animation data uploaded by the user. We also provide detailed model annotations describing overall characteristics, structural components, and intricate features. TexVerse offers a high-quality data resource with wide-ranging potential applications in texture synthesis, PBR material development, animation, and various 3D vision and graphics tasks.
  </details>  
  Keywords: texture synthesis  
- **[Enhancing Monocular 3D Hand Reconstruction with Learned Texture Priors](https://arxiv.org/abs/2508.09629v1)**  
  Authors: Giorgos Karvounas, Nikolaos Kyriazis, Iason Oikonomidis, Georgios Pavlakos, Antonis A. Argyros  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2508.09629v1.pdf)  
  <details><summary>Abstract</summary>

  We revisit the role of texture in monocular 3D hand reconstruction, not as an afterthought for photorealism, but as a dense, spatially grounded cue that can actively support pose and shape estimation. Our observation is simple: even in high-performing models, the overlay between predicted hand geometry and image appearance is often imperfect, suggesting that texture alignment may be an underused supervisory signal. We propose a lightweight texture module that embeds per-pixel observations into UV texture space and enables a novel dense alignment loss between predicted and observed hand appearances. Our approach assumes access to a differentiable rendering pipeline and a model that maps images to 3D hand meshes with known topology, allowing us to back-project a textured hand onto the image and perform pixel-based alignment. The module is self-contained and easily pluggable into existing reconstruction pipelines. To isolate and highlight the value of texture-guided supervision, we augment HaMeR, a high-performing...
  </details>  
- **[Deep Spectral Epipolar Representations for Dense Light Field Reconstruction](https://arxiv.org/abs/2508.08900v2)**  
  Authors: Noor Islam S. Mohammad  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2508.08900v2.pdf)  
  <details><summary>Abstract</summary>

  Accurate and efficient dense depth reconstruction from light field imagery remains a central challenge in computer vision, underpinning applications such as augmented reality, biomedical imaging, and 3D scene reconstruction. Existing deep convolutional approaches, while effective, often incur high computational overhead and are sensitive to noise and disparity inconsistencies in real-world scenarios. This paper introduces a novel Deep Spectral Epipolar Representation (DSER) framework for dense light field reconstruction, which unifies deep spectral feature learning with epipolar-domain regularization. The proposed approach exploits frequency-domain correlations across epipolar plane images to enforce global structural coherence, thereby mitigating artifacts and enhancing depth accuracy. Unlike conventional supervised models, DSER operates efficiently with limited training data while maintaining high reconstruction fidelity. Comprehensive experiments on the 4D Light Field Benchmark and a diverse set of real-world datasets demonstrate that DSER achieves superior performance in terms of precision, structural consistency, and computational efficiency compared to state-of-the-art methods. These results...
  </details>  
  Keywords: 3d scene reconstruction  
- **[DiffPhysCam: Differentiable Physics-Based Camera Simulation for Inverse Rendering and Embodied AI](https://arxiv.org/abs/2508.08831v1)**  
  Authors: Bo-Hsun Chen, Nevindu M. Batagoda, Dan Negrut  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2508.08831v1.pdf)  
  <details><summary>Abstract</summary>

  We introduce DiffPhysCam, a differentiable camera simulator designed to support robotics and embodied AI applications by enabling gradient-based optimization in visual perception pipelines. Generating synthetic images that closely mimic those from real cameras is essential for training visual models and enabling end-to-end visuomotor learning. Moreover, differentiable rendering allows inverse reconstruction of real-world scenes as digital twins, facilitating simulation-based robotics training. However, existing virtual cameras offer limited control over intrinsic settings, poorly capture optical artifacts, and lack tunable calibration parameters -- hindering sim-to-real transfer. DiffPhysCam addresses these limitations through a multi-stage pipeline that provides fine-grained control over camera settings, models key optical effects such as defocus blur, and supports calibration with real-world data. It enables both forward rendering for image synthesis and inverse rendering for 3D scene reconstruction, including mesh and material texture optimization. We show that DiffPhysCam enhances robotic perception performance in synthetic image tasks. As an illustrative example, we...
  </details>  
  Keywords: 3d scene reconstruction  
- **[Empowering Children to Create AI-Enabled Augmented Reality Experiences](https://arxiv.org/abs/2508.08467v1)**  
  Authors: Lei Zhang, Shuyao Zhou, Amna Liaqat, Tinney Mak, Brian Berengard, Emily Qian, Andrés Monroy-Hernández  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2508.08467v1.pdf)  
  <details><summary>Abstract</summary>

  Despite their potential to enhance children's learning experiences, AI-enabled AR technologies are predominantly used in ways that position children as consumers rather than creators. We introduce Capybara, an AR-based and AI-powered visual programming environment that empowers children to create, customize, and program 3D characters overlaid onto the physical world. Capybara enables children to create virtual characters and accessories using text-to-3D generative AI models, and to animate these characters through auto-rigging and body tracking. In addition, our system employs vision-based AI models to recognize physical objects, allowing children to program interactive behaviors between virtual characters and their physical surroundings. We demonstrate the expressiveness of Capybara through a set of novel AR experiences. We conducted user studies with 20 children in the United States and Argentina. Our findings suggest that Capybara can empower children to harness AI in authoring personalized and engaging AR experiences that seamlessly bridge the virtual and physical worlds.
  </details>  
  Keywords: text-to-3d, rigging  
- **[Enhanced Liver Tumor Detection in CT Images Using 3D U-Net and Bat Algorithm for Hyperparameter Optimization](https://arxiv.org/abs/2508.08452v1)**  
  Authors: Nastaran Ghorbani, Bitasadat Jamshidi, Mohsen Rostamy-Malkhalifeh  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2508.08452v1.pdf)  
  <details><summary>Abstract</summary>

  Liver cancer is one of the most prevalent and lethal forms of cancer, making early detection crucial for effective treatment. This paper introduces a novel approach for automated liver tumor segmentation in computed tomography (CT) images by integrating a 3D U-Net architecture with the Bat Algorithm for hyperparameter optimization. The method enhances segmentation accuracy and robustness by intelligently optimizing key parameters like the learning rate and batch size. Evaluated on a publicly available dataset, our model demonstrates a strong ability to balance precision and recall, with a high F1-score at lower prediction thresholds. This is particularly valuable for clinical diagnostics, where ensuring no potential tumors are missed is paramount. Our work contributes to the field of medical image analysis by demonstrating that the synergy between a robust deep learning architecture and a metaheuristic optimization algorithm can yield a highly effective solution for complex segmentation tasks.
  </details>  
- **[3D Human Mesh Estimation from Single View RGBD](https://arxiv.org/abs/2508.08178v2)**  
  Authors: Ozhan Suat, Bedirhan Uguz, Batuhan Karagoz, Muhammed Can Keles, Emre Akbas  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2508.08178v2.pdf)  
  <details><summary>Abstract</summary>

  Despite significant progress in 3D human mesh estimation from RGB images; RGBD cameras, offering additional depth data, remain underutilized. In this paper, we present a method for accurate 3D human mesh estimation from a single RGBD view, leveraging the affordability and widespread adoption of RGBD cameras for real-world applications. A fully supervised approach for this problem, requires a dataset with RGBD image and 3D mesh label pairs. However, collecting such a dataset is costly and challenging, hence, existing datasets are small, and limited in pose and shape diversity. To overcome this data scarcity, we leverage existing Motion Capture (MoCap) datasets. We first obtain complete 3D meshes from the body models found in MoCap datasets, and create partial, single-view versions of them by projection to a virtual camera. This simulates the depth data provided by an RGBD camera from a single viewpoint. Then, we train a masked autoencoder to complete the...
  </details>  
- **[Matrix-3D: Omnidirectional Explorable 3D World Generation](https://arxiv.org/abs/2508.08086v1)**  
  Authors: Zhongqi Yang, Wenhang Ge, Yuqi Li, Jiaqi Chen, Haoyuan Li, Mengyin An, Fei Kang, Hua Xue, Baixin Xu, Yuyang Yin, Eric Li, Yang Liu, Yikai Wang, Hao-Xiang Guo, Yahui Zhou  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2508.08086v1.pdf) | [![Project](https://img.shields.io/badge/-Project-blue)](https://matrix-3d.github.io)  
  <details><summary>Abstract</summary>

  Explorable 3D world generation from a single image or text prompt forms a cornerstone of spatial intelligence. Recent works utilize video model to achieve wide-scope and generalizable 3D world generation. However, existing approaches often suffer from a limited scope in the generated scenes. In this work, we propose Matrix-3D, a framework that utilize panoramic representation for wide-coverage omnidirectional explorable 3D world generation that combines conditional video generation and panoramic 3D reconstruction. We first train a trajectory-guided panoramic video diffusion model that employs scene mesh renders as condition, to enable high-quality and geometrically consistent scene video generation. To lift the panorama scene video to 3D world, we propose two separate methods: (1) a feed-forward large panorama reconstruction model for rapid 3D scene reconstruction and (2) an optimization-based pipeline for accurate and detailed 3D scene reconstruction. To facilitate effective training, we also introduce the Matrix-Pano dataset, the first large-scale synthetic collection comprising...
  </details>  
  Keywords: 3d scene reconstruction  
- **[Vertex Features for Neural Global Illumination](https://arxiv.org/abs/2508.07852v1)**  
  Authors: Rui Su, Honghao Dong, Haojie Jin, Yisong Chen, Guoping Wang, Sheng Li  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2508.07852v1.pdf)  
  <details><summary>Abstract</summary>

  Recent research on learnable neural representations has been widely adopted in the field of 3D scene reconstruction and neural rendering applications. However, traditional feature grid representations often suffer from substantial memory footprint, posing a significant bottleneck for modern parallel computing hardware. In this paper, we present neural vertex features, a generalized formulation of learnable representation for neural rendering tasks involving explicit mesh surfaces. Instead of uniformly distributing neural features throughout 3D space, our method stores learnable features directly at mesh vertices, leveraging the underlying geometry as a compact and structured representation for neural processing. This not only optimizes memory efficiency, but also improves feature representation by aligning compactly with the surface using task-specific geometric priors. We validate our neural representation across diverse neural rendering tasks, with a specific emphasis on neural radiosity. Experimental results demonstrate that our method reduces memory consumption to only one-fifth (or even less) of grid-based representations,...
  </details>  
  Keywords: 3d scene reconstruction  
- **[Touch-Augmented Gaussian Splatting for Enhanced 3D Scene Reconstruction](https://arxiv.org/abs/2508.07717v1)**  
  Authors: Yuchen Gao, Xiao Xu, Eckehard Steinbach, Daniel E. Lucani, Qi Zhang  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2508.07717v1.pdf)  
  <details><summary>Abstract</summary>

  This paper presents a multimodal framework that integrates touch signals (contact points and surface normals) into 3D Gaussian Splatting (3DGS). Our approach enhances scene reconstruction, particularly under challenging conditions like low lighting, limited camera viewpoints, and occlusions. Different from the visual-only method, the proposed approach incorporates spatially selective touch measurements to refine both the geometry and appearance of the 3D Gaussian representation. To guide the touch exploration, we introduce a two-stage sampling scheme that initially probes sparse regions and then concentrates on high-uncertainty boundaries identified from the reconstructed mesh. A geometric loss is proposed to ensure surface smoothness, resulting in improved geometry. Experimental results across diverse scenarios show consistent improvements in geometric accuracy. In the most challenging case with severe occlusion, the Chamfer Distance is reduced by over 15x, demonstrating the effectiveness of integrating touch cues into 3D Gaussian Splatting. Furthermore, our approach maintains a fully online pipeline, underscoring its...
  </details>  
  Keywords: 3d scene reconstruction  
- **[Make Your MoVe: Make Your 3D Contents by Adapting Multi-View Diffusion Models to External Editing](https://arxiv.org/abs/2508.07700v1)**  
  Authors: Weitao Wang, Haoran Xu, Jun Meng, Haoqian Wang  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2508.07700v1.pdf)  
  <details><summary>Abstract</summary>

  As 3D generation techniques continue to flourish, the demand for generating personalized content is rapidly rising. Users increasingly seek to apply various editing methods to polish generated 3D content, aiming to enhance its color, style, and lighting without compromising the underlying geometry. However, most existing editing tools focus on the 2D domain, and directly feeding their results into 3D generation methods (like multi-view diffusion models) will introduce information loss, degrading the quality of the final 3D assets. In this paper, we propose a tuning-free, plug-and-play scheme that aligns edited assets with their original geometry in a single inference run. Central to our approach is a geometry preservation module that guides the edited multi-view generation with original input normal latents. Besides, an injection switcher is proposed to deliberately control the supervision extent of the original normals, ensuring the alignment between the edited color and normal views. Extensive experiments show that our...
  </details>  
- **[DIP-GS: Deep Image Prior For Gaussian Splatting Sparse View Recovery](https://arxiv.org/abs/2508.07372v1)**  
  Authors: Rajaei Khatib, Raja Giryes  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2508.07372v1.pdf)  
  <details><summary>Abstract</summary>

  3D Gaussian Splatting (3DGS) is a leading 3D scene reconstruction method, obtaining high-quality reconstruction with real-time rendering runtime performance. The main idea behind 3DGS is to represent the scene as a collection of 3D gaussians, while learning their parameters to fit the given views of the scene. While achieving superior performance in the presence of many views, 3DGS struggles with sparse view reconstruction, where the input views are sparse and do not fully cover the scene and have low overlaps. In this paper, we propose DIP-GS, a Deep Image Prior (DIP) 3DGS representation. By using the DIP prior, which utilizes internal structure and patterns, with coarse-to-fine manner, DIP-based 3DGS can operate in scenarios where vanilla 3DGS fails, such as sparse view recovery. Note that our approach does not use any pre-trained models such as generative models and depth estimation, but rather relies only on the input frames. Among such methods,...
  </details>  
  Keywords: 3d scene reconstruction  
- **[UW-3DGS: Underwater 3D Reconstruction with Physics-Aware Gaussian Splatting](https://arxiv.org/abs/2508.06169v1)**  
  Authors: Wenpeng Xing, Jie Chen, Zaifeng Yang, Changting Lin, Jianfeng Dong, Chaochao Chen, Xun Zhou, Meng Han  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2508.06169v1.pdf)  
  <details><summary>Abstract</summary>

  Underwater 3D scene reconstruction faces severe challenges from light absorption, scattering, and turbidity, which degrade geometry and color fidelity in traditional methods like Neural Radiance Fields (NeRF). While NeRF extensions such as SeaThru-NeRF incorporate physics-based models, their MLP reliance limits efficiency and spatial resolution in hazy environments. We introduce UW-3DGS, a novel framework adapting 3D Gaussian Splatting (3DGS) for robust underwater reconstruction. Key innovations include: (1) a plug-and-play learnable underwater image formation module using voxel-based regression for spatially varying attenuation and backscatter; and (2) a Physics-Aware Uncertainty Pruning (PAUP) branch that adaptively removes noisy floating Gaussians via uncertainty scoring, ensuring artifact-free geometry. The pipeline operates in training and rendering stages. During training, noisy Gaussians are optimized end-to-end with underwater parameters, guided by PAUP pruning and scattering modeling. In rendering, refined Gaussians produce clean Unattenuated Radiance Images (URIs) free from media effects, while learned physics enable realistic Underwater Images (UWIs) with...
  </details>  
  Keywords: 3d scene reconstruction  
- **[Learning 3D Texture-Aware Representations for Parsing Diverse Human Clothing and Body Parts](https://arxiv.org/abs/2508.06032v2)**  
  Authors: Kiran Chhatre, Christopher Peters, Srikrishna Karanam  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2508.06032v2.pdf)  
  <details><summary>Abstract</summary>

  Existing methods for human parsing into body parts and clothing often use fixed mask categories with broad labels that obscure fine-grained clothing types. Recent open-vocabulary segmentation approaches leverage pretrained text-to-image (T2I) diffusion model features for strong zero-shot transfer, but typically group entire humans into a single person category, failing to distinguish diverse clothing or detailed body parts. To address this, we propose Spectrum, a unified network for part-level pixel parsing (body parts and clothing) and instance-level grouping. While diffusion-based open-vocabulary models generalize well across tasks, their internal representations are not specialized for detailed human parsing. We observe that, unlike diffusion models with broad representations, image-driven 3D texture generators maintain faithful correspondence to input images, enabling stronger representations for parsing diverse clothing and body parts. Spectrum introduces a novel repurposing of an Image-to-Texture (I2Tx) diffusion model (obtained by fine-tuning a T2I model on 3D human texture maps) for improved alignment with...
  </details>  
- **[ExploreGS: Explorable 3D Scene Reconstruction with Virtual Camera Samplings and Diffusion Priors](https://arxiv.org/abs/2508.06014v1)**  
  Authors: Minsu Kim, Subin Jeon, In Cho, Mijin Yoo, Seon Joo Kim  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2508.06014v1.pdf) | [![Project](https://img.shields.io/badge/-Project-blue)](https://exploregs.github.io)  
  <details><summary>Abstract</summary>

  Recent advances in novel view synthesis (NVS) have enabled real-time rendering with 3D Gaussian Splatting (3DGS). However, existing methods struggle with artifacts and missing regions when rendering from viewpoints that deviate from the training trajectory, limiting seamless scene exploration. To address this, we propose a 3DGS-based pipeline that generates additional training views to enhance reconstruction. We introduce an information-gain-driven virtual camera placement strategy to maximize scene coverage, followed by video diffusion priors to refine rendered results. Fine-tuning 3D Gaussians with these enhanced views significantly improves reconstruction quality. To evaluate our method, we present Wild-Explore, a benchmark designed for challenging scene exploration. Experiments demonstrate that our approach outperforms existing 3DGS-based methods, enabling high-quality, artifact-free rendering from arbitrary viewpoints.   https://exploregs.github.io
  </details>  
  Keywords: novel view synthesis, 3d scene reconstruction  
- **[AnimateScene: Camera-controllable Animation in Any Scene](https://arxiv.org/abs/2508.05982v1)**  
  Authors: Qingyang Liu, Bingjie Gao, Weiheng Huang, Jun Zhang, Zhongqian Sun, Yang Wei, Zelin Peng, Qianli Ma, Shuai Yang, Zhaohe Liao, Haonan Zhao, Li Niu  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2508.05982v1.pdf)  
  <details><summary>Abstract</summary>

  3D scene reconstruction and 4D human animation have seen rapid progress and broad adoption in recent years. However, seamlessly integrating reconstructed scenes with 4D human animation to produce visually engaging results remains challenging. One key difficulty lies in placing the human at the correct location and scale within the scene while avoiding unrealistic interpenetration. Another challenge is that the human and the background may exhibit different lighting and style, leading to unrealistic composites. In addition, appealing character motion videos are often accompanied by camera movements, which means that the viewpoints need to be reconstructed along a specified trajectory. We present AnimateScene, which addresses the above issues in a unified framework. First, we design an accurate placement module that automatically determines a plausible 3D position for the human and prevents any interpenetration within the scene during motion. Second, we propose a training-free style alignment method that adapts the 4D human representation...
  </details>  
  Keywords: 3d scene reconstruction  
- **[HOLODECK 2.0: Vision-Language-Guided 3D World Generation with Editing](https://arxiv.org/abs/2508.05899v2)**  
  Authors: Zixuan Bian, Ruohan Ren, Yue Yang, Chris Callison-Burch  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2508.05899v2.pdf)  
  <details><summary>Abstract</summary>

  3D scene generation plays a crucial role in gaming, artistic creation, virtual reality, and many other domains. However, current 3D scene design still relies heavily on extensive manual effort from creators, and existing automated methods struggle to generate open-domain scenes or support flexible editing. To address those challenges, we introduce HOLODECK 2.0, an advanced vision-language-guided framework for 3D world generation with support for interactive scene editing based on human feedback. HOLODECK 2.0 can generate diverse and stylistically rich 3D scenes (e.g., realistic, cartoon, anime, and cyberpunk styles) that exhibit high semantic fidelity to fine-grained input descriptions, suitable for both indoor and open-domain environments. HOLODECK 2.0 leverages vision-language models (VLMs) to identify and parse the objects required in a scene and generates corresponding high-quality assets via state-of-the-art 3D generative models. Then, HOLODECK 2.0 iteratively applies spatial constraints derived from the VLMs to achieve semantically coherent and physically plausible layouts. Both human...
  </details>  
- **[Progress and new challenges in image-based profiling](https://arxiv.org/abs/2508.05800v2)**  
  Authors: Erik Serrano, John Peters, Jesko Wagner, Rebecca E. Graham, Zhenghao Chen, Brian Feng, Gisele Miranda, Alexandr A. Kalinin, Loan Vulliard, Jenna Tomkinson, Cameron Mattson, Michael J. Lippincott, Ziqi Kang, Divya Sitani, Dave Bunten, Srijit Seal, Neil O. Carragher, Anne E. Carpenter, Shantanu Singh, Paula A. Marin Zapata, Juan C. Caicedo, Gregory P. Way  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2508.05800v2.pdf)  
  <details><summary>Abstract</summary>

  For over two decades, image-based profiling has revolutionized cell phenotype analysis. Image-based profiling processes rich, high-throughput, microscopy data into thousands of unbiased measurements that reveal phenotypic patterns powerful for drug discovery, functional genomics, and cell state classification. Here, we review the evolving computational landscape of image-based profiling, detailing the bioinformatics processes involved from feature extraction to normalization and batch correction. We discuss how deep learning has fundamentally reshaped the field. We examine key methodological advancements, such as single-cell analysis, the development of robust similarity metrics, and the expansion into new modalities like optical pooled screening, temporal imaging, and 3D organoid profiling. We also highlight the growth of public benchmarks and open-source software ecosystems as a key driver for fostering reproducibility and collaboration. Despite these advances, the field still faces substantial challenges, particularly in developing methods for emerging temporal and 3D data modalities, establishing robust quality control standards and workflows, and...
  </details>  
- **[Hi3DEval: Advancing 3D Generation Evaluation with Hierarchical Validity](https://arxiv.org/abs/2508.05609v1)**  
  Authors: Yuhan Zhang, Long Zhuo, Ziyang Chu, Tong Wu, Zhibing Li, Liang Pan, Dahua Lin, Ziwei Liu  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2508.05609v1.pdf) | [![Project](https://img.shields.io/badge/-Project-blue)](https://zyh482.github.io/Hi3DEval)  
  <details><summary>Abstract</summary>

  Despite rapid advances in 3D content generation, quality assessment for the generated 3D assets remains challenging. Existing methods mainly rely on image-based metrics and operate solely at the object level, limiting their ability to capture spatial coherence, material authenticity, and high-fidelity local details. 1) To address these challenges, we introduce Hi3DEval, a hierarchical evaluation framework tailored for 3D generative content. It combines both object-level and part-level evaluation, enabling holistic assessments across multiple dimensions as well as fine-grained quality analysis. Additionally, we extend texture evaluation beyond aesthetic appearance by explicitly assessing material realism, focusing on attributes such as albedo, saturation, and metallicness. 2) To support this framework, we construct Hi3DBench, a large-scale dataset comprising diverse 3D assets and high-quality annotations, accompanied by a reliable multi-agent annotation pipeline. We further propose a 3D-aware automated scoring system based on hybrid 3D representations. Specifically, we leverage video-based representations for object-level and material-subject evaluations to...
  </details>  
- **[DualMat: PBR Material Estimation via Coherent Dual-Path Diffusion](https://arxiv.org/abs/2508.05060v1)**  
  Authors: Yifeng Huang, Zhang Chen, Yi Xu, Minh Hoai, Zhong Li  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2508.05060v1.pdf)  
  <details><summary>Abstract</summary>

  We present DualMat, a novel dual-path diffusion framework for estimating Physically Based Rendering (PBR) materials from single images under complex lighting conditions. Our approach operates in two distinct latent spaces: an albedo-optimized path leveraging pretrained visual knowledge through RGB latent space, and a material-specialized path operating in a compact latent space designed for precise metallic and roughness estimation. To ensure coherent predictions between the albedo-optimized and material-specialized paths, we introduce feature distillation during training. We employ rectified flow to enhance efficiency by reducing inference steps while maintaining quality. Our framework extends to high-resolution and multi-view inputs through patch-based estimation and cross-view attention, enabling seamless integration into image-to-3D pipelines. DualMat achieves state-of-the-art performance on both Objaverse and real-world data, significantly outperforming existing methods with up to 28% improvement in albedo estimation and 39% reduction in metallic-roughness prediction errors.
  </details>  
  Keywords: image-to-3d  
- **[Deep Learning-based Scalable Image-to-3D Facade Parser for Generating Thermal 3D Building Models](https://arxiv.org/abs/2508.04406v1)**  
  Authors: Yinan Yu, Alex Gonzalez-Caceres, Samuel Scheidegger, Sanjay Somanath, Alexander Hollberg  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2508.04406v1.pdf)  
  <details><summary>Abstract</summary>

  Renovating existing buildings is essential for climate impact. Early-phase renovation planning requires simulations based on thermal 3D models at Level of Detail (LoD) 3, which include features like windows. However, scalable and accurate identification of such features remains a challenge. This paper presents the Scalable Image-to-3D Facade Parser (SI3FP), a pipeline that generates LoD3 thermal models by extracting geometries from images using both computer vision and deep learning. Unlike existing methods relying on segmentation and projection, SI3FP directly models geometric primitives in the orthographic image plane, providing a unified interface while reducing perspective distortions. SI3FP supports both sparse (e.g., Google Street View) and dense (e.g., hand-held camera) data sources. Tested on typical Swedish residential buildings, SI3FP achieved approximately 5% error in window-to-wall ratio estimates, demonstrating sufficient accuracy for early-stage renovation analysis. The pipeline facilitates large-scale energy renovation planning and has broader applications in urban development and planning.
  </details>  
  Keywords: image-to-3d  
- **[PIS3R: Very Large Parallax Image Stitching via Deep 3D Reconstruction](https://arxiv.org/abs/2508.04236v3)**  
  Authors: Muhua Zhu, Xinhao Jin, Chengbo Wang, Yongcong Zhang, Yifei Xue, Tie Ji, Yizhen Lao  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2508.04236v3.pdf)  
  <details><summary>Abstract</summary>

  Image stitching aim to align two images taken from different viewpoints into one seamless, wider image. However, when the 3D scene contains depth variations and the camera baseline is significant, noticeable parallax occurs-meaning the relative positions of scene elements differ substantially between views. Most existing stitching methods struggle to handle such images with large parallax effectively. To address this challenge, in this paper, we propose an image stitching solution called PIS3R that is robust to very large parallax based on the novel concept of deep 3D reconstruction. First, we apply visual geometry grounded transformer to two input images with very large parallax to obtain both intrinsic and extrinsic parameters, as well as the dense 3D scene reconstruction. Subsequently, we reproject reconstructed dense point cloud onto a designated reference view using the recovered camera parameters, achieving pixel-wise alignment and generating an initial stitched image. Finally, to further address potential artifacts such...
  </details>  
  Keywords: 3d scene reconstruction  
- **[IDCNet: Guided Video Diffusion for Metric-Consistent RGBD Scene Generation with Precise Camera Control](https://arxiv.org/abs/2508.04147v1)**  
  Authors: Lijuan Liu, Wenfa Li, Dongbo Zhang, Shuo Wang, Shaohui Jiao  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2508.04147v1.pdf) | [![Project](https://img.shields.io/badge/-Project-blue)](https://idcnet-scene.github.io)  
  <details><summary>Abstract</summary>

  We present IDC-Net (Image-Depth Consistency Network), a novel framework designed to generate RGB-D video sequences under explicit camera trajectory control. Unlike approaches that treat RGB and depth generation separately, IDC-Net jointly synthesizes both RGB images and corresponding depth maps within a unified geometry-aware diffusion model. The joint learning framework strengthens spatial and geometric alignment across frames, enabling more precise camera control in the generated sequences. To support the training of this camera-conditioned model and ensure high geometric fidelity, we construct a camera-image-depth consistent dataset with metric-aligned RGB videos, depth maps, and accurate camera poses, which provides precise geometric supervision with notably improved inter-frame geometric consistency. Moreover, we introduce a geometry-aware transformer block that enables fine-grained camera control, enhancing control over the generated sequences. Extensive experiments show that IDC-Net achieves improvements over state-of-the-art approaches in both visual quality and geometric consistency of generated scene sequences. Notably, the generated RGB-D sequences can...
  </details>  
  Keywords: 3d scene reconstruction  
- **[A Foundational Multi-Modal Model for Few-Shot Learning](https://arxiv.org/abs/2508.04746v1)**  
  Authors: Pengtao Dang, Tingbo Guo, Sha Cao, Chi Zhang  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2508.04746v1.pdf) | [![GitHub](https://img.shields.io/github/stars/ptdang1001/M3F?style=social)](https://github.com/ptdang1001/M3F)  
  <details><summary>Abstract</summary>

  Few-shot learning (FSL) is a machine learning paradigm that aims to generalize models from a small number of labeled examples, typically fewer than 10 per class. FSL is particularly crucial in biomedical, environmental, materials, and mechanical sciences, where samples are limited and data collection is often prohibitively costly, time-consuming, or ethically constrained. In this study, we present an innovative approach to FSL by demonstrating that a Large Multi-Modal Model (LMMM), trained on a set of independent tasks spanning diverse domains, task types, and input modalities, can substantially improve the generalization of FSL models, outperforming models based on conventional meta-learning on tasks of the same type. To support this, we first constructed a Multi-Modal Model Few-shot Dataset (M3FD, over 10K+ few-shot samples), which includes 2D RGB images, 2D/3D medical scans, tabular and time-course datasets, from which we manually curated FSL tasks such as classification. We further introduced M3F (Multi-Modal Model for...
  </details>  
- **[Uni3R: Unified 3D Reconstruction and Semantic Understanding via Generalizable Gaussian Splatting from Unposed Multi-View Images](https://arxiv.org/abs/2508.03643v3)**  
  Authors: Xiangyu Sun, Haoyi Jiang, Liu Liu, Seungtae Nam, Gyeongjin Kang, Xinjie Wang, Wei Sui, Zhizhong Su, Wenyu Liu, Xinggang Wang, Eunbyung Park  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2508.03643v3.pdf) | [![GitHub](https://img.shields.io/github/stars/HorizonRobotics/Uni3R?style=social)](https://github.com/HorizonRobotics/Uni3R)  
  <details><summary>Abstract</summary>

  Reconstructing and semantically interpreting 3D scenes from sparse 2D views remains a fundamental challenge in computer vision. Conventional methods often decouple semantic understanding from reconstruction or necessitate costly per-scene optimization, thereby restricting their scalability and generalizability. In this paper, we introduce Uni3R, a novel feed-forward framework that jointly reconstructs a unified 3D scene representation enriched with open-vocabulary semantics, directly from unposed multi-view images. Our approach leverages a Cross-View Transformer to robustly integrate information across arbitrary multi-view inputs, which then regresses a set of 3D Gaussian primitives endowed with semantic feature fields. This unified representation facilitates high-fidelity novel view synthesis, open-vocabulary 3D semantic segmentation, and depth prediction, all within a single, feed-forward pass. Extensive experiments demonstrate that Uni3R establishes a new state-of-the-art across multiple benchmarks, including 25.07 PSNR on RE10K and 55.84 mIoU on ScanNet. Our work signifies a novel paradigm towards generalizable, unified 3D scene reconstruction and understanding. The code...
  </details>  
  Keywords: novel view synthesis, 3d scene reconstruction  
- **[Unified Category-Level Object Detection and Pose Estimation from RGB Images using 3D Prototypes](https://arxiv.org/abs/2508.02157v1)**  
  Authors: Tom Fischer, Xiaojie Zhang, Eddy Ilg  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2508.02157v1.pdf) | [![GitHub](https://img.shields.io/github/stars/Fischer-Tom/unified-detection-and-pose-estimation?style=social)](https://github.com/Fischer-Tom/unified-detection-and-pose-estimation)  
  <details><summary>Abstract</summary>

  Recognizing objects in images is a fundamental problem in computer vision. Although detecting objects in 2D images is common, many applications require determining their pose in 3D space. Traditional category-level methods rely on RGB-D inputs, which may not always be available, or employ two-stage approaches that use separate models and representations for detection and pose estimation. For the first time, we introduce a unified model that integrates detection and pose estimation into a single framework for RGB images by leveraging neural mesh models with learned features and multi-model RANSAC. Our approach achieves state-of-the-art results for RGB category-level pose estimation on REAL275, improving on the current state-of-the-art by 22.9% averaged across all scale-agnostic metrics. Finally, we demonstrate that our unified method exhibits greater robustness compared to single-stage baselines. Our code and models are available at https://github.com/Fischer-Tom/unified-detection-and-pose-estimation.
  </details>  
- **[DisCo3D: Distilling Multi-View Consistency for 3D Scene Editing](https://arxiv.org/abs/2508.01684v1)**  
  Authors: Yufeng Chi, Huimin Ma, Kafeng Wang, Jianmin Li  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2508.01684v1.pdf)  
  <details><summary>Abstract</summary>

  While diffusion models have demonstrated remarkable progress in 2D image generation and editing, extending these capabilities to 3D editing remains challenging, particularly in maintaining multi-view consistency. Classical approaches typically update 3D representations through iterative refinement based on a single editing view. However, these methods often suffer from slow convergence and blurry artifacts caused by cross-view inconsistencies. Recent methods improve efficiency by propagating 2D editing attention features, yet still exhibit fine-grained inconsistencies and failure modes in complex scenes due to insufficient constraints. To address this, we propose \textbf{DisCo3D}, a novel framework that distills 3D consistency priors into a 2D editor. Our method first fine-tunes a 3D generator using multi-view inputs for scene adaptation, then trains a 2D editor through consistency distillation. The edited multi-view outputs are finally optimized into 3D representations via Gaussian Splatting. Experimental results show DisCo3D achieves stable multi-view consistency and outperforms state-of-the-art methods in editing quality.
  </details>  
  Keywords: 3d generator  
- **[Can3Tok: Canonical 3D Tokenization and Latent Modeling of Scene-Level 3D Gaussians](https://arxiv.org/abs/2508.01464v1)**  
  Authors: Quankai Gao, Iliyan Georgiev, Tuanfeng Y. Wang, Krishna Kumar Singh, Ulrich Neumann, Jae Shin Yoon  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2508.01464v1.pdf)  
  <details><summary>Abstract</summary>

  3D generation has made significant progress, however, it still largely remains at the object-level. Feedforward 3D scene-level generation has been rarely explored due to the lack of models capable of scaling-up latent representation learning on 3D scene-level data. Unlike object-level generative models, which are trained on well-labeled 3D data in a bounded canonical space, scene-level generations with 3D scenes represented by 3D Gaussian Splatting (3DGS) are unbounded and exhibit scale inconsistency across different scenes, making unified latent representation learning for generative purposes extremely challenging. In this paper, we introduce Can3Tok, the first 3D scene-level variational autoencoder (VAE) capable of encoding a large number of Gaussian primitives into a low-dimensional latent embedding, which effectively captures both semantic and spatial information of the inputs. Beyond model design, we propose a general pipeline for 3D scene data processing to address scale inconsistency issue. We validate our method on the recent scene-level 3D dataset...
  </details>  
  Keywords: text-to-3d, image-to-3d  
- **[MeshLLM: Empowering Large Language Models to Progressively Understand and Generate 3D Mesh](https://arxiv.org/abs/2508.01242v2)**  
  Authors: Shuangkang Fang, I-Chao Shen, Yufeng Wang, Yi-Hsuan Tsai, Yi Yang, Shuchang Zhou, Wenrui Ding, Takeo Igarashi, Ming-Hsuan Yang  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2508.01242v2.pdf)  
  <details><summary>Abstract</summary>

  We present MeshLLM, a novel framework that leverages large language models (LLMs) to understand and generate text-serialized 3D meshes. Our approach addresses key limitations in existing methods, including the limited dataset scale when catering to LLMs' token length and the loss of 3D structural information during mesh serialization. We introduce a Primitive-Mesh decomposition strategy, which divides 3D meshes into structurally meaningful subunits. This enables the creation of a large-scale dataset with 1500k+ samples, almost 50 times larger than previous methods, which aligns better with the LLM scaling law principles. Furthermore, we propose inferring face connectivity from vertices and local mesh assembly training strategies, significantly enhancing the LLMs' ability to capture mesh topology and spatial structures. Experiments show that MeshLLM outperforms the state-of-the-art LLaMA-Mesh in both mesh generation quality and shape understanding, highlighting its great potential in processing text-serialized 3D meshes.
  </details>  
- **[IGL-Nav: Incremental 3D Gaussian Localization for Image-goal Navigation](https://arxiv.org/abs/2508.00823v1)**  
  Authors: Wenxuan Guo, Xiuwei Xu, Hang Yin, Ziwei Wang, Jianjiang Feng, Jie Zhou, Jiwen Lu  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2508.00823v1.pdf) | [![Project](https://img.shields.io/badge/-Project-blue)](https://gwxuan.github.io/IGL-Nav)  
  <details><summary>Abstract</summary>

  Visual navigation with an image as goal is a fundamental and challenging problem. Conventional methods either rely on end-to-end RL learning or modular-based policy with topological graph or BEV map as memory, which cannot fully model the geometric relationship between the explored 3D environment and the goal image. In order to efficiently and accurately localize the goal image in 3D space, we build our navigation system upon the renderable 3D gaussian (3DGS) representation. However, due to the computational intensity of 3DGS optimization and the large search space of 6-DoF camera pose, directly leveraging 3DGS for image localization during agent exploration process is prohibitively inefficient. To this end, we propose IGL-Nav, an Incremental 3D Gaussian Localization framework for efficient and 3D-aware image-goal navigation. Specifically, we incrementally update the scene representation as new images arrive with feed-forward monocular prediction. Then we coarsely localize the goal by leveraging the geometric information for discrete...
  </details>  
- **[Cooperative Perception: A Resource-Efficient Framework for Multi-Drone 3D Scene Reconstruction Using Federated Diffusion and NeRF](https://arxiv.org/abs/2508.00967v1)**  
  Authors: Massoud Pourmandi  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2508.00967v1.pdf)  
  <details><summary>Abstract</summary>

  The proposal introduces an innovative drone swarm perception system that aims to solve problems related to computational limitations and low-bandwidth communication, and real-time scene reconstruction. The framework enables efficient multi-agent 3D/4D scene synthesis through federated learning of shared diffusion model and YOLOv12 lightweight semantic extraction and local NeRF updates while maintaining privacy and scalability. The framework redesigns generative diffusion models for joint scene reconstruction, and improves cooperative scene understanding, while adding semantic-aware compression protocols. The approach can be validated through simulations and potential real-world deployment on drone testbeds, positioning it as a disruptive advancement in multi-agent AI for autonomous systems.
  </details>  
  Keywords: 3d scene reconstruction  
- **[Guiding Diffusion-Based Articulated Object Generation by Partial Point Cloud Alignment and Physical Plausibility Constraints](https://arxiv.org/abs/2508.00558v1)**  
  Authors: Jens U. Kreber, Joerg Stueckler  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2508.00558v1.pdf)  
  <details><summary>Abstract</summary>

  Articulated objects are an important type of interactable objects in everyday environments. In this paper, we propose PhysNAP, a novel diffusion model-based approach for generating articulated objects that aligns them with partial point clouds and improves their physical plausibility. The model represents part shapes by signed distance functions (SDFs). We guide the reverse diffusion process using a point cloud alignment loss computed using the predicted SDFs. Additionally, we impose non-penetration and mobility constraints based on the part SDFs for guiding the model to generate more physically plausible objects. We also make our diffusion approach category-aware to further improve point cloud alignment if category information is available. We evaluate the generative ability and constraint consistency of samples generated with PhysNAP using the PartNet-Mobility dataset. We also compare it with an unguided baseline diffusion model and demonstrate that PhysNAP can improve constraint consistency and provides a tradeoff with generative ability.
  </details>  
  Keywords: articulated object generation  
- **[Sel3DCraft: Interactive Visual Prompts for User-Friendly Text-to-3D Generation](https://arxiv.org/abs/2508.00428v1)**  
  Authors: Nan Xiang, Tianyi Liang, Haiwen Huang, Shiqi Jiang, Hao Huang, Yifei Huang, Liangyu Chen, Changbo Wang, Chenhui Li  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2508.00428v1.pdf)  
  <details><summary>Abstract</summary>

  Text-to-3D (T23D) generation has transformed digital content creation, yet remains bottlenecked by blind trial-and-error prompting processes that yield unpredictable results. While visual prompt engineering has advanced in text-to-image domains, its application to 3D generation presents unique challenges requiring multi-view consistency evaluation and spatial understanding. We present Sel3DCraft, a visual prompt engineering system for T23D that transforms unstructured exploration into a guided visual process. Our approach introduces three key innovations: a dual-branch structure combining retrieval and generation for diverse candidate exploration; a multi-view hybrid scoring approach that leverages MLLMs with innovative high-level metrics to assess 3D models with human-expert consistency; and a prompt-driven visual analytics suite that enables intuitive defect identification and refinement. Extensive testing and user studies demonstrate that Sel3DCraft surpasses other T23D systems in supporting creativity for designers.
  </details>  
  Keywords: text-to-3d  
- **[Occlusion-robust Stylization for Drawing-based 3D Animation](https://arxiv.org/abs/2508.00398v1)**  
  Authors: Sunjae Yoon, Gwanhyeong Koo, Younghwan Lee, Ji Woo Hong, Chang D. Yoo  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2508.00398v1.pdf)  
  <details><summary>Abstract</summary>

  3D animation aims to generate a 3D animated video from an input image and a target 3D motion sequence. Recent advances in image-to-3D models enable the creation of animations directly from user-hand drawings. Distinguished from conventional 3D animation, drawing-based 3D animation is crucial to preserve artist's unique style properties, such as rough contours and distinct stroke patterns. However, recent methods still exhibit quality deterioration in style properties, especially under occlusions caused by overlapping body parts, leading to contour flickering and stroke blurring. This occurs due to a `stylization pose gap' between training and inference in stylization networks designed to preserve drawing styles in drawing-based 3D animation systems. The stylization pose gap denotes that input target poses used to train the stylization network are always in occlusion-free poses, while target poses encountered in an inference include diverse occlusions under dynamic motions. To this end, we propose Occlusion-robust Stylization Framework (OSF) for...
  </details>  
  Keywords: image-to-3d  



## Categorized Papers

### Cross-Modal Generation

*Showing the latest 50 out of 90 papers*

- **[FastPhysGS: Accelerating Physics-based Dynamic 3DGS Simulation via Interior Completion and Adaptive Optimization](https://arxiv.org/abs/2602.01723v1)**  
  Authors: Yikun Ma, Yiqing Li, Jingwen Ye, Zhongkai Wu, Weidong Zhang, Lin Gao, Zhi Jin  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2602.01723v1.pdf)  
  <details><summary>Abstract</summary>

  Extending 3D Gaussian Splatting (3DGS) to 4D physical simulation remains challenging. Based on the Material Point Method (MPM), existing methods either rely on manual parameter tuning or distill dynamics from video diffusion models, limiting the generalization and optimization efficiency. Recent attempts using LLMs/VLMs suffer from a text/image-to-3D perceptual gap, yielding unstable physics behavior. In addition, they often ignore the surface structure of 3DGS, leading to implausible motion. We propose FastPhysGS, a fast and robust framework for physics-based dynamic 3DGS simulation:(1) Instance-aware Particle Filling (IPF) with Monte Carlo Importance Sampling (MCIS) to efficiently populate interior particles while preserving geometric fidelity; (2) Bidirectional Graph Decoupling Optimization (BGDO), an adaptive strategy that rapidly optimizes material parameters predicted from a VLM. Experiments show FastPhysGS achieves high-fidelity physical simulation in 1 minute using only 7 GB runtime memory, outperforming prior works with broad potential applications.
  </details>  
  Keywords: image-to-3d  
- **[RoamScene3D: Immersive Text-to-3D Scene Generation via Adaptive Object-aware Roaming](https://arxiv.org/abs/2601.19433v1)**  
  Authors: Jisheng Chu, Wenrui Li, Rui Zhao, Wangmeng Zuo, Shifeng Chen, Xiaopeng Fan  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2601.19433v1.pdf) | [![GitHub](https://img.shields.io/github/stars/JS-CHU/RoamScene3D?style=social)](https://github.com/JS-CHU/RoamScene3D)  
  <details><summary>Abstract</summary>

  Generating immersive 3D scenes from texts is a core task in computer vision, crucial for applications in virtual reality and game development. Despite the promise of leveraging 2D diffusion priors, existing methods suffer from spatial blindness and rely on predefined trajectories that fail to exploit the inner relationships among salient objects. Consequently, these approaches are unable to comprehend the semantic layout, preventing them from exploring the scene adaptively to infer occluded content. Moreover, current inpainting models operate in 2D image space, struggling to plausibly fill holes caused by camera motion. To address these limitations, we propose RoamScene3D, a novel framework that bridges the gap between semantic guidance and spatial generation. Our method reasons about the semantic relations among objects and produces consistent and photorealistic scenes. Specifically, we employ a vision-language model (VLM) to construct a scene graph that encodes object relations, guiding the camera to perceive salient object boundaries and...
  </details>  
  Keywords: text-to-3d  
- **[Spherical Geometry Diffusion: Generating High-quality 3D Face Geometry via Sphere-anchored Representations](https://arxiv.org/abs/2601.13371v1)**  
  Authors: Junyi Zhang, Yiming Wang, Yunhong Lu, Qichao Wang, Wenzhe Qian, Xiaoyin Xu, David Gu, Min Zhang  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2601.13371v1.pdf)  
  <details><summary>Abstract</summary>

  A fundamental challenge in text-to-3D face generation is achieving high-quality geometry. The core difficulty lies in the arbitrary and intricate distribution of vertices in 3D space, making it challenging for existing models to establish clean connectivity and resulting in suboptimal geometry. To address this, our core insight is to simplify the underlying geometric structure by constraining the distribution onto a simple and regular manifold, a topological sphere. Building on this, we first propose the Spherical Geometry Representation, a novel face representation that anchors geometric signals to uniform spherical coordinates. This guarantees a regular point distribution, from which the mesh connectivity can be robustly reconstructed. Critically, this canonical sphere can be seamlessly unwrapped into a 2D map, creating a perfect synergy with powerful 2D generative models. We then introduce Spherical Geometry Diffusion, a conditional diffusion framework built upon this 2D map. It enables diverse and controllable generation by jointly modeling geometry...
  </details>  
  Keywords: text-to-3d, texture synthesis  
- **[A Comparative Study of 3D Model Acquisition Methods for Synthetic Data Generation of Agricultural Products](https://arxiv.org/abs/2601.03784v1)**  
  Authors: Steven Moonen, Rob Salaets, Kenneth Batstone, Abdellatif Bey-Temsamani, Nick Michiels  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2601.03784v1.pdf)  
  <details><summary>Abstract</summary>

  In the manufacturing industry, computer vision systems based on artificial intelligence (AI) are widely used to reduce costs and increase production. Training these AI models requires a large amount of training data that is costly to acquire and annotate, especially in high-variance, low-volume manufacturing environments. A popular approach to reduce the need for real data is the use of synthetic data that is generated by leveraging computer-aided design (CAD) models available in the industry. However, in the agricultural industry these models are not readily available, increasing the difficulty in leveraging synthetic data. In this paper, we present different techniques for substituting CAD files to create synthetic datasets. We measure their relative performance when used to train an AI object detection model to separate stones and potatoes in a bin picking environment. We demonstrate that using highly representative 3D models acquired by scanning or using image-to-3D approaches can be used to...
  </details>  
  Keywords: image-to-3d  
- **[UnrealPose: Leveraging Game Engine Kinematics for Large-Scale Synthetic Human Pose Data](https://arxiv.org/abs/2601.00991v1)**  
  Authors: Joshua Kawaguchi, Saad Manzur, Emily Gao Wang, Maitreyi Sinha, Bryan Vela, Yunxi Wang, Brandon Vela, Wayne B. Hayes  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2601.00991v1.pdf)  
  <details><summary>Abstract</summary>

  Diverse, accurately labeled 3D human pose data is expensive and studio-bound, while in-the-wild datasets lack known ground truth. We introduce UnrealPose-Gen, an Unreal Engine 5 pipeline built on Movie Render Queue for high-quality offline rendering. Our generated frames include: (i) 3D joints in world and camera coordinates, (ii) 2D projections and COCO-style keypoints with occlusion and joint-visibility flags, (iii) person bounding boxes, and (iv) camera intrinsics and extrinsics. We use UnrealPose-Gen to present UnrealPose-1M, an approximately one million frame corpus comprising eight sequences: five scripted "coherent" sequences spanning five scenes, approximately 40 actions, and five subjects; and three randomized sequences across three scenes, approximately 100 actions, and five subjects, all captured from diverse camera trajectories for broad viewpoint coverage. As a fidelity check, we report real-to-synthetic results on four tasks: image-to-3D pose, 2D keypoint detection, 2D-to-3D lifting, and person detection/segmentation. Though time and resources constrain us from an unlimited dataset,...
  </details>  
  Keywords: image-to-3d  
- **[SAM 3D for 3D Object Reconstruction from Remote Sensing Images](https://arxiv.org/abs/2512.22452v1)**  
  Authors: Junsheng Yao, Lichao Mou, Qingyu Li  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2512.22452v1.pdf)  
  <details><summary>Abstract</summary>

  Monocular 3D building reconstruction from remote sensing imagery is essential for scalable urban modeling, yet existing methods often require task-specific architectures and intensive supervision. This paper presents the first systematic evaluation of SAM 3D, a general-purpose image-to-3D foundation model, for monocular remote sensing building reconstruction. We benchmark SAM 3D against TRELLIS on samples from the NYC Urban Dataset, employing Frechet Inception Distance (FID) and CLIP-based Maximum Mean Discrepancy (CMMD) as evaluation metrics. Experimental results demonstrate that SAM 3D produces more coherent roof geometry and sharper boundaries compared to TRELLIS. We further extend SAM 3D to urban scene reconstruction through a segment-reconstruct-compose pipeline, demonstrating its potential for urban scene modeling. We also analyze practical limitations and discuss future research directions. These findings provide practical guidance for deploying foundation models in urban 3D reconstruction and motivate future integration of scene-level structural priors.
  </details>  
  Keywords: image-to-3d  
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
- **[Particulate: Feed-Forward 3D Object Articulation](https://arxiv.org/abs/2512.11798v1)**  
  Authors: Ruining Li, Yuxin Yao, Chuanxia Zheng, Christian Rupprecht, Joan Lasenby, Shangzhe Wu, Andrea Vedaldi  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2512.11798v1.pdf)  
  <details><summary>Abstract</summary>

  We present Particulate, a feed-forward approach that, given a single static 3D mesh of an everyday object, directly infers all attributes of the underlying articulated structure, including its 3D parts, kinematic structure, and motion constraints. At its core is a transformer network, Part Articulation Transformer, which processes a point cloud of the input mesh using a flexible and scalable architecture to predict all the aforementioned attributes with native multi-joint support. We train the network end-to-end on a diverse collection of articulated 3D assets from public datasets. During inference, Particulate lifts the network's feed-forward prediction to the input mesh, yielding a fully articulated 3D model in seconds, much faster than prior approaches that require per-object optimization. Particulate can also accurately infer the articulated structure of AI-generated 3D assets, enabling full-fledged extraction of articulated 3D objects from a single (real or synthetic) image when combined with an off-the-shelf image-to-3D generator. We further...
  </details>  
  Keywords: 3d generator, image-to-3d  
- **[Are We Ready for RL in Text-to-3D Generation? A Progressive Investigation](https://arxiv.org/abs/2512.10949v1)**  
  Authors: Yiwen Tang, Zoey Guo, Kaixin Zhu, Ray Zhang, Qizhi Chen, Dongzhi Jiang, Junli Liu, Bohan Zeng, Haoming Song, Delin Qu, Tianyi Bai, Dan Xu, Wentao Zhang, Bin Zhao  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2512.10949v1.pdf) | [![GitHub](https://img.shields.io/github/stars/Ivan-Tang-3D/3DGen-R1?style=social)](https://github.com/Ivan-Tang-3D/3DGen-R1)  
  <details><summary>Abstract</summary>

  Reinforcement learning (RL), earlier proven to be effective in large language and multi-modal models, has been successfully extended to enhance 2D image generation recently. However, applying RL to 3D generation remains largely unexplored due to the higher spatial complexity of 3D objects, which require globally consistent geometry and fine-grained local textures. This makes 3D generation significantly sensitive to reward designs and RL algorithms. To address these challenges, we conduct the first systematic study of RL for text-to-3D autoregressive generation across several dimensions. (1) Reward designs: We evaluate reward dimensions and model choices, showing that alignment with human preference is crucial, and that general multi-modal models provide robust signal for 3D attributes. (2) RL algorithms: We study GRPO variants, highlighting the effectiveness of token-level optimization, and further investigate the scaling of training data and iterations. (3) Text-to-3D Benchmarks: Since existing benchmarks fail to measure implicit reasoning abilities in 3D generation models,...
  </details>  
  Keywords: text-to-3d  

### Dynamic & Articulated Modeling

- **[GAOT: Generating Articulated Objects Through Text-Guided Diffusion Models](https://arxiv.org/abs/2512.03566v1)**  
  Authors: Hao Sun, Lei Fan, Donglin Di, Shaohui Liu  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2512.03566v1.pdf)  
  <details><summary>Abstract</summary>

  Articulated object generation has seen increasing advancements, yet existing models often lack the ability to be conditioned on text prompts. To address the significant gap between textual descriptions and 3D articulated object representations, we propose GAOT, a three-phase framework that generates articulated objects from text prompts, leveraging diffusion models and hypergraph learning in a three-step process. First, we fine-tune a point cloud generation model to produce a coarse representation of objects from text prompts. Given the inherent connection between articulated objects and graph structures, we design a hypergraph-based learning method to refine these coarse representations, representing object parts as graph vertices. Finally, leveraging a diffusion model, the joints of articulated objects-represented as graph edges-are generated based on the object parts. Extensive qualitative and quantitative experiments on the PartNet-Mobility dataset demonstrate the effectiveness of our approach, achieving superior performance over previous methods.
  </details>  
  Keywords: articulated object generation  
- **[FreeArt3D: Training-Free Articulated Object Generation using 3D Diffusion](https://arxiv.org/abs/2510.25765v2)**  
  Authors: Chuhao Chen, Isabella Liu, Xinyue Wei, Hao Su, Minghua Liu  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2510.25765v2.pdf) | [![Project](https://img.shields.io/badge/-Project-blue)](https://czzzzh.github.io/FreeArt3D)  
  <details><summary>Abstract</summary>

  Articulated 3D objects are central to many applications in robotics, AR/VR, and animation. Recent approaches to modeling such objects either rely on optimization-based reconstruction pipelines that require dense-view supervision or on feed-forward generative models that produce coarse geometric approximations and often overlook surface texture. In contrast, open-world 3D generation of static objects has achieved remarkable success, especially with the advent of native 3D diffusion models such as Trellis. However, extending these methods to articulated objects by training native 3D diffusion models poses significant challenges. In this work, we present FreeArt3D, a training-free framework for articulated 3D object generation. Instead of training a new model on limited articulated data, FreeArt3D repurposes a pre-trained static 3D diffusion model (e.g., Trellis) as a powerful shape prior. It extends Score Distillation Sampling (SDS) into the 3D-to-4D domain by treating articulation as an additional generative dimension. Given a few images captured in different articulation states,...
  </details>  
  Keywords: articulated object generation  
- **[Pain in 3D: Generating Controllable Synthetic Faces for Automated Pain Assessment](https://arxiv.org/abs/2509.16727v4)**  
  Authors: Xin Lei Lin, Soroush Mehraban, Abhishek Moturu, Babak Taati  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2509.16727v4.pdf)  
  <details><summary>Abstract</summary>

  Automated pain assessment from facial expressions is crucial for non-communicative patients, such as those with dementia. Progress has been limited by two challenges: (i) existing datasets exhibit severe demographic and label imbalance due to ethical constraints, and (ii) current generative models cannot precisely control facial action units (AUs), facial structure, or clinically validated pain levels. We present 3DPain, a large-scale synthetic dataset specifically designed for automated pain assessment, featuring unprecedented annotation richness and demographic diversity. Our three-stage framework generates diverse 3D meshes, textures them with diffusion models, and applies AU-driven face rigging to synthesize multi-view faces with paired neutral and pain images, AU configurations, PSPI scores, and the first dataset-level annotations of pain-region heatmaps. The dataset comprises 82,500 samples across 25,000 pain expression heatmaps and 2,500 synthetic identities balanced by age, gender, and ethnicity. We further introduce ViTPain, a Vision Transformer based cross-modal distillation framework in which a heatmap-trained teacher...
  </details>  
  Keywords: rigging  
- **[Empowering Children to Create AI-Enabled Augmented Reality Experiences](https://arxiv.org/abs/2508.08467v1)**  
  Authors: Lei Zhang, Shuyao Zhou, Amna Liaqat, Tinney Mak, Brian Berengard, Emily Qian, Andrés Monroy-Hernández  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2508.08467v1.pdf)  
  <details><summary>Abstract</summary>

  Despite their potential to enhance children's learning experiences, AI-enabled AR technologies are predominantly used in ways that position children as consumers rather than creators. We introduce Capybara, an AR-based and AI-powered visual programming environment that empowers children to create, customize, and program 3D characters overlaid onto the physical world. Capybara enables children to create virtual characters and accessories using text-to-3D generative AI models, and to animate these characters through auto-rigging and body tracking. In addition, our system employs vision-based AI models to recognize physical objects, allowing children to program interactive behaviors between virtual characters and their physical surroundings. We demonstrate the expressiveness of Capybara through a set of novel AR experiences. We conducted user studies with 20 children in the United States and Argentina. Our findings suggest that Capybara can empower children to harness AI in authoring personalized and engaging AR experiences that seamlessly bridge the virtual and physical worlds.
  </details>  
  Keywords: text-to-3d, rigging  
- **[Guiding Diffusion-Based Articulated Object Generation by Partial Point Cloud Alignment and Physical Plausibility Constraints](https://arxiv.org/abs/2508.00558v1)**  
  Authors: Jens U. Kreber, Joerg Stueckler  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2508.00558v1.pdf)  
  <details><summary>Abstract</summary>

  Articulated objects are an important type of interactable objects in everyday environments. In this paper, we propose PhysNAP, a novel diffusion model-based approach for generating articulated objects that aligns them with partial point clouds and improves their physical plausibility. The model represents part shapes by signed distance functions (SDFs). We guide the reverse diffusion process using a point cloud alignment loss computed using the predicted SDFs. Additionally, we impose non-penetration and mobility constraints based on the part SDFs for guiding the model to generate more physically plausible objects. We also make our diffusion approach category-aware to further improve point cloud alignment if category information is available. We evaluate the generative ability and constraint consistency of samples generated with PhysNAP using the PartNet-Mobility dataset. We also compare it with an unguided baseline diffusion model and demonstrate that PhysNAP can improve constraint consistency and provides a tradeoff with generative ability.
  </details>  
  Keywords: articulated object generation  
- **[From One to More: Contextual Part Latents for 3D Generation](https://arxiv.org/abs/2507.08772v2)**  
  Authors: Shaocong Dong, Lihe Ding, Xiao Chen, Yaokun Li, Yuxin Wang, Yucheng Wang, Qi Wang, Jaehyeok Kim, Chenjian Gao, Zhanpeng Huang, Zibin Wang, Tianfan Xue, Dan Xu  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2507.08772v2.pdf)  
  <details><summary>Abstract</summary>

  Recent advances in 3D generation have transitioned from multi-view 2D rendering approaches to 3D-native latent diffusion frameworks that exploit geometric priors in ground truth data. Despite progress, three key limitations persist: (1) Single-latent representations fail to capture complex multi-part geometries, causing detail degradation; (2) Holistic latent coding neglects part independence and interrelationships critical for compositional design; (3) Global conditioning mechanisms lack fine-grained controllability. Inspired by human 3D design workflows, we propose CoPart - a part-aware diffusion framework that decomposes 3D objects into contextual part latents for coherent multi-part generation. This paradigm offers three advantages: i) Reduces encoding complexity through part decomposition; ii) Enables explicit part relationship modeling; iii) Supports part-level conditioning. We further develop a mutual guidance strategy to fine-tune pre-trained diffusion models for joint part latent denoising, ensuring both geometric coherence and foundation model priors. To enable large-scale training, we construct Partverse - a novel 3D part dataset derived...
  </details>  
  Keywords: articulated object generation  
- **[EmbodiedGen: Towards a Generative 3D World Engine for Embodied Intelligence](https://arxiv.org/abs/2506.10600v2)**  
  Authors: Xinjie Wang, Liu Liu, Yu Cao, Ruiqi Wu, Wenkang Qin, Dehui Wang, Wei Sui, Zhizhong Su  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2506.10600v2.pdf) | [![Project](https://img.shields.io/badge/-Project-blue)](https://horizonrobotics.github.io/robot_lab/embodied_gen/index.html)  
  <details><summary>Abstract</summary>

  Constructing a physically realistic and accurately scaled simulated 3D world is crucial for the training and evaluation of embodied intelligence tasks. The diversity, realism, low cost accessibility and affordability of 3D data assets are critical for achieving generalization and scalability in embodied AI. However, most current embodied intelligence tasks still rely heavily on traditional 3D computer graphics assets manually created and annotated, which suffer from high production costs and limited realism. These limitations significantly hinder the scalability of data driven approaches. We present EmbodiedGen, a foundational platform for interactive 3D world generation. It enables the scalable generation of high-quality, controllable and photorealistic 3D assets with accurate physical properties and real-world scale in the Unified Robotics Description Format (URDF) at low cost. These assets can be directly imported into various physics simulation engines for fine-grained physical control, supporting downstream tasks in training and evaluation. EmbodiedGen is an easy-to-use, full-featured toolkit composed...
  </details>  
  Keywords: text-to-3d, articulated object generation, image-to-3d  
- **[AdaHuman: Animatable Detailed 3D Human Generation with Compositional Multiview Diffusion](https://arxiv.org/abs/2505.24877v1)**  
  Authors: Yangyi Huang, Ye Yuan, Xueting Li, Jan Kautz, Umar Iqbal  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2505.24877v1.pdf)  
  <details><summary>Abstract</summary>

  Existing methods for image-to-3D avatar generation struggle to produce highly detailed, animation-ready avatars suitable for real-world applications. We introduce AdaHuman, a novel framework that generates high-fidelity animatable 3D avatars from a single in-the-wild image. AdaHuman incorporates two key innovations: (1) A pose-conditioned 3D joint diffusion model that synthesizes consistent multi-view images in arbitrary poses alongside corresponding 3D Gaussian Splats (3DGS) reconstruction at each diffusion step; (2) A compositional 3DGS refinement module that enhances the details of local body parts through image-to-image refinement and seamlessly integrates them using a novel crop-aware camera ray map, producing a cohesive detailed 3D avatar. These components allow AdaHuman to generate highly realistic standardized A-pose avatars with minimal self-occlusion, enabling rigging and animation with any input motion. Extensive evaluation on public benchmarks and in-the-wild images demonstrates that AdaHuman significantly outperforms state-of-the-art methods in both avatar reconstruction and reposing. Code and models will be publicly available for...
  </details>  
  Keywords: rigging, image-to-3d  
- **[DIPO: Dual-State Images Controlled Articulated Object Generation Powered by Diverse Data](https://arxiv.org/abs/2505.20460v2)**  
  Authors: Ruiqi Wu, Xinjie Wang, Liu Liu, Chunle Guo, Jiaxiong Qiu, Chongyi Li, Lichao Huang, Zhizhong Su, Ming-Ming Cheng  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2505.20460v2.pdf)  
  <details><summary>Abstract</summary>

  We present DIPO, a novel framework for the controllable generation of articulated 3D objects from a pair of images: one depicting the object in a resting state and the other in an articulated state. Compared to the single-image approach, our dual-image input imposes only a modest overhead for data collection, but at the same time provides important motion information, which is a reliable guide for predicting kinematic relationships between parts. Specifically, we propose a dual-image diffusion model that captures relationships between the image pair to generate part layouts and joint parameters. In addition, we introduce a Chain-of-Thought (CoT) based graph reasoner that explicitly infers part connectivity relationships. To further improve robustness and generalization on complex articulated objects, we develop a fully automated dataset expansion pipeline, name LEGO-Art, that enriches the diversity and complexity of PartNet-Mobility dataset. We propose PM-X, a large-scale dataset of complex articulated 3D objects, accompanied by rendered...
  </details>  
  Keywords: articulated object generation  

### Scene & View Synthesis

*Showing the latest 50 out of 115 papers*

- **[Seeing Through Clutter: Structured 3D Scene Reconstruction via Iterative Object Removal](https://arxiv.org/abs/2602.04053v1)**  
  Authors: Rio Aguina-Kang, Kevin James Blackburn-Matzen, Thibault Groueix, Vladimir Kim, Matheus Gadelha  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2602.04053v1.pdf) | [![Project](https://img.shields.io/badge/-Project-blue)](https://rioak.github.io/seeingthroughclutter)  
  <details><summary>Abstract</summary>

  We present SeeingThroughClutter, a method for reconstructing structured 3D representations from single images by segmenting and modeling objects individually. Prior approaches rely on intermediate tasks such as semantic segmentation and depth estimation, which often underperform in complex scenes, particularly in the presence of occlusion and clutter. We address this by introducing an iterative object removal and reconstruction pipeline that decomposes complex scenes into a sequence of simpler subtasks. Using VLMs as orchestrators, foreground objects are removed one at a time via detection, segmentation, object removal, and 3D fitting. We show that removing objects allows for cleaner segmentations of subsequent objects, even in highly occluded scenes. Our method requires no task-specific training and benefits directly from ongoing advances in foundation models. We demonstrate stateof-the-art robustness on 3D-Front and ADE20K datasets. Project Page: https://rioak.github.io/seeingthroughclutter/
  </details>  
  Keywords: 3d scene reconstruction  
- **[Gaussian-Constrained LeJEPA Representations for Unsupervised Scene Discovery and Pose Consistency](https://arxiv.org/abs/2602.07016v1)**  
  Authors: Mohsen Mostafa  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2602.07016v1.pdf)  
  <details><summary>Abstract</summary>

  Unsupervised 3D scene reconstruction from unstructured image collections remains a fundamental challenge in computer vision, particularly when images originate from multiple unrelated scenes and contain significant visual ambiguity. The Image Matching Challenge 2025 (IMC2025) highlights these difficulties by requiring both scene discovery and camera pose estimation under real-world conditions, including outliers and mixed content. This paper investigates the application of Gaussian-constrained representations inspired by LeJEPA (Joint Embedding Predictive Architecture) to address these challenges. We present three progressively refined pipelines, culminating in a LeJEPA-inspired approach that enforces isotropic Gaussian constraints on learned image embeddings. Rather than introducing new theoretical guarantees, our work empirically evaluates how these constraints influence clustering consistency and pose estimation robustness in practice. Experimental results on IMC2025 demonstrate that Gaussian-constrained embeddings can improve scene separation and pose plausibility compared to heuristic-driven baselines, particularly in visually ambiguous settings. These findings suggest that theoretically motivated representation constraints offer a promising...
  </details>  
  Keywords: 3d scene reconstruction  
- **[EUGens: Efficient, Unified, and General Dense Layers](https://arxiv.org/abs/2601.22563v2)**  
  Authors: Sang Min Kim, Byeongchan Kim, Arijit Sehanobish, Somnath Basu Roy Chowdhury, Rahul Kidambi, Dongseok Shim, Avinava Dubey, Snigdha Chaturvedi, Min-hwan Oh, Krzysztof Choromanski  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2601.22563v2.pdf)  
  <details><summary>Abstract</summary>

  Efficient neural networks are essential for scaling machine learning models to real-time applications and resource-constrained environments. Fully-connected feedforward layers (FFLs) introduce computation and parameter count bottlenecks within neural network architectures. To address this challenge, in this work, we propose a new class of dense layers that generalize standard fully-connected feedforward layers, \textbf{E}fficient, \textbf{U}nified and \textbf{Gen}eral dense layers (EUGens). EUGens leverage random features to approximate standard FFLs and go beyond them by incorporating a direct dependence on the input norms in their computations. The proposed layers unify existing efficient FFL extensions and improve efficiency by reducing inference complexity from quadratic to linear time. They also lead to \textbf{the first} unbiased algorithms approximating FFLs with arbitrary polynomial activation functions. Furthermore, EuGens reduce the parameter count and computational overhead while preserving the expressive power and adaptability of FFLs. We also present a layer-wise knowledge transfer technique that bypasses backpropagation, enabling efficient adaptation of...
  </details>  
  Keywords: 3d scene reconstruction  
- **[Splat-Portrait: Generalizing Talking Heads with Gaussian Splatting](https://arxiv.org/abs/2601.18633v1)**  
  Authors: Tong Shi, Melonie de Almeida, Daniela Ivanova, Nicolas Pugeault, Paul Henderson  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2601.18633v1.pdf) | [![GitHub](https://img.shields.io/github/stars/stonewalking/Splat-portrait?style=social)](https://github.com/stonewalking/Splat-portrait)  
  <details><summary>Abstract</summary>

  Talking Head Generation aims at synthesizing natural-looking talking videos from speech and a single portrait image. Previous 3D talking head generation methods have relied on domain-specific heuristics such as warping-based facial motion representation priors to animate talking motions, yet still produce inaccurate 3D avatar reconstructions, thus undermining the realism of generated animations. We introduce Splat-Portrait, a Gaussian-splatting-based method that addresses the challenges of 3D head reconstruction and lip motion synthesis. Our approach automatically learns to disentangle a single portrait image into a static 3D reconstruction represented as static Gaussian Splatting, and a predicted whole-image 2D background. It then generates natural lip motion conditioned on input audio, without any motion driven priors. Training is driven purely by 2D reconstruction and score-distillation losses, without 3D supervision nor landmarks. Experimental results demonstrate that Splat-Portrait exhibits superior performance on talking head generation and novel view synthesis, achieving better visual quality compared to previous works....
  </details>  
  Keywords: novel view synthesis  
- **[POTR: Post-Training 3DGS Compression](https://arxiv.org/abs/2601.14821v1)**  
  Authors: Bert Ramlot, Martijn Courteaux, Peter Lambert, Glenn Van Wallendael  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2601.14821v1.pdf)  
  <details><summary>Abstract</summary>

  3D Gaussian Splatting (3DGS) has recently emerged as a promising contender to Neural Radiance Fields (NeRF) in 3D scene reconstruction and real-time novel view synthesis. 3DGS outperforms NeRF in training and inference speed but has substantially higher storage requirements. To remedy this downside, we propose POTR, a post-training 3DGS codec built on two novel techniques. First, POTR introduces a novel pruning approach that uses a modified 3DGS rasterizer to efficiently calculate every splat's individual removal effect simultaneously. This technique results in 2-4x fewer splats than other post-training pruning techniques and as a result also significantly accelerates inference with experiments demonstrating 1.5-2x faster inference than other compressed models. Second, we propose a novel method to recompute lighting coefficients, significantly reducing their entropy without using any form of training. Our fast and highly parallel approach especially increases AC lighting coefficient sparsity, with experiments demonstrating increases from 70% to 97%, with minimal loss...
  </details>  
  Keywords: novel view synthesis, 3d scene reconstruction  
- **[studentSplat: Your Student Model Learns Single-view 3D Gaussian Splatting](https://arxiv.org/abs/2601.11772v1)**  
  Authors: Yimu Pan, Hongda Mao, Qingshuang Chen, Yelin Kim  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2601.11772v1.pdf)  
  <details><summary>Abstract</summary>

  Recent advance in feed-forward 3D Gaussian splatting has enable remarkable multi-view 3D scene reconstruction or single-view 3D object reconstruction but single-view 3D scene reconstruction remain under-explored due to inherited ambiguity in single-view. We present \textbf{studentSplat}, a single-view 3D Gaussian splatting method for scene reconstruction. To overcome the scale ambiguity and extrapolation problems inherent in novel-view supervision from a single input, we introduce two techniques: 1) a teacher-student architecture where a multi-view teacher model provides geometric supervision to the single-view student during training, addressing scale ambiguity and encourage geometric validity; and 2) an extrapolation network that completes missing scene context, enabling high-quality extrapolation. Extensive experiments show studentSplat achieves state-of-the-art single-view novel-view reconstruction quality and comparable performance to multi-view methods at the scene level. Furthermore, studentSplat demonstrates competitive performance as a self-supervised single-view depth estimation method, highlighting its potential for general single-view 3D understanding tasks.
  </details>  
  Keywords: 3d scene reconstruction  
- **[SyncTwin: Fast Digital Twin Construction and Synchronization for Safe Robotic Grasping](https://arxiv.org/abs/2601.09920v1)**  
  Authors: Ruopeng Huang, Boyu Yang, Wenlong Gui, Jeremy Morgan, Erdem Biyik, Jiachen Li  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2601.09920v1.pdf)  
  <details><summary>Abstract</summary>

  Accurate and safe grasping under dynamic and visually occluded conditions remains a core challenge in real-world robotic manipulation. We present SyncTwin, a digital twin framework that unifies fast 3D scene reconstruction and real-to-sim synchronization for robust and safety-aware grasping in such environments. In the offline stage, we employ VGGT to rapidly reconstruct object-level 3D assets from RGB images, forming a reusable geometry library for simulation. During execution, SyncTwin continuously synchronizes the digital twin by tracking real-world object states via point cloud segmentation updates and aligning them through colored-ICP registration. The updated twin enables motion planners to compute collision-free and dynamically feasible trajectories in simulation, which are safely executed on the real robot through a closed real-to-sim-to-real loop. Experiments in dynamic and occluded scenes show that SyncTwin improves grasp accuracy and motion safety, demonstrating the effectiveness of digital-twin synchronization for real-world robotic execution.
  </details>  
  Keywords: 3d scene reconstruction  
- **[HERE: Hierarchical Active Exploration of Radiance Field with Epistemic Uncertainty Minimization](https://arxiv.org/abs/2601.07242v1)**  
  Authors: Taekbeom Lee, Dabin Kim, Youngseok Jang, H. Jin Kim  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2601.07242v1.pdf)  
  <details><summary>Abstract</summary>

  We present HERE, an active 3D scene reconstruction framework based on neural radiance fields, enabling high-fidelity implicit mapping. Our approach centers around an active learning strategy for camera trajectory generation, driven by accurate identification of unseen regions, which supports efficient data acquisition and precise scene reconstruction. The key to our approach is epistemic uncertainty quantification based on evidential deep learning, which directly captures data insufficiency and exhibits a strong correlation with reconstruction errors. This allows our framework to more reliably identify unexplored or poorly reconstructed regions compared to existing methods, leading to more informed and targeted exploration. Additionally, we design a hierarchical exploration strategy that leverages learned epistemic uncertainty, where local planning extracts target viewpoints from high-uncertainty voxels based on visibility for trajectory generation, and global planning uses uncertainty to guide large-scale coverage for efficient and comprehensive reconstruction. The effectiveness of the proposed method in active 3D reconstruction is demonstrated...
  </details>  
  Keywords: 3d scene reconstruction  
- **[Rotate Your Character: Revisiting Video Diffusion Models for High-Quality 3D Character Generation](https://arxiv.org/abs/2601.05722v1)**  
  Authors: Jin Wang, Jianxiang Lu, Comi Chen, Guangzheng Xu, Haoyu Yang, Peng Chen, Na Zhang, Yifan Xu, Longhuang Wu, Shuai Shao, Qinglin Lu, Ping Luo  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2601.05722v1.pdf)  
  <details><summary>Abstract</summary>

  Generating high-quality 3D characters from single images remains a significant challenge in digital content creation, particularly due to complex body poses and self-occlusion. In this paper, we present RCM (Rotate your Character Model), an advanced image-to-video diffusion framework tailored for high-quality novel view synthesis (NVS) and 3D character generation. Compared to existing diffusion-based approaches, RCM offers several key advantages: (1) transferring characters with any complex poses into a canonical pose, enabling consistent novel view synthesis across the entire viewing orbit, (2) high-resolution orbital video generation at 1024x1024 resolution, (3) controllable observation positions given different initial camera poses, and (4) multi-view conditioning supporting up to 4 input images, accommodating diverse user scenarios. Extensive experiments demonstrate that RCM outperforms state-of-the-art methods in both novel view synthesis and 3D generation quality.
  </details>  
  Keywords: novel view synthesis  
- **[360-GeoGS: Geometrically Consistent Feed-Forward 3D Gaussian Splatting Reconstruction for 360 Images](https://arxiv.org/abs/2601.02102v1)**  
  Authors: Jiaqi Yao, Zhongmiao Yan, Jingyi Xu, Songpengcheng Xia, Yan Xiang, Ling Pei  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2601.02102v1.pdf)  
  <details><summary>Abstract</summary>

  3D scene reconstruction is fundamental for spatial intelligence applications such as AR, robotics, and digital twins. Traditional multi-view stereo struggles with sparse viewpoints or low-texture regions, while neural rendering approaches, though capable of producing high-quality results, require per-scene optimization and lack real-time efficiency. Explicit 3D Gaussian Splatting (3DGS) enables efficient rendering, but most feed-forward variants focus on visual quality rather than geometric consistency, limiting accurate surface reconstruction and overall reliability in spatial perception tasks. This paper presents a novel feed-forward 3DGS framework for 360 images, capable of generating geometrically consistent Gaussian primitives while maintaining high rendering quality. A Depth-Normal geometric regularization is introduced to couple rendered depth gradients with normal information, supervising Gaussian rotation, scale, and position to improve point cloud and surface accuracy. Experimental results show that the proposed method maintains high rendering quality while significantly improving geometric consistency, providing an effective solution for 3D reconstruction in spatial perception...
  </details>  
  Keywords: 3d scene reconstruction  

### Surface & Appearance Modeling

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
- **[Spherical Geometry Diffusion: Generating High-quality 3D Face Geometry via Sphere-anchored Representations](https://arxiv.org/abs/2601.13371v1)**  
  Authors: Junyi Zhang, Yiming Wang, Yunhong Lu, Qichao Wang, Wenzhe Qian, Xiaoyin Xu, David Gu, Min Zhang  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2601.13371v1.pdf)  
  <details><summary>Abstract</summary>

  A fundamental challenge in text-to-3D face generation is achieving high-quality geometry. The core difficulty lies in the arbitrary and intricate distribution of vertices in 3D space, making it challenging for existing models to establish clean connectivity and resulting in suboptimal geometry. To address this, our core insight is to simplify the underlying geometric structure by constraining the distribution onto a simple and regular manifold, a topological sphere. Building on this, we first propose the Spherical Geometry Representation, a novel face representation that anchors geometric signals to uniform spherical coordinates. This guarantees a regular point distribution, from which the mesh connectivity can be robustly reconstructed. Critically, this canonical sphere can be seamlessly unwrapped into a 2D map, creating a perfect synergy with powerful 2D generative models. We then introduce Spherical Geometry Diffusion, a conditional diffusion framework built upon this 2D map. It enables diverse and controllable generation by jointly modeling geometry...
  </details>  
  Keywords: text-to-3d, texture synthesis  
- **[OSCAR: Optical-aware Semantic Control for Aleatoric Refinement in Sar-to-Optical Translation](https://arxiv.org/abs/2601.06835v1)**  
  Authors: Hyunseo Lee, Sang Min Kim, Ho Kyung Shin, Taeheon Kim, Woo-Jeoung Nam  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2601.06835v1.pdf)  
  <details><summary>Abstract</summary>

  Synthetic Aperture Radar (SAR) provides robust all-weather imaging capabilities; however, translating SAR observations into photo-realistic optical images remains a fundamentally ill-posed problem. Current approaches are often hindered by the inherent speckle noise and geometric distortions of SAR data, which frequently result in semantic misinterpretation, ambiguous texture synthesis, and structural hallucinations. To address these limitations, a novel SAR-to-Optical (S2O) translation framework is proposed, integrating three core technical contributions: (i) Cross-Modal Semantic Alignment, which establishes an Optical-Aware SAR Encoder by distilling robust semantic priors from an Optical Teacher into a SAR Student (ii) Semantically-Grounded Generative Guidance, realized by a Semantically-Grounded ControlNet that integrates class-aware text prompts for global context with hierarchical visual prompts for local spatial guidance; and (iii) an Uncertainty-Aware Objective, which explicitly models aleatoric uncertainty to dynamically modulate the reconstruction focus, effectively mitigating artifacts caused by speckle-induced ambiguity. Extensive experiments demonstrate that the proposed method achieves superior perceptual quality and...
  </details>  
  Keywords: texture synthesis  
- **[Conditional Morphogenesis: Emergent Generation of Structural Digits via Neural Cellular Automata](https://arxiv.org/abs/2512.08360v1)**  
  Authors: Ali Sakour  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2512.08360v1.pdf)  
  <details><summary>Abstract</summary>

  Biological systems exhibit remarkable morphogenetic plasticity, where a single genome can encode various specialized cellular structures triggered by local chemical signals. In the domain of Deep Learning, Differentiable Neural Cellular Automata (NCA) have emerged as a paradigm to mimic this self-organization. However, existing NCA research has predominantly focused on continuous texture synthesis or single-target object recovery, leaving the challenge of class-conditional structural generation largely unexplored. In this work, we propose a novel Conditional Neural Cellular Automata (c-NCA) architecture capable of growing distinct topological structures - specifically MNIST digits - from a single generic seed, guided solely by a spatially broadcasted class vector. Unlike traditional generative models (e.g., GANs, VAEs) that rely on global reception fields, our model enforces strict locality and translation equivariance. We demonstrate that by injecting a one-hot condition into the cellular perception field, a single set of local rules can learn to break symmetry and self-assemble into...
  </details>  
  Keywords: texture synthesis  
- **[TEXTRIX: Latent Attribute Grid for Native Texture Generation and Beyond](https://arxiv.org/abs/2512.02993v1)**  
  Authors: Yifei Zeng, Yajie Bao, Jiachen Qian, Shuang Wu, Youtian Lin, Hao Zhu, Buyu Li, Feihu Zhang, Xun Cao, Yao Yao  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2512.02993v1.pdf)  
  <details><summary>Abstract</summary>

  Prevailing 3D texture generation methods, which often rely on multi-view fusion, are frequently hindered by inter-view inconsistencies and incomplete coverage of complex surfaces, limiting the fidelity and completeness of the generated content. To overcome these challenges, we introduce TEXTRIX, a native 3D attribute generation framework for high-fidelity texture synthesis and downstream applications such as precise 3D part segmentation. Our approach constructs a latent 3D attribute grid and leverages a Diffusion Transformer equipped with sparse attention, enabling direct coloring of 3D models in volumetric space and fundamentally avoiding the limitations of multi-view fusion. Built upon this native representation, the framework naturally extends to high-precision 3D segmentation by training the same architecture to predict semantic attributes on the grid. Extensive experiments demonstrate state-of-the-art performance on both tasks, producing seamless, high-fidelity textures and accurate 3D part segmentation with precise boundaries.
  </details>  
  Keywords: texture synthesis  
- **[DensiCrafter: Physically-Constrained Generation and Fabrication of Self-Supporting Hollow Structures](https://arxiv.org/abs/2511.09298v2)**  
  Authors: Shengqi Dang, Fu Chai, Jiaxin Li, Chao Yuan, Wei Ye, Nan Cao  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2511.09298v2.pdf)  
  <details><summary>Abstract</summary>

  The rise of 3D generative models has enabled automatic 3D geometry and texture synthesis from multimodal inputs (e.g., text or images). However, these methods often ignore physical constraints and manufacturability considerations. In this work, we address the challenge of producing 3D designs that are both lightweight and self-supporting. We present DensiCrafter, a framework for generating lightweight, self-supporting 3D hollow structures by optimizing the density field. Starting from coarse voxel grids produced by Trellis, we interpret these as continuous density fields to optimize and introduce three differentiable, physically constrained, and simulation-free loss terms. Additionally, a mass regularization penalizes unnecessary material, while a restricted optimization domain preserves the outer surface. Our method seamlessly integrates with pretrained Trellis-based models (e.g., Trellis, DSO) without any architectural changes. In extensive evaluations, we achieve up to 43% reduction in material mass on the text-to-3D task. Compared to state-of-the-art baselines, our method could improve the stability and...
  </details>  
  Keywords: text-to-3d, texture synthesis  
- **[AvatarTex: High-Fidelity Facial Texture Reconstruction from Single-Image Stylized Avatars](https://arxiv.org/abs/2511.06721v1)**  
  Authors: Yuda Qiu, Zitong Xiao, Yiwei Zuo, Zisheng Ye, Weikai Chen, Xiaoguang Han  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2511.06721v1.pdf)  
  <details><summary>Abstract</summary>

  We present AvatarTex, a high-fidelity facial texture reconstruction framework capable of generating both stylized and photorealistic textures from a single image. Existing methods struggle with stylized avatars due to the lack of diverse multi-style datasets and challenges in maintaining geometric consistency in non-standard textures. To address these limitations, AvatarTex introduces a novel three-stage diffusion-to-GAN pipeline. Our key insight is that while diffusion models excel at generating diversified textures, they lack explicit UV constraints, whereas GANs provide a well-structured latent space that ensures style and topology consistency. By integrating these strengths, AvatarTex achieves high-quality topology-aligned texture synthesis with both artistic and geometric coherence. Specifically, our three-stage pipeline first completes missing texture regions via diffusion-based inpainting, refines style and structure consistency using GAN-based latent optimization, and enhances fine details through diffusion-based repainting. To address the need for a stylized texture dataset, we introduce TexHub, a high-resolution collection of 20,000 multi-style UV textures...
  </details>  
  Keywords: texture synthesis  
- **[Oitijjo-3D: Generative AI Framework for Rapid 3D Heritage Reconstruction from Street View Imagery](https://arxiv.org/abs/2511.00362v1)**  
  Authors: Momen Khandoker Ope, Akif Islam, Mohd Ruhul Ameen, Abu Saleh Musa Miah, Md Rashedul Islam, Jungpil Shin  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2511.00362v1.pdf)  
  <details><summary>Abstract</summary>

  Cultural heritage restoration in Bangladesh faces a dual challenge of limited resources and scarce technical expertise. Traditional 3D digitization methods, such as photogrammetry or LiDAR scanning, require expensive hardware, expert operators, and extensive on-site access, which are often infeasible in developing contexts. As a result, many of Bangladesh's architectural treasures, from the Paharpur Buddhist Monastery to Ahsan Manzil, remain vulnerable to decay and inaccessible in digital form. This paper introduces Oitijjo-3D, a cost-free generative AI framework that democratizes 3D cultural preservation. By using publicly available Google Street View imagery, Oitijjo-3D reconstructs faithful 3D models of heritage structures through a two-stage pipeline - multimodal visual reasoning with Gemini 2.5 Flash Image for structure-texture synthesis, and neural image-to-3D generation through Hexagen for geometry recovery. The system produces photorealistic, metrically coherent reconstructions in seconds, achieving significant speedups compared to conventional Structure-from-Motion pipelines, without requiring any specialized hardware or expert supervision. Experiments on landmarks...
  </details>  
  Keywords: texture synthesis, image-to-3d  
- **[Scale-DiT: Ultra-High-Resolution Image Generation with Hierarchical Local Attention](https://arxiv.org/abs/2510.16325v1)**  
  Authors: Yuyao Zhang, Yu-Wing Tai  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2510.16325v1.pdf)  
  <details><summary>Abstract</summary>

  Ultra-high-resolution text-to-image generation demands both fine-grained texture synthesis and globally coherent structure, yet current diffusion models remain constrained to sub-$1K \times 1K$ resolutions due to the prohibitive quadratic complexity of attention and the scarcity of native $4K$ training data. We present \textbf{Scale-DiT}, a new diffusion framework that introduces hierarchical local attention with low-resolution global guidance, enabling efficient, scalable, and semantically coherent image synthesis at ultra-high resolutions. Specifically, high-resolution latents are divided into fixed-size local windows to reduce attention complexity from quadratic to near-linear, while a low-resolution latent equipped with scaled positional anchors injects global semantics. A lightweight LoRA adaptation bridges global and local pathways during denoising, ensuring consistency across structure and detail. To maximize inference efficiency, we repermute token sequence in Hilbert curve order and implement a fused-kernel for skipping masked operations, resulting in a GPU-friendly design. Extensive experiments demonstrate that Scale-DiT achieves more than $2\times$ faster inference and lower...
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