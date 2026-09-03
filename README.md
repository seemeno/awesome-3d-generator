# Awesome Gaussian Splatting [![Awesome](https://awesome.re/badge.svg)](https://awesome.re)

A curated list of latest research papers, projects and resources related to Gaussian Splatting. Content is automatically updated daily.

> Last Update: 2026-09-03 01:38:20

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

- [Cross-Modal Generation](#cross-modal-generation) (92 papers) - Methods transforming text or 2D images into 3D assets.
- [Dynamic & Articulated Modeling](#dynamic-&-articulated-modeling) (13 papers) - Creation of 3D objects with moving parts or controllable skeletal structures.
- [Scene & View Synthesis](#scene-&-view-synthesis) (106 papers) - Focus on reconstructing complex scenes or synthesizing new perspectives from limited data.
- [Surface & Appearance Modeling](#surface-&-appearance-modeling) (21 papers) - Techniques for generating realistic textures, materials, and surface details.



## Table of Contents

- [Latest Papers](#latest-papers)
- [Categorized Papers](#categorized-papers)
- [Classic Papers](#classic-papers)
- [Open Source Projects](#open-source-projects)
- [Applications](#applications)
- [Tutorials & Blogs](#tutorials--blogs)



## Latest Papers
> 🔄 Updated Daily

### September 2026
- **[Kirin: Animal Motion Generation from In-the-Wild Video](https://arxiv.org/abs/2609.01823v1)**  
  Authors: Brian Nlong Zhao, Zhuoyang Pan, James M. Rehg, Jiajun Wu, Shangzhe Wu  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2609.01823v1.pdf) | [![Project](https://img.shields.io/badge/-Project-blue)](https://kirin-ani.github.io)  
  <details><summary>Abstract</summary>

  Understanding animal motion is fundamental to modeling animal behavior and biomechanics, yet progress in this area lags far behind human motion research due to the scarcity of high-quality motion data. While human motion can be captured in controlled environments, it is impractical for most animal species, resulting in small, domain-limited datasets that restrict downstream applications such as animation. To address this challenge, we introduce Kirin, a framework that reconstructs motion from video, learns motion priors at scale, and generates realistic motion that can be directly applied to animated assets. Using large collections of in-the-wild animal videos, we reconstruct 3D motion sequences and pair them with captions to create AiM3D, the first large-scale dataset offering aligned video-text-motion tuples for quadruped animals. Building on this dataset, we develop a visual-guided motion generation model that conditions on both text and image to guide the generation of realistic motion across diverse animal species. Finally,...
  </details>  
  Keywords: image-to-3d  
- **[Integrated Laser Scanning and Image-Based Topology Optimization Techniques for Detection and Quantification of Visible and Subsurface Structural Defects](https://arxiv.org/abs/2609.01808v1)**  
  Authors: Mehrdad Shafiei Dizaji, Devin Harris  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2609.01808v1.pdf)  
  <details><summary>Abstract</summary>

  Reliable characterization of structural defects requires methods capable of resolving both directly observable surface damage and damage that is not visible from the inspected surface. This study presents two complementary non-contact, vision-based approaches for the detection and quantitative characterization of defects in structural components. The first approach employs high-resolution laser scanning to generate three-dimensional (3D) point clouds of damaged steel specimens. Comparative processing of measured and reference point clouds is used to localize damaged regions, quantify geometric loss, and transfer the measured defect geometry to a finite element representation. The second approach combines full-field surface deformation measurements obtained using three-dimensional digital image correlation (3D-DIC) with finite element model updating and topology optimization. In this inverse framework, measured surface response is used to infer subsurface abnormalities through their influence on the spatial distribution of structural response. Experimental steel-beam specimens containing controlled smooth defects and randomly distributed defects are used to evaluate...
  </details>  
- **[ZipTok3D: High-Fidelity 3D Tokenization with Compact Token Prefixes](https://arxiv.org/abs/2609.01740v1)**  
  Authors: Mingda Lin, Weijie Wang, Zeyu Zhang, Bowen Cui, Yefei He, Haoyu Zhao, Yuanyu He, Donny Y. Chen, Feng Chen, Bohan Zhuang  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2609.01740v1.pdf)  
  <details><summary>Abstract</summary>

  Compact token sequences are essential for efficient 3D generation. However, existing 3D tokenizers typically organize latent representations either over spatial regions or as fixed-size sets of global tokens, both suffering sharp reconstruction degradation when compressed to extremely low token budgets. In this paper, we present ZipTok3D, a 3D tokenizer designed for high-fidelity reconstruction from extremely short token sequences. Its key idea is to organize object geometry into progressively informative global-token prefixes and unfold these compact representations through iterative decoding. Specifically, nested dropout randomly truncates the latent sequence after encoding during training and requires each retained prefix to reconstruct the complete object, thereby prioritizing essential geometric information in the leading tokens. The decoder then repeatedly applies a parameter-shared Transformer block to recover fine-grained geometry from each prefix without a separate generative sampling stage. With the same token dimension, ZipTok3D achieves reconstruction quality comparable to the 32-token COD-VAE baseline using only one...
  </details>  

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
- **[REST3D: Reconstructing Physically Stable 3D Scenes from a Single Image](https://arxiv.org/abs/2605.30338v2)**  
  Authors: Xiaoxuan Ma, Jiashun Wang, Nicolas Ugrinovic, Yehonathan Litman, Kris Kitani  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2605.30338v2.pdf)  
  <details><summary>Abstract</summary>

  Reconstructing physically stable 3D scenes from a single RGB image enables casual images to be converted into simulation-ready digital assets for applications such as immersive interaction and content creation. However, existing single-image reconstruction methods fall short in capturing the physical structure of a scene. As a result, they often produce geometrically plausible but physically inconsistent results, including object floating and penetration, which lead to unstable behavior in physics simulations. Image-conditioned scene generation methods improve physical plausibility but often rely on strong scene priors, yielding plausible yet inaccurate object arrangements that fail to match the input image. We propose REST3D, a single-image reconstruction framework that can reconstruct physically stable 3D scenes by integrating physical scene understanding with physics-constrained refinement. We first introduce an agentic physical scene understanding technique that constructs a scene-tree representation capturing object physical states and inter-object relationships from a gravity-support perspective, providing a structural prior for reconstruction. Leveraging...
  </details>  
  Keywords: image-to-3d  
- **[SuperVoxelGPT: Adaptive and Ordered 3D Tokenization for Autoregressive Shape Generation](https://arxiv.org/abs/2605.29655v3)**  
  Authors: Yuan Li, Congyi Zhang, Xifeng Gao, Xiaohu Guo  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2605.29655v3.pdf)  
  <details><summary>Abstract</summary>

  Autoregressive multimodal large language models (MLLMs) enable 3D generation but struggle to scale to high-resolution shapes due to inadequate 3D tokenizations. Compact set-based representations discard deterministic spatial ordering, leading to ambiguous sequence prediction, while uniform or octree-based voxel grids preserve ordering at the cost of severe redundancy and excessively long sequences. This structural trade-off limits stable and efficient autoregressive 3D generation. We present SuperVoxelGPT, a representation-first framework that resolves this tension through adaptive and deterministically ordered supervoxel tokenization. Given a prompt, we first predict a coarse geometric saliency distribution and construct a shape-adaptive supervoxel partition using saliency-guided centroidal Voronoi tessellation, allocating fine-grained cells to complex regions and larger cells to smooth regions. Conditioned on this prompt and ordered supervoxel layout, we introduce a SuperVoxelVAE and fine-tune a pretrained MLLM to autoregressively generate supervoxel tokens. Experiments using Trellis-500K data show that SuperVoxelGPT reduces token sequence length to 12.8% of uniform voxel...
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
- **[Feedforward 3D Editing Learns from Semantic-Part Transformation](https://arxiv.org/abs/2605.27351v5)**  
  Authors: Jiawei Weng, Saining Zhang, Zhenxin Diao, Peishuo Li, Henghaofan Zhang, Junhao Chen, Hao Zhao  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2605.27351v5.pdf)  
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
  Keywords: rigging, controllable 3d generation  
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
- **[Perceptually Lossless Tactile Texture Synthesis with Compact Spectral Envelope Models](https://arxiv.org/abs/2605.23804v2)**  
  Authors: Jagan K. Balasubramanian, Yasemin Vardar  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2605.23804v2.pdf)  
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
  Keywords: novel view synthesis, 3d scene reconstruction  
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
- **[Stream3D: Sequential Multi-View 3D Generation via Evidential Memory](https://arxiv.org/abs/2605.21472v5)**  
  Authors: Kaichen Zhou, Zeyang Bai, Xinhai Chang, Mengyu Wang, Paul Liang, Fangneng Zhan  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2605.21472v5.pdf) | [![Project](https://img.shields.io/badge/-Project-blue)](https://stream-3d.github.io/stream3d.github.io)  
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
- **[Structural Energy Guidance for View-Consistent Text-to-3D Generation](https://arxiv.org/abs/2605.19876v2)**  
  Authors: Qing Zhang, Jinguang Tong, Jing Zhang, Jie Hong, Xuesong Li  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2605.19876v2.pdf)  
  <details><summary>Abstract</summary>

  Text-to-3D generation based on diffusion models often suffers from the Janus problem, leading to inconsistent geometry across viewpoints. This work identifies viewpoint bias in 2D diffusion priors as the main cause and proposes Structural Energy-Guided Sampling (SEGS), a training-free and plug-and-play framework to improve multi-view consistency. SEGS constructs a structural energy in the PCA subspace of U-Net features and injects its gradient into the denoising process. It can be easily integrated into SDS/VSD pipelines without retraining. Experiments show that SEGS reduces the Janus Rate by about 10% on average and improves View-CS scores across multiple baselines, including DreamFusion, Magic3D, and LucidDreamer. This method effectively alleviates viewpoint artifacts while preserving appearance fidelity, providing a flexible solution for high-quality text-to-3D content generation.
  </details>  
  Keywords: text-to-3d  
- **[PhysOmni: Physics-Grounded Multi-Object Scene Generation from a Single Image with Real-Time Interaction](https://arxiv.org/abs/2605.20290v2)**  
  Authors: Xin Zhang, Yabo Chen, Yijie Fang, Wanying Qu, Haibin Huang, Chi Zhang, Feng Xu, Xuelong Li  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2605.20290v2.pdf) | [![Project](https://img.shields.io/badge/-Project-blue)](https://physomni.github.io)  
  <details><summary>Abstract</summary>

  Recent generative video models achieve impressive visual quality but remain constrained by limited physical consistency and controllability. Existing video generation methods provide minimal physical control, and single-image-to-3D conversion approaches often suffer from object interpenetration. Furthermore, physics-based scene-level 3D generation methods exhibit spatial misalignment, stylized artifacts, and inconsistencies with the input data, restricting their use in realistic interactive video synthesis. We propose PhysOmni, a training-free framework that converts a single image into a physically consistent and controllable video through holistic scene-level 3D reconstruction. By rep?resenting the full scene geometry in a unified spatial coordinate system, PhysOmni resolves object penetration and alignment ambiguity. Unlike prior methods, this formulation enables accurate scene?level multi-object interactions and introduces richer, complex control types for advanced mechanics?based manipulation. By decoupling simulation from rendering, PhysOmni bypasses latency-heavy priors, achieving real-time physical interaction previews paired while preserving photorealistic visual fidelity. Experimental results demonstrate that PhysOmni substantially outperforms prior methods...
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
  Keywords: image-to-3d, 3d scene reconstruction  
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
  Keywords: rigging, 3d asset generation  
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
- **[CADBench: A Multimodal Benchmark for AI-Assisted CAD Program Generation](https://arxiv.org/abs/2605.10873v2)**  
  Authors: Anna C. Doris, Jacob Thomas Sony, Ghadi Nehme, Era Syla, Amin Heyrani Nobari, Faez Ahmed  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2605.10873v2.pdf) | [![GitHub](https://img.shields.io/github/stars/anniedoris/CADBench?style=social)](https://github.com/anniedoris/CADBench)  
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
- **[DVD: Discrete Voxel Diffusion for 3D Generation and Editing](https://arxiv.org/abs/2605.07971v3)**  
  Authors: Zhengrui Xiang, Jiaqi Wu, Fupeng Sun, Heliang Zheng, Yingzhen Li  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2605.07971v3.pdf) | [![GitHub](https://img.shields.io/github/stars/TeCai/DVD?style=social)](https://github.com/TeCai/DVD)  
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
  Keywords: image-to-3d, 3d scene reconstruction  

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
- **[SceneExpander: Text-Guided 3D Scene Expansion via Free-Form View Insertion](https://arxiv.org/abs/2603.27084v3)**  
  Authors: Zijian He, Renjie Liu, Yihao Wang, Weizhi Zhong, Huan Yuan, Kun Gai, Guangrun Wang, Guanbin Li  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2603.27084v3.pdf)  
  <details><summary>Abstract</summary>

  World building with 3D scene representations is increasingly important for content creation, simulation, and interactive experiences, yet real workflows are inherently iterative: creators repeatedly extend existing scenes under user control. Motivated by this gap, we study text-guided 3D scene expansion via free-form view insertion. Starting from a real scene captured by multi-view images, a user specifies a text expansion intent, which a generative model materializes as an inserted view extending the scene coverage. Unlike simple object editing or style transfer within a fixed scene, the inserted view may be 3D-misaligned with the original reconstruction, introducing geometric shifts, hallucinated content, or view-dependent artifacts that disrupt global multi-view consistency. To address this challenge, we propose SceneExpander, which applies test-time adaptation to a parametric feed-forward 3D reconstruction model with two complementary distillation signals: anchor distillation stabilizes the captured scene using geometric cues from the captured views, while inserted-view self-distillation retains insertion-supported predictions to...
  </details>  
- **[GaussianGPT: Towards Autoregressive 3D Gaussian Scene Generation](https://arxiv.org/abs/2603.26661v2)**  
  Authors: Nicolas von Lützow, Barbara Rössle, Katharina Schmid, Matthias Nießner  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2603.26661v2.pdf)  
  <details><summary>Abstract</summary>

  Most recent advances in 3D generative modeling rely on diffusion or flow-matching formulations. We instead explore a fully autoregressive alternative and introduce GaussianGPT, a transformer-based model that directly generates 3D Gaussians via next-token prediction, thus facilitating full 3D scene generation. We first compress Gaussian primitives into a discrete latent grid using a sparse 3D convolutional autoencoder with vector quantization. The resulting tokens are serialized and modeled using a causal transformer with 3D rotary positional embedding, enabling sequential generation of spatial structure and appearance. Unlike diffusion-based methods that refine scenes holistically, our formulation constructs scenes step-by-step, naturally supporting completion, outpainting, controllable sampling via temperature, and flexible generation horizons. This formulation leverages the compositional inductive biases and scalability of autoregressive modeling while operating on explicit representations compatible with modern neural rendering pipelines, positioning autoregressive transformers as a complementary paradigm for controllable and context-aware 3D generation.
  </details>  
- **[ArtHOI: Taming Foundation Models for Monocular 4D Reconstruction of Hand-Articulated-Object Interactions](https://arxiv.org/abs/2603.25791v1)**  
  Authors: Zikai Wang, Zhilu Zhang, Yiqing Wang, Hui Li, Wangmeng Zuo  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2603.25791v1.pdf) | [![Project](https://img.shields.io/badge/-Project-blue)](https://arthoi-reconstruction.github.io)  
  <details><summary>Abstract</summary>

  Existing hand-object interactions (HOI) methods are largely limited to rigid objects, while 4D reconstruction methods of articulated objects generally require pre-scanning the object or even multi-view videos. It remains an unexplored but significant challenge to reconstruct 4D human-articulated-object interactions from a single monocular RGB video. Fortunately, recent advancements in foundation models present a new opportunity to address this highly ill-posed problem. To this end, we introduce ArtHOI, an optimization-based framework that integrates and refines priors from multiple foundation models. Our key contribution is a suite of novel methodologies designed to resolve the inherent inaccuracies and physical unreality of these priors. In particular, we introduce an Adaptive Sampling Refinement (ASR) method to optimize object's metric scale and pose for grounding its normalized mesh in world space. Furthermore, we propose a Multimodal Large Language Model (MLLM) guided hand-object alignment method, utilizing contact reasoning information as constraints of hand-object mesh composition optimization. To...
  </details>  
- **[ViewSplat: View-Adaptive 3D Gaussian Splatting for Feed-Forward Synthesis](https://arxiv.org/abs/2603.25265v2)**  
  Authors: Moonyeon Jeong, Seunggi Min, Suhyeon Lee, Hongje Seong  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2603.25265v2.pdf) | [![Project](https://img.shields.io/badge/-Project-blue)](https://cvlab-uos.github.io/ViewSplat)  
  <details><summary>Abstract</summary>

  We present ViewSplat, a view-adaptive 3D Gaussian splatting network for novel view synthesis from unposed images. While recent feed-forward 3D Gaussian splatting has significantly accelerated 3D scene reconstruction by bypassing per-scene optimization, a fundamental fidelity gap remains. We attribute this gap to the limited capacity of single-step feed-forward networks to regress static Gaussian primitives that satisfy all viewpoints. To address this limitation, we shift the paradigm from static primitive regression to view-adaptive splatting. Instead of a rigid Gaussian representation, our pipeline learns a view-adaptive latent representation. Specifically, ViewSplat initially predicts base Gaussian primitives alongside the weights of scene-conditioned View MLPs. During rendering, these MLPs take target-view coordinates as input and predict view-dependent residual updates for each Gaussian attribute (i.e., 3D position, scale, rotation, opacity, and color). This mechanism, which we term view-adaptive splatting, allows each primitive to rectify initial estimation errors, effectively capturing high-fidelity appearances. Extensive experiments demonstrate that ViewSplat...
  </details>  
  Keywords: novel view synthesis, 3d scene reconstruction  
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
- **[Know3D: Prompting 3D Generation with Knowledge from Vision-Language Models](https://arxiv.org/abs/2603.22782v2)**  
  Authors: Wenyue Chen, Wenjue Chen, Peng Li, Qinghe Wang, Xu Jia, Heliang Zheng, Rongfei Jia, Yuan Liu, Ronggang Wang  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2603.22782v2.pdf)  
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
- **[DreamPartGen: Semantically Grounded Part-Level 3D Generation via Collaborative Latent Denoising](https://arxiv.org/abs/2603.19216v2)**  
  Authors: Tianjiao Yu, Xinzhuo Li, Muntasir Wahed, Jerry Xiong, Yifan Shen, Ying Shen, Ismini Lourentzou  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2603.19216v2.pdf)  
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
- **[SEAR: Simple and Efficient Adaptation of Visual Geometric Transformers for Unpaired RGB+Thermal 3D Reconstruction](https://arxiv.org/abs/2603.18774v2)**  
  Authors: Vsevolod Skorokhodov, Chenghao Xu, Shuo Sun, Olga Fink, Malcolm Mielle  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2603.18774v2.pdf) | [![Project](https://img.shields.io/badge/-Project-blue)](https://doi.org/10.5281/ZENODO.21077295)  
  <details><summary>Abstract</summary>

  Foundational feed-forward visual geometry models enable accurate and efficient camera pose estimation and scene reconstruction by learning strong scene priors from massive RGB datasets. However, their effectiveness drops when applied to mixed sensing modalities, such as RGB-thermal (RGB-T) images. We observe that while a visual geometry grounded transformer pretrained on RGB data generalizes well to thermal-only reconstruction, it struggles to align RGB and thermal modalities when processed jointly. To address this, we propose SEAR, a simple yet efficient fine-tuning strategy that adapts a pretrained geometry transformer to multimodal RGB-T inputs. Despite being trained on a relatively small RGB-T dataset, our approach significantly outperforms state-of-the-art methods for 3D reconstruction and camera pose estimation, achieving significant improvements over all metrics and delivering higher detail and consistency between modalities with negligible overhead in inference time compared to the original pretrained model. Notably, SEAR enables reliable multimodal pose estimation and reconstruction even under challenging...
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
- **[Interact3D: Compositional 3D Generation of Interactive Objects](https://arxiv.org/abs/2603.16085v2)**  
  Authors: Hui Shan, Keyang Luo, Ming Li, Sizhe Zheng, Yanwei Fu, Zhen Chen, Xiangru Huang  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2603.16085v2.pdf)  
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
- **[SK-Adapter: Skeleton-Based Structural Control for Native 3D Generation](https://arxiv.org/abs/2603.14152v2)**  
  Authors: Anbang Wang, Yuzhuo Ao, Shangzhe Wu, Chi-Keung Tang  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2603.14152v2.pdf) | [![Project](https://img.shields.io/badge/-Project-blue)](https://sk-adapter.github.io)  
  <details><summary>Abstract</summary>

  Native 3D generative models have achieved remarkable fidelity and speed, yet they suffer from a critical limitation: inability to prescribe precise structural articulations, where precise structural control within the native 3D space remains underexplored. This paper proposes SK-Adapter, a simple yet efficient and effective framework that unlocks precise skeletal manipulation for native 3D generation. Moving beyond text or image prompts, which can be ambiguous for precise structure, we treat the 3D skeleton as a first-class control signal. SK-Adapter is a lightweight structural adapter network that encodes joint coordinates and topology into learnable tokens, which are injected into the frozen 3D generation backbone via cross-attention. This design allows the model to not only effectively ``attend'' to specific 3D structural constraints but also preserve its original generative priors. To bridge the data gap, we contribute the Objaverse-TMS dataset, a large-scale dataset of 24k text-mesh-skeleton pairs. Extensive experiments confirm that our method achieves...
  </details>  
- **[EI-Part: Explode for Completion and Implode for Refinement](https://arxiv.org/abs/2603.14021v1)**  
  Authors: Wanhu Sun, Zhongjin Luo, Heliang Zheng, Jiahao Chang, Chongjie Ye, Huiang He, Shengchu Zhao, Rongfei Jia, Xiaoguang Han  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2603.14021v1.pdf) | [![Project](https://img.shields.io/badge/-Project-blue)](https://cvhadessun.github.io/EI-Part)  
  <details><summary>Abstract</summary>

  Part-level 3D generation is crucial for various downstream applications, including gaming, film production, and industrial design. However, decomposing a 3D shape into geometrically plausible and meaningful components remains a significant challenge. Previous part-based generation methods often struggle to produce well-constructed parts, exhibiting poor structural coherence, geometric implausibility, inaccuracy, or inefficiency. To address these challenges, we introduce EI-Part, a novel framework specifically designed to generate high-quality 3D shapes with components, characterized by strong structural coherence, geometric plausibility, geometric fidelity, and generation efficiency. We propose utilizing distinct representations at different stages: an Explode state for part completion and an Implode state for geometry refinement. This strategy fully leverages spatial resolution, enabling flexible part completion and fine geometric detail generation. To maintain structural coherence between parts, a self-attention mechanism is incorporated in both exploded and imploded states, facilitating effective information perception and feature fusion among components during generation. Extensive experiments on multiple benchmarks...
  </details>  
- **[Scene Generation at Absolute Scale: Utilizing Semantic and Geometric Guidance From Text for Accurate and Interpretable 3D Indoor Scene Generation](https://arxiv.org/abs/2603.13910v2)**  
  Authors: Stefan Ainetter, Thomas Deixelberger, Edoardo A. Dominici, Philipp Drescher, Konstantinos Vardis, Markus Steinberger  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2603.13910v2.pdf)  
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
- **[InterEdit: Navigating Text-Guided 3D Dyadic Human Motion Editing](https://arxiv.org/abs/2603.13082v2)**  
  Authors: Yebin Yang, Di Wen, Lei Qi, Weitong Kong, Junwei Zheng, Ruiping Liu, Yufan Chen, Chengzhi Wu, Kailun Yang, Yuqian Fu, Danda Pani Paudel, Luc Van Gool, Kunyu Peng  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2603.13082v2.pdf) | [![GitHub](https://img.shields.io/github/stars/YNG916/InterEdit?style=social)](https://github.com/YNG916/InterEdit)  
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
- **[Spherical-GOF: Geometry-Aware Panoramic Gaussian Opacity Fields for 3D Scene Reconstruction](https://arxiv.org/abs/2603.08503v2)**  
  Authors: Zhe Yang, Guoqiang Zhao, Sheng Wu, Kai Luo, Kailun Yang  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2603.08503v2.pdf) | [![GitHub](https://img.shields.io/github/stars/1170632760/Spherical-GOF?style=social)](https://github.com/1170632760/Spherical-GOF)  
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
- **[Skill-Evolving Grounded Reasoning for Free-Text Promptable 3D Medical Image Segmentation](https://arxiv.org/abs/2603.08215v2)**  
  Authors: Tongrui Zhang, Chenhui Wang, Yongming Li, Zhihao Chen, Xufeng Zhan, Hongming Shan  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2603.08215v2.pdf) | [![Project](https://img.shields.io/badge/-Project-blue)](https://seer-medseg.github.io)  
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
- **[3DGS-HPC: Distractor-free 3D Gaussian Splatting with Hybrid Patch-wise Classification](https://arxiv.org/abs/2603.07587v2)**  
  Authors: Jiahao Chen, Yipeng Qin, Ganlong Zhao, Xin Li, Wenping Wang, Guanbin Li  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2603.07587v2.pdf) | [![Project](https://img.shields.io/badge/-Project-blue)](https://cnhaox.github.io/3DGS-HPC)  
  <details><summary>Abstract</summary>

  3D Gaussian Splatting (3DGS) has demonstrated remarkable performance in novel view synthesis and 3D scene reconstruction, but its quality often degrades in real-world environments due to transient distractors, such as moving objects and varying shadows. Existing methods commonly introduce semantic priors from pre-trained vision models either to group pixels into coherent regions or to define perceptual error metrics. However, semantic grouping is often misaligned with the binary static/transient distinction, while perceptual features can be fragile under appearance perturbations introduced during 3DGS optimization. We propose 3DGS-HPC, a framework that addresses these issues by combining two complementary principles: a patch-wise classification strategy that leverages local spatial consistency for robust region-level decisions, and a hybrid classification metric that adaptively integrates photometric and perceptual cues for more reliable separation. Extensive experiments demonstrate the superiority and robustness of our method in mitigating distractors to improve 3DGS-based novel view synthesis. Our project page is https://cnhaox.github.io/3DGS-HPC/ .
  </details>  
  Keywords: novel view synthesis, 3d scene reconstruction  
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
- **[3D Field of Junctions: A Noise-Robust, Training-Free Structural Prior for Volumetric Inverse Problems](https://arxiv.org/abs/2603.02149v2)**  
  Authors: Narges Moeini, Namhoon Kim, Justin Romberg, Sara Fridovich-Keil  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2603.02149v2.pdf) | [![GitHub](https://img.shields.io/github/stars/voilalab/3D-Field-of-Junctions?style=social)](https://github.com/voilalab/3D-Field-of-Junctions)  
  <details><summary>Abstract</summary>

  Volume denoising is a foundational problem in computational imaging, as many 3D imaging inverse problems face high levels of measurement noise. Inspired by the strong 2D image denoising properties of Field of Junctions (ICCV 2021), we propose a novel, fully volumetric 3D Field of Junctions (3D FoJ) representation that optimizes a junction of 3D wedges that best explain each 3D patch of a full volume, while encouraging consistency between overlapping patches. In addition to direct volume denoising, we leverage our 3D FoJ representation as a structural prior that: (i) requires no training data, and thus precludes the risk of hallucination, (ii) preserves and enhances sharp edge and corner structures in 3D, even under low signal to noise ratio (SNR), and (iii) can be used as a drop-in denoising representation via projected or proximal gradient descent for any volumetric inverse problem with low SNR. We demonstrate successful volume reconstruction and denoising...
  </details>  
- **[OnlineX: Unified Online 3D Reconstruction and Understanding with Active-to-Stable State Evolution](https://arxiv.org/abs/2603.02134v2)**  
  Authors: Chong Xia, Fangfu Liu, Yule Wang, Yize Pang, Yueqi Duan  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2603.02134v2.pdf)  
  <details><summary>Abstract</summary>

  Recent advances in generalizable 3D Gaussian Splatting (3DGS) have enabled rapid 3D scene reconstruction within seconds, eliminating the need for per-scene optimization. However, existing methods primarily follow an offline reconstruction paradigm, lacking the capacity for continuous reconstruction, which limits their applicability to online scenarios such as robotics and VR/AR. In this paper, we introduce OnlineX, a feed-forward framework that reconstructs both 3D visual appearance and language fields in an online manner using only streaming images. A key challenge in online formulation is the cumulative drift issue, which is rooted in the fundamental conflict between two opposing roles of the memory state: an active role that constantly refreshes to capture high-frequency local geometry, and a stable role that conservatively accumulates and preserves the long-term global structure. To address this, we introduce a decoupled active-to-stable state evolution paradigm. Our framework decouples the memory state into a dedicated active state and a persistent...
  </details>  
  Keywords: novel view synthesis, 3d scene reconstruction  
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
- **[TIMI: Training-Free Image-to-3D Multi-Instance Generation with Spatial Fidelity](https://arxiv.org/abs/2603.01371v2)**  
  Authors: Xiao Cai, Pengpeng Zeng, Ji Zhang, Heng Tao Shen, Jingkuan Song, Lianli Gao  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2603.01371v2.pdf)  
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
- **[WarpHammer: Densifying Scene Warps with 3D Object Priors for Extreme View Synthesis](https://arxiv.org/abs/2606.31258v1)**  
  Authors: Michael Green, Gavriel Habib, Dvir Samuel, Tal Berkovitz Shalev, Issar Tzachor, Rami Ben-Ari, Or Litany  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2606.31258v1.pdf)  
  <details><summary>Abstract</summary>

  Projection-conditioned novel view synthesis (NVS) warps an explicit 3D reconstruction of the input view into the target camera and conditions a generator on the warped rendering. This works well for small viewpoint changes but degrades sharply under large orbital motion: the warp becomes sparse around the orbited object, where hidden surfaces dominate the new view and mirror-like artifacts emerge, causing the generator to lose both pixel content and the implicit camera cue carried by the warp. We introduce WarpHammer, a training-free framework that resolves this failure mode by augmenting the warped scene with an explicit 3D reconstruction of the object obtained from a native 3D generative prior (e.g., SAM3D). The reconstructed object adds missing foreground surfaces and occludes background points that should no longer be visible, restoring both appearance and camera cues without fine-tuning the base model. The same explicit object representation further unlocks a capability current NVS pipelines do...
  </details>  
  Keywords: novel view synthesis  
- **[Knowledge-Driven Dimension Estimation from a Single Image -3D Asset Generation Technology for Digital Twin Construction](https://arxiv.org/abs/2606.30896v1)**  
  Authors: Hidenori Sakaniwa, Akihito Akai, Akihiko Hyodo  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2606.30896v1.pdf)  
  <details><summary>Abstract</summary>

  In the verification of in-vehicle cameras, simulation technology using virtual spaces has advanced, enabling pre-evaluation of false detections and missed detections in various scenarios. However, discrepancies in the scale of the object being verified between the virtual and real environments can lead to a decrease in camera recognition performance. For traffic signs installed at high altitudes, distance measurement using LiDAR or stereo cameras is difficult, requiring size estimation from monocular images. This paper proposes a method for estimating the scale of an object by decomposing it into multiple structural elements and integrating external knowledge regarding design rules, geometric relationships, and conventional dimensions. Specifically, this method detects each component from a monocular image and estimates the size of each component by considering its structural relationships and dimensional consistency with surrounding elements. Furthermore, it generates a 3D asset of the object by reconstructing the estimated components. This method makes it possible to...
  </details>  
  Keywords: 3d asset generation  
- **[Open-Vocabulary and Referring Segmentation for 3D Gaussians Using 2D Detectors](https://arxiv.org/abs/2606.30638v1)**  
  Authors: Jameel Hassan, Yasiru Ranasinghe, Vishal Patel  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2606.30638v1.pdf)  
  <details><summary>Abstract</summary>

  3D Gaussian Splatting (3DGS) has emerged at the forefront of 3D scene reconstruction. Extending 3DGS with language-driven, open-vocabulary understanding has gained significant attention for real-world applications such as embodied AI. Recent methods achieve this by learning an instance feature attribute and assigning semantics by distilling high-dimensional Contrastive Language-Image Pretraining (CLIP) features directly into the scene representation. However, the instance grouping mechanisms of these methods either require a predefined number of instances or suffer from noise in their bottom-up grouping strategies. Furthermore, the reliance on CLIP restricts semantic understanding to simple noun phrases, preventing complex spatial reasoning and referential expression grounding. We present GaussDet, a method that circumvents the need for dense CLIP features by leveraging discrete, open-vocabulary 2D object detectors with referring expression capabilities. We learn instance features for individual Gaussians to decompose the scene into 3D instance groups. By rendering these groups and aggregating semantic votes from multi-view 2D...
  </details>  
  Keywords: 3d scene reconstruction  
- **[Arko-T: A Foundation Model for Text-to-Structured 3D Generation](https://arxiv.org/abs/2606.30429v2)**  
  Authors: Liang Wang, Zhaoyang Xi, Zekai Xiang, Heng Meng, Qishan Zhang, Pingyi Zhou, Jin Liu, Litao Chen  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2606.30429v2.pdf)  
  <details><summary>Abstract</summary>

  Text-to-3D systems can now synthesize a model from a single sentence, yet the result is a shape to render, not a design to edit. We present Arko-T, a 4B-parameter text-to-design model that maps natural-language intent directly into executable, parametric CAD programs. Rather than optimizing for code executability alone, Arko-T aligns every stage of the pipeline to a formal notion of design state, so that data curation, code normalization, and execution-grounded supervision all work to preserve the features, parameters, and construction logic that make a CAD artifact editable. Benchmarked against seven frontier LLMs across 12 metrics, Arko-T attains the best score on 8 and the second-best on 3 more, at roughly one-tenth the per-benchmark cost. The results suggest that targeted design-level training at moderate scale can match frontier general-purpose models on structured CAD generation.
  </details>  
  Keywords: text-to-3d  
- **[FastPano3D: Feed-Forward Indoor Panoramic 3D Reconstruction from a Single Image](https://arxiv.org/abs/2606.30352v1)**  
  Authors: Jianqiang Li, Liumei Zhang, Wenjia Guo, Tianlong Feng, Yongzhi Liao, Di Lu, Hanchi Ren, Jingjing Deng  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2606.30352v1.pdf)  
  <details><summary>Abstract</summary>

  Recent advances in 3D scene reconstruction have highlighted the intricate trade-offs among rendering quality, inference efficiency, and data dependency. To address the challenge of rapidly reconstructing detailed 3D indoor scenes from minimal input, we introduce FastPano3D, an end-to-end framework that directly generates renderable 3D Gaussian representations from a single panoramic image. Unlike perspective-based methods, panoramic images inherently suffer from equirectangular projection distortions and spatially non-uniform feature distributions, making direct feed-forward Gaussian generation particularly challenging. In contrast to existing Gaussian Splatting based methods that rely on multi-view supervision or per-scene optimization, FastPano3D employs a lightweight feature encoder, adaptive Gaussian sampling, and a point-cloud-guided refinement strategy to achieve efficient and accurate scene generation without any test-time optimization. Our approach reconstructs high-fidelity 3D scenes within seconds, achieving up to 156 times faster inference than prior state-of-the-art methods such as Pano2Room, while using only half the parameters. Extensive experiments demonstrate that FastPano3D delivers rendering...
  </details>  
  Keywords: 3d scene reconstruction  
- **[Variance Reduction on the Camera Axis: Multi-View Score Distillation for 3D](https://arxiv.org/abs/2606.29964v1)**  
  Authors: Marian Lupascu, Mihai Sorin Stupariu, Ionut Mironica  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2606.29964v1.pdf)  
  <details><summary>Abstract</summary>

  Score distillation turns a pretrained 2D diffusion model into a 3D generator, but the per-step gradient is estimated from a single randomly chosen view: it is high-variance and blind to global shape consistency. Prior work addresses this by retraining the diffusion prior on multi-view data; this improves consistency but makes the sampling contribution inseparable from prior quality. We instead isolate the sampling axis. The per-step gradient is one noisy sample of an expectation over views; aggregating K samples per step at a fixed total UNet budget reduces variance without touching the prior. We introduce Multi-View Aggregated Score Distillation (MV-SDI), which aggregates gradients from K views per step via gradient accumulation, keeping peak memory unchanged and the 2D prior frozen, and draws views as antithetic antipodal pairs, a prior-independent geometric property, for balanced angular coverage. At a fixed 10,000-UNet-call budget, K=2 raises CLIP R-Precision from 74.8% to 83.8% and CLIP score...
  </details>  
  Keywords: 3d generator  
- **[Analytic Concept-Centric Memory for Agentic Embodied Manipulation](https://arxiv.org/abs/2606.29774v1)**  
  Authors: Mingyang Sun, Xiujian Liang, Jiude Wei, Qichen He, Donglin Wang, Cewu Lu, Jianhua Sun  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2606.29774v1.pdf)  
  <details><summary>Abstract</summary>

  Long-horizon embodied manipulation requires agents to remember persistent objects, track changing scene states, and reuse prior interaction knowledge. However, existing agent memories are often stored as unstructured histories or embedding-based records, making it difficult to retrieve manipulation-relevant object parts, physical states, action effects, and executable skills. We propose an analytic concept-centric memory framework for agentic embodied manipulation. Our memory organizes experience around structured analytic concepts, where objects are represented by semantic parts, parametric templates, grounded poses, affordances, and manipulation states. It further connects object and scene memories with transition memory for action-induced state changes and skill memory for template-grounded and policy-grounded execution. At runtime, the agent performs structured coarse-to-fine retrieval to identify relevant objects, states, transitions, and skills, supporting state-consistent reasoning and skill reuse. Experiments on memory-dependent manipulation, articulated-object generalization, real-world memory evaluation, and ablations show that our approach improves task completion, retrieval accuracy, object re-identification, and cross-object skill generalization...
  </details>  
- **[NaLA: A 3D Native LLM Layout Agent for High-quality 3D Scene Generation](https://arxiv.org/abs/2606.29395v2)**  
  Authors: Cheng Wan, Yongsen Mao, Wenzheng Wu, Yuxuan Xie, Chucheng Xiang, Runze Wang, Xiang Zhang, Zhongyuan Liu, Rushi Dai, Yuan Liu  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2606.29395v2.pdf)  
  <details><summary>Abstract</summary>

  Recently, Large Language Models (LLMs) have emerged as promising layout agents for 3D scene generation. Existing layout agents still suffer from implausible layout generation because most of them convert 3D assets and 3D layouts into textual descriptions as inputs and outputs, which involves severe information loss due to the modality gap between texts and 3D assets and 3D layouts. We propose NaLA, a native 3D LLM layout Agent for high-quality 3D scene generation by placing 3D assets in the scene. For the inputs, NaLA encodes 3D scene boundaries and 3D assets directly into the LLM, preserving fine-grained geometry and enabling explicit reasoning over relationships like collisions, surface supporting, and containment. To accurately output the positions and orientations of assets, NaLA adopts a coarse-to-fine prediction mechanism that first predicts discrete poses in an autoregressive manner and then refines the discrete poses with a continuous regression. Trained on diverse layout datasets, NaLA...
  </details>  
- **[Evidence-Based Text-Conditioned 3D CT Synthesis for Ovarian Cancer](https://arxiv.org/abs/2606.28980v1)**  
  Authors: Francesca Pia Panaccione, Eugenio Lomurno, Francesca Fati, Carlotta Pecchiari, Marina Rosanu, Luigi De Vitis, Lucia Ribero, Gabriella Schivardi, Giovanni Damiano Aletti, Nicoletta Colombo, Maria Francesca Spadea, Francesco Multinu, Matteo Matteucci, Elena De Momi  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2606.28980v1.pdf)  
  <details><summary>Abstract</summary>

  Ovarian cancer is frequently diagnosed at an advanced stage, making preoperative contrast-enhanced computed tomography (CT) central to staging and surgical planning; yet the scarcity of annotated imaging data, compounded by privacy regulations, limits the development of generalizable computational models in this domain. Text-conditioned 3D CT synthesis has shown promise, but existing pipelines depend on paired radiology reports and have been evaluated only on chest CT. We propose OvESyn (Ovarian Evidence-based Synthesis), a framework that constructs standardized Findings and Impression sections directly from CT-derived imaging descriptors and routine clinical metadata, without any original radiology report, and uses them to condition a latent diffusion model adapted to 493 high-grade serous ovarian carcinoma patients. This is the first text-conditioned 3D CT synthesis framework adapted to an abdomino-pelvic oncologic setting. A systematic ablation over two adaptation axes, vision-language encoder alignment and generator fine-tuning, identifies generator domain adaptation as the operative mechanism for crossing the...
  </details>  
- **[HAT-4D: Lifting Monocular Video for 4D Multi-Object Interactions via Human-Agent Collaboration](https://arxiv.org/abs/2606.28215v1)**  
  Authors: Jiaxin Li, Yuxiang Wu, Zhenkai Zhang, Xinrui Shi, Haoyuan Wang, Yichen Zhao, Su Linxiang, Chenyang Yu, Mingyu Zhang, Yifan Ding, Boran Wen, Li Zhang, Ruiyang Liu, Yong-Lu Li  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2606.28215v1.pdf) | [![Project](https://img.shields.io/badge/-Project-blue)](https://lijiaxin0111.github.io/HAT4D)  
  <details><summary>Abstract</summary>

  Extracting dynamic 4D object interactions from massive, in-the-wild monocular videos offers a highly efficient data collection pathway for scaling Embodied AI and training VLAs. However, existing monocular 4D reconstruction methods primarily focus on isolated objects, often failing under the severe occlusions and complex dynamics inherent in multi-object interactions. To bridge this gap, we propose HAT-4D, the first agentic framework designed to reconstruct the 3D geometry, temporal dynamics, and physical interactions of multiple objects from a single video. By integrating VLMs with a multi-level human-in-the-loop feedback mechanism, HAT-4D efficiently resolves depth ambiguities and interaction-induced occlusions during 3D generation and 4D propagation, yielding physically plausible assets without relying on expensive multicamera rigs. As a scalable data engine, HAT-4D facilitates the creation of MVOIK-4D, an open-world benchmark for monocular 4D interaction reconstruction, accompanied by a novel multi-dimensional evaluation protocol focused on physical plausibility and temporal consistency. Extensive experiments demonstrate that HAT-4D achieves SOTA...
  </details>  
- **[Home3D 1.0: A High-Fidelity Image-to-3D Asset Generation System for Interior Design](https://arxiv.org/abs/2606.27923v2)**  
  Authors: Yiyun Fei, Guoqiu Li, Jin Song, Chuqiao Wu, Delong Wu, Hong Wu, Ziru Zeng, Haohui Chen, Yindong Kong, Jing Li, Qi Wu, Feng Zhang, Jianan Jiang, Ruigao Yang  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2606.27923v2.pdf)  
  <details><summary>Abstract</summary>

  We present Home3D 1.0, a modular image-to-3D generation system that produces high-quality 3D assets from a single reference image, targeting interior design and e-commerce applications. Given a photograph of a furniture or decor item, the system outputs a mesh with physically-based rendering (PBR) materials, and the mesh can be decomposed into material-specific components. The pipeline is organized into four tightly coupled modules: Geometry reconstructs a watertight mesh through latent SDF modelling with a geometry VAE and a coarse-to-fine flow-matching DiT; Texture predicts multiview albedo observations, reprojects them onto the mesh, and completes unseen surface regions with a 3D texture field; Material uses MatWeaver to obtain component masks through video-based segmentation and UV-space voting, then retrieves and bakes PBR maps from a curated material library through hierarchical multi-modal matching; and Parts generates material-editable semantic part meshes with a PartVAE and PartDiT, decoding multi-head part-specific SDF fields in one pass. Each module...
  </details>  
  Keywords: image-to-3d, 3d asset generation  
- **[Scene and Human in One World: Reconstruction in a Feedforward Pass](https://arxiv.org/abs/2606.27720v1)**  
  Authors: Boao Shi, Qiao Feng, Yiming Huang, Lingjie Liu  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2606.27720v1.pdf)  
  <details><summary>Abstract</summary>

  Reconstructing humans in dynamic scenes from moving monocular cameras remains challenging due to scale ambiguity, human-scene misalignment, and occlusion interference. Rather than treating human mesh recovery and scene reconstruction as separate tasks, we believe that accurate human-scene reconstruction requires the two tasks to mutually inform each other: parametric human models offer semantic structure and metric-scale priors, while scene geometry provides spatial context for human localization and alignment. Built on this insight, we introduce SHOW, a mask-promptable human mesh recovery framework that couples feed-forward 3D scene reconstruction with Human Mesh Recovery in a unified metric space. SHOW injects human semantics and scale priors from parametric human models into normalized point-map prediction, enabling metric-scale scene reconstruction from inherently scale-ambiguous monocular input. In turn, the recovered scene geometry constrains human mesh estimation, encouraging spatially consistent human placement and improved human-scene alignment. To handle complex multi-person and cluttered scenes, SHOW further incorporates a promptable...
  </details>  
  Keywords: 3d scene reconstruction  
- **[GeoFace: Consistent Multi-View Face Generation with Geometry-Constrained Diffusion](https://arxiv.org/abs/2606.27659v1)**  
  Authors: Yeji Choi, Jinhyeok Choi, Jaewon Min, Minkyung Kwon, Jin Hyeon Kim, Seungryong Kim  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2606.27659v1.pdf)  
  <details><summary>Abstract</summary>

  We present GeoFace, a geometry-constrained multi-view diffusion framework for consistent face generation from a single input. % While recent multi-view diffusion models achieve photorealistic synthesis at the per-view level, they lack an explicit mechanism to enforce a shared 3D structure across views, often leading to inconsistent geometry across viewpoints. To address this, GeoFace proposes a unified dual-stream framework for joint generation of multi-view RGB images and 3D face geometry, where the appearance and geometry streams interact through shared attention layers. To encourage the two streams to mutually constrain each other, we introduce a geometry-guided attention alignment loss that supervises the cross-attention between appearance and geometry tokens with 3D-consistent correspondences, enabling the appearance stream to correctly reference pose-invariant geometric cues for robust alignment across viewpoints. Geometry is represented as a canonical UV position map, derived from a FLAME mesh fitted to multi-view observations, serving as a view-invariant shared constraint across all...
  </details>  
- **[Sculpting NeRF Geometry: Human-Preference Fine-Tuning of a 3D-Aware Face GAN](https://arxiv.org/abs/2606.27305v1)**  
  Authors: Archer Moore, Mingming Gong, Liam Hodgkinson  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2606.27305v1.pdf)  
  <details><summary>Abstract</summary>

  Reinforcement learning from human feedback (RLHF) for 3D generation is now established across a number of works, but most existing pipelines optimise explicit surface representations, often by converting radiance fields into meshes and training heavily on surface-supervised data. We instead fine-tune a pretrained 3D-aware generative model directly from a learned reward over radiance-field density ($σ$) values, with no externally supplied mesh or shape prior. The reward model requires no pretraining, trains easily on a small set of preference samples, and yields robust improvement in 3D geometry. Working on an unconditional 3D-aware face GAN (EG3D), our reward reads the continuous 3D density field of the neural radiance field (NeRF) directly and supplies a geometry-only learning signal, requiring neither text conditioning, mesh extraction, nor multi-view rendering. A density-consistency constraint keeps the 2D appearance qualitatively similar while the geometry is reshaped, at a measurable but bounded distributional cost (FID-50k rises from 4.09 to...
  </details>  
- **[Pseudo-Text-Conditioned 3D Grounding DINO for Organ Localization in Abdominal CT](https://arxiv.org/abs/2606.27084v1)**  
  Authors: Siqi Chen, Han Gong, Keyi Hou, Jingxuan Yang, Sheethal Bhat, Andreas Maier  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2606.27084v1.pdf)  
  <details><summary>Abstract</summary>

  Reliable organ localization in abdominal CT can provide spatial priors for downstream trauma analysis. We propose CT-3GDINO, a lightweight 3D detector that adapts a Grounding-DINO-style query-based architecture to fixed organ localization using frozen pseudo-text class tokens instead of a real text encoder. The model combines a Swin3D visual backbone, bidirectional feature enhancement, pseudo-text-guided query selection, and a cross-modality decoder to predict normalized 3D boxes for liver, spleen, left kidney, right kidney, and bowel. We train and evaluate on 193 matched RSNA/RATIC CT volumes with segmentation-derived boxes. The best multi-scale model, trained from scratch, achieves 0.5830 overall top-1 class-wise mAP over 3D IoU thresholds from 0.1 to 0.7, outperforming fixed- and trainable-backbone classification-pretrained variants with 0.5570 and 0.4657 mAP. Performance is strong for coarse localization, with 0.9649 AP at IoU 0.1, but remains limited for strict box alignment, with 0.1552 AP at IoU 0.7. These results establish CT-3GDINO as an open-source...
  </details>  
- **[Beyond a Shadow of a Doubt: Close Proximity Geometry Reconstruction Using FMCW Radar Shadow Effects](https://arxiv.org/abs/2606.25829v1)**  
  Authors: Felix de Trogoff du Boisguezennec, Benjamin Ramtoula, Daniele De Martini  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2606.25829v1.pdf)  
  <details><summary>Abstract</summary>

  Reliable perception in adverse conditions remains challenging for autonomous systems, as cameras and LiDAR degrade in poor lighting or weather. Millimetre-wave FMCW radar is robust to such conditions, but its elevation collapse limits geometric reasoning. We observe that vehicle chassis occlude radar rays and form a distinctive geometric shadow, and its consistency can enable us to infer useful information about objects whose returns intersect this shadow. Motivated by this observation, we propose a method to recover the 3D, in-plane inclination of nearby slender vertical objects from this cue. The object inclination is retrieved without assumptions about the wider scene, but through an analytical, closed-form mapping between its radar return boundaries and the opening angle. Validation in simulation and experimentation on a Navtech CTS350-X radar shows that inclinations can be estimated under practical conditions, with segmentation of the object in the radar scan emerging as the main bottleneck. This work highlights...
  </details>  
  Keywords: 3d scene reconstruction  
- **[Follow Your Track: Precise Skeleton Animation Controlled by 3D Trajectories](https://arxiv.org/abs/2606.25344v1)**  
  Authors: Yueting Liu, Yanqin Jiang, Nian Liu, Jingmen Zhou, Zhengjun Zha, Weiming Hu, Jin Gao  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2606.25344v1.pdf)  
  <details><summary>Abstract</summary>

  4D generation aims to animate 3D objects with realistic motion, holding great promise for applications. Existing methods typically decouple 3D asset generation from motion synthesis: acquire a 3D asset, prepare a structural representation like mesh and Gaussians, and synthesize motion from text or video control signals. However, dense mesh and Gaussian representations incur high computational costs and are prone to temporal artifacts, limiting animation quality and duration to only short clips. Meanwhile, text lacks fine-grained spatial and temporal details such as timing and coordination, while video entangles motion with appearance and background. Together, these limitations result in 4D animations that suffer from poor temporal consistency, wrong identification, and limited controllability. We address these issues with \texttt{ACT}, a trajectory-conditioned framework for topology-general skeletal animation. ACT uses skeletons as a compact structured and compute-efficient representation and 3D point trajectories from monocular video as explicit motion guidance which provide detailed motion patterns without...
  </details>  
  Keywords: 3d asset generation  
- **[CoGeoAD: Hierarchical Color-Geometric Fusion with Multi-View Attention for Zero-Shot 3D Anomaly Detection](https://arxiv.org/abs/2606.25273v1)**  
  Authors: Ke Xu, Xinle Wang, Yanning Hou, Xueliang Ma, Juan Xie, Jianfeng Qiu  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2606.25273v1.pdf) | [![GitHub](https://img.shields.io/github/stars/kingdomShu/CoGeoAD?style=social)](https://github.com/kingdomShu/CoGeoAD)  
  <details><summary>Abstract</summary>

  Zero-shot 3D anomaly detection is essential for industrial quality inspection, where labeled anomaly samples are scarce. Meanwhile, existing methods lack an effective mechanism to fuse complementary 2D color images with 3D geometric structures, limiting their ability to detect both surface and structural defects in a unified framework. To address these issues, we propose CoGeoAD, a unified CLIP-based framework that fuses color and geometric features by constructing pixel-aligned paired multi-view images. The framework introduces a Data-Driven Multi-View Attention (MVA) mechanism to adaptively aggregate 3D features and a Multi-Stage Color-Geometric Fusion (MS-CGF) module to hierarchically integrate multi-level features from both modalities. Extensive experiments on the MVTec3D-AD and Eyecandies benchmarks demonstrate that CoGeoAD achieves state-of-the-art performance, effectively capturing both structural and textural anomalies in complex industrial scenarios. our source code is available at https://github.com/kingdomShu/CoGeoAD.
  </details>  
- **[FLUX3D: High-Fidelity 3D Gaussian Generation with Diffusion-Aligned Sparse Representation](https://arxiv.org/abs/2606.24874v1)**  
  Authors: Haorui Ji, Weizhe Liu, Hongdong Li, Hengkai Guo  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2606.24874v1.pdf)  
  <details><summary>Abstract</summary>

  Sparse voxel representation has emerged as a scalable foundation for image-to-3D Gaussian Splatting (3DGS) generation, yet current methods struggle to preserve high-frequency visual details of input images due to two structural bottlenecks. First, they adopt discriminative 2D features optimized for semantic abstraction to construct sparse voxel latents, which suppress reconstructive cues and induce a representation bottleneck. Second, in the generation stage, standard diffusion transformers lack effective mechanisms to align dense 2D image tokens with sparse 3D voxel latents, resulting in a cross-modal correspondence bottleneck. To address these issues, we propose FLUX3D, a scalable image-to-3DGS framework that boosts both representation learning and cross-modal alignment during generation. We first revisit 2D feature selection for sparse-voxel-based 3D representation learning, propose Diffusion-Aligned Structured Latents (DA-SLAT) and couple it with a decoder-only architecture to improve 3DGS reconstruction fidelity. We also design a sparse-structure-aware diffusion framework, which integrates the Sparse-structure Multimodal Diffusion Transformer (SMDiT) and Modal-Aware...
  </details>  
  Keywords: image-to-3d  
- **[OrbitForge: Text-to-3D Scene Generation via Reconstruction-Anchored Video Synthesis](https://arxiv.org/abs/2606.24799v1)**  
  Authors: Chenrui Fan, Paolo Favaro  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2606.24799v1.pdf)  
  <details><summary>Abstract</summary>

  Generic text-to-video models can be used as rich open-world scene priors. Despite the high quality of today's generated videos, they do not directly yield reliable 3D assets: camera motion is difficult to control, view coverage is partial, and frames often contain inconsistencies across time. We introduce OrbitForge, an adapter built from frozen video priors and per-prompt Gaussian Splatting reconstruction optimization that converts a single text-generated video into a canonical closed-orbit 3D Gaussian Splatting scene. We use 3D reconstruction as an anchor to improve the 3D consistency of the generated video. We obtain a preliminary 3D reconstruction from a first generated video via Deformable Gaussian Splatting with a robust MedianGS proxy. We render views from a prescribed orbit to detect missing viewpoints. OrbitForge uses the text-to-video model to complete only the missing views, and reconstructs the completed orbit into a final Gaussian Splatting scene. This design requires no task-specific video or...
  </details>  
  Keywords: text-to-3d  
- **[MM-TRELLIS: Point-Cloud Guided Multi-Modal 3D Vehicle Generation in Autonomous Driving](https://arxiv.org/abs/2606.24301v1)**  
  Authors: Hongli Xiao, Youjian Zhang, Yucai Bai, Chaoyue Wang, Yaohui Jin, Xiaoguang Ren, Wenjing Yang, Long Lan  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2606.24301v1.pdf) | [![GitHub](https://img.shields.io/github/stars/HongliXiao/MM-TRELLIS?style=social)](https://github.com/HongliXiao/MM-TRELLIS)  
  <details><summary>Abstract</summary>

  Recovering realistic 3D vehicle models from autonomous driving scenes is crucial for synthesizing training data and building simulation environment. However, most existing vehicle generation methods fail to fully exploit multimodal sensors i.e. multi-view images and LiDAR point clouds) and rely on neural rendering based reconstruction, leading to low-quality mesh. Recently, native 3D generative models have made significant progress, yet they are not built for arbitrary multi-view inputs and often struggle with in-the-wild driving images. In this work, we present MM-TRELLIS, a multi-modal version of TRELLIS for in-the-wild 3D vehicle generation that integrates LiDAR and image sensors from autonomous driving datasets into native 3D generative models. Specifically, multi-view images are cycled as conditioning inputs, while LiDAR point clouds provide test-time guidance to ensure geometric accuracy and cross-view consistency. During denoising, we first align the guidance point cloud with the model priors, then enforce consistency between the generated geometry and the guidance...
  </details>  
- **[3DCarGen: Scalable 3D Car Generation via 3D-consistent Multi-view Synthesis](https://arxiv.org/abs/2606.24257v2)**  
  Authors: Hongli Xiao, Youjian Zhang, Yaohui Jin, Xiaoguang Ren, Wenjing Yang, Long Lan  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2606.24257v2.pdf) | [![Project](https://img.shields.io/badge/-Project-blue)](https://honglixiao.github.io/3dcargen.github.io)  
  <details><summary>Abstract</summary>

  High-quality 3D vehicle assets are essential for autonomous driving simulation. Although multi-view diffusion-based paradigms enable controllable single-image reconstruction, they typically produce limited viewpoints and exhibit cross-view geometric inconsistencies, thereby reducing reconstruction fidelity in real-world scenarios. In this work, we introduce 3DCarGen, a scalable single-view 3D car generation framework designed for real-world images by synthesizing an arbitrary number of 3D-consistent multi-view images. Specifically, given a single image as input, we first synthesize a set of images from fixed viewpoints. These images are then fed into a feed-forward reconstruction model, resulting in a coarse 3D representation based on 3D Gaussian Splatting. Conditioned on this explicit 3D prior, our multi-view diffusion model generates 3D-consistent images from arbitrary camera viewpoints. We further extend a fast mesh reconstruction algorithm by incorporating color-normal joint optimization to recover detailed and coherent 3D vehicle models from the synthesized dense views. Extensive experiments on synthetic and real-world datasets demonstrate...
  </details>  
  Keywords: multi-view synthesis  
- **[Social Structure Matters in 3D Human-Human Interaction Generation](https://arxiv.org/abs/2606.24255v1)**  
  Authors: Zhongju Wang, Beier Wang, Yatao Bian, Pichao Wang, Zhi Wang, Daoyi Dong, Hongdong Li, Huadong Mo, Zhenhong Sun  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2606.24255v1.pdf)  
  <details><summary>Abstract</summary>

  Although text-to-motion generation has achieved strong progress in synthesizing realistic single-person motions from language, extending it to text-driven 3D human-human interaction (HHI) remains non-trivial, as HHI requires modeling the underlying \textbf{social structure} that governs phase progression, actor roles, and inter-actor coordination. In this paper, we formulate HHI generation as a social structure modeling and grounding problem: the model must first infer how an interaction unfolds and how the two actors coordinate their roles, and then realize this structure as continuous, physically plausible, and partner-aware 3D motion. To study how such structure should be modeled, we first examine the capability boundary of large language models (LLMs) for HHI generation. Our analysis shows that LLMs can \textit{think} by recovering phase decompositions and partner-aware roles, but cannot directly \textit{move}, as they fail to generate dynamic, physically plausible, and interaction-aware motion. This motivates our planner-executor paradigm, \textbf{Think with LLM, Move with Motion Skill}. The...
  </details>  
- **[Inclusive Interactive Collisions for Multi-View Consistent Compositional 3D Generation](https://arxiv.org/abs/2606.24206v1)**  
  Authors: Chang Liu, Mingwen Shao, Xiang Lv, Xinyuan Chen, Lingzhuang Meng, Qiao Zhang, Zhengyi Gong, Jinghao Hu  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2606.24206v1.pdf)  
  <details><summary>Abstract</summary>

  Recent breakthroughs in 3D generation have advanced notably with the development of text-to-image diffusion model. However, existing methods remain two practical challenges: (1) They primarily generate single 3D object, but struggle to generate multi-object compositional 3D assets due to the lack of the modeling for Gaussian primitives in reasonable interactions. (2) They often suffer from cross-view inconsistency during 3D optimization, as Score Distillation Sampling inherently performs on each single view, inevitably resulting in cross-view hallucinations. To solve above issues, we propose I2C-3D, a novel optimization-based method to generate multi-view consistent compositional 3D assets with reasonable interactions. Specifically, we propose an Inclusive Interactive Collisions strategy to guide Gaussian primitives appearing in reasonable interaction regions naturally, thereby ensuring objects in the compositional scene interact in a physically plausible and visually coherent way. Additionally, to enhance multi-view consistency, Multi-View Adaptive Score Distillation Sampling is devised to distill multi-view consistency prior and layout prior...
  </details>  
- **[Zero-Shot Test-Time Canonicalization using Out-of-Distribution Scoring](https://arxiv.org/abs/2606.24178v1)**  
  Authors: Dominik Lindner, Johann Schmidt, Tom Siegl, Martin Becker, Sebastian Stober  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2606.24178v1.pdf) | [![GitHub](https://img.shields.io/github/stars/johschm/its?style=social)](https://github.com/johschm/its)  
  <details><summary>Abstract</summary>

  Pretrained vision models often misclassify inputs that are rotated, scaled, or sheared, even though these affine transformations leave the object class unchanged. Robustness is usually restored either by building equivariance into the architecture or by retraining with augmentation, both of which require changing or retraining the model. Test-time canonicalization instead leaves the classifier untouched. It undoes the transformation of each input, mapping it to a canonical form near the training distribution before classification. Existing canonicalizers, however, rely on a narrow set of logit-based energy scores and bespoke search procedures, leaving the design space of scoring functions and optimizers unexplored. We reframe canonicalization as out-of-distribution (OOD) detection, which lets any OOD score serve as the energy minimized over transformations. Across benchmarks ranging from handwritten characters and sketches to natural images and 3D point clouds, we systematically evaluate around twenty OOD scores and nine search algorithms, finding that distance-based scores paired with...
  </details>  
- **[Arbor: Explicit Geometric Conditioning for Controllable 3D Asset Generation](https://arxiv.org/abs/2606.23514v1)**  
  Authors: Jan-Niklas Dihlmann, Andreas Engelhardt, Simon Donne, Hendrik P. A. Lensch, Mark Boss  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2606.23514v1.pdf)  
  <details><summary>Abstract</summary>

  Text and image conditioned 3D models now generate convincing assets, but they still offer little direct control over the space an object should occupy or avoid. In authoring, this spatial intent is often known before generation starts. A chair should fit a seating envelope, a prop should leave clearance for motion, or a part should expose a contact surface. Prompts and image views are poor carriers for such constraints, requiring the need for an explicit control interface. We present Arbor, a trainable attachment for text conditioned latent 3D generation. Arbor introduces constraint meshes as a native 3D control interface. The interface uses hull regions where geometry should exist, avoidance regions that should remain empty, and touch regions the object should contact. Unlike completion or whole object scaffold control, these meshes are not target evidence. They are local typed requirements and can include regions where no surface should appear. Arbor keeps...
  </details>  
  Keywords: 3d asset generation  
- **[Can Single-View Mesh Reconstruction Generalize to Robot Camera Rotation?](https://arxiv.org/abs/2606.22987v1)**  
  Authors: Yu Zhan, Guangcheng Chen, Hanjing Ye, Zhiqin Cheng, Zanjia Tong, Wenjun Xu, Hong Zhang  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2606.22987v1.pdf)  
  <details><summary>Abstract</summary>

  Single-view mesh reconstruction predicts object meshes and spatial layouts from a single observation, making it attractive for fast robot spatial reasoning and real-to-sim digital twins. However, robot-mounted cameras naturally rotate during manipulation and navigation, while learned single-view reconstruction models often rely on view-dependent priors and may generalize poorly to out-of-distribution camera rotations. Such rotations can introduce 3D inconsistencies, incorrect layouts, and violations of physical constraints, but this failure mode remains under-evaluated. We introduce an evaluation protocol with controlled axis-wise roll, pitch, and yaw sweeps to trace errors in monocular depth estimation (MDE), canonical object meshes, camera-space layout, and physical plausibility within a representative SAM3D-style pipeline. On the Aria Digital Twin dataset and a real Franka wrist-camera sequence, camera rotations induce MDE distortion, layout drift, and collision penetration, while canonical mesh predictions remain relatively stable. A two-stage SAM3D+FoundationPose pipeline is more robust than one-stage feed-forward layout prediction, and our Gravity-Aware Refinement...
  </details>  
  Keywords: single-view reconstruction  
- **[$φ$-Scene: Physically Grounded Image-to-3D Scene Reconstruction](https://arxiv.org/abs/2606.21596v2)**  
  Authors: Haodong Li, Lulu Shao, Haolin Lu, Yu Fu, Yen-Ru Chen, Seemandhar Jain, Manmohan Chandraker  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2606.21596v2.pdf)  
  <details><summary>Abstract</summary>

  Recent image-to-3D scene methods recover high-fidelity 3D objects with plausible arrangements, but often leave floatings and interpenetrations that limit physical validity and downstream use in interactive environments. We present $φ$-Scene, a physically grounded approach for open-vocabulary and compositional image-to-3D scene reconstruction that treats a scene not merely as a set of objects with predicted poses, but as a globally stable physical system. $φ$-Scene formulates reconstruction as topology-driven physical assembly: given an initial scene of reconstructed objects, it infers how objects support one another and settles them one by one in topological order. For each object, SDF-based optimization first resolves penetrations against the already-settled support context, and rigid-body simulation then settles the object into a stable equilibrium under real-world physical constraints. The resulting scene stays aligned to the reference image, with every object resting at a physically valid, stable contact configuration. On the 3D-Front benchmark, $φ$-Scene achieves the strongest overall performance...
  </details>  
  Keywords: image-to-3d, 3d scene reconstruction  
- **[Enhancing Creativity in 3D Generative Design via a TRIZ-Inspired Text-to-CAD Framework](https://arxiv.org/abs/2606.21378v1)**  
  Authors: Dongeon Lee, Leekyo Jeong, Soyoung Yoo, Sunwoong Yang, Namwoo Kang  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2606.21378v1.pdf)  
  <details><summary>Abstract</summary>

  Recent advances in large language models (LLMs) have demonstrated significant potential in supporting engineering design tasks, including computer-aided design (CAD) automation. However, most existing LLM-based 3D CAD generation approaches primarily focus on geometric precision and instruction-following performance, often overlooking the fundamental aspect of creative design exploration. This study presents a TRIZ-inspired text-to-CAD framework that leverages LLMs to generate high-quality, editable CAD models while systematically exploring creative design alternatives. The framework integrates the Theory of Inventive Problem Solving (TRIZ)-embedding deep human insights from extensive patent records-into LLM prompting strategies, enabling autonomous generation of innovative CAD variants that address technical contradictions. Through a comprehensive three-stage pipeline of design generation, enhancement, and optimization, the framework produces structurally diverse CAD models from well-crafted prompts. The present study implements and evaluates the first two stages, while positioning the design optimization stage as future work. A product design case study (chair) demonstrates that the TRIZ-inspired text-to-CAD...
  </details>  
- **[JanusMesh: Fast and Zero-Shot 3D Visual Illusion Generation via Cross-Space Denoising](https://arxiv.org/abs/2606.20563v2)**  
  Authors: Siang-Ling Zhang, Huai-Hsun Cheng, Tsung-Ju Yang, Yu-Lun Liu  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2606.20563v2.pdf) | [![Project](https://img.shields.io/badge/-Project-blue)](https://siang1105.github.io/JanusMesh.github.io)  
  <details><summary>Abstract</summary>

  Creating 3D visual illusions, a single 3D mesh that reveals entirely different semantics from various viewing angles, is a fascinating but tough challenge. Existing optimization-based methods are slow and can produce oversaturated colors. In contrast, naive stitching approaches fail to produce geometrically coherent objects. This results in visible unnatural seams and semantic leaks. In this paper, we present a fast and training-free framework for generating text-driven 3D visual illusions. Our approach decouples the generation into two stages. First, we propose a cross-space dual-branch denoising process. This process dynamically decodes 3D latents into voxel space for CLIP-guided orientation alignment and Signed Distance Field (SDF) blending, which ensures seamless geometric fusion. Second, we introduce a view-conditioned texture synthesis module that projects and aggregates view-specific 2D diffusion priors onto the fused geometry. Extensive experiments demonstrate that our method generates highly realistic, dual-semantic 3D illusions in just 3-5 minutes. It significantly outperforms existing methods...
  </details>  
  Keywords: texture synthesis  
- **[Judging to Improve: A De-biased VLM-as-3D-Judge Protocol for Single-Image 3D Generation](https://arxiv.org/abs/2606.20364v1)**  
  Authors: Ali Asaria, Tony Salomone, Deep Gandhi  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2606.20364v1.pdf)  
  <details><summary>Abstract</summary>

  A companion study established a de-biased, cross-model VLM-as-3D-judge that reliably ranks single-image-to-3D mesh quality where cheap geometry and CLIP proxies fall short. This paper asks: can that judge's preferences specialize a strong open generator, TRELLIS, on one asset class (furniture), cheaply and without human labels? Taking the judge from ranking to optimization is where the work lives. Pushing a VLM judge into the training and evaluation loop exposes failure modes ranking never triggered, so our contribution is an optimization-grade hardening of the judge: a training judge (Qwen2.5-VL-7B) held distinct from an evaluation judge (InternVL3-8B) to break circularity; position-bias correction; and fixes for three failure modes (image overload, geometry-hiding splat renders, and reference-free judging that rewards clean-but-wrong outputs), with calibration evidence (clear-gap win-rate 0.83-1.0; base-vs-base ~0.5). Using this protocol as an independent evaluator, and working only from public models and data with lightweight parameter-efficient adaptation, we find our methods match the...
  </details>  
  Keywords: image-to-3d  
- **[NeuMesh++: Towards Versatile and Efficient Volumetric Editing with Disentangled Neural Mesh-based Implicit Field](https://arxiv.org/abs/2606.19316v1)**  
  Authors: Chong Bao, Yuan Li, Bangbang Yang, Yujun Shen, Hujun Bao, Zhaopeng Cui, Yinda Zhang, Guofeng Zhang  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2606.19316v1.pdf) | [![Project](https://img.shields.io/badge/-Project-blue)](https://zju3dv.github.io/neumeshplusplus)  
  <details><summary>Abstract</summary>

  Recently neural implicit rendering techniques have evolved rapidly and demonstrated significant advantages in novel view synthesis and 3D scene reconstruction. However, existing neural rendering methods for editing purposes offer limited functionalities, e.g., rigid transformation and category-specific editing. In this paper, we present a novel mesh-based representation by encoding the neural radiance field with disentangled geometry, texture, and semantic codes on mesh vertices, which empowers a set of efficient and comprehensive editing functionalities, including mesh-guided geometry editing, designated texture editing with texture swapping, filling and painting operations, and semantic-guided editing. To this end, we develop several techniques including a novel local space parameterization to enhance rendering quality and training stability, a learnable modification color on vertex to improve the fidelity of texture editing, a spatial-aware optimization strategy to realize precise texture editing, and a semantic-aided region selection to ease the laborious annotation of implicit field editing. Extensive experiments and editing examples...
  </details>  
  Keywords: novel view synthesis, 3d scene reconstruction  
- **[FlowObject: Flow Steering for Bridging Generative Priors and Reconstruction Fidelity](https://arxiv.org/abs/2606.19019v1)**  
  Authors: Yuchen Rao, Xuqian Ren, Yinyu Nie, Sayan Deb Sarkar, Biao Zhang, Vincent Lepetit, Friedrich Fraundorfer  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2606.19019v1.pdf)  
  <details><summary>Abstract</summary>

  Recovering complete 3D representations of objects from few casual image captures remains a significant challenge. Recent 3D generative models, particularly those based on Flow-Matching (FM), can synthesize high-quality textured assets; however, they often suffer from ''synthetic bias'' where learned priors override observational evidence, alongside a lack of alignment with the observed instance. Conversely, optimization-based methods like 3D Gaussian Splatting (3DGS) provide high fidelity on visible surfaces but fail to reason about unobserved geometry. In this paper, we present FlowObject, a framework that reformulates sparse-view 3D reconstruction as a training-free, guided inverse problem. Our approach applies a dual-space guidance strategy to steer the Ordinary Differential Equation (ODE) trajectory of a flow-matching model, enabling the completion of unseen regions through learned generative priors while enforcing strict consistency with real-world observations. By integrating a 3DGS refinement stage, FlowObject further bridges the gap between ''synthetic-looking'' generative outputs and photorealistic reconstructions. Comprehensive benchmarks on synthetic...
  </details>  
- **[Splaxel: Efficient Distributed Training of 3D Gaussian Splatting for Large-scale Scene Reconstruction via Pixel-level Communication](https://arxiv.org/abs/2606.18588v1)**  
  Authors: Wenqi Jia, Zhewen Hu, Ying Huang, Yu Gong, Stavros Kalafatis, Yuke Wang, Wei Niu, Chengming Zhang, Ang Li, Sheng Di, Yuede Ji, Bo Fang, Miao Yin  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2606.18588v1.pdf)  
  <details><summary>Abstract</summary>

  3D Gaussian Splatting (3DGS) enables high-fidelity and real-time 3D scene reconstruction, but scaling training to large-scale scenes requires optimizing hundreds of millions of Gaussians across multiple GPUs. Existing distributed approaches either partition scenes into isolated regions, causing global inconsistency, or rely on global Gaussian-level exchanges, which lead to substantial growth in inter-GPU communication and quickly dominate iteration time.   We propose Splaxel, a communication-efficient distributed 3DGS training framework based on pixel-level local rendering and global composition. Instead of synchronizing Gaussians, each GPU renders its local subset and exchanges only partial pixel values, maintaining mathematical consistency while keeping communication cost stable as the scene size increases. Splaxel further reduces pixel-level redundancy through geometric and transmittance visibility prediction and improves GPU utilization via conflict-free camera-view consolidation. Evaluated on large-scale datasets with up to 120M Gaussians, Splaxel achieves up to 7.6$\times$ speedup over the state-of-the-art distributed 3DGS framework while preserving high reconstruction quality.
  </details>  
  Keywords: 3d scene reconstruction  
- **[A Cross-Model VLM-Judge Protocol for Single-Image 3D Mesh Quality (and Why Cheap Proxies Fall Short)](https://arxiv.org/abs/2606.18451v1)**  
  Authors: Ali Asaria, Tony Salomone, Deep Gandhi  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2606.18451v1.pdf)  
  <details><summary>Abstract</summary>

  Single-image-to-3D generators are improving quickly, but there is no agreed, human-free way to tell whether one generated mesh is better than another. Practitioners commonly rely on cheap automatic proxies (render-space CLIP similarity and mesh geometry-validity statistics), yet how well these track perceived quality is unestablished. We make two contributions. First, we propose and validate a reproducible VLM-judge evaluation protocol: a fixed 24-view headless render rig, two independent vision-language judge families, and a mandatory position-bias correction that queries both presentation orders and keeps only order-consistent verdicts. The two judge families agree substantially with each other (Cohen's kappa = 0.66), well above the chance-agreement floor. Second, using this protocol as the reference, we show the cheap proxies do not substitute for it. Geometry validity is only a weak signal on average (because, as we show, it is bimodal) and stays below our pre-registered target, while render-CLIP is at chance. A learned Bradley-Terry...
  </details>  
  Keywords: 3d generator, image-to-3d  
- **[Instruct-Particulate: Scaling Feed-Forward 3D Object Articulation with Kinematic Control](https://arxiv.org/abs/2606.14699v1)**  
  Authors: Ruining Li, Yuxin Yao, Matt Zhou, Chuanxia Zheng, Christian Rupprecht, Joan Lasenby, Shangzhe Wu, Andrea Vedaldi  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2606.14699v1.pdf)  
  <details><summary>Abstract</summary>

  Reconstructing articulated 3D objects is important for animation, gaming, and robotic simulations. Recent neural networks can estimate the articulated structure of 3D objects, but their generalization remains limited by the scarcity of annotated data for this task. To address this gap, we introduce Instruct-Particulate, a model that takes a 3D mesh together with a target kinematic specification, including part descriptions, connectivity, joint types, and optional point prompts, and predicts the corresponding kinematic part segmentation and joint motion parameters. The kinematic specification disambiguates the task and allows the model to target annotations of different granularity, thereby making it possible to use more abundant heterogeneous training data. At test time, the kinematic specification can be obtained automatically from large-scale vision-language models, so the model can be applied to any input mesh. To train our model at scale, we construct a heterogeneous dataset of more than 150,000 articulated 3D objects, extending existing publicly...
  </details>  
  Keywords: image-to-3d  
- **[MUSE: Agentic 3D Scene Authoring via Memory-Grounded Incremental Requirement Satisfaction](https://arxiv.org/abs/2606.14168v1)**  
  Authors: Ruijie Xu, Xinnan Zhu, Jiayu Ying, Daoguo Dong, Yuzhou Ji, Xin Tan  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2606.14168v1.pdf)  
  <details><summary>Abstract</summary>

  Text-driven 3D scene generation is a promising technique for digital content creation, embodied AI simulation, and interactive design, yet practical workflows often require refining, extending, or correcting existing scenes while preserving non-target content. Existing methods can produce realistic and structurally plausible scenes, but they generally lack editability with requirement-level state tracking, so part-level failures often lead to full-scene regeneration or manual intervention. To tackle this challenge, we formulate controllable 3D scene authoring as incremental requirement satisfaction, unifying construction and editing. In this paper, we present MUSE, a memory-grounded multi-agent framework in which an Architect compiles instructions into structured requirements, a Sculptor executes local scene operations, and an Inspector verifies each step while updating Working, Scene, and Skill Memory. To evaluate requirement-level controllability and preservation-aware editing, we introduce AuthorBench, offering 145 constrained construction cases and a 1,584-case preservation-aware editing pool paired with external structured checks. On full construction cases, MUSE improves...
  </details>  
- **[World Tracing: Generative Pixel-Aligned Geometry Beyond the Visible](https://arxiv.org/abs/2606.13652v1)**  
  Authors: Hao Zhang, Mohamed El Banani, Jen-Hao Cheng, Paul Zhang, Yi Hua, Ben Mildenhall, Christoph Lassner, Narendra Ahuja, Gengshan Yang  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2606.13652v1.pdf)  
  <details><summary>Abstract</summary>

  Image-to-3D methods often trade off faithfulness and completeness: depth estimators are anchored to input pixels but stop at the visible surface, while image-to-3D models generate complete shapes that are often misaligned with the input. We introduce World Tracing, a generative pixel-aligned geometry representation that predicts 3D points aligned with observed pixels while completing geometry beyond the visible surface. For each input pixel, World Tracing predicts an ordered stack of camera-space 3D points, where the first layer represents the visible surface and subsequent layers represent front-to-back intersections with occluded surfaces. We instantiate this representation with a world-tracing diffusion transformer, WT-DiT, which treats multiple geometry layers as separate denoising tokens coupled through factorized and global attention. WT-DiT is trained with pixel-space flow matching and a mixed noise schedule that balances visible-surface reconstruction with occluded-geometry generation. World Tracing achieves strong performance on visible-surface reconstruction and complete geometry generation across object, scene, and dynamic...
  </details>  
  Keywords: 3d generator, image-to-3d  
- **[Pipette: An Embodied Simulation Platform, Benchmark, and Data-Efficient Augmentation Framework for Wet-Lab Robotics](https://arxiv.org/abs/2606.12936v3)**  
  Authors: Zhe Liu, Huanbo Jin, Zhaohui Du, Zhe Wang, Dongzhan Zhou, Minting Pan, He Xu, Peijia Li, Jiaming Gu, Quan Lu, Qi Wang, Bin Ji, Ting Xiao  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2606.12936v3.pdf)  
  <details><summary>Abstract</summary>

  Wet-lab robots can improve the reproducibility, throughput, and safety of biomedical experiments, but scaling their learning requires customizable simulators for safe and reproducible task generation, open editable laboratory assets, and efficient pipelines that turn limited demonstrations into usable training data. We present Pipette, an embodied simulation platform, benchmark, and data-efficient augmentation framework for wet-lab robot learning. Pipette provides over 100 open-source and re-editable wet-lab assets through an extensible asset-building pipeline with built-in Tencent Hunyuan support for text- and image-conditioned 3D asset generation, and supports three robotic-arm embodiments through a unified simulation interface for task construction, data collection, augmentation, and evaluation. A key component of Pipette is its simulation-based data augmentation pipeline, which replays human demonstrations in simulation, applies lighting, camera, speed, and action perturbations, and filters generated episodes with automatic task success checks, rapidly expanding usable training data from limited manual demonstrations. We further introduce a 12-task wet-lab embodied benchmark...
  </details>  
  Keywords: 3d asset generation  
- **[Performance Analysis and Optimization of 3D Generative Diffusion Models across GPU Architectures](https://arxiv.org/abs/2606.19365v1)**  
  Authors: Jeeho Ryoo, Yongchan Jung, Muhammad Ali Khaliq, Weidong Zhang, Jiatong Han, Byeong Kil Lee  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2606.19365v1.pdf)  
  <details><summary>Abstract</summary>

  Diffusion models have become essential for high-fidelity 3D MRI synthesis, yet their deployment remains constrained by substantial GPU resource demands arising from hundreds of U-Net evaluations per sample and a highly heterogeneous kernel behavior. This paper performs a comprehensive performance analysis of the state-of-the-art medical diffusion model, Med-DDPM, across three generations of NVIDIA architectures to study kernel-level runtime breakdowns, instruction-mix characteristics, memory system utilization, warp-level activities, and profiler priority-score estimates. We show that training is overwhelmingly dominated by cuDNN convolution and implicit-GEMM kernels, with inefficiencies arising from memory-access patterns, tensor-layout conversions, and limited Tensor Core utilization. Guided by these insights, we evaluate two architecture-aware optimizations TF32 Tensor Core activation and a 3D channels-last layout and demonstrate that they reduce SM cycles by up to 100x, cut dynamic instructions by 100x, raise Tensor Core utilization from 1.45 to 9.98x, and increase IPC by 7% on A100, all without degrading synthesis quality.
  </details>  
- **[HairPort: In-context 3D-aware Hair Import and Transfer for Images](https://arxiv.org/abs/2606.12562v2)**  
  Authors: Alireza Heidari, Amirhossein Alimohammadi, Ali Mahdavi-Amiri  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2606.12562v2.pdf)  
  <details><summary>Abstract</summary>

  Transferring hairstyles between images is an important but challenging task in computer graphics, computer vision, and visual effects. It enables users to explore new looks without physically altering their hair, with applications in virtual try-on systems, augmented reality, and entertainment. Most prior works operate best under small pose gaps, and they fall short under large viewpoint and scale differences, where missing hair content must be synthesized rather than transferred. We propose HairPort, a 3D-aware hairstyle transfer framework that attempts to solve these issues by explicitly separating hair removal from transfer and enforcing geometric consistency before synthesis. We introduce a Bald Converter, which produces realistic bald versions of faces through LoRA-based in-context adaptation of FLUX.1 Kontext. To train our Bald Converter, we introduce a new dataset, Baldy, containing 6,000 paired bald and original images across diverse identities and conditions. We also use a 3D-Aware Transfer Pipeline that reconstructs and re-renders the...
  </details>  
- **[ISAP-3D: Identity-Slot Aligned Part-Aware 3D Generation](https://arxiv.org/abs/2606.12099v1)**  
  Authors: Junlin Hao, Haoshuai Fu, Xibin Song, Wei Li, Ruigang Yang, Xinggong Zhang, Jinchuan Zhang  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2606.12099v1.pdf)  
  <details><summary>Abstract</summary>

  Part-aware 3D generation aims to synthesize structured objects with semantically meaningful components, yet often suffers from structural ambiguity due to identity-layout entanglement. Existing methods either infer part identity and spatial layout implicitly, which can lead to unstable part allocation (e.g., slot swapping or part merging), or rely on strong layout conditions that are difficult to obtain in practice. We attribute this ambiguity to identity-slot permutation freedom: without explicit identity-slot alignment, the correspondence between semantic parts and generation slots is not identifiable during training, allowing multiple slot assignments to fit the same supervision and leading to inconsistent decomposition. Based on this insight, we argue that stable part-aware generation requires identity-aligned one-to-one slot modelling. We therefore propose an identity-slot aligned framework, ISAP-3D, which anchors each part with semantic identity tokens and performs identity-conditioned one-to-one layout prediction, followed by layout-conditioned geometry synthesis. Structured local-global conditioning maintains identity alignment across semantic, spatial, and geometric...
  </details>  
- **[TextHOI-3D: Text-to-3D Hand-Object Interaction via Discrete Multi-View Generation and Joint Mesh Optimization](https://arxiv.org/abs/2606.11805v1)**  
  Authors: Zixiong Hao, Zhencun Jiang  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2606.11805v1.pdf)  
  <details><summary>Abstract</summary>

  Text-conditioned 3D generation has progressed rapidly for images and isolated objects, but producing a hand-object mesh remains challenging: the output must preserve language semantics, cross-view consistency, object geometry, articulated hand shape, and physically plausible contact. We present TextHOI-3D, a staged framework that uses generated multi-view observations as an explicit interface between text-conditioned visual generation and geometry-aware hand-object recovery. TextHOI-3D learns a compact VQ token space for fixed-camera hand-object observations, predicts multi-view visual tokens from text with a CLIP-conditioned visual autoregressive model, and recovers a unified hand-object mesh through prior initialization, multi-view joint optimization, and anti-penetration refinement. The design separates semantic generation from geometric recovery while keeping both stages connected by a discrete multi-view representation. On HO3D-derived evaluations, the multi-view setting reduces object CD from 17.26 mm to 4.92 mm and penetration volume from 5.3721 cm^3 to 0.2193 cm^3 compared with a single-view counterpart, while improving hand errors and surface F-scores....
  </details>  
  Keywords: text-to-3d  
- **[3D-CBM: A Framework for Concept-Based Interpretability in Generative 3D Modeling](https://arxiv.org/abs/2606.11446v1)**  
  Authors: Ahmad Al-Kabbany  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2606.11446v1.pdf)  
  <details><summary>Abstract</summary>

  This research introduces a framework for incorporating Concept Bottleneck Models (CBMs) into 3D generative architectures to address the inherent 'semantic gap' in deep geometric learning. As deep models become central to 3D content creation, explainability shifts from a peripheral feature to a fundamental requirement for trust and accountability in safety-critical domains such as healthcare and manufacturing. CBMs provide an intrinsic interpretability solution by constraining latent representations to align with human-defined concepts, yet their application to unstructured 3D data remains largely unexplored. We design, implement, and validate a formal 3D-CBM architecture that maps raw geometric inputs, including point clouds and meshes, into a multi-tiered taxonomy of interpretable primitives and functional attributes. The framework further identifies strategic datasets, such as PartNet and ShapeNet, specialized for concept-based supervision. Experimental results from a 3D part-manipulation proof-of-concept experiment demonstrate the framework's efficacy, achieving a concept prediction accuracy of 88.8\% and a Chamfer Distance of 0.0115....
  </details>  
- **[P3D-Bench: Benchmarking MLLMs for Parametric 3D Generation and Structural Reasoning](https://arxiv.org/abs/2606.11152v2)**  
  Authors: Yikang Yang, Zhanpeng Hu, Youtian Lin, Mengqi Zhou, Jingxi Xu, Feihu Zhang, Jiaheng Liu, Yao Yao  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2606.11152v2.pdf)  
  <details><summary>Abstract</summary>

  Multimodal large language models can write code to produce complex programs as well as use programs to do 3D modeling, which opens up a new avenue for 3D generation powered by their priors, world knowledge and reasoning. Yet existing benchmarks rarely evaluate 3D modeling through code. Such modeling demands more than runnable code: from a text or visual specification, a model must generate a parametric 3D program that is geometrically precise, semantically aligned and assembly-consistent. We introduce P3D-Bench, a benchmark for parametric 3D generation. Unlike a 3D mesh, a parametric 3D program exposes explicit dimensions, construction operations and part relations, revealing whether a model recovers a design's structure, not just its appearance. Under a unified protocol, P3D-Bench covers three task families (Text-to-3D, Image-to-3D and Assembly-3D) and scores each output for executability, geometric fidelity, topology, text-grounded constraints, multiview semantic alignment and part-level structure. We evaluate frontier MLLMs and text-only LLMs on...
  </details>  
  Keywords: text-to-3d, image-to-3d  
- **[DB-3DME: From Dataset to Benchmark for Human-aligned Automatic 3D Mesh Evaluation](https://arxiv.org/abs/2606.10142v1)**  
  Authors: Nanshan Jia, Zhenyu Zhao, Sui Huang, Jingshen Wang, Zeyu Zheng  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2606.10142v1.pdf)  
  <details><summary>Abstract</summary>

  Recent advances in 3D generation have led to substantial improvements in realism, controllability, and efficiency, yet the evaluation of 3D assets remains underexplored. Existing evaluation paradigms, including human evaluation, learned metrics, and vision-language models (VLMs) as judges, suffer from limitations in cost, scalability, resolution handling, or task-specific alignment. In this work, we focus on 3D mesh evaluation and introduce DB-3DME, the Dataset and Benchmark for 3D Mesh Evaluation. DB-3DME contains 2,619 synthetic 3D meshes paired with human ratings on Geometry and Prompt Adherence. Using this dataset, we systematically benchmark state-of-the-art VLMs and identify visual encoding of 3D representations as a key factor for human-aligned evaluation performance. Motivated by this finding, we fine-tune an open-weight VLM, Qwen-2.5-VL-7B, for 3D mesh evaluation by adapting the visual encoder while freezing the language model. The fine-tuned model substantially outperforms existing pre-trained VLMs across multiple evaluation dimensions, establishing a new benchmark for automatic 3D mesh...
  </details>  
- **[CP4D: Compositional Physics-aware 4D Scene Generation](https://arxiv.org/abs/2606.09187v1)**  
  Authors: Hanxin Zhu, Cong Wang, Tianyu He, Long Chen, Xin Jin, Chen Gao, Zhibo Chen  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2606.09187v1.pdf) | [![Project](https://img.shields.io/badge/-Project-blue)](https://anonymous.4open.science/w/CP4D)  
  <details><summary>Abstract</summary>

  4D generation (\textit{i.e.}, dynamic 3D generation) has recently emerged as a rapidly growing research frontier due to its powerful spatiotemporal modeling capabilities. However, despite notable advances, existing approaches typically fail to capture the underlying physical principles, producing results that are both physically inconsistent and visually implausible. To overcome this limitation, we present CP4D, a novel paradigm for photorealistic 4D scene synthesis with faithful adherence to complex physical dynamics. Drawing inspiration from the compositional nature of real-world scenes, where immutable static backgrounds coexist with dynamic, physically plausible foregrounds, CP4D reformulates 4D generation as the integration of a static 3D environment with physically grounded dynamic objects. On this basis, our framework follows a three-stage pipeline: \textbf{1)} Firstly, we leverage pre-trained expert models to generate high-fidelity 3D representations of the environment and foreground objects respectively. \textbf{2)} Subsequently, to produce physically plausible trajectories and realistic interactions for these objects, we propose a hybrid motion...
  </details>  
- **[Leveraging NeRF-Rendered Images for 3D Gaussian Splatting](https://arxiv.org/abs/2606.09034v1)**  
  Authors: Mizuki Morikawa, Yuta Shimizu, Chunyu Li, Yusuke Monno, Masatoshi Okutomi  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2606.09034v1.pdf)  
  <details><summary>Abstract</summary>

  Neural radiance field (NeRF) and 3D Gaussian splatting (3DGS) are two mainstream approaches for novel view synthesis. They often show complementary performance, i.e., 3DGS demonstrating faster rendering speed and NeRF demonstrating higher rendering quality. Motivated by this, we propose leveraging NeRF-rendered images for 3DGS. Specifically, we target street scenes and utilize a pre-trained street-specific NeRF method to produce training images for a target 3DGS method. In our 3DGS training, NeRF-rendered images are used to remove transient objects in street-level input views and to generate bird's-eye views as additional views, inheriting the higher-quality rendering of NeRF into 3DGS. We further incorporate a diffusion-based image enhancement to improve the image quality of the additional views. Experimental results on one synthetic and two real datasets demonstrate that our proposed method improves street-scene rendering while preserving the speed of 3DGS and the quality of NeRF.
  </details>  
  Keywords: novel view synthesis  
- **[EPS3D: End-to-End Feed-Forward 3D Panoptic Segmentation](https://arxiv.org/abs/2606.08980v1)**  
  Authors: Runsong Zhu, Jiaxin Guo, Xiaoyang Guo, Zhengzhe Liu, Ka-Hei Hui, Wei Yin, Kai Chen, Wei Chen, Weiqiang Ren, Yunhui Liu, Pheng-Ann Heng, Chi-Wing Fu  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2606.08980v1.pdf)  
  <details><summary>Abstract</summary>

  This paper introduces EPS3D, a new end-to-end feed-forward framework for open-vocabulary 3D panoptic segmentation. Unlike existing methods relying on additional preprocessing, we design an end-to-end architecture, with a distillation-based training strategy on diverse 3D scenes to predict 3D-aware semantic and instance features from multi-view images, improving 3D consistency and avoiding error accumulation. We further propose a mutual enhancement module to enforce inherent semantic-instance consistency. By aligning semantics within instances (Ins2Sem) and refining instance features with semantic guidance (Sem2Ins), we achieve more coherent 3D scene understanding. Ultimately, EPS3D outperforms SOTA baselines on two benchmarks (e.g., +13% mIoU for semantics on Replica) with high efficiency (e.g., 1s per scene), supporting tasks like robotic manipulation and 3D scene editing.
  </details>  
- **[WaveDiT: Distribution-Aware Wavelet Flow Matching for Efficient 3D Brain MRI Synthesis](https://arxiv.org/abs/2606.08670v2)**  
  Authors: Danilo Danese, Angela Lombardi, Giuseppe Fasano, Matteo Attimonelli, Tommaso Di Noia  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2606.08670v2.pdf) | [![GitHub](https://img.shields.io/github/stars/sisinflab/WaveDiT?style=social)](https://github.com/sisinflab/WaveDiT)  
  <details><summary>Abstract</summary>

  Large and demographically balanced datasets are essential for reliable neuroimaging biomarkers. Full-resolution 3D brain MRI synthesis can support data augmentation in this setting, but existing approaches either incur prohibitive computational cost at volumetric scale or rely on lossy latent compression that may compromise anatomical detail. As a result, practical 3D generative augmentation often requires specialized compute infrastructure. We propose WaveDiT, a conditional flow matching framework operating in the coefficient space of a 3D Haar Discrete Wavelet Transform. The model combines factorized spatio-depth attention with band-wise heteroscedastic uncertainty modeling derived from higher-order wavelet statistics. Predicted log-variance is integrated directly into both the flow objective and conditioning pathway, enabling adaptive precision consistent with the heavy-tailed and input-dependent variance structure of anatomical detail. This formulation supports full-resolution 3D synthesis under practical memory and time constraints on a single modern GPU. Evaluation on a multi-site cohort demonstrates improved alignment between generated and real MRI...
  </details>  
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
  Keywords: novel view synthesis, 3d scene reconstruction  
- **[MeshFlow: Efficient Artistic Mesh Generation via MeshVAE and Flow-based Diffusion Transformer](https://arxiv.org/abs/2606.04621v2)**  
  Authors: Weiyu Li, Antoine Toisoul, Tom Monnier, Roman Shapovalov, Rakesh Ranjan, Ping Tan, Andrea Vedaldi  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2606.04621v2.pdf) | [![GitHub](https://img.shields.io/github/stars/facebookresearch/meshflow?style=social)](https://github.com/facebookresearch/meshflow) | [![Project](https://img.shields.io/badge/-Project-blue)](https://mesh-flow.github.io)  
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

### July 2026
- **[OASIS: Occlusion-aware Single-image Hand Avatar Reconstruction via 3D Gaussian Splatting](https://arxiv.org/abs/2607.29633v1)**  
  Authors: Zhisheng Han, Shiyao Wu, Jiayan Qiu, Yakun Ju, Lu Liu, Le Zhang, Pengfei Feng, Huiyu Zhou, Zheheng Jiang  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2607.29633v1.pdf)  
  <details><summary>Abstract</summary>

  Single-image 3D hand avatar reconstruction is fundamentally ill-posed and particularly challenging due to limited visual evidence under severe self-occlusion and the complex pose-dependent deformation of highly articulated hands. Existing methods predominantly rely on implicit NeRF-style representations, whose volumetric fitting is computationally expensive and often struggles to preserve fine-grained hand details. In this work, we present OASIS, a tailored 3D Gaussian Splatting framework for single-image hand avatar reconstruction. To faithfully encode sparse image-specific appearance cues in single-view reconstruction, we construct geometry-aligned visual evidence tokens by explicitly aligning input image observations with 3D hand geometry and context-adaptively tokenizing the resulting visual evidence. Since severe self-occlusion makes the reliability of image evidence inherently visibility-dependent, we introduce a visibility-conditioned point-image attention to reliably transfer visual evidence to geometric tokens, yielding occlusion-aware Gaussian features for faithful and robust reconstruction. To further capture non-rigid deformation of articulated hands, we introduce a Feature-on-Mesh representation to enable Gaussian...
  </details>  
  Keywords: single-view reconstruction  
- **[ROAD: Reciprocal-Objective Alignment of Discriminative Semantics for 3D Shape Generation](https://arxiv.org/abs/2607.28581v1)**  
  Authors: Xiao Luo, Mingyang Du, Xin Zhou, Tianrui Feng, Xiwu Chen, Xiaofan Li, Jiangning Zhang, Dingkang Liang  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2607.28581v1.pdf) | [![GitHub](https://img.shields.io/github/stars/H-EmbodVis/ROAD?style=social)](https://github.com/H-EmbodVis/ROAD)  
  <details><summary>Abstract</summary>

  High-fidelity 3D generation predominantly relies on scaling model capacity and data, which incurs prohibitive computational costs. This paradigm typically requires learning geometry from scratch and overlooks the rich semantic and structural priors already encapsulated in discriminative 3D foundation models. We contend that leveraging the profound understanding of the 3D world possessed by these discriminative models can significantly reduce generative cost. To this end, we propose ROAD, a framework that reduces the training cost of 3D generation by transferring these rich discriminative priors into diffusion transformers. To address the inherent semantic-structural heterogeneity between generative and discriminative latents, we introduce a reciprocal-objective alignment strategy. This method synergizes Holistic Semantic Condensing to enforce global semantic coherence and Structural Optimal Alignment, which is formulated as a bipartite matching problem to rigorously align microscopic geometric details between disparate latent spaces. The 3D foundation model is only used for training-time supervision of alignment and is not...
  </details>  
- **[SpatialQ: Understanding 3D Gaussian Splatting Scene Quality via Visual-based MLLM](https://arxiv.org/abs/2607.26595v1)**  
  Authors: Jingxuan Su, Shenglin Wang, Tiesong Zhao, Ge Li, Wei Gao  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2607.26595v1.pdf)  
  <details><summary>Abstract</summary>

  3D Gaussian Splatting (3DGS) has emerged as an effective representation for novel view synthesis and 3D scene reconstruction, creating an increasing demand for reliable quality assessment. Unlike conventional image quality assessment (IQA), the quality of a 3DGS scene depends not only on the perceptual fidelity of rendered views, but also on scene-level factors such as spatial structure and cross-view consistency. Existing IQA methods are limited by their reliance on 2D perceptual cues, whereas general multimodal large language models (MLLMs) are not designed for stable quality regression and may produce unreliable judgments. To address these limitations, a multimodal quality assessment framework is developed for 3DGS scene understanding. First, a 3D-aware quality representation learning framework is introduced by augmenting a VGGT-based encoder with a dedicated quality head. Multi-view images are encoded into view-specific features and aggregated to capture cross-view consistency, while geometric cues are incorporated through joint modeling of depth and point-cloud-related...
  </details>  
  Keywords: novel view synthesis, 3d scene reconstruction  
- **[DreamStyle3D: Efficient 3D Stylized Asset Generation via Dual-Attention Disentanglement](https://arxiv.org/abs/2607.24721v3)**  
  Authors: Kai Wang, Ziheng Ouyang, Xuying Zhang, Ming-Ming Cheng, Qibin Hou  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2607.24721v3.pdf) | [![GitHub](https://img.shields.io/github/stars/NK-JittorCV/nk-3D?style=social)](https://github.com/NK-JittorCV/nk-3D)  
  <details><summary>Abstract</summary>

  With the growth of gaming, animation, and virtual reality industries, the demand for efficient generation of stylized 3D assets is rapidly increasing. However, existing approaches still struggle to jointly preserve style fidelity, geometric consistency, and generation efficiency, as most of them still rely on indirect 2D-to-3D stylization pipelines. This motivates a native 3D stylization framework that can explicitly disentangle style from geometry while remaining efficient. To this end, we propose DreamStyle3D, an efficient framework for stylized 3D asset generation built on a Decoupled Dual Cross-Attention mechanism. Our method explicitly separates geometric and stylistic features to enable efficient style injection while preserving structural consistency, and further adopts a lightweight training strategy to enhance style consistency and model generalization. In addition, we build an automated data pipeline and construct a dataset of about 15K content-style-stylized triplets for training and evaluation. Extensive experiments demonstrate that our DreamStyle3D can generate high-fidelity, geometrically consistent stylized...
  </details>  
  Keywords: 3d asset generation  
- **[MSVS-VAE: Multi-Scale Anchored VecSet for High-Fidelity 3D Reconstruction](https://arxiv.org/abs/2607.24436v1)**  
  Authors: Dehao Hao, Kaiyi Zhang, Tanghui Jia, Xiangjun Gao, Dongyu Yan, Weikai Chen, Zeyu Hu, Lingting Zhu, Yingda Yin, Runze Zhang, Li Yuan, Xin Wang, Long Quan  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2607.24436v1.pdf)  
  <details><summary>Abstract</summary>

  High-fidelity 3D generative modeling increasingly relies on the latent diffusion paradigm, where the reconstruction quality of the underlying 3D VAE becomes a primary bottleneck. Existing approaches largely follow two paradigms: sparse voxel-based representations achieve strong reconstruction quality but incur significant memory and computational overhead, while set-based representations are compact and continuous yet typically lag in fidelity due to latent sparsity and excessive global smoothness. We propose MSVS-VAE, a hierarchical set-based VAE that closes this fidelity gap without sacrificing compactness. Our key idea is to progressively densify anchored VecSet latents via hierarchical point-shuffle upsampling, increasing spatial capacity for fine-grained geometry modeling. To efficiently decode from the densified hierarchy, we replace global cross-attention with AVS-Conv, a geometry-aware local aggregation operator operating within local neighborhoods rather than the exhaustive latent set. We further introduce multi-scale query decoding to fuse coarse-to-fine latent features, where coarse scales provide stable global context, and fine scales refine...
  </details>  
- **[UMI3D: Robust 3D Generation on Unconstrained Multi-Image Inputs via Simultaneous Focus Cross-Attention Routing](https://arxiv.org/abs/2607.24298v1)**  
  Authors: Zefan Qu, Zhenwei Wang, Gerhard Petrus Hancke, Rynson W. H. Lau  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2607.24298v1.pdf)  
  <details><summary>Abstract</summary>

  Recent 3D foundation models can generate high-quality assets from a single image, but degrade markedly on unconstrained multi-image inputs, often producing distorted geometry, over-smoothed textures, and chaotic colors. We argue that this failure stems not from limited model capacity, but from a mismatch between single-image cross-attention and the multi-image setting: existing models lack a principled way to decide which image each 3D voxel should trust at each denoising step. Revisiting recent single-image 3D foundation models, we show that explicitly routing each voxel to its most informative image is sufficient to unlock strong performance on inconsistent multi-image inputs. Based on this observation, we propose UMI3D, a training-free and plug-and-play framework that restructures cross-attention for unconstrained multi-image 3D generation. Its core, Simultaneous Focus Cross-Attention (SFC-Attn), activates all conditioning images at each denoising step while allowing each voxel to focus on the single image that best explains it. To enable this routing, we...
  </details>  
- **[A Dual Path Framework with Hotspot Guided Fusion for Three Dimensional CT to PET Synthesis in Head and Neck Cancer](https://arxiv.org/abs/2607.21800v1)**  
  Authors: Mohd Maaz Khan, Oluwaseyi Oderinde  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2607.21800v1.pdf)  
  <details><summary>Abstract</summary>

  18F-FDG PET/CT plays a central role in staging, treatment planning, and response assessment for head and neck cancer by providing functional information that complements anatomical CT imaging. However, PET acquisition requires radiotracer administration, specialized infrastructure, and additional cost, limiting its availability for repeated imaging. We present a proof of concept deep learning framework for synthesizing PET like images directly from routine CT scans with the goal of providing complementary metabolic information that may support imaging triage and clinical decision support rather than replace diagnostic PET. Forty-four patients from the publicly available QIN-HEADNECK dataset were retrospectively analyzed using five fold cross-validation. We propose a fully three dimensional dual path architecture consisting of (i) a regression U-Net optimized for voxel-wise quantitative SUV estimation and (ii) a conditional generative adversarial network optimized for realistic PET texture. Their outputs are integrated using hotspot guided Laplacian pyramid blending, allowing quantitative information from the regression pathway...
  </details>  
  Keywords: texture synthesis  
- **[Recurrent Sinusoidal INRs for Efficient High-Fidelity Representation](https://arxiv.org/abs/2607.21485v1)**  
  Authors: Hyunmin Cho, Jaejun Yoo, Kyong Hwan Jin  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2607.21485v1.pdf)  
  <details><summary>Abstract</summary>

  We study sinusoidal recurrence as an iterative mechanism for harmonic spectral enrichment in implicit neural representations (INRs). Our analysis reveals that sinusoidal activations induce a harmonic line spectrum, providing a spectral account of how recurrent unrolling enriches the effective spectral support. We realize this principle with a shared sinusoidal block that iteratively refines the latent representation. We empirically validate the resulting spectral behavior against feed-forward INRs, non-sinusoidal recurrent variants, and equilibrium-style sinusoidal models. Complementing this analysis, we evaluate the proposed architecture across image and 3D representation tasks. On RGB image benchmarks, our method achieves higher fidelity than feed-forward baselines with fewer parameters and fewer optimization steps, and it further transfers favorably to super-resolution, NeRF, and SDF tasks.
  </details>  
- **[Axolotl3D: a Unified Framework for Faithful 3D Shape Completion](https://arxiv.org/abs/2607.20660v1)**  
  Authors: Anita Hu, Maria Shugrina  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2607.20660v1.pdf)  
  <details><summary>Abstract</summary>

  Recent 3D generative models produce high-quality geometry from a single image using large-scale priors and diffusion architectures. However, they assume complete visibility and single-view inputs, limiting applicability in multi-view, occluded, or editing scenarios. Although prior works address these challenges individually, they lack a unified framework for controllable 3D completion under diverse conditioning signals. We present Axolotl3D, a multi-modal and occlusion-aware 3D generation model that jointly conditions on images, visibility masks, camera parameters, and a partial point cloud. The point cloud serves as a geometric anchor promoting faithful shape completion, while camera parameters ensure consistent multi-view alignment in a shared 3D coordinate system. A unified training strategy synthesizes diverse conditioning regimes from large-scale 3D data, enabling robust cross-modal reasoning. Experiments on Toys4K and OmniObject3D demonstrate state-of-the-art performance under both clean and occluded settings, as well as strong results in real-world reconstruction and geometry-consistent editing.
  </details>  
- **[RealVDeblur: One-Step Diffusion for Generalizable Real-World Video Deblurring](https://arxiv.org/abs/2607.20628v1)**  
  Authors: Renbiao Jin, Mingxin Yang, Yutian Chen, Junhao Zhuang, Xin Cai, Mulin Yu, Linning Xu, Wenxian Yu, Danping Zou, Shi Guo, Tianfan Xue  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2607.20628v1.pdf) | [![Project](https://img.shields.io/badge/-Project-blue)](https://rbjin.github.io/RealVDeblur)  
  <details><summary>Abstract</summary>

  Real-world video deblurring remains challenging due to diverse motion patterns, complex degradations, and the scarcity of realistic training data, yet robust restoration is critical for downstream pipelines such as mobile imaging and 3D reconstruction. This work presents \textbf{RealVDeblur}, an efficient generative framework designed to improve in-the-wild robustness under diverse real capture conditions. First, a large-scale, physically grounded blur synthesis pipeline is constructed from scene-level 3D Gaussian Splatting (3DGS) assets and high-frame-rate videos, providing realistic training data covering both camera-induced and object-motion blur. Second, a video diffusion prior is leveraged for restoration; to better accommodate frame-dependent blur variations, temporal compression in the VAE is disabled and a frame-wise encoding scheme is adopted. For practical deployment on long videos, multi-step diffusion sampling is distilled into an efficient one-step generator, and a training-free Temporal Window Mask stabilizes inference beyond the training horizon with constant memory usage. Extensive experiments on diverse real-world benchmarks demonstrate...
  </details>  
- **[Nova3D: Code-Native Generation of Programmable 3D Assets](https://arxiv.org/abs/2607.22738v1)**  
  Authors: Nimra Noor, Muhammad Bilal, Abdullah Hussain, Hassan Baig  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2607.22738v1.pdf)  
  <details><summary>Abstract</summary>

  Current 3D generative models mostly produce a final surface: a visually strong but largely opaque mesh. Interactive 3D worlds need more than a surface. They need named parts, an assembly hierarchy, measurable constraints, local edit handles, and joints for articulation. We present Nova3D, a system that generates 3D assets as executable Blender source code; the compiled mesh, a binary glTF (GLB), is treated as the artifact, not the asset. Because the output is a program, semantic handles exist at generation time rather than being recovered afterward by segmentation or rigging. We evaluate on Nova3D-Bench, a frozen, spec-grounded benchmark of 54 items across six domains and three difficulty levels with text and image inputs, against eleven baselines in four families (mesh-native, part-structured, code-native, and CAD) plus a same-LLM ablation. Nova3D produces an executable program and a valid artifact for 54/54 items. Every asset exposes named parts organized in a parent-child assembly...
  </details>  
  Keywords: rigging  
- **[Look Before You Edit: Attention-Guided Camera Placement and Multi-View Alignment for 3D Gaussian Splatting Editing](https://arxiv.org/abs/2607.19777v1)**  
  Authors: Jaeyeon Park, Taeho Kang, Youngki Lee  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2607.19777v1.pdf)  
  <details><summary>Abstract</summary>

  Text-driven 3D scene editing with 3D Gaussian Splatting (3DGS) typically applies a 2D diffusion editor to views rendered from fixed training cameras, limiting both the spatial coverage of edits and the user's freedom to target specific objects in complex scenes. We present LB-Edit, a framework that addresses two coupled problems: where to place editing cameras for localized edits, and how to make per-view edits agree with one another so that the 3D scene remains consistent after fine-tuning. First, Attention-Guided Editing Camera Placement (ACP) probes the diffusion model's self- and cross-attention at multiple candidate camera distances to find where attention is well-contained in the region of interest, then places a compact, geometrically diverse editing camera set at that attention-optimal distance. Second, Multi-view Attention Alignment (MAA) steers the editor toward the same edit across views along two axes: it aligns appearance by sharing self-attention features via token-level correspondence, and aligns spatial location...
  </details>  
- **[Fast Wave-optics Rendering of Multiplane Images for 3D Holographic Displays](https://arxiv.org/abs/2607.19731v1)**  
  Authors: Brian Chao, Dario Seyb, Nathan Matsuda, Oliver Cossairt, Yang Zhou, Douglas Lanman, Gordon Wetzstein, Grace Kuo, Changwon Jang  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2607.19731v1.pdf)  
  <details><summary>Abstract</summary>

  Recent advances in neural rendering have unlocked unprecedented capabilities in 3D reconstruction and novel view synthesis, giving rise to applications such as virtual fly-throughs of a 3D scene reconstructed from a set of sparse, casually captured images. However, these renderings are viewed on a computer screen or conventional VR headsets as 2D images, greatly limiting the perceptual realism and immersiveness of such experiences. The rapid development in novel 3D scene representations calls for dedicated rendering algorithms that convert these readily-available 3D contents into formats that are compatible with emerging 3D display technologies, such as holographic displays. In this paper, we propose a wave-optics rendering pipeline that works with multiplane images (MPIs) for efficient and high-quality hologram synthesis. Our MPI-based computer-generated holography algorithm greatly outperforms state-of-the-art primitive-based CGH algorithms in terms of runtime, achieving speedups up to 250,000x while achieving comparable image quality, and significantly outperforms conventional layer-based CGH algorithms in...
  </details>  
  Keywords: novel view synthesis  
- **[Boltzmann-Expected Molecular Design with Decoupled Annealing Flows](https://arxiv.org/abs/2607.19519v1)**  
  Authors: Selma Moqvist, Richard Beckmann, Ross Irwin, Rocío Mercado, Simon Olsson  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2607.19519v1.pdf)  
  <details><summary>Abstract</summary>

  Most 3D properties relevant to molecular design, including free energies and shape descriptors, are $\textit{expectations}$ over the Boltzmann distribution over 3D configurations of a molecular graph. However, existing property-guided generative models tie each property to a single structure, ignoring the underlying ensemble. We recast 3D molecular design as $\textbf{Boltzmann-expected design}$ and realise it with $\textbf{DECAF}$ (Decoupled Annealing Flows), which factorise the joint distribution over graphs and coordinates into two conditional flow models: a graph-conditioned flow $p(x\mid\mathcal{G})$, acting as a $\textit{Boltzmann emulator}$, and a coordinate-conditioned flow $p(\mathcal{G}\mid x)$, proposing new graphs from 3D information. By alternating the two flows, DECAF optimises molecular graphs with a simulated-annealing acceptance rule whose scoring function is evaluated on ensembles drawn from $p(x\mid\mathcal{G})$, making ensemble statistics, not single-conformer properties, the design target. The resulting loop requires no retraining to change objectives. On GEOM-Drugs, we show that ensemble-aware optimisation produces graphs whose mean radius of gyration and...
  </details>  
- **[Latent Riemannian Flow Matching for Geometry-Grounded 3D Foundation Models](https://arxiv.org/abs/2607.19120v1)**  
  Authors: Lisa Weijler, Irene Ballester, Guofeng Mei, Tolga Birdal, Pedro Hermosilla  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2607.19120v1.pdf) | [![Project](https://img.shields.io/badge/-Project-blue)](https://lisaweijler.github.io/geometry-grounded-rfm)  
  <details><summary>Abstract</summary>

  Geometric foundation models, such as the Visual Geometry Grounded Transformer (VGGT), provide strong 3D priors from unposed images. However, such models operate purely in a feed-forward, deterministic regime, \ie~they cannot generate plausible geometry beyond what the input views directly support. Generative models for 3D scenes, on the other hand, must rely on strong geometric priors to produce coherent outputs from sparse inputs. We bridge these two paradigms by performing flow matching directly in VGGT's latent space, leveraging its learned 3D priors without committing to any explicit downstream representation such as Gaussians, meshes, or video-VAE latents. This requires respecting the latent geometry: VGGT tokens occupy a product of high-dimensional hyperspheres on which standard Euclidean flow matching fails. We address this with a Riemannian Flow Matching framework defined on a product manifold of four hyperspheres, aligned with VGGT's multi-scale encoder, which keeps generated tokens on the valid data manifold required by the...
  </details>  
- **[Text2Villa: Hierarchical Generation of 3D Indoor Environments with Physics-Aware Analysis-by-Synthesis](https://arxiv.org/abs/2607.17145v2)**  
  Authors: Xiang Tang, Ruotong Li, Xiaopeng Fan  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2607.17145v2.pdf)  
  <details><summary>Abstract</summary>

  Generating 3D indoor scenes from natural language holds tremendous potential, yet existing methods predominantly fail to generate multi-room structures with vertical connectivity and arbitrary polygonal boundaries. Furthermore, they lack a deep grounding in continuous 3D physical laws, leading to severe geometric penetrations and floating artifacts. In this work, we propose Text2Villa, a novel hierarchical generative framework. At the macro level, we construct a multi-story dataset to fine-tune an autoregressive layout generator, ensuring the direct parsing of text into 3D building foundations featuring polygonal boundaries and multi-story connectivity. To enforce physical laws during micro-level asset arrangement, we introduce the Affordance-driven Physical-Semantic Scene Graph (A-PSSG) to explicitly abstract physical affordances (such as support surfaces and containment cavities) into node attributes, establishing strict geometric and semantic edge constraints. Guided by the A-PSSG, we formulate scene instantiation as a constrained closed-loop optimization problem following the analysis-by-synthesis paradigm. By integrating an underlying geometric collision detection...
  </details>  
- **[Autoregressive B-Rep Shape Generation with Parametric Surfaces](https://arxiv.org/abs/2607.17093v1)**  
  Authors: Dafei Qin, Rui Xu, Zeyu Shen, Kaichun Qiao, Hongyang Lin, Qixuan Zhang, Huaijin Pi, Lan Xu, Jingyi Yu, Wenping Wang, Taku Komura  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2607.17093v1.pdf)  
  <details><summary>Abstract</summary>

  Generative CAD modeling has broad design and application potential. Despite significant advances in Boundary Representation (B-Rep) generation, the dominant representation in CAD, existing methods largely depend on uniformly sampled point- or grid-based geometry representations, sacrificing native surface types and parameters and thereby limiting geometric fidelity and downstream usability. We present ParaCAD, an autoregressive framework for point-cloud-conditioned B-Rep generation that directly operates on native parametric surfaces. ParaCAD introduces a surface-centric tokenization that explicitly encodes each face by its exact surface type and continuous parameters, preserving the intrinsic semantics of CAD geometry. Our model first generates parametric surfaces with constrained UV domains, and then constructs a valid B-Rep by globally intersecting these surfaces to recover edges and vertices. ParaCAD places point-cloud-conditioned generation at the core of B-Rep synthesis, making it practical for user-guided reconstruction and seamless integration into existing 3D generation pipelines. Extensive experiments demonstrate that ParaCAD produces accurate B-Reps with faithful...
  </details>  
- **[Splat-based 3D Scene Reconstruction with Extreme Motion-blur](https://arxiv.org/abs/2607.16926v1)**  
  Authors: Hyeonjoong Jang, Dongyoung Choi, Donggun Kim, Woohyun Kang, Min H. Kim  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2607.16926v1.pdf) | [![GitHub](https://img.shields.io/github/stars/KAIST-VCLAB/gs-extreme-motion-blur?style=social)](https://github.com/KAIST-VCLAB/gs-extreme-motion-blur)  
  <details><summary>Abstract</summary>

  We propose a splat-based 3D scene reconstruction method from RGB-D input that effectively handles extreme motion blur, a frequent challenge in low-light environments. Under dim illumination, RGB frames often suffer from severe motion blur due to extended exposure times, causing traditional camera pose estimation methods, such as COLMAP, to fail. This results in inaccurate camera pose and blurry color input, compromising the quality of 3D reconstructions. Although recent 3D reconstruction techniques like Neural Radiance Fields and Gaussian Splatting have demonstrated impressive results, they rely on accurate camera trajectory estimation, which becomes challenging under fast motion or poor lighting conditions. Furthermore, rapid camera movement and the limited field of view of depth sensors reduce point cloud overlap, limiting the effectiveness of pose estimation with the ICP algorithm. To address these issues, we introduce a method that combines camera pose estimation and image deblurring using a Gaussian Splatting framework, leveraging both 3D...
  </details>  
  Keywords: 3d scene reconstruction  
- **[Scene-SAM3D: Multi-View Scene Asset Generation Without Fine-Tuning](https://arxiv.org/abs/2607.16805v1)**  
  Authors: Yuqi Zhang, Yadan Luo, Xiangyu Sun, Fengyi Zhang, Zi Huang, Xin Tan  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2607.16805v1.pdf) | [![GitHub](https://img.shields.io/github/stars/xibi777/Scene-SAM3D?style=social)](https://github.com/xibi777/Scene-SAM3D)  
  <details><summary>Abstract</summary>

  High-quality 3D scene assets are critical for embodied applications such as robotic manipulation, navigation, and simulation. Despite their strong object priors, recent single-image 3D generation models such as SAM3D remain insufficient for real-world scenes, where severe occlusions, redundant observations, and cross-view inconsistencies make reliable scene generation challenging. We introduce Scene-SAM3D, a training-free framework that extends SAM3D from single-view object generation to calibrated multi-view scene asset generation. Scene-SAM3D selects a compact set of complementary views, reducing observation redundancy while providing additional evidence for regions occluded in individual views. Based on the selected views, it performs step-efficient latent velocity fusion to integrate multi-view evidence and suppress cross-view conflicts in canonical space. Finally, a lightweight rigid-object Gaussian optimization refines the scene layout within 200 iterations while preserving the generated object geometry. Experiments on Replica and ScanNet++ demonstrate consistent improvements at both instance and scene levels, with our method reducing scene-level CD by 43.8%...
  </details>  
- **[CNS-Edit++: Category-Agnostic 3D Editing with Coupled Neural Shape Representation](https://arxiv.org/abs/2607.16577v2)**  
  Authors: Jingyu Hu, Weilong Yan, Zhengzhe Liu, Haipeng Li, Ka-Hei Hui, Hao Zhang, Chi-Wing Fu  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2607.16577v2.pdf)  
  <details><summary>Abstract</summary>

  This paper presents a latent-space 3D shape editing framework built upon a coupled neural shape (CNS) representation and a neural feature volume optimization. This work extends CNS-Edit, built on Coupled Neural Shape optimization, to CNS-Edit++, by generalizing the category-specific coupled representation to category-agnostic 3D shape editing with foundation models. The Coupled Neural Shape (CNS) representation couples a global latent code that captures high-level shape semantics with a 3D neural feature volume that provides spatial context for local shape manipulation. Then we formulate a coupled neural shape optimization procedure that co-optimizes these two components subject to a given editing operation. Our framework can be instantiated on both the category-specific 3D inversion model and category-agnostic 3D foundation models. We provide various shape editing operators, including copy, resize, delete, mix, point-wise drag, and region-wise drag, each of which is formulated as an objective to guide the CNS optimization. To preserve regions outside the...
  </details>  
- **[OmniStyle-INR: Universal and Multimodal Style Transfer for INRs](https://arxiv.org/abs/2607.16362v1)**  
  Authors: Rafał Kajca, Michał Miziołek, Kornel Howil, Rafał Tobiasz, Przemysław Spurek  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2607.16362v1.pdf)  
  <details><summary>Abstract</summary>

  Style transfer remains a fundamental and highly important task across various data modalities, enabling creative manipulation conditioned by both reference images and textual descriptions. Recently, methods utilizing Gaussian Splatting have emerged as a unified representation for 2D images, video, 3D scenes, and 4D dynamics. However, representing videos and 2D images with Gaussian Splatting is structurally sub-optimal for dense continuous domains. The number of required Gaussians often approaches the total number of pixels, raising questions about the actual utility of such a representation for these specific modalities. In contrast, Implicit Neural Representations have established themselves as a much more popular and natural choice across all these data domains. Implicit Neural Representations naturally provide significant advantages, including data compression, inherent capabilities for super resolution, and seamless integration with deep generative models. To this end, we introduce OmniStyle-INR, a novel framework that leverages network-based continuous representations as a truly universal domain. Our approach...
  </details>  
- **[Clarify Before Executing: A Self-Evolving Agent for Resolving Intent Asymmetry in 3D Tool Orchestration](https://arxiv.org/abs/2607.16352v1)**  
  Authors: Xiaoye Zhu, Weixin Li, Junan Huo, Bozhong Wang, Jia Zeng, Yi Yang, Cen Chen, Qi Liu  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2607.16352v1.pdf) | [![GitHub](https://img.shields.io/github/stars/xyzhu1225/CLARE?style=social)](https://github.com/xyzhu1225/CLARE)  
  <details><summary>Abstract</summary>

  A fundamental intent asymmetry plagues modern 3D asset creation: while state-of-the-art 3D toolchains demand precise, executable parameters, ordinary users typically provide vague, underspecified instructions. Current 3D agents treat this ambiguity as noise, defaulting to blind execution under a single-turn assumption. To address this limitation, we introduce CLARE, a clarification-aware and evolutionary 3D agent that treats intent asymmetry not as an execution error, but as an opportunity for strategic dialogue. By decoupling the generation pipeline into four specialized cognitive roles, CLARE intercepts and resolves underspecified instructions before invoking computationally expensive 3D tools to seamlessly execute tasks across five diverse domains: text-to-3D generation, single-view reconstruction, multi-view reconstruction, point cloud editing, and post-processing. Crucially, rather than relying on rigid manual rules, CLARE self-evolves its clarification policy via simulated multi-turn interactions. By optimizing a Multi-turn Reward, the agent internalizes the delicate balance between interaction efficiency and task completion. To rigorously test this, we construct...
  </details>  
  Keywords: text-to-3d, single-view reconstruction  
- **[CSS-BA: Gate-Guided Column Space Search for Bundle Adjustment](https://arxiv.org/abs/2607.15652v1)**  
  Authors: Ayano Kaneda, Takafumi Taketomi, Shugo Yamaguchi, Shigeo Morishima  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2607.15652v1.pdf)  
  <details><summary>Abstract</summary>

  Bundle adjustment (BA) remains a critical refinement module for image-based 3D reconstruction and continues to improve geometric accuracy even in learning-based pipelines. However, in low-parallax and near-rotational regimes, classical Schur-based Levenberg--Marquardt (LM) often becomes ill-conditioned and yields unreliable pose and calibration estimates. We propose Gate-Guided CSS-BA, a solver-side modification of Schur-LM that preserves the classical BA objective and trust-region framework while constraining each update to a geometrically informed low-dimensional subspace. By integrating Column Space Search (CSS) with geometry-aware gating, the method stabilizes the Schur-LM update without altering the estimation problem. In contrast to keyframe or state-selection approaches, all camera and point parameters remain in the optimization problem; only the update direction is restricted. The method serves as a drop-in replacement for existing BA pipelines. Experiments on both generic and challenging weak-geometry scenarios show more stable optimization, improved relative pose accuracy, and competitive calibration behavior while maintaining reprojection quality.
  </details>  
- **[TanGO: Training-Free 3D Editing via Tangent-Space Guidance and Optimization](https://arxiv.org/abs/2607.14927v1)**  
  Authors: Siwoo Lim, Sunjae Yoon, Gwanhyeong Koo, Hyeonseo Yun, Chang D. Yoo  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2607.14927v1.pdf) | [![GitHub](https://img.shields.io/github/stars/siw00-lim/TanGO?style=social)](https://github.com/siw00-lim/TanGO)  
  <details><summary>Abstract</summary>

  While recent flow-matching 3D generative models (e.g., VecSet) adopt structured representations, their tokens share global context, causing conventional training-free editing to suffer from semantic artifacts such as collapsed preserved regions or incomplete transformations. To address this, we propose TanGO, a training-free framework that enables adaptive per-token steering in the tangent space of generative dynamics. To realize this selective control, we formulate a one-step optimal control rule and determine the strength of each token's control signal using a von Mises-Fisher inspired directional discrepancy derived from the source and target velocity fields. Experiments show that TanGO substantially reduces structural artifacts and achieves state-of-the-art performance, outperforming existing 3D editing baselines. The code is publicly available at https://github.com/siw00-lim/TanGO.
  </details>  
- **[$K$-NeAS: Scalable Multi-Material CT Reconstruction Using Neural SDFs](https://arxiv.org/abs/2607.14415v1)**  
  Authors: Daksh K. Shah, Emmanouil Nikolakakis, Razvan V. Marinescu  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2607.14415v1.pdf)  
  <details><summary>Abstract</summary>

  Computed Tomography (CT) carries significant ionizing radiation risks, driving the need for sparse-view reconstruction. Implicit scene representations (ISRs) address this by recovering continuous volumetric attenuation fields directly from sparse projections, and recent geometry-aware extensions jointly model surface geometry alongside attenuation to improve fidelity and enable clean tissue segmentation without manual thresholding. However, these methods remain limited by manually tuned attenuation bounds and rigid two-material constraints. This paper proposes $K$-NeAS, a unified and scalable architecture for automated, multi-material surface reconstruction. We replace independent material networks with a shared latent backbone and introduce a fully differentiable $K$-material sequential soft selector to model an arbitrary number of overlapping tissues. To eliminate manual tuning, we automate attenuation bounding using a Gaussian Mixture Model (GMM) and implement a scheduled auxiliary floater loss to mitigate geometric hallucinations common under extreme sparsity. Evaluated across four clinical Cone-Beam CT (CBCT) datasets, $K$-NeAS successfully scales to arbitrary material counts,...
  </details>  
- **[T3HG-Editor: Text-driven 3D Human Garment Editing with Body Priors Embedded in SMPL-X](https://arxiv.org/abs/2607.13654v1)**  
  Authors: Shaoru Sun, Xingtao Wang, Zihan Ma, Wenrui Li, Jiantao Zhou, Debin Zhao, Xiaopeng Fan  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2607.13654v1.pdf)  
  <details><summary>Abstract</summary>

  While 3D Gaussian Editing (3DGE) has seen substantial progress, text-driven 3D human garment editing remains largely underexplored. Existing 3DGE works typically follow a paradigm that applies 2D editing techniques to multi-view rendered images and updates 3D Gaussians based on the modified images. Extending such methods to 3D human garment editing suffers from low-fidelity outcomes, caused by introduced distortions and garment inconsistencies. A promising breakthrough opportunity arises from the SMPL eXpressive (SMPL-X) model that embodies rich prior information for virtual humans. Motivated by this insight, we propose a text-driven 3D human garment editor termed T3HG-Editor, which delivers high-fidelity and garment consistent results by leveraging geometry and joint priors embedded in SMPL-X. Specifically, T3HG-Editor contains three stages, namely obtainment of editable Gaussians, garment consistent editing, and Gaussian updating with overflow pruning. The obtainment of editable Gaussians begins with seeding Gaussians along SMPL-X normals to generate sufficient near surface Gaussians, followed by a...
  </details>  
- **[DiffGI: Differentiable Geometry Images for High-Fidelity Thin-Shell 3D Generation](https://arxiv.org/abs/2607.13365v1)**  
  Authors: Eungjune Shim, Hansol Lee, Eunjung Ju  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2607.13365v1.pdf)  
  <details><summary>Abstract</summary>

  Existing 3D generative models predominantly rely on implicit volumetric representations, which enforce watertight topology and struggle to represent thin-shell and non-manifold geometries such as garments. Geometry image-based approaches offer a surface-centric alternative, but existing methods rely on discrete binary occupancy maps whose resolution-dependent boundary encoding causes staircase artifacts and information loss upon downsampling, while surface reconstruction remains a non-differentiable post-processing step disconnected from the learning pipeline. To address this, we propose Differentiable Geometry Image (DiffGI), an end-to-end 3D-to-2D mapping framework that seamlessly integrates surface representation and geometric optimization. DiffGI replaces binary maps with a continuous 2D Truncated Signed Distance Function (TSDF), which encodes boundary position at subpixel precision within a fixed grid resolution, eliminating resolution-dependent staircase artifacts even under aggressive downsampling. Building on this continuous field, we introduce a differentiable Marching Squares algorithm based on analytical linear interpolation, allowing gradients from 3D surface losses to propagate back to the 2D...
  </details>  
- **[Hallo4D: Multi-Modal Hallucination Mitigation for Consistent Spatio-Temporal Generation](https://arxiv.org/abs/2607.12752v2)**  
  Authors: Hongbo Wang, Huaibo Huang, Jie Cao, Jin Liu, Haoyang Tong, Ran He  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2607.12752v2.pdf)  
  <details><summary>Abstract</summary>

  While recent advances in 3D generation have enabled impressive visual synthesis, existing methods often rely on 2D diffusion supervision without explicit mechanisms for geometric consistency, leading to spatial hallucinations such as duplicated structures and misaligned geometry. These issues become more severe in 4D generation, where maintaining consistency across viewpoints and temporal evolution introduces additional challenges, including jitter, identity flicker, and structural drift. We present \textbf{Hallo4D}, a unified and model-agnostic framework for mitigating spatiotemporal hallucinations in 3D and 4D content generation. Hallo4D introduces a generation-detection-correction paradigm that leverages large multimodal language models (LMMs) to identify and summarize spatial and temporal inconsistencies from multi-view and multi-frame renderings. These insights guide a consensus-driven image-space consistency optimization, where an LMM-based selector evaluates candidate corrections through multi-model voting, without requiring retraining or architectural modifications. To further improve temporal consistency and optimization efficiency, Hallo4D incorporates motion-aware keyframe sampling, LMM-guided initialization, and appearance alignment. We additionally introduce...
  </details>  
- **[RealSkin: Spatio-Spectral Partial Neural Adjoint Maps for Image-to-3D Attribute Transfer](https://arxiv.org/abs/2607.12495v1)**  
  Authors: Jing Li, Yawei Luo, Xiangze Meng, Ying Li, Tieru Wu, Rui Ma  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2607.12495v1.pdf)  
  <details><summary>Abstract</summary>

  Creating photorealistic 3D assets requires bridging the appearance gap between real-world observations and synthetic models. A promising approach is to transfer visual attributes from real images onto synthetic 3D surfaces. Traditional methods struggle with resolution mismatch and the inherent discreteness of point correspondences. In contrast, resolution-robust functional maps enable smooth attribute propagation but rely on near-isometry assumptions and topological consistency. To address these limitations, we propose RealSkin, a self-supervised framework that performs correspondence optimization in a learned spectral domain, guided by spatial correspondences. We first introduce a spatial-guided registration algorithm to establish coarse correspondences under severe topological discrepancies. To relax strict isometric assumptions and handle partial correspondences, we further design a spectral-aware neural adjoint network that incorporates partial correspondences into a neural function space and models non-isometric residuals for correspondence refinement. Experimental results demonstrate that our method achieves state-of-the-art performance on challenging real-to-synthetic scenarios. The code will be publicly released.
  </details>  
  Keywords: image-to-3d  
- **[3D-DefectBench: A Controlled Factorial Study of Vision-Language Model Evaluation Pipelines for Fine-Grained 3D Generation Defects](https://arxiv.org/abs/2607.10826v1)**  
  Authors: Zhenyu Zhao, Nanshan Jia, Jihyeon Je, Yifu Tang, Alvin Chan, Michael Spedden, Michael V. Palleschi, Sui Huang, Jingshen Wang, Zeyu Zheng  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2607.10826v1.pdf)  
  <details><summary>Abstract</summary>

  Automated evaluation is essential for scaling generative 3D systems, where exhaustive human review is costly and slow. However, the reliability of an automated judge depends on the entire evaluation pipeline, not only the underlying vision-language model (VLM), but also how assets are rendered, what visual evidence is provided, how the task is specified, and how human reference labels are constructed. We introduce 3D-DefectBench, a benchmark and framework for systematic analysis of VLM-based 3D defect detection pipelines. It complements holistic ratings and pairwise preferences with nine fine-grained binary defects spanning geometry, texture, and prompt adherence, providing actionable diagnostics for generator development and judge evaluation. Using a balanced factorial design, we vary four pipeline factors, VLM, camera protocol, visual input, and prompt schema, across 84 inference designs and approximately 3.2 million scored defect decisions, followed by staged validation on a broader set of frontier models. Model choice is the largest determinant of...
  </details>  
- **[PoseAlign: Sculpting Pose-Consistent Meshes via Text-Guided Deformation](https://arxiv.org/abs/2607.10560v2)**  
  Authors: Shijin Wang, Zichong Chen, Yang Zhou, Hui Huang  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2607.10560v2.pdf) | [![Project](https://img.shields.io/badge/-Project-blue)](https://cousingrade6.github.io/PoseAlign)  
  <details><summary>Abstract</summary>

  Mesh deformation, the process of altering the vertex positions of a 3D mesh while preserving its topological structure, is a cornerstone of computer graphics. Despite the recent emergence of numerous text-guided 3D mesh deformation methods, deforming an initial mesh into one that both adheres to text prompts and preserves its pose remains challenging. This paper proposes PoseAlign, which decomposes text-guided mesh deformation into two stages: global pose scaling and local detail sculpting. Specifically, in the first stage, we introduce the Laplacian as a differentiable mesh representation to enable more efficient yet smoother global deformation. Then, we propose a novel pose-aligned SDS loss by adapting score distillation sampling (SDS) with an attention-sharing mechanism, which sculptures fine-grained geometric details for the deformed mesh while preserving its original pose. PoseAlign significantly enhances the controllability of the overall deformation process, achieving a favorable balance between pose preservation and text alignment. Experiments demonstrate the competitive...
  </details>  
- **[LTM: Large-scale Terrain Model for Wildfire-prone Landscapes](https://arxiv.org/abs/2607.08711v1)**  
  Authors: Xiao Fu, Yue Hu, Meida Chen, Peter Anthony Beerel, Barath Raghavan  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2607.08711v1.pdf)  
  <details><summary>Abstract</summary>

  Accurate 3D terrain maps are essential for emergency response when assessing wildfire hazards. However, wildfire-prone regions often span vast areas where conventional reconstruction methods underperform. Airborne LiDAR systems provide high-resolution terrain data, but they are expensive and infrequently updated. Image-based methods offer a lower-cost alternative, but struggle due to sparse visual features and limited image overlap. We propose a multi-modal reconstruction framework leveraging outdated Digital Elevation Models (DEMs) as geometric priors for image-based 3D reconstruction. Our key innovation is physics-based pixel-pixel alignment between images and DEM data, dramatically reducing computational complexity by eliminating expensive feature matching procedures. To validate our approach, we developed a large-terrain simulator based on a real wildfire-prone area, generating realistic images enabling a comprehensive evaluation. Given posed images and legacy DEMs, our method produces high-fidelity depth maps while maintaining real-time performance. We find significant improvements in reconstruction accuracy and computational efficiency over existing techniques, offering a...
  </details>  
- **[DreamCharacter-1: From 3D Generative Foundation Models to Product-Ready Character Generation](https://arxiv.org/abs/2607.07817v1)**  
  Authors: Weizhe Liu, Yunjie Wu, Xiangqian Shu, Guangwei Wang, Xiangyu Xu, Peng Li, Yujie Li, Hengkai Guo  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2607.07817v1.pdf)  
  <details><summary>Abstract</summary>

  We present DreamCharacter-1, a lightweight post-adaptation framework that calibrates pretrained 3D foundation models toward high-fidelity, production-ready 3D character generation. Building upon a 3D foundation backbone, our pipeline incorporates three task-oriented components: (1) geometry post-training, which enhances fine-grained surface details through geometric preference optimization; (2) texture post-training, which synthesizes high-resolution textures and refines the appearance of occluded regions; and (3) inference acceleration, which enables scalable deployment. Extensive quantitative and qualitative experiments demonstrate that DreamCharacter-1 produces visually compelling and structurally robust 3D character assets, consistently surpassing state-of-the-art character generation methods.
  </details>  
- **[EmbodiedGen V2: An Agentic, Simulation-Ready 3D World Engine for Embodied AI](https://arxiv.org/abs/2607.07459v3)**  
  Authors: Xinjie Wang, Liu Liu, Taojun Ding, Andrew Choi, Chaodong Huang, Mengao Zhao, Ziang Li, Jackson Jiang, Chunlei Yu, Shengxiang Liu, Wei Xu, Zhizhong Su  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2607.07459v3.pdf)  
  <details><summary>Abstract</summary>

  We present EmbodiedGen V2, a generative 3D world engine for building executable policy-ready environments for embodied intelligence. Sim-ready 3D asset generation has advanced rapidly, yet assembling such assets into policy-ready task environments remains largely manual, limiting scalable closed-loop learning. EmbodiedGen V2 addresses this gap through a unified sim-ready representation that connects cross-simulator assets, interaction affordances, task-driven worlds, large-scale multi-room scenes, and stateful Vibe Coding into a generative, editable, and reusable simulation pipeline. The generated environments support manipulation, navigation, mobile manipulation, cross-simulator deployment, and embodied policy training. In evaluation, the asset pipeline achieves 96.5% human acceptance and 98.6% collision success, and 83.3% of task-driven worlds are directly usable for downstream simulation without manual modification. Online reinforcement learning with generated environments further improves simulation success from 9.7% to 79.8%, and transfers to real robots with task success increasing from 21.7% to 75.0%. These results establish EmbodiedGen V2 as scalable simulation infrastructure for...
  </details>  
  Keywords: 3d asset generation  
- **[EditVerse3D: High-Quality 3D Object Editing with Region-Aware Learning](https://arxiv.org/abs/2607.07187v1)**  
  Authors: Youtan Yin, Yanning Zhou, Jiacheng Wei, Xiaofeng Yang, Jun Zhang, Jiayang Bai, Jingwen Ye, Weidong Zhang, Guosheng Lin  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2607.07187v1.pdf) | [![Project](https://img.shields.io/badge/-Project-blue)](https://editverse3d.github.io)  
  <details><summary>Abstract</summary>

  Local editing of 3D objects remains a long-standing challenge. When interacting with 3D content, humans naturally tend to specify a coarse region of interest for modification rather than defining precise editing boundaries. However, previous methods rely on fully edited 2D images, precise 3D masks, or redundant pipelines, which present a gap. To bridge this gap, we propose EditVerse3D, a novel 3D editing framework that enables high-quality object editing under such coarse guidance. Our approach takes as input a 3D object to be edited, a coarse 3D bounding box indicating the target region, and a reference 2D image describing the desired modification. It produces a coherent, high-fidelity edited 3D object. To facilitate this editing, we introduce a novel region-aware adaptive loss that emphasizes hard-to-learn regions and balances the objective between target and preserved areas. Complementing our loss function, we enhance model robustness and generalization through targeted data augmentations, such as training...
  </details>  
- **[ELSA3D: Elastic Semantic Anchoring for Unified 3D Understanding and Generation](https://arxiv.org/abs/2607.06565v1)**  
  Authors: Tianjiao Yu, Xinzhuo Li, Yifan Shen, Onkar Susladkar, Yuanzhe Liu, Xiaona Zhou, Ismini Lourentzou  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2607.06565v1.pdf)  
  <details><summary>Abstract</summary>

  Unified 3D foundation models aspire to generate 3D assets and reason about them in language within a single backbone, but their text-3D interaction remains largely implicit. Existing methods concatenate text and 3D tokens into a flat sequence and rely on self-attention, collapsing coarse structural cues and fine geometric details into one undifferentiated representation. We introduce ELSA3D, a unified 3D model that addresses this with elastic semantic anchoring, structuring language and geometric reasoning jointly along matched abstraction scales. ELSA3D represents geometry with a scale-aware octree tokenizer and introduces Anchor Tokens, sparse cross-modal units that select semantic cues, route them to the most relevant 3D scale, retrieve scale-specific geometric evidence, and write the fused signal back into the unified representation, keeping interaction sparse yet precise. A lightweight per-block router makes both computation and reasoning elastic, choosing which text tokens instantiate anchors at which geometric scale so that cross-modal capacity concentrates where alignment...
  </details>  
  Keywords: text-to-3d, image-to-3d  
- **[VaseMuseum: Digital Intelligent Museum for Ancient Greek Pottery](https://arxiv.org/abs/2607.06374v1)**  
  Authors: Jiazi Wang, Nonghai Zhang, Qiushi Xie, Zeyu Zhang, Yufeng Chen, Yang Zhao, Ling Shao, Hao Tang  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2607.06374v1.pdf)  
  <details><summary>Abstract</summary>

  Vision-language models (VLMs) have made interactive digital museums increasingly feasible by connecting 3D digitization with natural-language artifact exploration. However, in cultural heritage domains such as ancient Greek pottery, reliable VLM assistance is limited by two challenges. First, open-ended interpretation requires grounding fine-grained 2D/3D visual evidence in specialized curatorial knowledge, yet the retrieval process may introduce weak sources and unverifiable references. Second, when the available evidence is incomplete, noisy, or ambiguous, VLMs often produce confident but unsupported answers instead of calibrated uncertainty. To address these challenges, we propose VaseMuseum, a lightweight and modular multimodal agent framework for intelligent digital museums of ancient Greek pottery. VaseMuseum combines an interactive virtual museum with VaseAgent, which supports both 2D images and 3D artifacts through multimodal perception, 3D-aware reasoning, external knowledge retrieval, and inference-time reliability control. Specifically, VaseAgent retrieves evidence from authoritative web and museum knowledge sources, and source-level control selects diverse and verifiable evidence...
  </details>  
- **[Recovering Cloud Microstructures with Cascaded Diffusion Inversion](https://arxiv.org/abs/2607.05637v1)**  
  Authors: Hanan Gani, Guy Pulik, Daniel Rosenfeld, Duncan Watson-Parris, Salman Khan  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2607.05637v1.pdf) | [![GitHub](https://img.shields.io/github/stars/hananshafi/superresolution-cloud-microphysics?style=social)](https://github.com/hananshafi/superresolution-cloud-microphysics)  
  <details><summary>Abstract</summary>

  High-resolution satellite imagery is critical for observing fine-scale cloud structures that inform weather modification strategies like cloud seeding for rain-enhancement. However, the spatial resolution of current geostationary and polar-orbiting satellites is often insufficient for capturing small cloud features. Current super-resolution methodologies are suited for natural images and, therefore, struggle to generalize to satellite-captured spectral images of cloud cover. To address this, we propose a two-stage diffusion-based super-resolution framework to enhance the resolution of multi-spectral cloud microstructures by a factor of $4\times$. Specifically, we use inverse diffusion to recover the high resolution properties from low resolution. Stage 1 utilizes real-world paired data to learn robust degradation handling and inter-sensor alignment, while Stage 2 employs a self-supervised internal downgrading of high resolution data to refine structural learning and texture synthesis. Our approach outperforms the state-of-the-art transformer and diffusion-based baselines in both reconstruction accuracy and visual quality. We demonstrate that the two-stage method...
  </details>  
  Keywords: texture synthesis  
- **[SynCity 3000: Bootstrapping Scene-Scale 3D Diffusion](https://arxiv.org/abs/2607.05392v1)**  
  Authors: Paul Engstler, Iro Laina, Christian Rupprecht, Andrea Vedaldi  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2607.05392v1.pdf)  
  <details><summary>Abstract</summary>

  We present SynCity 3000, a framework for generating 3D scenes that are globally coherent while enabling fine-grained layout control. Building on the ability of current image-to-3D generators to produce complex 3D assets from a single image, we extend this capability to the scale of entire scenes by adapting the generator to be applicable as a convolutional operator. We achieve this by fine-tuning the model on scene-like data generated by a new synthetic data engine, which we propose to address the scarcity of 3D scene data for training. The convolutional generator is then applied to a dimetric image of the entire scene, generated from the user prompt, resulting in 3D scenes of arbitrary size and complexity. Across diverse prompts and layouts, SynCity 3000 produces large, coherent, and detailed scenes, addressing the shortcomings of prior approaches to 3D scene generation.
  </details>  
  Keywords: 3d generator, image-to-3d  
- **[MV-Forcing: Long Multi-View Video Generation via 4D-Grounded Spatio-Temporal Self-Forcing](https://arxiv.org/abs/2607.05376v1)**  
  Authors: Gal Fiebelman, Hadar Averbuch-Elor, Sagie Benaim  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2607.05376v1.pdf)  
  <details><summary>Abstract</summary>

  Recent advances in video diffusion models have enabled either long single-view generation through temporal autoregression, or short multi-view synthesis through bidirectional attention. However, generating long, multi-view consistent videos of dynamic scenes remains unsolved. In this work, we present MV-Forcing, a framework that composes temporal and view-wise autoregression within a single diffusion model by introducing a 4D geometric bridge between sequentially generated views. Our key insight is that an autoregressive 3D reconstruction model naturally interfaces between autoregressively generated views. Given a completed source view, we reconstruct its 3D structure and render a geometric prior of the next target viewpoint, which the diffusion model refines into a high-quality video. To extend generation beyond the teacher's fixed temporal window, we introduce a joint denoising regime where both view slots are initialized from noise during training, enabling temporally unbounded generation. We distill the model via Distribution Matching Distillation with Spatio-Temporal Self-Forcing, closing the train-inference...
  </details>  
  Keywords: multi-view synthesis  
- **[Graph Representation Learning of Longitudinal Medical Imaging Trajectories for Treatment Response Prediction](https://arxiv.org/abs/2607.04912v1)**  
  Authors: Johannes Kiechle, Richard Osuala, Daniel M. Lang, Stefan M. Fischer, Ivana Janíčková, Karim Lekadir, Julia A. Schnabel, Jan C. Peeken  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2607.04912v1.pdf)  
  <details><summary>Abstract</summary>

  In patients with breast cancer, pathological complete response (pCR) has been established as a clinically meaningful surrogate marker for long-term outcomes. While commonly treated with neoadjuvant chemotherapy (NACT), effective treatment decision-making remains challenging, as therapeutic response can vary substantially across patients, calling for predictive models capable of accurately estimating individualized treatment response. To address this, we propose an imaging-based 3D spatio-temporal framework for treatment response prediction that integrates a state-of-the-art graph neural network with relational modeling of temporal interactions across timepoints alongside three novel complementary self-supervised treatment trajectory representation learning objectives. Experiments across a cohort of 585 patients from the public ISPY-2 dataset demonstrate that our method substantially outperforms both vision and self-supervised learning baselines across several classification metrics. Alongside establishing a breast cancer pCR prediction benchmark, we include a principled ablation of our method and further introduce and empirically assess the impact of the available number of DCE-MRI timepoints...
  </details>  
- **[InSpace: Structure-Aware 3D Indoor Scene Generation from a Single 360° Image](https://arxiv.org/abs/2607.03990v1)**  
  Authors: Gwanhyeong Koo, Hyunsu Kim, Youngji Kim, Taejae Lee, Siwoo Lim, Sunjae Yoon, Suyong Yeon, Chang D. Yoo  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2607.03990v1.pdf) | [![Project](https://img.shields.io/badge/-Project-blue)](https://kookie12.github.io/InSpace-Project-Page)  
  <details><summary>Abstract</summary>

  Recent advances in single image-to-3D generation have enabled high-quality asset synthesis, yet extending these capabilities to indoor scene generation remains challenging. Existing methods focus on asset-level generation while neglecting the structural layout, which is essential for downstream applications and serves as the spatial anchor for grounding assets. However, a single image with a limited field of view lacks the spatial coverage to recover a coherent global layout. To this end, we use a 360° image represented in equirectangular projection (ERP) and propose InSpace, a structure-aware framework for 3D indoor scene generation. InSpace comprises three stages: (1) estimating partial scene geometry as spatial priors, (2) generating coarse scene structure with view-selective cross-attention, and (3) producing detailed layout and asset geometry with textures through a global-local hybrid attention, using flow matching. We also propose ERP-FRONT, a paired ERP-Image-to-3D indoor scene dataset based on 3D-FRONT. Experiments show that InSpace generates complete 3D indoor...
  </details>  
  Keywords: image-to-3d  
- **[BAT3R: Bootstrapping Articulated 3D Reconstruction from 2D Image Collections](https://arxiv.org/abs/2607.03891v1)**  
  Authors: Jakub Zadrozny, Oisin Mac Aodha, Hakan Bilen  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2607.03891v1.pdf)  
  <details><summary>Abstract</summary>

  3D reconstruction of articulated objects from a single image is challenging because large training datasets with paired image and 3D supervision are difficult to obtain. Recent point map-based methods achieve strong performance but rely on synthetic datasets rendered from manually created articulated 3D assets with carefully curated pose distributions. While camera viewpoints can be easily sampled, generating realistic object articulations remains costly and labor-intensive. We propose a training framework that reduces this requirement by leveraging unannotated 2D images collections with only a single rigged canonical mesh per category. Starting from a weak 3D shape predictor trained on canonical-pose renders, we iteratively estimate object articulation and camera pose by fitting the mesh to predicted point maps. The recovered articulations and viewpoints are then used to render updated synthetic training data, progressively improving the predictor. Despite using substantially weaker 3D supervision, our models achieve performance comparable with DualPM, which requires manually curated...
  </details>  
- **[PRISM3D: Probabilistic Refinement and Robust Initialization for Physically Consistent Scene Modeling under Extreme Motion Blur](https://arxiv.org/abs/2607.03855v1)**  
  Authors: Gopi Raju Matta, Reddypalli Trisha, Vemunuri Divya Madhuri, Kaushik Mitra  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2607.03855v1.pdf)  
  <details><summary>Abstract</summary>

  We address the inverse problem of blind 3D scene reconstruction from extremely motion-blurred images, a scenario where traditional Structure-from-Motion (SfM) pipelines fail. Existing approaches typically circumvent this bottleneck by relying on impractical sharp-image supervision. In this work, we introduce PRISM3D, a unified framework enabling robust reconstruction directly from severely degraded inputs. To overcome the lack of a reliable starting point, we propose a Robust Initialization strategy utilizing deep dense tracking method (VGGSfM) to recover global topology where feature matching fails. To the best of our knowledge, we are the first to effectively leverage this paradigm to bootstrap 3D Gaussian Splatting from extreme motion blur. However, while robust, this initialization yields sparse and noisy geometry that causes deterministic optimization to diverge. To resolve this, we propose a coupled solution driven by probability and physics: we adopt a probabilistic formulation for geometric densification via Markov Chain Monte Carlo (MCMC) to robustly populate...
  </details>  
  Keywords: 3d scene reconstruction  
- **[CGGS: Consistency-Augmented Geometric Gaussian Splatting for Ego-Centric 3D Scene Generation](https://arxiv.org/abs/2607.03819v3)**  
  Authors: Zhenyu Sun, Xiaohan Zhang, Qi Liu, Huan Wang  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2607.03819v3.pdf)  
  <details><summary>Abstract</summary>

  Challenges remain in ego-centric 3D scene generation due to limited view overlap and the dominant influence of individual perspectives on scene interpretation. These factors hinder the creation of viewpoint-consistent and semantically aligned visual content, as well as the construction of accurate geometric structures. In this paper, we propose CGGS, a text-to-3D framework aiming to enhance 3D-content-awareness and address geometric distortions in ego-centric scene generation. Firstly, the Ego-centric Generator is proposed by fine-tuning a Multi-View Latent Diffusion Model with consistency-augmented loss to generate consistent, high-fidelity 2D content aligned with textual descriptions. Then, Layout Decorator leverages optical flow and point-track correspondence to estimate depth, therefore producing dense point clouds as coarse layouts from the ego-centric 2D priors. Building on this initialization, Geometric Refiner is proposed to enhance 3D Gaussian reconstruction via an entropy-based Mutual Information Depth Loss (MID) combined with a hierarchical optimization scheme for improving visual quality and geometric structure. Comprehensive...
  </details>  
  Keywords: text-to-3d  
- **[CoGen3D: An Agentic Human-AI Co-Design Pipeline for 3D Asset Generation for Virtual Reality](https://arxiv.org/abs/2607.03731v1)**  
  Authors: Weiwei Jiang, Wanyu He, Zheyu Tan, Zheyuan Kuang, Difeng Yu, Shinobu Hasegawa, Sven Mayer, Zhanna Sarsenbayeva  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2607.03731v1.pdf)  
  <details><summary>Abstract</summary>

  Creating 3D assets for virtual reality requires modeling expertise, which restricts the authorship of immersive experiences. Existing generative AI tools rely on unconstrained, command-driven prompting, lacking the conversational scaffolding needed for users to articulate their intent and validate designs prior to rendering. To address this, we introduce CoGen3D, an agentic human-AI co-design pipeline that proactively guides users through conversational intent elicitation, a concept image confirmation, and image-to-3D generation that directly deploys to immersive scenes. We evaluated this system through a user study (N=120) across six affectively diverse immersive scenes, observing 60 Design group participants who co-created 3D assets for the scenes, and 60 Validation group participants who experienced the scenes with generated assets. Our findings show that co-designed assets are associated with higher scene engagement and shifted affective responses, while participants generally preferred concept images over the final 3D assets, with no increased leniency toward degradation in their own creations....
  </details>  
  Keywords: image-to-3d, 3d asset generation  
- **[Property-Constrained 3D Porous Media Reconstruction from 2D Images via Conditional Generative Adversarial Networks](https://arxiv.org/abs/2607.02693v1)**  
  Authors: Ali Sadeghkhani, Brandon Bennett, Arash Rabbani  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2607.02693v1.pdf)  
  <details><summary>Abstract</summary>

  This study presents a conditional Generative Adversarial Network (cGAN) framework for generating 3D porous media volumes with controlled porosity, trained exclusively on 2D thin section images. The key innovation lies in combining property-conditioned generation with 2D-to-3D reconstruction, eliminating the need for expensive 3D training data while maintaining control over petrophysical properties. The framework employs a hybrid architecture with a 3D generator and 2D discriminator, where multi-axis slice extraction enables learning 3D-consistent structures from 2D training data. Porosity labels are extracted using an Enhanced U-Net segmentation model. The methodology was demonstrated on two carbonate samples with different lithologies: dolomite-anhydrite and pure dolomite. Results show that the framework successfully generates realistic 3D volumes capturing lithological features such as anhydrite inclusions and fine crystalline textures. Porosity control achieved an $R^2$ of 0.93, with mean absolute errors of 0.019 and 0.010 for the heterogeneous and homogeneous samples, respectively.
  </details>  
  Keywords: 3d generator  
- **[EmoteGPT: 3D Human Facial Expressions from Natural Language Descriptions](https://arxiv.org/abs/2607.02674v1)**  
  Authors: Haoran Wang, Mohit Mendiratta, Christian Theobalt, Adam Kortylewski  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2607.02674v1.pdf)  
  <details><summary>Abstract</summary>

  Precise control of 3D facial expressions from text is crucial for virtual avatars, animation, and human-computer interaction, yet existing text-to-3D methods jointly generate identity, expression, and texture, making fine-grained expression control difficult. We instead formulate text-driven expression synthesis as a regression problem in the disentangled parameter space of a 3D Morphable Model (3DMM). This setting, however, requires paired data linking detailed language to precise expression parameters, which are missing from existing resources. To fill this gap, we introduce Txt2Emote, a benchmark of diverse 3D facial expressions with fine-grained textual annotations obtained from GPT-4o and a high-fidelity face tracker, providing both explicit descriptions detailing facial features and implicit descriptions referencing the situational context behind the expression. Leveraging this dataset, we present EmoteGPT, a text-to-3D expression framework based on a Multimodal Large Language Model (MLLM) with a dedicated <Expr> token to semantically ground expression representations, which are then decoded into 3DMM parameters....
  </details>  
  Keywords: text-to-3d, image-to-3d  
- **[Text-Driven 3D Indoor Scene Synthesis in Non-Manhattan Environments](https://arxiv.org/abs/2607.02407v1)**  
  Authors: Xianhui Meng, Zirui Song, Yuchen Zhang, Li Zhang, Yongxuan Lv, Xiuying Chen, Kun Wang, Yan Luo, Kai Chen, Hangjun Ye, Long Chen, Jun Liu, Xiaoshuai Hao  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2607.02407v1.pdf)  
  <details><summary>Abstract</summary>

  Large Language Models (LLMs) have demonstrated remarkable capabilities in 3D indoor synthesis for Manhattan environments. However, existing methods often fail to capture plausible object layout patterns in non-Manhattan settings, primarily because they struggle to model non-orthogonal spatial relationships, leading to high geometric violations and low physical fidelity. To address this challenge, we propose SPG-Layout, a novel text-driven framework designed to generate physically plausible indoor scenes within complex non-Manhattan environments. Specifically, we first utilize statistical priors of object distributions to guide the training process, enhancing environmental understanding and fidelity. Furthermore, mirroring human design workflows, we adopt a hierarchical layout strategy that prioritizes the placement of large objects, thereby substantially minimizing layout violations. By synergizing these components, SPG-Layout achieves a balanced optimization of semantic realism and physical plausibility. To evaluate performance in these complex settings, we constructed a new benchmark comprising 500 diverse non-Manhattan environments. Extensive experiments demonstrate that SPG-Layout consistently and...
  </details>  
- **[PWM-ArtGen: Part World Model for Articulated Object Generation](https://arxiv.org/abs/2607.02045v1)**  
  Authors: Wentao Zheng, Ancong Wu  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2607.02045v1.pdf)  
  <details><summary>Abstract</summary>

  The key challenge in articulated 3D object generation from a single image is accurately predicting the underlying kinematic structure. Existing methods either infer kinematic parameters directly from a static image that lacks dynamic part-level kinematic relationships, or estimate parameters from visual dynamics generated from a single image, which is prone to accumulated errors of two steps. Moreover, the limited scale and diversity of existing annotated datasets further hinder generalization to complex, real-world objects. To overcome these limitations, we propose to learn the joint distribution of visual dynamics and kinematic parameters. Recognizing that articulated objects can be formulated as dynamic systems, we propose a unified Part World Model called PWM-ArtGen. To leverage unannotated data, this model couples action diffusion and image diffusion with independent diffusion timesteps, which enables visual branch co-training. We further curate a photorealistic dataset of 19.7k part-level image pairs without kinematic annotations, to support co-training. Experiments demonstrate that...
  </details>  
  Keywords: articulated object generation  
- **[Ink3D: Sculpting 3D Assets with Extremely Complex Textures via Video Generative Models](https://arxiv.org/abs/2607.01222v1)**  
  Authors: Yue Han, Chong Li, Zhening Liu, Cong Huang, Fang Deng, Yong Liu, Fangyun Wei, Yan Lu  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2607.01222v1.pdf)  
  <details><summary>Abstract</summary>

  Recent 3D generative models can synthesize high-quality geometry but often struggle to reproduce intricate textures from reference images, largely due to the scarcity of large-scale 3D training data with rich surface appearance. In contrast, visual generative models are trained on datasets several orders of magnitude larger and excel at modeling complex visual patterns. Motivated by this gap, we introduce Ink3D, a framework that bridges 3D generation with large-scale video generative models to synthesize extremely complex textures. Ink3D first reconstructs a white-mesh geometry using an off-the-shelf 3D generation model. It then employs OrbitPainter, a conditional video generative model, to produce dense orbit-scan videos capturing object appearance across viewpoints. To convert these views into coherent textures, we introduce TextureOptimizer, a neural baking module that integrates dense multi-view observations while mitigating geometry inconsistencies arising from video generation. By decoupling geometry and texture synthesis and leveraging large-scale pretrained video priors, Ink3D enables significantly richer...
  </details>  
  Keywords: texture synthesis  
- **[Pano2World: End-to-End 3D Generation via Unified Multi-View Sequences](https://arxiv.org/abs/2607.00832v2)**  
  Authors: Zhenjia Li, Jinrang Jia, Yifeng Shi  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2607.00832v2.pdf)  
  <details><summary>Abstract</summary>

  A single panorama captures the full visual sphere from one camera center, yet confines users to looking around in place without enabling true scene exploration. Converting a single panorama into a persistent, renderable 3D representation for free-viewpoint navigation has attracted growing interest; existing methods either adopt iterative per-view completion that propagates inpainting results to update the underlying geometry, leading to progressive error accumulation and cumbersome multi-step pipelines, or leverage the temporal consistency priors of video generation models, yet the continuous-trajectory constraint intrinsic to such models limits their flexibility in covering scenes from multiple directions simultaneously. We present Pano2World, which takes a single indoor panorama as input and directly outputs a persistent, explorable 3D Gaussian scene. Given the source panorama, Pano2World first reconstructs a coarse 3D Gaussian proxy and renders it at adaptively sampled nearby poses to obtain geometrically aligned guidance panoramas; a panoramic diffusion model then jointly denoises all target...
  </details>  
- **[Vitality-Aware Compression for Efficient Image-to-Shape Diffusion Transformers](https://arxiv.org/abs/2607.00382v1)**  
  Authors: Jaeah Lee, Hyunjin Kim, Jaewoong Cho, Gihyun Kwon  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2607.00382v1.pdf)  
  <details><summary>Abstract</summary>

  We propose the first compression approach for image-to-shape Diffusion Transformers (DiTs) that substantially reduces model size while preserving geometric fidelity. Despite remarkable progress in 3D shape generation, large DiT-based models remain computationally prohibitive in resource-constrained settings. Furthermore, it is difficult to directly transfer existing diffusion model compression strategies developed for different domains to 3D generation, and prior 3D efficiency approaches focus primarily on inference speed rather than backbone compression. To address this limitation, we build a geometry-aware compression framework tailored to image-to-shape DiTs. Guided by the observation that 3D DiT layers exhibit non-uniform importance for geometry synthesis, we introduce a vitality-guided framework integrating structured pruning, adaptive quantization, and targeted fine-tuning. Our method achieves up to 66% model-size reduction across state-of-the-art image-to-3D models while maintaining synthesis fidelity comparable to full-sized counterparts. This highlights the potential of our framework as a plug-and-play solution for efficient 3D shape generation across diverse models.
  </details>  
  Keywords: image-to-3d  

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
- **[Neural Particle Automata: Learning Self-Organizing Particle Dynamics](https://arxiv.org/abs/2601.16096v3)**  
  Authors: Hyunsoo Kim, Ehsan Pajouheshgar, Sabine Süsstrunk, Wenzel Jakob, Jinah Park  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2601.16096v3.pdf)  
  <details><summary>Abstract</summary>

  We introduce Neural Particle Automata (NPA), a Lagrangian generalization of Neural Cellular Automata (NCA) from static lattices to dynamic particle systems. Unlike classical Eulerian NCA where cells are pinned to pixels or voxels, NPA model each cell as a particle with a continuous position and internal state, both updated by a shared, learnable neural rule. This particle-based formulation yields clear individuation of cells, allows heterogeneous dynamics, and concentrates computation only on regions where activity is present. At the same time, particle systems pose challenges: neighborhoods are dynamic, and a naive implementation of local interactions scale quadratically with the number of particles. We address these challenges by replacing grid-based neighborhood perception with differentiable Smoothed Particle Hydrodynamics (SPH) operators backed by memory-efficient, CUDA-accelerated kernels, enabling scalable end-to-end training. Across tasks including morphogenesis, point-cloud classification, and particle-based texture synthesis, we show that NPA retain key NCA behaviors such as robustness and self-regeneration, while...
  </details>  
  Keywords: texture synthesis  
- **[ThermoSplat: Cross-Modal 3D Gaussian Splatting with Feature Modulation and Geometry Decoupling](https://arxiv.org/abs/2601.15897v3)**  
  Authors: Zhaoqi Su, Shihai Chen, Xinyan Lin, Liqin Huang, Zhipeng Su, Xiaoqiang Lu  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2601.15897v3.pdf)  
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
- **[One-Shot Feed-Forward 360$^{\circ}$ Animatable Avatar via Inpainted UV-Space Gaussian Modeling](https://arxiv.org/abs/2601.12770v2)**  
  Authors: Shuling Zhao, Dan Xu  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2601.12770v2.pdf)  
  <details><summary>Abstract</summary>

  Building one-shot 3D animatable head avatars is an important yet challenging problem. Existing methods generally collapse under large camera pose variations, compromising the realism of 3D avatars. In this work, we propose a new framework to tackle the novel setting of one-shot 3D full-head animatable avatar reconstruction in a single forward pass via inpainted UV-space Gaussian modeling, enabling 360$^\circ$ rendering views and real-time animation. To facilitate efficient animation control, we model 3D head avatars with Gaussian primitives embedded on the surface of a parametric face model within the UV space, and project the input image features to the UV space, resulting in incomplete local UV feature maps. To inpaint the missing regions, we obtain knowledge of full-head geometry and textures from rich 3D full-head priors within a pretrained 3D generative adversarial network (GAN) for global full-head feature extraction and multi-view supervision. Specifically, to enhance the fidelity of 3D reconstruction during...
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
- **[CRAG: Can 3D Generative Models Help 3D Assembly?](https://arxiv.org/abs/2602.22629v3)**  
  Authors: Zeyu Jiang, Sihang Li, Siqi Tan, Chenyang Xu, Juexiao Zhang, Julia Galway-Witham, Xue Wang, Scott A. Williams, Radu Iovita, Chen Feng, Jing Zhang  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2602.22629v3.pdf) | [![Project](https://img.shields.io/badge/-Project-blue)](https://ai4ce.github.io/CRAG)  
  <details><summary>Abstract</summary>

  Most existing 3D assembly methods treat the problem as pure pose estimation, rearranging observed parts via rigid transformations. In contrast, human assembly naturally couples structural reasoning with holistic shape inference. Inspired by this intuition, we reformulate 3D assembly as a joint problem of assembly and generation. We show that these two processes are mutually reinforcing: assembly provides part-level structural priors for generation, while generation injects holistic shape context that resolves ambiguities in assembly. Unlike prior methods that cannot synthesize missing geometry, we propose CRAG, which simultaneously generates plausible complete shapes and predicts poses for input parts. Extensive experiments demonstrate state-of-the-art performance across in-the-wild objects with diverse geometries, varying part counts, and missing pieces. Project Page: https://ai4ce.github.io/CRAG/
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
  Keywords: rigging, articulated object generation  
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
- **[Symmetry Matters: Auditing and Symmetrizing 3D Generative Models](https://arxiv.org/abs/2512.18953v3)**  
  Authors: Nicolas Caytuiro, Ivan Sipiran  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2512.18953v3.pdf)  
  <details><summary>Abstract</summary>

  Symmetry is a strong prior present in many object categories, yet standard benchmarks for 3D generative models rarely report whether this prior is preserved. We study symmetry preservation in unconditional point cloud generation. We first audit the symmetry of generated shapes by several 3D generative models and compute a normalized symmetry score based on the Chamfer Distance (CD). We show that although current 3D generative models achieve competitive results under standard evaluation, they reveal a persistent symmetry gap when a symmetry-aware evaluation protocol is applied. To test whether this gap is merely inherited from the training data, we evaluate these models over a mirrored-objects dataset derived from ShapeNet and analyze symmetry dynamics during training. Mechanism-inspired diagnostic tests were conducted at the sampling and latent-representation levels to further show that reflection symmetry is not reliably encoded in the learned generative process. Finally, to address this gap, we propose a data-centric symmetry-based...
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
  Keywords: novel view synthesis, 3d scene reconstruction  
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

### August 2026
- **[Instance-Guided Report Anchoring for Text-Free 3D Abnormality Segmentation in Chest CT](https://arxiv.org/abs/2609.00447v1)**  
  Authors: Zhenyu Bu, Haoyan Ding, Chushu Shen, Xinyuan Zheng, Peiyu Duan, Xueqi Guo, Sepehr Farhand, Yoshihisa Shinagawa, Gerardo Hermosillo, Chaowei Wu  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2609.00447v1.pdf)  
  <details><summary>Abstract</summary>

  Accurate 3D abnormality segmentation in chest CT requires dense spatial supervision, but obtaining expert voxel-level labels is costly. Radiology reports, however, are routinely generated during clinical interpretation and contain instance-specific descriptions that can provide additional guidance without new dense annotation. Existing vision-language grounding methods typically require report-derived findings at inference, making localization dependent on paired text and limiting each forward pass to a queried finding. We propose Instance-Guided Report Anchoring (IGRA), a model-agnostic module that preserves the correspondence between each annotated abnormality instance and the report finding that describes it. IGRA pools each instance representation and anchors it to the corresponding finding embedding during training; all text-related components are discarded at inference. We further reformulate free-text grounding on ReXGroundingCT as multi-label volumetric segmentation by merging same-category instances, allowing all abnormality categories to be predicted in one image-only forward pass. IGRA improves Dice by 22.5% over the strongest image-only baseline (30.93...
  </details>  
- **[SeqAlign3DVG: A Sequence-Aligned Benchmark and Voxel Reasoning Framework for 3D Visual Grounding](https://arxiv.org/abs/2608.30451v1)**  
  Authors: Yi Zhang, Yi Wang, Yueting Wu, Kaiyue Yang, Yuejiao Su, Lap-Pui Chau  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2608.30451v1.pdf)  
  <details><summary>Abstract</summary>

  Image-based 3D visual grounding is critical for embodied agents, yet existing benchmarks suffer from loose text-observation alignment and neglect temporal ordering. We introduce SeqAlign3DVG, a novel benchmark dedicated to temporally ordered and strictly observation-aligned image-based 3D visual grounding. Unlike prior works using order-agnostic views or global point clouds, SeqAlign3DVG ensures all expressions are human-verified and strictly grounded in the provided RGB observations (single frames or ordered observation sequences). It comprises 9,622 single-view and 14,493 sequence samples featuring rich descriptions, complex relations, and multi-instance ambiguities. To tackle this benchmark, we propose a unified voxel-based pipeline featuring Relevance-Ordered Voxel Memory (ROVM) and Progressive Language-Voxel Fusion (PLVF). ROVM dynamically ranks and aggregates multi-view evidence via a conservative memory to mitigate noisy observations, while PLVF performs coarse-to-fine spatial-linguistic reasoning for precise disambiguation. Our approach achieves state-of-the-art performance under the depth-free protocol, significantly improving localization for targets defined by complex relations and appearance cues.
  </details>  
- **[ScenePilot: Grow-and-Repair Policy for Text-Driven 3D Indoor Scene Generation](https://arxiv.org/abs/2608.30307v1)**  
  Authors: Jiawei Zhang, Hongsong Wang, Pan Zhou  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2608.30307v1.pdf)  
  <details><summary>Abstract</summary>

  Text-driven 3D indoor scene generation has advanced from dataset-bound layout modeling to open-vocabulary synthesis with large language and vision-language models. Yet existing methods remain limited: one-pass generators often yield geometrically invalid layouts, heavy post-hoc optimization is costly and unstable, and prompt-only planners lack reusable layout priors for functional grouping and object relations. We propose \textbf{ScenePilot}, a retrieval-augmented \textbf{Grow-and-Repair} framework that formulates scene generation as prior-guided incremental growth with learned rectification. Given a prompt, the Hierarchical Retrieval-Augmented Planning (HRAP) module retrieves room-, group-, and anchor-level layout priors to support functional group planning. A text-driven base generator then inserts object groups sequentially, while the Reinforcement Multimodal Repair (RMR) module performs lightweight local correction after each insertion and a final global repair after completion. To train this policy, we construct \textbf{SceneReverse-17k}, a repair-trajectory dataset built by perturbing high-quality 3D scenes in position, rotation, and scale, then using inverse operations as executable rectification targets....
  </details>  
- **[Generalization over Memorization: Generalization-Aware Diffusion Adaptation for Single-Image Multi-View Synthesis](https://arxiv.org/abs/2608.29233v1)**  
  Authors: Jie Li, Xingchen Zou, Yuxuan Liang  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2608.29233v1.pdf)  
  <details><summary>Abstract</summary>

  We present the winning solution to the ACM Multimedia 2026 Grand Challenge on Single-Image Guided Multi-Angle Image Synthesis. It ranks first among 293 registered teams; 56 teams obtained at least one scored submission on the public Phase-A leaderboard. With only 40 training scenes, the challenge requires 26 target views from one RGB model and one forward pass per view; it prohibits explicit geometry, external rendering, chained generation, candidate selection, and post-processing. We identify a critical model-selection failure: shared training and validation scenes make memorization appear as transferable view control. We therefore introduce GoM. Short for Generalization over Memorization, the framework combines scene-disjoint validation, exposure-matched selection, and targeted diffusion adaptation. Its synthesis model adapts a 4B rectified-flow DiT using rank-32 LoRA, optimizer restarts, late-checkpoint averaging, and VAE decoder tuning. More than 300 offline experiments and 24 online submissions show that validation design and training-trajectory control can matter as much as architecture...
  </details>  
  Keywords: multi-view synthesis  
- **[Ground-to-Satellite Localization in Unconstrained Image Collections for 3D Scene Reconstruction](https://arxiv.org/abs/2608.29211v1)**  
  Authors: Angel Daruna, Ben Southall, Niluthpol Chowdhury Mithun, Kshitij Minhas, Nicholas Meegan, Qiao Wang, Bogdan Matei, Supun Samarasekera, Rakesh Kumar  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2608.29211v1.pdf)  
  <details><summary>Abstract</summary>

  Ground image localization with respect to satellite imagery is a key enabler for metrically-accurate, geo-localized 3D scene reconstruction from unconstrained image collections. Existing cross-view localization methods have strict requirements such as panoramic imagery or known initial locations, limiting their applicability for in-the-wild reconstruction settings. We propose a robust hierarchical cross-view localization framework that leverages geometric constraints from Structure-from-Motion (SfM) models derived from unconstrained ground image collections. Our method generates coarse-to-fine pose hypotheses through a cross-view matching approach and aggregates noisy predictions across SfM model(s) using Kernel Density Estimation to recover consensus alignments while filtering outliers. Experiments demonstrate reliable localization performance from challenging image collections. Empirically we found satellite-referenced alignment enables accurate metric scale estimation, doppelgänger detection, and merging of disjoint SfM reconstructions, resulting in more complete, geo-localized site models than are possible with SfM alone.
  </details>  
  Keywords: 3d scene reconstruction  
- **[Chat-Edit-3D++: Interactive 3D and 4D Scene Editing via Large Language Models](https://arxiv.org/abs/2608.29137v1)**  
  Authors: Shuangkang Fang, Yufeng Wang, Yi-Hsuan Tsai, Wenrui Ding, Yi Yang, Shuchang Zhou, Ming-Hsuan Yang  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2608.29137v1.pdf) | [![GitHub](https://img.shields.io/github/stars/Fangkang515/CE3D?style=social)](https://github.com/Fangkang515/CE3D)  
  <details><summary>Abstract</summary>

  Recent work on image content manipulation based on vision-language pre-training models has been effectively extended to text-driven 3D scene editing. However, existing schemes for 3D scene editing still have certain shortcomings, hindering their further development as interactive design tools. Such schemes typically adhere to fixed input patterns, limiting flexibility in text input. Furthermore, their editing capabilities are constrained by a single or a few 2D visual models and require intricate pipeline design to integrate these models into 3D reconstruction processes. To address the aforementioned issues, we propose the Hash-Atlas network, which reformulates 3D scene editing as operations on 2D atlas images, thereby achieving a workflow decoupling of the 2D editing and 3D reconstruction processes. Building on this foundation, we introduce a dialogue-based 3D scene editing approach, termed CE3D++, which is centered on a large language model (LLM) that allows arbitrary textual input from users and interprets their intentions, subsequently facilitating...
  </details>  
- **[ReconSplat: Generalizable 3D Scene Reconstruction Beyond Observed Views](https://arxiv.org/abs/2608.28895v1)**  
  Authors: Giuseppe Stracquadanio, Kevin Raj, Julia Grabinski, Stefan Roth  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2608.28895v1.pdf)  
  <details><summary>Abstract</summary>

  We introduce ReconSplat, a feed-forward model for 3D scene reconstruction that aims to address the longstanding trade-off between plausible view generation for unobserved regions and geometric consistency, providing both geometrically aligned novel views and sharp depth estimates. Our approach builds on 3D Gaussian splatting (3DGS) as an intermediate differentiable scene representation and integrates it with a multi-view latent diffusion model (MV-LDM) trained to act simultaneously as a refiner and an inpainter for appearance and scene geometry. We enforce geometric consistency by guiding the diffusion process with variational 3D latent features for appearance and geometry, encoded by the feed-forward 3DGS representation and rasterized to 2D latent space. ReconSplat produces both photorealistic novel views and accurate depth maps on real-world benchmarks, RealEstate10K and DL3DV-10K, outperforming existing methods in challenging extrapolation setups. Notably, ReconSplat allows the extrapolation of unseen and challenging viewpoints jointly with coherent and precise scene geometry.
  </details>  
  Keywords: 3d scene reconstruction  
- **[Inter-3D VQA: A Roadside Multimodal Benchmark for 3D Spatiotemporally Grounded Visual Question Answering](https://arxiv.org/abs/2608.28762v1)**  
  Authors: Shaozu Ding, Linan Song, Dajiang Suo  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2608.28762v1.pdf) | [![GitHub](https://img.shields.io/github/stars/ASU-Suo-Lab/Inter-3D-VQA?style=social)](https://github.com/ASU-Suo-Lab/Inter-3D-VQA)  
  <details><summary>Abstract</summary>

  Recent advances in visual question answering (VQA) and multimodal large language models (MLLMs) have enabled natural-language reasoning over traffic scenes. However, existing benchmarks are largely built from ego-vehicle views or 2D roadside videos, limiting their ability to evaluate 3D-grounded reasoning over real-world distances, trajectories, infrastructure topology, and safety-critical interactions. We introduce Inter-3D VQA, a large-scale roadside multimodal benchmark for 3D spatiotemporally grounded VQA at intersections. Built from synchronized point clouds and multi-view images, Inter-3D VQA contains 407K QA pairs covering lane-level positions, object relationships, motion patterns, and near-miss-oriented interaction reasoning. We further propose Inter-Geo, an MLLM baseline that integrates object- and scene-level aligned LiDAR representations, and Inter-Metrics, a unified evaluation framework for textual consistency, numerical accuracy, and semantic correctness. Experiments show that Inter-Geo outperforms image-based VLMs, especially on grounded spatial and temporal reasoning tasks. Our benchmark and codes are available at https://github.com/ASU-Suo-Lab/Inter-3D-VQA .
  </details>  
- **[WilLaGS: Latent-Conditional 3D Appearance Fields for Robust Gaussian Splatting In-the-Wild](https://arxiv.org/abs/2608.28240v1)**  
  Authors: Yuhao Bai, Qianqiu Tan, Lilong Chen, Huanhuan Lv, Lijun Chen  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2608.28240v1.pdf)  
  <details><summary>Abstract</summary>

  3D Gaussian Splatting (3DGS) delivers real-time and high-fidelity rendering but remains challenged by unconstrained in-the-wild scenes, where drastic appearance variations and transient objects violate multi-view consistency. Existing methods are fundamentally limited by independent and discrete embeddings that struggle to capture continuous environmental changes or model spatially-varying local illumination. To address these limitations, we propose \textbf{WilLaGS}, a unified framework for robust 3D scene reconstruction and generative appearance synthesis under unconstrained settings. Specifically, we introduce a generative appearance model where a $β$-VAE learns a structured and continuous manifold of global appearance. Conditioned on the latent code, we construct a 3D neural appearance field that generates dynamic Tri-Plane features to encode spatially-varying local illumination effects. Furthermore, to suppress transient artifacts, we present a self-supervised perceptual masking mechanism that leverages a Teacher-Student (EMA) architecture to derive a stable scene consensus, robustly identifying inconsistent regions via perceptual discrepancies. Extensive experiments on multiple datasets demonstrate that...
  </details>  
  Keywords: 3d scene reconstruction  
- **[Cyc3D: Evaluating Cyclic Structural Stability and Asset Usability in Image-to-3D Generation](https://arxiv.org/abs/2608.28080v1)**  
  Authors: Liwen Zhang  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2608.28080v1.pdf)  
  <details><summary>Abstract</summary>

  Image-conditioned 3D generation has advanced rapidly, yet existing evaluation protocols largely judge rendered-view plausibility and semantic alignment, overlooking whether a generator forms a stable 3D interpretation and produces assets usable in graphics pipelines. We introduce Cyc3D, a multidimensional benchmark that evaluates image-to-3D generation along two complementary axes: Cross-View Object Consistency and Representation Quality. At the asset level, Cyc3D measures whether object identity remains semantically coherent across rendered viewpoints. At the model level, we propose View-Cycle Structural Consistency, a closed-loop render-regenerate-align protocol that repeatedly re-observes a generated asset from novel views and quantifies geometric, perceptual, and semantic drift across generations. To assess native asset usability beyond rendered appearance, Cyc3D further evaluates geometric structure, reference-image fidelity, mesh discretization and efficiency, and UV parameterization quality. Together, these diagnostics expose failures obscured by a single perceptual score and provide interpretable evidence of both model instability and representation defects. Experiments on five representative image-to-3D systems...
  </details>  
  Keywords: image-to-3d  
- **[Procedura: Agentic 3D Modeling with Procedural Control](https://arxiv.org/abs/2608.26238v1)**  
  Authors: Youtian Lin, Yikang Yang, Zhanpeng Hu, Mengqi Zhou, Feihu Zhang, Xun Cao, Jiaheng Liu, Yao Yao  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2608.26238v1.pdf)  
  <details><summary>Abstract</summary>

  Native 3D generators now recover impressive mesh geometry from a single image. However, a dense mesh stays soft where a machined object should be sharp, it carries no part decomposition, and it exposes no parameter a user could edit. To address this, we explore the paradigm of 3D shape as code, leveraging and scaling the coding ability of an LLM for 3D modeling. We introduce Procedura, a novel 3D modeling agent framework that writes an object as a procedural assembly, a parametric program whose named parts are joined by typed, machine-checkable mates. From a text prompt, the agent plans the object as an assembly graph and writes the program part by part, solving each placement from the mated frames rather than guessing it, and admitting a part only once compile, mate, and connectivity checks pass. A decoupled vision critic then refines the assembly one diagnosed fix at a time. Moreover,...
  </details>  
  Keywords: 3d generator  
- **[Luce: Relightable Gaussians for 3D Asset Generation](https://arxiv.org/abs/2608.23943v1)**  
  Authors: Mayank Singh, Michele Stoppa, Alvise Memo, Rui Yu, Harsha Kalli, Srimanth Gunturi, Muhammad Ahmed Riaz, Behrooz Shahsavari, Waleed Abdulla, David E. Jacobs  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2608.23943v1.pdf)  
  <details><summary>Abstract</summary>

  High-fidelity image-to-3D generation requires a 3D representation that captures both geometry and appearance. To support relighting and integration into standard rendering pipelines, the representation should include physically based rendering (PBR) modalities such as albedo, metallic-roughness, and surface normals. We propose Luce, a 3D representation that unifies geometry and PBR materials within a voxelized multimodal Gaussian cloud, using dedicated Gaussian primitives for each modality. A variational autoencoder compresses this representation into a unified material-aware latent space. A rectified-flow transformer generates this latent from a single image, conditioned on multi-layer features from a pretrained image encoder that preserve both semantic context and fine spatial detail. The latent then decodes into relightable PBR Gaussians and an optional textured mesh with a tangent-space normal map. On Toys4K, Luce achieves state-of-the-art single-image-to-3D generation, improving FID by 28% over the strongest baseline. We further introduce a benchmark of AI-generated images, on which Luce improves the CLIP...
  </details>  
  Keywords: pbr materials, image-to-3d, 3d asset generation  
- **[SceneReGen: Generative Reconstruction of 3D Scenes from a Single Image](https://arxiv.org/abs/2608.23930v1)**  
  Authors: Zefan Tian, Yuteng Ye, Yiheng Zhang, Yuhang Yang, Xueqiang Lv, Shizhou Zhang, Le Liu, Di Xu  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2608.23930v1.pdf)  
  <details><summary>Abstract</summary>

  Single-image 3D scene reconstruction must complete partially observed objects and place them coherently in a shared observation-aligned scene frame. Object-level generative priors offer strong completion ability, but their centered, scale-normalized outputs are typically expressed in an object frame, creating a fundamental representation gap between object generation and scene reconstruction. We introduce SceneReGen, a generative reconstruction framework that reinterprets scene reconstruction as the generation and assembly of complete object assets in a shared observation-aligned scene frame. SceneReGen addresses the generation-reconstruction gap through selective pose factorization: each object's observed orientation is encoded directly in the generated mesh, while translation and scale are estimated from instance-level and global scene evidence. Given a scene image and instance masks, a geometry encoder extracts dense cues; learnable shape queries condition a pretrained DiT-based 3D generator to produce complete meshes in their observed orientations, while position queries fuse object and scene features to assemble them in the...
  </details>  
  Keywords: 3d generator, 3d scene reconstruction  
- **[Seeing the Unseen: Semantic-in-Gaussian for Sparse-View 3D Generalization](https://arxiv.org/abs/2608.22740v1)**  
  Authors: Zeyang Bai, Yunpeng Wang, Yunbiao Wang, Jun Xiao  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2608.22740v1.pdf)  
  <details><summary>Abstract</summary>

  Generalizable 3D Gaussian Splatting (G-3DGS) has emerged as a promising approach for novel view synthesis undersparse-view settings. However, existing frameworks remain restricted by pixel-aligned Gaussian estimation, whichstruggles in partially observed or occluded regions and often leads to incomplete surfaces or structural collapse. Toaddress these challenges, we propose SeeU (Seeing the Unseen), a novel G-3DGS framework. We frame its core design asSemantic-in-Gaussian: semantic-conditioned refinement in Gaussian space. Specifically, we introduce a Cross-viewEntropy-Aware (CEA) module that aggregates multi-view semantic and geometric cues into compact embeddings. Theseembeddings guide the Conditional Gaussian Transformer, which applies residual updates to coarse Gaussians, helpingrecover under-constrained regions of partially observed structures while preserving surface consistency. Comprehensiveexperiments on multiple benchmarks demonstrate that SeeU consistently improves rendering quality and structuralcompleteness while retaining efficient feed-forward inference. Especially under challenging extrapolation settings,SeeU achieves an average improvement of 2.44 dB in PSNR compared to recent SOTA G-3DGS methods.
  </details>  
  Keywords: novel view synthesis  
- **[DiGS-Avatar: Single-Image Animatable 3D Human Reconstruction via UV-Space Diffusion](https://arxiv.org/abs/2608.20759v1)**  
  Authors: Jiakun Li, Li Fang, Hao Zhu, Fei Hu, Long Ye, Yuan Zhang, Jinyao Yan  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2608.20759v1.pdf) | [![GitHub](https://img.shields.io/github/stars/KLMAV-CUC/DiGS-Avatar?style=social)](https://github.com/KLMAV-CUC/DiGS-Avatar)  
  <details><summary>Abstract</summary>

  Single-image 3D human reconstruction often suffers from over-smoothed textures and geometric inconsistencies. While diffusion models improve generative quality, their reliance on multi-view synthesis prior to 3D reconstruction is computationally expensive and prone to view inconsistency. We propose DiGS-Avatar, which reformulates this task as an efficient, diffusion-based UV-latent completion task, ensuring 3D consistency by design. To capture accurate spatial structure, we introduce a teacher-student framework where a multi-view teacher provides geometrically aligned pseudo-ground-truth latents to supervise a single-view diffusion student. Treating this inferred latent as a robust structural skeleton, our method injects high-level semantic features to accurately recover fine textural details without disrupting spatial integrity. The refined representation is then decoded into 3D Gaussian primitives. Extensive experiments demonstrate that DiGS-Avatar achieves state-of-the-art or highly competitive visual fidelity and zero-shot generalization, while reconstructing a fully animatable 3D avatar in just 0.71 seconds. Code is available at https://github.com/KLMAV-CUC/DiGS-Avatar.
  </details>  
  Keywords: multi-view synthesis  
- **[AffordAny: Open-World 3D Affordance Grounding from Monocular RGB Images via Vision-Language-Guided Geometric Reasoning](https://arxiv.org/abs/2608.20720v1)**  
  Authors: Junqi Wu, Kaihua Tang, Xuanwen Chen, Hongzhi Li, Jianqiang Huang, Xian-Sheng Hua  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2608.20720v1.pdf)  
  <details><summary>Abstract</summary>

  Open-world 3D affordance grounding requires localizing functional object parts in 3D given free-form language queries. Existing methods typically assume pre-built object-centric 3D geometry and closed affordance ontologies, limiting deployment from raw RGB observations. We present AffordAny, an end-to-end framework that uses one monocular RGB image to construct large-scale text-conditioned 3D part supervision, ground affordances with a frozen vision-language model (VLM) guided decoder, and improve open-world generalization through pseudo-label self-training. Our automated pipeline produces a benchmark of 5,334 objects and 10,633 part-level samples spanning 473 categories, an order-of-magnitude increase in categorical diversity over prior work. The decoder progressively fuses frozen Cosmos-2B features with 3D geometry through spatial projection, instruction-conditioned semantic compression, and bidirectional geometry-semantics interaction. Minimal-perturbation pseudo-label self-training further adds new objects without human annotation. Under a systematic generalization protocol evaluating unseen objects, unseen categories, and unseen instruction paraphrases, our approach achieves 0.428 IoU on unseen objects and 0.315 IoU on...
  </details>  
- **[MultiCube: Compositional 3D Generation With Part-Level Semantic and Spatial Control](https://arxiv.org/abs/2608.20448v1)**  
  Authors: Ava Pun, Kangle Deng, Yiheng Zhu, Jun-Yan Zhu, Maneesh Agrawala, Tinghui Zhou  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2608.20448v1.pdf) | [![Project](https://img.shields.io/badge/-Project-blue)](https://multi-cube.github.io)  
  <details><summary>Abstract</summary>

  Digital 3D objects used in games and animation are often required to be compositional; that is, decomposed into semantically meaningful parts. Recent 3D generation methods can produce high-quality compositional objects conditioned on image or text prompts. Yet, such global conditioning lacks the precise part-level controllability required for professional creative workflows. To address this, we introduce MultiCube, a novel compositional 3D generation method that provides explicit, independent control over both the semantics and spatial arrangement of each part. MultiCube takes as input a global text prompt, a text schema specifying the desired parts, and a spatial layout indicating the bounding boxes of the parts in the given schema. It outputs a 3D object composed of distinct meshes, one per specified part, that adhere to the given semantic and spatial conditions. Our approach employs a two-stage diffusion process, first generating a schema- and layout-aligned monolithic mesh, then decomposing the mesh into individual...
  </details>  
- **[Block3D: Efficient Text-to-3D Generation via Block-Wise Diffusion](https://arxiv.org/abs/2608.19567v4)**  
  Authors: Bowen Cui, Weijie Wang, Zeyu Zhang, Yefei He, Mingda Lin, Haoyu Zhao, Yuanyu He, Donny Y. Chen, Feng Chen, Bohan Zhuang  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2608.19567v4.pdf)  
  <details><summary>Abstract</summary>

  While text-to-3D generation has advanced rapidly, achieving high geometric fidelity at low inference cost remains challenging. Existing text-to-3D methods either decode discrete shape tokens autoregressively or iteratively refine global 3D representations with diffusion or flow models. However, autoregressive decoding is sequential and cannot revise errors, whereas diffusion and flow-matching models repeatedly process the full representation, making high-quality generation increasingly expensive. In this paper, we propose Block3D, a block-wise diffusion framework that partitions the discrete shape-token sequence into contiguous blocks, generates the blocks autoregressively, and jointly denoises all tokens within the current block. To alleviate error accumulation, we introduce confidence-guided intra-block correction, which revises low-confidence tokens before each block is finalized. On a held-out set from TRELLIS-500K, Block3D reduces mean end-to-end generation time from 25.71 seconds to 4.99 seconds, achieving a $5.15\times$ speedup over the fine-tuned autoregressive baseline without sacrificing geometric fidelity.
  </details>  
  Keywords: text-to-3d  
- **[CL4D: Contrastive Language-4D Pretraining for Vision-Language Reasoning in Dynamic Scenes](https://arxiv.org/abs/2608.18734v2)**  
  Authors: Kumal Hewagamage, Isuranga Senavirathne, Sasika Amarasinghe, Hasitha Gallella, Dulanga Weerakoon, Vigneshwaran Subbaraju, Ranga Rodrigo  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2608.18734v2.pdf)  
  <details><summary>Abstract</summary>

  4D understanding and reasoning is a fundamental capability for embodied AI agents operating in dynamic physical environments. However, existing vision encoders are largely limited to static 2D images or 3D point clouds without temporal modeling, or to 2D videos that lack accurate geometric depth reasoning. Consequently, current approaches fail to jointly capture spatial structure and motion evolution in dynamic scenes. We present CL4D, the first foundational 4D vision encoder that directly operates on dynamic point clouds, trained with a contrastive learning objective to align spatio-temporal geometric representations with natural language descriptions. By learning a shared embedding space between text and 4D scene dynamics, CL4D enables zero-shot motion-to-text and text-to-motion retrieval in dynamic environments and serves as a foundational 4D vision encoder for downstream 4D vision-language tasks. Building on this encoder, we introduce 4DVLM, a 4D vision-language model that conditions language generation on dynamic geometric representations. 4DVLM is the first VLM...
  </details>  
- **[SemanticSlider3D: Training-Free Continuous Semantic Editing for 3D Objects](https://arxiv.org/abs/2608.18560v2)**  
  Authors: Ru Wang, Rahul Jain, Koichiro Niinuma, Aakar Gupta  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2608.18560v2.pdf)  
  <details><summary>Abstract</summary>

  Fine-grained control over continuous semantic attributes of 3D objects is essential for 3D content creation, but is not well supported by conventional 3D modeling workflows or prompt-based interaction with existing generative AI tools. While slider-based methods have proven effective for fine-grained semantic control in 2D image generation, no equivalent approach exists for 3D. Extending these 2D methods to 3D is non-trivial due to challenges unique to 3D, including geometric integrity and cross-view coherence. We present SemanticSlider3D, a technique for continuous semantic attribute editing of 3D objects that requires no per-attribute training. Given a user-specified attribute, our pipeline constructs a semantic editing direction in the latent space of a state-of-the-art 3D generation model, presenting a diverse and coherent spectrum of 3D variations. A technical validation on a dataset of 50 3D object-attribute pairs shows our method was preferred by all five human assessors across variation range, consistency, 3D object quality, and...
  </details>  
  Keywords: image-to-3d  
- **[GS-Voxel: Fitting-Free Structured Latents for Large-Scale 3DGS Generation](https://arxiv.org/abs/2608.17988v1)**  
  Authors: Ming Qian, Zijian Wang, Minchao Sun, Jincheng Xiong, Hang Zhang, Mu Xu, Chi Wang, Baoquan Chen  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2608.17988v1.pdf)  
  <details><summary>Abstract</summary>

  Many scalable latent 3D generators operate on structured tensors, whereas pre-optimized 3D Gaussian Splatting (3DGS) reconstructions are unordered, spatially irregular, and vary widely in primitive count. We present GS-Voxel, a fitting-free structured latent framework, and evaluate it for large-scale aerial 3D Gaussian scene generation. GS-Voxel deterministically converts a compatible pre-optimized 3DGS reconstruction into sparse active voxels without additional per-scene optimization, retaining the sub-voxel positions and rendering attributes of the selected primitives. A GS-specific factorized VAE then separately encodes voxel geometry and local Gaussian attributes into sparse 3D latents whose size grows with the number of occupied voxels rather than being limited by a fixed scene-wide primitive count. We train image-conditioned flow models in the GS-Voxel latent space to generate aerial 3DGS scenes. A key application enabled by GS-Voxel is large-area scene generation: overlap-aware tiled inference extends synthesis beyond a single training crop conditioned on satellite-view images. Our results show that...
  </details>  
  Keywords: 3d generator  
- **[DesignAgent3D: Interactive 3D Scene Editing via Designer-like Multimodal Reasoning](https://arxiv.org/abs/2608.21438v1)**  
  Authors: Xiujin Liu, Tianyu Yang, Yilun Zhao, Xiangliang Zhang  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2608.21438v1.pdf)  
  <details><summary>Abstract</summary>

  Text guided 3D scene editing provides an intuitive interface for modifying reconstructed environments, but remains difficult because natural language design requests are often semantically underspecified and must be grounded in cluttered 3D scenes. Existing methods typically formulate the task as one-shot conditional generation from a single prompt, failing to resolve ambiguous user intents or achieve precise spatial grounding. Consequently, they suffer from severe object localization drift, tracking failure under occlusions, and the notorious multi-view "sticker effect." To overcome these limitations, we present DesignAgent3D, an interactive multimodal agentic framework that reformulates 3D scene editing as a designer-like Plan-Perceive-Act paradigm. The agent first plans by interacting with the user to clarify underspecified design goals, then perceives by grounding the intended edit to specific objects or regions in the 3D scene, and finally acts by applying controlled visual modifications while preserving scene consistency. The edits are further integrated into the underlying 3D representation,...
  </details>  
- **[Spatial Temporal Synergy: Balancing Change and Invariance in Text Driven 3D Human Motion Editing](https://arxiv.org/abs/2608.16008v1)**  
  Authors: Shaohui Lin, Zhenwu Shi, Jingyu Gong, Jiao Xie, Yu Zhou, Baochang Zhang, Lizhuang Ma, Chia-Wen Lin  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2608.16008v1.pdf) | [![GitHub](https://img.shields.io/github/stars/ZhenwuShi/CIME.git?style=social)](https://github.com/ZhenwuShi/CIME.git)  
  <details><summary>Abstract</summary>

  Text-driven human motion editing aims to modify existing motion sequences according to natural language instructions while maintaining the structural consistency of the original motion. Existing diffusion-based approaches struggle to balance text-responsive "change" and inertial "invariance". They often rely on coarse spatial constraints and rigid uniform time assumptions, leading to spatial motion distortions and the destruction of intrinsic physical rhythms during variable-length editing. To handle these challenges, we propose Change and Invariance Motion Editing (CIME), a unified framework that comprehensively decouples change and invariance into spatial pose and temporal rhythm dimensions. For spatial poses, our method integrates an omni-supervised positive-negative learning mechanism comprising hierarchical retrospective feature supervision, subtle motion preservation, and triplet-based semantic alignment. For temporal rhythms, we introduce the Riemannian Non-uniform Integral Manifold Mapping (RNIMM) module, which achieves high-fidelity reproduction of physical beats in the edited text via kinematics-aware non-uniform timestamps. Extensive experiments on the MotionFix and STANCE Adjustment datasets...
  </details>  
- **[ES3D: Embedding Semantics into 3D Space for Component-Aware Editing](https://arxiv.org/abs/2608.15749v1)**  
  Authors: Xuancheng Jin, Rengan Xie, Jiayuan Lu, Wenting Zheng, Rui Wang, Yuchi Huo, Lincheng Li, Yingfeng Chen  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2608.15749v1.pdf)  
  <details><summary>Abstract</summary>

  Existing 3D editing methods have made notable progress in controllability, yet they remain limited in several important ways. Most approaches rely on text-driven editing, which struggles to express fine-grained visual changes intended by the user. Moreover, many methods require manually supplied 3D masks or introduce unintended changes to regions that should remain untouched. These limitations largely arise from the absence of fine-grained semantic understanding, making it difficult for existing models to retrieve or modify specific 3D components. We introduce ES3D, a framework that embeds semantics directly into 3D space, enabling component-aware retrieval and editing of a 3D asset conditioned on multiple local reference images and optional text queries. We first construct a 3D semantic embedding by projecting multi-view semantic features into the voxelized space of the asset. We then perform 3D component retrieval by computing feature similarity between the 3D semantic embedding and the semantic embeddings of image or text...
  </details>  
- **[MegaParts: Scaling Part-Aware 3D Object Generation to 300 Parts via Token-Efficient Autoregressive Modeling](https://arxiv.org/abs/2608.14783v1)**  
  Authors: Manwen Liao, Xinyu Lian, Jian Mao, Kaixu Chen, Li Luo, Jinghao Yan, Wanshui Gan, Qiao Yu, Weitian Zhang, Chunhua Shen, Guang Chen, Bo Dai, Xudong Xu, Zhaoyang Lyu  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2608.14783v1.pdf) | [![Project](https://img.shields.io/badge/-Project-blue)](https://expmaster.github.io/megaparts_webpage)  
  <details><summary>Abstract</summary>

  Part-aware 3D object generation is essential for graphics applications such as controllable modeling, editing, and articulation, where objects are represented as coherent assemblies of semantic parts. However, existing part-aware generation methods, do not scale well to highly complex objects. As the number of parts increases, generating detailed geometry becomes prohibitively expensive in token length and memory. We introduce MegaParts, a scalable autoregressive 3D generation framework to address this challenge by combining structured sequence modeling with a token-efficient vector-quantized shape tokenizer. Our tokenizer learns discrete latent representations for part-level geometry by minimizing token usage subject to high-fidelity reconstruction, enabling adaptive-length tokenization based on geometric complexity. On top of this compact representation, we train a large language model to generate object bounding boxes, part bounding boxes, and part shape tokens within a unified structured sequence. Combined with efficient long-context training strategy, our token-efficient formulation scales to objects with up to 300 parts...
  </details>  
- **[Synthesizing Post-Acetazolamide Cerebral Blood Flow Maps from Baseline MRI in Moyamoya Using 3D Generative AI](https://arxiv.org/abs/2608.14758v1)**  
  Authors: Julia Huang, Camila Gonzalez, Rydham Goyal, Aja Zou, Sasha Alexander, Michael Moseley, Moss Y. Zhao, Gary K. Steinberg  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2608.14758v1.pdf)  
  <details><summary>Abstract</summary>

  For patients with Moyamoya disease, impaired cerebrovascular reserve (CVR) is an important hemodynamic criterion for recommending extracranial-to-intracranial bypass surgery. Standard CVR assessment in this cohort uses paired arterial spin labeling (ASL) perfusion MRI acquired before and after acetazolamide (ACZ). When ACZ is contraindicated or avoided, the post-ACZ cerebral blood flow (CBF) map needed for hemodynamic assessment is unavailable. We propose CAE3D, a deterministic 3D conditional autoencoder that synthesizes post-ACZ CBF maps directly from pre-ACZ ASL input. We evaluated CAE3D against ten comparators, including deterministic and diffusion-style 3D baselines, a 2D contextual baseline, and frozen-encoder foundation-model adapters. CAE3D achieved the lowest held-out MAE (0.066), with SSIM 0.80 and PSNR 24.0 dB, and near-zero full-brain mean bias. Its MAE advantage was statistically significant over seven of eight trained-from-scratch baselines, excluding the 2D CAE_2D comparator; its SSIM and PSNR advantages were significant over all eight. Regional delta-CBF predictions compressed the dynamic range in...
  </details>  
- **[SCULPT: Subtractive Composition for 3D Part Generation](https://arxiv.org/abs/2608.13541v1)**  
  Authors: Sikuang Li, Chen Yang, Jiemin Fang, Jiazhong Cen, Yuhe Wei, Jichen Pang, Wei Shen, Qi Tian  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2608.13541v1.pdf)  
  <details><summary>Abstract</summary>

  Part-aware 3D generation aims to create digital assets that are coherent as complete objects while exposing structural parts for editing, material assignment, animation, and reuse. Existing methods impose this structure outside the native generation loop: segmentation-based methods partition an already generated shape, while additive methods synthesize parts from predefined layouts, boxes, or tokens and then reconcile them into a whole. The former preserves the generated geometry but fixes the object before part boundaries are determined; the latter exposes part cardinality but often leaves shared boundaries vulnerable to gaps, interpenetrations, and material discontinuities. In this paper, we propose SCULPT, a framework that addresses these challenges through subtractive composition. Given a complete object represented in a structured 3D latent space, SCULPT iteratively applies a joint split predictor to generate one extracted part together with the remaining object. The predictor performs a coupled denoising process conditioned on both the image and the current...
  </details>  
- **[GS$^{2}$CI: Robust Gaussian Splatting For Snapshot Compressive Imaging via Large Vision Model Priors](https://arxiv.org/abs/2608.13502v1)**  
  Authors: Yanming Yang, Chenxi Song, Ping Wang, Xin Yuan, Chi Zhang  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2608.13502v1.pdf)  
  <details><summary>Abstract</summary>

  Snapshot Compressive Imaging (SCI) offers an efficient solution for high-speed video acquisition and, under exposure-time camera--scene relative motion, multi-view scene capture by compressing temporal or spatial information into a single 2D measurement. While recent studies have explored SCI for 3D scene reconstruction, existing methods struggle with significant challenges due to information loss, limited viewpoint diversity, and the computational burden of jointly optimizing 3D representations and camera poses. In this work, we propose a novel framework that reconstructs high-quality 3D scenes from a single SCI measurement by leveraging 3D Gaussian Splatting (3DGS) and the powerful priors of large-scale vision foundation models (VFMs). Our primary reconstruction combines measurement-derived 3D VFM initialization with SCI-aware Gaussian optimization. After coarse-stage convergence, an auxiliary 2D VFM provides pseudo-view supervision at synthesized viewpoints for local appearance refinement. To further address the instability caused by ambiguous SCI supervision during 3DGS optimization, we introduce Opacity-Guided Splitting and Growth Regulation...
  </details>  
  Keywords: 3d scene reconstruction  
- **[PixSDS: Why Latent SDS Makes Noisy Pixels](https://arxiv.org/abs/2608.12997v1)**  
  Authors: Vsevolod Skorokhodov  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2608.12997v1.pdf) | [![Project](https://img.shields.io/badge/-Project-blue)](https://sevashasla.github.io/pixsds-webpage)  
  <details><summary>Abstract</summary>

  Score Distillation Sampling (SDS) enables text-to-3D generation by optimizing rendered images with a pretrained diffusion prior, but latent SDS often produces structured color artifacts and high-frequency texture noise. We identify a failure mode of latent SDS caused by VAE-induced pixel drift: the optimized image can move along pixel-space directions that are weakly constrained by the VAE encoder, so its latent representation remains clean and semantically meaningful while the image itself accumulates visible artifacts. We support this diagnosis with controlled 2D SDS experiments, VAE-only optimization, and a simplified analysis showing that encoder-like latent objectives can amplify image-space noise when the inverse mapping to pixels is underconstrained. Motivated by this observation, we propose PixSDS, a lightweight VAE-consistent gradient repair method. PixSDS decodes a latent SDS lookahead step and uses the decoded image as a clean direction for pixel-space optimization, reducing motion in VAE-inconsistent directions without retraining the diffusion model, changing the renderer,...
  </details>  
  Keywords: text-to-3d  
- **[TGRHuman: Text-Guided Realistic 3D Human Generation via Diffusion Renderer](https://arxiv.org/abs/2608.12175v1)**  
  Authors: Muxin Zhang, Chaohui Yu, Yuanwang Yang, Min Wei, Zhuo Su, Kun Li  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2608.12175v1.pdf)  
  <details><summary>Abstract</summary>

  Realistic 3D human generation plays a crucial role in many graphics applications. However, current methods still struggle to generate high-quality human geometry and texture while maintaining 3D consistency and inference efficiency. In this work, we address these limitations by introducing TGRHuman, a novel approach for generating realistic 3D humans from text. Our method decouples geometry and texture generation to alleviate the issues commonly encountered in NeRF-based methods. Instead of relying on slow, implicit score-distillation-based optimization, we directly use explicit multi-view observation generation and optimization for efficient 3D synthesis. For geometry generation, we propose a high-resolution generative module for multi-view normals together with a geometry-carving strategy that preserves view consistency and supports loose clothing. For texture generation, we produce spatially consistent RGB observations from densely sampled surrounding views using a carefully designed texture-prior acquisition strategy and a diffusion renderer, enabling detailed human texture synthesis. Experiments show that our method can generate...
  </details>  
  Keywords: text-to-3d, texture synthesis  
- **[Compact Feed-Forward 3D Gaussians via Saliency-Guided Primitive Merging](https://arxiv.org/abs/2608.10712v2)**  
  Authors: Tim-Felix Fassch, Jochen Kall, Cyrill Stachniss  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2608.10712v2.pdf)  
  <details><summary>Abstract</summary>

  3D scene reconstruction, modeling, and rendering are highly relevant for numerous tasks, and 3D Gaussian splatting has become a standard choice in this context. Its feed-forward variants provide fast reconstruction from sparse input views but often produce per-pixel primitives, leading to highly redundant and thus inefficient representations. We present a structure-aware merging pipeline that takes per-pixel primitives from any feed-forward method and consolidates them into a compact, content-adaptive Gaussian set while largely retaining visual quality at just $\frac{1}{20}^\text{th}$ of the Gaussians of a per-pixel method. We group spatially coherent Gaussians of similar appearance into variable-size clusters via adaptive superpixel segmentation guided by a saliency map, which allocates fine segments to textured regions and coarse segments to homogeneous areas. We compress each cluster into a compact latent representation through a learned encoder, then match and consolidate representations across views based on geometric overlap and feature similarity via a learned merger. A...
  </details>  
  Keywords: 3d scene reconstruction  
- **[MRIComp4Flow: Compression of 3D Brain MRI for Training Multi-Modal Generative Models](https://arxiv.org/abs/2608.10291v1)**  
  Authors: Lisa K. Fischer, Mykhailo Riabets, Daniel Rueckert, Benedikt Wiestler, Anke Meyer-Baese, Sandeep Nagar  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2608.10291v1.pdf) | [![GitHub](https://img.shields.io/github/stars/lisafis/MRIComp4Flow?style=social)](https://github.com/lisafis/MRIComp4Flow)  
  <details><summary>Abstract</summary>

  Large-scale multi-modal MRI datasets impose substantial storage and I/O costs, limiting the training of 3D generative models on commodity infrastructure. While lossy compression is known to preserve accuracy for discriminative segmentation networks, its effect on generative models, which must learn the full data distribution rather than a decision boundary, is unexplored. We study whether standard image codecs can effectively compress semantically rich brain tumor MRI while preserving the fidelity required to train and deploy a 3D MRI generative model. Each 3D volume is compressed with JPEG2000 or a near-lossless JPEG-LS pipeline. Next, a Wavelet Flow Matching model, conditioned on BraTS image sequences (T1n, T1c, T2, T2f), is trained on compressed data, and the resulting models are evaluated on the validation set. At a 20:1 compression ratio, synthesis quality is statistically equivalent to a model trained on uncompressed data within a pre-specified margin ($Δ$PSNR $<1$,dB, $Δ$SSIM $<0.02$; paired TOST $p=[[p]]$): mean...
  </details>  
- **[RAGMesh with FaME-G2E: Long-Form Text-Driven 3D Face Generation and Editing](https://arxiv.org/abs/2608.09186v1)**  
  Authors: Hao Li, Ju Dai, Feng Zhou, Mengting Shi, Haofei Wang, Zhen Song, Wei Zhou, Lei Li, Junjun Pan  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2608.09186v1.pdf) | [![Project](https://img.shields.io/badge/-Project-blue)](https://youtu.be/Yr0_XkpWcNk,) | [![Video](https://img.shields.io/badge/-Video-red)](https://youtu.be/Yr0_XkpWcNk)  
  <details><summary>Abstract</summary>

  Text-driven 3D face generation and editing remains challenging due to the difficulty of translating long-form descriptions into fine-grained facial geometry. Existing methods primarily align global textual semantics with facial structures but often struggle to capture subtle local deformations, such as eyebrow tension, cheek contraction, and asymmetric mouth motions, resulting in limited geometric fidelity and editing precision. To facilitate fine-grained text-driven facial modeling, we first construct FaME-G2E, a large-scale multimodal dataset containing detailed text--mesh annotations and paired text--blendshape samples for unified 3D facial generation and editing. Based on this dataset, we propose RAGMesh, a retrieval-augmented framework that leverages text-correlated geometric priors to improve high-fidelity facial synthesis and editing. Specifically, the Multi-Scale Retrieval Fusion (MSRF) module retrieves semantically consistent global and regional facial priors and fuses them in the blendshape space, suppressing conflicting local deformations while preserving coherent deformation patterns. Furthermore, we introduce Adaptive RAG-guided Supervision (AdaRAGS), a region-aware constraint that explicitly...
  </details>  
- **[View-Adaptive Renderer for View-Consistent 2D-to-3D Generation](https://arxiv.org/abs/2608.09110v1)**  
  Authors: U-Chae Jun, Jaeeun Ko, Jiwoo Kang  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2608.09110v1.pdf)  
  <details><summary>Abstract</summary>

  Reconstructing 3D shapes from a single image remains a fundamental yet challenging problem in computer vision. Traditional monocular 3D generation pipelines typically synthesize multiple views from a single input image before applying Neural Radiance Field (NeRF)-based reconstruction. However, inherent projective ambiguities often produce visual discontinuities across generated viewpoints, leading to inaccuracies in reconstructed 3D models. Current solutions either incur significant additional computational burdens or fail to adequately resolve practical inconsistencies between synthesized views. To address these limitations, we propose a novel viewpoint-adaptive neural rendering framework that enables robust 3D reconstruction even when given partially inconsistent multi-view inputs. Our approach introduces view-adaptive neural renderers that independently correct viewpoint-dependent errors while simultaneously sharing a global feature backbone to preserve structural coherence. Furthermore, we propose a self-attention fusion module that adaptively integrates multi-view information, ensuring geometric consistency without relying heavily on indirect regularizations or computationally intensive methods. Through extensive experiments, we demonstrate that...
  </details>  
- **[OccAnyScene: Towards Unified Indoor-Outdoor 3D Occupancy Prediction](https://arxiv.org/abs/2608.08696v2)**  
  Authors: Junjie Liu, Wanshui Gan, Zitong Dai, Guiping Cao, Yan Li, Ke Chen, Dongmei Jiang, Xiangyuan Lan, Jianguo Zhang  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2608.08696v2.pdf)  
  <details><summary>Abstract</summary>

  3D occupancy prediction is fundamental to scene understanding, yet existing 3D semantic occupancy methods are typically specialized to fixed scene types and occupancy protocols. We introduce Cross-Scene 3D Semantic Occupancy Prediction, a new task setting which requires a single model to handle heterogeneous indoor and outdoor scenes with varying cameras, spatial ranges, voxel specifications, and semantic taxonomies. This setting poses a fundamental challenge: achieving metric-consistent yet scene-adaptive image-to-3D lifting across varying camera configurations and scene scales. To address this challenge, we propose OccAnyScene, a pixel-frustum-centered Gaussian framework built upon a pretrained depth foundation model. Specifically, the framework employs Pixel-Aligned Frustum Feature Aggregation to construct a camera-aware frustum query for each feature pixel, and Frustum-Parameterized Gaussian Construction to decode each query into multiple Gaussians whose positions and sizes are constrained by the predicted pixel depth and corresponding frustum geometry. OccAnyScene sets new state-of-the-art results, achieving 59.92% mIoU on the indoor Occ-ScanNet...
  </details>  
  Keywords: image-to-3d  
- **[ERF-GS: Reconstructing Fast Motion from Disjoint Event-RGB Viewpoints](https://arxiv.org/abs/2608.08531v1)**  
  Authors: Xiaoyang Bai, Zhenyang Li, Weiwei Xu, Edmund Y. Lam, Yifan Peng  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2608.08531v1.pdf) | [![GitHub](https://img.shields.io/github/stars/andrewbxy/ERF-GS?style=social)](https://github.com/andrewbxy/ERF-GS)  
  <details><summary>Abstract</summary>

  Deep learning-driven representations such as neural radiance fields (NeRFs) and 3D Gaussian splatting (3DGS) have revolutionized the field of dynamic 3D scene reconstruction with improved visual precision and scalability. However, the reconstruction of fast-moving objects remains a challenge; existing methods based on conventional frame-based videos often struggle in scenarios such as sports events and animal videography. We propose an event-RGB fusion Gaussian splatting (ERF-GS) framework that integrates event information into both optimization and densification stages of the Gaussian splatting pipeline, taking advantage of novel event sensors with high frame-rate. Unlike many other event-assisted scene reconstruction methods, ERF-GS was developed using realistic simulation settings and realizes event-based learning detached from RGB inputs. This design enables its application beyond straightforward synthetic data into the realm of natural video with complex layout, low frame rates and severe motion blur. Our experiments show that ERF-GS outperforms both the 4DGS baseline and the concurrent E-D3DGS...
  </details>  
  Keywords: 3d scene reconstruction  
- **[PhysX-CoT: Structured Physical Reasoning from a Single Image to Simulation-Ready 3D Assets](https://arxiv.org/abs/2608.08053v1)**  
  Authors: Jie Huang, Xiaohe Li, Jiahao Li, Fangli Mou, Chen Qian, Yuqiang Fang, Junhao Fan, Kaixin Zhang, Zide Fan  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2608.08053v1.pdf)  
  <details><summary>Abstract</summary>

  Simulation-ready 3D assets are central to robotics and embodied AI. Generating them from a single image is usually framed as a vision-language model that emits a serialized asset for a decoder to turn into geometry and physical fields, leaving the image-to-3D reasoning implicit. We argue the limiting factor is this output-centric view: part placement and local shape are entangled in one global-coordinate token stream, and the intermediate physical states are never exposed for supervision, conditioning, or verification. PhysX-CoT instead casts single-image asset generation as an explicit structured physical reasoning process, an ordered and machine-parseable trajectory of part-level states covering decomposition, 2D and 3D grounding, relations, coarse geometry, and surface cues that we separately supervise, use to condition geometry, and treat as reward targets. Geometry is factorized so that 3D boxes carry placement and local codes carry shape, and CoT-aligned GRPO optimizes parse validity, grounding, geometry, placement, and physical consistency. Under...
  </details>  
  Keywords: image-to-3d  
- **[CANIS: Generation-Assisted 3D Canonicalization via an Image-Semantic Bridge](https://arxiv.org/abs/2608.07256v1)**  
  Authors: Kendong Liu, Yuxin Yao, Junhui Hou  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2608.07256v1.pdf) | [![Project](https://img.shields.io/badge/-Project-blue)](https://kenkenzaii.github.io/Canis)  
  <details><summary>Abstract</summary>

  Canonicalizing 3D object orientation is fundamental to 3D understanding and analysis. Existing approaches often rely on geometric cues, although 3D canonicalization ultimately requires a semantically meaningful orientation. To address this gap, we propose CANIS, a category-agnostic, generation-assisted framework that introduces the semantic orientation prior of a frozen image-to-3D generative model into 3D canonicalization, without canonicalization-specific training or category-specific templates. Specifically, CANIS first renders the input object from candidate viewpoints, selects an informative view, and generates a proxy in a canonical orientation. During generation, a sparse structural latent encoded from the input guides the proxy to preserve the geometry of an object. CANIS then uses the selected image as a semantic bridge between the input and the proxy. Image patches identify semantic regions on the proxy, and depth back-projection locates the corresponding regions on the input. The resulting semantic anchors constrain geometric matching, from which we estimate the rigid transformation that...
  </details>  
  Keywords: image-to-3d  
- **[Scenix: Sparse-View 3D Scene Reconstruction via Executable Scene Programs](https://arxiv.org/abs/2608.07012v1)**  
  Authors: Kai Li, Lutao Jiang, Zhenyang Li, Jiayu Dong, Jierui Zhang, Yingda Yin, Runze Zhang, Kai Yan, Xiaoyang Huang, Keyang Luo, Xin Wang, Xiangyu Zhao, Weikai Chen  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2608.07012v1.pdf)  
  <details><summary>Abstract</summary>

  Synthesizing a structured and editable 3D indoor scene from a few uncalibrated RGB views requires more than generating high-quality individual assets: a system must infer the room structure, associate objects across incomplete observations, and recover a globally consistent spatial configuration. Previous methods mainly focus on 3D scene generation with text input or require continuous visual inputs with additional priors, \ e.g., human-annotated masks or accurate 3D layouts, which makes these methods labor demanding and hard to apply in general cases. We present \textsc{Scenix}, a sparse-view 3D scene reconstruction framework via executable scene programs, a structured representation that can be directly instantiated into editable 3D scenes. Given sparse views, \textsc{Scenix} predicts executable scene programs through perception-grounded asset instantiation and closed-loop spatial refinement. % We present \method, a framework that predicts an executable scene representation from sparse views and realizes it through perception-grounded asset instantiation and closed-loop spatial refinement. To support this...
  </details>  
  Keywords: 3d scene reconstruction  
- **[To See a World in a Living Context: Unified Indoor-Outdoor Urban World Generation](https://arxiv.org/abs/2608.05879v2)**  
  Authors: Xiaobin Huang, Zilong Huang, Yang Luo, Hongchao Fan, Yiping Chen, Ting Han  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2608.05879v2.pdf) | [![Project](https://img.shields.io/badge/-Project-blue)](https://huangxb326.github.io/HoloWorld)  
  <details><summary>Abstract</summary>

  Text-driven 3D generation has advanced rapidly in creating large-scale outdoor environments and detailed indoor scenes, but these domains are usually synthesized independently, lacking the correspondence required for a coherent urban world. We present HoloWorld, a unified indoor-outdoor urban world generation framework built on a continuously updated cross-scale world context. Initializing from a user description, HoloWorld progressively represents and updates the diverse world information, from city-scale planning to individual buildings, allowing generated interiors to maintain explicit correspondence with their associated exterior buildings. Conditioned on the evolving context and previously generated neighboring blocks, HoloWorld autoregressively generates urban exteriors with consistent spatial organization and visual identity across blocks. The generated exterior representations are further grounded in 3D building instances and footprints, enabling building-specific indoor generation with geometry-constrained layouts and inherited appearance characteristics. To our knowledge, HoloWorld is the first framework to unify indoor and outdoor generation within a coherent 3D urban world. Extensive...
  </details>  
- **[CoordRefer: Coordinate-Aware 3D Visual Grounding from Multiview Images](https://arxiv.org/abs/2608.05569v1)**  
  Authors: Haijie Li, Jiaxin Zhang, Dave Zhenyu Chen, Youyu Chen, Yanmin Wu, Jian Zhang  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2608.05569v1.pdf)  
  <details><summary>Abstract</summary>

  Multiview image-based 3D visual grounding predicts a coordinate frame to define a coordinate system and then regresses a 3D bounding box for localization. However, existing methods jointly optimize coordinate frame selection and box regression, leading to coordinate-relative box ambiguity and degraded grounding performance. This ambiguity arises because the same box admits different numerical representations across coordinate frames, creating multiple optimization targets and yielding invalid compromise predictions. To tackle this challenge, we propose CoordRefer, a coordinate-aware framework that decouples coordinate frame selection from coordinate-conditioned grounding. CoordRefer first selects a reference frame to define the coordinate system and then conditions 3D box prediction on the coordinate system. We perform coordinate-aware supervised fine-tuning to establish coordinate frame selection and coordinate-conditioned box regression, followed by Group Relative Policy Optimization with 3D IoU-based rewards to align both stages with downstream grounding quality. On ScanRefer with Qwen3-VL-2B, CoordRefer achieves gains of 11% in Acc@0.25 and 7%...
  </details>  
- **[CDSeg: A Renderable Gaussian Carrier for Image-to-3D Label Transfer](https://arxiv.org/abs/2608.05482v1)**  
  Authors: Wentao Sun, Yiping Chen, Zhengsen Xu, Jonathan Li, John S. Zelek  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2608.05482v1.pdf)  
  <details><summary>Abstract</summary>

  Modern image models provide strong cues about \emph{what} should be segmented in each view, but their masks do not by themselves determine \emph{where} those labels should persist in 3D. We present Cross-Domain Segmentation via Gaussian Splatting (CDSeg), a label-transfer interface that requires no task-specific 3D segmentation training and uses Gaussian primitives as a renderable label carrier. An external mask source supplies the labels, while renderer-derived visibility determines which 3D primitives receive them. The carrier is instantiated either by completing each input point into one Gaussian, preserving its index, or by reusing the native primitives of an optimized Gaussian scene. CDSeg records pixel--primitive associations during rendering and fuses multi-view masks through voting and a local filter. The resulting labels can be returned to the original points, retained on the native Gaussian scene, or rendered into other views. CDSeg covers promptable, automatic instance, semantic, and LiDAR settings and processes scenes with millions...
  </details>  
  Keywords: image-to-3d  
- **[OutLangSplat: 3D Language Gaussian Splatting for UAV Outdoor Scenes](https://arxiv.org/abs/2608.04560v1)**  
  Authors: Xia Yan, He Wu, Yanghui Xu, Zizhao Wu, Jiazhou Chen  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2608.04560v1.pdf)  
  <details><summary>Abstract</summary>

  3D Language Gaussian Splatting embeds open-vocabulary language features into 3D Gaussian Splatting, providing an efficient explicit representation for text-driven 3D scene understanding. However, existing methods are limited to indoor or small-scale scenes, and tend to fail in Unmanned Aerial Vehicle (UAV) outdoor scenes, where severe occlusions and long distance viewpoints often lead to incorrect semantic activations and missing target responses. In this paper, we present OutLangSplat which adapts language Gaussian representations to UAV outdoor scenes by improving feature representation and aggregation reliability. For the feature representation, a 2D-3D dual-branch representation with region-based alignment and fusion is designed to improve spatial consistency, reducing incomplete target responses and background misactivations. For the feature aggregation, we introduce a training-free contribution and consistency-aware Gaussian feature aggregation strategy that leverages pixel contribution reliability and cross-view semantic consistency to suppress unreliable responses from noisy viewpoints. A new dataset is provided by manually annotating various objects on...
  </details>  
- **[EditFlow3D: Automated Local Editing of 3D Assets with Trajectory Preservation](https://arxiv.org/abs/2608.03179v1)**  
  Authors: Rui Nie, Chuang Wang, Haitao Zhou, Jiahe Song, Buyu Li, Sheng Wang, Qian Yu  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2608.03179v1.pdf)  
  <details><summary>Abstract</summary>

  Controllable local editing of 3D assets requires precise target localization and appropriate visual guidance. However, existing methods lack a simple yet accurate way to obtain 3D masks and struggle to achieve the desired edit while faithfully preserving the structure and appearance of non-target regions. To address these challenges, we present EditFlow3D, a training-free framework for local 3D editing. Given a source asset and an edit instruction, a VLM-driven workflow interprets the editing intent and automatically constructs a visual guidance image and a refined 3D editing mask, enabling localized editing in the native representation space of a pretrained 3D generative model. Specifically, mask-guided differential flow focuses the edit on the target region, while step-wise trajectory preservation maintains consistency between non-target regions and the source asset without directly replacing intermediate features. Since the existing Edit3D-Bench covers only a limited range of local editing categories, we further introduce EditFlow-Bench as a complementary benchmark...
  </details>  
- **[Hunyuan3D-Buffalo 1.0: A Unified Multimodal Model for Scalable 3D Generation, Understanding, and Editing](https://arxiv.org/abs/2608.02711v3)**  
  Authors: Junliang Ye, Kenkun Liu, Guocun Wang, Yang Li, Yansong Qu, Chunshi Wang, Jingwei Xu, Yunhan Yang, Zibo Zhao, Jiachen Xu, Jiaao Yu, Lifu Wang, Zhihao Liang, Xin Huang, Zhuo Chen, Chunchao Guo  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2608.02711v3.pdf) | [![Project](https://img.shields.io/badge/-Project-blue)](https://tencent-hunyuan.github.io/Hunyuan3D-Buffalo1.0)  
  <details><summary>Abstract</summary>

  Recent advances in image generation have demonstrated the potential of unified multimodal models that integrate understanding, generation, and editing. However, unified 3D modeling remains constrained by scarce multimodal data, particularly the lack of large-scale and geometrically consistent editing data. To address this limitation, we propose Hunyuan3D-Buffalo 1.0, a unified framework supporting 3D understanding, text-to-3D generation, instruction-guided 3D editing, and text-grounded part generation within a single architecture. To enable scalable training, we construct an 87M-scale 3D multimodal corpus, comprising 25M understanding samples, 50M text-to-3D pairs, and 12M editing pairs generated using Nano3D-v2. Architecturally, the framework combines Hunyuan3D-VLM for semantic, structural, and spatial understanding with Hunyuan3D DiT for high-fidelity 3D synthesis. The VLM provides multimodal semantic conditions for generation, while editing and part generation additionally condition the diffusion process on the source object representation to preserve its overall structure and unedited regions. Extensive experiments show that Hunyuan3D-Buffalo 1.0 achieves state-of-the-art or leading...
  </details>  
  Keywords: text-to-3d  
- **[Beyond Global Latents: Chunk-Based Sparse Grid VAE for Scalable 3D Modeling](https://arxiv.org/abs/2608.02016v1)**  
  Authors: Kaiyi Zhang, Zhihao Liang, Haolin Liu, Qingxiang Lin, Zeqiang Lai, Yunfei Zhao, Bowen Zhang, Xianghui Yang, Zibo Zhao, Chunchao Guo, Long Quan  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2608.02016v1.pdf)  
  <details><summary>Abstract</summary>

  Sparse voxel grids preserve the spatial structure needed for detailed 3D reconstruction, but their memory still grows rapidly with resolution as active surface cells increase. We introduce ChunkVAE, a sparse grid variational autoencoder organized around local chunks rather than a global latent volume. Local learned operators permit independently chosen encoder and decoder partitions and allow inference chunk sizes to differ from training. Two complementary data operators make this flexibility practical: Balanced Binary Object Partitioning distributes active cells while limiting replicated overlap, while S-Curve weighted stitching attenuates unreliable boundary features when assembling a global latent or reconstruction. Across three object benchmarks, ChunkVAE is competitive with or better than strong baselines from $512^3$ to $1536^3$; smaller chunks lower peak allocated memory and shorten per-chunk compute, enabling faster parallel inference. Stable stitched latents and improved image to 3D metrics indicate that local compression can scale geometry while retaining the global interface required downstream.
  </details>  
- **[ASTRA: Asynchronous Spatio-Temporal Reconstruction via Trajectory Alignment](https://arxiv.org/abs/2608.02006v2)**  
  Authors: Junyu Zhu, Hao Zhu, Xinzhuo Zhang, Hongdong Li, Zhan Ma, Xun Cao  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2608.02006v2.pdf)  
  <details><summary>Abstract</summary>

  Dynamic 3D scene reconstruction has achieved remarkable success under the assumption of strictly synchronized multi-camera inputs. However, in real-world scenarios, temporal asynchrony among capturing devices remains a critical challenge, leading to severe motion blur and geometric artifacts. Existing asynchronous reconstruction methods typically estimate temporal offsets through photometric supervision, but appearance matching provides weak temporal cues under large offsets and complex motions. We attribute this limitation to two major bottlenecks: texture-induced collapse, where low-texture regions provide nearly vanishing alignment signals, and deformation-induced coupling, where temporal errors are absorbed into distorted geometry or motion rather than being explicitly corrected. To address these issues, we propose ASTRA (Asynchronous Spatio-Temporal Reconstruction via Trajectory Alignment), a framework that introduces 2D motion trajectories as explicit, texture-agnostic supervision for asynchronous dynamic reconstruction. Instead of synchronizing cameras solely through rendered color residuals, ASTRA jointly optimizes temporal offsets and dynamic 3D representations by aligning the projected motion of reconstructed...
  </details>  
  Keywords: 3d scene reconstruction  
- **[PartMat: Material-Aware 3D Part Decomposition with a Single Global Latent](https://arxiv.org/abs/2608.01825v1)**  
  Authors: Guangming Fu, Jin Song, Yiyun Fei, Guoqiu Li, Ruigao Yang, Jianan Jiang  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2608.01825v1.pdf)  
  <details><summary>Abstract</summary>

  Part-level 3D generation has recently attracted increasing attention for producing structured and editable 3D assets. However, existing methods typically decompose objects according to functional semantics rather than the editable material boundaries (e.g., fabric, wood, metal) required in practical 3D applications such as interior design. Additionally, current methods often generate parts independently, causing computational costs to scale linearly with the part count. To address these limitations, we present PartMat, an efficient material-aware 3D part decomposition pipeline that represents multi-part geometry with a single global latent. Given a reference image and a single whole-object geometry, PartMat decomposes the object into parts that follow material boundaries. First, we propose PartVAE to learn such a unified representation and decode all material parts in a single forward pass, thereby decoupling inference cost from the number of parts. Second, with this representation, a diffusion model is trained for part generation and refined via reinforcement learning for...
  </details>  
- **[ETHead: Generating Expressive 3D Facial Animation and Head Movement from Speech](https://arxiv.org/abs/2608.01605v1)**  
  Authors: Jiu-Cheng Xie, Jiwang Zheng, Yongkang Xia, Jian Xiong, Chi-Man Pun, Hao Gao, Feng Xu  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2608.01605v1.pdf) | [![Project](https://img.shields.io/badge/-Project-blue)](https://verdure-oss.github.io/ETHead.github.io)  
  <details><summary>Abstract</summary>

  Generating expressive 3D talking heads solely from speech remains a significant challenge due to the scarcity of high-fidelity 3D data, which limits the modeling of complex emotional motion patterns. In this paper, we introduce \textbf{E}xpressive \textbf{T}alking \textbf{Head} (ETHead), a method for generating 3D facial and head motions that vividly align with the emotional content of input speech. To overcome the data limitations, we design a self-distillation framework that leverages large-scale 2D talking videos to pre-train a specialized speech encoder. By incorporating a novel emotion-modulated probabilistic masking mechanism, this framework aligns speech representations with expressive visual dynamics, allowing the encoder to extract features highly correlated with facial and head motions directly from audio. These features are then leveraged to guide 3D generation, enriching input cues and providing explicit supervision through a joint speech-motion latent space. Extensive experiments demonstrate that ETHead substantially outperforms state-of-the-art methods. Furthermore, our motion-aligned speech encoder can serve...
  </details>  
- **[Assistant Placement Aria: A Benchmark for Egocentric Placement Assistance](https://arxiv.org/abs/2608.00652v1)**  
  Authors: Amir Belder, Gonçalo Dias Pais, Refael Vivanti, Omri Carmi, Daniel DeTone, Oren Shrout, Ido Gattegno, Ayellet Tal  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2608.00652v1.pdf)  
  <details><summary>Abstract</summary>

  Human assistance in robotics spans around several tasks such as navigation, object manipulation, and placement, where a key challenge is selecting target destinations that align with human intentions or preferences. We focus on this challenge in the context of Virtual Placement (VP), the task of identifying all plausible target locations given scene context and human-centric constraints. This differs from traditional placement tasks that typically focus on a single, predefined target location. The VP problem is complex, as it requires both global and local reasoning about the scene's geometry, semantics, and plausibility. To address this gap, we introduce {\bf Assistant Placement Aria}, the first benchmark to explore diverse aspects of VP, including global, local, and human-centric constraints. It contains both synthetic and real indoor scenes annotated for three tasks: (i)~2D Panel Placement, (ii)~Sitting Suggestion, and (iii)~TV Placement. Each scene includes 2D images, a 3D point cloud, and a textual description of...
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
  Keywords: rigging, 3d asset generation  
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
- **[E3VS-Bench: A Benchmark for Viewpoint-Dependent Active Perception in 3D Gaussian Splatting Scenes](https://arxiv.org/abs/2604.17969v3)**  
  Authors: Koya Sakamoto, Taiki Miyanishi, Daichi Azuma, Shuhei Kurita, Shu Morikuni, Naoya Chiba, Motoaki Kawanabe, Yusuke Iwasawa, Yutaka Matsuo  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2604.17969v3.pdf)  
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
- **[LAGS: Low-Altitude Gaussian Splatting with Groupwise Heterogeneous Graph Learning](https://arxiv.org/abs/2604.16910v3)**  
  Authors: Yikun Wang, Yujie Wan, Wei Zuo, Shuai Wang, Yik-Chung Wu, Chengzhong Xu, Huseyin Arslan  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2604.16910v3.pdf)  
  <details><summary>Abstract</summary>

  Low-altitude Gaussian splatting (LAGS) facilitates 3D scene reconstruction by aggregating aerial images from distributed drones. However, as LAGS prioritizes maximizing reconstruction quality over communication throughput, existing low-altitude resource allocation schemes become inefficient. This inefficiency stems from their failure to account for image diversity introduced by varying viewpoints. To fill this gap, we propose a groupwise heterogeneous graph neural network (GW-HGNN) for LAGS resource allocation. GW-HGNN explicitly models the non-uniform contribution of different image groups to the reconstruction process, thus automatically balancing data fidelity and transmission cost. The key insight of GW-HGNN is to transform LAGS losses and communication constraints into graph learning costs for dual-level message passing. Experiments on real-world LAGS datasets demonstrate that GW-HGNN significantly outperforms state-of-the-art benchmarks across key rendering metrics, including PSNR, SSIM, and LPIPS. Furthermore, GW-HGNN reduces computational latency by approximately 100x compared to the widely-used MOSEK solver, achieving millisecond-level inference suitable for real-time deployment.
  </details>  
  Keywords: 3d scene reconstruction  
- **[Repurposing 3D Generative Model for Autoregressive Layout Generation](https://arxiv.org/abs/2604.16299v2)**  
  Authors: Haoran Feng, Yifan Niu, Zehuan Huang, Yang-Tian Sun, Yuxin Peng, Lu Sheng  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2604.16299v2.pdf) | [![GitHub](https://img.shields.io/github/stars/fenghora/LaviGen?style=social)](https://github.com/fenghora/LaviGen)  
  <details><summary>Abstract</summary>

  We introduce LaviGen, a framework that repurposes 3D generative models for 3D layout generation. Unlike previous methods that infer object layouts from textual descriptions, LaviGen operates directly in the native 3D space, formulating layout generation as an autoregressive process that explicitly models geometric relations and physical constraints among objects, producing coherent and physically plausible 3D scenes. To further enhance this process, we propose an adapted 3D diffusion model that integrates scene, object, and instruction information and employs a dual-guidance self-rollout distillation mechanism to improve efficiency and spatial accuracy. Extensive experiments on the LayoutVLM benchmark show LaviGen achieves superior 3D layout generation performance, with 19% higher physical plausibility than the state of the art and 65% faster computation. Our code is publicly available at https://github.com/fenghora/LaviGen.
  </details>  
- **[Beyond Prompts: Unconditional 3D Inversion for Out-of-Distribution Shapes](https://arxiv.org/abs/2604.14914v2)**  
  Authors: Victoria Yue Chen, Emery Pierson, Léopold Maillard, Maks Ovsjanikov  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2604.14914v2.pdf) | [![Project](https://img.shields.io/badge/-Project-blue)](https://daidedou.sorpi.fr/publication/beyondprompts)  
  <details><summary>Abstract</summary>

  Text-driven inversion of generative models is a core paradigm for manipulating 2D or 3D content, unlocking numerous applications such as text-based editing, style transfer, or inverse problems. However, it relies on the assumption that generative models remain sensitive to natural language prompts. We demonstrate that for state-of-the-art native text-to-3D generative models, this assumption often collapses. We identify a critical failure mode where generation trajectories are drawn into latent "sink traps": regions where the model becomes insensitive to prompt modifications. In these regimes, changes to the input text fail to alter internal representations in a way that alters the output geometry. Crucially, we observe that this is not a limitation of the model's \textit{geometric} expressivity; the same generative models possess the ability to produce a vast diversity of shapes but, as we demonstrate, become insensitive to out-of-distribution \textit{text} guidance. We investigate this behavior by analyzing the sampling trajectories of the generative...
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
- **[Physically Grounded 3D Generative Reconstruction under Hand Occlusion using Proprioception and Multi-Contact Touch](https://arxiv.org/abs/2604.09100v2)**  
  Authors: Gabriele Mario Caddeo, Pasquale Marra, Lorenzo Natale  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2604.09100v2.pdf)  
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
  Keywords: rigging, 3d asset generation  
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
  Keywords: novel view synthesis, 3d scene reconstruction  
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
  Keywords: novel view synthesis, 3d scene reconstruction  
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
- **[Control-DINO: Feature Space Conditioning for Controllable Image-to-Video Diffusion](https://arxiv.org/abs/2604.01761v2)**  
  Authors: Edoardo A. Dominici, Thomas Deixelberger, Konstantinos Vardis, Markus Steinberger  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2604.01761v2.pdf)  
  <details><summary>Abstract</summary>

  Video diffusion models have recently been applied with success to problems in content generation, novel view synthesis, and, more broadly, world simulation. Many applications in generation and transfer rely on conditioning these models, typically through perceptual, geometric, or simple semantic signals, fundamentally using them as generative renderers. At the same time, high-dimensional features obtained from large-scale self-supervised learning on images or point clouds are increasingly used as a general-purpose interface for vision models. The connection between the two has been explored for subject specific editing, aligning and training video diffusion models, but not in the role of a dense conditioning signal for pretrained video diffusion models. Features obtained through self-supervised learning like DINOv3, contain a lot of entangled information about style, lighting and semantics of the scene. This makes them great at reconstruction tasks but limits their generative capabilities. In this paper, we show how we can use the features...
  </details>  
  Keywords: novel view synthesis  
- **[Satellite-Free Training for Drone-View Geo-Localization](https://arxiv.org/abs/2604.01581v3)**  
  Authors: Tao Liu, Yingzhi Zhang, Kan Ren, Xiaoqi Zhao  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2604.01581v3.pdf)  
  <details><summary>Abstract</summary>

  Drone-view geo-localization (DVGL) aims to determine the location of drones in GPS-denied environments by retrieving the corresponding geotagged satellite tile from a reference gallery given UAV observations of a location. In many existing formulations, these observations are represented by a single oblique UAV image. In contrast, our satellite-free setting is designed for multi-view UAV sequences, which are used to construct a geometry-normalized UAV-side location representation before cross-view retrieval. Existing approaches rely on satellite imagery during training, either through paired supervision or unsupervised alignment, which limits practical deployment when satellite data are unavailable or restricted. In this paper, we propose a satellite-free training (SFT) framework that converts drone imagery into cross-view compatible representations through three main stages: drone-side 3D scene reconstruction, geometry-based pseudo-orthophoto generation, and satellite-free feature aggregation for retrieval. Specifically, we first reconstruct dense 3D scenes from multi-view drone images using 3D Gaussian splatting and project the reconstructed geometry into...
  </details>  
  Keywords: 3d scene reconstruction  
- **[RecGen3D: Reconstruction-Guided 3D Generation in a Shared Canonical Space](https://arxiv.org/abs/2604.01479v3)**  
  Authors: Zhisheng Huang, Jiahao Chen, Cheng Lin, Chenyu Hu, Hanzhuo Huang, Zhengming Yu, Mengfei Li, Yuheng Liu, Zekai Gu, Zibo Zhao, Yuan Liu, Xin Li, Wenping Wang  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2604.01479v3.pdf)  
  <details><summary>Abstract</summary>

  Sparse-view 3D modeling represents a fundamental tension between reconstruction fidelity and generative plausibility. While feed-forward reconstruction excels in efficiency and input alignment, it often lacks the global priors needed for structural completeness. Conversely, diffusion-based generation provides rich geometric details but struggles with multi-view consistency. We present RecGen3D, a framework that combines these two paradigms into a cooperative system. To overcome inherent conflicts in coordinate spaces, 3D representations, and training objectives, we align both models within a shared canonical space. We employ decoupled cooperative learning, which maintains stable training while enabling seamless collaboration during inference. Specifically, the reconstruction module is adapted to provide canonical geometric anchors, while the diffusion generator leverages latent-augmented conditioning to refine and complete the geometric structure. Experimental results demonstrate that RecGen3D achieves superior fidelity and robustness, outperforming existing methods in creating complete and consistent 3D models from sparse observations.
  </details>  
- **[DirectFisheye-GS: Enabling Native Fisheye Input in Gaussian Splatting with Cross-View Joint Optimization](https://arxiv.org/abs/2604.00648v2)**  
  Authors: Zhengxian Yang, Fei Xie, Xutao Xue, Rui Zhang, Taicheng Huang, Yang Liu, Mengqi Ji, Tao Yu  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2604.00648v2.pdf) | [![Project](https://img.shields.io/badge/-Project-blue)](https://yzxqh.github.io/DirectFisheye-GS)  
  <details><summary>Abstract</summary>

  3D Gaussian Splatting (3DGS) has enabled efficient 3D scene reconstruction from everyday images with real-time, high-fidelity rendering, greatly advancing VR/AR applications. Fisheye cameras, with their wider field of view (FOV), promise high-quality reconstructions from fewer inputs and have recently attracted much attention. However, since 3DGS relies on rasterization, most subsequent works involving fisheye camera inputs first undistort images before training, which introduces two problems: 1) Black borders at image edges cause information loss and negate the fisheye's large FOV advantage; 2) Undistortion's stretch-and-interpolate resampling spreads each pixel's value over a larger area, diluting detail density -- causes 3DGS overfitting these low-frequency zones, producing blur and floating artifacts. In this work, we integrate fisheye camera model into the original 3DGS framework, enabling native fisheye image input for training without preprocessing. Despite correct modeling, we observed that the reconstructed scenes still exhibit floaters at image edges: Distortion increases toward the periphery, and...
  </details>  
  Keywords: 3d scene reconstruction  



## Categorized Papers

### Cross-Modal Generation

*Showing the latest 50 out of 92 papers*

- **[Kirin: Animal Motion Generation from In-the-Wild Video](https://arxiv.org/abs/2609.01823v1)**  
  Authors: Brian Nlong Zhao, Zhuoyang Pan, James M. Rehg, Jiajun Wu, Shangzhe Wu  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2609.01823v1.pdf) | [![Project](https://img.shields.io/badge/-Project-blue)](https://kirin-ani.github.io)  
  <details><summary>Abstract</summary>

  Understanding animal motion is fundamental to modeling animal behavior and biomechanics, yet progress in this area lags far behind human motion research due to the scarcity of high-quality motion data. While human motion can be captured in controlled environments, it is impractical for most animal species, resulting in small, domain-limited datasets that restrict downstream applications such as animation. To address this challenge, we introduce Kirin, a framework that reconstructs motion from video, learns motion priors at scale, and generates realistic motion that can be directly applied to animated assets. Using large collections of in-the-wild animal videos, we reconstruct 3D motion sequences and pair them with captions to create AiM3D, the first large-scale dataset offering aligned video-text-motion tuples for quadruped animals. Building on this dataset, we develop a visual-guided motion generation model that conditions on both text and image to guide the generation of realistic motion across diverse animal species. Finally,...
  </details>  
  Keywords: image-to-3d  
- **[Cyc3D: Evaluating Cyclic Structural Stability and Asset Usability in Image-to-3D Generation](https://arxiv.org/abs/2608.28080v1)**  
  Authors: Liwen Zhang  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2608.28080v1.pdf)  
  <details><summary>Abstract</summary>

  Image-conditioned 3D generation has advanced rapidly, yet existing evaluation protocols largely judge rendered-view plausibility and semantic alignment, overlooking whether a generator forms a stable 3D interpretation and produces assets usable in graphics pipelines. We introduce Cyc3D, a multidimensional benchmark that evaluates image-to-3D generation along two complementary axes: Cross-View Object Consistency and Representation Quality. At the asset level, Cyc3D measures whether object identity remains semantically coherent across rendered viewpoints. At the model level, we propose View-Cycle Structural Consistency, a closed-loop render-regenerate-align protocol that repeatedly re-observes a generated asset from novel views and quantifies geometric, perceptual, and semantic drift across generations. To assess native asset usability beyond rendered appearance, Cyc3D further evaluates geometric structure, reference-image fidelity, mesh discretization and efficiency, and UV parameterization quality. Together, these diagnostics expose failures obscured by a single perceptual score and provide interpretable evidence of both model instability and representation defects. Experiments on five representative image-to-3D systems...
  </details>  
  Keywords: image-to-3d  
- **[Luce: Relightable Gaussians for 3D Asset Generation](https://arxiv.org/abs/2608.23943v1)**  
  Authors: Mayank Singh, Michele Stoppa, Alvise Memo, Rui Yu, Harsha Kalli, Srimanth Gunturi, Muhammad Ahmed Riaz, Behrooz Shahsavari, Waleed Abdulla, David E. Jacobs  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2608.23943v1.pdf)  
  <details><summary>Abstract</summary>

  High-fidelity image-to-3D generation requires a 3D representation that captures both geometry and appearance. To support relighting and integration into standard rendering pipelines, the representation should include physically based rendering (PBR) modalities such as albedo, metallic-roughness, and surface normals. We propose Luce, a 3D representation that unifies geometry and PBR materials within a voxelized multimodal Gaussian cloud, using dedicated Gaussian primitives for each modality. A variational autoencoder compresses this representation into a unified material-aware latent space. A rectified-flow transformer generates this latent from a single image, conditioned on multi-layer features from a pretrained image encoder that preserve both semantic context and fine spatial detail. The latent then decodes into relightable PBR Gaussians and an optional textured mesh with a tangent-space normal map. On Toys4K, Luce achieves state-of-the-art single-image-to-3D generation, improving FID by 28% over the strongest baseline. We further introduce a benchmark of AI-generated images, on which Luce improves the CLIP...
  </details>  
  Keywords: pbr materials, image-to-3d, 3d asset generation  
- **[Block3D: Efficient Text-to-3D Generation via Block-Wise Diffusion](https://arxiv.org/abs/2608.19567v4)**  
  Authors: Bowen Cui, Weijie Wang, Zeyu Zhang, Yefei He, Mingda Lin, Haoyu Zhao, Yuanyu He, Donny Y. Chen, Feng Chen, Bohan Zhuang  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2608.19567v4.pdf)  
  <details><summary>Abstract</summary>

  While text-to-3D generation has advanced rapidly, achieving high geometric fidelity at low inference cost remains challenging. Existing text-to-3D methods either decode discrete shape tokens autoregressively or iteratively refine global 3D representations with diffusion or flow models. However, autoregressive decoding is sequential and cannot revise errors, whereas diffusion and flow-matching models repeatedly process the full representation, making high-quality generation increasingly expensive. In this paper, we propose Block3D, a block-wise diffusion framework that partitions the discrete shape-token sequence into contiguous blocks, generates the blocks autoregressively, and jointly denoises all tokens within the current block. To alleviate error accumulation, we introduce confidence-guided intra-block correction, which revises low-confidence tokens before each block is finalized. On a held-out set from TRELLIS-500K, Block3D reduces mean end-to-end generation time from 25.71 seconds to 4.99 seconds, achieving a $5.15\times$ speedup over the fine-tuned autoregressive baseline without sacrificing geometric fidelity.
  </details>  
  Keywords: text-to-3d  
- **[SemanticSlider3D: Training-Free Continuous Semantic Editing for 3D Objects](https://arxiv.org/abs/2608.18560v2)**  
  Authors: Ru Wang, Rahul Jain, Koichiro Niinuma, Aakar Gupta  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2608.18560v2.pdf)  
  <details><summary>Abstract</summary>

  Fine-grained control over continuous semantic attributes of 3D objects is essential for 3D content creation, but is not well supported by conventional 3D modeling workflows or prompt-based interaction with existing generative AI tools. While slider-based methods have proven effective for fine-grained semantic control in 2D image generation, no equivalent approach exists for 3D. Extending these 2D methods to 3D is non-trivial due to challenges unique to 3D, including geometric integrity and cross-view coherence. We present SemanticSlider3D, a technique for continuous semantic attribute editing of 3D objects that requires no per-attribute training. Given a user-specified attribute, our pipeline constructs a semantic editing direction in the latent space of a state-of-the-art 3D generation model, presenting a diverse and coherent spectrum of 3D variations. A technical validation on a dataset of 50 3D object-attribute pairs shows our method was preferred by all five human assessors across variation range, consistency, 3D object quality, and...
  </details>  
  Keywords: image-to-3d  
- **[PixSDS: Why Latent SDS Makes Noisy Pixels](https://arxiv.org/abs/2608.12997v1)**  
  Authors: Vsevolod Skorokhodov  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2608.12997v1.pdf) | [![Project](https://img.shields.io/badge/-Project-blue)](https://sevashasla.github.io/pixsds-webpage)  
  <details><summary>Abstract</summary>

  Score Distillation Sampling (SDS) enables text-to-3D generation by optimizing rendered images with a pretrained diffusion prior, but latent SDS often produces structured color artifacts and high-frequency texture noise. We identify a failure mode of latent SDS caused by VAE-induced pixel drift: the optimized image can move along pixel-space directions that are weakly constrained by the VAE encoder, so its latent representation remains clean and semantically meaningful while the image itself accumulates visible artifacts. We support this diagnosis with controlled 2D SDS experiments, VAE-only optimization, and a simplified analysis showing that encoder-like latent objectives can amplify image-space noise when the inverse mapping to pixels is underconstrained. Motivated by this observation, we propose PixSDS, a lightweight VAE-consistent gradient repair method. PixSDS decodes a latent SDS lookahead step and uses the decoded image as a clean direction for pixel-space optimization, reducing motion in VAE-inconsistent directions without retraining the diffusion model, changing the renderer,...
  </details>  
  Keywords: text-to-3d  
- **[TGRHuman: Text-Guided Realistic 3D Human Generation via Diffusion Renderer](https://arxiv.org/abs/2608.12175v1)**  
  Authors: Muxin Zhang, Chaohui Yu, Yuanwang Yang, Min Wei, Zhuo Su, Kun Li  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2608.12175v1.pdf)  
  <details><summary>Abstract</summary>

  Realistic 3D human generation plays a crucial role in many graphics applications. However, current methods still struggle to generate high-quality human geometry and texture while maintaining 3D consistency and inference efficiency. In this work, we address these limitations by introducing TGRHuman, a novel approach for generating realistic 3D humans from text. Our method decouples geometry and texture generation to alleviate the issues commonly encountered in NeRF-based methods. Instead of relying on slow, implicit score-distillation-based optimization, we directly use explicit multi-view observation generation and optimization for efficient 3D synthesis. For geometry generation, we propose a high-resolution generative module for multi-view normals together with a geometry-carving strategy that preserves view consistency and supports loose clothing. For texture generation, we produce spatially consistent RGB observations from densely sampled surrounding views using a carefully designed texture-prior acquisition strategy and a diffusion renderer, enabling detailed human texture synthesis. Experiments show that our method can generate...
  </details>  
  Keywords: text-to-3d, texture synthesis  
- **[OccAnyScene: Towards Unified Indoor-Outdoor 3D Occupancy Prediction](https://arxiv.org/abs/2608.08696v2)**  
  Authors: Junjie Liu, Wanshui Gan, Zitong Dai, Guiping Cao, Yan Li, Ke Chen, Dongmei Jiang, Xiangyuan Lan, Jianguo Zhang  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2608.08696v2.pdf)  
  <details><summary>Abstract</summary>

  3D occupancy prediction is fundamental to scene understanding, yet existing 3D semantic occupancy methods are typically specialized to fixed scene types and occupancy protocols. We introduce Cross-Scene 3D Semantic Occupancy Prediction, a new task setting which requires a single model to handle heterogeneous indoor and outdoor scenes with varying cameras, spatial ranges, voxel specifications, and semantic taxonomies. This setting poses a fundamental challenge: achieving metric-consistent yet scene-adaptive image-to-3D lifting across varying camera configurations and scene scales. To address this challenge, we propose OccAnyScene, a pixel-frustum-centered Gaussian framework built upon a pretrained depth foundation model. Specifically, the framework employs Pixel-Aligned Frustum Feature Aggregation to construct a camera-aware frustum query for each feature pixel, and Frustum-Parameterized Gaussian Construction to decode each query into multiple Gaussians whose positions and sizes are constrained by the predicted pixel depth and corresponding frustum geometry. OccAnyScene sets new state-of-the-art results, achieving 59.92% mIoU on the indoor Occ-ScanNet...
  </details>  
  Keywords: image-to-3d  
- **[PhysX-CoT: Structured Physical Reasoning from a Single Image to Simulation-Ready 3D Assets](https://arxiv.org/abs/2608.08053v1)**  
  Authors: Jie Huang, Xiaohe Li, Jiahao Li, Fangli Mou, Chen Qian, Yuqiang Fang, Junhao Fan, Kaixin Zhang, Zide Fan  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2608.08053v1.pdf)  
  <details><summary>Abstract</summary>

  Simulation-ready 3D assets are central to robotics and embodied AI. Generating them from a single image is usually framed as a vision-language model that emits a serialized asset for a decoder to turn into geometry and physical fields, leaving the image-to-3D reasoning implicit. We argue the limiting factor is this output-centric view: part placement and local shape are entangled in one global-coordinate token stream, and the intermediate physical states are never exposed for supervision, conditioning, or verification. PhysX-CoT instead casts single-image asset generation as an explicit structured physical reasoning process, an ordered and machine-parseable trajectory of part-level states covering decomposition, 2D and 3D grounding, relations, coarse geometry, and surface cues that we separately supervise, use to condition geometry, and treat as reward targets. Geometry is factorized so that 3D boxes carry placement and local codes carry shape, and CoT-aligned GRPO optimizes parse validity, grounding, geometry, placement, and physical consistency. Under...
  </details>  
  Keywords: image-to-3d  
- **[CANIS: Generation-Assisted 3D Canonicalization via an Image-Semantic Bridge](https://arxiv.org/abs/2608.07256v1)**  
  Authors: Kendong Liu, Yuxin Yao, Junhui Hou  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2608.07256v1.pdf) | [![Project](https://img.shields.io/badge/-Project-blue)](https://kenkenzaii.github.io/Canis)  
  <details><summary>Abstract</summary>

  Canonicalizing 3D object orientation is fundamental to 3D understanding and analysis. Existing approaches often rely on geometric cues, although 3D canonicalization ultimately requires a semantically meaningful orientation. To address this gap, we propose CANIS, a category-agnostic, generation-assisted framework that introduces the semantic orientation prior of a frozen image-to-3D generative model into 3D canonicalization, without canonicalization-specific training or category-specific templates. Specifically, CANIS first renders the input object from candidate viewpoints, selects an informative view, and generates a proxy in a canonical orientation. During generation, a sparse structural latent encoded from the input guides the proxy to preserve the geometry of an object. CANIS then uses the selected image as a semantic bridge between the input and the proxy. Image patches identify semantic regions on the proxy, and depth back-projection locates the corresponding regions on the input. The resulting semantic anchors constrain geometric matching, from which we estimate the rigid transformation that...
  </details>  
  Keywords: image-to-3d  

### Dynamic & Articulated Modeling

- **[Nova3D: Code-Native Generation of Programmable 3D Assets](https://arxiv.org/abs/2607.22738v1)**  
  Authors: Nimra Noor, Muhammad Bilal, Abdullah Hussain, Hassan Baig  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2607.22738v1.pdf)  
  <details><summary>Abstract</summary>

  Current 3D generative models mostly produce a final surface: a visually strong but largely opaque mesh. Interactive 3D worlds need more than a surface. They need named parts, an assembly hierarchy, measurable constraints, local edit handles, and joints for articulation. We present Nova3D, a system that generates 3D assets as executable Blender source code; the compiled mesh, a binary glTF (GLB), is treated as the artifact, not the asset. Because the output is a program, semantic handles exist at generation time rather than being recovered afterward by segmentation or rigging. We evaluate on Nova3D-Bench, a frozen, spec-grounded benchmark of 54 items across six domains and three difficulty levels with text and image inputs, against eleven baselines in four families (mesh-native, part-structured, code-native, and CAD) plus a same-LLM ablation. Nova3D produces an executable program and a valid artifact for 54/54 items. Every asset exposes named parts organized in a parent-child assembly...
  </details>  
  Keywords: rigging  
- **[PWM-ArtGen: Part World Model for Articulated Object Generation](https://arxiv.org/abs/2607.02045v1)**  
  Authors: Wentao Zheng, Ancong Wu  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2607.02045v1.pdf)  
  <details><summary>Abstract</summary>

  The key challenge in articulated 3D object generation from a single image is accurately predicting the underlying kinematic structure. Existing methods either infer kinematic parameters directly from a static image that lacks dynamic part-level kinematic relationships, or estimate parameters from visual dynamics generated from a single image, which is prone to accumulated errors of two steps. Moreover, the limited scale and diversity of existing annotated datasets further hinder generalization to complex, real-world objects. To overcome these limitations, we propose to learn the joint distribution of visual dynamics and kinematic parameters. Recognizing that articulated objects can be formulated as dynamic systems, we propose a unified Part World Model called PWM-ArtGen. To leverage unannotated data, this model couples action diffusion and image diffusion with independent diffusion timesteps, which enables visual branch co-training. We further curate a photorealistic dataset of 19.7k part-level image pairs without kinematic annotations, to support co-training. Experiments demonstrate that...
  </details>  
  Keywords: articulated object generation  
- **[Fishbone: From One 3D Asset to a Million Controllable Edits](https://arxiv.org/abs/2605.24805v1)**  
  Authors: Yumeng He, Xiaoying Wang, Peihao Li, Yanjia Huang, Joe Masterjohn, Jiajun Wu, Leonidas Guibas, Yin Yang, Ying Jiang, Chenfanfu Jiang  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2605.24805v1.pdf)  
  <details><summary>Abstract</summary>

  Large-scale controllable 3D assets are critical for computer graphics, embodied AI, robotics, and interactive content creation, yet creating diverse 3D assets remains challenging due to the high cost of manual modeling and rigging. Shape deformation offers a natural way to generate variations from existing meshes, but existing data-driven methods often rely on sparse user inputs, while parametric editing frameworks require manually designed control structures and category-specific configurations. Inspired by natural creatures, where a central spine governs global shape and cross-sectional ribs control local variation, we introduce Fishbone, a unified rib-spine representation for general shapes that supports controllable parametric mesh deformation, reduced-space dynamics, and animation. Given an input mesh, Fishbone computes a geodesic scalar field with an adaptive heat method, extracts iso-contours as cross-sectional ribs, constructs a smooth geometry-aware spine through rib centers, and associates surface vertices with nearby rib and spine structures using Gaussian-weighted skinning. The resulting representation enables real-time...
  </details>  
  Keywords: rigging, controllable 3d generation  
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
  Keywords: rigging, 3d asset generation  
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
  Keywords: rigging, 3d asset generation  
- **[AniGen: Unified $S^3$ Fields for Animatable 3D Asset Generation](https://arxiv.org/abs/2604.08746v2)**  
  Authors: Yi-Hua Huang, Zi-Xin Zou, Yuting He, Chirui Chang, Cheng-Feng Pu, Ziyi Yang, Yuan-Chen Guo, Yan-Pei Cao, Xiaojuan Qi  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2604.08746v2.pdf) | [![Project](https://img.shields.io/badge/-Project-blue)](https://yihua7.github.io/AniGen-web)  
  <details><summary>Abstract</summary>

  Animatable 3D assets, defined as geometry equipped with an articulated skeleton and skinning weights, are fundamental to interactive graphics, embodied agents, and animation production. While recent 3D generative models can synthesize visually plausible shapes from images, the results are typically static. Obtaining usable rigs via post-hoc auto-rigging is brittle and often produces skeletons that are topologically inconsistent with the generated geometry. We present AniGen, a unified framework that directly generates animate-ready 3D assets conditioned on a single image. Our key insight is to represent shape, skeleton, and skinning as mutually consistent $S^3$ Fields (Shape, Skeleton, Skin) defined over a shared spatial domain. To enable the robust learning of these fields, we introduce two technical innovations: (i) a confidence-decaying skeleton field that explicitly handles the geometric ambiguity of bone prediction at Voronoi boundaries, and (ii) a dual skin feature field that decouples skinning weights from specific joint counts, allowing a fixed-architecture...
  </details>  
  Keywords: rigging, 3d asset generation  
- **[Points-to-3D: Structure-Aware 3D Generation with Point Cloud Priors](https://arxiv.org/abs/2603.18782v3)**  
  Authors: Jiatong Xia, Zicheng Duan, Anton van den Hengel, Lingqiao Liu  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2603.18782v3.pdf) | [![Project](https://img.shields.io/badge/-Project-blue)](https://jiatongxia.github.io/points2-3D)  
  <details><summary>Abstract</summary>

  Recent progress in 3D generation has been driven largely by models conditioned on images or text, while readily available 3D priors are still underused. In many real-world scenarios, the visible-region point cloud are easy to obtain from active sensors such as LiDAR or from feed-forward predictors like VGGT, offering explicit geometric constraints that current methods fail to exploit. In this work, we introduce Points-to-3D, a diffusion-based framework that leverages point cloud priors for geometry-controllable 3D asset and scene generation. Built on a latent 3D diffusion model TRELLIS, Points-to-3D first replaces pure-noise sparse structure latent initialization with a point cloud priors tailored input formulation.A structure inpainting network, trained within the TRELLIS framework on task-specific data designed to learn global structural inpainting, is then used for inference with a staged sampling strategy (structural inpainting followed by boundary refinement), completing the global geometry while preserving the visible regions of the input priors. In...
  </details>  
  Keywords: controllable 3d generation  

### Scene & View Synthesis

*Showing the latest 50 out of 106 papers*

- **[Generalization over Memorization: Generalization-Aware Diffusion Adaptation for Single-Image Multi-View Synthesis](https://arxiv.org/abs/2608.29233v1)**  
  Authors: Jie Li, Xingchen Zou, Yuxuan Liang  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2608.29233v1.pdf)  
  <details><summary>Abstract</summary>

  We present the winning solution to the ACM Multimedia 2026 Grand Challenge on Single-Image Guided Multi-Angle Image Synthesis. It ranks first among 293 registered teams; 56 teams obtained at least one scored submission on the public Phase-A leaderboard. With only 40 training scenes, the challenge requires 26 target views from one RGB model and one forward pass per view; it prohibits explicit geometry, external rendering, chained generation, candidate selection, and post-processing. We identify a critical model-selection failure: shared training and validation scenes make memorization appear as transferable view control. We therefore introduce GoM. Short for Generalization over Memorization, the framework combines scene-disjoint validation, exposure-matched selection, and targeted diffusion adaptation. Its synthesis model adapts a 4B rectified-flow DiT using rank-32 LoRA, optimizer restarts, late-checkpoint averaging, and VAE decoder tuning. More than 300 offline experiments and 24 online submissions show that validation design and training-trajectory control can matter as much as architecture...
  </details>  
  Keywords: multi-view synthesis  
- **[Ground-to-Satellite Localization in Unconstrained Image Collections for 3D Scene Reconstruction](https://arxiv.org/abs/2608.29211v1)**  
  Authors: Angel Daruna, Ben Southall, Niluthpol Chowdhury Mithun, Kshitij Minhas, Nicholas Meegan, Qiao Wang, Bogdan Matei, Supun Samarasekera, Rakesh Kumar  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2608.29211v1.pdf)  
  <details><summary>Abstract</summary>

  Ground image localization with respect to satellite imagery is a key enabler for metrically-accurate, geo-localized 3D scene reconstruction from unconstrained image collections. Existing cross-view localization methods have strict requirements such as panoramic imagery or known initial locations, limiting their applicability for in-the-wild reconstruction settings. We propose a robust hierarchical cross-view localization framework that leverages geometric constraints from Structure-from-Motion (SfM) models derived from unconstrained ground image collections. Our method generates coarse-to-fine pose hypotheses through a cross-view matching approach and aggregates noisy predictions across SfM model(s) using Kernel Density Estimation to recover consensus alignments while filtering outliers. Experiments demonstrate reliable localization performance from challenging image collections. Empirically we found satellite-referenced alignment enables accurate metric scale estimation, doppelgänger detection, and merging of disjoint SfM reconstructions, resulting in more complete, geo-localized site models than are possible with SfM alone.
  </details>  
  Keywords: 3d scene reconstruction  
- **[ReconSplat: Generalizable 3D Scene Reconstruction Beyond Observed Views](https://arxiv.org/abs/2608.28895v1)**  
  Authors: Giuseppe Stracquadanio, Kevin Raj, Julia Grabinski, Stefan Roth  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2608.28895v1.pdf)  
  <details><summary>Abstract</summary>

  We introduce ReconSplat, a feed-forward model for 3D scene reconstruction that aims to address the longstanding trade-off between plausible view generation for unobserved regions and geometric consistency, providing both geometrically aligned novel views and sharp depth estimates. Our approach builds on 3D Gaussian splatting (3DGS) as an intermediate differentiable scene representation and integrates it with a multi-view latent diffusion model (MV-LDM) trained to act simultaneously as a refiner and an inpainter for appearance and scene geometry. We enforce geometric consistency by guiding the diffusion process with variational 3D latent features for appearance and geometry, encoded by the feed-forward 3DGS representation and rasterized to 2D latent space. ReconSplat produces both photorealistic novel views and accurate depth maps on real-world benchmarks, RealEstate10K and DL3DV-10K, outperforming existing methods in challenging extrapolation setups. Notably, ReconSplat allows the extrapolation of unseen and challenging viewpoints jointly with coherent and precise scene geometry.
  </details>  
  Keywords: 3d scene reconstruction  
- **[WilLaGS: Latent-Conditional 3D Appearance Fields for Robust Gaussian Splatting In-the-Wild](https://arxiv.org/abs/2608.28240v1)**  
  Authors: Yuhao Bai, Qianqiu Tan, Lilong Chen, Huanhuan Lv, Lijun Chen  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2608.28240v1.pdf)  
  <details><summary>Abstract</summary>

  3D Gaussian Splatting (3DGS) delivers real-time and high-fidelity rendering but remains challenged by unconstrained in-the-wild scenes, where drastic appearance variations and transient objects violate multi-view consistency. Existing methods are fundamentally limited by independent and discrete embeddings that struggle to capture continuous environmental changes or model spatially-varying local illumination. To address these limitations, we propose \textbf{WilLaGS}, a unified framework for robust 3D scene reconstruction and generative appearance synthesis under unconstrained settings. Specifically, we introduce a generative appearance model where a $β$-VAE learns a structured and continuous manifold of global appearance. Conditioned on the latent code, we construct a 3D neural appearance field that generates dynamic Tri-Plane features to encode spatially-varying local illumination effects. Furthermore, to suppress transient artifacts, we present a self-supervised perceptual masking mechanism that leverages a Teacher-Student (EMA) architecture to derive a stable scene consensus, robustly identifying inconsistent regions via perceptual discrepancies. Extensive experiments on multiple datasets demonstrate that...
  </details>  
  Keywords: 3d scene reconstruction  
- **[SceneReGen: Generative Reconstruction of 3D Scenes from a Single Image](https://arxiv.org/abs/2608.23930v1)**  
  Authors: Zefan Tian, Yuteng Ye, Yiheng Zhang, Yuhang Yang, Xueqiang Lv, Shizhou Zhang, Le Liu, Di Xu  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2608.23930v1.pdf)  
  <details><summary>Abstract</summary>

  Single-image 3D scene reconstruction must complete partially observed objects and place them coherently in a shared observation-aligned scene frame. Object-level generative priors offer strong completion ability, but their centered, scale-normalized outputs are typically expressed in an object frame, creating a fundamental representation gap between object generation and scene reconstruction. We introduce SceneReGen, a generative reconstruction framework that reinterprets scene reconstruction as the generation and assembly of complete object assets in a shared observation-aligned scene frame. SceneReGen addresses the generation-reconstruction gap through selective pose factorization: each object's observed orientation is encoded directly in the generated mesh, while translation and scale are estimated from instance-level and global scene evidence. Given a scene image and instance masks, a geometry encoder extracts dense cues; learnable shape queries condition a pretrained DiT-based 3D generator to produce complete meshes in their observed orientations, while position queries fuse object and scene features to assemble them in the...
  </details>  
  Keywords: 3d generator, 3d scene reconstruction  
- **[Seeing the Unseen: Semantic-in-Gaussian for Sparse-View 3D Generalization](https://arxiv.org/abs/2608.22740v1)**  
  Authors: Zeyang Bai, Yunpeng Wang, Yunbiao Wang, Jun Xiao  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2608.22740v1.pdf)  
  <details><summary>Abstract</summary>

  Generalizable 3D Gaussian Splatting (G-3DGS) has emerged as a promising approach for novel view synthesis undersparse-view settings. However, existing frameworks remain restricted by pixel-aligned Gaussian estimation, whichstruggles in partially observed or occluded regions and often leads to incomplete surfaces or structural collapse. Toaddress these challenges, we propose SeeU (Seeing the Unseen), a novel G-3DGS framework. We frame its core design asSemantic-in-Gaussian: semantic-conditioned refinement in Gaussian space. Specifically, we introduce a Cross-viewEntropy-Aware (CEA) module that aggregates multi-view semantic and geometric cues into compact embeddings. Theseembeddings guide the Conditional Gaussian Transformer, which applies residual updates to coarse Gaussians, helpingrecover under-constrained regions of partially observed structures while preserving surface consistency. Comprehensiveexperiments on multiple benchmarks demonstrate that SeeU consistently improves rendering quality and structuralcompleteness while retaining efficient feed-forward inference. Especially under challenging extrapolation settings,SeeU achieves an average improvement of 2.44 dB in PSNR compared to recent SOTA G-3DGS methods.
  </details>  
  Keywords: novel view synthesis  
- **[DiGS-Avatar: Single-Image Animatable 3D Human Reconstruction via UV-Space Diffusion](https://arxiv.org/abs/2608.20759v1)**  
  Authors: Jiakun Li, Li Fang, Hao Zhu, Fei Hu, Long Ye, Yuan Zhang, Jinyao Yan  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2608.20759v1.pdf) | [![GitHub](https://img.shields.io/github/stars/KLMAV-CUC/DiGS-Avatar?style=social)](https://github.com/KLMAV-CUC/DiGS-Avatar)  
  <details><summary>Abstract</summary>

  Single-image 3D human reconstruction often suffers from over-smoothed textures and geometric inconsistencies. While diffusion models improve generative quality, their reliance on multi-view synthesis prior to 3D reconstruction is computationally expensive and prone to view inconsistency. We propose DiGS-Avatar, which reformulates this task as an efficient, diffusion-based UV-latent completion task, ensuring 3D consistency by design. To capture accurate spatial structure, we introduce a teacher-student framework where a multi-view teacher provides geometrically aligned pseudo-ground-truth latents to supervise a single-view diffusion student. Treating this inferred latent as a robust structural skeleton, our method injects high-level semantic features to accurately recover fine textural details without disrupting spatial integrity. The refined representation is then decoded into 3D Gaussian primitives. Extensive experiments demonstrate that DiGS-Avatar achieves state-of-the-art or highly competitive visual fidelity and zero-shot generalization, while reconstructing a fully animatable 3D avatar in just 0.71 seconds. Code is available at https://github.com/KLMAV-CUC/DiGS-Avatar.
  </details>  
  Keywords: multi-view synthesis  
- **[GS$^{2}$CI: Robust Gaussian Splatting For Snapshot Compressive Imaging via Large Vision Model Priors](https://arxiv.org/abs/2608.13502v1)**  
  Authors: Yanming Yang, Chenxi Song, Ping Wang, Xin Yuan, Chi Zhang  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2608.13502v1.pdf)  
  <details><summary>Abstract</summary>

  Snapshot Compressive Imaging (SCI) offers an efficient solution for high-speed video acquisition and, under exposure-time camera--scene relative motion, multi-view scene capture by compressing temporal or spatial information into a single 2D measurement. While recent studies have explored SCI for 3D scene reconstruction, existing methods struggle with significant challenges due to information loss, limited viewpoint diversity, and the computational burden of jointly optimizing 3D representations and camera poses. In this work, we propose a novel framework that reconstructs high-quality 3D scenes from a single SCI measurement by leveraging 3D Gaussian Splatting (3DGS) and the powerful priors of large-scale vision foundation models (VFMs). Our primary reconstruction combines measurement-derived 3D VFM initialization with SCI-aware Gaussian optimization. After coarse-stage convergence, an auxiliary 2D VFM provides pseudo-view supervision at synthesized viewpoints for local appearance refinement. To further address the instability caused by ambiguous SCI supervision during 3DGS optimization, we introduce Opacity-Guided Splitting and Growth Regulation...
  </details>  
  Keywords: 3d scene reconstruction  
- **[Compact Feed-Forward 3D Gaussians via Saliency-Guided Primitive Merging](https://arxiv.org/abs/2608.10712v2)**  
  Authors: Tim-Felix Fassch, Jochen Kall, Cyrill Stachniss  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2608.10712v2.pdf)  
  <details><summary>Abstract</summary>

  3D scene reconstruction, modeling, and rendering are highly relevant for numerous tasks, and 3D Gaussian splatting has become a standard choice in this context. Its feed-forward variants provide fast reconstruction from sparse input views but often produce per-pixel primitives, leading to highly redundant and thus inefficient representations. We present a structure-aware merging pipeline that takes per-pixel primitives from any feed-forward method and consolidates them into a compact, content-adaptive Gaussian set while largely retaining visual quality at just $\frac{1}{20}^\text{th}$ of the Gaussians of a per-pixel method. We group spatially coherent Gaussians of similar appearance into variable-size clusters via adaptive superpixel segmentation guided by a saliency map, which allocates fine segments to textured regions and coarse segments to homogeneous areas. We compress each cluster into a compact latent representation through a learned encoder, then match and consolidate representations across views based on geometric overlap and feature similarity via a learned merger. A...
  </details>  
  Keywords: 3d scene reconstruction  
- **[ERF-GS: Reconstructing Fast Motion from Disjoint Event-RGB Viewpoints](https://arxiv.org/abs/2608.08531v1)**  
  Authors: Xiaoyang Bai, Zhenyang Li, Weiwei Xu, Edmund Y. Lam, Yifan Peng  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2608.08531v1.pdf) | [![GitHub](https://img.shields.io/github/stars/andrewbxy/ERF-GS?style=social)](https://github.com/andrewbxy/ERF-GS)  
  <details><summary>Abstract</summary>

  Deep learning-driven representations such as neural radiance fields (NeRFs) and 3D Gaussian splatting (3DGS) have revolutionized the field of dynamic 3D scene reconstruction with improved visual precision and scalability. However, the reconstruction of fast-moving objects remains a challenge; existing methods based on conventional frame-based videos often struggle in scenarios such as sports events and animal videography. We propose an event-RGB fusion Gaussian splatting (ERF-GS) framework that integrates event information into both optimization and densification stages of the Gaussian splatting pipeline, taking advantage of novel event sensors with high frame-rate. Unlike many other event-assisted scene reconstruction methods, ERF-GS was developed using realistic simulation settings and realizes event-based learning detached from RGB inputs. This design enables its application beyond straightforward synthetic data into the realm of natural video with complex layout, low frame rates and severe motion blur. Our experiments show that ERF-GS outperforms both the 4DGS baseline and the concurrent E-D3DGS...
  </details>  
  Keywords: 3d scene reconstruction  

### Surface & Appearance Modeling

- **[Luce: Relightable Gaussians for 3D Asset Generation](https://arxiv.org/abs/2608.23943v1)**  
  Authors: Mayank Singh, Michele Stoppa, Alvise Memo, Rui Yu, Harsha Kalli, Srimanth Gunturi, Muhammad Ahmed Riaz, Behrooz Shahsavari, Waleed Abdulla, David E. Jacobs  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2608.23943v1.pdf)  
  <details><summary>Abstract</summary>

  High-fidelity image-to-3D generation requires a 3D representation that captures both geometry and appearance. To support relighting and integration into standard rendering pipelines, the representation should include physically based rendering (PBR) modalities such as albedo, metallic-roughness, and surface normals. We propose Luce, a 3D representation that unifies geometry and PBR materials within a voxelized multimodal Gaussian cloud, using dedicated Gaussian primitives for each modality. A variational autoencoder compresses this representation into a unified material-aware latent space. A rectified-flow transformer generates this latent from a single image, conditioned on multi-layer features from a pretrained image encoder that preserve both semantic context and fine spatial detail. The latent then decodes into relightable PBR Gaussians and an optional textured mesh with a tangent-space normal map. On Toys4K, Luce achieves state-of-the-art single-image-to-3D generation, improving FID by 28% over the strongest baseline. We further introduce a benchmark of AI-generated images, on which Luce improves the CLIP...
  </details>  
  Keywords: pbr materials, image-to-3d, 3d asset generation  
- **[TGRHuman: Text-Guided Realistic 3D Human Generation via Diffusion Renderer](https://arxiv.org/abs/2608.12175v1)**  
  Authors: Muxin Zhang, Chaohui Yu, Yuanwang Yang, Min Wei, Zhuo Su, Kun Li  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2608.12175v1.pdf)  
  <details><summary>Abstract</summary>

  Realistic 3D human generation plays a crucial role in many graphics applications. However, current methods still struggle to generate high-quality human geometry and texture while maintaining 3D consistency and inference efficiency. In this work, we address these limitations by introducing TGRHuman, a novel approach for generating realistic 3D humans from text. Our method decouples geometry and texture generation to alleviate the issues commonly encountered in NeRF-based methods. Instead of relying on slow, implicit score-distillation-based optimization, we directly use explicit multi-view observation generation and optimization for efficient 3D synthesis. For geometry generation, we propose a high-resolution generative module for multi-view normals together with a geometry-carving strategy that preserves view consistency and supports loose clothing. For texture generation, we produce spatially consistent RGB observations from densely sampled surrounding views using a carefully designed texture-prior acquisition strategy and a diffusion renderer, enabling detailed human texture synthesis. Experiments show that our method can generate...
  </details>  
  Keywords: text-to-3d, texture synthesis  
- **[A Dual Path Framework with Hotspot Guided Fusion for Three Dimensional CT to PET Synthesis in Head and Neck Cancer](https://arxiv.org/abs/2607.21800v1)**  
  Authors: Mohd Maaz Khan, Oluwaseyi Oderinde  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2607.21800v1.pdf)  
  <details><summary>Abstract</summary>

  18F-FDG PET/CT plays a central role in staging, treatment planning, and response assessment for head and neck cancer by providing functional information that complements anatomical CT imaging. However, PET acquisition requires radiotracer administration, specialized infrastructure, and additional cost, limiting its availability for repeated imaging. We present a proof of concept deep learning framework for synthesizing PET like images directly from routine CT scans with the goal of providing complementary metabolic information that may support imaging triage and clinical decision support rather than replace diagnostic PET. Forty-four patients from the publicly available QIN-HEADNECK dataset were retrospectively analyzed using five fold cross-validation. We propose a fully three dimensional dual path architecture consisting of (i) a regression U-Net optimized for voxel-wise quantitative SUV estimation and (ii) a conditional generative adversarial network optimized for realistic PET texture. Their outputs are integrated using hotspot guided Laplacian pyramid blending, allowing quantitative information from the regression pathway...
  </details>  
  Keywords: texture synthesis  
- **[Recovering Cloud Microstructures with Cascaded Diffusion Inversion](https://arxiv.org/abs/2607.05637v1)**  
  Authors: Hanan Gani, Guy Pulik, Daniel Rosenfeld, Duncan Watson-Parris, Salman Khan  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2607.05637v1.pdf) | [![GitHub](https://img.shields.io/github/stars/hananshafi/superresolution-cloud-microphysics?style=social)](https://github.com/hananshafi/superresolution-cloud-microphysics)  
  <details><summary>Abstract</summary>

  High-resolution satellite imagery is critical for observing fine-scale cloud structures that inform weather modification strategies like cloud seeding for rain-enhancement. However, the spatial resolution of current geostationary and polar-orbiting satellites is often insufficient for capturing small cloud features. Current super-resolution methodologies are suited for natural images and, therefore, struggle to generalize to satellite-captured spectral images of cloud cover. To address this, we propose a two-stage diffusion-based super-resolution framework to enhance the resolution of multi-spectral cloud microstructures by a factor of $4\times$. Specifically, we use inverse diffusion to recover the high resolution properties from low resolution. Stage 1 utilizes real-world paired data to learn robust degradation handling and inter-sensor alignment, while Stage 2 employs a self-supervised internal downgrading of high resolution data to refine structural learning and texture synthesis. Our approach outperforms the state-of-the-art transformer and diffusion-based baselines in both reconstruction accuracy and visual quality. We demonstrate that the two-stage method...
  </details>  
  Keywords: texture synthesis  
- **[Ink3D: Sculpting 3D Assets with Extremely Complex Textures via Video Generative Models](https://arxiv.org/abs/2607.01222v1)**  
  Authors: Yue Han, Chong Li, Zhening Liu, Cong Huang, Fang Deng, Yong Liu, Fangyun Wei, Yan Lu  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2607.01222v1.pdf)  
  <details><summary>Abstract</summary>

  Recent 3D generative models can synthesize high-quality geometry but often struggle to reproduce intricate textures from reference images, largely due to the scarcity of large-scale 3D training data with rich surface appearance. In contrast, visual generative models are trained on datasets several orders of magnitude larger and excel at modeling complex visual patterns. Motivated by this gap, we introduce Ink3D, a framework that bridges 3D generation with large-scale video generative models to synthesize extremely complex textures. Ink3D first reconstructs a white-mesh geometry using an off-the-shelf 3D generation model. It then employs OrbitPainter, a conditional video generative model, to produce dense orbit-scan videos capturing object appearance across viewpoints. To convert these views into coherent textures, we introduce TextureOptimizer, a neural baking module that integrates dense multi-view observations while mitigating geometry inconsistencies arising from video generation. By decoupling geometry and texture synthesis and leveraging large-scale pretrained video priors, Ink3D enables significantly richer...
  </details>  
  Keywords: texture synthesis  
- **[JanusMesh: Fast and Zero-Shot 3D Visual Illusion Generation via Cross-Space Denoising](https://arxiv.org/abs/2606.20563v2)**  
  Authors: Siang-Ling Zhang, Huai-Hsun Cheng, Tsung-Ju Yang, Yu-Lun Liu  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2606.20563v2.pdf) | [![Project](https://img.shields.io/badge/-Project-blue)](https://siang1105.github.io/JanusMesh.github.io)  
  <details><summary>Abstract</summary>

  Creating 3D visual illusions, a single 3D mesh that reveals entirely different semantics from various viewing angles, is a fascinating but tough challenge. Existing optimization-based methods are slow and can produce oversaturated colors. In contrast, naive stitching approaches fail to produce geometrically coherent objects. This results in visible unnatural seams and semantic leaks. In this paper, we present a fast and training-free framework for generating text-driven 3D visual illusions. Our approach decouples the generation into two stages. First, we propose a cross-space dual-branch denoising process. This process dynamically decodes 3D latents into voxel space for CLIP-guided orientation alignment and Signed Distance Field (SDF) blending, which ensures seamless geometric fusion. Second, we introduce a view-conditioned texture synthesis module that projects and aggregates view-specific 2D diffusion priors onto the fused geometry. Extensive experiments demonstrate that our method generates highly realistic, dual-semantic 3D illusions in just 3-5 minutes. It significantly outperforms existing methods...
  </details>  
  Keywords: texture synthesis  
- **[Advances in Neural 3D Mesh Texturing: A Survey](https://arxiv.org/abs/2606.00137v1)**  
  Authors: Sai Raj Kishore Perla, Hao Zhang, Ali Mahdavi-Amiri  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2606.00137v1.pdf)  
  <details><summary>Abstract</summary>

  Texturing 3D meshes plays a vital role in determining the visual realism of digital objects and scenes. Although recent generative 3D approaches based on Neural Radiance Fields and Gaussian Splatting can produce textured assets directly, polygonal meshes remain the core representation across modeling, animation, visual effects, and gaming pipelines. Neural 3D mesh texturing therefore continues to be an essential and active area of research. In this survey, we present a comprehensive review of recent advances in neural 3D mesh texturing, covering methods for texture synthesis, transfer, and completion. We first summarize key foundations in mesh geometry, texture mapping, differentiable rendering, and neural generative models, and then organize the literature into a unified taxonomy spanning early GAN-based methods to modern diffusion-based pipelines. We further analyze common architectures and supervision strategies, review datasets and evaluation protocols, and discuss emerging applications, practical/commercial systems, and open challenges. Together, these insights provide a structured perspective...
  </details>  
  Keywords: texture synthesis  
- **[Perceptually Lossless Tactile Texture Synthesis with Compact Spectral Envelope Models](https://arxiv.org/abs/2605.23804v2)**  
  Authors: Jagan K. Balasubramanian, Yasemin Vardar  
  Links: [![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2605.23804v2.pdf)  
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