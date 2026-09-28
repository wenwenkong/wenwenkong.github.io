---
layout: post
title: Swin Transformer Notes
date: 2026-09-27
description: Learning notes of Swin Transformer
tags: computer-vision
#categories: course-summary
giscus_comments: false
related_posts: false
thumbnail: assets/img/posts/swin_notes/3_shifted_window.jpg 
toc:
  sidebar: left
---

This post covers the essence of the Swin Transformer (Liu et al., 2021). In the original work, the authors found that
> The proposed Swin Transformer achieves strong performance on the recognition tasks of image classification, object detection, and semantic segmentation.

Rather than going into the experimental details, this post focuses on the core ideas behind the model. In a nutshell, Swin Transformer is a **hierarchical transformer** whose representations are computed using **s**hifted **win**dows - that's where its name ("**S**hifted **Win**dow **Transformer**") comes from. The hierarchical design and the window-based self-attention with shifted windows are the two key structures of Swin that we will walk through in more detail later. 

--- 

## **Motivation**

Transformer is the default backbone of NLP applications: the same basic architecture can power language modeling, translation, question answering, and many other tasks. In computer vision, CNNs play a similar "general-purpose backbone" role. The Swin Transformer paper sets out to close this gap: 
> In this paper, we seek to expand the applicability of Transformer such that it can serve as a general-purpose backbone for computer vision, as it does for NLP and as CNNs do in vision.

However, it's challenging to transfer Transformer's high performance in the language domain to the vision domain. The challenge comes from differences between the two modalities: **scale** and **resolution**.

### Scale challenge
For typical sentence-level NLP tasks, Transformers operate on a single scale of tokens in a 1D sequence and do not rely on explicit multi-scale representations.[^nlp-hier] In contrast, visual elements can vary substantially in scale: small textures, medium-sized objects, large structures. Classical CNN backbones such as ResNet (He et al., 2016), often combined with a feature pyramid network (FPN; Lin et al., 2017), handle this wide range of scales using **hierarchical feature maps** that gradually reduce spatial resolution while increasing channel depth. ViT (see more in [this post](https://wenwenkong.com/blog/2025/ViT-notes/)), on the other hand, uses fixed-sized patches throughout the network and keeps a single feature resolution (see Figure 1b). Though this works well for image classification, in its original form (Dosovitskiy et al., 2021) it is not well suited as a general-purpose backbone for dense prediction tasks such as object detection or semantic segmentation.[^vit-variants]

[^nlp-hier]: Some NLP models do use hierarchical structures for longer documents (e.g., sentence -> paragraph -> document), such as Hierarchical Attention Networks for document classification (Yang et al., 2016), but those are more specialized and not the typical setup for the tasks we are referring to here.

[^vit-variants]: There are ViT-based variants that introduce multi-scale or hierarchical features and are used successfully for detection and segmentation. Examples include Pyramid Vision Transformer (PVT; Wang et al., 2021) and Multi-scale Vision Transformers (MViT; Fan et al., 2021). These architectures are ViT-inspired but explicitly add the kind of hierarchical structure and multi-scale features that the original single-scale ViT lacks.

### Resolution challenge
Images are also much higher resolution than typical text sequences. Standard self-attention has $$O(N^2)$$ complexity in the number of tokens $$N$$, because it computes interactions between all pairs of tokens. As a result, applying global self-attention at high resolution quickly becomes prohibitively expensive, especially for dense tasks that need fine spatial detail at the pixel level. 

### How Swin addresses these challenges
Swin Transformer is designed around these challenges. It introduces a **hierarchical representation** to handle scale variation, and **window-based self-attention with shifted windows** to make attention feasible at high resolution - reducing the computational complexity from quadratic in the number of tokens to approximately linear - while still allowing information to flow across the image.

--- 

## Core ideas
In this section, we introduce Swin’s two structural ideas: **hierarchical representation** and **shifted windows**. We also explain why Swin achieves an approximate **linear computational complexity**. 

### Hierarchical representation

<div class="row mt-3">
     <div class="col-sm mt-3 mt-md-0">
         {% include figure.liquid loading="eager" path="assets/img/posts/swin_notes/1_Swin_vs_ViT.png" class="img-fluid rounded z-depth-1" %}
     </div>
</div>
<div class="caption">
     <b>Figure 1.</b> Swin builds a hierarchical feature pyramid by merging patches in deeper stages (gray grids), similar to CNN backbones. Compared to ViT’s single-scale feature map, this multi-stage structure supports multi-scale visual patterns (adapted from Liu et al., 2021).
</div>

The gray grids in Figure 1a represent patch-level feature maps at different spatial resolutions. Shallow layers contain many patch tokens arranged at a higher resolution. As neighboring tokens are merged, deeper layers contain fewer, coarser patch tokens, where each token represents a larger area of the original image and typically has a larger embedding dimension.  This progression from high-resolution, fine-grained features to low-resolution, more abstract features is what we mean by a **hierarchical feature representation**.

The red boxes represent the local windows within which self-attention is computed. Instead of attending over all patches globally, Swin partitions the feature map into these red-outlined local windows, and applies self-attention only inside each window. This greatly reduces the computational cost compared with global attention.

In summary, Swin combines a CNN-like feature hierarchy with local window-based attention,  giving it both multi-scale representations and efficient computation. Here, the gray grids show what the representation consists of (i.e. the spatial grid of patch tokens), while the red boxes show where self-attention is computed (i.e. within local windows over those tokens).

Figure 2 provides an alternative visualization of the architectural differences between ViT and Swin Transformer. 
<div class="row mt-3">
     <div class="col-sm mt-3 mt-md-0">
         {% include figure.liquid loading="eager" path="assets/img/posts/swin_notes/2_Swin_vs_ViT.png" class="img-fluid rounded z-depth-1" %}
     </div>
</div>
<div class="caption">
     <b>Figure 2.</b> An alternative visualization of the architectural differences between ViT and Swin Transformer (adapted from Arkin et al., 2022).
</div>


### Why window-based attention scales linearly

We mentioned above that Swin’s self-attention is much cheaper than global self-attention. This efficiency comes from limiting self-attention to fixed-sized local windows. To understand why, let us introduce some notation:

- $$N$$: the total number of tokens (gray patches in Figure 1a) in the feature map at a given stage
- $$M$$: the side length of each window
- $$M^2$$: number of tokens in each local attention window (red boxes in Figure 1a). For example, the original Swin paper sets the side length of $$M = 7$$ by default, so each $$7 \times 7$$ window contains $$M^2 = 49$$ tokens.
- Note:
    - Each token corresponds to a progressively larger region of the original image at deeper stages due to patch merging. 
    - Swin tiles the feature map with $$N/M^2$$ non-overlapping windows, with each token belonging to exactly one window.

**Global self-attention (ViT-style)**

In global self-attention, every token can attend to every other token. This means:
- We construct an $$N \times N$$ attention matrix, where each entry represents the interaction between a pair of tokens.
- The pairwise-attention cost therefore scales like $$O(N^2)$$.

As image resolution increases while the patch size remains fixed, the image is divided into more patches and therefore produces more tokens ($$N$$). The quadratic cost of global self-attentil thus quickly becomes prohibitive for high-resolution feature maps.

**Window-based self-attention (Swin)**

Swin changes the picture by restricting attention to local windows:
- The feature map is partitioned into $$N/M^2$$ non-overlapping windows, each with $$M^2$$ tokens.
- Within each window, we compute self-attention only among those $$M^2$$ tokens. The cost per window is: $$O({M^2}^2) = O(M^4)$$
- Across the whole feature map, the total cost becomes: $$(N/M^2) \times O(M^4) = O(NM^2)$$

Here, $$M$$ (the side length of each window) is a fixed hyperparameter of the model (e.g., always $$7×7$$ in the original Swin paper). It does not grow with image size. As we increase image resolution, $$N$$ increases (i.e. more gray patches), but $$M^2$$ (i.e. the number of tokens per window) stays the same. Therefore, **for a fixed window size, the self-attention cost grows approximately linearly with the number of tokens, instead of quadratically as in global self-attention.**

Note: The full complexity expressions in Equations (1)  and (2) of Liu et al. (2021) also include the linear projections used to compute queries $$(Q)$$, keys $$(K)$$, values $$(V)$$, and attention output. These projection costs are the same for global and window-based attention. The simplified derivation here focuses on the pairwise-attention term, which is where their computational complexities differ. 


### Shifted window 
Though Swin can reduce cost by restricting attention to local windows, that creates an isolation problem: tokens in different windows cannot interact within the same attention block. Swin addresses this in the following Swin Transformer block at the same stage using the shifted window approach, allowing information to flow across window boundaries. 

<div class="row mt-3">
     <div class="col-sm mt-3 mt-md-0">
         {% include figure.liquid loading="eager" path="assets/img/posts/swin_notes/3_shifted_window.jpg" class="img-fluid rounded z-depth-1" %}
     </div>
</div>
<div class="caption">
     <b>Figure 3.</b> Window-based and shifted window self-attention. In one layer, self-attention is computed only within non-overlapping local windows (red boxes). In the next layer, the windows are shifted so that tokens can attend across previous window boundaries (reproduced from Liu et al. 2021).
</div>

#### How shifted window enables cross-window communication

Figure 3 illustrates how shifting the window partition between two consecutive blocks enables cross-window communication. In one block ($$L$$), Swin computes self-attention within non-overlapping windows. In the following block ($$L + 1$$) at the same stage, Swin shifts the window partition by half-window size in all directions, resulting in new windows that cross the previous window boundaries. As a result, some tokens that previously belonged to separate windows now appear in the same shifted window and attend to one another.

The window partition alternates between these two configurations across consecutive blocks. Thus, block ($$L+2$$) returns to the regular window partition used in ($$L$$), while block ($$L+3$$) uses the shifted partition again, as in ($$L+1$$). In other words, the shift does not accumulate across blocks. In practice, Swin implements the shifted-window configuration using a cyclic shift and an attention mask, which we will discuss in the Architecture section.


#### Sliding windows vs shifted windows

How does the shifted-window mechanism differ from sliding-window attention (Ramachandran et al., 2019)? Although both restrict self-attention to local neighborhoods, they differ in how those neighborhoods are defined and organized. In Swin, the feature map is partitioned into non-overlapping windows, and all query tokens within the same window share the same set of key and value tokens. In contrast, in sliding-window self-attention, the local neighborhood is centered on each query token, so different queries attend to different sets of keys and values. Figures 4-5 illustrate this distinction.


<div class="row mt-3">
     <div class="col-sm mt-3 mt-md-0">
         {% include figure.liquid loading="eager" path="assets/img/posts/swin_notes/4_Query_centered_attention.png" class="img-fluid rounded z-depth-1" %}
     </div>
</div>
<div class="caption">
     <b>Figure 4.</b> Query-centered local self-attention. For each query, attention is computed only over a local spatial neighborhood centered on that query. If the query position changes, its neighborhood moves with it. Adapted from Figures 1 and 3 of Ramachandran et al. (2019).  
</div>

<div class="row mt-3">
     <div class="col-sm mt-3 mt-md-0">
         {% include figure.liquid loading="eager" path="assets/img/posts/swin_notes/5_sliding_vs_shifted_windows.png" class="img-fluid rounded z-depth-1" %}
     </div>
</div>
<div class="caption">
     <b>Figure 5.</b> Visual comparison of (a) sliding-window, (b) window-based, and (c) shifted-window self-attention. In sliding-window attention, the local neighborhood moves with the query. In window-based attention, queries within the same non-overlapping window share a common key and value set. Swin shifts the window partition between consecutive blocks to enable cross-window communication. Panel (c) previews the cyclic-shift operation used to implement this shifted partition, which will be discussed in the Architecture section. Adapted from Hassani et al. (2023), Neighborhood Attention Transformer, CVPR 2023 presentation slides. 
</div>


## Architecture
In this section, we explore how the core ideas are assembled into the actual Swin backbone. 

### Overall architecture

<div class="row mt-3">
     <div class="col-sm mt-3 mt-md-0">
         {% include figure.liquid loading="eager" path="assets/img/posts/swin_notes/6_Swin_architecture.png" class="img-fluid rounded z-depth-1" %}
     </div>
</div>
<div class="caption">
     <b>Figure 6.</b> Overall architecture of the Swin Transformer. The left panel shows the four hierarchical stages and patch-merging layers, and the right panel expands two successive Swin Transformer blocks, illustrating the alternation between W-MSA and SW-MSA. Reproduced from Liu et al. (2021).
</div>

Let’s first have a high-level overview of the Swin model architecture. Figure 6 (reproduced from Figure 3 of Liu et al. 2021) covers the main pieces, and we will unpack the key components below. 

**Stages**

The original Swin Transformer models consist of four hierarchical stages, with spatial resolution progressively reduced between stages through patch merging (Figure 6a).

**Patch size and embedding dimensions**

Liu et al. (2021) use a patch size of $$4 \times 4$$, each patch therefore contains $$(4\times 4\times 3 = 48)$$ raw pixel values. After patch partitioning, the image is represented as a grid of $$(H/4 \times W/4)$$ patches, each with a $$48-$$dimensional feature vector. A linear embedding layer then projects each patch vector into a learned $$C-$$dimensional feature vector, giving the input to Stage 1 a shape of $$H/4 \times W/4 \times C$$. Note that here $$C$$ plays conceptually the same role as the embedding dimension $$D$$ in ViT. 

**Patch merging between stages**

Patch merging reduces spatial resolution while increasing the embedding dimension. At each patch-merging layer, Swin groups neighboring tokens in fixed $$2\times 2$$ blocks. This $$2\times 2$$ grouping is a design choice that gives a natural $$2\times$$ spatial downsampling in both height and width, similar to stride-2 downsampling in a CNN. 

If the incoming tokens at a given stage have embedding dimension $$d$$, where $$d$$ is used here as a generic notation for that stage’s current embedding dimension, the four $$d$$-dimensional token vectors are concatenated into a $$4d$$-dimensional vector and then linearly projected to $$2d$$. Thus, each patch-merging step halves the spatial resolution in both directions while doubling the embedding dimension. 

Therefore, starting with embedding dimension $$C$$ in Stage 1, repeated patch-merging produces the following stage-wise progression:

$$
\frac{H}{4}\times\frac{W}{4}\times C
\rightarrow
\frac{H}{8}\times\frac{W}{8}\times 2C
\rightarrow
\frac{H}{16}\times\frac{W}{16}\times 4C
\rightarrow
\frac{H}{32}\times\frac{W}{32}\times 8C.
$$

**Swin Transformer blocks**

Each stage contains multiple Swin Transformer blocks; the number of blocks varies across Swin variants. Within each stage, successive blocks always alternate between W-MSA (window-based multi-head self-attention) and SW-MSA (shifted-window multi-head self-attention), as illustrated by the two consecutive blocks in Figure 6b.

### Cyclic shifting

<div class="row mt-3">
     <div class="col-sm mt-3 mt-md-0">
         {% include figure.liquid loading="eager" path="assets/img/posts/swin_notes/7_Cyclic_shift_schematic_diagram.png" class="img-fluid rounded z-depth-1" %}
     </div>
</div>
<div class="caption">
     <b>Figure 7.</b> Schematic of the cyclic-shift and masking mechanism. Reproduced from Wang et al. (2025). This schematic illustrates the same mechanism as Figure 4 of Liu et al. (2021), but presents it in a clearer way.
</div>

We have seen conceptually how shifted windows enable cross-window communication (Figure 3). But directly implementing the shifted partition introduces an efficiency problem at the feature-map boundaries: it creates more windows, including smaller partial windows along the edges. For example, in the example shown in Figure 7, the number of windows increases from 4 to 9. Swin addresses this using cyclic shifting and attention masking. 

Cyclic shifting works by shifting the feature map toward the top-left by approximately half a window size $$M/2$$, where $$M$$ denotes the side length of each $$M\times M$$ attention window. Tokens that cross the top boundary wrap to the bottom, those that cross the left boundary warp to the right, and those that cross both wrap to the bottom-right (Figure 7). This rearrangement allows the shifted configuration to be processed using the same number of regular $$M\times M$$ computational windows as regular window partitioning. However, it also places sub-windows that were not adjacent in the original feature map into the same computational window. Masked MSA therefore ensures tokens only attend to other tokens within the same sub-window. For example, in Figure 8, all tokens within region 5 can attend to one another, whereas regions 8 and 2 share a computational window only because of cyclic wraparound, so attention between regions 8 and 2 is masked out. After Masked MSA, a reverse cyclic shift moves the entire feature map back to its original spatial arrangement. See also my sketch in Figure 8 for a visual explanation. 


<div class="row mt-3">
     <div class="col-sm mt-3 mt-md-0">
         {% include figure.liquid loading="eager" path="assets/img/posts/swin_notes/8_cyclic_shift_sketch.png" class="img-fluid rounded z-depth-1" %}
     </div>
</div>
<div class="caption">
     <b>Figure 8.</b> My sketch to better understand how the cyclic shifting and masked MSA work.  
</div>


---

## References

- Arkin, E., Yadikar, N., Xu, X., Aysa, A., & Ubul, K. (2023). [A survey: Object detection methods from CNN to transformer](https://doi.org/10.1007/s11042-022-13801-3). *Multimedia Tools and Applications, 82*, 21353–21383.

- Dosovitskiy, A., Beyer, L., Kolesnikov, A., Weissenborn, D., Zhai, X., Unterthiner, T., Dehghani, M., Minderer, M., Heigold, G., Gelly, S., Uszkoreit, J., & Houlsby, N. (2021). [An image is worth 16×16 words: Transformers for image recognition at scale](https://arxiv.org/abs/2010.11929). *International Conference on Learning Representations (ICLR)*.

- Hassani, A., Walton, S., Li, J., Li, S., & Shi, H. (2023). [*Neighborhood Attention Transformer* — CVPR 2023 presentation slides](https://cvpr2023.thecvf.com/media/cvpr-2023/Slides/23172.pdf). IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR 2023).

- He, K., Zhang, X., Ren, S., & Sun, J. (2016). [Deep residual learning for image recognition](https://doi.org/10.1109/CVPR.2016.90). *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*, 770–778.

- Lin, T.-Y., Dollár, P., Girshick, R., He, K., Hariharan, B., & Belongie, S. (2017). [Feature pyramid networks for object detection](https://doi.org/10.1109/CVPR.2017.106). *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*, 2117–2125.

- Liu, Z., Lin, Y., Cao, Y., Hu, H., Wei, Y., Zhang, Z., Lin, S., & Guo, B. (2021). [Swin Transformer: Hierarchical Vision Transformer using shifted windows](https://doi.org/10.1109/ICCV48922.2021.00986). *Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)*, 10012–10022.

- Ramachandran, P., Parmar, N., Vaswani, A., Bello, I., Levskaya, A., & Shlens, J. (2019). [Stand-alone self-attention in vision models](https://proceedings.neurips.cc/paper/2019/hash/3416a75f4cea9109507cacd8e2f2aefc-Abstract.html). *Advances in Neural Information Processing Systems (NeurIPS), 32*.

- Wang, H., Zhang, F., & Yi, R. (2025). [Multi-scale deformable transformer with iterative query refinement for hot-rolled steel surface defect detection](https://doi.org/10.3390/s25226890). *Sensors, 25*(22), 6890.

## Footnote
