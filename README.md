# mHC: Manifold-Constrained Hyper-Connections

The Hyper-Connections (HC) architecture, while powerful, breaks the identity mapping property of standard residual connections. This leads to unbounded signal amplification or attenuation (exploding or vanishing signals) when stacking many layers, causing severe training instability and limiting the scalability of deep models. Additionally, the widened residual stream in HC introduces significant memory access overhead.

### Paper Summary

The paper identifies a key problem with a recent architecture called Hyper-Connections (HC): while widening the residual stream improves performance, the unconstrained nature of its learnable connection matrices leads to severe training instability. This is because it disrupts the crucial "identity mapping" property of standard residual connections, causing signals to explode or vanish over many layers. The authors propose **Manifold-Constrained Hyper-Connections (mHC)**, a framework that projects the residual connection matrix (`H_res`) onto the manifold of doubly stochastic matrices. This projection, achieved via the Sinkhorn-Knopp algorithm, ensures that the matrix acts as a stable convex combination of its inputs, restoring the identity mapping property and enabling stable, scalable training while retaining the performance benefits of HC.

### What I'll Implement (Coming Soon)

Build a small, decoder-only Transformer from scratch to test the experiment. We'll implement three versions of the residual connection within the Transformer blocks:
1.  **Standard Connection:** The classic `x + F(x)` from ResNets and Transformers.
2.  **Hyper-Connection (HC):** The unstable predecessor, with unconstrained `tanh` activated matrices.
3.  **Manifold-Constrained HC (mHC):** The paper's proposed solution, using the Sinkhorn-Knopp algorithm to enforce a doubly stochastic constraint.

By training these models on a simple sequence-copying task, we will directly compare their training stability (via loss and gradient norms) and performance, aiming to replicate the paper's key findings in a simplified, educational setting.

## Problem Intuition

### The "Game of Telephone" in Deep Networks

Imagine training a very deep neural network as a game of telephone. Each layer must pass information to the next. 

*   **Standard Residual Connections (`x_new = x_old + F(x_old)`)**: This is like whispering the original message (`x_old`) and then adding a small, computed correction (`F(x_old)`). The core message gets through almost perfectly at each step. This is the **identity mapping property**, and it's why we can train networks with hundreds of layers.

*   **Hyper-Connections (HC)**: HC tries to be cleverer. Instead of just adding, it *transforms* the message at each step: `x_new = H_res * x_old + ...`. The matrix `H_res` is learned. The problem is, if `H_res` isn't carefully controlled, it can act as an amplifier or a muffler. After many layers, a small amplification at each step can lead to a deafening, meaningless shout (exploding gradients), while a small muffling can lead to silence (vanishing gradients). The original message is lost.

### The Key Insight: Stable Mixing

The authors of mHC propose a brilliant solution: what if we constrain `H_res` so it can only *mix* the information, not amplify or muffle it? 

They enforce a **doubly stochastic** constraint on `H_res`. This means two things:
1.  All its entries are non-negative.
2.  Every row and every column sums to exactly 1.

A matrix with these properties can only perform a **convex combination**. It's like mixing different colored paints: you can change the final color, but the total volume of paint remains constant. This constraint ensures that the overall signal energy is preserved across layers, preventing explosions or vanishing. The Sinkhorn-Knopp algorithm is a classic and efficient way to project any matrix onto this "stable mixing" manifold.

In essence, **mHC gets the best of both worlds**: it allows for learnable, dynamic mixing of features between residual streams (the benefit of HC) while guaranteeing the signal propagation stability of standard residual connections.

## Dataset & Tokenization

I'll use a synthetic sequence-copying task. The model is given a sequence of unique characters (e.g., `<s>FJRBE</s>`) and must learn to reproduce it exactly. This task is a powerful probe for stability. An unstable model like HC will struggle to preserve the information of early tokens (like 'F') across multiple layers. A stable model like mHC should have no problem.

I create a simple character-level vocabulary and a PyTorch `Dataset` to handle tokenization and batching.

## Model Architecture

We now implement the core components of the mHC architecture. The implementation is modular, allowing us to easily swap between `StandardConnection`, `HCModule`, and `mHCModule`.

### Core Components:

1.  **`SinkhornKnopp`**: A faithful implementation of Equation (9). It takes an unconstrained matrix, exponentiates it to ensure positivity, and then iteratively normalizes rows and columns to approximate a doubly stochastic matrix.

2.  **`RMSNorm`**: A standard Root Mean Square Layer Normalization module, used throughout modern Transformers.

3.  **Connection Modules (`mHCModule`, `HCModule`, `StandardConnection`)**: These are the heart of our experiment. Each module takes the input `x` and the sublayer function `F` (e.g., Attention or FFN) and computes the output of the block according to its specific paradigm.
    - `mHCModule` follows Equations (7) and (8) precisely.
    - `HCModule` follows Equation (5), using `tanh` for unconstrained mappings.
    - `StandardConnection` implements the simple `x + F(x)`.

4.  **`TransformerBlock`**: A standard pre-norm Transformer block that integrates a connection module. The data flow is `Norm -> Attention -> Connection -> Norm -> FFN -> Connection`.

5.  **`HyperTransformer`**: The final model. It orchestrates the embeddings, positional encoding, Transformer blocks, and the final output projection. For HC and mHC, it handles the initial expansion of the residual stream from dimension `C` to `n*C` and the final collapse back to `C` for the language model head.

<img width="1014" height="529" alt="Screenshot 2026-02-11 at 15 20 43" src="https://github.com/user-attachments/assets/01720661-546a-4e12-aadb-43f0410defcf" />


# Citations

Implementation of the architecture is based on this [paper](https://arxiv.org/abs/2512.24880)
```
@misc{xie2026mhcmanifoldconstrainedhyperconnections,
      title={mHC: Manifold-Constrained Hyper-Connections}, 
      author={Zhenda Xie and Yixuan Wei and Huanqi Cao and Chenggang Zhao and Chengqi Deng and Jiashi Li and Damai Dai and Huazuo Gao and Jiang Chang and Kuai Yu and Liang Zhao and Shangyan Zhou and Zhean Xu and Zhengyan Zhang and Wangding Zeng and Shengding Hu and Yuqing Wang and Jingyang Yuan and Lean Wang and Wenfeng Liang},
      year={2026},
      eprint={2512.24880},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2512.24880}, 
}
```
