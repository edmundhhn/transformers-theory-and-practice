# LongLoRA

Original [Paper](https://arxiv.org/abs/2309.12307): LongLoRA: Efficient Fine-tuning of Long-Context Large Language Models 
Yukang Chen, Shengju Qian, Haotian Tang, Xin Lai, Zhijian Liu, Song Han, Jiaya Jia, 2013

Github [Repository](https://github.com/dvlab-research/LongLoRA)

## Overview

__Problem__: 
- Training transformers with longer sequence lengths is a major problem, as computational costs scale $O(n^2)$ with context length. This begs the question of whether there is a method to extend the context window effectively. 

__Solution__:
- By making two notable changes to the LoRA model, LongLoRA demonstrates accuracy close to that of full fine tuning whilst utilizing far fewer GPU resources and training hours.

![image](https://github.com/edmundhhn/transformers-theory-and-practice/assets/97279107/3cc4eeed-fe73-4c77-be54-0fd584cdca4d)

## Question One: How Do You Think LoRA Performs for Longer Context Lengths?

- __LoRA Recap__: Hypothesizes that weight updates in pretrained models have a low intrinsic rank during adaptation. Thus, weights are updated with a low rank decomposition, increasing efficiency and reducing the number of trainable parameters

$$W + \Delta W = W + BA \textrm{ where } B \in \mathbb{R}^{d \times r}, A \in \mathbb{R}^{r \times k}$$ 

 ![image](https://github.com/edmundhhn/transformers-theory-and-practice/assets/97279107/3c45da3d-cf83-4c56-aa4e-30553b4a483e)

__Problem__ :
- As context length increases, LoRA is neither effective nor efficient. Longer contexts result in a higher perplexity and computational costs when compared to the full-fine tuning procedure.

__Solution__:
Two important changes make the LongLoRA suprerior to LoRA for longer contexts:

__Trainable Normalization and Embedding Layers__:
- The authors empirically show that making the embedding and normalization layers learnable is immesnsely helpful in getting LoRA to learn long contexts.
- This is despite the fact that these layers only take up a very small percentage of total parameters i.e. <2% for embeddings and <0.004% for normalization in LLama2.
![image](https://github.com/edmundhhn/transformers-theory-and-practice/assets/97279107/4b58ee86-9ef4-40d7-b598-c3d84390055e)

__Shifted Sparse Attention ($S^2$ Attention)__: 
- To save compute the authors propose a neat trick called shifted sparse attention.
- When we split a context into groups, long contexts cause perplexity to rapidly increase, as there is no information exchange between different groups
- To achieve information flow, shifted sparse attention introduces "half attention heads", splitting attention heads along the head dimension into two chunks. Each half head receives tokens that are shifted relative to each other by half the group size.
- For example with a context length of 8192, the first group uses two patterns in each half self attention head, 1-2048 and 1025-3072.
- This method enables information flow between groups without introducing additional compute.

![image](https://github.com/edmundhhn/transformers-theory-and-practice/assets/97279107/d9ddbf42-5cf5-4987-8ec0-74da9ed25e6b)


## Question Two: How Do You Think the Performance on Longer Contexts can be Measured? 

__Long Sequence Language Modelling__: 
- Perplexity is evaluated on proof-pile and PG19 datasets. Under various training context lengths, the authors show that better perplexity is achieved by increasing the perplexity from 8192 to 32768
- For the 7B model, perplexity decreases from 2.72 to 2.50, for the 13B model, perplexity reduces from 2.60 to 2.32. 

![Screenshot 2024-03-03 at 2 02 07 PM](https://github.com/edmundhhn/transformers-theory-and-practice/assets/97279107/a7735522-ee89-4af2-ad31-f73f0c604a99)


__Retrieval Based Evaluation__: 
- Topic Retrieval: retrieving target topics from a very long conversation with lengths varying from 3k, 6k, 10k, 13k and 16k. Using a LongLoRA model fine tunde on Llama 13B with a 18k context length, comparable performance is achieved to the state of the art LongChat-13B, despite the lower fine tuning costs.
![image](https://github.com/edmundhhn/transformers-theory-and-practice/assets/97279107/1e367f94-5f6f-4cbf-a88a-60ccf5554a08)
- Passkey Retrieval: Finding a random passkey hidden in a long document. In this experiemnt, Llama 7B was finetuned with LongLoRA using 32768 context length. It was also found that modifying max position interpolation to 48k resulted in extended performance, without further fine tuning. 
![image](https://github.com/edmundhhn/transformers-theory-and-practice/assets/97279107/002cf882-be7f-4f40-b205-093999675f96)


## Architecture Overview

__Modification of Attention Mechanism During Training__

```
# Computes a single (masked) self or cross attention head
# Input: X, Z, vector representaitons of primary and context sequence
# Output: V_tilda: Updated representations of tokens in X, folding in information from tokens in Z
# Parameters: B_q, A_q, B_k, A_k, B_v, A_v, B_v, A_v

Attention(X, Z):
 Q = (B_q @ A_q) @ X + b_q @ ones.T
 K = (B_k @ A_k) @ Z + b_k @ ones.T
 V = (B_v @ A_v) @ Z + b_v @ ones.T
 S = K.T @ Q
 S = mask(S) # apply mask
 V_tilda = dot(V, softmax(S / sqrt(d_attn)))
 return V_tilda
```

__Modification of MHAttention__

```
# Input X, Z vector representaitons of primary and context sequence
# Output: V_tilda: Updated representations of tokens in X, folding in information from tokens in Z

num_groups = 4
group_size = len(X) / num_groups

# Split sequences into groups
groups = [lst[i:i + num_groups] for i in range(0, len(X), num_groups)]

# Shift Sequences
shifted_groups = 

# For a given group we have two patterns, X and X_shift
MHAttention(X, Z):
 for h_1 in H[:len(h)/2]:
  Y_h = Attention(X, Z)
 for h_2 in H[len(h)/2:]:
  Y_h = Attention(X_shift, Z_shift)

 Y = [Y_1, Y_2 ..... Y_H]...
 return V_tilda = (B_o @ A_o) + b_o @ ones.T
```

__Implementation of S2 Attention (From Paper)__
```
# B: batch size; S: sequence length or number of tokens; G: group size;
# H: number of attention heads; D: dimension of each attention head

# qkv in shape (B, N, 3, H, D), projected queries, keys, and values
# Key line 1: split qkv on H into 2 chunks, and shift G/2 on N
qkv = cat((qkv.chunk(2, 3)[0], qkv.chunk(2, 3)[1].roll(-G/2, 1)), 3).view(B*N/G,G,3,H,D)

# standard self-attention function
out = self_attn(qkv)

# out in shape (B, N, H, D)
# Key line 2: split out on H into 2 chunks, and then roll back G/2 on N
out = cat((out.chunk(2, 2)[0], out.chunk(2, 2)[1].roll(G/2, 1)), 2)
```

## Critical Analysis

__Dataset Usage__
- The datasets used in the evaluation process are not very mainstream. The main two being [proof-pile, Azerbayev et al](https://github.com/zhangir-azerbayev/proof-pile) and [PG-19, Rae et al](https://openreview.net/pdf?id=SylKikSYDH). Both of these datasets are not very mainstream and could raise questions about the reliability and comparability of these evaluations.

__Base Models__
- Throughout the whole process, only the fine-tuned Llama-2 is used, it is unsure whether the same results can be replicated with other base models.

__Emperical Metrics__
- The main results presented in the paper are exclusively based around perplexity metrics and retrieval processes. Very little is done in terms of evaluating how it useful it actually is for specific use cases (E.g. question answering, text summarization, sentiment analysis)

__Comparing to LoRA__
The original LoRA paper uses much more Robust experiments, including a variety of base models including BERT, RoBERTA, GPT-2 and GPT-3, for a variety of downstream tasks (Including natural language generation and understanding) on very mainstream and standardized datasets (e.g. GLUE, wikiSQL). Although this paper may have been impeded by resource costs, the comparisons on these benchmarks would be a lot more reliable. 

## Resource Links

1. LoRA Paper: Edward J. Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, Weizhu Chen. 2021. LoRA: Low-Rank Adaptation of Large Language Models. [Paper](https://arxiv.org/abs/2106.09685)
2. Llama 2 Paper: Touvron et al. Llama 2: Open Foundation and Fine-Tuned Chat Models [Paper](https://arxiv.org/abs/2307.09288)
3. Proof-Pile: Dataset used for long sequence modelling evaluation [Link](https://huggingface.co/datasets/hoskinson-center/proof-pile)
4. LongChat: Data used for passkey retrieval task: [Link](https://lmsys.org/blog/2023-06-29-longchat/)
5. Learn more about LongLoRA through this YouTube [Video](https://www.youtube.com/watch?v=hf5N-SlqRmA&t=812s&pp=ygUIbG9uZ2xvcmE%3D)


## Appendix

__Perplexity Recap__
Perplexity is a measure of how well a probability distribution or a probability model predicts a sample. In the context of a Large Language Model (LLM), perplexity is often used to evaluate the model's performance in predicting the next word in a sequence of words. The formula for calculating perplexity is as follows:

$$
\
\text{Perplexity}(\mathcal{D}) = \exp\left(-\frac{1}{N}\sum_{i=1}^{N}\log P(w_i|\text{context}_i)\right)
\
$$
