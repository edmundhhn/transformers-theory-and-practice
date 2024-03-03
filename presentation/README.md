# LongLoRA

Original [Paper](https://arxiv.org/abs/2309.12307): LongLoRA: Efficient Fine-tuning of Long-Context Large Language Models 
Yukang Chen, Shengju Qian, Haotian Tang, Xin Lai, Zhijian Liu, Song Han, Jiaya Jia, 2013

## Overview

__Problem__: 
- Training transformers with longer sequence lengths is a major problem, as computational costs scale quadratically context length. This begs the question of whether there is a method to extend the context window effectively. 

__Solution__:
- By making two notable changes to the LoRA model, LongLoRA demonstrates accuracy close to that of full fine tuning whilst utilizing far fewer GPU resources and training hours.

![image](https://github.com/edmundhhn/transformers-theory-and-practice/assets/97279107/3cc4eeed-fe73-4c77-be54-0fd584cdca4d)

## Question One: How Does LongLoRA improve over LoRA?

- __LoRA Recap__: Utilizes low rank matrices to compute weight updates, increasing efficiency and reducing the number of trainable parameters

 ![image](https://github.com/edmundhhn/transformers-theory-and-practice/assets/97279107/3c45da3d-cf83-4c56-aa4e-30553b4a483e)

__Problem__ :
- As context length increases, LoRA is neither effective nor efficient. Longer contexts result in a higher perpplexity and computational costs

__Solution__:
Two changes make the LongLoRA suprerior to LoRA for longer contexts:
- Firstly

## Question Two: How do We Define Better Performance in Long Contexts? 



## Architecture Overview



## Critical Analysis
