# Multihead_Attention_PyTorch
Implementing the Multi-Head Attention Layer in PyTorch.

<p align="center">
  <img src="https://github.com/VictorFrancheto/Multihead_Attention_PyTorch/blob/main/image_neural.jpg">
</p>

# 🧠 Head Attention Layer in PyTorch  

**Head Attention** is the core mechanism behind **self-attention**, allowing models to dynamically weigh input elements based on relevance. This fundamental operation powers models like **Transformers, BERT, and GPT**, making them highly effective for sequence-based tasks. 🚀  

Let’s explore how a **single attention head** works and why it’s essential! 🔥  

---

## 🎯 What is Head Attention?  

A **single attention head** computes the importance of each input element relative to others. Unlike traditional RNNs, which process sequences step by step, **attention enables direct interactions between all elements** in a single pass.  

At its core, an **attention head** takes three key inputs:  

- **Q (Query)** – The element making a request for information.
- **K (Key)** – The reference used to compare against queries.
- **V (Value)** – The actual data being retrieved.  

These components allow the model to determine **which words (or elements) are most relevant** for each query.  

---

## ⚙️ How Does a Single Attention Head Work?  

1️⃣ **Compute Attention Scores**:  
   The attention mechanism calculates the similarity between **Q** and **K** using a scaled dot product:  

   $$ \text{Score} = QK^T $$  

2️⃣ **Apply Scaling and Softmax**:  
   The scores are normalized to prevent large values, ensuring stable gradients:  

   $$ \text{Attention Weights} = \text{softmax} \left( \frac{QK^T}{\sqrt{d_k}} \right) $$  

   This step ensures that attention values sum to **1**, creating a probability distribution.  

3️⃣ **Weight the Values**:  
   The computed weights are used to reweight the **V (Values)**, determining how much information each element contributes:  

   $$ \text{Output} = \text{Attention Weights} \times V $$  

4️⃣ **Final Projection**:  
   The output of the attention head is transformed via a linear layer to ensure it fits the model’s architecture.  

---

## 🔍 Why Use a Single Attention Head?  

✅ **Captures dependencies across sequence elements.**  
✅ **Eliminates recurrence, enabling parallel processing.**  
✅ **Foundation of Multi-Head Attention (MHA).**  

While a single head provides **context-aware representations**, stacking multiple heads (as in Multi-Head Attention) allows the model to capture diverse relationships in the data.  

---

## 📊 Key Considerations  

📌 **Dimensionality Matching**: The input and output dimensions must align for correct computation.
📌 **Computational Efficiency**: Compared to MHA, a single head is **faster but less expressive**.
📌 **Scalability**: Used in lightweight attention-based architectures where fewer parameters are desired.  


