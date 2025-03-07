# Multihead_Attention_PyTorch
Implementing the Multi-Head Attention Layer in PyTorch.

<p align="center">
  <img src="https://github.com/VictorFrancheto/Multihead_Attention_PyTorch/blob/main/image_neural.jpg">
</p>

# 🧠 Multi-Head Attention: A Deep Dive

**Multi-Head Attention (MHA)** is one of the fundamental building blocks of **Transformers**, revolutionizing sequence processing in tasks like **machine translation, text generation, and contextual understanding**. 🚀

Let's explore how it works and how it is implemented in **PyTorch**! 🔥

---

## 🎯 What is Multi-Head Attention?

**Attention** allows a model to focus on different parts of the input when processing a sequence. However, a single attention matrix can be **limited**. **Multi-Head Attention** solves this by computing **multiple** attention representations in parallel, enabling the model to capture **different aspects of the context**.

Each **attention head** processes the input independently, and the results are combined to produce a richer representation of the sequence. 🔄

---

## ⚙️ How Does It Work?

1️⃣ **Linear Projections**: The input is transformed into three matrices:
   - **Q (Queries)** – Represents the words making the query.\
   - **K (Keys)** – Represents the "indices" used to fetch information.\
   - **V (Values)** – Contains the actual information to be retrieved.

2️⃣ **Scaled Dot-Product Attention**: For each pair \( (Q, K) \), the attention weights are computed as:

   $$ \text{Attention}(Q, K, V) = \text{softmax} \left( \frac{QK^T}{\sqrt{d_k}} \right) V $$

   The factor \( \sqrt{d_k} \) prevents exploding gradients and improves training stability.

3️⃣ **Splitting into Multiple Heads**: Instead of using a single attention matrix, **MHA** splits the input into multiple “heads,” allowing each to learn different patterns. The number of heads is a model hyperparameter.

4️⃣ **Concatenation and Final Projection**: The outputs of all heads are concatenated and passed through a **final linear layer**, generating the **MHA** output.

---

## 📊 Why Use Multi-Head Attention?

✅ **Learns multiple patterns** – Each head focuses on different aspects of the input.\
✅ **Enhanced expressiveness** – The model learns richer sequence representations.\
✅ **Better generalization** – Captures deeper contextual relationships between words.

---

## 🔍 Key Considerations

📌 **Number of Heads**: Too many heads increase computational complexity, while too few may limit model expressiveness.\
📌 **Model Dimension**: The sum of the head dimensions must match the input dimension for consistency.\
📌 **Use in Transformers**: MHA is a core component in **BERT, GPT, and Vision Transformers (ViTs)**.

---

Now you understand **Multi-Head Attention** and how it enhances attention-based models! 🚀

🔜 Ready to implement it in **PyTorch**? Let's code! 🧑‍💻

