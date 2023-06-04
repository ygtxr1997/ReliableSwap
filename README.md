# ReliableSwap: Boosting General Face Swapping Via Reliable Supervision

[ArXiv]() | [Project Page]()

<div>
<span class="author-block">
<a href="https://github.com/ygtxr1997" target="_blank">Ge Yuan</a><sup>1,2,+</sup></span>,
<span class="author-block">
<a href="https://scholar.google.com/citations?user=ym_t6QYAAAAJ&hl=zh-CN&oi=sra" target="_blank">Maomao Li</a><sup>2,+</sup>,
</span>
<span class="author-block">
    <a href="https://yzhang2016.github.io" target="_blank">Yong Zhang</a><sup>2,*</sup>,
</span>
<span class="author-block">
<a href="https://scholar.google.com/citations?user=CCUQi50AAAAJ" target="_blank">Huicheng Zheng</a><sup>1,*</sup>
</span> (+ Equal Contributions, * Corresponding Authors)
</div>

  
<div class="is-size-5 publication-authors">
    <span class="author-block">
    <sup>1</sup> Sun Yat-sen University &nbsp;&nbsp;&nbsp;
    <sup>2</sup> Tencent AI Lab &nbsp;&nbsp;&nbsp;
    </span>
</div>
<br>

**TL;DR: A general face swapping training framework that:**

✅ solves no-guidance problems <br>
✅ enhances source identity preservation <br>
✅ is orthogonal and compatiable with existing methods <br>

![Fig1](./assets/Fig1.png)

### What Problems We Solve

![Fig3](./assets/Fig3.png)

During face swapping training, the re-construction task (used when $X_{\rm{t}}=X_{\rm{s}}$) cannot be used as the proxy anymore when $X_{\rm{t}} \neq X_{\rm{s}}$, lacking pixel-wise supervison $\Gamma$.

### How It Works

![Fig4](./assets/Fig4.png)

![Tab2](./assets/Tab2.png)

We first use real images $C_{\rm{a}}$ and $C_{\rm{b}}$ to synthesize fake images $C_{\rm{ab}}$ and $C_{\rm{ba}}$.
This synthesizing stage preserves the true source identity based on Multi-Band Blending.

![Fig2](./assets/Fig2.png)

Then based on the **cycle relationship**, for face swapping training stage, we use *fake* images as inputs while *real* images as pixel-level supervisons $\Gamma$, keeping the output domain close to the *real* and natural distribution and solving the non-supervision issue.
In this way, the trainable face swapping network is guided to generate identity-consistency swapping results.


More details can be found in our [project]() page.


### TODO

- [ ] release code
- [ ] extending to $512^2$ resolution

### BibTex


```tex
TBD
```


