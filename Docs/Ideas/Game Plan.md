### 0.1.  GNN 
- Thevindu 
- Anotated data (JA) NEEDED
### 0.2.  Fuzzy Entropy
- Janith
- Use currently anotated data
### 0.3.  SPARC 
- Janith
- Use currently anotated data
### 0.4.  LDLJ-A
- Janith
- Use currently anotated data
### 0.5.  fPCA
- Paveen
- - Use currently anotated data

An fPCA score (Functional Principal Component Analysis score) serves a completely different purpose than SPARC or LDLJ-A. While SPARC and LDLJ-A calculate a single, explicit index for movement smoothness, fPCA is a data-driven statistical technique used to capture and quantify the dominant modes of variation across an entire movement curve. [1, 2, 3]

Instead of focusing only on "jerkiness," an fPCA score tells you how much and in what specific way an individual's movement trajectory deviates from the average population curve. [4]

---

## 1.  How the fPCA Score Works

Think of standard PCA as extracting patterns from static columns of numbers. fPCA extracts patterns from continuous time-series curves (like a velocity profile over time, or hip/knee angles during a stride). [1, 5]

1. The Mean Curve: fPCA computes the average trajectory across all subjects/trials. [6]
2. Functional Principal Components (fPCs): It extracts characteristic "modes of variation" (curves, not single numbers) that explain where the data varies the most. [1, 4]
3. The Score: For any individual trial, the fPCA score is a weight (a single number) indicating how strongly that specific trial matches a given fPC curve. [3, 4]

Mathematically, an individual movement curve $X_i(t)$ is reconstructed as:  
$$X_i(t) \approx \mu(t) + \text{score}_{i,1}\psi_1(t) + \text{score}_{i,2}\psi_2(t) + \dots$$

- μ(t) = Mean curve of the population.
- $\psi_k(t)$ = The k-th functional component (the pattern shape).
- $\text{score}_{i,k}$ = The fPCA score (how much of that pattern this individual exhibits). [4, 6]

---

## 2.  Biomechanical Meaning of fPCA Scores

An fPCA score allows researchers to turn complex, multi-second movement graphs into simple coordinates for clinical analysis: [2]

- Score = 0: The subject's movement perfectly mirrors the average shape of the group. [7]
- Positive vs. Negative Scores: Indicate opposite physical deviations from the norm.
    
    - _Example:_ If ψ₁(t) represents an early peak in velocity, a high positive score means the subject accelerated aggressively early on. A negative score means they accelerated much later than average. [8]
    

---

## 3.  Comparison: SPARC vs. LDLJ-A vs. fPCA Score

|Feature|SPARC / LDLJ-A|fPCA Score|
|---|---|---|
|Primary Goal|Direct evaluation of smoothness.|Mapping overall coordination variability.|
|Output Type|Fixed clinical score (e.g., -1.5 is always smooth).|Data-dependent weight (relative to the dataset).|
|What it Captures|Micro-fluctuations, tremors, and jerks.|Macro-shape shifts, timing delays, and amplitude changes.|
|Interpretation|Absolute (higher is always better).|Relative (positive/negative depend entirely on the pattern).|

---

## 4.  Can fPCA be used to measure smoothness?

Yes, but indirectly. If you perform fPCA on a messy dataset containing both healthy controls and tremor patients, the first or second functional components ($\psi_k(t)$) will likely capture the "shakiness" pattern.  
In that specific study, patients with higher fPCA scores for the shaky component are classified as having less smooth movements. However, unlike SPARC, those scores cannot be easily compared to external studies because the fPCA patterns are custom-built from your specific dataset. [1, 9, 10]

Are you planning to use fPCA to compare patient groups (e.g., stroke vs. control), or are you trying to reduce the dimensions of a massive kinematics dataset? Let me know if you need help choosing the right method. [1, 10]

  

[1] [https://www.jmp.com](https://www.jmp.com/en/analytics-in-practice/functional-data-analysis/understanding-functional-principal-component-analysis)

[2] [https://www.tandfonline.com](https://www.tandfonline.com/doi/full/10.1080/14763141.2024.2398508)

[3] [https://www.jmp.com](https://www.jmp.com/en/analytics-in-practice/functional-data-analysis/understanding-functional-principal-component-analysis)

[4] [https://towardsdatascience.com](https://towardsdatascience.com/functional-principal-component-analysis-and-functional-data-91d21261ab7f/)

[5] [https://casi.ie](https://casi.ie/2021/analysis-of-multivariate-and-multilevel-functional-data-in-sports-biomechanics/)

[6] [https://www.youtube.com](https://www.youtube.com/watch?v=2GbCtDH9q7o&t=179)

[7] [https://www.researchgate.net](https://www.researchgate.net/figure/Functional-principal-component-analysis-FPCA-scores-for-post-release-and-undisturbed_fig6_359962879)

[8] [https://www.sfu.ca](https://www.sfu.ca/science/stat/cao/Papers/ParametricFPCA.pdf)

[9] [https://stats.stackexchange.com](https://stats.stackexchange.com/questions/23566/functional-principal-component-analysis-fpca-what-is-it-all-about)

[10] [https://pure.ul.ie](https://pure.ul.ie/en/publications/common-functional-principal-components-analysis-a-new-approach-to/)