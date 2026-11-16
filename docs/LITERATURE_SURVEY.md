# Literature Survey

**Project:** Smart Model Router: cost-aware routing between retrieval, a small local LLM and a large LLM

> Each reference links to its arXiv or publisher page.

---

## 1. Introduction

Large language models (LLMs) differ widely in quality, cost and speed. A frontier model answers
almost anything well but is expensive per token (or, if run locally, needs costly hardware). A
3-billion-parameter model runs free on a laptop but fails on harder reasoning. In real traffic,
such as student questions to a department chatbot, most queries are easy and a few are hard.
Sending every query to the strongest model therefore wastes money, and sending every query to the
weakest wastes quality.

This survey reviews the research that addresses this trade-off. It is organised into six themes:

1. LLM cascades (try cheap first, escalate if needed)
2. Learned routers (predict the best model *before* calling any)
3. Benchmarks for routing
4. Confidence and self-evaluation of LLMs (the signal cascades rely on)
5. LLM-as-a-judge (how routing labels are produced)
6. Retrieval and embeddings (answering without an LLM)

It closes with code-mixed and Indian-language NLP, a comparison table, and the research gap this
project targets.

---

## 2. Background: conditional computation

The idea of sending each input to only part of a system is older than LLMs.

- **Cascade classifiers.** Viola and Jones (2001) used a chain of increasingly expensive
  classifiers for face detection. Most image windows are rejected by cheap early stages, and only
  hard ones reach later stages. LLM cascades follow the same principle.
- **Mixture of Experts.** Shazeer et al. (2017) introduced a sparsely gated mixture-of-experts layer
  in which a learned gating network sends each token to a few "expert" sub-networks. Routing there
  happens *inside* one model at the token level. This project routes *between* separate models at
  the query level.

---

## 3. LLM cascades

**FrugalGPT** (Chen, Zaharia & Zou, 2023) is the most cited starting point. It proposes three cost
reduction strategies for LLM APIs: prompt adaptation, LLM approximation (caching or fine-tuning a
cheap model), and the **LLM cascade**. In the cascade, a query goes to the cheapest model first, a
learned scoring function judges whether the answer is reliable, and if not the query moves to the
next, more expensive model. The authors report matching the best single model's performance at a
large cost reduction on several datasets. FrugalGPT's cascade needs a *trained scorer* and pays for
every cheap attempt, even when the query later escalates.

**Mixture of Thoughts cascades** (Yue et al., 2024, ICLR) apply cascades to reasoning tasks. The
weaker model is sampled several times, with chain-of-thought and program-of-thought prompts, and the
query escalates to the stronger model only when the answers *disagree*. Answer consistency serves as
the confidence signal, with no trained scorer, and large savings are reported on math and reasoning
benchmarks.

**AutoMix** (Aggarwal, Madaan et al., NeurIPS 2024) lets the small model *verify its own answer*
with a few-shot self-verification prompt. Because that self-verification is noisy, a meta-verifier
(modelled as a POMDP) decides whether to escalate.

**Uncertainty-based two-tier selection** (Ramírez et al., 2024) shows that a simple signal, the
small model's own output uncertainty, can decide when to call the large model, and that it is
competitive with trained routers across many tasks.

**Relevance to this project.** Our *cascade router* follows this line. It uses the simplest
confidence signal (the small model states a 1–10 confidence), so no extra calls or training are
needed, and our evaluation charges it for the wasted small-model call, as FrugalGPT does.

---

## 4. Learned routers (decide before calling)

A cascade always pays for the cheap attempt and adds its latency. A **router** predicts the right
model from the query alone, before any model is called.

**Hybrid LLM** (Ding et al., ICLR 2024) trains a BERT-style router to predict whether a small model's
answer will be as good as a large model's for a given query. A tunable threshold trades quality for
cost at test time. It reports up to 40% fewer calls to the large model with no drop in response
quality. This is the closest formulation to our *learned router*: a binary "will the small model
suffice?" classifier with a threshold.

**RouteLLM** (Ong et al., 2024) learns routers between one strong and one weak model from **human
preference data** (Chatbot Arena), with data augmentation. It compares several router types
(similarity-weighted ranking, matrix factorisation, a BERT classifier and a causal-LLM classifier)
and reports cost reductions of more than 2× on MT-Bench without losing quality. It also shows that
routers transfer to model pairs they were not trained on. It is released as an open-source framework.

**Fly-swat or cannon?** (Šakota et al., WSDM 2024) uses *meta-modelling*: a model predicts each
LLM's performance on a query from its text, and an optimiser then picks the cheapest model expected
to succeed, under a budget.

**Routing with benchmark datasets** (Shnitzer et al., 2023) learns, from many existing benchmark
datasets, which model is best for which kind of input. It frames routing as a set of binary
"is this model correct?" classifiers, which matches our label definition.

**Tryage** (Hari & Thomson, 2023) is a context-aware router that picks among a library of expert
models, using a perceptive router that predicts downstream performance and respects user
constraints such as model size.

**Zooter** (Lu et al., 2024) distils reward-model scores into a routing function, so that each query
is sent to the expert LLM most likely to answer it well, without running every model at inference.

**LLM-Blender** (Jiang, Ren & Lin, ACL 2023) is the opposite design point. It runs *all* models,
ranks their outputs pairwise (PairRanker) and fuses the best (GenFuser). Quality improves, but the
cost is the sum of all models. It shows the upper end of the quality-cost trade-off that routing
tries to avoid.

**Relevance to this project.** Our *learned router* is deliberately simple: logistic regression on
MiniLM sentence embeddings plus two handcrafted features. It is fast, explainable and trainable on a
few hundred labels, which is what a student project or a department deployment can collect. Hybrid
LLM and RouteLLM suggest a fine-tuned BERT classifier as a natural next step.

---

## 5. Benchmarks for routing

**RouterBench** (Hu et al., 2024) provides a standard benchmark and evaluation framework for
multi-LLM routing. It contains pre-computed outputs of many LLMs on many tasks, so routing
strategies can be compared offline without new API calls. Its key evaluation tool is the
**cost-quality plane**: each strategy is a point or curve of average cost against average quality,
the non-decreasing convex hull of a router's points summarises it, and an **oracle router** gives
the upper bound.

**Relevance to this project.** We follow the same method. `build_labels.py` pre-computes and grades
both models' answers once, and `evaluate.py` replays every strategy offline, plots the
cost-vs-accuracy plane and includes an oracle.

---

## 6. Confidence and self-evaluation

Cascades depend on a model knowing when it might be wrong.

**Language Models (Mostly) Know What They Know** (Kadavath et al., 2022) shows that large models are
reasonably well calibrated on multiple-choice questions and can estimate the probability that their
own answer is correct. That ability improves with scale, which means *small* models are less
reliable judges of themselves.

**Relevance to this project.** We ask a 3B model for a verbal 1–10 confidence. Kadavath et al.
predict this signal will be useful but noisy. Our evaluation measures exactly how useful it is
compared with a router that never calls the small model first.

---

## 7. LLM-as-a-judge

Training a router needs labels saying whether the small model's answer was good enough. Human
grading of hundreds of answers is slow.

**Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena** (Zheng et al., NeurIPS 2023 Datasets &
Benchmarks) shows that strong LLM judges agree with human preferences about as often as humans
agree with each other (over 80%). It also documents biases: **position bias**, **verbosity bias**
and **self-enhancement bias** (judges favour their own answers).

**Relevance to this project.** We use the large model as a judge with a strict YES/NO rubric and
optional human-written reference answers. Self-enhancement bias is a known limitation: when no
reference exists, the large model's answer is treated as correct. Adding reference answers, or
using a different judge model (`JUDGE_MODEL`), reduces this bias.

---

## 8. Retrieval and embeddings

Many chatbot questions ("When is IA-2?") have fixed answers that an LLM cannot know anyway.

**Retrieval-Augmented Generation** (Lewis et al., NeurIPS 2020) combines a retriever over a document
store with a generator, and shows that retrieval improves factual accuracy on knowledge-intensive
tasks.

**Sentence-BERT** (Reimers & Gurevych, EMNLP 2019) produces sentence embeddings that can be compared
with cosine similarity. That makes semantic search over thousands of sentences fast enough for real
time. **MiniLM** (Wang et al., NeurIPS 2020) distils large transformers into small, fast ones. The
`all-MiniLM-L6-v2` model used here comes from that line of work and runs comfortably on a CPU.

**Semantic caching** systems such as GPTCache (Zilliz, 2023, open source) store previous LLM answers
and return them for semantically similar new queries, reducing LLM calls.

**Relevance to this project.** Our retrieval tier is the simplest form of this idea. We embed FAQ
questions, and if a new question's cosine similarity to an FAQ passes a threshold, we return the
stored answer with no LLM call at a cost of zero. The same embeddings are reused as features for
the learned router.

---

## 9. Small open models

**Qwen2.5** (Qwen Team, 2024) is a family of open-weight models from 0.5B to 72B parameters. The 3B
and 14B versions used here are free to run locally through Ollama. Small open models like these
make a *free* small tier practical, which is the basis of this project's cost savings.

---

## 10. Code-mixed and Indian-language NLP

Students in Karnataka often write in **Kanglish**, Kannada written in Latin script and mixed with
English ("Pointer andre enu?").

- **GLUECoS** (Khanuja et al., ACL 2020) is an evaluation benchmark for code-switched NLP
  (Hindi-English, Spanish-English). It shows that models trained on monolingual data drop in
  performance on code-mixed text.
- **DravidianCodeMix** (Chakravarthi et al., 2022) provides code-mixed Tamil-English,
  Malayalam-English and **Kannada-English** social-media datasets for sentiment and offensive
  language detection. It confirms that Kannada-English code-mixing is common and under-resourced.

**Relevance to this project.** None of the routing papers above (FrugalGPT, RouteLLM, Hybrid LLM,
RouterBench) evaluate on code-mixed or Indian-language queries. Our separate Kanglish test set
(`data/queries_kanglish.csv`) measures whether a router trained on English questions still makes
good decisions on code-mixed ones.

---

## 11. Comparison of approaches

| Work | Type | Decision signal | Needs training? | Calls before deciding | Evaluated on |
|---|---|---|---|---|---|
| FrugalGPT (2023) | Cascade | Learned answer scorer | Yes | Cheap model(s) first | Classification and QA APIs |
| MoT cascade (2024) | Cascade | Answer consistency | No | Several weak-model samples | Math and reasoning |
| AutoMix (2024) | Cascade | Self-verification + POMDP | Light | Small model + verification | Context-grounded QA |
| Hybrid LLM (2024) | Router | BERT quality-gap predictor | Yes | None | Instruction following |
| RouteLLM (2024) | Router | Preference-trained classifiers | Yes | None | MT-Bench, MMLU, GSM8K |
| Šakota et al. (2024) | Router | Per-model performance prediction | Yes | None | Many NLP tasks |
| LLM-Blender (2023) | Ensemble | Pairwise ranking of all outputs | Yes | **All** models | Instruction following |
| RouterBench (2024) | Benchmark | — | — | — | Many tasks, many LLMs |
| **This project** | **Retrieval + router + cascade** | **Similarity / embedding classifier / self-confidence** | **Small (logistic regression)** | **0 (router) or 1 (cascade)** | **Engineering student questions, English + Kanglish** |

---

## 12. Research gap and contribution

1. **Three tiers, not two.** Most routing work chooses between LLMs only. A real department chatbot
   also has fixed-answer questions that need *no* LLM. We add a retrieval tier in front of the
   router and measure the whole pipeline.
2. **Fully free stack.** Published routers mostly route between paid APIs. Here both tiers are free
   (a local small model, plus a free-tier API or a local large model). Cost is reported as an
   API-equivalent figure so results stay comparable with the literature.
3. **Like-for-like comparison of the three families.** Rules, a learned router and a confidence
   cascade are compared on the same data, labels and cost-accuracy plane, with an oracle upper bound
   (following RouterBench).
4. **Domain and language.** The test set is engineering-student questions (C, data structures,
   DBMS, OS, CN), including Kanglish code-mixed queries. That domain and language are not covered
   by existing routing benchmarks.
5. **Lightweight and explainable.** The learned router is logistic regression that trains in
   seconds on a few hundred labels and runs on a CPU, which suits a college deployment.

---

## References

1. P. Viola and M. Jones. "Rapid Object Detection using a Boosted Cascade of Simple Features." *CVPR*, 2001.
2. N. Shazeer et al. "Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer." *ICLR*, 2017. [arXiv:1701.06538](https://arxiv.org/abs/1701.06538)
3. L. Chen, M. Zaharia, J. Zou. "FrugalGPT: How to Use Large Language Models While Reducing Cost and Improving Performance." 2023. [arXiv:2305.05176](https://arxiv.org/abs/2305.05176)
4. M. Yue et al. "Large Language Model Cascades with Mixture of Thoughts Representations for Cost-efficient Reasoning." *ICLR*, 2024. [arXiv:2310.03094](https://arxiv.org/abs/2310.03094)
5. P. Aggarwal, A. Madaan et al. "AutoMix: Automatically Mixing Language Models." *NeurIPS*, 2024. [arXiv:2310.12963](https://arxiv.org/abs/2310.12963)
6. D. Ramírez et al. "Optimising Calls to Large Language Models with Uncertainty-Based Two-Tier Selection." 2024. [arXiv:2405.02134](https://arxiv.org/abs/2405.02134)
7. D. Ding et al. "Hybrid LLM: Cost-Efficient and Quality-Aware Query Routing." *ICLR*, 2024. [arXiv:2404.14618](https://arxiv.org/abs/2404.14618)
8. I. Ong et al. "RouteLLM: Learning to Route LLMs with Preference Data." 2024. [arXiv:2406.18665](https://arxiv.org/abs/2406.18665)
9. M. Šakota, M. Peyrard, R. West. "Fly-Swat or Cannon? Cost-Effective Language Model Choice via Meta-Modeling." *WSDM*, 2024. [arXiv:2308.06077](https://arxiv.org/abs/2308.06077)
10. T. Shnitzer et al. "Large Language Model Routing with Benchmark Datasets." 2023. [arXiv:2309.15789](https://arxiv.org/abs/2309.15789)
11. S. N. Hari, M. Thomson. "Tryage: Real-time, Intelligent Routing of User Prompts to Large Language Models." 2023. [arXiv:2308.11601](https://arxiv.org/abs/2308.11601)
12. K. Lu et al. "Routing to the Expert: Efficient Reward-guided Ensemble of Large Language Models." *NAACL*, 2024. [arXiv:2311.08692](https://arxiv.org/abs/2311.08692)
13. D. Jiang, X. Ren, B. Y. Lin. "LLM-Blender: Ensembling Large Language Models with Pairwise Ranking and Generative Fusion." *ACL*, 2023. [arXiv:2306.02561](https://arxiv.org/abs/2306.02561)
14. Q. J. Hu et al. "RouterBench: A Benchmark for Multi-LLM Routing System." 2024. [arXiv:2403.12031](https://arxiv.org/abs/2403.12031)
15. S. Kadavath et al. "Language Models (Mostly) Know What They Know." 2022. [arXiv:2207.05221](https://arxiv.org/abs/2207.05221)
16. L. Zheng et al. "Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena." *NeurIPS Datasets and Benchmarks*, 2023. [arXiv:2306.05685](https://arxiv.org/abs/2306.05685)
17. P. Lewis et al. "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks." *NeurIPS*, 2020. [arXiv:2005.11401](https://arxiv.org/abs/2005.11401)
18. N. Reimers, I. Gurevych. "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks." *EMNLP*, 2019. [arXiv:1908.10084](https://arxiv.org/abs/1908.10084)
19. W. Wang et al. "MiniLM: Deep Self-Attention Distillation for Task-Agnostic Compression of Pre-Trained Transformers." *NeurIPS*, 2020. [arXiv:2002.10957](https://arxiv.org/abs/2002.10957)
20. Zilliz. "GPTCache: A Library for Creating Semantic Cache for LLM Queries." 2023. https://github.com/zilliztech/GPTCache
21. Qwen Team. "Qwen2.5 Technical Report." 2024. [arXiv:2412.15115](https://arxiv.org/abs/2412.15115)
22. S. Khanuja et al. "GLUECoS: An Evaluation Benchmark for Code-Switched NLP." *ACL*, 2020. [arXiv:2004.12376](https://arxiv.org/abs/2004.12376)
23. B. R. Chakravarthi et al. "DravidianCodeMix: Sentiment Analysis and Offensive Language Identification Dataset for Dravidian Languages in Code-Mixed Text." *Language Resources and Evaluation*, 2022. [arXiv:2106.09460](https://arxiv.org/abs/2106.09460)
