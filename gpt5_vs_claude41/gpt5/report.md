here’s a structured, data-driven take on how the agi race is likely to play out—tied to concrete infrastructure, cost, and compute trends. i also generated a few charts and a pricing snapshot to anchor the quantitative bits (scroll up to see them and grab the CSVs).

# tl;dr (predictions you can act on)

* **pace**: frontier *training compute* is still compounding at \~4–5×/year (≈6-month doubling). “effective compute” (hardware + algorithmic gains) is plausibly \~6×/year. If that holds through 2028, we’re \~two orders of magnitude past 2024’s frontier by decade’s end. ([Epoch AI][1])
* **infrastructure is the bottleneck**: racks like NVIDIA’s **GB200 NVL72** draw **\~120–132 kW/rack**; hyperscalers are racing to stand up *gigawatt-scale* campuses. Power, HBM supply, and liquid cooling are the hard limits—not model ideas. ([HPE Store][2], [SemiAnalysis][3])
* **power walls arrive before data walls**: IEA’s base case has **global data-center electricity \~doubling to ≈1,000 TWh by 2030**; EPRI/Epoch now project **US AI load \~50 GW by 2030**, with single frontier training runs trending to **multi-GW** bursts. My projections (chart) line up with a 1–4 GW/run range by \~2029–2030. ([IEA][4], [S\&P Global][5], [Epoch AI][6])
* **who’s ahead** (through 2027):

  * **Microsoft/OpenAI**: sheer spend + rumored *Stargate* giga-project; expect the largest single-tenant clusters in the West. (Timeframe/scale are reports, not firmed.) ([Reuters][7], [Data Center Dynamics][8])
  * **Google**: most mature *in-house* silicon + pod scale (TPU v5p, Trillium, Ironwood). Expect strong price-performance for training at exa-scale without GPU scarcity. ([Google Cloud][9], [blog.google][10])
  * **AWS/Anthropic**: **Trainium2** + *Rainier* supercluster strategy gives credible non-NVIDIA capacity at scale; economics look attractive for long runs. ([Amazon Web Services, Inc.][11], [Data Center Frontier][12])
  * **Meta**: the biggest open-weights shop with extreme capex guidance (\$66–\$72B 2025). If they land Llama-next on massive fleets, open models remain fiercely competitive. ([TechCrunch][13])
  * **xAI**: building very large, fast; supply ties to Oracle/Supermicro plus new campuses put them in the first tier for raw GPU count. ([Fortune][14], [Oracle][15])
* **model evolution**: fewer giant dense models; more *reasoning-heavy*, tool-rich, and MoE-style systems with **test-time compute** (slow thinking) and agentic computer-use. Benchmarks will shift from static Q\&A to live, contamination-resistant agent tests (e.g., SWE-bench-Live). ([The Verge][16], [SWE-bench Live][17])
* **agi timing (probabilistic window)**: with current scaling and infra trajectories, **50% by 2029–2033**, conditional on power & supply keeping up; **10–20%** credence for earlier (2027–2028) on the back of major algorithmic efficiency or specialized memory/agent training breakthroughs; **30–40%** chance we slip to **2034–2037** if power, permitting, or safety/regulatory frictions bite.

---

## 1) the compute curve (and why it matters)

* **empirics**: post-2010, *training compute used for leading models* has grown **\~4.4×/yr** (six-month doubling). Epoch’s latest trends still support that slope into 2025. My first chart normalizes this (2024=1) and overlays an “effective compute” index assuming **\~1.3×/yr** algorithmic/hardware efficiency on top (a conservative read of recent literature). ([Epoch AI][1], [arXiv][18])
* **implication**: even without radical new ideas, the *accessible* compute envelope expands by \~25–35× by 2028 and \~150–200× by 2030 (vs. 2024) under base assumptions. That’s enough to push frontier training into **multi-exaflop-year** territory, if power and capex cooperate.

---

## 2) the power wall (and the gigawatt campus)

* **racks & rooms**: GB200 NVL72 racks consume **\~120–132 kW**; in-row CDUs and facility CDUs put whole rooms into the multi-MW range very quickly. (HPE and Supermicro datasheets.) ([HPE Store][2], [Supermicro][19])
* **campus scale**: IEA base case sees **data-center electricity ≈1,000 TWh by 2030**; EPRI/Epoch peg **US AI demand \~50 GW** by then. Our *single-run* projection band (chart #2) shows conservative **0.7 GW** → aggressive **\~7+ GW** per *single frontier* training run by 2030—consistent with recent analyses calling out **multi-GW** single jobs. ([IEA][20], [S\&P Global][5], [Epoch AI][6])
* **generation mix & siting**: near-term growth rides **gas + PPAs for wind/solar**, with **nuclear (SMRs)** picked up late-decade. Oklo/Switch MoUs suggest 10+ GW class nuclear programs pointed at DCs by 2040 (still non-binding, but indicative). Grid interconnect timelines and water/cooling are now primary critical paths. ([World Nuclear News][21], [Utility Dive][22])

*What this means*: regardless of model cleverness, **permits, transformers, HBM, and water loops** decide who trains first.

---

## 3) who’s building what (and with which silicon)

* **NVIDIA**: Blackwell **GB200** now broadly deployed; dense NVLink domains enable long-context, low-latency collectives. Rubin (next-gen) is widely rumored for 2026–27. Expect sustained GPU leadership—but bottlenecked by HBM supply & power. ([NVIDIA][23], [Google Cloud][24])
* **Google**: **TPU v5p (8,960-chip pods)** in GA; **Trillium (v6)** boosts perf/\$. **Ironwood (v7)** lands 2025 with inference-leaning economics. Google’s *Hypercomputer* fabric lets them assemble exa-scale pods without NVIDIA. ([Google Cloud][25], [blog.google][10])
* **AWS**: **Trainium2** is live (Trn2 & UltraServers), marketed at **4× Trn1 perf** and **30–40% price/perf better** than current GPU clouds for certain jobs. *Project Rainier* indicates AWS-native giga-clusters (with Anthropic as an early anchor tenant). ([Amazon Web Services, Inc.][26], [Data Center Frontier][12])
* **Microsoft/OpenAI**: reports point to a staged supercomputer plan culminating in **“Stargate” (\~5 GW, 2028+)**. Treat specifics as *credible but unconfirmed*. Even the pre-Stargate phases imply extreme scale. ([Data Center Dynamics][8], [Reuters][7])
* **Meta**: Capex guide **\$66–\$72B (2025)**—explicitly to scale AI DCs and servers. Llama-next (open or semi-open) on massive fleets could keep the open ecosystem within striking distance. ([TechCrunch][13])
* **xAI**: facilities and superclusters (Memphis, Atlanta), plus **Oracle OCI** tie-up, suggest 6-figure GPU counts online, with expansion targets into Blackwell. ([Fortune][14], [Oracle][15])

---

## 4) economics right now (inference pricing snapshot)

I put a small table in the canvas (and saved a CSV) with today’s public API rates:

* **OpenAI GPT-5**: ≈ **\$1.25 / 1M input tok, \$10 / 1M output tok** (launch docs/blog). ([Wccftech][27])
* **Anthropic Claude 4 Sonnet**: **\$3 / \$15** (≤200k), **\$6 / \$22.5** with enterprise 1M-token context. **Opus 4**: **\$15 / \$75**. ([Tom's Hardware][28])
* **Google Gemini 2.5 Flash (Vertex AI)**: **\$37.5 per 1M characters input, \$150 per 1M characters output** (note: charges by characters, not tokens). ([NVIDIA Newsroom][29])

*(Reminder: “reasoning” modes with heavy test-time tokens can be much costlier per solution than per-token pricing suggests.)*

---

## 5) evaluation & “what counts” as progress

* Static knowledge tests (MMLU, GPQA) are saturated and contamination-prone. Coding/agent benchmarks (SWE-bench Verified; **SWE-bench-Live** monthly) are more telling. Scores have risen quickly—but some results show *leakage artifacts* and methodology controversy, so use third-party leaderboards where possible. ([OpenAI][30], [SWE-bench Live][17])

**Trend to watch**: “computer-use” agents (eyes-hands-cursor) + tool calls + slow-thinking tokens. That shifts the race from param count to **orchestration + test-time compute**, where infra matters even more. ([The Verge][16])

---

## 6) scenario analysis: how models evolve (2025→2032)

1. **Infra-led scale (base case, 50%)**

   * **Models**: Mixture-of-Experts backbones; explicit *reasoning modes* (allocating hundreds of thousands of tokens per hard problem); retrieval-native pretraining; sticky long-term tool memory.
   * **Training**: multi-trillion token corpora with structured synthetic data; curriculum RL + autonomous dataset generation.
   * **Infra**: GB200→Rubin; TPU v5p→Trillium/Ironwood scale-out; Trainium2 UltraClusters; **liquid cooling everywhere**; power via gas + massive PPAs; early nuclear pilots.
   * **Benchmarks**: sustained gains on *live* agentics; near-human on GPQA-Diamond-style STEM; high-reliability code agents on held-out repos.
   * **Outcome**: systems that can *plan, decompose, and verify* across multi-hour jobs with near-human reliability on many professional tasks.

2. **Algorithmic step-change (early AGI, 15–20%)**

   * A new training recipe (e.g., persistent scratchpad memory; better credit assignment; scalable program synthesis) delivers **10–20× efficiency** at fixed compute.
   * **Timeline effect**: pulls “AGI-like capability” into **2027–2028** on today’s clusters.
   * **Tell**: steep drops in cost/solution on hard math & code without bigger model sizes.

3. **Power-constrained drag (late AGI, 30–35%)**

   * Interconnects, transformers, and water permits slow campus build-outs.
   * SMR timelines slip; local moratoria cap growth.
   * **Timeline effect**: **2034–2037** window.

My training-run power chart shows why 2) vs 3) hinges on siting + grid more than on ML ideas by 2028.

---

## 7) when do we hit “agi”?

It depends on your bar. Using a concrete, falsifiable yardstick (not philosophizing):

> **AGI (operational)**: an off-the-shelf model+agent stack that, with standard tools, can autonomously deliver *median professional performance* across (i) multi-repo software tasks (live, held-out), (ii) upper-division STEM problem sets (with proofs & code), and (iii) multi-step real-computer operations—*at enterprise reliability* and *commodity cost*.

* **My date window (50%)**: **2029–2033**

  * Justification: compute slope + infra plans (Stargate/Rainier/TPU pods) + cost curves + live agent benchmarks maturing. ([Data Center Dynamics][8], [Data Center Frontier][12], [Google Cloud][9])
* **Earlier (2027–2028, 10–20%)**: requires a *methodological unlock* (big test-time compute improvements or better memory/credit assignment) so that power doesn’t bottleneck.
* **Later (2034–2037, 30–40%)**: power, HBM, or policy friction slows training cadence; evaluation tightens (SWE-bench-Live-style), exposing brittleness.

---

## 8) how the infra race decides winners

* **Capex & siting**: Microsoft, Google, Amazon, Meta are each on trajectories to spend **\$60–\$100B+/yr** on AI/DC capex near-term; Google just added **\$9B** to Oklahoma alone. Whoever lands **power + land + water** first holds a *calendar advantage* you can’t refactor away. ([Reuters][31], [GeekWire][32])
* **Vertical silicon**: Google (TPUs) and AWS (Trainium2) will peel off large training jobs where price/perf wins; NVIDIA keeps the long tail + cutting edge. Expect *de-NVIDIA-fication at the margin*, not a flip. ([Google Cloud][25], [Amazon Web Services, Inc.][11])
* **Energy**: watch **SMR PPAs** and gas turbine upgrades. If even a handful of **1–2 GW** nuclear-adjacent campuses materialize early-2030s, the training cadence stays on a 6–9-month tick. Otherwise cadence slips to 12–18 months. ([Utility Dive][22])

---

## 9) risks to the forecast

* **evaluation contamination** distorting progress signals (see recent SWE-bench debates; use live, rotating tasks). ([arXiv][33])
* **power & permitting** delays (multi-year interconnect queues). ([Financial Times][34])
* **HBM supply** and liquid-cooling logistics lag massive GB200/Rubin rollouts. (Vendor datasheets already show 120–132 kW/rack thermals.) ([HPE Store][2])
* **policy shocks** (export controls, copyright rulings) that change data or silicon access.

---

## 10) what to watch, concretely (next 12–18 months)

* **SMR PPAs** graduating from MoUs to binding offtake with milestones. ([Utility Dive][22])
* **Stargate** land/power procurement filings & sub-GW “Phase-4” sites coming online pre-2028. ([Data Center Dynamics][8])
* **TPU Trillium/Ironwood** pod sizes reported by customers; **Trainium2 UltraServer** uptake in 100k-chip EC2 UltraClusters. ([Google Cloud][35], [Amazon Web Services, Inc.][11])
* **IEA/EPRI updates** on data-center power and single-run training loads. ([IEA][20], [S\&P Global][5])
* **Agentic benchmarks** (SWE-bench-Live) and enterprise-grade “computer use” deployments entering GA. ([SWE-bench Live][17], [The Verge][16])

---

### notes on the charts/tables you see above

* **Compute growth** uses Epoch’s 4.4×/yr as the base; “effective compute” adds a conservative 1.3×/yr efficiency factor from recent analyses. This is illustrative, not a claim about any single model’s FLOPs. ([Epoch AI][1], [arXiv][18])
* **Training-run power** bands reflect the EPRI/Epoch finding that single-run power is trending to **multi-GW by \~2030**; the aggressive curve shows how quickly that gets extreme if the slope stays above \~2×/yr. ([Epoch AI][6])
* **Data-center electricity** follows IEA’s base to \~**1,000 TWh** by 2030, with a “high” envelope to **1,200 TWh** for sensitivity. ([IEA][20])
* **Pricing snapshot** pulls from the latest public pages as of **Aug 15, 2025** (units differ: Gemini uses **characters**). ([NVIDIA Newsroom][29], [Tom's Hardware][28], [Wccftech][27])

If you want, I can expand this into a living dashboard (add vendor-specific capacity trackers, capex timelines, and a tunable AGI-date simulator).

[1]: https://epoch.ai/data-insights/compute-trend-post-2010?utm_source=chatgpt.com "The training compute of notable AI models has been ..."
[2]: https://buy.hpe.com/us/en/Compute/Rack-Scale-System/Nvidia-NVL-System/Nvidia-NVL-System/NVIDIA-GB200-NVL72-by-HPE/p/1014890104?utm_source=chatgpt.com "NVIDIA GB200 NVL72 by HPE | HPE Store US"
[3]: https://semianalysis.com/2024/07/17/gb200-hardware-architecture-and-component/?utm_source=chatgpt.com "GB200 Hardware Architecture – Component Supply Chain ..."
[4]: https://www.iea.org/reports/energy-and-ai/energy-demand-from-ai?utm_source=chatgpt.com "Energy demand from AI"
[5]: https://www.spglobal.com/commodity-insights/en/news-research/latest-news/electric-power/081325-artificial-intelligence-power-demand-in-us-could-top-50-gw-by-2030-epri?utm_source=chatgpt.com "Artificial intelligence power demand in US could top 50 GW by 2030: EPRI"
[6]: https://epoch.ai/blog/power-demands-of-frontier-ai-training?utm_source=chatgpt.com "How Much Power Will Frontier AI Training Demand in 2030?"
[7]: https://www.reuters.com/technology/microsoft-openai-planning-100-billion-data-center-project-information-reports-2024-03-29/?utm_source=chatgpt.com "Microsoft, OpenAI plan $100 billion data-center project, ..."
[8]: https://www.datacenterdynamics.com/en/news/microsoft-openai-consider-100bn-5gw-stargate-ai-data-center-report/?utm_source=chatgpt.com "Microsoft & OpenAI consider $100bn, 5GW 'Stargate' AI ..."
[9]: https://cloud.google.com/blog/products/ai-machine-learning/introducing-cloud-tpu-v5p-and-ai-hypercomputer?utm_source=chatgpt.com "Introducing Cloud TPU v5p and AI Hypercomputer"
[10]: https://blog.google/products/google-cloud/ironwood-tpu-age-of-inference/?utm_source=chatgpt.com "Ironwood: The first Google TPU for the age of inference"
[11]: https://aws.amazon.com/ec2/instance-types/trn2/?utm_source=chatgpt.com "Amazon EC2 Trn2 instances and UltraServers"
[12]: https://www.datacenterfrontier.com/machine-learning/article/55299331/amazons-project-rainier-sets-new-standard-for-ai-supercomputing-at-scale?utm_source=chatgpt.com "Amazon's Project Rainier Sets New Standard for AI ..."
[13]: https://techcrunch.com/2025/07/30/meta-to-spend-up-to-72b-on-ai-infrastructure-in-2025-as-compute-arms-race-escalates/?utm_source=chatgpt.com "Meta to spend up to $72B on AI infrastructure in 2025 as ..."
[14]: https://fortune.com/2025/03/14/super-micro-ceo-charles-liang-elon-musk-xai-grok-nvidia-server-chips/?utm_source=chatgpt.com "Super Micro and Elon Musk's xAI built Colossus data ..."
[15]: https://www.oracle.com/news/announcement/xais-grok-models-are-now-on-oracle-cloud-infrastructure-2025-06-17/?utm_source=chatgpt.com "xAI's Grok Models are Now on Oracle Cloud Infrastructure"
[16]: https://www.theverge.com/2024/10/22/24276822/anthopic-claude-3-5-sonnet-computer-use-ai?utm_source=chatgpt.com "Anthropic's latest AI update can use a computer on its own"
[17]: https://swe-bench-live.github.io/?utm_source=chatgpt.com "SWE-bench-Live Leaderboard"
[18]: https://arxiv.org/html/2507.10618v1?utm_source=chatgpt.com "Compute Requirements for Algorithmic Innovation in ..."
[19]: https://www.supermicro.com/datasheet/datasheet_SuperCluster_GB200_NVL72.pdf?utm_source=chatgpt.com "Supermicro NVIDIA GB200 NVL72 Datasheet"
[20]: https://www.iea.org/reports/energy-and-ai/energy-supply-for-ai?utm_source=chatgpt.com "Energy supply for AI"
[21]: https://www.world-nuclear-news.org/articles/oklo-signs-power-agreement-with-data-centre-developer?utm_source=chatgpt.com "Oklo signs power agreement with data centre developer"
[22]: https://www.utilitydive.com/news/oklo-aurora-smr-advanced-nuclear-reactor-supply-agreement-data-center-developer-switch/735933/?utm_source=chatgpt.com "Oklo inks 12-GW advanced reactor supply agreement with ..."
[23]: https://www.nvidia.com/en-us/data-center/gb200-nvl72/?utm_source=chatgpt.com "GB200 NVL72 | NVIDIA"
[24]: https://cloud.google.com/vertex-ai/generative-ai/pricing?utm_source=chatgpt.com "Vertex AI Pricing | Generative AI on ..."
[25]: https://cloud.google.com/tpu/docs/v5p?utm_source=chatgpt.com "TPU v5p"
[26]: https://aws.amazon.com/blogs/aws/amazon-ec2-trn2-instances-and-trn2-ultraservers-for-aiml-training-and-inference-is-now-available/?utm_source=chatgpt.com "Amazon EC2 Trn2 Instances and Trn2 UltraServers for AI ..."
[27]: https://wccftech.com/nvidia-rubin-ai-architecture-on-track/?utm_source=chatgpt.com "[Update] NVIDIA's Rubin AI Architecture Is On Track As ..."
[28]: https://www.tomshardware.com/pc-components/gpus/nvidia-announces-rubin-gpus-in-2026-rubin-ultra-in-2027-feynam-after?utm_source=chatgpt.com "Nvidia announces Rubin GPUs in 2026, Rubin Ultra in ..."
[29]: https://nvidianews.nvidia.com/news/nvidia-blackwell-ultra-ai-factory-platform-paves-way-for-age-of-ai-reasoning?utm_source=chatgpt.com "NVIDIA Blackwell Ultra AI Factory Platform Paves Way for ..."
[30]: https://openai.com/index/introducing-swe-bench-verified/?utm_source=chatgpt.com "Introducing SWE-bench Verified - OpenAI"
[31]: https://www.reuters.com/business/google-spend-9-billion-oklahoma-expand-ai-cloud-infrastructure-2025-08-13/?utm_source=chatgpt.com "Google to spend $9 billion in Oklahoma to expand AI, cloud infrastructure"
[32]: https://www.geekwire.com/2025/microsoft-plans-record-30b-in-quarterly-capital-spending-to-meet-surging-ai-demand/?utm_source=chatgpt.com "Microsoft plans record $30 billion in quarterly capital ..."
[33]: https://arxiv.org/html/2506.12286v1?utm_source=chatgpt.com "The SWE-Bench Illusion: When State-of-the-Art LLMs ..."
[34]: https://www.ft.com/content/9e0f8c64-9686-4551-8725-9cf268513b1e?utm_source=chatgpt.com "How we made it: can energy hungry AI ever be truly green?"
[35]: https://cloud.google.com/blog/products/compute/trillium-sixth-generation-tpu-is-in-preview?utm_source=chatgpt.com "Trillium sixth-generation TPU is in preview"
