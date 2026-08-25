# Simple Mode analysis source dossier

This dossier records the sources that carry methodological or linguistic weight in
`13-simple-mode-analysis-research-plan.md`. It was frozen before outcome modelling.

Status meanings:

- **verified-open** — canonical metadata and the relevant public text were checked.
- **verified-paywalled** — canonical metadata was checked, but no claim is made that closed
  full text was inspected.
- **to-verify** — the exact passage or a stable edition-specific identifier remains unresolved.

## Dimension A — illocution and clause type

- **Searle (1969), _Speech Acts_.** DOI
  [10.1017/CBO9781139173438](https://doi.org/10.1017/CBO9781139173438);
  user-supplied book. Pages 30–31 distinguish illocutionary force from propositional content
  and list word order, stress, intonation, punctuation, verb mood, and performative verbs as
  force-indicating devices. **verified; human-confirmed**.
- **Searle (1976), “A Classification of Illocutionary Acts.”** DOI
  [10.1017/S0047404500006837](https://doi.org/10.1017/S0047404500006837).
  User-supplied article. Pages 1, 3, and 11 define the five illocutionary classes and define
  directives as attempts by the speaker to get the hearer to act. This anchors directive
  theory, not the grammatical clause taxonomy. **verified; human-confirmed**.
- **Sadock & Zwicky (1985), “Speech Act Distinctions in Syntax,” in Shopen,
  _Language Typology and Syntactic Description_, vol. I, ch. 6.**
  [Author PDF](https://web.stanford.edu/~zwicky/speech-act-distinctions.pdf).
  §§1.1 and 2.1, pp. 155–160, distinguish sentence type from communicative use, distinguish
  yes-no from information questions, and identify declarative, interrogative, and imperative
  as the three frequent basic sentence types. These passages anchor the core clause inventory
  but not the project’s exact UD classifier. **verified-open; human-confirmed**.
- **Huddleston & Pullum (2002), _The Cambridge Grammar of the English Language_.**
  ISBN 9780521431460; user-supplied book. Chapter 10, pp. 854, 856, 858, 929, 939, and
  941, distinguishes clause form from use, gives separate open/closed interrogative syntax,
  and documents imperative, interrogative, and declarative directives. This is the primary
  source for the four project clause labels. **verified; human-confirmed**.
- **Portner (2018), _Mood_.** DOI
  [10.1093/oso/9780199547524.001.0001](https://doi.org/10.1093/oso/9780199547524.001.0001).
  User-supplied book. Chapter 3, pp. 121, 123, and 127, distinguishes clause type,
  sentence mood, and sentential force and recognizes declarative, interrogative, and
  imperative as basic types. **verified; human-confirmed**.

## Dimension B — answerhood, exhaustivity, and presupposition

- **Dayal (2016), _Questions_.** DOI
  [10.1093/acprof:oso/9780199281268.001.0001](https://doi.org/10.1093/acprof:oso/9780199281268.001.0001).
  User-supplied book. Chapters 1 and 3, pp. 16, 57, 67, and 82–87, define mention-some,
  weakly exhaustive, and strongly exhaustive answers and use NPI licensing as evidence for a
  grammatical strong-versus-weak/non-exhaustive distinction. Dayal also records caveats on
  negation diagnostics. This does not make the presence of *any* an automatic
  `negative_conclusiveness` label. **verified; human-confirmed**.
- **Groenendijk & Stokhof (1984), _Studies on the Semantics of Questions and the
  Pragmatics of Answers_.**
  [Part I](https://pure.uva.nl/ws/files/1989717/27444_Proefschrift_001_257.PDF);
  [Part II](https://pure.uva.nl/ws/files/1989719/27445_Proefschrift_258_577.PDF).
  Part I pp. 214–216 represents questions as partitions whose cells are possible semantic
  answers. Part II p. 278 contrasts implicitly exhaustive answers with contextually
  non-exhaustive answers; pp. 394–395 names mention-some and mention-all interrogatives.
  This supports partition semantics and the mention-some/mention-all distinction, but not the
  project’s separate `weakly_exhaustive` level. **verified-open; human-confirmed**.
- **Karttunen (1977), “Syntax and Semantics of Questions.”** DOI
  [10.1007/BF00351935](https://doi.org/10.1007/BF00351935);
  user-supplied article. Pages 7–10 model questions as sets of true-answer propositions while
  treating less-than-exhaustive direct answers as context-sensitive pragmatics rather than a
  strict semantic ambiguity. **verified; human-confirmed**.
- **Beck & Rullmann (1999), “A Flexible Approach to Exhaustivity in Questions.”**
  DOI [10.1023/A:1008373224343](https://doi.org/10.1023/A:1008373224343).
  User-supplied article. Pages 249–250 present a Karttunen-based semantics with separate
  weakly and strongly exhaustive answerhood notions and motivate retaining a flexible
  question-denotation system. **verified; human-confirmed**.
- **George (2011), _Question Embedding and the Semantics of Answers_.**
  [UCLA dissertation PDF](https://linguistics.ucla.edu/wp-content/uploads/2021/11/old_brgeorge_dissertation_web_june2011.pdf).
  The abstract and Chapter 1, pp. 13–18, define mention-some, weakly exhaustive, and strongly
  exhaustive answers and motivate their differences. George also questions whether weak
  exhaustivity is independently required in every theory, so the source supports retaining the
  category as an annotation possibility rather than assuming its necessity.
  **verified-open; human-confirmed**.
- **Karttunen (1971), “Some Observations on Factivity.”** DOI
  [10.1080/08351817109370248](https://doi.org/10.1080/08351817109370248);
  user-supplied article. Pages 55–56 describe the standard factive presupposition account and
  immediately caution that verb identity alone does not determine whether a complement is
  presupposed. This supports contextual `presupposition_load`, not a verb-list rule.
  **verified; human-confirmed**.
- **Roberts (2012), “Information Structure in Discourse.”** DOI
  [10.3765/sp.5.6](https://doi.org/10.3765/sp.5.6);
  user-supplied article. Page 6 defines relevant alternatives by the question/topic under
  discussion. This provides a QUD-based contextual-relevance anchor.
  **verified; human-confirmed**.
- **Ginzburg (2012), _The Interactive Stance_.** DOI
  [10.1093/acprof:oso/9780199697922.001.0001](https://doi.org/10.1093/acprof:oso/9780199697922.001.0001).
  User-supplied full book. Chapters 2–4, pp. 18, 20, 32, and 66, define QUD as the current
  question/topic under discussion, describe questions as conditioning what can be said and how,
  and define a dialogue gameboard containing `FACTS`, `LatestMove`, and a partially ordered
  `QUD`. This anchors dialogue context and interaction-sensitive answer relevance, not the
  exhaustivity label inventory. **verified; human-confirmed**.

## Dimension C — retrieval decomposition

- **Wolfson et al. (2020), “Break It Down.”** DOI
  [10.1162/tacl_a_00309](https://doi.org/10.1162/tacl_a_00309);
  user-supplied article. Section 2, p. 185 defines QDMR as a sequence of single-query steps.
  **verified; human-confirmed**.
- **Yang et al. (2018), HotpotQA.** DOI
  [10.18653/v1/D18-1259](https://doi.org/10.18653/v1/D18-1259);
  user-supplied article. Section 2, pp. 2370–2371 distinguishes bridge entities and comparison
  questions. **verified; human-confirmed**.
- **Ho et al. (2020), 2WikiMultiHopQA.** DOI
  [10.18653/v1/2020.coling-main.580](https://doi.org/10.18653/v1/2020.coling-main.580);
  user-supplied article. Section 2.2, p. 6611 defines comparison, inference, compositional,
  and bridge-comparison question types. **verified; human-confirmed**.
- **Trivedi et al. (2022), MuSiQue.** DOI
  [10.1162/tacl_a_00475](https://doi.org/10.1162/tacl_a_00475). Sections 2–3 anchor
  connected single-hop composition and shortcut filtering. Pages 541 and 543 define critical
  predecessor dependence and identify disconnected shortcut-solvable edges.
  **verified; human-confirmed**.
- **Jeong et al. (2024), Adaptive-RAG.** DOI
  [10.18653/v1/2024.naacl-long.389](https://doi.org/10.18653/v1/2024.naacl-long.389).
  User-supplied article. Figure 2, p. 7037, describes query-complexity routing among iterative,
  single, and no-retrieval strategies. **verified; human-confirmed**.
- **Li & Roth (2002), “Learning Question Classifiers.”** DOI
  [10.3115/1072228.1072378](https://doi.org/10.3115/1072228.1072378);
  user-supplied article. Section 2.2 defines a two-level hierarchy of 6 coarse and 50 fine
  question/answer classes. **verified; human-confirmed**.

## Dimension D — reference and lexical anchoring

- **Gundel, Hedberg & Zacharski (1993), “Cognitive Status and the Form of Referring
  Expressions.”** DOI [10.2307/416535](https://doi.org/10.2307/416535);
  user-supplied article. Pages 274–277 introduce six implicationally related cognitive
  statuses and map referring forms to those statuses. This anchors the general relation between
  form and identifiability, not the project’s full-name/alias/email labels.
  **verified; human-confirmed**.

## Dimension E — morphosyntactic complexity

- **de Marneffe et al. (2021), “Universal Dependencies.”** DOI
  [10.1162/coli_a_00402](https://doi.org/10.1162/coli_a_00402);
  user-supplied article. Sections 2–4 define UD relations, morphological features, and tree
  structure. **verified; human-confirmed**.
- **Lu (2010), “Automatic Analysis of Syntactic Complexity in Second Language Writing.”**
  DOI [10.1075/ijcl.15.4.02lu](https://doi.org/10.1075/ijcl.15.4.02lu);
  user-supplied article. Sections 2–3 and the measure definitions anchor clauses, dependent
  clauses, complex nominals, DC/C, and CN/C. The project’s UD rules are analogous
  operationalizations, not a faithful L2SCA reimplementation. **verified; human-confirmed**.
- **Kyle (2016), TAASSC dissertation.**
  [Georgia State full text](https://scholarworks.gsu.edu/alesl_diss/35/);
  user-supplied dissertation. The tooling chapters support dependency-based fine-grained
  indices, but not the project’s exact UD relation sets. **verified; human-confirmed**.
- **Kyle & Crossley (2018), “Measuring Syntactic Complexity in L2 Writing.”** DOI
  [10.1111/modl.12468](https://doi.org/10.1111/modl.12468);
  user-supplied article. The abstract and pp. 1–2 motivate fine-grained clausal and phrasal
  indices because broad measures do not identify the structures driving complexity. The
  evidence comes from TOEFL essays and does not validate this project’s exact UD rules or
  short-question setting. **verified; human-confirmed**.
- **Petrov, Das & McDonald (2012), “A Universal Part-of-Speech Tagset.”**
  [Open text](https://aclanthology.org/L12-1115/); user-supplied article. The universal
  inventory supports coarse grammatical categories, but not the project’s clause or
  complex-nominal rules. **verified; human-confirmed**.
- **Gibson (1998), “Linguistic Complexity: Locality of Syntactic Dependencies.”** DOI
  [10.1016/S0010-0277(98)00034-1](https://doi.org/10.1016/S0010-0277(98)00034-1);
  user-supplied local PDF. The abstract and §2.2, pp. 1, 8, and 11–13, state that longer
  head-dependent integrations require greater resources. Gibson’s formal `I(n)` counts
  intervening discourse referents, so this source supplies theoretical locality context rather
  than the project’s exact mean token-distance definition. **verified; human-confirmed**.
- **Futrell, Mahowald & Gibson (2015), dependency-length minimization.** DOI
  [10.1073/pnas.1502134112](https://doi.org/10.1073/pnas.1502134112);
  user-supplied article. It defines arc length by word distance, while the project’s mean
  over selected arcs is its own aggregation. **verified; human-confirmed**.
- **Yngve (1960), “A Model and an Hypothesis for Language Structure.”**
  [JSTOR 985230](https://www.jstor.org/stable/985230); user-supplied article. Its
  temporary-storage depth for left-to-right constituency generation is not equivalent to
  the project’s maximum UD root-to-token arc count. **verified; human-confirmed**.

The codebook explicitly labels its UD clause/CN definitions as operationalizations; it does
not claim byte-for-byte equivalence to constituency-based L2SCA.

## Dimension F — exploratory lexical features

- **Brysbaert & New (2009), SUBTLEX-US.** DOI
  [10.3758/BRM.41.4.977](https://doi.org/10.3758/BRM.41.4.977). Anchors mean log
  lexical frequency. **verified-open**.
- **Hale (2001), probabilistic Earley parsing.** DOI
  [10.3115/1073336.1073357](https://doi.org/10.3115/1073336.1073357), and
  **Levy (2008), expectation-based comprehension.** DOI
  [10.1016/j.cognition.2007.05.006](https://doi.org/10.1016/j.cognition.2007.05.006).
  These define and interpret surprisal. **verified-open**.
- **McCarthy & Jarvis (2010), “MTLD, vocd-D, and HD-D.”** DOI
  [10.3758/BRM.42.2.381](https://doi.org/10.3758/BRM.42.2.381);
  user-supplied article. The abstract reports strong MTLD validity and no observed text-length
  effect, while p. 384 states that shorter texts are more difficult to evaluate confidently and
  reports `100` tokens as the shortest development-tested length. This supports excluding MTLD
  from roughly 15-token questions without claiming general unreliability.
  **verified; human-confirmed**.
- **Flesch (1948), “A New Readability Yardstick.”** DOI
  [10.1037/h0057532](https://doi.org/10.1037/h0057532);
  user-supplied article. Pages 221–223 describe applications to documents, a 100-word sample,
  and reading-test passages. **Kincaid et al. (1975), DTIC ADA006655**
  ([open report](https://apps.dtic.mil/sti/citations/ADA006655)) likewise uses passages.
  Neither validates readability formulas for approximately 15-token questions; these sources
  anchor a scope exclusion only. **verified; Flesch human-confirmed**.
- **Lee, Jang & Lee (2021), LingFeat.** DOI
  [10.18653/v1/2021.emnlp-main.834](https://doi.org/10.18653/v1/2021.emnlp-main.834).
  The feature inventory is exploratory only. **verified-open**.

## Dimension G — task and intent framing

- **Oard & Webber (2013), “Information Retrieval for E-Discovery.”** DOI
  [10.1561/1500000025](https://doi.org/10.1561/1500000025);
  [author PDF](https://user.eng.umd.edu/~oard/pdf/fntir13.pdf). Workflow, review, and
  evaluation sections anchor high-recall orientation. **verified-open**.
- **Broder (2002), “A Taxonomy of Web Search.”** DOI
  [10.1145/792550.792552](https://doi.org/10.1145/792550.792552), and
  **Rose & Levinson (2004), “Understanding User Goals in Web Search.”** DOI
  [10.1145/988672.988675](https://doi.org/10.1145/988672.988675). Intent taxonomies
  anchor precision-oriented known-item versus recall-oriented review. **verified-open**.
- **Ingwersen & Järvelin (2005), _The Turn_.** DOI
  [10.1007/1-4020-3851-8](https://doi.org/10.1007/1-4020-3851-8);
  user-supplied book. Pages 3 and 19–20 argue that information seeking/retrieval must be
  studied in context and distinguish work tasks from search tasks. This anchors task/context
  framing but not the binary `recall_orientation` coding. **verified; human-confirmed**.
- **Belkin, Oddy & Brooks (1982), “ASK for Information Retrieval: Part I. Background and
  Theory.”** DOI [10.1108/eb026722](https://doi.org/10.1108/eb026722);
  user-supplied repository PDF. The abstract and introduction, p. 61, state that information
  needs are not in principle precisely specifiable and define an ASK as a recognized anomaly
  in the user’s state of knowledge that the user cannot precisely specify how to resolve.
  This anchors open-ended information-need context but not the binary `recall_orientation`
  coding. **verified-open; human-confirmed**.
- **Anderson & Krathwohl (2001), revised educational taxonomy.** ISBN 9780801319037.
  User-supplied book. Pages 5 and 30–31 define
  `remember < understand < apply < analyze < evaluate < create` along an assumed
  cognitive-complexity continuum. This supports treating `cognitive_process_level` as ordinal,
  activating the later ordinal-validation branch. **verified; human-confirmed**.
- **Graesser & Person (1994), “Question Asking During Tutoring.”** DOI
  [10.3102/00028312031001104](https://doi.org/10.3102/00028312031001104). Table 1's
  content categories and classification dimensions provide supporting taxonomy context;
  they do not add labels to the frozen codebook. **verified-open**.

## Dimension H — evidence demand

Dimension H is anchored in source semantics rather than an external linguistic theory.
At `r1-evals-new` revision
[`41978dbd459ec3dfed9adcf3cc3695aae31a1551`](https://github.com/relativityone/r1-evals/commit/41978dbd459ec3dfed9adcf3cc3695aae31a1551)
(human-confirmed):

- `src/r1_evals/rubrics/models_v2.py` defines `document_ids` as optional supporting-document
  metadata. A revision-scoped source search finds direct references only in models and parsers;
  no prompt, grader, or scorer directly names the field. **verified-local**.
- `src/r1_evals/rubrics/rubric_data/air_assist/mallinckrodt/rubrics_for_ga/4-03.rubric.toml`
  assigns 21, 26, and 28 IDs to three individual factual expectations. This confirms high
  multiplicity, but does not prove independent attestation or conjunctive/disjunctive semantics.
  **verified-local**.

Accordingly, `expectation_count` is the demand measure and `expectation_document_count` is a
two-sided, potentially confounded redundancy proxy.

## Annotation and measurement validation

- **Artstein & Poesio (2008), “Inter-Coder Agreement for Computational Linguistics.”**
  DOI [10.1162/coli.07-034-R2](https://doi.org/10.1162/coli.07-034-R2);
  [open text](https://aclanthology.org/J08-4004/). Sections 2–4 distinguish reliability
  from validity and match agreement coefficients to scale. **verified-open**.
- **Krippendorff (2019), _Content Analysis_, 4th ed.** DOI
  [10.4135/9781071878781](https://doi.org/10.4135/9781071878781);
  ISBN 9781506395661; user-supplied book. Pages 5, 88, and 90 support explicit coding
  instructions and replicability; p. 291 motivates alpha as a flexible agreement measure.
  **verified; human-confirmed**.
- **Zapf et al. (2016), nominal inter-rater reliability.** DOI
  [10.1186/s12874-016-0200-9](https://doi.org/10.1186/s12874-016-0200-9);
  [open text](https://pmc.ncbi.nlm.nih.gov/articles/PMC4974794/). Coefficient and interval
  choice depend on design and scale. **verified-open**.
- **Cohen (1960), “A Coefficient of Agreement for Nominal Scales.”** DOI
  [10.1177/001316446002000104](https://doi.org/10.1177/001316446002000104);
  user-supplied article. Pages 37–39 define chance-corrected nominal two-rater agreement.
  **Cohen (1968), “Weighted Kappa.”** DOI
  [10.1037/h0026256](https://doi.org/10.1037/h0026256); user-supplied article. Pages 213 and
  215 define chance-corrected weighted agreement and require weights to be set before data
  collection. These sources support the fixed ordinal metric, while alpha remains the primary
  reliability coefficient. **verified; human-confirmed**.
- **Gilardi, Alizadeh & Kubli (2023), LLM text annotation.** DOI
  [10.1073/pnas.2305016120](https://doi.org/10.1073/pnas.2305016120). Human-labelled
  evaluation supports testing LLM annotators but not treating inter-LLM agreement as validity.
  **verified-open**.
- **Egami, Hinck, Stewart & Wei (2023), “Using Imperfect Surrogates for Downstream
  Inference.”** [arXiv:2306.04746](https://arxiv.org/abs/2306.04746);
  [NeurIPS paper](https://proceedings.neurips.cc/paper_files/paper/2023/hash/d862f7f5445255090de13b825b880d59-Abstract-Conference.html).
  Naive predicted-label substitution can bias downstream estimators. The paper does not
  validate the plan’s two-way-clustered extension; the separate DSL derivation gate remains
  mandatory. **verified-open**.
- **Horvitz & Thompson (1952), unequal-probability sampling.** DOI
  [10.2307/2280784](https://doi.org/10.2307/2280784). The inverse-inclusion-probability
  estimator anchors weighted gold confusion cells. **verified-open**.
- **Mashreghi, Haziza & Léger (2016), finite-population bootstrap survey.** DOI
  [10.1214/16-SS113](https://doi.org/10.1214/16-SS113). Stratified resampling and
  percentile-interval guidance supports the frozen gold bootstrap. **verified-open**.
- **Brodersen et al. (2010), balanced accuracy.** DOI
  [10.1109/ICPR.2010.764](https://doi.org/10.1109/ICPR.2010.764). The class-balanced
  discrimination measure anchors the binary validity metric. **verified-open**.

## Statistical sources that carry the frozen analysis

- **Efron & Morris (1975), “Data Analysis Using Stein’s Estimator and Its
  Generalizations.”** DOI
  [10.1080/01621459.1975.10479864](https://doi.org/10.1080/01621459.1975.10479864);
  user-supplied article. The abstract and pp. 311–312 describe James–Stein estimation in an
  empirical-Bayes context and shrinking related estimates toward a shared mean. This is general
  shrinkage context, not validation of the exact Layer 1 model. **verified; human-confirmed**.
- **Gelman & Hill (2007), _Data Analysis Using Regression and Multilevel/Hierarchical
  Models_.** DOI
  [10.1017/CBO9780511790942](https://doi.org/10.1017/CBO9780511790942);
  ISBN 9780521686891; user-supplied book. Pages 7, 251, and 253–254 contrast complete pooling
  and no pooling, motivate partial pooling, model group coefficients jointly, and explain how
  small groups shrink more toward the overall mean. **verified; human-confirmed**.
- **Liang & Zeger (1986), GEE.** DOI
  [10.1093/biomet/73.1.13](https://doi.org/10.1093/biomet/73.1.13), and
  **Cameron, Gelbach & Miller (2011), multiway clustering.** DOI
  [10.1198/jbes.2010.07136](https://doi.org/10.1198/jbes.2010.07136). These anchor
  estimating equations and the inclusion–exclusion covariance. **verified-open metadata**;
  CGM full text was checked.
- **Kline & Santos (2012), score-based wild bootstrap.** DOI
  [10.1515/2156-6674.1006](https://doi.org/10.1515/2156-6674.1006);
  [author PDF](https://eml.berkeley.edu/~pkline/papers/ScoreFinal_web.pdf). Score
  perturbation supports a bootstrap without invalid Bernoulli responses. **verified-open**.
- **MacKinnon, Nielsen & Webb (2021), “Wild Bootstrap and Asymptotic Inference with
  Multiway Clustering.”** DOI
  [10.1080/07350015.2019.1677473](https://doi.org/10.1080/07350015.2019.1677473);
  Queen’s Working Paper 1415. The three-term CRVE and arm-clustered restricted WCR pairing
  are used exactly as qualified in §5.3. **verified-open**.
- **Davidson & Flachaire (2008), “The Wild Bootstrap, Tamed at Last.”** DOI
  [10.1016/j.jeconom.2008.08.003](https://doi.org/10.1016/j.jeconom.2008.08.003);
  user-supplied article. **Cameron, Gelbach & Miller (2008), “Bootstrap-Based Improvements
  for Inference with Clustered Errors.”** DOI
  [10.1162/rest.90.3.414](https://doi.org/10.1162/rest.90.3.414); user-supplied article.
  Their abstracts, p. 162 and p. 414, respectively provide heteroskedastic linear-regression
  and few-cluster bootstrap background. They do not validate the project’s exact restricted
  multiway GLM score-bootstrap. **verified; human-confirmed with scope limits**.
- **Benjamini & Hochberg (1995), FDR step-up.** DOI
  [10.1111/j.2517-6161.1995.tb02031.x](https://doi.org/10.1111/j.2517-6161.1995.tb02031.x).
  The exact step-up rule is used with the dependence qualification stated in the plan.
  **verified-open**.
- **Benjamini & Yekutieli (2001), FDR under dependency.** DOI
  [10.1214/aos/1013699998](https://doi.org/10.1214/aos/1013699998). The PRDS and
  arbitrary-dependence results anchor the plan's limitation on the classical BH guarantee.
  **verified-open**.
- **Mundlak (1978), pooling time series and cross-sections.**
  [JSTOR 1913646](https://www.jstor.org/stable/1913646). Cluster means anchor the frozen
  within/between decomposition. **verified-open**.
- **Phipson & Smyth (2010), finite Monte Carlo p-values.** DOI
  [10.2202/1544-6115.1585](https://doi.org/10.2202/1544-6115.1585). Including the
  observed arrangement anchors the sampled-regime `+1` formula. **verified-open**.
- **McCullagh (1980), “Regression Models for Ordinal Data.”** DOI
  [10.1111/j.2517-6161.1980.tb01109.x](https://doi.org/10.1111/j.2517-6161.1980.tb01109.x);
  user-supplied article. Pages 109–110 develop ordinal-response models without assuming
  cardinal scores and identify proportional odds as a practical model.
  **Agresti (2010), _Analysis of Ordinal Categorical Data_, 2nd ed.** DOI
  [10.1002/9780470594001](https://doi.org/10.1002/9780470594001);
  user-supplied book. Pages 10, 44, 47–48, and 58 develop cumulative logits and the
  proportional-odds restriction. **Liddell & Kruschke (2018).** DOI
  [10.1016/j.jesp.2018.08.009](https://doi.org/10.1016/j.jesp.2018.08.009).
  These anchor grade-model ordering and ordinal interpretation. **verified; human-confirmed**
  for McCullagh/Agresti and **verified-open** for Liddell–Kruschke.
- **Cameron & Trivedi (2005), _Microeconometrics: Methods and Applications_.** DOI
  [10.1017/CBO9780511811241](https://doi.org/10.1017/CBO9780511811241);
  user-supplied book. The overview and Chapter 24, pp. xii and 813–814, identify nonlinear
  models, robust inference, bootstrap methods, and clustered samples, and explain why
  independence assumptions fail within clusters. This is general implementation context only,
  not validation of the project’s exact multiway covariance/bootstrap. **verified; human-confirmed**.

## Open verification queue

The unresolved items are deliberately visible rather than promoted to verified:

1. Keep Graesser & Person as supporting context; do not use it to add labels.
2. Preserve the explicit qualification that neither MacKinnon–Nielsen–Webb nor
   Kline–Santos establishes the complete two-way-clustered GLM procedure end to end.
