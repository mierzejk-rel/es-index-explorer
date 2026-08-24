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
  [10.1017/CBO9781139173438](https://doi.org/10.1017/CBO9781139173438). Chapters 2–3
  separate propositional content from illocutionary force. **verified-paywalled**.
- **Searle (1976), “A Classification of Illocutionary Acts.”** DOI
  [10.1017/S0047404500006837](https://doi.org/10.1017/S0047404500006837).
  The directive class anchors requests that are not surface interrogatives.
  **verified-paywalled**.
- **Sadock & Zwicky (1985), “Speech Act Distinctions in Syntax,” in Shopen,
  _Language Typology and Syntactic Description_, vol. I, ch. 6.**
  [Author PDF](https://web.stanford.edu/~zwicky/speech-act-distinctions.pdf).
  §§1.1 and 2.1, pp. 155–160, distinguish sentence type from communicative use, distinguish
  yes-no from information questions, and identify declarative, interrogative, and imperative
  as the three frequent basic sentence types. These passages anchor the core clause inventory
  but not the project’s exact UD classifier. **verified-open; human-confirmed**.
- **Huddleston & Pullum (2002), _The Cambridge Grammar of the English Language_.**
  ISBN 9780521431460. Chapter 10 distinguishes open and closed interrogatives,
  declaratives, and imperatives while separating clause type from force.
  **verified-paywalled**.
- **Portner (2018), _Mood_.** DOI
  [10.1093/oso/9780199547524.001.0001](https://doi.org/10.1093/oso/9780199547524.001.0001).
  The sentence-mood and imperative chapters support using `Mood=Imp` as evidence rather
  than equating morphology with force. **verified-paywalled**.

## Dimension B — answerhood, exhaustivity, and presupposition

- **Dayal (2016), _Questions_.** DOI
  [10.1093/acprof:oso/9780199281268.001.0001](https://doi.org/10.1093/acprof:oso/9780199281268.001.0001).
  Chapters 2–3 support mention-some, weakly exhaustive, and mention-all distinctions and
  the use of negative-polarity forms as diagnostics. **verified-paywalled**.
- **Groenendijk & Stokhof (1984), _Studies on the Semantics of Questions and the
  Pragmatics of Answers_.**
  [Part I](https://pure.uva.nl/ws/files/1989717/27444_Proefschrift_001_257.PDF);
  [Part II](https://pure.uva.nl/ws/files/1989719/27445_Proefschrift_258_577.PDF).
  Part I pp. 214–216 represents questions as partitions whose cells are possible semantic
  answers. Part II p. 278 contrasts implicitly exhaustive answers with contextually
  non-exhaustive answers; pp. 394–395 names mention-some and mention-all interrogatives.
  This supports partition semantics and the mention-some/mention-all distinction, but not the
  project’s separate `weakly_exhaustive` level. **verified-open; human-confirmed**.
- **Hamblin (1973), “Questions in Montague English.”** Original record
  [JSTOR 25000703](https://www.jstor.org/stable/25000703); later reprint DOI
  [10.1007/978-94-010-2506-5_9](https://doi.org/10.1007/978-94-010-2506-5_9).
  Questions as sets of possible answers anchor alternative semantics.
  **verified-paywalled**.
- **Karttunen (1977), “Syntax and Semantics of Questions.”** DOI
  [10.1007/BF00351935](https://doi.org/10.1007/BF00351935). True complete and partial
  answers anchor the answerhood code. **verified-paywalled**.
- **Beck & Rullmann (1999), “A Flexible Approach to Exhaustivity in Questions.”**
  DOI [10.1023/A:1008373224343](https://doi.org/10.1023/A:1008373224343).
  The construction-sensitive treatment motivates separate weak and strong exhaustivity
  labels. **verified-paywalled**.
- **George (2011), _Question Embedding and the Semantics of Answers_.**
  [UCLA dissertation PDF](https://linguistics.ucla.edu/wp-content/uploads/2021/11/old_brgeorge_dissertation_web_june2011.pdf).
  The abstract and Chapter 1, pp. 13–18, define mention-some, weakly exhaustive, and strongly
  exhaustive answers and motivate their differences. George also questions whether weak
  exhaustivity is independently required in every theory, so the source supports retaining the
  category as an annotation possibility rather than assuming its necessity.
  **verified-open; human-confirmed**.
- **Karttunen (1971), “Some Observations on Factivity.”** DOI
  [10.1080/08351817109370248](https://doi.org/10.1080/08351817109370248), and
  **Kiparsky & Kiparsky (1970), “Fact.”** DOI
  [10.1515/9783111350219.143](https://doi.org/10.1515/9783111350219.143).
  Factive predicates anchor `presupposition_load`. **verified-paywalled**.
- **Roberts (2012), “Information Structure in Discourse.”** DOI
  [10.3765/sp.5.6](https://doi.org/10.3765/sp.5.6). Sections 2–3 provide the
  question-under-discussion account of contextual answer relevance. **verified-open**.
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
  [open text](https://aclanthology.org/2020.tacl-1.13/). Sections 2–3 define QDMR steps and
  operators. **verified-open**.
- **Yang et al. (2018), HotpotQA.** DOI
  [10.18653/v1/D18-1259](https://doi.org/10.18653/v1/D18-1259). Sections 2–3 anchor bridge
  and comparison questions. **verified-open**.
- **Ho et al. (2020), 2WikiMultiHopQA.** DOI
  [10.18653/v1/2020.coling-main.580](https://doi.org/10.18653/v1/2020.coling-main.580).
  Sections 2–3 support compositional and comparison structures. **verified-open**.
- **Trivedi et al. (2022), MuSiQue.** DOI
  [10.1162/tacl_a_00475](https://doi.org/10.1162/tacl_a_00475). Sections 2–3 anchor
  connected single-hop composition and shortcut filtering. **verified-open**.
- **Jeong et al. (2024), Adaptive-RAG.** DOI
  [10.18653/v1/2024.naacl-long.389](https://doi.org/10.18653/v1/2024.naacl-long.389).
  Sections 2–3 support complexity-sensitive routing. **verified-open**.
- **Li & Roth (2002), “Learning Question Classifiers.”** DOI
  [10.3115/1072228.1072378](https://doi.org/10.3115/1072228.1072378);
  [open text](https://aclanthology.org/C02-1150/). The coarse/fine hierarchy anchors
  question-type controls. **verified-open**.

## Dimension D — reference and lexical anchoring

- **Gundel, Hedberg & Zacharski (1993), “Cognitive Status and the Form of Referring
  Expressions.”** DOI [10.2307/416535](https://doi.org/10.2307/416535). The Givenness
  Hierarchy anchors form and identifiability. **verified-paywalled**.
- **Ariel (1990), _Accessing Noun-Phrase Antecedents_.** ISBN 9780415055932. The
  accessibility hierarchy supports contrasts among conventional names, aliases, and
  maximally identifying forms. **verified-paywalled**.

## Dimension E — morphosyntactic complexity

- **de Marneffe et al. (2021), “Universal Dependencies.”** DOI
  [10.1162/coli_a_00402](https://doi.org/10.1162/coli_a_00402). Sections 2–4 define
  UD relations, morphological features, and tree structure. **verified-open**.
- **Lu (2010), “Automatic Analysis of Syntactic Complexity in Second Language Writing.”**
  DOI [10.1075/ijcl.15.4.02lu](https://doi.org/10.1075/ijcl.15.4.02lu). Sections 2–3
  and the measure definitions anchor clauses, dependent clauses, complex nominals, DC/C,
  and CN/C. **verified-open**.
- **Kyle (2016), TAASSC dissertation.**
  [Georgia State full text](https://scholarworks.gsu.edu/alesl_diss/35/). The tooling
  chapters support dependency-based fine-grained indices. **verified-open**.
- **Kyle & Crossley (2018), “Measuring Syntactic Complexity in L2 Writing.”** DOI
  [10.1111/modl.12468](https://doi.org/10.1111/modl.12468). Fine-grained clausal and
  phrasal indices motivate reporting separate dimensions. **verified-paywalled**.
- **Petrov, Das & McDonald (2012), “A Universal Part-of-Speech Tagset.”**
  [Open text](https://aclanthology.org/L12-1115/). The universal inventory supports coarse
  grammatical categories. **verified-open**.
- **Gibson (1998), “Linguistic Complexity: Locality of Syntactic Dependencies.”** DOI
  [10.1016/S0010-0277(98)00034-1](https://doi.org/10.1016/S0010-0277(98)00034-1);
  user-supplied local PDF. The abstract and §2.2, pp. 1, 8, and 11–13, state that longer
  head-dependent integrations require greater resources. Gibson’s formal `I(n)` counts
  intervening discourse referents, so this source supplies theoretical locality context rather
  than the project’s exact mean token-distance definition. **verified; human-confirmed**.
- **Futrell, Mahowald & Gibson (2015), dependency-length minimization.** DOI
  [10.1073/pnas.1502134112](https://doi.org/10.1073/pnas.1502134112). Cross-linguistic
  evidence validates dependency length as a structural property. **verified-open**.
- **Yngve (1960), “A Model and an Hypothesis for Language Structure.”**
  [JSTOR 985230](https://www.jstor.org/stable/985230). Stack-depth limits anchor the
  tree-depth measure. **verified-open**.

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
- **Flesch (1948).** DOI [10.1037/h0057532](https://doi.org/10.1037/h0057532), and
  **Kincaid et al. (1975), DTIC ADA006655.**
  [Open report](https://apps.dtic.mil/sti/citations/ADA006655). Their applications use
  passages and do not validate readability formulas for approximately 15-token questions;
  this anchors a scope exclusion only. **verified-open** for Kincaid;
  **verified-paywalled** for Flesch.
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
  [10.1007/1-4020-3851-8](https://doi.org/10.1007/1-4020-3851-8). The contextual framework
  anchors work/search-task distinctions. **verified-paywalled**.
- **Belkin, Oddy & Brooks (1982), “ASK for Information Retrieval: Part I. Background and
  Theory.”** DOI [10.1108/eb026722](https://doi.org/10.1108/eb026722);
  user-supplied repository PDF. The abstract and introduction, p. 61, state that information
  needs are not in principle precisely specifiable and define an ASK as a recognized anomaly
  in the user’s state of knowledge that the user cannot precisely specify how to resolve.
  This anchors open-ended information-need context but not the binary `recall_orientation`
  coding. **verified-open; human-confirmed**.
- **Anderson & Krathwohl (2001), revised educational taxonomy.** ISBN 9780801319037.
  The cognitive-process dimension supplies the named exploratory labels.
  **verified-paywalled**.
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
- **Krippendorff (2019), _Content Analysis_, 4th ed.** ISBN 9781506395661. Reliability
  chapters anchor alpha, disagreement functions, and codebook discipline.
  **verified-paywalled**.
- **Zapf et al. (2016), nominal inter-rater reliability.** DOI
  [10.1186/s12874-016-0200-9](https://doi.org/10.1186/s12874-016-0200-9);
  [open text](https://pmc.ncbi.nlm.nih.gov/articles/PMC4974794/). Coefficient and interval
  choice depend on design and scale. **verified-open**.
- **Cohen (1960), kappa.** DOI
  [10.1177/001316446002000104](https://doi.org/10.1177/001316446002000104), and
  **Cohen (1968), weighted kappa.** DOI
  [10.1037/h0026256](https://doi.org/10.1037/h0026256), and
  **Fleiss (1971), multi-rater agreement.** DOI
  [10.1037/h0031619](https://doi.org/10.1037/h0031619). Historical comparators, not
  substitutes for the frozen alpha contract. **verified-paywalled**.
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
- **Davidson & Flachaire (2008), wild-bootstrap weights.** DOI
  [10.1016/j.jeconom.2008.08.003](https://doi.org/10.1016/j.jeconom.2008.08.003), and
  **Cameron, Gelbach & Miller (2008), bootstrap-based improvements.** DOI
  [10.1162/rest.90.3.414](https://doi.org/10.1162/rest.90.3.414). These anchor
  Rademacher weights and few-cluster inference. **verified-paywalled**.
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
- **McCullagh (1980), proportional odds.** DOI
  [10.1111/j.2517-6161.1980.tb01109.x](https://doi.org/10.1111/j.2517-6161.1980.tb01109.x);
  **Agresti (2010), _Analysis of Ordinal Categorical Data_, 2nd ed., ISBN 9780470082898**;
  and **Liddell & Kruschke (2018).** DOI
  [10.1016/j.jesp.2018.08.009](https://doi.org/10.1016/j.jesp.2018.08.009).
  These anchor grade-model ordering and ordinal interpretation. **verified-open** for
  Liddell–Kruschke; **verified-paywalled** for McCullagh/Agresti.
- **Cameron & Trivedi (2005), _Microeconometrics_, ISBN 9780521848053**;
  **Davidson & MacKinnon (2004), _Econometric Theory and Methods_, ISBN 9780195123722**;
  **Carroll et al. (2006), _Measurement Error in Nonlinear Models_, ISBN
  9781584886334**; and **Hardin & Hilbe (2013), _Generalized Estimating Equations_,
  ISBN 9781439881132** remain closed-access implementation references.
  **verified-paywalled metadata; exact passages to-verify**.

## Open verification queue

The unresolved items are deliberately visible rather than promoted to verified:

1. Keep Graesser & Person as supporting context; do not use it to add labels.
2. Keep Hamblin’s original record distinct from the later reprint DOI.
3. Preserve the explicit qualification that neither MacKinnon–Nielsen–Webb nor
   Kline–Santos establishes the complete two-way-clustered GLM procedure end to end.
