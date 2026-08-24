# Simple Mode source quotation audit

Status: draft for human verification  
Source inventory: `reports/13a-source-dossier.md`  
Claim inventory: `reports/13-simple-mode-analysis-research-plan.md`

## How to use this file

Each record distinguishes four questions:

1. Does the cited source exist?
2. Is its primary text accessible?
3. Does the quoted passage exist at the stated locator?
4. Does that passage support the claim or operational definition used by the Research Plan?

`SUPPORTED` means the quotation directly supports the cited use. `PARTIAL` means it supports
the general construct but not the exact project operationalization. `NOT SUPPORTED` means the
quoted source does not establish the stated claim. `USER ACCESS REQUIRED` means no quotation
has been supplied because the primary text was not accessible. No quotation in this file is
reconstructed from model memory.

# 1. Verified-open primary sources

## Dimension A — illocution and clause type

### Searle (1969) — human-confirmed

- **Full source:** John R. Searle, *Speech Acts: An Essay in the Philosophy of Language*,
  Cambridge University Press.
- **Links:** [DOI](https://doi.org/10.1017/CBO9781139173438);
  [user-supplied local PDF](sources/Searle%20-%20Speech%20Acts%20An%20Essay%20in%20the%20Philosophy%20of%20Language.pdf).
- **Research Plan use:** §12 Dimension A: separating propositional content from illocutionary
  force.
- **Quotations:**

  > “I am distinguishing between the illocutionary act and the propositional content of the
  > illocutionary act.”

  > “The illocutionary force indicator shows how the proposition is to be taken…”

  > “Illocutionary force indicating devices in English include at least: word order, stress,
  > intonation contour, punctuation, the mood of the verb, and the so-called performative
  > verbs.”

- **Locator:** pp. 30–31.
- **Why used:** Supports treating grammatical and prosodic form as evidence about force while
  maintaining a distinction between force and propositional content.
- **Assessment:** **SUPPORTED** as general speech-act theory. It does not define the project’s
  four clause labels or its UD rules.
- **Human verdict:** [x] quotations confirmed [x] locators confirmed [x] retain source

### Searle (1976) — human-confirmed

- **Full source:** John R. Searle, “A Classification of Illocutionary Acts,” *Language in
  Society* 5(1):1–23.
- **Links:** [DOI](https://doi.org/10.1017/S0047404500006837);
  [user-supplied local PDF](sources/Searle%20-%20A%20Classification%20of%20Illocutionary%20Acts%20(Unknown)%20(z-library.sk,%201lib.sk,%20z-lib.sk).pdf).
- **Research Plan use:** §12 Dimension A: directive illocutionary acts.
- **Quotations:**

  > “The five basic kinds of illocutionary acts are: representatives (or assertives),
  > directives, commissives, expressives, and declarations.”

  > “The illocutionary point of request is the same as that of commands: both are attempts to
  > get hearers to do something.”

  > “The illocutionary point of [directives] consists in the fact that they are attempts … by
  > the speaker to get the hearer to do something.”

- **Locator:** Abstract, p. 1; pp. 3 and 11.
- **Why used:** Defines directives and distinguishes their illocutionary purpose.
- **Assessment:** **SUPPORTED** for directive theory, not as a grammatical clause-type
  classifier.
- **Human verdict:** [x] quotations confirmed [x] locators confirmed [x] retain source

### Sadock & Zwicky (1985) — human-confirmed

- **Full source:** Jerrold M. Sadock and Arnold M. Zwicky, “Speech Act Distinctions in
  Syntax,” in Timothy Shopen (ed.), *Language Typology and Syntactic Description*, vol. I,
  ch. 6, pp. 155–196.
- **Link:** [Author PDF](https://web.stanford.edu/~zwicky/speech-act-distinctions.pdf).
- **Research Plan use:** §12 Dimension A, lines 2087–2096: sentence type versus communicative
  use, open/closed interrogatives, and the declarative/interrogative/imperative inventory.
- **Quotations:**

  > “Such a coincidence of grammatical structure and conventional conversational use we call
  > a sentence type.”

  > “Now any sentence with a conventionally indicated force can be put to uses other than, or
  > in addition to, the one to which it is conventionally suited…”

  > “Most languages are similar in presenting three basic sentence types with similar functions
  > and often strikingly similar forms. These are the declarative, interrogative, and
  > imperative.”

- **Locator:** §1.1, pp. 155–156; §2.1, p. 160. The yes-no and information-question
  distinction is also in §1.1, p. 156.
- **Why used:** Supports separating grammatical form from actual speech-act use, distinguishing
  polar from information questions, and retaining the three core sentence-type families.
- **Assessment:** **SUPPORTED** as a theoretical taxonomy anchor. It does not validate the
  project’s exact four-level mapping or UD heuristics.
- **Human verdict:** [x] quotations confirmed [x] locators confirmed [x] retain source

### Huddleston & Pullum (2002) — human-confirmed

- **Full source:** Rodney Huddleston and Geoffrey K. Pullum, *The Cambridge Grammar of the
  English Language*, Cambridge University Press.
- **Links:** ISBN 9780521431460;
  [user-supplied local PDF](sources/The%20Cambridge%20Grammar%20of%20the%20English%20Language%20(Rodney%20Huddleston%20Geoffrey%20K.%20Pullum)%20(z-library.sk,%201lib.sk,%20z-lib.sk).pdf).
- **Research Plan use:** §12 Dimension A: the four clause labels and the form/use distinction.
- **Quotations:**

  > “It is essential therefore to maintain a sharp conceptual distinction between the
  > grammatical clause types and the categories of meaning or use…”

  > “Open interrogatives contain an interrogative phrase based on one of the interrogative
  > words who, whom, whose, which, what, when, where, how, etc.”

  > “Imperatives are characteristically used as directives…”

  > “Declaratives can be used with either direct or indirect directive force.”

- **Locator:** Chapter 10, pp. 854, 856, 858, 929, 939, and 941.
- **Why used:** Directly distinguishes closed and open interrogatives and documents imperative,
  interrogative, and declarative directives.
- **Assessment:** **SUPPORTED** as the primary source for `open_interrogative`,
  `closed_interrogative`, `directive_imperative`, and `declarative_request`. The project’s
  deterministic UD precedence remains its own operationalization.
- **Human verdict:** [x] quotations confirmed [x] locators confirmed [x] retain as primary
  clause-type source

### Portner (2018) — human-confirmed

- **Full source:** Paul Portner, *Mood*, Oxford University Press.
- **Links:** [DOI](https://doi.org/10.1093/oso/9780199547524.001.0001);
  [user-supplied local PDF](sources/Mood%20(Paul%20Portner).pdf).
- **Research Plan use:** §12 Dimension A: sentence mood, clause type, and sentential force.
- **Quotations:**

  > “The difference between the concepts of clause type and sentence mood is that the former
  > are primarily identified in terms of grammatical form, while the latter are primarily
  > identified in terms of meaning.”

  > “Traditional grammatical descriptions make use of such clause types as declarative,
  > interrogative, and imperative.”

  > “Sentential forces are the fundamental conversational functions with which sentence moods
  > are associated.”

- **Locator:** Chapter 3, pp. 121, 123, and 127.
- **Why used:** Supports deriving form-based evidence while keeping clause form distinct from
  conversational force.
- **Assessment:** **SUPPORTED** as theoretical architecture. It does not itself determine the
  project’s four labels or parser heuristics.
- **Human verdict:** [x] quotations confirmed [x] locators confirmed [x] retain source

## Dimension B — answerhood and exhaustivity

### Dayal (2016) — human-confirmed

- **Full source:** Veneeta Dayal, *Questions*, Oxford University Press.
- **Links:** [DOI](https://doi.org/10.1093/acprof:oso/9780199281268.001.0001);
  [user-supplied local PDF](sources/Dayal,%20Veneeta%20-%20Questions.pdf).
- **Research Plan use:** §12 Dimension B: mention-some, weak/strong exhaustivity, and NPI
  evidence about exhaustivity.
- **Quotations:**

  > “Three types of answers are analyzed from this perspective: weakly exhaustive, strongly
  > exhaustive, and non-exhaustive answers.”

  > “The terms non-exhaustive answers and mention-some answers are used interchangeably, as
  > are the terms exhaustive answers and mention-all answers.”

  > “Negative polarity items in questions … provide a strong argument in favor of a
  > grammatical distinction between strong exhaustiveness and weak/non-exhaustiveness.”

  > “They argue that NPI licensing goes hand in hand with strong exhaustiveness.”

- **Locator:** Chapter 1, p. 16; Chapter 3, pp. 57 and 82–87. Page 67 records caveats on the
  negation diagnostic, including domain uncertainty and complementation failure.
- **Why used:** Directly anchors the three answer types and the grammatical evidence NPIs can
  provide about exhaustivity.
- **Assessment:** **SUPPORTED** for answer-type and exhaustivity diagnostics. It does not
  define the project’s `negative_conclusiveness` label and does not permit classification from
  the token *any* alone.
- **Human verdict:** [x] quotations confirmed [x] locators confirmed [x] retain source with
  diagnostic caveat

### Karttunen (1977) — human-confirmed

- **Full source:** Lauri Karttunen, “Syntax and Semantics of Questions,” *Linguistics and
  Philosophy* 1:3–44.
- **Links:** [DOI](https://doi.org/10.1007/BF00351935);
  [user-supplied local PDF](sources/Karttunen%20-%20Syntax%20and%20semantics%20of%20questions%20.pdf).
- **Research Plan use:** §12 Dimension B: question denotations, true answers, and contextual
  partial answers.
- **Quotations:**

  > “Hamblin’s idea was to let every direct question denote a set of propositions, namely, the
  > set of propositions expressed by possible answers to it.”

  > “I choose to make questions denote the set of propositions expressed by their true answers
  > instead of the set of propositions expressed by their possible answers.”

  > “If indirect questions denote sets of propositions that jointly constitute a true and
  > complete answer to the question…”

- **Locator:** pp. 7–10. Footnote 4 on p. 7 separately notes that direct wh-questions can
  pragmatically solicit more or less complete answers depending on context.
- **Why used:** Provides true-answer-set semantics while preserving a semantic/pragmatic
  distinction for partial answers.
- **Assessment:** **SUPPORTED** for true-answer sets and weak-exhaustive background. It does
  not by itself supply the project’s three annotation labels.
- **Human verdict:** [x] quotations confirmed [x] locators confirmed [x] retain with caveat

### Beck & Rullmann (1999) — human-confirmed

- **Full source:** Sigrid Beck and Hotze Rullmann, “A Flexible Approach to Exhaustivity in
  Questions,” *Natural Language Semantics* 7:249–298.
- **Links:** [DOI](https://doi.org/10.1023/A:1008373224343);
  [user-supplied local PDF](sources/Beck%20-%20A%20Flexible%20Approach%20to%20Exhaustivity%20in%20Questions.pdf).
- **Research Plan use:** §12 Dimension B: weak and strong exhaustivity.
- **Quotations:**

  > “A semantics for interrogatives is presented which is based on Karttunen’s theory, but in a
  > flexible manner incorporates both weak and strong exhaustivity.”

  > “The first of these is weakly exhaustive and the second strongly exhaustive.”

  > “…the basic denotation of a question is a set of propositions which intuitively constitute
  > its possible answers.”

- **Locator:** Abstract and Introduction, pp. 249–250.
- **Why used:** Direct primary-source support for distinct weakly and strongly exhaustive
  answerhood notions.
- **Assessment:** **SUPPORTED** as a primary weak/strong exhaustivity source. The project’s
  coding rules remain a separate operationalization.
- **Human verdict:** [x] quotations confirmed [x] locators confirmed [x] retain source

### Groenendijk & Stokhof (1984) — human-confirmed

- **Full source:** Jeroen Groenendijk and Martin Stokhof, *Studies on the Semantics of
  Questions and the Pragmatics of Answers*, University of Amsterdam doctoral dissertation.
- **Links:** [Part I](https://pure.uva.nl/ws/files/1989717/27444_Proefschrift_001_257.PDF);
  [Part II](https://pure.uva.nl/ws/files/1989719/27445_Proefschrift_258_577.PDF).
- **Research Plan use:** §12 Dimension B, lines 2098–2109: partition semantics, semantic
  answerhood, exhaustive answers, and mention-some/mention-all distinctions.
- **Quotations:**

  > “In this paper we will view questions as partitions of the set of indices…”

  > “If we view a question as a partition of the set of indices I, each element of that
  > partition, a set of indices, represents a proposition, a possible semantic answer to that
  > question.”

  > “This means that the partition I/Q is the set of possible semantic answers to Q.”

  > “Both answers are implicitly exhaustive. All answers are taken to be exhaustive, unless
  > they are explicitly marked as being non-exhaustive, or … the non-linguistic context makes
  > it perfectly obvious that the question itself is meant to be interpreted non-exhaustively.”

  > “We leave these ‘choice-interrogatives’ (and ‘mention-some interrogatives’) out of
  > consideration in this paper. The interrogatives which are treated here are often called
  > ‘mention-all interrogatives’.”

- **Locator:** Part I pp. 214–216; Part II p. 278 and note 13 on pp. 394–395.
- **Why used:** Supports questions-as-partitions, possible semantic answers, default
  exhaustivity, contextual mention-some answers, and the mention-some/mention-all terminology.
- **Assessment:** **SUPPORTED** for those constructs. It does not establish the project’s
  separate `weakly_exhaustive` category, which requires other evidence.
- **Human verdict:** [x] quotations confirmed [x] locators confirmed [x] retain source

### George (2011) — human-confirmed

- **Full source:** Benjamin Ross George, *Question Embedding and the Semantics of Answers*,
  UCLA doctoral dissertation.
- **Link:** [Dissertation PDF](https://linguistics.ucla.edu/wp-content/uploads/2021/11/old_brgeorge_dissertation_web_june2011.pdf).
- **Research Plan use:** §12 Dimension B: mention-some, weakly exhaustive, and strongly
  exhaustive answerhood.
- **Quotations:**

  > “Many accounts consider multiple types of answer, and some of them do this non-uniformly.
  > For example, Beck and Rullmann (1999) consider three kinds of answers. Strongly exhaustive
  > answers … ‘weakly exhaustive’ answers, and ‘mention-some’ answers.”

  > “A weakly exhaustive answer is all the information in a strongly exhaustive answer except
  > the ‘that’s all’ component…”

  > “A mention-some answer is an answer that identifies at least one … thing with the property
  > picked out by the question…”

  > “Starting with the ‘mention all’ idea, weak exhaustivity is a natural theoretical move…”

  > “I propose that a question’s semantic contribution is the set of its possible answers,
  > identifying the meaning of a question with its answerhood conditions…”

- **Locator:** Abstract, pp. 13–14; Chapter 1, pp. 13 and 16–17; Chapter 2, p. 18.
- **Why used:** Directly defines the three answer types and explains the positive versus
  negative information that distinguishes weak from strong exhaustivity.
- **Assessment:** **SUPPORTED** for retaining the three labels as annotation possibilities.
  George also questions whether weak exhaustivity is always independently required, so the
  source does not justify assuming that every item has a weakly exhaustive reading.
- **Human verdict:** [x] quotations confirmed [x] locators confirmed [x] retain source with
  caveat

### Roberts (2012)

- **Full source:** Craige Roberts, “Information Structure in Discourse: Towards an Integrated
  Formal Theory of Pragmatics,” *Semantics & Pragmatics* 5, Article 6:1–69.
- **Links:** [DOI](https://doi.org/10.3765/sp.5.6);
  [full text](https://semprag.org/index.php/sp/article/download/sp.5.6/pdf).
- **Research Plan use:** §12 Dimension B, lines 2105–2109: question-under-discussion context
  and answer relevance.
- **Quotation:**

  > “The relevant alternatives are those proffered by the question, or topic, under
  > discussion.”

- **Locator:** p. 6, paragraph beginning “Assertions are, as for Stalnaker, choices among
  alternatives.”
- **Why used:** Makes answer relevance conditional on the active question under discussion.
- **Assessment:** **SUPPORTED** for contextual relevance.
- **Human verdict:** [ ] accept [ ] reject

### Ginzburg (2012) — human-confirmed

- **Full source:** Jonathan Ginzburg, *The Interactive Stance: Meaning for Conversation*,
  Oxford University Press.
- **Links:** [DOI](https://doi.org/10.1093/acprof:oso/9780199697922.001.0001);
  [user-supplied local PDF](sources/Jonathan%20Ginzburg%20-%20The%20Interactive%20Stance%20Meaning%20for%20Conversation.pdf).
- **Research Plan use:** §12 Dimension B: dialogue context, QUD, and interaction-sensitive
  answer relevance.
- **Quotations:**

  > “…the issue or question currently under discussion.”

  > “…those questions that are raised for discussion and which condition both what can be said
  > (information pertaining to that question) and how (‘dialogue ellipsis’).”

  > “QUD (‘questions under discussion’): a partially ordered set that specifies the currently
  > discussed questions.”

- **Locator:** Chapter 2, pp. 18 and 20; Chapter 4, p. 66. Chapter 3, p. 32, additionally
  describes questions and propositions as structuring context and notes a theory of questions
  providing an account of answerhood.
- **Why used:** Supports the dialogue gameboard model in which current questions constrain
  coherent contributions and answer relevance.
- **Assessment:** **SUPPORTED** for dialogue context/QUD. It does not define the project’s
  mention-some, weakly exhaustive, or mention-all labels.
- **Human verdict:** [x] quotations confirmed [x] locators confirmed [x] retain as context

### Karttunen (1971) — human-confirmed

- **Full source:** Lauri Karttunen, “Some Observations on Factivity,” *Papers in Linguistics*
  4(1):55–69.
- **Links:** [DOI](https://doi.org/10.1080/08351817109370248);
  [user-supplied local PDF](sources/Karttunen%20-%20Some%20observations%20on%20factivity.pdf).
- **Research Plan use:** §12 Dimension B: contextual `presupposition_load`.
- **Quotations:**

  > “There is a general agreement that factive verbs involve presuppositions…”

  > “…a sentence with a factive predicate is said to presuppose the truth of its complement
  > sentence.”

  > “The main verb does not alone determine whether the complement is actually presupposed to
  > be true. The mood of the main sentence and the type of the complement also have to be taken
  > into account.”

- **Locator:** pp. 55–56.
- **Why used:** Provides factivity/presupposition background while explicitly requiring
  contextual and structural qualification.
- **Assessment:** **SUPPORTED** for contextual annotation. It rules out a simple verb-list
  implementation of `presupposition_load`.
- **Human verdict:** [x] quotations confirmed [x] locators confirmed [x] retain Karttunen
  [x] remove unsupplied Kiparsky & Kiparsky

## Dimension C — retrieval-decomposition complexity

### Wolfson et al. (2020)

- **Full source:** Tomer Wolfson et al., “Break It Down: A Question Understanding Benchmark,”
  *TACL* 8:183–198.
- **Links:** [DOI](https://doi.org/10.1162/tacl_a_00309);
  [full text](https://aclanthology.org/2020.tacl-1.13.pdf).
- **Research Plan use:** §12 Dimension C, lines 2115–2126: QDMR steps and operators.
- **Quotation:**

  > “QDMR Definition Given a question x, its QDMR is a sequence of n steps, s = ⟨s1, ..., sn⟩,
  > where each step si corresponds to a single query”.

- **Locator:** §2, “QDMR Definition,” article p. 185 (PDF p. 3).
- **Assessment:** **SUPPORTED** for counting decomposition steps and treating each as one query
  operation. The project’s three-way applicability rule remains its own convention.
- **Human verdict:** [ ] accept [ ] reject

### Yang et al. (2018), HotpotQA

- **Full source:** Zhilin Yang et al., “HotpotQA: A Dataset for Diverse, Explainable Multi-hop
  Question Answering,” *EMNLP 2018*, pp. 2369–2380.
- **Links:** [DOI](https://doi.org/10.18653/v1/D18-1259);
  [full text](https://aclanthology.org/D18-1259.pdf).
- **Research Plan use:** §12 Dimension C, lines 2117–2118 and 2126–2127: bridge and comparison
  `hop_structure`.
- **Quotations:**

  > “We call ‘Thom Yorke’ a bridge entity in this example.”

  > “Finally, we also collected a novel type of questions—comparison questions—as part of
  > HOTPOTQA, in which we require systems to compare two entities on some shared properties...”

- **Locator:** §2, article pp. 2370–2371 (PDF pp. 2–3).
- **Assessment:** **SUPPORTED** for distinct bridge and comparison structures.
- **Human verdict:** [ ] accept [ ] reject

### Ho et al. (2020), 2WikiMultiHopQA

- **Full source:** Xanh Ho, Anh-Khoa Duong Nguyen, Saku Sugawara, and Akiko Aizawa,
  “Constructing A Multi-hop QA Dataset for Comprehensive Evaluation of Reasoning Steps,”
  *COLING 2020*, pp. 6609–6625.
- **Links:** [DOI](https://doi.org/10.18653/v1/2020.coling-main.580);
  [full text](https://aclanthology.org/2020.coling-main.580.pdf).
- **Research Plan use:** §12 Dimension C: compositional, comparison, and bridge structures.
- **Quotation:**

  > “In our dataset, we have the following four types of questions: (1) comparison,
  > (2) inference, (3) compositional, and (4) bridge comparison. The inference and
  > compositional questions are the two subtypes of the bridge question which comprises a
  > bridge entity that connects the two paragraphs.”

- **Locator:** §2.2, “Question Types,” article p. 6611 (PDF p. 3).
- **Assessment:** **SUPPORTED** as source taxonomy context. Mapping it to the project’s
  four-level `hop_structure` remains an operational simplification.
- **Human verdict:** [ ] accept [ ] reject [ ] revise mapping

### Trivedi et al. (2022), MuSiQue

- **Full source:** Harsh Trivedi, Niranjan Balasubramanian, Tushar Khot, and Ashish Sabharwal,
  “MuSiQue: Multihop Questions via Single-hop Question Composition,” *TACL* 10:539–554.
- **Links:** [DOI](https://doi.org/10.1162/tacl_a_00475);
  [full text](https://aclanthology.org/2022.tacl-1.31.pdf).
- **Research Plan use:** §12 Dimension C: connected hop composition and shortcut filtering.
- **Quotations:**

  > “An edge (qj, qi) ∈ edges(GQ) indicates that the reasoning step qi relies critically on
  > the output of the predecessor step qj.”

  > “If, on held out data, the model can identify a subquestion’s answer (ai) without its
  > predecessor’s answer (aj), we say the edge (qj, qi) is disconnected.”

- **Locator:** article p. 541 (PDF p. 3) and p. 543 (PDF p. 5).
- **Assessment:** **SUPPORTED** for connected hops and filtering shortcut-solvable
  compositions.
- **Human verdict:** [ ] accept [ ] reject

### Jeong et al. (2024), Adaptive-RAG

- **Full source:** Soyeong Jeong et al., “Adaptive-RAG: Learning to Adapt
  Retrieval-Augmented Large Language Models through Question Complexity,” *NAACL 2024*,
  pp. 7036–7050.
- **Links:** [DOI](https://doi.org/10.18653/v1/2024.naacl-long.389);
  [full text](https://aclanthology.org/2024.naacl-long.389.pdf).
- **Research Plan use:** §12 Dimension C, lines 2128–2130: complexity-sensitive routing.
- **Quotation:**

  > “Our adaptive approach can select the most suitable strategy for retrieval-augmented LLMs,
  > ranging from iterative, to single, to even no retrieval approaches, based on the complexity
  > of given queries determined by our classifier.”

- **Locator:** Figure 2 caption, article p. 7037 (PDF p. 2).
- **Assessment:** **SUPPORTED** for the routing premise, not for this project’s exact features.
- **Human verdict:** [ ] accept [ ] reject

### Li & Roth (2002)

- **Full source:** Xin Li and Dan Roth, “Learning Question Classifiers,” *COLING 2002*.
- **Links:** [DOI](https://doi.org/10.3115/1072228.1072378);
  [full text](https://aclanthology.org/C02-1150.pdf).
- **Research Plan use:** §12 Dimension C: controlled question-type classification.
- **Quotation:**

  > “We define a two-layered taxonomy, which represents a natural semantic classification for
  > typical answers in the TREC task. The hierarchy contains 6 coarse classes ... and 50 fine
  > classes.”

- **Locator:** §2.2, “Question Hierarchy,” PDF p. 2.
- **Assessment:** **PARTIAL.** It supports hierarchical question classification, not this
  project’s decomposition labels.
- **Human verdict:** [ ] accept as context [ ] remove

## Dimension D — reference and lexical anchoring

### Gundel, Hedberg & Zacharski (1993) — human-confirmed

- **Full source:** Jeanette K. Gundel, Nancy Hedberg, and Ron Zacharski, “Cognitive Status and
  the Form of Referring Expressions in Discourse,” *Language* 69(2):274–307.
- **Links:** [DOI](https://doi.org/10.2307/416535);
  [user-supplied local PDF](sources/Gundel%20-%20Cognitive%20Status%20and%20the%20Form%20of%20Referring%20Expressions%20in%20Discourse.pdf).
- **Research Plan use:** §12 Dimension D: general relation between referring-expression form,
  cognitive status, and identifiability.
- **Quotations:**

  > “We propose six implicationally related cognitive statuses relevant for explicating the use
  > of referring expressions in natural language discourse.”

  > “We propose that there are six cognitive statuses relevant to the form of referring
  > expressions in natural language discourse…”

  > “The statuses are thus ordered from most restrictive (in focus) to least restrictive
  > (type identifiable)…”

- **Locator:** Abstract, p. 274; §2, pp. 275–277.
- **Why used:** Establishes that referring forms conventionally signal assumptions about the
  addressee’s cognitive status and identifiability.
- **Assessment:** **SUPPORTED** as general referring-form theory. It does not validate the
  project’s `full_name_form`, `alias_or_handle`, and `email_address` categories or their
  precedence.
- **Human verdict:** [x] quotations confirmed [x] locators confirmed [x] retain with scope
  caveat

## Dimension E — morphosyntactic complexity

### de Marneffe, Manning, Nivre & Zeman (2021)

- **Full source:** Marie-Catherine de Marneffe, Christopher D. Manning, Joakim Nivre, and
  Daniel Zeman, “Universal Dependencies,” *Computational Linguistics* 47(2):255–308.
- **Links:** [DOI](https://doi.org/10.1162/coli_a_00402);
  [open text](https://aclanthology.org/2021.cl-2.11/).
- **Research Plan use:** §12 Dimension E, lines 2142–2158: UD relations, morphology, tree
  structure, and Stanza-based implementation.
- **Quotation:**

  > “Through these dependencies, the words of a sentence are organized into a tree structure
  > with the main predicate as the root.”

  > “A consequence of this decision … is that a UD tree represents a sentence’s observed
  > surface predicate–argument structure rather than necessarily accurately capturing
  > phrase-internal syntactic constituency.”

- **Locator:** §2.1.1, “Heads and Dependents.”
- **Why used:** Supports representing Stanza output as UD dependency trees.
- **Assessment:** **PARTIAL.** It supports the parse representation but explicitly warns that
  UD does not reproduce phrase-internal constituency. It cannot by itself validate
  constituency-equivalent L2SCA measurements.
- **Human verdict:** [ ] accept [ ] reject [ ] needs more evidence

### Lu (2010)

- **Full source:** Xiaofei Lu, “Automatic Analysis of Syntactic Complexity in Second Language
  Writing,” *International Journal of Corpus Linguistics* 15(4):474–496.
- **Links:** [DOI](https://doi.org/10.1075/ijcl.15.4.02lu);
  [publisher PDF](https://www.jbe-platform.com/docserver/fulltext/ijcl.15.4.02lu.pdf).
- **Research Plan use:** §12 Dimension E, lines 2144–2158: L2SCA clauses, dependent clauses,
  DC/C, complex nominals, and CN/C.
- **Quotation:**

  > “A clause is defined as a structure with a subject and a finite verb … Non-finite verb
  > phrases are excluded in the definition of clauses.”

  > “A dependent clause is defined as a finite adjective, adverbial, or nominal clause.”

  > “Complex nominals comprise (i) nouns plus adjective, possessive, prepositional phrase,
  > relative clause, participle, or appositive, (ii) nominal clauses, and (iii) gerunds and
  > infinitives in subject position.”

- **Locator:** §3.2, “Clauses,” “Dependent clauses,” and “Complex nominals”; Table 1 for DC/C
  and CN/C.
- **Why used:** Supplies the original constituency/Tregex definitions and ratio names.
- **Assessment:** **NOT SUPPORTED for exact equivalence.** The implemented UD rules include
  `xcomp` without Lu’s finite-subject requirement, add relations/categories not present in
  Lu’s definition, and omit some Lu classes. They are analogous project operationalizations,
  not a faithful L2SCA reimplementation.
- **Human verdict:** [ ] accept [ ] reject [ ] revise implementation

### Kyle (2016)

- **Full source:** Kristopher Kyle, *Measuring Syntactic Development in L2 Writing:
  Fine Grained Indices of Syntactic Complexity and Usage-Based Indices of Syntactic
  Sophistication*, PhD dissertation, Georgia State University.
- **Link:** [DOI and full text](https://doi.org/10.57709/8501051).
- **Research Plan use:** §12 Dimension E, lines 2148–2158: TAASSC and dependency-based
  complexity tooling.
- **Quotation:**

  > “Dependency parsing is useful for automatic syntactic analyses … because it reveals the
  > syntactic relationships between words and phrases in a text.”

  > “Finally, both finite and non-finite clauses are considered clauses by TAASSC.”

- **Locator:** Chapter 3, “Dependency parsing” and TAASSC description.
- **Why used:** Supports dependency-based fine-grained features.
- **Assessment:** **PARTIAL.** It supports dependency features generally, not this project’s
  exact UD relation sets. Its inclusion of non-finite clauses also differs from Lu.
- **Human verdict:** [ ] accept [ ] reject [ ] needs more evidence

### Petrov, Das & McDonald (2012)

- **Full source:** Slav Petrov, Dipanjan Das, and Ryan McDonald, “A Universal Part-of-Speech
  Tagset,” *LREC 2012*, pp. 2089–2096.
- **Link:** [Open text](https://aclanthology.org/L12-1115/).
- **Research Plan use:** §12 Dimension E, lines 2148–2152: universal POS categories.
- **Quotation:**

  > “We did not rely on intrinsic definitions of the above categories. Instead, each category
  > is defined operationally.”

- **Locator:** §2, “Tagset,” p. 2090.
- **Why used:** Supports coarse operational POS categories.
- **Assessment:** **PARTIAL.** Its twelve-tag inventory predates the current UD/Stanza UPOS
  inventory and does not validate clause or complex-nominal rules.
- **Human verdict:** [ ] accept [ ] reject [ ] needs more evidence

### Gibson (1998) — human-confirmed

- **Full source:** Edward Gibson, “Linguistic Complexity: Locality of Syntactic Dependencies,”
  *Cognition* 68:1–76.
- **Links:** [DOI](https://doi.org/10.1016/S0010-0277(98)00034-1);
  [user-supplied local PDF](sources/Gibson%20-%20Linguistic%20complexity,%20locality%20of%20syntactic%20dependencies.pdf).
- **Research Plan use:** §12 Dimension E: theoretical locality rationale for
  `mean_dependency_length`.
- **Quotations:**

  > “…the greater the distance between an incoming word and the most local head or dependent
  > to which it attaches, the greater the integration cost.”

  > “The important idea in both of these components of the theory is locality: syntactic
  > predictions held in memory over longer distances are more expensive … and longer distance
  > head-dependent integrations are more expensive.”

  > “It is assumed that each linguistic integration requires a fixed quantity of computational
  > resources to perform the integration plus additional resources proportional to the distance
  > between the elements being integrated. Thus longer-distance integrations require more
  > resources, other factors being equal.”

- **Locator:** Abstract, p. 1; introductory summary, p. 8; §2.2, p. 11. The formal `I(n)`
  proposal and its limitations are on pp. 12–13.
- **Why used:** Establishes the theoretical claim that longer head-dependent integrations are
  more costly.
- **Assessment:** **SUPPORTED as theoretical context only.** Gibson’s formal distance counts
  intervening discourse referents, not raw token positions. Futrell et al. (2015) is the direct
  anchor for the word-distance measure, and the project’s averaging rule remains its own
  operationalization.
- **Human verdict:** [x] quotations confirmed [x] locators confirmed [x] retain 1998 source
  [x] remove inaccessible Gibson (2000) source

### Futrell, Mahowald & Gibson (2015)

- **Full source:** Richard Futrell, Kyle Mahowald, and Edward Gibson, “Large-Scale Evidence of
  Dependency Length Minimization in 37 Languages,” *PNAS* 112(33):10336–10341.
- **Links:** [DOI](https://doi.org/10.1073/pnas.1502134112);
  [open text](https://pmc.ncbi.nlm.nih.gov/articles/PMC4547262/).
- **Research Plan use:** §12 Dimension E, lines 2144–2152: dependency length.
- **Quotation:**

  > “We calculate the length of a single dependency arc as the number of words between a head
  > and a dependent, including the dependent.”

- **Locator:** “Materials and Methods — Measuring Dependency Length,” p. 10339.
- **Why used:** Defines word-distance dependency length.
- **Assessment:** **PARTIAL.** The project’s mean over selected non-root, non-punctuation arcs
  is a project-specific aggregation.
- **Human verdict:** [ ] accept [ ] reject [ ] needs more evidence

### Yngve (1960)

- **Full source:** Victor H. Yngve, “A Model and an Hypothesis for Language Structure,”
  *Proceedings of the American Philosophical Society* 104(5):444–466.
- **Links:** [MIT record](https://dspace.mit.edu/handle/1721.1/4453);
  [public PDF](https://aclanthology.org/www.mt-archive.info/50/ProcAmPhilSoc-1960-Yngve.pdf).
- **Research Plan use:** §12 Dimension E, lines 2144–2152; the dossier linked it to
  `dependency_tree_depth`.
- **Quotation:**

  > “It is tempting to identify the temporary storage of our model in the case of spoken
  > language with the facility that we use for immediate memory.”

- **Locator:** Depth-hypothesis passage, approximately pp. 449–450.
- **Why used:** Intended as theoretical background for structural depth.
- **Assessment:** **NOT SUPPORTED.** Yngve depth concerns temporary storage of unexpanded
  constituency nodes during left-to-right generation. Maximum UD root-to-word arc count is a
  different construct.
- **Human verdict:** [ ] accept [ ] remove source [ ] change feature rationale

## Dimension F — exploratory lexical and information-theoretic sources

### Brysbaert & New (2009)

- **Full source:** Marc Brysbaert and Boris New, “Moving Beyond Kučera and Francis: A Critical
  Evaluation of Current Word Frequency Norms and the Introduction of a New and Improved Word
  Frequency Measure for American English,” *Behavior Research Methods* 41(4):977–990.
- **Links:** [DOI](https://doi.org/10.3758/BRM.41.4.977);
  [institutional PDF](https://www.ugent.be/pp/experimentele-psychologie/en/research/documents/subtlexus/brysbaertnew.pdf).
- **Research Plan use:** §12 Dimension F, lines 2160–2163: SUBTLEX-US frequency.
- **Quotation:**

  > “We found that frequencies based on television and film subtitles are better than
  > frequencies based on written sources, certainly for the monosyllabic and bisyllabic words
  > used in psycholinguistic research.”

- **Locator:** Abstract, p. 977.
- **Why used:** Supports SUBTLEX-US as a lexical frequency norm.
- **Assessment:** **PARTIAL.** It does not define the project’s token aggregation or validate
  the measure for short questions.
- **Human verdict:** [ ] accept [ ] reject [ ] needs more evidence

### Hale (2001)

- **Full source:** John Hale, “A Probabilistic Earley Parser as a Psycholinguistic Model,”
  *NAACL 2001*.
- **Links:** [DOI](https://doi.org/10.3115/1073336.1073357);
  [open PDF](https://aclanthology.org/N01-1021.pdf).
- **Research Plan use:** §12 Dimension F, lines 2162–2163: per-token surprisal.
- **Quotation:**

  > “This report considers a definition of cognitive load in terms of … the surprisal of word
  > \(w_i\) given its prefix \(w_0…i-1\) on a phrase-structural language model.”

- **Locator:** Abstract, p. 159.
- **Why used:** Defines incremental word surprisal.
- **Assessment:** **SUPPORTED** for the general construct, not for any particular modern model.
- **Human verdict:** [ ] accept [ ] reject

### Levy (2008)

- **Full source:** Roger Levy, “Expectation-Based Syntactic Comprehension,” *Cognition*
  106(3):1126–1177.
- **Links:** [DOI](https://doi.org/10.1016/j.cognition.2007.05.006);
  [author PDF](https://www.mit.edu/~rplevy/papers/levy-2008-cognition.pdf).
- **Research Plan use:** §12 Dimension F, lines 2162–2163: surprisal interpretation.
- **Quotation:**

  > “The difficulty of a word is proportional to its surprisal (its negative log-probability)
  > in the context within which it appears.”

- **Locator:** Abstract, p. 1126.
- **Why used:** Defines expectation-based surprisal.
- **Assessment:** **SUPPORTED** for the general construct.
- **Human verdict:** [ ] accept [ ] reject

### McCarthy & Jarvis (2010) — human-confirmed

- **Full source:** Philip M. McCarthy and Scott Jarvis, “MTLD, vocd-D, and HD-D:
  A Validation Study of Sophisticated Approaches to Lexical Diversity Assessment,”
  *Behavior Research Methods* 42:381–392.
- **Links:** [DOI](https://doi.org/10.3758/BRM.42.2.381);
  [user-supplied local PDF](sources/MTLD,%20vocd-D,%20and%20HD-D-%20A%20validation%20study%20of%20sophisticated%20approaches%20to%20lexical%20diversity%20assessment.pdf).
- **Research Plan use:** §12 Dimension F: deciding whether MTLD is appropriate for roughly
  15-token questions.
- **Quotations:**

  > “MTLD performs well with respect to all four types of validity and is, in fact, the only
  > index not found to vary as a function of text length.”

  > “The shorter the text, the greater will be the part of its value that is composed of a
  > remainder.”

  > “Consequently, shorter texts will be more difficult to evaluate with confidence.”

  > “In testing the tool during the development process, we found that texts as short as
  > 100 tokens can be used.”

- **Locator:** Abstract, p. 381; “MTLD — Partial Factors,” p. 384.
- **Why used:** Establishes that MTLD performs well generally but that reliable evaluation of
  very short texts is limited by remainder approximation and was tested only down to 100 tokens.
- **Assessment:** **SUPPORTED for exclusion from this corpus.** The reason is that roughly
  15-token questions fall far below the paper’s tested lower bound—not that MTLD is generally
  unreliable.
- **Human verdict:** [x] quotations confirmed [x] locators confirmed [x] retain source and
  correct the plan wording

### Kincaid et al. (1975)

- **Full source:** J. Peter Kincaid, Robert P. Fishburne Jr., Richard L. Rogers, and Brad S.
  Chissom, *Derivation of New Readability Formulas … for Navy Enlisted Personnel*, Research
  Branch Report 8-75, DTIC ADA006655.
- **Links:** [DOI](https://doi.org/10.21236/ADA006655);
  [open report](https://apps.dtic.mil/sti/citations/ADA006655).
- **Research Plan use:** §12 Dimension F, lines 2165–2166: excluding classical readability
  formulas for roughly 15-token questions.
- **Quotation:**

  > “They were derived from test results of 531 Navy enlisted personnel enrolled in four
  > technical training schools … [and] 18 passages taken from Rate Training Manuals.”

- **Locator:** Report abstract.
- **Why used:** Establishes the source population and passage domain.
- **Assessment:** **NOT SUPPORTED for ‘unreliable’.** It supports “not validated for short
  questions,” not unreliability at approximately 15 tokens.
- **Human verdict:** [ ] retain as design caution [ ] remove claim [ ] find better source

### Lee, Jang & Lee (2021)

- **Full source:** Bruce W. Lee, Yoo Sung Jang, and Jason Lee, “Pushing on Text Readability
  Assessment: A Transformer Meets Handcrafted Linguistic Features,” *EMNLP 2021*,
  pp. 10669–10686.
- **Link:** [DOI and open text](https://aclanthology.org/2021.emnlp-main.834/).
- **Research Plan use:** §12 Dimension F, lines 2167–2168: LingFeat, exploratory only.
- **Quotation:**

  > “We extract 255 handcrafted linguistic features using self-developed extraction software.”

- **Locator:** Abstract, p. 10669.
- **Why used:** Establishes the existence and breadth of LingFeat.
- **Assessment:** **PARTIAL.** “Exploratory only” is the project’s caution, not the source’s
  conclusion.
- **Human verdict:** [ ] accept [ ] reject

## Dimension G — task and intent framing

### Belkin, Oddy & Brooks (1982) — human-confirmed

- **Full source:** N. J. Belkin, R. N. Oddy, and H. M. Brooks, “ASK for Information Retrieval:
  Part I. Background and Theory,” *Journal of Documentation* 38(2):61–71.
- **Links:** [DOI](https://doi.org/10.1108/eb026722);
  [user-supplied local PDF](sources/Belkin,%20Oddy%20&%20Brooks%20-%20ASK%20FOR%20INFORMATION%20RETRIEVAL_%20PART%20I.%20BACKGROUND%20AND%20THEORY.pdf).
- **Research Plan use:** §12 Dimension G: anomalous states of knowledge and open-ended
  information needs.
- **Quotations:**

  > “Basic premises of the project were: that information needs are not in principle precisely
  > specifiable…”

  > “This new approach recognizes that a fundamental element in the IR situation is the
  > development of an information need out of an inadequate state of knowledge.”

  > “The ASK hypothesis is that an information need arises from a recognized anomaly in the
  > user’s state of knowledge concerning some topic or situation and that, in general, the user
  > is unable to specify precisely what is needed to resolve that anomaly.”

- **Locator:** Abstract and Introduction, article p. 61 (local PDF pp. 2–3).
- **Why used:** Supports treating some information needs as open-ended and imperfectly
  expressible rather than as precise best-match queries.
- **Assessment:** **SUPPORTED as conceptual context.** It does not define the project’s binary
  `recall_orientation` label; Oard & Webber is the direct high-recall/high-precision anchor.
- **Human verdict:** [x] quotations confirmed [x] locators confirmed [x] retain as context

### Oard & Webber (2013)

- **Full source:** Douglas W. Oard and William Webber, “Information Retrieval for
  E-Discovery,” *Foundations and Trends in Information Retrieval* 7(2–3):99–237.
- **Links:** [DOI](https://doi.org/10.1561/1500000025);
  [author PDF](https://user.eng.umd.edu/~oard/pdf/fntir13.pdf).
- **Research Plan use:** §12 Dimension G, lines 2172–2176: `recall_orientation`.
- **Quotation:**

  > “E-discovery focuses on high recall, even in large collections, in contrast to the
  > high-precision focus of many end-user applications, such as Web search.”

- **Locator:** §1, p. 101.
- **Why used:** Directly distinguishes high-recall review from high-precision retrieval.
- **Assessment:** **SUPPORTED.**
- **Human verdict:** [ ] accept [ ] reject

### Broder (2002)

- **Full source:** Andrei Broder, “A Taxonomy of Web Search,” *SIGIR Forum* 36(2):3–10.
- **Links:** [DOI](https://doi.org/10.1145/792550.792552);
  [public PDF](https://sigir.org/files/forum/F2002/broder.pdf).
- **Research Plan use:** §12 Dimension G, lines 2175–2177: intent taxonomy.
- **Quotation:**

  > “We classify web queries according to their intent into 3 classes: 1. Navigational …
  > 2. Informational … 3. Transactional.”

- **Locator:** §3, “A taxonomy of web searches,” approximately p. 5.
- **Why used:** Supports classifying search intent.
- **Assessment:** **PARTIAL.** The project’s binary recall-orientation mapping is new.
- **Human verdict:** [ ] accept [ ] reject [ ] needs more evidence

### Rose & Levinson (2004)

- **Full source:** Daniel E. Rose and Danny Levinson, “Understanding User Goals in Web Search,”
  *WWW 2004*, pp. 13–19.
- **Links:** [DOI](https://doi.org/10.1145/988672.988675);
  [public PDF](https://www.cse.iitb.ac.in/~soumen/doc/www2013/QirWoo/RoseL2004FunctionalIntent.pdf).
- **Research Plan use:** §12 Dimension G, lines 2175–2177: known-item versus open review needs.
- **Quotation:**

  > “My goal is to go to [a] specific known website that I already have in mind.”

  > “I want to get an answer to an open-ended question, or one with unconstrained depth.”

- **Locator:** Table 1, “The Search Goal Hierarchy,” approximately p. 15.
- **Why used:** Supports known-item and open-ended intent distinctions.
- **Assessment:** **PARTIAL.** It does not establish the exact project binary label.
- **Human verdict:** [ ] accept [ ] reject [ ] needs more evidence

### Graesser & Person (1994)

- **Full source:** Arthur C. Graesser and Natalie K. Person, “Question Asking During Tutoring,”
  *American Educational Research Journal* 31(1):104–137.
- **Links:** [DOI](https://doi.org/10.3102/00028312031001104);
  [public full text](https://gwern.net/doc/psychology/spaced-repetition/1994-graesser.pdf).
- **Research Plan use:** §12 Dimension G, lines 2177–2178; supporting taxonomy context only.
- **Quotation:**

  > “Table 1 presents the 18 question-content categories in the GPH scheme … These categories
  > are defined according to the content of the information sought rather than on the question
  > stems.”

- **Locator:** “Theoretical Dimensions That Address Question Quality,” pp. 108–109; Table 1.
- **Why used:** Supports question-content taxonomy context.
- **Assessment:** **PARTIAL.** It should not add labels to the frozen codebook.
- **Human verdict:** [ ] accept as context [ ] remove

## Dimension H — revision-certified local sources

- **Repository:** `relativityone/r1-evals`.
- **Verified revision:**
  [`41978dbd459ec3dfed9adcf3cc3695aae31a1551`](https://github.com/relativityone/r1-evals/commit/41978dbd459ec3dfed9adcf3cc3695aae31a1551),
  dated 2026-07-21, commit message `Scorer harness.` The local `r1-evals-new` checkout has this
  exact commit as `HEAD`.
- **Research Plan use:** §12 Dimension H: interpretation of `document_ids` and
  `expectation_document_count`.

### `models_v2.py`

- **Source:** [`src/r1_evals/rubrics/models_v2.py`, lines 293–315](https://github.com/relativityone/r1-evals/blob/41978dbd459ec3dfed9adcf3cc3695aae31a1551/src/r1_evals/rubrics/models_v2.py#L293-L315).
- **Quotation:**

  > `document_ids : list[str] | None` — `Optional list of supporting document IDs.`

- **Assessment:** **SUPPORTED** for optional supporting-document metadata. A revision-scoped
  `git grep` finds direct Python references only in model definitions and parsers; no prompt,
  grader, or scorer directly names the field. This does not rule out indirect serialization.

### `4-03.rubric.toml`

- **Source:** [`src/r1_evals/rubrics/rubric_data/air_assist/mallinckrodt/rubrics_for_ga/4-03.rubric.toml`,
  lines 22–35](https://github.com/relativityone/r1-evals/blob/41978dbd459ec3dfed9adcf3cc3695aae31a1551/src/r1_evals/rubrics/rubric_data/air_assist/mallinckrodt/rubrics_for_ga/4-03.rubric.toml#L22-L35).
- **Quotations:**

  > `description = "Michael Brennan, MD was one of three physician presenters at Covidien's
  > 2012 Exalgo national broadcast event"`

  > `description = "Charles Argoff, MD was one of three physician presenters at Covidien's
  > 2012 Exalgo national broadcast event"`

- **Assessment:** The three expectations carry 21, 26, and 28 IDs. This proves high
  multiplicity for individual factual expectations, but **does not prove** that every document
  independently attests the fact or that the list is conjunctive/disjunctive.
- **Human verdict:** [x] revision confirmed [x] model quotation confirmed [x] rubric rows
  confirmed [x] retain with semantic caveat

## Annotation and measurement validation

### Artstein & Poesio (2008)

- **Full source:** Ron Artstein and Massimo Poesio, “Inter-Coder Agreement for Computational
  Linguistics,” *Computational Linguistics* 34(4):555–596.
- **Links:** [DOI](https://doi.org/10.1162/coli.07-034-R2);
  [open text](https://aclanthology.org/J08-4004/).
- **Research Plan use:** §13.2: agreement is reliability, not validity; coefficients must match
  measurement scale.
- **Quotation:**

  > “Reliability is the extent to which the data collection procedure yields the same results
  > on repeated trials. Validity is the extent to which the data collection procedure measures
  > the intended concept.”

- **Locator:** §§2.1–2.2, pp. 557–558.
- **Assessment:** **SUPPORTED** for separating reproducibility from construct validity. It does
  not validate this project’s coding schema.
- **Human verdict:** [ ] accept [ ] reject

### Zapf et al. (2016)

- **Full source:** Antonia Zapf et al., “Measuring inter-rater reliability for nominal
  data—which coefficients and confidence intervals are appropriate?” *BMC Medical Research
  Methodology* 16:93.
- **Links:** [DOI](https://doi.org/10.1186/s12874-016-0200-9);
  [open text](https://pmc.ncbi.nlm.nih.gov/articles/PMC4974794/).
- **Research Plan use:** §13.2: design- and scale-sensitive coefficient and interval choice.
- **Quotation — abstract:**

  > “The choice of the appropriate coefficient and of the appropriate confidence interval
  > depends on the study design.”

- **Locator:** Abstract, Conclusions.
- **Assessment:** **PARTIAL.** It does not validate this project’s exact bootstrap or missing
  code-pair treatment.
- **Human verdict:** [ ] accept [ ] reject [ ] needs more evidence

### Gilardi, Alizadeh & Kubli (2023)

- **Full source:** Fabrizio Gilardi, Meysam Alizadeh, and Maël Kubli, “ChatGPT Outperforms
  Crowd-Workers for Text-Annotation Tasks,” *PNAS* 120(30):e2305016120.
- **Link:** [DOI](https://doi.org/10.1073/pnas.2305016120).
- **Research Plan use:** §13.2: evaluate LLM annotations against human-labelled data.
- **Quotation — abstract:**

  > “We find that ChatGPT outperforms crowd-workers for four out of five annotation tasks in
  > terms of accuracy, while the intercoder agreement of ChatGPT exceeds that of both
  > crowd-workers and trained annotators for all tasks.”

- **Assessment:** **PARTIAL.** It supports task-specific human comparison, not validity from
  inter-LLM agreement or generalization to this project.
- **Human verdict:** [ ] accept [ ] reject

### Egami, Hinck, Stewart & Wei (2023)

- **Full source:** Naoki Egami, Musashi Hinck, Brandon M. Stewart, and Hanying Wei, “Using
  Imperfect Surrogates for Downstream Inference.”
- **Links:** [arXiv:2306.04746](https://arxiv.org/abs/2306.04746);
  [NeurIPS paper](https://proceedings.neurips.cc/paper_files/paper/2023/hash/d862f7f5445255090de13b825b880d59-Abstract-Conference.html).
- **Research Plan use:** §13.4: predicted-label error can bias downstream inference.
- **Quotation — abstract:**

  > “Unfortunately, using these predictions without accounting for prediction error can lead
  > to biased estimates and invalid confidence intervals.”

- **Assessment:** **PARTIAL.** It supports correction for imperfect labels but not the plan’s
  two-way rubric/arm clustered extension.
- **Human verdict:** [ ] accept general claim [ ] reject [ ] require extension derivation

### Horvitz & Thompson (1952)

- **Full source:** D. G. Horvitz and D. J. Thompson, “A Generalization of Sampling Without
  Replacement From a Finite Universe,” *JASA* 47(260):663–685.
- **Link:** [DOI](https://doi.org/10.2307/2280784).
- **Research Plan use:** §13.2a: inverse-inclusion weighting of gold confusion cells.
- **Quotation:**

  > “The method of estimation suggested consists of weighting the value of the character for
  > each unit in the sample by the reciprocal of the probability that the unit is included in
  > the sample.”

- **Locator:** Introduction, p. 664.
- **Assessment:** **SUPPORTED** for the Horvitz–Thompson total; it does not establish all
  downstream ratio estimators or their variance.
- **Human verdict:** [ ] accept [ ] reject

### Mashreghi, Haziza & Léger (2016)

- **Full source:** Zeinab Mashreghi, David Haziza, and Christian Léger, “A Survey of Bootstrap
  Methods in Finite Population Sampling,” *Statistics Surveys* 10:1–52.
- **Link:** [DOI](https://doi.org/10.1214/16-SS113).
- **Research Plan use:** §13.2a: resampling independently within gold strata.
- **Quotation:**

  > “In stratified simple random sampling without replacement, the bootstrap methods described
  > in Section 4.1.1 can be applied independently in each stratum.”

- **Locator:** §4.1.2, p. 21.
- **Assessment:** **PARTIAL.** It supports stratum-wise resampling, not the unique choice of
  this project’s exact algorithm, percentile interval, or nonlinear metric implementation.
- **Human verdict:** [ ] accept [ ] reject [ ] revise method

### Brodersen et al. (2010)

- **Full source:** Kay Henning Brodersen et al., “The Balanced Accuracy and Its Posterior
  Distribution,” *20th ICPR*, pp. 3121–3124.
- **Link:** [DOI](https://doi.org/10.1109/ICPR.2010.764).
- **Research Plan use:** §13.2a: balanced accuracy for binary annotation validity.
- **Quotation:**

  > “The balanced accuracy is defined as the average accuracy obtained on either class.”

- **Locator:** §III, p. 3122.
- **Assessment:** **SUPPORTED** for balanced accuracy, not for survey weighting or intervals.
- **Human verdict:** [ ] accept [ ] reject

## Statistical methodology

### Efron & Morris (1975) — human-confirmed

- **Full source:** Bradley Efron and Carl Morris, “Data Analysis Using Stein’s Estimator and
  Its Generalizations,” *Journal of the American Statistical Association* 70(350):311–319.
- **Links:** [DOI](https://doi.org/10.1080/01621459.1975.10479864);
  [user-supplied local PDF](sources/efron1975.pdf).
- **Research Plan use:** §5.1: general empirical-Bayes shrinkage motivation for Layer 1.
- **Quotations:**

  > “This estimator is reviewed briefly in an empirical Bayes context.”

  > “The estimator (1.4) arises quite naturally in an empirical Bayes context.”

  > “…shrinking all \(X_i\) toward \(\bar X\)…”

- **Locator:** Abstract and Introduction, p. 311; empirical-Bayes derivation and baseball
  example, p. 312.
- **Why used:** Provides historical and applied evidence for estimating related units jointly
  and shrinking noisy unit estimates toward a shared center.
- **Assessment:** **SUPPORTED as general context only.** The paper’s normal-means/James–Stein
  setting does not validate the project’s exact hierarchical count likelihood, covariance
  structure, or decision probabilities.
- **Human verdict:** [x] quotations confirmed [x] locators confirmed [x] retain as context

### Gelman & Hill (2007) — human-confirmed

- **Full source:** Andrew Gelman and Jennifer Hill, *Data Analysis Using Regression and
  Multilevel/Hierarchical Models*, Cambridge University Press.
- **Links:** [DOI](https://doi.org/10.1017/CBO9780511790942);
  [user-supplied local PDF](sources/Data%20Analysis%20Using%20Regression%20and%20MultilevelHierarchical%20Models%20(Andrew%20Gelman,%20Jennifer%20Hill)%20(z-library.sk,%201lib.sk,%20z-lib.sk).pdf).
- **Research Plan use:** §5.1: multilevel partial pooling and joint group/data-level estimation.
- **Quotations:**

  > “…no pooling ignores information and can give unacceptably variable inferences, and
  > complete pooling suppresses variation that can be important…”

  > “…ultimately we prefer the partial pooling that comes out of a multilevel analysis.”

  > “The crucial multilevel modeling step is that these \(J\) coefficients are then themselves
  > given a model…”

  > “The group-level model is estimated simultaneously with the data-level regression of
  > \(y\).”

- **Locator:** p. 7; Chapter 12, p. 251. Pages 253–254 further show the weighted
  group/overall estimate and stronger shrinkage for smaller groups.
- **Why used:** Directly supports partial pooling, jointly modeled group effects, and
  sample-size-sensitive shrinkage.
- **Assessment:** **SUPPORTED** for the multilevel partial-pooling architecture. It does not
  determine the project’s exact distributional or computational choices.
- **Human verdict:** [x] quotations confirmed [x] locators confirmed [x] retain source

### Liang & Zeger (1986)

- **Full source:** Kung-Yee Liang and Scott L. Zeger, “Longitudinal Data Analysis Using
  Generalized Linear Models,” *Biometrika* 73(1):13–22.
- **Link:** [DOI](https://doi.org/10.1093/biomet/73.1.13).
- **Research Plan use:** §5.2: GEE-style scores and robust sandwich covariance.
- **Quotation:**

  > “The consistency of β̂ and of its variance estimate depends only on the correct
  > specification of the mean, not on the correct choice of R.”

- **Locator:** §3, pp. 17–18.
- **Assessment:** **PARTIAL.** It does not establish two-way clustering, the plan’s finite
  sample corrections, or its wild bootstrap.
- **Human verdict:** [ ] accept [ ] reject

### Cameron, Gelbach & Miller (2011)

- **Full source:** A. Colin Cameron, Jonah B. Gelbach, and Douglas L. Miller, “Robust Inference
  With Multiway Clustering,” *JBES* 29(2):238–249.
- **Link:** [DOI](https://doi.org/10.1198/jbes.2010.07136).
- **Research Plan use:** §5.2: two-way inclusion–exclusion covariance.
- **Quotation:**

  > “The variance matrix is computed by adding the two one-way cluster-robust variance
  > matrices, and subtracting the cluster-robust variance matrix computed using clustering on
  > the intersection of the two cluster dimensions.”

- **Locator:** §2; working-paper p. 8, published pp. 240–241.
- **Assessment:** **SUPPORTED** for the three-term structure. It does not validate every
  finite-sample correction frozen by the plan.
- **Human verdict:** [ ] accept [ ] reject [ ] inspect correction

### Kline & Santos (2012)

- **Full source:** Patrick Kline and Andres Santos, “A Score Based Approach to Wild Bootstrap
  Inference,” *Journal of Econometric Methods* 1(1):23–41.
- **Links:** [DOI](https://doi.org/10.1515/2156-6674.1006);
  [author PDF](https://eml.berkeley.edu/~pkline/papers/ScoreFinal_web.pdf).
- **Research Plan use:** §5.3: score perturbation instead of invalid bootstrap Bernoulli
  responses.
- **Quotation — abstract:**

  > “We propose a new wild bootstrap procedure for inference in models defined by moment
  > restrictions. Our approach perturbs the scores of the unrestricted model in a manner that
  > is robust to heteroskedasticity of unknown form.”

- **Assessment:** **PARTIAL.** It does not establish the plan’s GLM-specific, two-way
  clustered, restricted-arm construction.
- **Human verdict:** [ ] accept general framework [ ] reject [ ] require derivation

### MacKinnon, Nielsen & Webb (2021)

- **Full source:** James G. MacKinnon, Morten Ørregaard Nielsen, and Matthew D. Webb, “Wild
  Bootstrap and Asymptotic Inference With Multiway Clustering,” *JBES* 39(2):505–519.
- **Link:** [DOI](https://doi.org/10.1080/07350015.2019.1677473).
- **Research Plan use:** §5.3: three-term CRVE and bootstrap clustering on the few-cluster
  dimension.
- **Quotation:**

  > “When one dimension has few clusters and the other has many, clustering only by the former
  > dimension, with or without the wild cluster bootstrap, generally seems to be the best
  > approach.”

- **Locator:** §7, Conclusion.
- **Assessment:** **PARTIAL.** It supports the few-cluster rationale, not the exact GLM score
  implementation or an unconditional arm-only rule.
- **Human verdict:** [ ] accept rationale [ ] reject [ ] require derivation

### Benjamini & Hochberg (1995)

- **Full source:** Yoav Benjamini and Yosef Hochberg, “Controlling the False Discovery Rate:
  A Practical and Powerful Approach to Multiple Testing,” *JRSS B* 57(1):289–300.
- **Link:** [DOI](https://doi.org/10.1111/j.2517-6161.1995.tb02031.x).
- **Research Plan use:** §5.6: ten-family BH step-up computation.
- **Quotation:**

  > “Let \(k\) be the largest \(i\) for which \(P_{(i)} \leq (i/m)q^*\); then reject all
  > \(H_{(i)}\), \(i=1,2,\ldots,k\).”

- **Locator:** §3.1, p. 292.
- **Assessment:** **SUPPORTED** for the computation. The original proof does not provide an
  unqualified guarantee for the plan’s dependent Wald tests.
- **Human verdict:** [ ] accept procedure [ ] reject

### Benjamini & Yekutieli (2001)

- **Full source:** Yoav Benjamini and Daniel Yekutieli, “The Control of the False Discovery
  Rate in Multiple Testing Under Dependency,” *Annals of Statistics* 29(4):1165–1188.
- **Link:** [DOI](https://doi.org/10.1214/aos/1013699998).
- **Research Plan use:** §5.6: PRDS and arbitrary-dependence qualifications.
- **Quotation:**

  > “When the test statistics have positive regression dependency on each of the test
  > statistics corresponding to the true null hypotheses, the Benjamini and Hochberg procedure
  > controls the FDR at level less than or equal to \(m_0q/m\).”

  > “For all other forms of dependency, the procedure with
  > \(q/\sum_{i=1}^{m}(1/i)\) in place of \(q\) controls the FDR at level less than or equal to
  > \(m_0q/m\).”

- **Locator:** Theorems 1.2–1.3, pp. 1167–1168.
- **Assessment:** **SUPPORTED** for the caveat. The plan establishes neither PRDS nor the
  arbitrary-dependence correction and therefore must not claim finite-sample 5% FDR control.
- **Human verdict:** [ ] accept limitation [ ] revise multiplicity plan

### Mundlak (1978)

- **Full source:** Yair Mundlak, “On the Pooling of Time Series and Cross Section Data,”
  *Econometrica* 46(1):69–85.
- **Link:** [JSTOR](https://www.jstor.org/stable/1913646).
- **Research Plan use:** within/between cluster-mean decomposition.
- **Quotation — abstract only:**

  > “The analysis is presented in a unified framework in which it is shown that the random
  > effects model is a special case of the fixed effects model.”

- **Assessment:** **PARTIAL.** The abstract does not establish the exact cluster-mean coding,
  nonlinear interpretation, or project estimands.
- **Human verdict:** [ ] supply full text [ ] retain as context [ ] remove

### Phipson & Smyth (2010)

- **Full source:** Belinda Phipson and Gordon K. Smyth, “Permutation P-values Should Never Be
  Zero,” *Statistical Applications in Genetics and Molecular Biology* 9(1), Article 39.
- **Link:** [DOI](https://doi.org/10.2202/1544-6115.1585).
- **Research Plan use:** §5.3: sampled-regime `(b+1)/(B+1)` p-values.
- **Quotation:**

  > “The exact p-value when \(B\) permutations are randomly drawn with replacement is
  > \((b+1)/(B+1)\), where \(b\) is the number of permutation statistics greater than or equal
  > to the observed statistic.”

- **Locator:** §4, Monte Carlo p-value equation.
- **Assessment:** **PARTIAL.** Exactness does not transfer automatically to the restricted
  multiway wild bootstrap without the same randomization/exchangeability structure.
- **Human verdict:** [ ] accept formula with caveat [ ] reject [ ] require derivation

### Liddell & Kruschke (2018)

- **Full source:** Torrin M. Liddell and John K. Kruschke, “Analyzing Ordinal Data with Metric
  Models: What Could Possibly Go Wrong?” *Journal of Experimental Social Psychology*
  79:328–348.
- **Link:** [DOI](https://doi.org/10.1016/j.jesp.2018.08.009).
- **Research Plan use:** §8: model ordered grades as ordinal rather than interval-scaled.
- **Quotation — abstract only:**

  > “Ordinal data are commonly analyzed as if they were metric, but this can systematically
  > lead to errors.”

- **Assessment:** **PARTIAL.** It supports an ordinal model but not proportional odds,
  specific thresholds, or proportional-odds validity for this dataset.
- **Human verdict:** [ ] accept general claim [ ] supply full text [ ] reject

# 2. Verified-paywalled sources — quotations unavailable

No quotation is provided in this section. Metadata or publisher contents may have been checked,
but the closed primary text was not available. Supplying wording from training memory would be
fabrication. The user must supply the relevant pages or confirm access before any passage can
be treated as evidence.

## Dimension E

### Kyle & Crossley (2018)

- **Lookup:** “Measuring Syntactic Complexity in L2 Writing Using Fine-Grained Clausal and
  Phrasal Indices,” *Modern Language Journal* 102(2):333–349.
  [DOI](https://doi.org/10.1111/modl.12468).
- **Research Plan use:** §12 Dimension E: fine-grained clausal and phrasal complexity.
- **Quotation:** **USER ACCESS REQUIRED unless an authorized author copy is supplied.**
- **Human verdict:** [ ] supply article [ ] remove [ ] replace

## Dimension F

### Flesch (1948), “A New Readability Yardstick”

- **Lookup:** *Journal of Applied Psychology* 32(3):221–233.
  [DOI](https://doi.org/10.1037/h0057532).
- **Research Plan use:** §12 Dimension F: exclusion of classical readability formulas for
  roughly 15-token questions.
- **Quotation:** **USER ACCESS REQUIRED.**
- **Assessment:** Even if the formula is verified, a separate source is needed for the
  approximately-15-token unreliability claim.
- **Human verdict:** [ ] supply article [ ] remove claim [ ] find short-text source

## Dimension G

### Ingwersen & Järvelin (2005), _The Turn_

- **Lookup:** Springer. [DOI](https://doi.org/10.1007/1-4020-3851-8);
  ISBN 9781402038501.
- **Research Plan use:** §12 Dimension G: contextual information seeking, work tasks, and
  search tasks.
- **Requested passage:** Chapters 2 and 6.
- **Quotation:** **USER ACCESS REQUIRED.**
- **Human verdict:** [ ] supply chapters [ ] remove [ ] replace

### Anderson & Krathwohl (2001)

- **Lookup:** *A Taxonomy for Learning, Teaching, and Assessing*; ISBN 9780321084057.
- **Research Plan use:** §12 Dimension G: six cognitive-process categories.
- **Requested passage:** Chapters 3–5.
- **Quotation:** **USER ACCESS REQUIRED.**
- **Human verdict:** [ ] supply chapters [ ] remove [ ] replace

## Annotation and statistical methodology

### Krippendorff (2019), _Content Analysis_, 4th edition

- **Lookup:** SAGE. [DOI](https://doi.org/10.4135/9781071878781);
  ISBN 9781506395661.
- **Research Plan use:** §13: alpha, disagreement functions, reliability, and codebook
  discipline.
- **Requested passage:** Chapters 7 and 12, especially §§12.2–12.4.
- **Quotation:** **USER ACCESS REQUIRED.**
- **Human verdict:** [ ] supply chapters [ ] remove [ ] replace

### Cohen (1960), Cohen (1968), and Fleiss (1971)

- **Lookup:** Cohen (1960) [DOI](https://doi.org/10.1177/001316446002000104);
  Cohen (1968) [DOI](https://doi.org/10.1037/h0026256);
  Fleiss (1971) [DOI](https://doi.org/10.1037/h0031619).
- **Research Plan use:** §13: nominal kappa, weighted kappa, and multi-rater historical
  comparators.
- **Quotation:** **USER ACCESS REQUIRED for all three.**
- **Human verdict:** [ ] supply sources [ ] remove one/more [ ] replace

### Davidson & Flachaire (2008) and Cameron, Gelbach & Miller (2008)

- **Lookup:** Davidson–Flachaire
  [DOI](https://doi.org/10.1016/j.jeconom.2008.08.003); CGM
  [DOI](https://doi.org/10.1162/rest.90.3.414).
- **Research Plan use:** §5.3: Rademacher weights and few-cluster bootstrap inference.
- **Quotation:** **USER ACCESS REQUIRED for final journal versions.** Public working-paper
  copies should be audited separately before these are accepted as verified-open.
- **Human verdict:** [ ] audit working papers [ ] supply final articles [ ] remove/replace

### McCullagh (1980) and Agresti (2010)

- **Lookup:** McCullagh, “Regression Models for Ordinal Data,”
  [DOI](https://doi.org/10.1111/j.2517-6161.1980.tb01109.x); Agresti,
  *Analysis of Ordinal Categorical Data*, 2nd ed.,
  [DOI](https://doi.org/10.1002/9780470594001), ISBN 9780470082898.
- **Research Plan use:** §8: proportional odds and ordinal response interpretation.
- **Quotation:** **USER ACCESS REQUIRED for both.**
- **Human verdict:** [ ] supply sources [ ] remove one/both [ ] replace

### Closed implementation references

- **Sources:** Cameron & Trivedi (2005), *Microeconometrics*,
  [DOI](https://doi.org/10.1017/CBO9780511811241); Carroll et al. (2006),
  *Measurement Error in Nonlinear Models*,
  [DOI](https://doi.org/10.1201/9781420010138); Hardin & Hilbe (2013),
  *Generalized Estimating Equations*, [DOI](https://doi.org/10.1201/b13880); Davidson &
  MacKinnon (2004), *Econometric Theory and Methods*, ISBN 9780195123722.
- **Research Plan use:** general implementation references for binary models,
  measurement error, and GEE.
- **Quotation:** **USER ACCESS REQUIRED.**
- **Human verdict:** [ ] supply selected chapters [ ] remove source [ ] replace

# 3. To-verify or downgraded sources

# 4. Removed sources

## Ladusaw (1979/1980)

- **Source:** William A. Ladusaw, *Polarity Sensitivity as Inherent Scope Relations*.
- **Supplied evidence:** a 24-page University Microfilms extract containing the introductory
  pages and the beginning of Chapter II, not the complete dissertation.
- **Former Research Plan use:** NPI licensing as support for the project-defined
  `negative_conclusiveness` diagnostic.
- **Decision:** **REMOVED.** The extract verifies the work’s existence and its general treatment
  of polarity-sensitive items such as *any*, *yet*, and *already*, but it does not verify the
  required bridge from NPI licensing to the project’s operational negative-conclusiveness
  label. The source is not necessary to retain that later P4 label.
- **Impact:** documentation only; no completed feature value or code path depends on this
  citation.

## Prince (1981)

- **Source:** Ellen F. Prince, “Toward a Taxonomy of Given-New Information,” in
  *Radical Pragmatics*.
- **Former Research Plan use:** discourse-old/new and hearer-old/new context for Dimension D.
- **Decision:** **REMOVED at user request.** No verifiable primary copy was available to the
  user, so no quotation or locator could be confirmed.
- **Impact:** documentation only. Gundel et al. and Ariel remain the Dimension D theoretical
  anchors; no implemented feature or completed artifact depends on Prince.

## Aitchison (1986)

- **Source:** John Aitchison, *The Statistical Analysis of Compositional Data*.
- **Former Research Plan use:** external support for additive log-ratio coordinates in Layer 1.
- **Decision:** **REMOVED at user request.** No source copy was available for human
  verification, so no quotation or locator was confirmed.
- **Impact:** no completed Segment 1–3 computation depends on this citation; Layer 1 has not
  been implemented. The additive log-ratio transform remains a frozen mathematical
  parameterization in the Research Plan, but it must not be described as externally validated
  by Aitchison.

## Hamblin (1973)

- **Source:** Charles L. Hamblin, “Questions in Montague English.”
- **Former Research Plan use:** questions as sets of possible answers.
- **Decision:** **REMOVED at user request.** The user could not obtain a primary source for
  human verification.
- **Impact:** documentation only. Groenendijk & Stokhof and George retain the
  questions/answers and answerhood foundations used by Dimension B.

## Ariel (1990)

- **Source:** Mira Ariel, *Accessing Noun-Phrase Antecedents*.
- **Former Research Plan use:** accessibility hierarchy context for Dimension D.
- **Decision:** **REMOVED at user request.** The user could not obtain a primary source for
  human verification.
- **Impact:** documentation only. Gundel, Hedberg & Zacharski remains the direct
  referring-form theoretical anchor.

## Kiparsky & Kiparsky (1970)

- **Source:** Paul Kiparsky and Carol Kiparsky, “Fact.”
- **Former Research Plan use:** original factivity/presupposition analysis.
- **Decision:** **REMOVED.** The supplied file was Karttunen (1971), not the Kiparskys’
  source, so no primary passage could be human-verified.
- **Impact:** documentation only. Karttunen (1971) remains the verified contextual factivity
  anchor.

# 5. Impact guide if a source fails verification

- **Dossier-only or contextual source:** documentation change only; no rerun.
- **Same operational rule with a replacement source:** update citations/codebook; deterministic
  feature values remain unchanged.
- **Dimension A or E operational definition changes:** revise codebook, extractor, and tests;
  regenerate Segment 3 features. Segment 1 and Segment 2 science remain unchanged.
- **P4/annotation source fails:** revise before Segment 4; completed Segments 1–3 remain.
- **Statistical source fails:** revise future Segments 6–9; completed joins and deterministic
  features remain.
- **Research Plan or codebook changes:** the analysis lock may require an automated `join` and
  `features` refresh. This is reproducibility bookkeeping, not repeating experiments.
