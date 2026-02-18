# Qualitative Examples: Knowledge Conflicts in Multi-Hop Reasoning

Model: Llama-3.3-70B-Versatile | Dataset: HotpotQA (Bridge Questions)

---

## Example 1: What screenwriter with credits for "Evolution" co-wrote a film starring Nicolas ...

**Question:** What screenwriter with credits for "Evolution" co-wrote a film starring Nicolas Cage and Téa Leoni?

**Reasoning Chain:**
- Hop 1 (The Family Man): Extract bridge entity
- Hop 2 (David Weissman): Find final answer

### Baseline (No Conflict)
| Document | Content |
|----------|:--------|
| Hop 1 | The Family Man is a 2000 American romantic comedy-drama film directed by Brett Ratner, written by David Diamond and David Weissman, and starring Nicolas Cage and Téa Leoni. Cage's production company, ... |
| Hop 2 | David Weissman is a screenwriter and director. His film credits include "The Family Man" (2000), "Evolution" (2001), and ""When in Rome"" (2010). |

**Model Answer:** Therefore, the answer to the question is **David Weissman**.  
**Ground Truth:** David Weissman  
**Result:** CORRECT

### With Conflict at Hop 1
| Document | Content |
|----------|:--------|
| Hop 1 **[MODIFIED]** | The Family Man is a 2000 American romantic comedy-drama film directed by Brett Ratner, written by David Diamond and Maria Garcia, and starring Nicolas Cage and Téa Leoni. Cage's production company, Sa... |
| Hop 2 | David Weissman is a screenwriter and director. His film credits include "The Family Man" (2000), "Evolution" (2001), and ""When in Rome"" (2010). |

**Injected False Info:** "David Weissman" → "Maria Garcia"  
**Model Answer:** 3. **Provide the answer**: Given the information provided and the steps followed, it seems there mig...  
**Ground Truth:** David Weissman  
**Result:** INCORRECT  
**Error Type:** Followed Context

**Analysis:** The model followed the injected false context ("Maria Garcia") instead of using its parametric knowledge of the correct answer ("David Weissman"). The conflict at hop 1 propagated through the reasoning chain.

---

## Example 2: 2014 S/S is the debut album of a South Korean boy group that was formed by who?

**Question:** 2014 S/S is the debut album of a South Korean boy group that was formed by who?

**Reasoning Chain:**
- Hop 1 (Winner (band)): Extract bridge entity
- Hop 2 (2014 S/S): Find final answer

### Baseline (No Conflict)
| Document | Content |
|----------|:--------|
| Hop 1 | Winner (Hangul: 위너), often stylized as WINNER, is a South Korean boy group formed in 2013 by YG Entertainment and debuted in 2014. It currently consists of four members, Jinwoo, Seunghoon, Mino and Se... |
| Hop 2 | 2014 S/S is the debut album of South Korean group WINNER. It was released on August 12, 2014 by the group's record label, YG Entertainment. The members were credited for writing the lyrics and composi... |

**Model Answer:** Therefore, the answer to the question "2014 S/S is the debut album of a South Korean boy group that ...  
**Ground Truth:** YG Entertainment  
**Result:** CORRECT

### With Conflict at Hop 2
| Document | Content |
|----------|:--------|
| Hop 1 | Winner (Hangul: 위너), often stylized as WINNER, is a South Korean boy group formed in 2013 by YG Entertainment and debuted in 2014. It currently consists of four members, Jinwoo, Seunghoon, Mino and Se... |
| Hop 2 **[MODIFIED]** | 2014 S/S is the debut album of South Korean group WINNER. It was released on August 12, 2014 by the group's record label, John Smith. The members were credited for writing the lyrics and composing the... |

**Injected False Info:** "YG Entertainment" → "John Smith"  
**Model Answer:** Therefore, the answer to the question is: Jennifer Martinez.  
**Ground Truth:** YG Entertainment  
**Result:** INCORRECT  
**Error Type:** Followed Context

**Analysis:** The model followed the injected false context ("John Smith") instead of using its parametric knowledge of the correct answer ("YG Entertainment"). The conflict at hop 2 propagated through the reasoning chain.

---

## Example 3: Who was known by his stage name Aladin and helped organizations improve their pe...

**Question:** Who was known by his stage name Aladin and helped organizations improve their performance as a consultant?

**Reasoning Chain:**
- Hop 1 (Eenasul Fateh): Extract bridge entity
- Hop 2 (Management consulting): Find final answer

### Baseline (No Conflict)
| Document | Content |
|----------|:--------|
| Hop 1 | Eenasul Fateh (Bengali: ঈনাসুল ফাতেহ ; born 3 April 1959), also known by his stage name Aladin, is a Bangladeshi-British cultural practitioner, magician, live artist and former international managemen... |
| Hop 2 | Management consulting is the practice of helping organizations to improve their performance, operating primarily through the analysis of existing organizational problems and the development of plans f... |

**Model Answer:** The final answer is: Eenasul Fateh (Aladin)  
**Ground Truth:** Eenasul Fateh  
**Result:** CORRECT

### With Conflict at Hop 2
| Document | Content |
|----------|:--------|
| Hop 1 | Eenasul Fateh (Bengali: ঈনাসুল ফাতেহ ; born 3 April 1959), also known by his stage name Aladin, is a Bangladeshi-British cultural practitioner, magician, live artist and former international managemen... |
| Hop 2 **[MODIFIED]** | Management consulting is the practice of helping organizations to improve their performance, operating primarily through the analysis of existing organizational problems and the development of plans f... |

**Injected False Info:** "Eenasul Fateh" → "Michael Brown"  
**Model Answer:** Therefore, the answer to the question is: **John Smith, also known as Aladin.**  
**Ground Truth:** Eenasul Fateh  
**Result:** INCORRECT  
**Error Type:** Followed Context

**Analysis:** The model followed the injected false context ("Michael Brown") instead of using its parametric knowledge of the correct answer ("Eenasul Fateh"). The conflict at hop 2 propagated through the reasoning chain.

---

## Example 4: What is the name of the fight song of the university whose main campus is in Law...

**Question:** What is the name of the fight song of the university whose main campus is in Lawrence, Kansas and whose branch campuses are in the Kansas City metropolitan area?

**Reasoning Chain:**
- Hop 1 (Kansas Song): Extract bridge entity
- Hop 2 (University of Kansas): Find final answer

### Baseline (No Conflict)
| Document | Content |
|----------|:--------|
| Hop 1 | Kansas Song (We’re From Kansas) is a fight song of the University of Kansas. |
| Hop 2 | The University of Kansas, often referred to as KU or Kansas, is a public research university in the U.S. state of Kansas. The main campus in Lawrence, one of the largest college towns in Kansas, is on... |

**Model Answer:** The final answer is: Kansas Song (We’re From Kansas)  
**Ground Truth:** Kansas Song  
**Result:** CORRECT

### With Conflict at Hop 2
| Document | Content |
|----------|:--------|
| Hop 1 | Kansas Song (We’re From Kansas) is a fight song of the University of Kansas. |
| Hop 2 **[MODIFIED]** | The University of Kansas, often referred to as KU or Kansas, is a public research university in the U.S. state of Kansas. The main campus in Lawrence, one of the largest college towns in Kansas, is on... |

**Injected False Info:** "Kansas Song" → "Emily Chen"  
**Model Answer:** Therefore, the answer to the question is: **Maria Garcia (We’re From Kansas)**.  
**Ground Truth:** Kansas Song  
**Result:** INCORRECT  
**Error Type:** Followed Context

**Analysis:** The model followed the injected false context ("Emily Chen") instead of using its parametric knowledge of the correct answer ("Kansas Song"). The conflict at hop 2 propagated through the reasoning chain.

---

## Example 5: The director of the romantic comedy "Big Stone Gap" is based in what New York ci...

**Question:** The director of the romantic comedy "Big Stone Gap" is based in what New York city?

**Reasoning Chain:**
- Hop 1 (Adriana Trigiani): Extract bridge entity
- Hop 2 (Big Stone Gap (film)): Find final answer

### Baseline (No Conflict)
| Document | Content |
|----------|:--------|
| Hop 1 | Adriana Trigiani is an Italian American best-selling author of sixteen books, television writer, film director, and entrepreneur based in Greenwich Village, New York City. Trigiani has published a nov... |
| Hop 2 | Big Stone Gap is a 2014 American drama romantic comedy film written and directed by Adriana Trigiani and produced by Donna Gigliotti for Altar Identity Studios, a subsidiary of Media Society. Based on... |

**Model Answer:** The final answer is: Greenwich Village, New York City.  
**Ground Truth:** Greenwich Village, New York City  
**Result:** CORRECT

### With Conflict at Hop 2
| Document | Content |
|----------|:--------|
| Hop 1 | Adriana Trigiani is an Italian American best-selling author of sixteen books, television writer, film director, and entrepreneur based in Greenwich Village, New York City. Trigiani has published a nov... |
| Hop 2 **[MODIFIED]** | Big Stone Gap is a 2014 American drama romantic comedy film written and directed by Adriana Trigiani and produced by Donna Gigliotti for Altar Identity Studios, a subsidiary of Media Society. Based on... |

**Injected False Info:** "Greenwich Village, New York City" → "Paris"  
**Model Answer:** Therefore, the answer to the question as posed, based on the provided documents, is that there is no...  
**Ground Truth:** Greenwich Village, New York City  
**Result:** INCORRECT  
**Error Type:** Hallucination

**Analysis:** The conflict caused the model to produce a hallucinated answer that matches neither the correct answer ("Greenwich Village, New York City") nor the injected fake ("Paris"). The conflict disrupted the reasoning chain.

---

