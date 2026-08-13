# Redistributing Ilm-NMR-P31: upstream licensing and the "link, don't host" proposal

**This is not legal advice, and I am not a lawyer.** It is a research memo describing how
the relevant law is generally understood, written so that the maintainer can put sharp
questions to KIT's legal office instead of vague ones. Every legal source cited below was
fetched and read; where I could not verify something I say so inline, marked
`[UNVERIFIED]`. Do not act on this without the legal office signing off on the two or
three points flagged as decisive.

---

## The two answers, up front

### Q1 — "The authors already published it, so it's more or less public. Can we take it?"

**No, that reasoning is wrong — but the conclusion is closer to "yes, mostly" than you'd
expect, for entirely different reasons.**

Publishing something under CC-BY-4.0 cannot grant rights the publisher never held. The
licence says so itself: the grant is limited to rights "**that the Licensor has authority
to license**" ([CC BY 4.0 §1(7)](https://creativecommons.org/licenses/by/4.0/legalcode.en)),
and §5(a) expressly disclaims any warranty of title or non-infringement. So the CC-BY tag
on the Zenodo record is not a shield. That is the `nemo dat` point and it is correct.

But "is it protected at all?" is the question that actually decides this, and the answer
splits the dataset cleanly in two:

| Source block | 3D structures | EU database right | Status |
|---|---:|---|---|
| **HBP** — Tebby, CRC Press, **1991** | **6,608** | Term expired **31 Dec 2012** (§137g(2) UrhG); maker is US, likely outside Art. 11 | **Very likely clean** |
| **Lan1/2/3** — Landolt-Börnstein III/40H, Springer, **2013/14** | **5,114** | Term runs to **~2029**; German maker; investment is in *obtaining*, not creating | **This is the exposure** |
| NMRexp (journal-scraped) | 0 | — | n/a |

The individual chemical-shift values are **not protected as such** anywhere — not in
Germany, not in the US. What is potentially protected is the *compilation*. And the
compilation right on the larger of the two blocks (56% of your 3D structures) ran out
thirteen years ago.

So the honest position is: **the Landolt-Börnstein-derived 5,114 structures are the
problem, and they are separable.** Your `source` column already distinguishes them. See
[the middle path](#the-option-nobody-has-proposed) below.

### Q2 — "What if we fetch from Zenodo instead of hosting it ourselves?"

**Legally it does not help you, and it is the wrong counterparty. Engineering-wise it is a
structural change, not a small one, and it destroys the main thing the project is for.**

Not hosting avoids "extraction". It does not avoid "**re-utilisation**", which is defined
as "any form of making available to the public ... by ... on-line **or other forms of
transmission**" (Art. 7(2)(b), Directive 96/9/EC). The CJEU in *Innoweb* held that an
operator who merely publishes a retrieval mechanism re-utilises the database — the
mechanism "comes close to the manufacture of a parasitical competing product ... **albeit
without copying the information stored in the database concerned**"
([C‑202/12, para 48](https://eur-lex.europa.eu/legal-content/EN/TXT/HTML/?uri=CELEX:62012CJ0202)).
*CV-Online* puts it even more bluntly: hyperlinks are "merely **external manifestations, of
secondary importance**, of that extraction and that re-utilisation"
([C‑762/19, para 37](https://eur-lex.europa.eu/legal-content/EN/TXT/HTML/?uri=CELEX:62019CJ0762)).

And the deeper problem: **the party you'd be linking to is not the party with the
complaint.** The Ilm-NMR authors already consented — their CC-BY grant over their own layer
is valid. Springer never consented to anything and is not on the other end of that link.
Routing through Zenodo changes nothing about Springer's position.

On the engineering side: `connectors.py` already contains a `ZenodoSource`, but it is
imported **only** by the build-time `create_graph_datasets__*.py` scripts. Nothing in the
load path touches it. Wiring it into `main.py:ensure_dataset()` means a new source
abstraction, a version-aware cache (the current one is keyed on `{name}.{typ}` with no
version and no checksum), a pinned client-side processing pipeline, a 155 MB download per
user instead of a gzipped `.mpack`, and the loss of offline use. **Details below.**

---

## 1. The analysis: what is actually protected

### 1.1 The individual shift values — not protected, clearly

Three independent confirmations, and this one is not close:

- **Recital 45**, Directive 96/9/EC: "the right to prevent unauthorized extraction and/or
  re-utilization **does not in any way constitute an extension of copyright protection to
  mere facts or data**."
- **Recital 46**: the right "should **not give rise to the creation of a new right in the
  works, data or materials themselves**."
- **Art. 3(2)**: "The copyright protection of databases ... **shall not extend to their
  contents**."

German law is the same: §2(2) UrhG protects only a *persönliche geistige Schöpfung*, which
a measured ppm value is not. In the US, *Feist* reaches the identical result by a different
route: "**No one may claim originality as to facts** ... The distinction is one between
creation and discovery: The first person to find and report a particular fact has not
created the fact"
([499 U.S. 340](https://www.law.cornell.edu/supremecourt/text/499/340)).

So nobody can stop you from using a ³¹P shift value. The fight, if there is one, is over
taking *the collection*.

### 1.2 Do the books count as "databases" at all? — Yes

This was the right question to ask, and the answer is yes, printed reference works qualify.
Art. 1(2) defines a database as a collection "individually accessible by electronic **or
other means**" — the "or other means" is doing exactly this work. The CJEU confirmed the
breadth in *Fixtures Marketing v OPAP*: the term "refers to **any collection** of works,
data or other materials, separable from one another without the value of their contents
being affected, including a method or system of some sort for the retrieval of each of its
constituent materials"
([C‑444/02, summary point 1](https://eur-lex.europa.eu/legal-content/EN/TXT/HTML/?uri=CELEX:62002CJ0444)).

A ³¹P NMR reference table indexed by compound is squarely inside that. Germany transposes
it identically in §87a(1) UrhG ("*einzeln mit Hilfe elektronischer Mittel **oder auf andere
Weise** zugänglich*").

### 1.3 Is there "substantial investment"? — Yes, and this cuts *against* you

There's a well-known escape hatch here that does **not** apply. In *BHB* and *OPAP* the
CJEU held that investment in **creating** data doesn't count — only investment in
**obtaining** it, i.e. "the resources used to **seek out existing independent materials and
collect them** in the database"
([C‑203/02, summary point 1](https://eur-lex.europa.eu/legal-content/EN/TXT/HTML/?uri=CELEX:62002CJ0203)).
That is why the horseracing board lost: they created the fixture data themselves.

Springer and Tebby did the opposite. They did not measure a single shift; they trawled
thousands of primary papers (the derived dataset cites 3,648 references at v1, 6,853 at
v3) and collected the values. **That is the paradigm case of "obtaining."** So unlike
*BHB*, the substantial-investment limb is likely satisfied for both books. No help here.

### 1.4 Term — and this is where the case actually turns

**Sui generis protection lasts 15 years**, not 70. §87d UrhG: "*Die Rechte des
Datenbankherstellers erlöschen **fünfzehn Jahre** nach der Veröffentlichung der
Datenbank*." Art. 10(1)–(2) is the same.

**Tebby, *Handbook of Phosphorus-31 NMR Data*, CRC Press, 1991** — the source of **6,608 of
your 11,722** 3D structures:

- §137g(2) UrhG: for databases made between **1 Jan 1983 and 31 Dec 1997**, "*Die
  Schutzfrist beginnt in diesen Fällen am 1. Januar 1998*." Fifteen years from 1 Jan 1998
  → **expired 31 December 2012.** (Art. 14(3)/(5) of the Directive is to the same effect.)
- Independently, **Art. 11** limits the right to makers who are EU nationals/residents or
  EU-formed companies. CRC Press is Boca Raton, Florida. Art. 11(3) allows the Council to
  extend protection to third countries by agreement, and Recital 56 conditions it on
  reciprocity. `[UNVERIFIED: I could not retrieve a Council decision under Art. 11(3)
  covering the US. My searching surfaced only a reference to an Isle of Man arrangement,
  which I did not retrieve or confirm. Treat the "no US extension" point as probable but
  unconfirmed — the term point above is the solid one and does not depend on it.]`

**Landolt-Börnstein III/40H, Springer, 2013/2014** — the source of **5,114** structures.
The Ilm-NMR paper itself dates it: it digitised the Landolt-Börnstein "**from 2014**"
([J. Cheminform. 2023](https://jcheminf.biomedcentral.com/articles/10.1186/s13321-023-00792-y),
Methods; the cited reference is Gupta, Lechner, Kumar et al., vol. 40, Springer, 2013).
Springer-Verlag is German, so Art. 11 is satisfied. **Term runs to roughly the end of 2028
or 2029.** This one is live, and 5,114 structures out of a 2,713 + 1,596 + 805 breakdown is
plainly a substantial part both quantitatively (it is the bulk of those subvolumes' 3D-
representable content) and qualitatively — *BHB* defines the qualitative test as "the scale
of the investment in the obtaining, verification or presentation of the contents ...
**regardless of whether that subject represents a quantitatively substantial part**"
([C‑203/02, summary point 3](https://eur-lex.europa.eu/legal-content/EN/TXT/HTML/?uri=CELEX:62002CJ0203)).

One more trap worth knowing: §87a(1) sentence 2 — a database "*wesentlich geänderte*"
counts as a **new** database with a fresh 15-year term, if the change required substantial
investment (Art. 10(3) likewise). So a revised Landolt-Börnstein subvolume would restart
the clock. Worth checking whether III/40H has been revised since 2014.

### 1.5 Copyright in the compilation — a separate, weaker claim

Distinct from the database right, §4 UrhG protects a *Sammelwerk* where "*die Auswahl oder
Anordnung der Elemente eine persönliche geistige Schöpfung*" is — for 70 years post mortem
auctoris. But it protects only the **selection or arrangement**, expressly "*unbeschadet
eines an den einzelnen Elementen gegebenenfalls bestehenden Urheberrechts*".

The Ilm-NMR authors have a decent answer to this: per the paper's Methods, they re-drew
**every structure by hand** in ACD/ChemSketch, generated their own SMILES, their own IDs
and their own schema. They took the values, not the presentation. Whether taking *the
entire selection* is itself an infringement of the selection is the arguable residue, and
it is genuinely arguable — but it is a weaker claim than the database right and I would not
expect it to be the thing that bites.

### 1.6 The TDM exceptions do not cover you — this is clear

This is the argument people reach for and it does not work.

**§44b UrhG** (implementing DSM Art. 4) permits "*Vervielfältigungen von rechtmäßig
zugänglichen Werken für das Text und Data Mining*" — **reproductions only**. It says nothing
about distribution or making available. It also requires deletion once no longer needed
("*Die Vervielfältigungen sind zu löschen, wenn sie für das Text und Data Mining nicht mehr
erforderlich sind*") and is subject to machine-readable opt-out (§44b(3)). §87c(1) Nr. 4
extends it to substantial parts of databases — again, reproduction only.

**§60d UrhG** (DSM Art. 3) is broader for research organisations and *does* permit making
copies available — but only "*einem bestimmt abgegrenzten Kreis von Personen für deren
gemeinsame wissenschaftliche Forschung*" and to individual third parties for quality
verification, and it must **stop** when the research concludes (§60d(4)).

Neither of these authorises publishing a corpus on PyPI and a public Nextcloud to an
unbounded worldwide audience. **The TDM exceptions let you train a model on the data. They
do not let you ship the data.** ChemMatData's entire product is shipping the data.

(Also note DSM Art. 7(1): contractual terms overriding the Art. 3 research exception are
unenforceable — helpful for *your own* mining, irrelevant to redistribution.)

### 1.7 Good-faith reliance — helps with damages, not with an injunction

Under the database right, infringement does not require knowledge. What good faith affects
is **remedies**, and [§97 UrhG](https://www.gesetze-im-internet.de/urhg/__97.html) splits
them cleanly. §97(1) gives removal and injunctive relief against anyone who "*das
Urheberrecht oder ein anderes nach diesem Gesetz geschütztes Recht **widerrechtlich**
verletzt*" — unlawfulness only, **no fault required**, and it extends to a first
threatened infringement. §97(2) gives damages only against someone who acts "*vorsätzlich
oder fahrlässig*". So:

- **Takedown / injunction:** available against you even if you acted in perfect good faith.
- **Damages:** require fault. Reliance on a facially valid CC-BY grant from a peer-reviewed
  J. Cheminformatics paper is decent evidence against negligence.
- **But:** this memo destroys that. Once you have read an analysis saying the shifts came
  out of a 2014 Springer reference work with no documented permission, you have actual
  knowledge, and the good-faith argument for *future* distribution is gone. That is an
  unwelcome consequence of asking the question, and I'd rather say it than not.

*GS Media* ([C‑160/15](https://eur-lex.europa.eu/legal-content/EN/TXT/HTML/?uri=CELEX:62015CJ0160))
does make knowledge and profit-motive decisive — but it is a **copyright** case about
communication to the public under Art. 3(1) of Directive 2001/29, not a database-right
case. Do not assume its knowledge test transfers to §87b.

### 1.8 The US contrast — the same act is lawful there

*Feist* killed "sweat of the brow" outright: the doctrine's "most glaring" flaw was that
"it extended copyright protection in a compilation **beyond selection and arrangement** —
the compiler's original contributions — **to the facts themselves**," which "eschewed the
most fundamental axiom of copyright law"
([499 U.S. 340](https://www.law.cornell.edu/supremecourt/text/499/340)). US compilations
get only thin protection in original selection/arrangement, per 17 U.S.C. §103(b), and
**there is no US sui generis database right** — Congress considered database bills in the
late 1990s/2000s and none passed. `[UNVERIFIED: I did not retrieve legislative history for
the failed database bills; the absence of a federal database right is however evident from
the Feist framework and from the Directive's own Recital 56 reciprocity condition, which
would be pointless if the US had an equivalent.]`

**Concretely:** a US-based maintainer redistributing the Ilm-NMR shift values, including
the Landolt-Börnstein-derived ones, is doing something lawful. You, in Germany, hosting on
a German Nextcloud, are not in the same position for the Landolt-Börnstein subset.
Liability follows **where you act**, not where your users are — so a worldwide user base
does not dilute your exposure, and your US users' lawful downloads do not launder your
distribution.

---

## 2. Question 2(a): does linking change the legal position?

**Extraction:** yes, avoided. Art. 7(2)(a) requires "the permanent or temporary **transfer**
of all or a substantial part of the contents ... **to another medium**". If no bytes touch
your server, you don't transfer. (Note *Directmedia*, [C‑304/07](https://eur-lex.europa.eu/legal-content/EN/TXT/HTML/?uri=CELEX:62007CJ0304),
holds that even manual on-screen consultation followed by re-keying is capable of being
extraction — so this is narrower than it sounds, but the no-hosting version does escape
it.)

**Re-utilisation:** no, not avoided. This is where the proposal fails.

*Innoweb* is the case on point, and it is close to fatal. The Court held that the operator
of a dedicated meta search engine re-utilises the database, reasoning that:

> "the relevant aspect of the activity of the operator ... — that is to say, **making that
> search engine available on the Internet** — comes close to the manufacture of a
> parasitical competing product ..., **albeit without copying the information stored in the
> database concerned**. In view of the search options offered, such a dedicated meta search
> engine **resembles a database, but without having any data itself**." (para 48)

And on timing — which kills the "but the *user* does the downloading" defence:

> "actions actually carried out by the operator ... occur **prior to** activities carried
> out by end users and before a search is made in response to a given query. Such actions
> consist in making a dedicated meta search engine available on the Internet..." (para 29)

The operator's act is complete when the mechanism ships. That is `pip install
chem_mat_database`.

*CV-Online* ([C‑762/19](https://eur-lex.europa.eu/legal-content/EN/TXT/HTML/?uri=CELEX:62019CJ0762))
then dismissed the "but it's just a link" framing directly: hyperlinks and meta tags "are
merely **external manifestations, of secondary importance**, of that extraction and that
re-utilisation" (para 37).

**Why *Svensson* doesn't rescue this.** *Svensson* holds that "the provision on a website of
clickable links to works freely available on another website does not constitute an 'act of
communication to the public'"
([C‑466/12, operative part](https://eur-lex.europa.eu/legal-content/EN/TXT/HTML/?uri=CELEX:62012CJ0466)).
That is a **copyright** holding under Art. 3(1) of Directive 2001/29. It is not a holding
about Art. 7 of Directive 96/9, and *Innoweb* — decided two months *earlier* — reaches the
opposite result on the database right for functionally similar conduct. The two coexist
because they concern different rights. Anyone telling you "linking is always fine, see
*Svensson*" is conflating them.

**The one authority that genuinely helps you** is *Renckhoff*, and it is worth understanding
why. The Court distinguished linking from re-hosting precisely on control:

> "As regards the act of communication constituted by the posting on a website of a
> hyperlink ..., the preventive nature of the rights of the holder **are preserved, since it
> is open to the author, if he no longer wishes to communicate his work on the website
> concerned, to remove it** ..., rendering obsolete any hyperlink leading to it. However
> ... the posting on another website of a work gives rise to a **new communication,
> independent** of the communication initially authorised."
> ([C‑161/17, para 44](https://eur-lex.europa.eu/legal-content/EN/TXT/HTML/?uri=CELEX:62017CJ0161))

That is a real difference and it is in your favour on the copyright layer: if Springer ever
forced a takedown at Zenodo, a link-based loader would go dark automatically, whereas your
Nextcloud copy would not. It does not, however, answer the database-right point.

**Caching on the user's machine** does not change the operator's position. The user may have
their own §44b defence for their own copy; you don't inherit it, and per *Innoweb* para 29
your act completed before theirs began.

**Shipping the processing recipe alongside makes it worse, not better.** *Innoweb*'s first
limb is whether the tool "provides the end user with a search form which essentially offers
the same range of functionality as ... the database site". A loader that fetches the raw
Zenodo archive *and* applies your curation, filtering and featurisation offers materially
*more* functionality than the source. You would be arguing that you're less of a
re-utiliser than Innoweb while shipping a strictly more capable product.

### The genuinely open question

*CV-Online* did not stop at "this is re-utilisation." It made harm a **condition**:

> re-utilisation "**may be prohibited** by the maker of such a database **where those acts
> adversely affect its investment** ..., namely that they constitute a risk to the
> possibility of redeeming that investment through the normal operation of the database in
> question" (operative part)

and the Court framed the exercise as striking "a **fair balance**" between makers redeeming
investment and "that of users and competitors ... in having access to the information ...
and the possibility of **creating innovative products** based on that information" (para
140).

Both *Innoweb* and *CV-Online* involved commercial products that diverted advertising
revenue from a competitor. A non-commercial academic loader that sends users to Zenodo —
where Springer earns nothing either way — is a long way from that fact pattern. **Nobody has
litigated whether the harm condition is met when the "maker" whose investment is at risk is
two hops upstream of the resource being linked, and when the linker is non-commercial.**

That is the untested question, and it is the one worth putting to the legal office. My
reading is that ChemMatData has a real argument on the harm limb and a poor one on the
"is it re-utilisation at all" limb — which means the linking scheme buys you a *defence*,
not *safety*, at a large engineering cost.

---

## 3. Question 2(b): engineering viability

I read `main.py`, `cache.py`, `dataset.py`, `web.py`, `config.py`, `connectors.py` and
`cli.py` in the worktree. **This is a structural change, not a small one.** Six specific
reasons.

### 3.1 The load path has exactly one source model

`load_graph_dataset` / `load_smiles_dataset` / `load_xyz_dataset` all funnel into
`chem_mat_data/main.py:77` `ensure_dataset()`, which does: local file → `config.cache` →
`web.py:NextcloudFileShare`. `web.py:269` `construct_download_url()` hard-codes a Nextcloud
WebDAV path (`{base_url}/public.php/dav/files/{share_token}/{file_name}`), and
`ensure_dataset` assumes one file per dataset per extension with a `.gz`/`.zip` fallback
ladder. There is a `FILE_SHARE_CLASS_MAP` in `main.py:26` but it has exactly one entry
(`'nextcloud'`), and `metadata.yml` is fetched from the same share to decide what exists at
all (`main.py:145`). A per-dataset source override is new plumbing through all of it.

### 3.2 `ZenodoSource` exists — but only at build time

`connectors.py` has `ZenodoSource`, `FigshareSource` and `FileDownloadSource`. Grepping for
importers: **every one is a `chem_mat_data/scripts/create_graph_datasets__*.py`** plus
`tests/test_connectors.py`. Nothing in the user-facing load path imports `connectors` at
all. That separation is currently doing real work — it's the line between "maintainer's
machine, once" and "user's machine, every time." The proposal erases it.

`ZenodoSource` also isn't built for this. `prepare()` does `tempfile.mkdtemp()`, `fetch()`
downloads the whole archive and extracts it, and `cleanup()` runs from `__del__`. It is
**deliberately ephemeral and has no `Cache` integration whatsoever**. On the code as it
stands, every single load re-downloads the whole archive.

### 3.3 The cache cannot express versions

`cache.py:171` `construct_base_name()` returns `f'{name}.{typ}'`. That is the entire cache
key. No version, no checksum, no source URL. Two different Zenodo versions of Ilm-NMR-P31
would occupy the same slot with nothing to distinguish them, and `contains_dataset()`
(`cache.py:21`) checks only whether the `.yml` exists — so a stale v2.0 cache entry would
satisfy a v3.0 request silently. Adding version-awareness means changing the key format,
which invalidates every existing user's cache.

### 3.4 Reproducibility: this record is a moving target

I queried the Zenodo API. The concept record is `8260782`; there are **five versions in
under three years**:

| Record | DOI | Version | Published | Files |
|---|---|---|---|---:|
| 8260783 | `10.5281/zenodo.8260783` | 1.0 | 2023-08-18 | 4 |
| 15658468 | `10.5281/zenodo.15658468` | 2.0 | 2025-06-13 | 5 |
| 17649620 | `10.5281/zenodo.17649620` | 2.1 | 2025-11-19 | 5 |
| 17952686 | `10.5281/zenodo.17952686` | 2.2 | 2025-12-17 | 5 |
| **20762603** | `10.5281/zenodo.20762603` | **3.0** | **2026-06-19** | 5 |

The content is not stable in any sense that matters:

- Entry count went **14,250 → 43,317**; molecules **13,730 → 39,107**; references
  **3,648 → 6,853**.
- **`xyz_nmr_data.zip` did not exist before v2.0.** The 3D geometries you want appeared in
  June 2025, then grew 136.4 MB → 155.1 MB by v3.0.
- v3.0 **removed the `origin` column** and changed the SDF `NMREDATA_*` tags.

So: a **concept-DOI** pin (`10.5281/zenodo.8260782`) always resolves to the newest version
and would have silently swapped a 14k-molecule dataset for a 43k one under your users'
feet, and would have broken any script reading `origin`. A **versioned-DOI** pin
(`20762603`) is stable — but then you hand-re-pin on every upstream release, which is a
maintenance burden you don't have today, and Zenodo depositors can restrict or withdraw old
versions. Neither option is as reproducible as a frozen `.mpack` on your own share.

Note also that the paper cites `10.5281/zenodo.8260783` — the **versioned** DOI of v1.0.
The v1.0 the paper describes doesn't even contain 3D coordinates.

### 3.5 It discards the entire point of the `.mpack` model

The value proposition of ChemMatData is that expensive processing happens **once,
centrally**. `load_graph_dataset` is `ensure_dataset(...)` then `load_graphs(file_path)` —
a msgpack deserialise into arrays. Fetch-at-load-time means every user instead:

- downloads **155 MB** (`xyz_nmr_data.zip` at v3.0) rather than a gzipped `.mpack`;
- extracts it;
- re-runs the RDKit parsing and featurisation in `processing.py` (56 KB) and the
  `XyzDataset` / `GraphDataset` path in `dataset.py` (53 KB);
- and gets results that depend on **their** RDKit/OpenBabel versions.

That last one is the serious one. Today an RDKit change is your problem, absorbed once at
build time. Under the proposal it becomes a silent per-user data divergence — the same
`load_xyz_dataset('ilm_nmr_p31')` call returning different featurisations on different
machines. To avoid that you'd have to pin and ship a reproducible processing environment,
which is a much bigger commitment than a download path.

### 3.6 Offline use, rate limits, failure modes

Offline use currently works: `main.py:112` returns early if the file is on disk, and
`main.py:124` consults the cache before any network call. A Zenodo-backed path is offline-
capable only to the extent you build a cache for it — i.e. you re-implement the thing you
already have, against a second backend.

You also inherit a dependency you don't control: Zenodo availability, Zenodo's rate limits
(a CI matrix or a lecture hall behind one NAT is a realistic trigger)
`[UNVERIFIED: I did not fetch Zenodo's current published rate-limit figures, so treat the
severity as unquantified — the dependency itself is certain, the thresholds are not]`, and
per-file URLs that change shape between record versions. `ZenodoSource.fetch()` currently
does a bare `requests.get(..., stream=True)` with `raise_for_status()` and no retry, no
backoff, no resume, no checksum verification — for a 155 MB download.

### Verdict

Small change: adding a `source:` field to `metadata.yml` and a build-time connector. **You
already have that.** Structural change: making the *user's* load path fetch from arbitrary
third-party repositories, with a version-aware cache, a pinned client-side processing
environment, and a second failure domain. That is the one being proposed, and it buys a
legal defence that *Innoweb* and *CV-Online* suggest is weak anyway.

---

## 4. <a name="the-option-nobody-has-proposed"></a>The option nobody has proposed

The framing so far has been all-or-nothing: ship everything, or ship nothing, or link. The
term analysis in §1.4 opens a fourth door.

**Ship the HBP-derived subset. Hold back the Landolt-Börnstein-derived subset until 2029.**

- **HBP (Tebby / CRC Press, 1991): 6,608 structures — 56% of your 3D data.** The German
  database right expired **31 Dec 2012** under §137g(2) UrhG. The values were never
  protected as such. The compilation-copyright claim on selection/arrangement is weak given
  that the Ilm-NMR authors re-drew every structure and built their own schema. This block is
  on firm ground.
- **Lan1/2/3 (Landolt-Börnstein / Springer, 2013/14): 5,114 structures.** Live database
  right until ~2029. This is the block to hold, or to get permission for.

You said the 3D subset "is 100% book-derived and cannot be filtered clean" — which is true
as stated, but the two *books* are not in the same legal position, and the dataset
distinguishes them. Your own count is broken out by source (`HBP` 6,608; `Lan1` 2,713;
`Lan2` 1,596; `Lan3` 805), so the filter is a `source` column predicate, not a research
project.

A 6,608-structure ³¹P dataset with DFT geometries is still a genuinely useful public
resource, and it ships now instead of in three years.

**In parallel, ask Springer.** A written permission request to Springer Nature for use of
Landolt-Börnstein III/40H-derived shift values in a non-commercial academic ML dataset is
not an outlandish ask, it costs one email, and a "yes" resolves the whole question. It is
also worth telling the Ilm-NMR authors what you found — they may already hold a permission
that simply isn't documented in the paper, which would collapse this entire analysis.

---

## 5. A general rule for the collection

Proposed addition to `.claude/skills/dataset-extraction/references/license.md`. It extends
the existing "Trap 1 — the upstream database" section, which already gets the principle
right but stops one step short.

### 5.1 The rule

> **A dataset's licence describes the last link in its provenance chain. Your right to
> redistribute is determined by the *most restrictive* link, not the last one.**
>
> Every compiled dataset has a chain: primary measurement → some compilation → possibly
> another compilation → the record you found. A CC-BY tag on the final record binds only
> the rights that depositor actually held. CC BY 4.0 §1(7) limits the grant to rights "that
> the Licensor has authority to license", and §5(a) disclaims any warranty of title — the
> licence itself tells you it is not a guarantee.

### 5.2 The classification

Add a **BLOCK** tier. The current table has only PASS and ESCALATE, which forces genuinely
unshippable cases into the same bucket as merely-awkward ones.

| Verdict | Trigger |
|---|---|
| **BLOCK** | Data digitised or scraped from a **commercially sold** reference work, database or journal corpus, where (a) no publisher permission is documented, **and** (b) the source was published **within the last 15 years**, **and** (c) the source publisher is EU/EEA-based. All three must hold. |
| **ESCALATE** | Any two of (a)/(b)/(c). Also: everything already in the existing ESCALATE row (NC, SA, copyleft, bespoke terms, no licence stated). Also: an upstream reference work **more than 15 years old** where the *selection* is being taken wholesale rather than the values. |
| **PASS** | Upstream is primary literature, a public database with permissive terms, the depositor's own measurements or computations, **or** a compilation whose sui generis term has expired **and** whose values are being taken rather than its arrangement. |

### 5.3 Three new manifest fields

The existing `upstream_database` / `upstream_terms_url` / `upstream_license` fields cannot
express any of the above. Add:

```json
{
  "upstream_publication_year": 2014,
  "upstream_publisher_jurisdiction": "DE",
  "upstream_is_commercial_product": true
}
```

These are cheap to establish (they're on the book's copyright page) and they mechanise the
BLOCK/ESCALATE decision instead of leaving it to judgement at the point where, as
`license.md` already observes, "an ambiguous license gets read charitably."

### 5.4 Two rules of thumb worth stating flatly

- **"It's already public" is never an answer.** Publication by a third party is not a grant.
  Nor is "everyone else uses this dataset." The existing file says this about missing
  licences; it should say it about *upstream* rights too, which is the harder case.
- **Linking is not a workaround.** If a dataset would be BLOCKed when hosted, it is BLOCKed
  when fetched at load time. *Innoweb* para 48 and *CV-Online* para 37 are the reason.
  Record this so that the next person to have this idea finds the answer instead of
  re-deriving it.

### 5.5 One structural gap

`metadata.yml` has **46 datasets and no licence field at all**. The keys in use are:
`compounds`, `description`, `raw`, `tags`, `target_type`, `targets`, `verbose`, `sources`,
`target_descriptions`, `full`, `notes`, `category`, `min_version`. `sources` is free text
and appears on 36 of 46.

So the skill's licence gate produces a finding that is then **discarded** — it lives in
`.dataset_work/<slug>/manifest.yml` and never reaches the shipped metadata or the user. A
`license:` block in `metadata.yml`, surfaced by `cmdata info`, would mean attribution
obligations actually travel with the data (which CC-BY §3(a) requires of *you* when you
Share) and that any future problem is auditable across the collection rather than one
dataset at a time. That is a small change with a large payoff, and it is worth doing
regardless of how Ilm-NMR-P31 is resolved.

---

## 6. What to ask the legal office

Not "is this OK". These:

1. **Term and beneficiary.** Do you agree the Tebby/CRC 1991 handbook's sui generis right
   expired 31 Dec 2012 under §137g(2) UrhG, and that as a US-made database it likely never
   attracted the right under §87e-relevant Art. 11 in the first place? If so, is
   redistributing the 6,608 HBP-derived structures acceptable, given that the shift values
   themselves are unprotected facts?
2. **The live block.** For the 5,114 Landolt-Börnstein-derived structures (Springer,
   2013/14): do you regard that as a substantial part of the III/40H subvolumes under §87b?
   And has III/40H been substantially revised since 2014 in a way that would restart the
   term under §87a(1) sentence 2?
3. **Compilation copyright.** Independently of the database right — the Ilm-NMR authors
   re-drew all structures and built their own schema, taking values but not arrangement. Is
   that sufficient to avoid §4 UrhG *Sammelwerk* copyright, which would otherwise run for
   70 years p.m.a. and would outlast the database right by decades?
4. **Downstream position.** ChemMatData relies on a CC-BY-4.0 grant from a peer-reviewed
   publication. Given CC BY §1(7) ("rights ... that the Licensor has authority to license")
   and §5(a)'s disclaimer of title, what is KIT's exposure if the grant is partly invalid —
   and does the analysis change now that we have actual knowledge of the provenance?
5. **The linking question.** Does *Innoweb* (C‑202/12, esp. para 48) combined with
   *CV-Online* (C‑762/19, para 37) mean that shipping a client that fetches from Zenodo at
   load time is "re-utilisation" under §87b, notwithstanding that no data sits on our
   server? And does *CV-Online*'s harm-to-investment condition offer a real defence for a
   non-commercial academic tool where the upstream maker is two hops removed?
6. **Institutional posture.** Does KIT want a standing policy for this class of case, given
   it will recur across the collection? Specifically: is the BLOCK criterion in §5.2 above
   (commercial + within 15 years + EU publisher, no permission) one legal is comfortable
   endorsing as an automated gate?
7. **Practical.** Is there an institutional route for a permission request to Springer
   Nature for Landolt-Börnstein-derived values in a non-commercial ML dataset, and would
   legal prefer to send it?

---

## 7. What I could not establish

- **Whether any Council agreement under Art. 11(3) of Directive 96/9/EC extends the sui
  generis right to US-made databases.** Searching surfaced a reference to an Isle of Man
  arrangement which I did not retrieve. The Tebby conclusion does not depend on this — the
  term expiry under §137g(2) is independently decisive — but the Art. 11 argument is
  unconfirmed.
- **Whether Landolt-Börnstein III/40H has been substantially revised since 2014.** If it
  has, §87a(1) sentence 2 restarts a fresh 15-year term and pushes the block past 2029.
  Checkable from Springer's catalogue; I did not.
- **The actual licence terms under which the Ilm-NMR authors obtained the Landolt-Börnstein
  and CRC PDFs.** The paper says both "are available as PDF documents and could thus
  partially be read by machines," which implies institutional subscription access. Springer
  and CRC subscription agreements routinely contain TDM and redistribution clauses that
  would govern independently of copyright — a **contractual** restriction that no statutory
  analysis above touches. I could not see those agreements. **This may matter more than
  everything above**, because contract can bind where copyright does not.
- **Whether the Ilm-NMR authors in fact hold permission.** Absent from the paper, but
  absence of documentation is not documentation of absence. Worth one email.
- **German case law on printed reference works as `Datenbanken` under §87a.** I relied on
  Art. 1(2)'s "or other means" and the CJEU's broad definition in *OPAP*; I did not retrieve
  a German decision squarely on a printed scientific reference table.
- **Zenodo's current published rate limits.** The runtime dependency is certain; the
  thresholds are unquantified here.
- **No US case law is cited beyond *Feist*.** I did not attempt to find US authority on
  scientific-data compilations specifically, and the "no federal database right" statement
  rests on the *Feist* framework rather than on retrieved legislative history.

---

## Sources

All fetched and read in preparing this memo.

**Legislation**
- [Directive 96/9/EC on the legal protection of databases](https://eur-lex.europa.eu/legal-content/EN/TXT/HTML/?uri=CELEX:31996L0009) — Arts. 1, 3, 5–11, 13, 14, 16; Recitals 42, 45–47, 49, 55, 56
- [Directive (EU) 2019/790 (DSM)](https://eur-lex.europa.eu/legal-content/EN/TXT/HTML/?uri=CELEX:32019L0790) — Arts. 2, 3, 4, 7; Recital 18
- UrhG: [§4](https://www.gesetze-im-internet.de/urhg/__4.html) · [§44b](https://www.gesetze-im-internet.de/urhg/__44b.html) · [§60d](https://www.gesetze-im-internet.de/urhg/__60d.html) · [§87a](https://www.gesetze-im-internet.de/urhg/__87a.html) · [§87b](https://www.gesetze-im-internet.de/urhg/__87b.html) · [§87c](https://www.gesetze-im-internet.de/urhg/__87c.html) · [§87d](https://www.gesetze-im-internet.de/urhg/__87d.html) · [§87e](https://www.gesetze-im-internet.de/urhg/__87e.html) · [§97](https://www.gesetze-im-internet.de/urhg/__97.html) · [§137g](https://www.gesetze-im-internet.de/urhg/__137g.html)

**A note on paragraph citations.** CJEU paragraph numbers below were read off the numbered
judgment text in the EUR-Lex HTML, not inferred. The *Innoweb* "parasitical competing
product" passage being para 48 is independently corroborated by the Court's own
cross-reference to it in *CV-Online* para 40 ("*see, to that effect, judgment of 19
December 2013, Innoweb, C‑202/12, EU:C:2013:850, paragraph 48*").

**Case law**
- [C‑203/02 *British Horseracing Board v William Hill*](https://eur-lex.europa.eu/legal-content/EN/TXT/HTML/?uri=CELEX:62002CJ0203) — obtaining vs creating; substantial part
- [C‑444/02 *Fixtures Marketing v OPAP*](https://eur-lex.europa.eu/legal-content/EN/TXT/HTML/?uri=CELEX:62002CJ0444) — definition of "database"
- [C‑304/07 *Directmedia Publishing*](https://eur-lex.europa.eu/legal-content/EN/TXT/HTML/?uri=CELEX:62007CJ0304) — extraction by consultation and re-keying
- [C‑202/12 *Innoweb v Wegener*](https://eur-lex.europa.eu/legal-content/EN/TXT/HTML/?uri=CELEX:62012CJ0202) — re-utilisation without copying
- [C‑762/19 *CV-Online Latvia v Melons*](https://eur-lex.europa.eu/legal-content/EN/TXT/HTML/?uri=CELEX:62019CJ0762) — hyperlinks as secondary manifestation; harm condition
- [C‑466/12 *Svensson*](https://eur-lex.europa.eu/legal-content/EN/TXT/HTML/?uri=CELEX:62012CJ0466) — linking under copyright
- [C‑160/15 *GS Media*](https://eur-lex.europa.eu/legal-content/EN/TXT/HTML/?uri=CELEX:62015CJ0160) — knowledge and financial gain
- [C‑161/17 *Renckhoff*](https://eur-lex.europa.eu/legal-content/EN/TXT/HTML/?uri=CELEX:62017CJ0161) — re-posting vs linking
- [*Feist Publications v Rural Telephone*, 499 U.S. 340 (1991)](https://www.law.cornell.edu/supremecourt/text/499/340) — rejection of sweat of the brow

**Licence and dataset**
- [CC BY 4.0 legal code](https://creativecommons.org/licenses/by/4.0/legalcode.en) — §§1(7), 1(10), 2(a), 3(a), 4, 5
- [Ilm-NMR-P31, *J. Cheminform.* 15, 122 (2023)](https://jcheminf.biomedcentral.com/articles/10.1186/s13321-023-00792-y) — `10.1186/s13321-023-00792-y`
- Zenodo API, concept record `8260782` and versions `8260783`, `15658468`, `17649620`, `17952686`, [`20762603`](https://zenodo.org/records/20762603)
