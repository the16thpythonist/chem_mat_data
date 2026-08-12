# Data sources

Where dataset files actually live, and how to get a *direct download URL* rather than
a landing page. This distinction is the single most common stage-1 failure: saving a
landing page as `data.csv` produces an HTML file that stage 2 then fails to parse.

Sanity check every download:

```bash
file .dataset_work/downloads/data.csv    # should not say "HTML document"
head -c 300 .dataset_work/downloads/data.csv
```

## Zenodo

Landing page: `https://zenodo.org/records/<id>`

Direct download, one per file in the "Files" section:

```
https://zenodo.org/records/<id>/files/<filename>?download=1
```

The API lists files without scraping — useful when the page is JS-heavy:

```bash
curl -s "https://zenodo.org/api/records/<id>" | python -m json.tool | grep -A3 '"key"'
```

Zenodo DOIs (`10.5281/zenodo.<id>`) resolve to the landing page. A "concept DOI"
points at the latest version; a version DOI pins one. Prefer the version DOI in
`sources` so the record stays reproducible.

## Figshare

Landing page: `https://figshare.com/articles/dataset/<slug>/<article_id>`

Direct download:

```
https://ndownloader.figshare.com/files/<file_id>
https://figshare.com/ndownloader/files/<file_id>     # equivalent
```

The `<file_id>` is per-file and is *not* the article id — find it in the download
button's href, or via the API:

```bash
curl -s "https://api.figshare.com/v2/articles/<article_id>/files"
```

**Figshare frequently has certificate-chain problems.** This is what
`FileDownloadSource(..., ssl_verify=False)` exists for, and why several existing
scripts use it. Reach for it as soon as you see an SSL error rather than fighting it.

ACS-hosted Figshare collections (`acs.figshare.com`) behave the same way.

## GitHub / GitLab

Never link the HTML blob viewer — it returns a web page, not the file.

```
# raw file
https://raw.githubusercontent.com/<owner>/<repo>/<branch>/<path>

# release asset
https://github.com/<owner>/<repo>/releases/download/<tag>/<filename>

# GitLab raw
https://gitlab.com/<group>/<project>/-/raw/<branch>/<path>?inline=false
```

Repos often hold the data under `data/`, `datasets/`, or in a release rather than the
tree. Check the releases page when the tree looks like code only. Pin a tag or commit
SHA instead of `main`/`master` when one exists — `main` moves.

## OSF

```
https://osf.io/<node>/download           # whole component
https://osf.io/download/<file_guid>/     # single file
```

## Dryad

```
https://datadryad.org/api/v2/datasets/doi%3A10.5061%2Fdryad.<id>/download
```

The DOI must be percent-encoded. Dryad downloads are zipped by default.

## Materials Cloud / NOMAD

Materials Cloud archive entries expose a direct file URL on the record page. NOMAD
generally requires its API and returns archives of calculation outputs — usually the
wrong shape for this pipeline. If a NOMAD entry is the only source, say so and expect
significant parsing work rather than a tabular file.

## Supplementary information from publishers

When the data only exists as journal SI, the file is typically at a publisher-specific
URL reachable from the article page:

- ACS: `https://pubs.acs.org/doi/suppl/<doi>/suppl_file/<filename>`
- Nature: linked from the "Supplementary information" section
- RSC / Wiley / Elsevier: linked from the article page

**These are frequently bot-blocked even when the article is fully open access.** ACS in
particular returns 403 to `curl` and to every scripted client regardless of headers. Do
not spend attempts fighting it — go to Europe PMC instead.

The contrast, measured on the same paper (`10.1021/acs.estlett.3c00526`, open access):

| URL | Result |
|---|---|
| `pubs.acs.org/doi/suppl/.../ez3c00526_si_001.xlsx` | **403**, 5.6 kB of HTML |
| `ebi.ac.uk/.../PMC10569042/supplementaryFiles` | **200**, 2.1 MB `application/zip` |

### Europe PMC — the way around a bot-blocked SI

For anything with a PMC ID, Europe PMC serves the publisher-deposited supplementary
files as a plain ZIP, with no bot check and no authentication:

```
https://www.ebi.ac.uk/europepmc/webservices/rest/PMC<id>/supplementaryFiles
```

The ZIP contains every deposited file, typically the SI workbook plus article figures,
under the publisher's own filenames (`ez3c00526_si_001.xlsx`). Extract the member you
need and ignore the images.

Find the PMC ID from the DOI:

```bash
curl -s "https://www.ebi.ac.uk/europepmc/webservices/rest/search?query=DOI:<doi>&format=json" \
  | python -c "import json,sys; print([r.get('pmcid') for r in json.load(sys.stdin)['resultList']['result']])"
```

This route is stable enough to put directly in a generated script's `load_dataset`,
which a Playwright click-through or a browser-mode download never is.

### ChemRxiv

The article id in a ChemRxiv URL (`/engage/chemrxiv/article-details/<id>`) is opaque and
is **not** the DOI. The old `public-api/v1/items/<id>` endpoint is dead since the
migration to Atypon and now 404s.

Live patterns, both of which need the DOI rather than the article id:

```
https://chemrxiv.org/engage/chemrxiv/doi/full/<doi>
https://chemrxiv.org/engage/chemrxiv/doi/suppl/<doi>/suppl_file/<filename>
```

Get the DOI by rendering the article page with `crwl`, the default reader. Do not fall
back to `defuddle` or `WebFetch` here — ChemRxiv 403s every non-browser client, so this
is the host that the crwl-first default exists for. Crossref, DataCite and Europe PMC
search all fail to resolve the opaque article id; an **OpenAlex** search filtered to the
ChemRxiv source does work — the source id is `S4393918830`:

```bash
curl -s "https://api.openalex.org/works?filter=primary_location.source.id:S4393918830&search=<title>"
```

**Check for a published version before using the preprint.** A preprint and its journal
article routinely carry *different* licenses — CC-BY-NC-ND on ChemRxiv against CC-BY on
the publisher's site is a common and consequential pairing, since NoDerivatives forbids
redistributing a processed dataset at all. Crossref will show the published version:

```bash
curl -s "https://api.crossref.org/works/<preprint-doi>" | python -m json.tool | grep -i relation -A5
```

Prefer the published version when one exists, and license against the version you
actually downloaded. See `license.md`.

Excel SI files often carry headers, footnotes, and merged cells above the real table.
Inspect before parsing, and use `skiprows=` / `sheet_name=` rather than assuming row 0.

## Preferring an open-access copy

For paywalled articles you only need the Data Availability section, so a preprint is
usually enough: arXiv, ChemRxiv, bioRxiv, PubMed Central, or the author's own page.
Search `"<paper title>" arxiv OR chemrxiv` and read that instead.
