> Replace the lambeq sentence that lives on discopy.org with one that points at discopy.lambek. Do the change in this repo only (rel-int/discopy-next). It will be merged upstream later; do not touch discopy/discopy or discopy/discopy.github.io.
>
> What is true, measured:
>
> - https://discopy.org is built from discopy/discopy.github.io, branch main, file index.md. The only lambeq link on the live page is this sentence in the How section:
>
>   This application has now been packaged into its own library, [lambeq](https://github.com/Quantinuum/lambeq).
>
> - That sentence is not in this repo. docs.discopy.org is built from this repo's docs/, which include ../README.md from line 9 and docs/extra/blogs.md. README.md's Features section already names no lambeq; it names the grammar modules and depccg, spaCy, NLTK. docs/extra/blogs.md still has two historical lambeq mentions (a github.com/CQCL/lambeq link and "try it yourself with lambeq"). Leave those: they name the library the blog posts were written about.
>
> What to do:
>
> - Add the How sentence to this repo so the docs site carries it, pointed at discopy.lambek rather than lambeq. The natural place is README.md, since docs/index.rst includes it: a short How section, after Features and before Quickstart, with the sentence rewritten so the link is the docs page for discopy.lambek (https://docs.discopy.org/en/main/api/lambek.html is the page this repo's docs/api/lambek.rst generates; confirm the path against docs/api/lambek.rst and how the other modules are linked in README.md, and use the same style). Keep the rest of the sentence's meaning: DisCoPy began as an implementation of DisCoCat and QNLP, and that application now lives in discopy.lambek. Do not claim it has been packaged into a separate library.
> - A changelog line under [Unreleased], in the file's existing style, naming this pull request once it exists.
> - Open a pull request. Follow this repo's RULES.md (fresh TODO.md with the prompt verbatim, deleted when done) and its pull request template. The Why is this sentence, quoted, and that the change stays in next until it goes upstream.
>
> When the pull request is open, reply to the thread that spawned you with its URL, the sentence as it now reads, and where it landed.

- [x] README.md: a How section after Features, the discopy.org sentence pointed at discopy.lambek
- [x] CHANGELOG.md: a bullet under [Unreleased] naming this pull request
- [x] Open the pull request with the template's Why, What and How
