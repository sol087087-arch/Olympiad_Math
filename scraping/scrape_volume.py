#!/usr/bin/env python3
"""
High-volume olympiad scraper.

Sources:
  1. AoPS Wiki  — problem + solution pages, LaTeX in page source (no JS needed)
  2. Putnam archive (kskedlaya.org) — clean TeX files
  3. GitHub repos with olympiad collections

Usage:
    pip install requests beautifulsoup4 tqdm

    # AoPS wiki — best quality, ~5000+ problems available
    python scrape_volume.py aops --out ./raw

    # Putnam (Kedlaya archive)
    python scrape_volume.py putnam --out ./raw

    # GitHub repos
    python scrape_volume.py github --out ./raw

    # Everything
    python scrape_volume.py all --out ./raw

After scraping, convert to training pairs:
    python scrape_volume.py convert --input ./raw --output big_dataset.jsonl
"""

import os, re, json, time, argparse, sys
from pathlib import Path
from urllib.parse import urljoin, quote

try:
    import requests
    from bs4 import BeautifulSoup
except ImportError:
    print("pip install requests beautifulsoup4")
    sys.exit(1)

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False
    def tqdm(x, **kw): return x


# ─────────────────────────────────────────────────────────────
# HTTP
# ─────────────────────────────────────────────────────────────

def make_session():
    s = requests.Session()
    s.headers.update({
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                      "AppleWebKit/537.36 (KHTML, like Gecko) "
                      "Chrome/122.0 Safari/537.36",
        "Accept-Language": "en-US,en;q=0.9",
    })
    return s

def get(session, url, retries=3, delay=1.0):
    for attempt in range(retries):
        try:
            r = session.get(url, timeout=20)
            if r.status_code == 200:
                return r
            if r.status_code == 429:
                wait = 30 * (attempt + 1)
                print(f"  [rate limit] waiting {wait}s...")
                time.sleep(wait)
        except Exception as e:
            if attempt == retries - 1:
                print(f"  [!] {url}: {e}")
        time.sleep(delay)
    return None


# ─────────────────────────────────────────────────────────────
# AoPS WIKI SCRAPER
# Key insight: AoPS wiki pages contain raw LaTeX in the HTML
# inside <script type="math/tex"> tags OR as plain text in
# the wikitext before MediaWiki processes it.
# The math is stored as: <annotation encoding="application/x-tex">RAW LATEX</annotation>
# ─────────────────────────────────────────────────────────────

# Contests and their AoPS wiki URL patterns
AOPS_CONTESTS = {
    # format: (url_pattern, years, problems_per_year)
    "IMO":          ("https://artofproblemsolving.com/wiki/index.php/{year}_IMO_Problems/Problem_{n}",
                     range(1959, 2025), 6),
    "USAMO":        ("https://artofproblemsolving.com/wiki/index.php/{year}_USAMO_Problems/Problem_{n}",
                     range(1972, 2025), 6),
    "USAJMO":       ("https://artofproblemsolving.com/wiki/index.php/{year}_USAJMO_Problems/Problem_{n}",
                     range(2010, 2025), 6),
    "AIME_I":       ("https://artofproblemsolving.com/wiki/index.php/{year}_AIME_I_Problems/Problem_{n}",
                     range(2000, 2025), 15),
    "AIME_II":      ("https://artofproblemsolving.com/wiki/index.php/{year}_AIME_II_Problems/Problem_{n}",
                     range(2000, 2025), 15),
    "AMC_10A":      ("https://artofproblemsolving.com/wiki/index.php/{year}_AMC_10A_Problems/Problem_{n}",
                     range(2002, 2025), 25),
    "AMC_12A":      ("https://artofproblemsolving.com/wiki/index.php/{year}_AMC_12A_Problems/Problem_{n}",
                     range(2002, 2025), 25),
    "Putnam":       ("https://artofproblemsolving.com/wiki/index.php/{year}_Putnam_Problems/Problem_{n}",
                     range(1985, 2024), 12),  # A1-A6, B1-B6
    "EGMO":         ("https://artofproblemsolving.com/wiki/index.php/{year}_EGMO_Problems/Problem_{n}",
                     range(2012, 2025), 6),
    "APMO":         ("https://artofproblemsolving.com/wiki/index.php/{year}_APMO_Problems/Problem_{n}",
                     range(1989, 2025), 5),
    "Canada_MO":    ("https://artofproblemsolving.com/wiki/index.php/{year}_Canadian_MO_Problems/Problem_{n}",
                     range(1969, 2025), 5),
    "Brazil_MO":    ("https://artofproblemsolving.com/wiki/index.php/{year}_Brazil_MO_Problems/Problem_{n}",
                     range(1979, 2024), 6),
    "Balkan_MO":    ("https://artofproblemsolving.com/wiki/index.php/{year}_Balkan_MO_Problems/Problem_{n}",
                     range(1984, 2025), 4),
    "Iran_MO":      ("https://artofproblemsolving.com/wiki/index.php/{year}_Iran_MO_3rd_Round_Problems/Problem_{n}",
                     range(2005, 2024), 6),
    "Turkey_MO":    ("https://artofproblemsolving.com/wiki/index.php/{year}_Turkey_MO_(2nd_Round)_Problems/Problem_{n}",
                     range(2005, 2024), 6),
}

# Olympiad-level only (skip AMC/AIME for quality dataset)
OLYMPIAD_ONLY = {"IMO", "USAMO", "USAJMO", "EGMO", "APMO", "Canada_MO",
                 "Brazil_MO", "Balkan_MO", "Iran_MO", "Turkey_MO", "Putnam"}


def extract_math_from_aops_page(html):
    """
    Extract LaTeX from AoPS wiki page HTML.
    AoPS stores math in <annotation encoding="application/x-tex"> tags.
    Plain text is in standard paragraph elements.
    """
    soup = BeautifulSoup(html, "html.parser")

    # Get the main content div
    content = soup.find("div", {"id": "mw-content-text"})
    if not content:
        content = soup.find("div", class_="mw-parser-output")
    if not content:
        return None

    # Replace math annotations with LaTeX
    # AoPS uses: <math alttext="..."><semantics><annotation encoding="application/x-tex">LATEX</annotation>
    for math_elem in content.find_all("math"):
        annotation = math_elem.find("annotation", {"encoding": "application/x-tex"})
        if annotation:
            latex = annotation.get_text()
            # Determine if inline or display
            if math_elem.get("display") == "block":
                math_elem.replace_with(f"$${latex}$$")
            else:
                math_elem.replace_with(f"${latex}$")
        else:
            # Fallback: use alttext
            alttext = math_elem.get("alttext", "")
            if alttext:
                math_elem.replace_with(f"${alttext}$")

    # Also handle <span class="mwe-math-element"> which wraps the math
    # (already handled above via the <math> tag inside)

    return content


def parse_aops_problem_page(html, url=""):
    """
    Parse an AoPS wiki problem page.
    Returns {"problem": str, "solutions": [str]} or None.
    """
    content = extract_math_from_aops_page(html)
    if not content:
        return None

    full_text = content.get_text(separator="\n")
    full_text = re.sub(r'\n{3,}', '\n\n', full_text).strip()

    # Split into problem + solutions sections
    # AoPS wiki structure: Problem\n...\nSolution\n...\nSolution 2\n...
    sections = re.split(
        r'\n(?=(?:Problem|Solution\s*\d*|Alternate [Ss]olution|[Ss]olution \d+)\s*\n)',
        full_text
    )

    problem_text = ""
    solution_texts = []

    for section in sections:
        section = section.strip()
        if not section:
            continue

        if re.match(r'^Problem\s*$', section.split('\n')[0], re.IGNORECASE):
            # Everything after "Problem" header
            lines = section.split('\n')
            problem_text = '\n'.join(lines[1:]).strip()

        elif re.match(r'^(?:Solution|Alternate [Ss]olution|[Ss]olution \d+)', 
                      section.split('\n')[0], re.IGNORECASE):
            lines = section.split('\n')
            sol = '\n'.join(lines[1:]).strip()
            if len(sol) > 100:
                solution_texts.append(sol)

    # Fallback: if no clear sections, try to split at "Solution"
    if not problem_text:
        split = re.split(r'\n(?=Solution)', full_text, maxsplit=1)
        if len(split) == 2:
            problem_text = split[0].strip()
            solution_texts = [split[1].strip()]

    if not problem_text or not solution_texts:
        return None

    # Clean up "See Also" cruft from solutions
    solution_texts = [
        re.split(r'\nSee [Aa]lso\b|\n==', s)[0].strip()
        for s in solution_texts
    ]
    solution_texts = [s for s in solution_texts if len(s) > 100]

    if not solution_texts:
        return None

    return {
        "url": url,
        "problem": problem_text,
        "solutions": solution_texts,
    }


def scrape_aops(out_dir, contests=None, olympiad_only=True, delay=0.5):
    """Scrape AoPS wiki problem pages."""
    os.makedirs(out_dir, exist_ok=True)
    session = make_session()

    if contests is None:
        contests = OLYMPIAD_ONLY if olympiad_only else list(AOPS_CONTESTS.keys())

    total = 0
    for contest_name in contests:
        if contest_name not in AOPS_CONTESTS:
            print(f"Unknown contest: {contest_name}")
            continue

        url_pattern, years, n_problems = AOPS_CONTESTS[contest_name]
        out_file = os.path.join(out_dir, f"aops_{contest_name.lower()}.jsonl")

        print(f"\n── {contest_name} ({len(list(years))} years × {n_problems} problems)")
        contest_total = 0

        # Build URL list
        urls = []
        for year in years:
            for n in range(1, n_problems + 1):
                urls.append(url_pattern.format(year=year, n=n))

        # Check cache
        existing_urls = set()
        if os.path.exists(out_file):
            with open(out_file, encoding="utf-8") as f:
                for line in f:
                    try:
                        d = json.loads(line)
                        existing_urls.add(d.get("url", ""))
                    except:
                        pass

        urls_to_fetch = [u for u in urls if u not in existing_urls]
        print(f"   {len(existing_urls)} cached, {len(urls_to_fetch)} to fetch")

        with open(out_file, "a", encoding="utf-8") as f_out:
            for url in tqdm(urls_to_fetch, desc=contest_name):
                r = get(session, url, delay=delay)
                if not r:
                    continue

                # Skip redirect pages (problem doesn't exist)
                if "There is currently no text in this page" in r.text:
                    continue

                data = parse_aops_problem_page(r.text, url=url)
                if data:
                    f_out.write(json.dumps(data, ensure_ascii=False) + "\n")
                    contest_total += 1

                time.sleep(delay)

        print(f"   → {contest_total} new pairs saved")
        total += contest_total

    print(f"\nAoPS total: {total} new pairs")


# ─────────────────────────────────────────────────────────────
# PUTNAM ARCHIVE (kskedlaya.org)
# Clean PDF/TeX, structured, 1938-present
# ─────────────────────────────────────────────────────────────

def scrape_putnam(out_dir, delay=1.0):
    """
    Scrape Putnam problems from kskedlaya.org.
    The archive has LaTeX source available for many years.
    """
    os.makedirs(out_dir, exist_ok=True)
    session = make_session()
    out_file = os.path.join(out_dir, "putnam_raw.jsonl")

    base = "https://kskedlaya.org/putnam-archive/"
    print(f"\n── Putnam archive ({base})")

    r = get(session, base)
    if not r:
        print("  [!] Could not reach kskedlaya.org")
        return

    soup = BeautifulSoup(r.text, "html.parser")
    # Find links to individual year pages or .tex files
    tex_links = []
    for a in soup.find_all("a", href=True):
        href = a["href"]
        if href.endswith(".tex") or href.endswith(".pdf"):
            tex_links.append(urljoin(base, href))

    print(f"  Found {len(tex_links)} files")

    downloaded = 0
    tex_dir = os.path.join(out_dir, "putnam_tex")
    os.makedirs(tex_dir, exist_ok=True)

    for url in tqdm(tex_links, desc="Putnam"):
        fname = os.path.basename(url)
        dest = os.path.join(tex_dir, fname)
        if os.path.exists(dest):
            continue
        r = get(session, url)
        if r:
            with open(dest, "wb") as f:
                f.write(r.content)
            downloaded += 1
        time.sleep(delay)

    print(f"  Downloaded {downloaded} files → {tex_dir}")
    print(f"  Run parse_tex_to_dataset.py --input {tex_dir} to convert")


# ─────────────────────────────────────────────────────────────
# GITHUB REPOS
# Many people upload olympiad solution collections as .tex
# ─────────────────────────────────────────────────────────────

GITHUB_REPOS = [
    # Format: (owner, repo, subpath_with_tex_files)
    ("vEnhance", "von",             ""),          # Evan Chen's problem manager
    ("vEnhance", "olympiad",        ""),          # his olympiad collection
    ("bamos", "olympiad-problems",  ""),
    ("mathzeta2", "olympiad",       ""),
    ("gilcu3", "olympiad-notes",    ""),
    # Add more as you find them
]

GITHUB_API = "https://api.github.com/repos/{owner}/{repo}/git/trees/HEAD?recursive=1"
GITHUB_RAW = "https://raw.githubusercontent.com/{owner}/{repo}/HEAD/{path}"


def scrape_github(out_dir, delay=0.5):
    """Download .tex files from GitHub olympiad repos."""
    os.makedirs(out_dir, exist_ok=True)
    session = make_session()
    # Use token if available
    token = os.environ.get("GITHUB_TOKEN")
    if token:
        session.headers["Authorization"] = f"token {token}"
        print("  Using GITHUB_TOKEN")
    else:
        print("  No GITHUB_TOKEN — rate limited to 60 req/hr. Set env var to increase.")

    total = 0
    for owner, repo, subpath in GITHUB_REPOS:
        print(f"\n── github.com/{owner}/{repo}")
        url = GITHUB_API.format(owner=owner, repo=repo)
        r = get(session, url)
        if not r:
            continue

        try:
            tree = r.json().get("tree", [])
        except:
            continue

        tex_files = [
            item["path"] for item in tree
            if item["path"].endswith(".tex")
            and (not subpath or item["path"].startswith(subpath))
        ]
        print(f"  {len(tex_files)} .tex files found")

        repo_dir = os.path.join(out_dir, f"github_{owner}_{repo}")
        os.makedirs(repo_dir, exist_ok=True)

        for path in tqdm(tex_files, desc=f"{owner}/{repo}"):
            fname = path.replace("/", "_")
            dest = os.path.join(repo_dir, fname)
            if os.path.exists(dest):
                continue
            raw_url = GITHUB_RAW.format(owner=owner, repo=repo, path=path)
            r = get(session, raw_url)
            if r and len(r.content) > 200:
                with open(dest, "wb") as f:
                    f.write(r.content)
                total += 1
            time.sleep(delay)

    print(f"\nGitHub total: {total} files downloaded")
    print(f"Run parse_tex_to_dataset.py on each subdirectory to convert")


# ─────────────────────────────────────────────────────────────
# CONVERT: raw AoPS JSON → training JSONL
# ─────────────────────────────────────────────────────────────

def convert_aops_raw(raw_dir, output_file, min_sol_len=150):
    """Convert raw AoPS scraped JSONL files to training format."""
    all_pairs = []
    seen = set()

    raw_files = list(Path(raw_dir).glob("aops_*.jsonl"))
    print(f"Converting {len(raw_files)} raw files...")

    for fpath in raw_files:
        contest = fpath.stem.replace("aops_", "").upper()
        count = 0
        with open(fpath, encoding="utf-8") as f:
            for line in f:
                try:
                    d = json.loads(line)
                except:
                    continue

                problem = d.get("problem", "").strip()
                solutions = d.get("solutions", [])

                if not problem or not solutions:
                    continue

                # Pick best solution (longest that's not too long = copy-paste)
                solutions = [s for s in solutions if min_sol_len < len(s) < 15000]
                if not solutions:
                    continue
                best_sol = max(solutions, key=len)

                # Dedup
                key = re.sub(r'\s+', ' ', problem[:120].lower())
                if key in seen:
                    continue
                seen.add(key)

                all_pairs.append({
                    "messages": [
                        {"role": "user",
                         "content": f"Solve the following olympiad problem:\n\n{problem}"},
                        {"role": "assistant",
                         "content": best_sol},
                    ]
                })
                count += 1

        print(f"  {fpath.name}: {count} pairs")

    print(f"\nTotal: {len(all_pairs)} pairs")
    with open(output_file, "w", encoding="utf-8") as f:
        for p in all_pairs:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    print(f"Saved → {output_file}")


# ─────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("source", choices=["aops", "putnam", "github", "all", "convert"])
    parser.add_argument("--out", default="./olympiad_raw")
    parser.add_argument("--output", default="./aops_training.jsonl",
                        help="Output file for convert command")
    parser.add_argument("--contests", nargs="+",
                        help="AoPS: specific contests e.g. IMO USAMO EGMO")
    parser.add_argument("--all-contests", action="store_true",
                        help="AoPS: include AMC/AIME, not just olympiad")
    parser.add_argument("--delay", type=float, default=0.5,
                        help="Delay between requests (seconds)")
    args = parser.parse_args()

    if args.source == "aops" or args.source == "all":
        contests = args.contests
        scrape_aops(
            out_dir=os.path.join(args.out, "aops"),
            contests=contests,
            olympiad_only=not args.all_contests,
            delay=args.delay,
        )

    if args.source == "putnam" or args.source == "all":
        scrape_putnam(
            out_dir=os.path.join(args.out, "putnam"),
            delay=args.delay,
        )

    if args.source == "github" or args.source == "all":
        scrape_github(
            out_dir=os.path.join(args.out, "github"),
            delay=args.delay,
        )

    if args.source == "convert":
        convert_aops_raw(
            raw_dir=os.path.join(args.out, "aops"),
            output_file=args.output,
        )


if __name__ == "__main__":
    main()