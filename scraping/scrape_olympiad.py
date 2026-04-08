#!/usr/bin/env python3
"""
Olympiad content scraper.
Sources:
  1. Evan Chen's site (web.evanchen.cc) — .tex files, best quality
  2. AoPS (artofproblemsolving.com) — problem + solution threads
  3. IMO official (imo-official.org) — problem PDFs

Usage:
    pip install requests beautifulsoup4
    
    python scrape_olympiad.py evanchen           # Evan Chen .tex files only
    python scrape_olympiad.py aops --contest IMO # AoPS threads for IMO
    python scrape_olympiad.py all                # everything
    
    python scrape_olympiad.py evanchen --dry-run # just list what would be downloaded
    python scrape_olympiad.py aops --contest USAMO --years 2010-2023
"""

import os, re, time, json, argparse, sys
from urllib.parse import urljoin, urlparse

try:
    import requests
    from bs4 import BeautifulSoup
except ImportError:
    print("pip install requests beautifulsoup4")
    sys.exit(1)


# ─────────────────────────────────────────────
# SHARED
# ─────────────────────────────────────────────

def make_session():
    s = requests.Session()
    s.headers["User-Agent"] = (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0 Safari/537.36"
    )
    return s


def get(session, url, **kw):
    try:
        r = session.get(url, timeout=15, **kw)
        if r.status_code == 200:
            return r
    except Exception as e:
        print(f"  [!] {url} → {e}")
    return None


# ─────────────────────────────────────────────
# SOURCE 1: EVAN CHEN
# ─────────────────────────────────────────────

EVAN_BASES = [
    "https://web.evanchen.cc/upload/",
    "https://web.evanchen.cc/exams/",
    "https://web.evanchen.cc/",
]

EVAN_PATTERNS = []
for y in range(1994, 2026):
    EVAN_PATTERNS += [
        f"IMO-{y}-notes.tex",
        f"USAMO-{y}-notes.tex",
        f"EGMO-{y}-notes.tex",
        f"JMO-{y}-notes.tex",
        f"APMO-{y}-notes.tex",
        f"sols-TST-IMO-{y}.tex",
        f"sols-TSTST-{y}.tex",
        f"sols-ELMO-{y}.tex",
        f"ELMO-{y}-notes.tex",
        f"RMM-{y}-notes.tex",
        f"IMO-{y}-SL-notes.tex",
        f"sols-USAMO-{y}.tex",
    ]

EVAN_SCRAPE_PAGES = [
    "https://web.evanchen.cc/olympiad.html",
    "https://web.evanchen.cc/problems.html",
]


def scrape_evanchen(out_dir, dry_run=False):
    os.makedirs(out_dir, exist_ok=True)
    session = make_session()
    to_try = set()

    # 1a. Scrape index pages for links
    print("── Evan Chen: scraping index pages")
    for page in EVAN_SCRAPE_PAGES:
        r = get(session, page)
        if not r:
            continue
        soup = BeautifulSoup(r.text, "html.parser")
        for a in soup.find_all("a", href=True):
            if a["href"].endswith(".tex"):
                to_try.add(urljoin(page, a["href"]))
        time.sleep(0.4)

    # 1b. Try known patterns
    for fname in EVAN_PATTERNS:
        for base in EVAN_BASES:
            to_try.add(base + fname)

    print(f"   {len(to_try)} URLs to probe")

    ok, skip = 0, 0
    for url in sorted(to_try):
        fname = os.path.basename(urlparse(url).path)
        dest = os.path.join(out_dir, fname)
        if os.path.exists(dest):
            skip += 1
            continue
        if dry_run:
            print(f"  [?] {url}")
            continue
        r = get(session, url)
        if r and len(r.content) > 300:
            preview = r.content[:200].decode("utf-8", errors="ignore")
            if any(kw in preview for kw in ["\\documentclass", "\\begin", "Evan Chen", "%"]):
                with open(dest, "wb") as f:
                    f.write(r.content)
                print(f"  [+] {fname}  ({len(r.content)//1024}KB)")
                ok += 1
        time.sleep(0.25)

    print(f"\n  Downloaded {ok} new files, skipped {skip} existing.\n")


# ─────────────────────────────────────────────
# SOURCE 2: AOPS
# ─────────────────────────────────────────────

# AoPS forum IDs for major contests
AOPS_FORUMS = {
    "IMO":       6,
    "USAMO":     26,
    "USAJMO":    227,
    "AIME":      20,
    "AMC":       22,
    "EGMO":      389,
    "APMO":      56,
    "Putnam":    33,
    "Shortlist": 7,
}

AOPS_SEARCH_URL = "https://artofproblemsolving.com/community/c{forum_id}"
AOPS_TOPIC_URL  = "https://artofproblemsolving.com/community/c{forum_id}h{topic_id}"


def parse_aops_post(post_div):
    """Extract text content from an AoPS post div."""
    # AoPS uses MathJax — we want the raw LaTeX from data-src or just text
    # Try to get raw post content
    content = post_div.get_text(separator="\n", strip=True)
    # Clean up some AoPS artifacts
    content = re.sub(r'\n{3,}', '\n\n', content)
    return content.strip()


def scrape_aops_thread(session, url):
    """
    Scrape a single AoPS thread.
    Returns {"problem": str, "solutions": [str]} or None.
    """
    r = get(session, url)
    if not r:
        return None
    soup = BeautifulSoup(r.text, "html.parser")

    posts = soup.select("div.cmty-post-body, div.post-body, .post_text")
    if not posts:
        return None

    result = {"url": url, "problem": "", "solutions": []}
    for i, post in enumerate(posts):
        text = parse_aops_post(post)
        if not text:
            continue
        if i == 0:
            result["problem"] = text
        else:
            # Only keep posts that look like solutions (have math or reasoning)
            if len(text) > 100 and ("$" in text or "proof" in text.lower() or "solution" in text.lower()):
                result["solutions"].append(text)
    
    if result["problem"] and result["solutions"]:
        return result
    return None


def scrape_aops_forum(out_dir, forum_name, year_start=2000, year_end=2025, max_threads=200):
    """
    Scrape AoPS forum for a given contest.
    Searches for threads matching the contest + year.
    """
    forum_id = AOPS_FORUMS.get(forum_name)
    if not forum_id:
        print(f"  Unknown forum: {forum_name}. Options: {list(AOPS_FORUMS)}")
        return

    os.makedirs(out_dir, exist_ok=True)
    session = make_session()
    out_file = os.path.join(out_dir, f"aops_{forum_name.lower()}.jsonl")

    # AoPS search API
    search_url = "https://artofproblemsolving.com/community/search"
    collected = 0

    for year in range(year_start, year_end + 1):
        query = f"{forum_name} {year}"
        print(f"  Searching: {query}")

        params = {
            "q": query,
            "fo": forum_id,
            "search_type": "1",
        }
        r = get(session, search_url, params=params)
        if not r:
            continue

        soup = BeautifulSoup(r.text, "html.parser")
        thread_links = []
        for a in soup.select("a.cmty-topic-title, a.topic_link"):
            href = a.get("href", "")
            if "/community/c" in href:
                thread_links.append(urljoin("https://artofproblemsolving.com", href))

        for url in thread_links[:10]:  # max 10 threads per year
            time.sleep(0.5)
            data = scrape_aops_thread(session, url)
            if data:
                with open(out_file, "a", encoding="utf-8") as f:
                    f.write(json.dumps(data, ensure_ascii=False) + "\n")
                collected += 1
                print(f"    [+] {url}")
                if collected >= max_threads:
                    print(f"  Reached max {max_threads} threads.")
                    return

        time.sleep(1)

    print(f"  Collected {collected} threads → {out_file}")


# ─────────────────────────────────────────────
# POST-PROCESS: AoPS JSONL → TRAINING JSONL
# ─────────────────────────────────────────────

def aops_to_training(in_file, out_file):
    """
    Convert raw AoPS scrape to training pairs.
    Takes the first solution that looks complete.
    """
    written = 0
    with open(in_file) as f_in, open(out_file, "w") as f_out:
        for line in f_in:
            d = json.loads(line)
            problem = d.get("problem", "").strip()
            solutions = d.get("solutions", [])
            if not problem or not solutions:
                continue
            # Pick longest solution as the "best"
            best = max(solutions, key=len)
            if len(best) < 200:
                continue
            pair = {
                "messages": [
                    {"role": "user", "content": f"Solve the following olympiad problem:\n\n{problem}"},
                    {"role": "assistant", "content": best},
                ]
            }
            f_out.write(json.dumps(pair, ensure_ascii=False) + "\n")
            written += 1
    print(f"  Wrote {written} training pairs → {out_file}")


# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Olympiad content scraper")
    parser.add_argument("source", choices=["evanchen", "aops", "all"],
                        help="Which source to scrape")
    parser.add_argument("--out", default="./olympiad_raw",
                        help="Output directory (default: ./olympiad_raw)")
    parser.add_argument("--dry-run", action="store_true",
                        help="List URLs without downloading")
    parser.add_argument("--contest", default="IMO",
                        help="AoPS: which contest (IMO, USAMO, EGMO, ...)")
    parser.add_argument("--years", default="2000-2024",
                        help="AoPS: year range e.g. 2010-2023")
    parser.add_argument("--max-threads", type=int, default=300,
                        help="AoPS: max threads to collect")
    args = parser.parse_args()

    year_start, year_end = map(int, args.years.split("-"))

    if args.source in ("evanchen", "all"):
        print("═══ EVAN CHEN .tex files ═══")
        scrape_evanchen(
            out_dir=os.path.join(args.out, "evanchen_tex"),
            dry_run=args.dry_run,
        )

    if args.source in ("aops", "all"):
        print("═══ AOPS threads ═══")
        raw_dir = os.path.join(args.out, "aops_raw")
        scrape_aops_forum(
            out_dir=raw_dir,
            forum_name=args.contest,
            year_start=year_start,
            year_end=year_end,
            max_threads=args.max_threads,
        )
        # Convert to training format
        raw_file = os.path.join(raw_dir, f"aops_{args.contest.lower()}.jsonl")
        if os.path.exists(raw_file):
            aops_to_training(
                in_file=raw_file,
                out_file=os.path.join(args.out, f"training_aops_{args.contest.lower()}.jsonl"),
            )


if __name__ == "__main__":
    main()
