#!/usr/bin/env python3
"""Organize downloaded MVSEP PolarFormer stems into per-song folders.

Convention (hard user rule):
  * one folder per song, named by the song title
  * folder names: LOWERCASE, no spaces / ' / " / numbers / special chars
  * separator is underscore '_'
  * stems renamed simple & uniform: vocals.* , instrumental.*
  * MVSEP returns base + '-fullness-level-1' variants; 'fullness' preferred.
  * ALWAYS copies source; never deletes or moves them.
"""
import argparse, os, re, shutil, sys

SONG_FOLDER_MAP = {
    "the-alabama-waltz": "the_alabama_waltz",
    "cold-cold-heart": "cold_cold_heart",
    "hey-good-lookin": "hey_good_lookin",
    "honky-tonk-blues": "honky_tonk_blues",
    "honky-tonkin": "honky_tonkin",
    "i-saw-the-light": "i_saw_the_light",
    "i-ll-never-get-out-of-this-world-alive": "ill_never_get_out_of_this_world_alive",
    "i-m-so-lonesome-i-could-cry": "im_so_lonesome_i_could_cry",
    "jambalaya-on-the-bayou": "jambalaya_on_the_bayou",
    "kaw-liga": "kaw_liga",
    "long-gone-lonesome-blues": "long_gone_lonesome_blues",
    "lost-highway": "lost_highway",
    "lovesick-blues": "lovesick_blues",
    "move-it-on-over": "move_it_on_over",
    "ramblin-man": "ramblin_man",
    "there-s-a-tear-in-my-beer": "theres_a_tear_in_my_beer",
    "weary-blues-from-waitin": "weary_blues_from_waitin",
    "wild-side-of-life": "wild_side_of_life",
    "your-cheatin-heart": "your_cheatin_heart",
}

def parse(fname):
    """-> (song_slug, stem_type, is_fullness, ext) or None."""
    low = fname.lower()
    if "_bs_polarformer_" not in low or not low.endswith((".flac",".wav",".mp3")):
        return None
    is_full = "-fullness-level-1" in low
    if "instrumental" in low:
        stem = "instrumental"
    elif "vocals" in low:
        stem = "vocals"
    else:
        return None
    m = re.search(r"-[0-9a-f]{10,}-(.+?)\.(?:flac|wav|mp3)", low)
    slug = m.group(1).strip() if m else ""
    slug = re.sub(r"\.?_.*(bs_polarformer).*", "", slug)
    slug = slug.split("._")[0].split("_")[0]
    return slug, stem, is_full, os.path.splitext(low)[1]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="/home/fritz/Downloads")
    ap.add_argument("--dest", default="/home/fritz/Music/hank_sr/stems")
    ap.add_argument("--variant", choices=["full","base","both"], default="full")
    ap.add_argument("--commit", action="store_true")
    ap.add_argument("--only", default="")
    a = ap.parse_args()
    if not os.path.isdir(a.src):
        sys.exit(f"source not found: {a.src}")
    os.makedirs(a.dest, exist_ok=True)
    only = {x.strip() for x in a.only.split(",") if x.strip()} if a.only else None
    bysong = {}
    for fname in sorted(os.listdir(a.src)):
        r = parse(fname)
        if not r:
            continue
        slug, stem, is_full, ext = r
        bysong.setdefault(slug, {}).setdefault("full" if is_full else "base", {})[stem] = (fname, ext)
    copied, skipped = 0, 0
    for slug in sorted(bysong):
        folder = SONG_FOLDER_MAP.get(slug)
        if folder is None:
            print(f"[?? UNMAPPED] {slug}")
            skipped += 1
            continue
        if only and folder not in only:
            continue
        variants = bysong[slug]
        writes = []
        if a.variant == "both":
            have_full = "full" in variants
            for stem in sorted(set(list(variants.get("full", {})) + list(variants.get("base", {})))):
                if have_full and stem in variants["full"]:
                    writes.append((f"{stem}_fullness{variants['full'][stem][1]}", variants["full"][stem][0]))
                if stem in variants["base"]:
                    writes.append((f"{stem}_base{variants['base'][stem][1]}", variants["base"][stem][0]))
        else:
            want = a.variant
            other = "base" if want == "full" else "full"
            pool = variants.get(want) or variants.get(other) or {}
            for stem in sorted(pool):
                writes.append((f"{stem}{pool[stem][1]}", pool[stem][0]))
        print(f"[{'copy' if a.commit else 'dry '}] {folder}/")
        for dest, srcf in writes:
            print(f"        {dest:20} <- {srcf}")
        if a.commit:
            dd = os.path.join(a.dest, folder)
            os.makedirs(dd, exist_ok=True)
            for dest, srcf in writes:
                shutil.copy2(os.path.join(a.src, srcf), os.path.join(dd, dest))
                copied += 1
    print(f"\nskipped unmapped: {skipped}" + ("" if not a.commit else f" | copied {copied} files"))
    if not a.commit:
        print("Run with --commit to actually copy (dest source stays intact).")

if __name__ == "__main__":
    main()
