#!/usr/bin/env python3
"""Reconcile finhank.json to the confirmed 14-song Hank Williams Sr. dataset scope.
Drops Alabama Waltz, re-points audio to originals_backup1/ sources, applies corrected
metadata, sets custom_tag, strips baked-in trigger tag from captions."""
import json, sys, shutil, os

SRC = "/home/fritz/Music/hank_sr"
PATH = SRC + "/finhank.json"
OB1 = SRC + "/originals_backup1"

# sample_id -> dict of authoritative fields
FIX = {
    "3299aa71": dict(file="I_Saw_The_Light.flac", genre="Country Gospel / Honky-Tonk", key="F major", ts="4/4", bpm=103),
    "1ae2fbda": dict(file="Move_It_On_Over.flac", genre="Country Boogie / 12-Bar Blues", key="C major", ts="4/4", bpm=157),
    "a90ffc11": dict(file="Im_So_Lonesome_I_Could_Cry.flac", genre="Country Ballad / Country Blues", key="E major", ts="3/4", bpm=111),
    "e06c4410": dict(file="Jambalaya_(On_The_Bayou)", genre="Cajun Country / Honky-Tonk", key="F major", ts="4/4", bpm=161),
    "430c3186": dict(file="Ill_Never_Get_Out_Of_This_World_Alive.flac", genre="Honky-Tonk / Country Blues", key="C major", ts="4/4", bpm=118),
    "1ba95d20": dict(file="Lovesick_Blues.flac", genre="Honky-Tonk / Vaudeville Blues", key="F major", ts="4/4", bpm=126),
    "fc654d18": dict(file="Honky_Tonk_Blues.flac", genre="Honky-Tonk / 12-Bar Blues", key="A major", ts="4/4", bpm=130),
    "c6ddc38b": dict(file="Hey_Good_Lookin.flac", genre="Honky-Tonk / Traditional Country", key="C major", ts="4/4", bpm=141),
    "b28da653": dict(file="Lost_Highway.flac", genre="Honky-Tonk Ballad / Folk", key="D major", ts="4/4", bpm=120),
    "4d435432": dict(file="Cold_Cold_Heart.flac", genre="Honky-Tonk / Country Ballad", key="D major", ts="4/4", bpm=114),
    "9d49706b": dict(file="Long_Gone_Lonesome_Blues.flac", genre="Country Blues / Honky-Tonk", key="E major", ts="4/4", bpm=112),
    "21751afb": dict(file="Honky_Tonkin.flac", genre="Honky-Tonk / Uptempo Swing", key="E major", ts="4/4", bpm=167),
    "f398169b": dict(file="Wild_Side_of_Life.flac", genre="Honky-Tonk / Traditional Country", key="G major", ts="4/4", bpm=120),
    "cf782abc": dict(file="Your_Cheatin_Heart.flac", genre="Honky-Tonk / Country Ballad", key="C major", ts="4/4", bpm=129),
}
DROP = {"1ed0a775"}  # The_Alabama_Waltz
TAG = "hank_williams_sr"
PREFIX = TAG + ", "

# backup
shutil.copy2(PATH, PATH + ".bak-before-reconcile")

with open(PATH, encoding="utf-8") as f:
    d = json.load(f)

before_ids = [s["id"] for s in d["samples"]]
missing = [i for i in FIX if i not in before_ids]
if missing:
    sys.exit(f"target ids not present in file: {missing}")

kept = []
for s in d["samples"]:
    if s["id"] in DROP:
        continue
    fix = FIX[s["id"]]
    ap = OB1 + "/" + fix["file"]
    if not os.path.exists(ap):
        sys.exit(f"audio source missing: {ap}")
    s["audio_path"] = ap
    s["filename"] = fix["file"]
    s["genre"] = fix["genre"]
    s["keyscale"] = fix["key"]
    s["timesignature"] = fix["ts"]
    s["bpm"] = fix["bpm"]
    s["custom_tag"] = TAG
    cap = s.get("caption") or ""
    if cap.startswith(PREFIX):
        s["caption"] = cap[len(PREFIX):]
    kept.append(s)

d["samples"] = kept
d["metadata"] = d.get("metadata") or {}
d["metadata"]["custom_tag"] = TAG
# tag_position prepended via custom_tag now; captions already clean
if d["metadata"].get("tag_position") != "prepend":
    d["metadata"]["tag_position"] = "prepend"
d["metadata"]["num_samples"] = len(kept)
d["metadata"]["instrumental_mode"] = "mixed"

with open(PATH, "w", encoding="utf-8") as f:
    json.dump(d, f, ensure_ascii=False, indent=2)

print("samples:", len(kept))
print("dropped:", len(DROP))
print("all kept ids mapped + audio resolve OK")
