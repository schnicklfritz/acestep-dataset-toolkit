# core/file_system.py
import os
import re
import shutil
import time
import logging

logger = logging.getLogger("file_system")

def execute_disk_rename(samples, rename_mode, options):
    """
    Natively mutates filenames directly inside the track folders on disk.
    Upholds strict Gentoo performance metrics and returns a list of mapping updates.
    """
    renamed_count = 0
    updates = []
    seen_names = set()
    counter = options.get("start_number", 1)

    for sample in samples:
        old_path = sample.get("audio_path", "")
        if not old_path or not os.path.exists(old_path):
            continue
            
        base_dir = os.path.dirname(old_path)
        old_name = sample.get("filename", "")
        stem, ext = os.path.splitext(old_name)

        # 🎛️ Dynamic Pattern Selection Mapping
        if rename_mode == "Song name (spaces → _)":
            # Strip prefixes like track numbers and replace whitespaces with underscores
            clean_stem = re.sub(r"^\s*\d{1,3}\s*[-._]\s*", "", stem)
            if " - " in clean_stem:
                clean_stem = clean_stem.split(" - ")[-1]
            new_name = re.sub(r"\s+", "_", clean_stem).strip("_") + ext
        elif rename_mode == "Find & Replace":
            find_txt = options.get("find_text", "")
            repl_txt = options.get("replace_text", "")
            new_name = stem.replace(find_txt, repl_txt) + ext if find_txt in stem else old_name
        elif rename_mode == "Prefix":
            new_name = options.get("prefix_text", "") + old_name
        elif rename_mode == "Suffix":
            new_name = stem + options.get("suffix_text", "") + ext
        else: # Number Sequence Mode
            pattern = options.get("pattern", "track_{n:03d}")
            try:
                new_name = pattern.format(n=counter) + ext
            except (KeyError, ValueError):
                new_name = pattern.replace("{n}", str(counter)) + ext
            counter += 1

        if new_name == old_name or new_name in seen_names:
            continue
            
        seen_names.add(new_name)
        new_path = os.path.join(base_dir, new_name)

        # 🛡️ Non-destructive filesystem protection backup
        if options.get("create_backup", True):
            stamp = time.strftime("%Y%m%d-%H%M%S")
            shutil.copy2(old_path, f"{old_path}.bak-{stamp}")

        try:
            os.rename(old_path, new_path)
            updates.append({
                "id": sample.get("id"),
                "new_filename": new_name,
                "new_audio_path": new_path
            })
            renamed_count += 1
        except OSError as e:
            print(f"OS File system rename collision error: {e}")

    return updates, renamed_count
