# core/file_system.py
import os
import re
import shutil
import time
import zipfile
import tempfile
import logging

logger = logging.getLogger("file_system")

def compress_dataset_folder(samples) -> str:
    """
    Gentoo Modularity: Single Responsibility Archiving.
    Gathers only active tracks from the memory matrix, completely skipping directory bloat.
    Returns the absolute path to the temporary .zip file archive on success.
    """
    if not samples:
        logger.error("Compression abort: Sample array is empty.")
        return ""

    try:
        # Create a unique temporary directory sandbox destination to avoid system file lock issues
        temp_dir = tempfile.gettempdir()
        zip_output_path = os.path.join(temp_dir, "ace_step_upload_bundle.zip")
        
        # Overwrite previous compression snapshots if they exist to keep disk spaces lean
        if os.path.exists(zip_output_path):
            os.remove(zip_output_path)

        print(f"📦 Commencing zip compilation for {len(samples)} tracking targets...")
        
        with zipfile.ZipFile(zip_output_path, 'w', zipfile.ZIP_DEFLATED) as zip_file:
            for sample in samples:
                audio_path = sample.get("audio_path", "")
                filename = sample.get("filename", "")
                
                if not audio_path or not os.path.exists(audio_path):
                    print(f"⚠️ Skipping missing source track during archiving: {filename}")
                    continue
                
                # Write the target asset file into the root layer inside the archive container
                # This guarantees the remote layout is flat: /kaggle/input/ace-step-upload-bundle/song.wav
                zip_file.write(audio_path, arcname=filename)
                
        print(f"✅ Archive compilation complete: {zip_output_path}")
        return zip_output_path

    except Exception as e:
        print(f"❌ Structural compression failure: {str(e)}")
        return ""

def launch_remote_kaggle_console(username: str, notebook_slug: str):
    """Zero-overhead browser cross-platform dispatcher."""
    import webbrowser
    url = f"https://kaggle.com{username}/{notebook_slug}"
    webbrowser.open_new_tab(url)

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

def launch_remote_kaggle_console(username, notebook_slug):
    """
    Launches a dedicated browser tab directly to the Kaggle notebook instance.
    Zero-overhead, cross-platform call.
    """
    import webbrowser
    
    # Construct the canonical URL targeting your specific container space
    url = f"https://www.kaggle.com/{username}/{notebook_slug}"
    
    # 🚀 Opens instantly inside the user's default browser profile tab
    webbrowser.open_new_tab(url)
