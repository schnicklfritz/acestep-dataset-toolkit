import os
import csv
import json
import google.generativeai as genai
from pydantic import BaseModel, Field

# 1. API Configuration
genai.configure(api_key=os.environ["GEMINI_API_KEY"])

# 2. ACE-Step 1.5 Strict JSON Schema
class AceStepAudioSchema(BaseModel):
    song_id: str = Field(description="Unique tracking ID for the audio sample file.")
    time_signature: str = Field(description="Dedicated metadata field for structural time signature (e.g., '4/4', '3/3'). Never put in the caption.")
    tempo_bpm: int = Field(description="Numeric metadata field for BPM. Never place inside text blocks.")
    caption: str = Field(
        description="Global conditioning string. Format: [Trigger Tag], [Primary Genre], [Subgenre/Mood], [2-3 Specific Instruments], [Vocal Style], [Production & Mix], [Era/Aesthetic]. Followed by 2-3 sentences detailing dynamic build, arrangement transitions, and energy flow."
    )
    lyrics: str = Field(
        description="Temporal script. Must use capitalized section markers enclosed in brackets, separated by a blank line (\\n\\n). Must include maximum 3 comma-separated architectural descriptors attached by a dash, e.g., '[Intro - acoustic rhythm]' or '[Verse 1 - raspy vocal, low energy]'. Line length should target 6 to 10 syllables."
    )

# Initialize the optimized flash engine for fast catalog lookups
model = genai.GenerativeModel('gemini-2.5-flash')

# 3. System Blueprint Rule Injection
system_instruction = """
You are a music annotation expert for the ACE-Step 1.5 music generation dataset wrapper.
Your task is to transcribe or generate data structures based on historical accuracy for specified classic country tracks.

For the 'lyrics' field, you MUST strictly adhere to these rules:
1. Every structural section must use capitalized markers: [Intro], [Verse], [Chorus], [Outro], [Instrumental].
2. Append maximum 3 descriptors after a dash. For example: [Chorus - powerful belting, high energy].
3. For Hank Williams, valid architectural tags include: 'acoustic rhythm', 'steel guitar solo', 'fiddle lead', 'raspy vocal', 'yodeling', 'clear vocals', 'low energy', 'high energy', 'stripped back'.
4. Separate independent brackets using two newlines (\\n\\n).
5. Capitalize entire lyric lines only if the section represents a belted or shouted vocal delivery.
"""

# 4. Target Batch Configuration
track_list = [
    {"id": "hw_01", "title": "Alabama Waltz", "artist": "Hank Williams"},
    {"id": "hw_02", "title": "I'm So Lonesome I Could Cry", "artist": "Hank Williams"},
    {"id": "hw_03", "title": "Hey, Good Lookin'", "artist": "Hank Williams"},
    {"id": "hw_04", "title": "Cold, Cold Heart", "artist": "Hank Williams"}
]

dataset_output = []

# 5. Extraction Engine
print("Commencing ACE-Step 1.5 Master Dataset Tagging...")
for track in track_list:
    prompt = f"Process metadata, global sound portraits, and full structural lyric timelines for '{track['title']}' by {track['artist']}."
    
    try:
        response = model.generate_content(
            f"{system_instruction}\n\nUser Request: {prompt}",
            generation_config=genai.GenerationConfig(
                response_mime_type="application/json",
                response_schema=AceStepAudioSchema,
                temperature=0.1 # Lock low variance to enforce schema adherence
            ),
        )
        
        parsed_data = json.loads(response.text)
        dataset_output.append(parsed_data)
        print(f"Successfully processed: {track['title']}")
        
    except Exception as e:
        print(f"Failed parsing validation execution on track {track['title']}: {e}")

# 6. Save out to Dataset Master Repository
with open("ace_step_dataset_manifest.csv", mode="w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=["song_id", "time_signature", "tempo_bpm", "caption", "lyrics"])
    writer.writeheader()
    writer.writerows(dataset_output)

print("Export complete. Dataset saved to disk.")

