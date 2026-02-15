# helpers/planner_config.py
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any


@dataclass
class CharacterSpec:
    """Placeholder character input. Merge/replace with your real character schema later."""
    enabled: bool = False
    name: str = ""
    description: str = ""
    reference_image_path: str = ""  # optional


@dataclass
class PlannerConfig:
    """
    Single source of truth for the wizard.
    Keep this lightweight: allow AUTO and leave the heavy resolving to your pipeline.
    """
    # Wizard mode
    advanced_mode: bool = False

    # Core idea
    prompt: str = ""
    extra_details: str = ""

    # Characters (max 2 in default flow)
    character_1: CharacterSpec = field(default_factory=CharacterSpec)
    character_2: CharacterSpec = field(default_factory=CharacterSpec)

    # Output type
    output_type: str = "narrated_story"  # narrated_story | music_videoclip

    # Music / narration choices
    want_music: bool = False
    music_mode: str = "auto"  # auto | ace_step | heartmula | user_file
    user_music_path: str = ""
    lyrics_mode: str = "auto_from_story"  # auto_from_story | user_lyrics | user_audio
    user_lyrics: str = ""
    want_narration: bool = True
    narration_mode: str = "tts"  # tts | user_voice
    user_voice_path: str = ""

    # Mix (placeholder)
    music_volume: int = 45
    voice_volume: int = 70

    # Output settings
    quality: str = "balanced"  # fast | balanced | best
    duration_sec: int = 30
    aspect: str = "16:9"  # 16:9 | 9:16 | 1:1

    # Model control (advanced)
    image_model_strategy: str = "auto"  # auto | fixed | priority_list
    video_model_strategy: str = "auto"
    fixed_image_model: str = ""
    fixed_video_model: str = ""
    image_model_priority: List[str] = field(default_factory=list)
    video_model_priority: List[str] = field(default_factory=list)

    # Advanced: allow stage editing pauses
    allow_edit_images: bool = False  # pause after images stage completes
    allow_edit_videos: bool = False  # pause after videos stage completes

    # Runtime / resolved selections (filled by pipeline; shown on Preview page)
    resolved: Dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "advanced_mode": self.advanced_mode,
            "prompt": self.prompt,
            "extra_details": self.extra_details,
            "character_1": self.character_1.__dict__,
            "character_2": self.character_2.__dict__,
            "output_type": self.output_type,
            "want_music": self.want_music,
            "music_mode": self.music_mode,
            "user_music_path": self.user_music_path,
            "lyrics_mode": self.lyrics_mode,
            "user_lyrics": self.user_lyrics,
            "want_narration": self.want_narration,
            "narration_mode": self.narration_mode,
            "user_voice_path": self.user_voice_path,
            "music_volume": self.music_volume,
            "voice_volume": self.voice_volume,
            "quality": self.quality,
            "duration_sec": self.duration_sec,
            "aspect": self.aspect,
            "image_model_strategy": self.image_model_strategy,
            "video_model_strategy": self.video_model_strategy,
            "fixed_image_model": self.fixed_image_model,
            "fixed_video_model": self.fixed_video_model,
            "image_model_priority": list(self.image_model_priority),
            "video_model_priority": list(self.video_model_priority),
            "allow_edit_images": self.allow_edit_images,
            "allow_edit_videos": self.allow_edit_videos,
            "resolved": dict(self.resolved),
        }
