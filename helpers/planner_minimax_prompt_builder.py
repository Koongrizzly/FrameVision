from __future__ import annotations
import re
from typing import Any, Dict, List, Sequence, Tuple

STYLE_PROFILES = {'Cinematic realism': 'Naturalistic production design with physically accurate materials, motivated practical lighting, nuanced skin and surface texture, cinematic contrast, controlled depth of field and believable environmental interaction. Movement carries real weight and inertia; avoid synthetic gloss and generic stock-video polish.', 'Live action': 'Grounded live-action photography with authentic locations, practical light sources, natural exposure roll-off, convincing wardrobe and props, restrained color grading, human micro-expressions and realistic lens behavior. Preserve physical plausibility and documentary-level environmental detail.', '3D animation': 'Premium feature-quality 3D animation with appealing sculpted forms, expressive facial rigs, detailed materials, soft global illumination, controlled subsurface scattering and confident animated posing. Motion uses readable anticipation, follow-through, squash and stretch without becoming weightless.', 'Cartoon': 'Bold graphic cartoon design with clean silhouettes, expressive shape language, simplified but intentional backgrounds, punchy color separation and highly readable poses. Use elastic timing, visual exaggeration and crisp comedic reactions while keeping character construction consistent.', 'Anime': 'Polished cinematic anime with precise linework, controlled cel shading, expressive eyes, dynamic perspective, atmospheric painted backgrounds, speed accents and dramatic color scripting. Use held poses punctuated by fluid bursts of action and carefully composed emotional close-ups.', 'Illustrated': 'Living editorial illustration with visible authored linework, layered pigment or brush texture, selective detail, designed negative space and sophisticated color harmony. Motion should feel like the illustration has come alive while preserving its handmade surface and graphic composition.', 'Game cinematic': 'High-end real-time game cinematic with detailed characters and environments, volumetric atmosphere, dramatic rim lighting, physically based materials, heroic composition and responsive action animation. Camera and editing feel authored for a premium narrative cutscene.', 'Gameplay / first-person': 'Immersive first-person gameplay presentation with stable player geography, responsive head and weapon motion, readable environmental navigation, game-authentic lighting and tactile interaction. Camera acceleration, recoil and impacts remain controlled enough to preserve spatial clarity.', 'Stop motion': 'Handcrafted stop-motion production with tactile puppets, miniature sets, visible fabric, clay, wood or paper texture, practical miniature lighting, shallow macro depth of field and intentionally stepped frame-by-frame movement. Include tiny puppet-settle imperfections while avoiding smooth CGI motion.', 'Mixed live action and hand-drawn animation': 'Live-action photography integrated with expressive hand-drawn marks that wrap around surfaces, cast light, react to movement and inhabit the same perspective. Preserve natural footage texture while animated lines, paint and symbols retain visible human variation.', 'Graphic motion design': 'Precision motion design with bold typography, geometric systems, controlled grids, clean masking, deliberate transitions and rhythmically choreographed shape animation. Every movement reinforces hierarchy and composition; surfaces remain crisp and production-ready.', 'Animated poster': 'A striking poster composition that evolves through restrained parallax, animated lighting, atmospheric particles, moving type and one memorable visual transformation. Maintain a strong hero layout and finish on a clean, readable key art frame.', 'Premium product commercial': 'Luxury commercial photography with immaculate product geometry, controlled studio reflections, refined material rendering, macro detail, elegant camera motion and sculpted highlight falloff. Interactions showcase function and craftsmanship without inventing labels, features or claims.', 'Visceral cinematic horror': 'Tactile cinematic horror with oppressive darkness, sickly practical light, damp and decayed surfaces, uncomfortable proximity, deep negative space and brief fragments of disturbing detail. Withhold the threat before revealing it; use imperfect handheld movement, abrupt stillness and low-frequency physical sound rather than constant spectacle.', 'Psychological thriller': 'Controlled psychological-thriller imagery with compressed space, reflections, frames within frames, symmetrical compositions that gradually destabilize, muted color contaminated by one recurring accent and slow invasive camera movement. Emphasize uncertain perception, micro-expressions, off-screen implication and subjective sound.', 'Gothic whimsy': 'Playfully macabre storybook gothic design with crooked architecture, elongated silhouettes, spindly trees, theatrical miniature-like sets, moonlit fog and handcrafted surface imperfections. Use charcoal, bone-white and faded jewel tones, angular compositions and expressive movement that balances childlike wonder with elegant unease; avoid direct imitation of any named filmmaker.', 'Dark fairy tale': 'Lush but threatening folklore imagery with ancient forests, worn storybook textures, candlelit chiaroscuro, jewel-toned shadows, weathered costumes and beautiful objects carrying subtle danger. Frame the world with mythic scale, enchanted atmosphere and a constant tension between wonder and menace.', 'Cosmic horror': 'Overwhelming cosmic-horror scale with tiny human figures, impossible geometry, ancient nonhuman structures, distorted horizons, starless voids and light behaving in physically unsettling ways. Reveal incomprehensible forms only partially; emphasize awe, insignificance and deep subsonic resonance over conventional monsters.', 'Supernatural mystery': 'Atmospheric supernatural mystery with ordinary locations disturbed by one impossible detail, cool nocturnal color, pools of practical light, drifting haze, reflective surfaces and patient observational framing. Build evidence gradually through environmental changes, reactions and suggestive off-screen sound.', 'Neo-noir crime': 'Modern neo-noir crime photography with hard directional light, deep blacks, wet streets, sodium and neon color contrast, glass reflections, smoke and morally charged close-ups. Use long lenses, oblique framing and deliberate urban camera moves with restrained, dangerous energy.', 'Analog found footage': 'Degraded consumer-video authenticity with imperfect autofocus, sensor noise, clipped highlights, rolling exposure, timestamp-era color, nervous reframing and accidental obstructions. Events must feel captured rather than staged; preserve plausible operator behavior and unsettling off-camera audio without decorative digital glitch overload.', 'Retro science fiction': 'Tactile retro-futurism built from practical miniatures, painted control panels, CRT displays, analog switches, brushed metal, colored instrument light and optimistic mid-century industrial design. Combine clean graphic shapes with visible model-making detail and period-authentic optical effects.', 'Dystopian future': 'Severe dystopian worldbuilding with monumental surveillance architecture, dense infrastructure, polluted atmosphere, utilitarian clothing, harsh industrial lighting and controlled institutional color. Contrast overwhelming systems with vulnerable human-scale details and credible environmental wear.', 'Disaster spectacle': 'Large-scale disaster cinema with clearly established geography, escalating structural failure, credible mass and debris physics, atmospheric depth, human reaction inserts and wide shots that communicate enormous scale. Destruction unfolds as connected cause and effect rather than random visual noise.', 'Action blockbuster': 'Premium action-blockbuster imagery with strong chase geography, bold silhouettes, dynamic parallax, practical-feeling impacts, readable stunt motion, aggressive but motivated camera placement and escalating shot scale. Maintain screen direction and physical continuity through every cut.', 'Pulp adventure': 'Colorful pulp-adventure energy with exotic practical locations, weathered maps and machinery, heroic silhouettes, golden light, dangerous terrain and bold serialized storytelling. Camera movement feels athletic and optimistic; action favors ingenious escapes and tactile set pieces.', 'Romantic fantasy': 'Luminous romantic fantasy with ethereal natural light, flowing fabric, enchanted landscapes, delicate particles, elegant production design and intimate expressive close-ups. Use graceful camera movement and rich color transitions to make emotional connection feel physically present in the environment.', 'Surreal dreamscape': 'Poetic surrealism with lucid visual logic, seamless impossible transitions, symbolic objects, altered scale, gravity-defying but graceful motion and environments that transform through visual association. Maintain coherent lighting and composition so the dream feels intentional rather than randomly generated.', 'Stand-up comedy': 'Authentic live stand-up staging with one clearly anchored comedian, microphone, stage and believable audience geography. Favor confident medium shots and close-ups for delivery, occasional wider room views and selective audience reaction cuts. Preserve natural performance gestures, pauses and facial timing; keep spoken material concise enough for the clip and avoid unnecessary cinematic action that distracts from the comedian.', 'TV comedy': 'Polished episodic television comedy with grounded locations, readable character blocking, natural ensemble performances and clean coverage that supports conversation and reaction. Let humor come from personality, situation, dry remarks, awkward timing or mild escalation rather than forcing every beat into a punchline. Use flexible medium shots, two-shots, close reactions and motivated inserts while preserving spatial continuity.', 'Sitcom': 'Grounded episodic television comedy built around familiar domestic, workplace or neighborhood situations, natural ensemble performances, clear character blocking and warm practical interiors. Let humor develop through everyday inconvenience, personality clashes, misunderstandings, dry remarks, awkward pauses and readable reaction beats rather than constant exaggeration. Favor conversational medium shots, two-shots, doorway entrances, restrained close-ups and reaction cuts that clearly establish who is speaking and listening. Stage each interaction as a simple setup, interruption or complication, response and comic payoff while keeping characters, eyelines and room geography consistent.', 'Slapstick comedy': 'Physical comedy with crystal-clear spatial geography, readable anticipation, cause and effect, exaggerated but coherent body mechanics and a visible reaction after each impact or failed attempt. Favor wider coverage for the main gag, then cut closer for contact details and reactions. Escalate physical complications without random motion, identity drift or impossible character positions unless the absurdity is explicitly part of the joke.', 'Sketch comedy': 'Short-form comedy built around a fast readable setup, escalation and decisive punchline. Use distinct visual beats, purposeful cuts, strong character reactions and concise dialogue when useful; each shot should advance the gag rather than repeat it. Preserve character identity, wardrobe, props and screen geography so the joke remains understandable even when the pacing becomes rapid.', 'Dark comedy': 'Grounded, serious-looking comedy in which irony, uncomfortable timing, inappropriate reactions or absurd consequences create the humor. Keep cinematography restrained and believable, performances controlled and reactions specific rather than cartoonish. Let the contrast between the sober presentation and the comic situation do the work; do not drift into horror merely because the subject matter is dark.', 'Parody': 'Convincingly reproduce the visual grammar, staging and dramatic conventions of the genre being spoofed, then exaggerate selected clichés for comic effect. Keep the underlying filmmaking competent enough that the parody is recognizable, use clear setups and reaction beats, and avoid random silliness that is unrelated to the target genre.', 'Surreal comedy': 'Matter-of-fact comedy built from impossible events, dream logic and absurd juxtapositions while preserving clear character identity, geography and internal continuity. Treat bizarre events as normal within the scene, use calm reactions or precise contrast to make the absurdity readable, and escalate through deliberate visual logic rather than uncontrolled randomness.'}

STYLE_NAMES = tuple(STYLE_PROFILES.keys())
CAMERA_PRESETS = (
    "Automatic cinematic camera",
    "Static composition",
    "Gentle push-in",
    "Smooth tracking shot",
    "Handheld realism",
    "Orbit around the subject",
    "Crane reveal",
    "Fast FPV movement",
)
FLOW_PRESETS = (
    "Let H3 decide",
    "One continuous shot",
    "Multiple cinematic shots",
    "Dynamic action sequence",
    "Suspense / thriller buildup",
    "Horror escalation and reveal",
    "Fast commercial-style cuts",
    "Slow, deliberate pacing",
)

_ACTION_RE = re.compile(
    r"\b(fight|fights|fighting|combat|punch|kick|strike|block|dodge|slam|throw|"
    r"attack|attacks|ambush|chase|pursuit|escape|run|sprint|jump|leap|shoot|gunfire|"
    r"battle|duel|martial|stunt|crash|explosion|explodes|weapon|intercept|confront)\b", re.I
)
_SUSPENSE_RE = re.compile(r"\b(suspense|stalk|stalking|hiding|hide|search|threat|unease|mystery|watching|follows?|creeps?)\b", re.I)
_DIALOGUE_RE = re.compile(r"\b(says?|asks?|replies?|answers?|whispers?|shouts?|yells?|speaks?|dialogue|conversation|explains?|reveals?)\b", re.I)
_HORROR_RE = re.compile(r"\b(horror|monster|demon|ghost|haunt|terrifying|gore|nightmare|creature)\b", re.I)

def infer_project_style(text: str) -> str:
    s = (text or "").lower()
    rules = [
        ("Gameplay / first-person", ("first-person gameplay","first person gameplay","fps game","gameplay")),
        ("Anime", ("anime","manga")),
        ("Cartoon", ("cartoon","toon")),
        ("3D animation", ("pixar","3d animation","3d animated","animated feature","cgi animation")),
        ("Stop motion", ("stop motion","stop-motion","claymation")),
        ("Mixed live action and hand-drawn animation", ("live action and hand-drawn","live-action and hand-drawn","mixed animation")),
        ("Graphic motion design", ("motion design","motion graphics")),
        ("Animated poster", ("animated poster","motion poster")),
        ("Game cinematic", ("game cinematic","game trailer","cutscene")),
        ("Analog found footage", ("found footage","vhs","camcorder")),
        ("Visceral cinematic horror", ("visceral horror","body horror")),
        ("Psychological thriller", ("psychological thriller",)),
        ("Neo-noir crime", ("neo-noir","neo noir")),
        ("Retro science fiction", ("retro science fiction","retro sci-fi","retro scifi")),
        ("Dystopian future", ("dystopian","dystopia")),
        ("Disaster spectacle", ("disaster movie","disaster film","catastrophe")),
        ("Pulp adventure", ("pulp adventure",)),
        ("Romantic fantasy", ("romantic fantasy",)),
        ("Surreal dreamscape", ("surreal dream","dreamscape")),
        ("Stand-up comedy", ("stand-up","standup comedy")),
        ("Sitcom", ("sitcom",)),
        ("Slapstick comedy", ("slapstick",)),
        ("Sketch comedy", ("sketch comedy",)),
        ("Dark comedy", ("dark comedy",)),
        ("Parody", ("parody","spoof")),
        ("Surreal comedy", ("surreal comedy",)),
        ("TV comedy", ("tv comedy","television comedy")),
    ]
    for style, needles in rules:
        if any(n in s for n in needles):
            return style
    # Movie / cinematic / named movie-like prompts are deliberately kept in a stable
    # photoreal movie language. Action is selected per shot through shot-flow instead.
    if any(n in s for n in ("movie","film","cinematic","matrix","live action","live-action","realistic")):
        return "Cinematic realism"
    return "Cinematic realism"

def choose_flow(text: str, story_role: str = "", section_key: str = "") -> str:
    s = " ".join((text or "", story_role or "", section_key or ""))
    if _HORROR_RE.search(s):
        return "Horror escalation and reveal"
    if _ACTION_RE.search(s):
        return "Dynamic action sequence"
    if _SUSPENSE_RE.search(s):
        return "Suspense / thriller buildup"
    if _DIALOGUE_RE.search(s):
        return "Multiple cinematic shots"
    return "Let H3 decide"

def choose_camera(text: str, flow: str) -> str:
    s = text or ""
    if flow == "Dynamic action sequence":
        # H3 tends to blur when action is described as physically ultra-fast.
        # Camera can be energetic while subject motion remains readable.
        return "Handheld realism"
    if flow in ("Suspense / thriller buildup", "Horror escalation and reveal"):
        return "Gentle push-in"
    if _DIALOGUE_RE.search(s):
        return "Static composition"
    if re.search(r"\b(walk|walking|run|running|follow|follows|approach|approaches|chase)\b", s, re.I):
        return "Smooth tracking shot"
    if re.search(r"\b(reveal|arrival|arrives|enters|skyline|tower|large|vast)\b", s, re.I):
        return "Crane reveal"
    return "Automatic cinematic camera"

def shot_count(duration: float, flow: str) -> int:
    d = max(1.0, float(duration or 1.0))
    if flow == "One continuous shot":
        return 5 if d >= 12 else 4 if d >= 8 else 3
    if d >= 13:
        return 6
    if d >= 9:
        return 5
    return 4 if d >= 6 else 3

def timeline_ranges(duration: float, flow: str) -> List[Tuple[float,float]]:
    count = shot_count(duration, flow)
    d = max(1.0, float(duration or 1.0))
    return [(d*i/count, d*(i+1)/count) for i in range(count)]

def _fmt_time(sec: float) -> str:
    sec = max(0.0, float(sec))
    mins = int(sec // 60)
    rem = sec - mins*60
    return f"{mins}:{rem:04.1f}"

def style_profile(style: str) -> str:
    return STYLE_PROFILES.get(style) or STYLE_PROFILES["Cinematic realism"]

def flow_profile(flow: str, duration: float) -> str:
    count = shot_count(duration, flow)
    if flow == "Dynamic action sequence":
        return (
            f"Use {count} clearly differentiated cinematic action beats. Establish geography first, then use "
            "controlled tracking, readable close combat or pursuit coverage, one clear impact/reaction beat, "
            "and a decisive climax. Keep body motion crisp and physically readable; do not ask subjects to move "
            "at extreme or blurry speed. Preserve screen direction and spatial continuity."
        )
    if flow == "Suspense / thriller buildup":
        return f"Use {count} progressively tighter beats, delaying confirmation and ending on a decisive reveal."
    if flow == "Horror escalation and reveal":
        return f"Use {count} escalating beats from normality to evidence, threatened reaction and a final reveal."
    if flow == "Multiple cinematic shots":
        return f"Use {count} distinct motivated shots that advance the beat and end on a consequence or reaction."
    if flow == "Slow, deliberate pacing":
        return "Use patient compositions and restrained camera movement so every action and reaction registers."
    return f"Choose {count} clear cinematic beats that progress the action without repeating the same composition."

def camera_profile(camera: str, flow: str) -> str:
    if camera == "Static composition":
        return "Use stable medium/two-shot framing with motivated cut-ins and clear eyelines."
    if camera == "Gentle push-in":
        return "Use a restrained push-in that increases tension while preserving subject clarity."
    if camera == "Smooth tracking shot":
        return "Use controlled tracking that follows the subject while keeping geography readable."
    if camera == "Handheld realism":
        return "Use controlled action handheld coverage with stable readable framing at impact moments; avoid shaky blur."
    if camera == "Orbit around the subject":
        return "Use a deliberate orbit only while the subject action remains spatially clear."
    if camera == "Crane reveal":
        return "Use a measured crane reveal to expose scale or a new story element."
    if camera == "Fast FPV movement":
        return "Use energetic first-person camera travel but keep subject motion and environmental detail readable."
    if flow == "Dynamic action sequence":
        return "Alternate wide geography, controlled low tracking, medium combat coverage, close impact detail and reaction shots."
    return "Use motivated cinematic framing and camera movement that serves the story beat."

def _split_actions(text: str) -> List[str]:
    s = re.sub(r"\s+", " ", (text or "").strip())
    if not s:
        return ["The current story beat unfolds clearly and reaches a visible consequence."]
    # Preserve the story; this only creates internal timing chunks.
    parts = [p.strip(" .") for p in re.split(r"(?<=[.!?])\s+|;\s+|\bthen\b", s, flags=re.I) if p.strip(" .")]
    return parts or [s]

def _fit_actions(parts: Sequence[str], count: int) -> List[str]:
    arr = list(parts) or ["The current story beat progresses."]
    if len(arr) == count:
        return arr
    if len(arr) > count:
        out=[]
        for i in range(count):
            a=int(i*len(arr)/count); b=max(a+1,int((i+1)*len(arr)/count))
            out.append("; then ".join(arr[a:b]))
        return out
    # Unlike the old Planner, never repeat the same beat verbatim. Extend it with
    # cause/effect stages while preserving the original event.
    stages = [
        "Establish the subjects and immediate geography before the decisive movement.",
        "Develop the same event through a visible cause-and-effect action.",
        "Show the physical consequence and a readable reaction.",
        "Escalate the existing action without introducing an unrelated new plot.",
        "Land the beat on a decisive pose, result or story change.",
        "Hold just long enough for the consequence to register before the cut.",
    ]
    out=[]
    for i in range(count):
        if i < len(arr):
            out.append(arr[i])
        else:
            out.append(stages[min(i-len(arr), len(stages)-1)])
    return out

def _subjectize(text: str, subjects: Sequence[Dict[str,Any]]) -> str:
    out = text or ""
    # Longer names first to avoid partial replacements.
    for sub in sorted(subjects, key=lambda x: len(str(x.get("name") or "")), reverse=True):
        name = str(sub.get("name") or "").strip()
        num = int(sub.get("subject_number") or 0)
        if not name or num <= 0:
            continue
        aliases = {name, re.sub(r"[-_]+"," ",name)}
        if name.lower().startswith("mr "):
            aliases.add(name[3:])
        if name.lower().startswith("the "):
            aliases.add(name[4:])
        for alias in sorted(aliases, key=len, reverse=True):
            if len(alias) < 3:
                continue
            out = re.sub(rf"(?<!<Subject )\b{re.escape(alias)}\b", f"<Subject {num}>", out, flags=re.I)
    return out

def _dialogue_tags(text: str, subjects: Sequence[Dict[str,Any]]) -> str:
    s = text or ""
    by_name = {str(x.get("name") or "").lower(): int(x.get("subject_number") or 0) for x in subjects}
    # NAME says, "..." / NAME: "..."
    pat = re.compile(r'\b([A-Za-z][A-Za-z0-9 _-]{1,35})\s+(?:says?|asks?|replies?|whispers?|shouts?|yells?)\s*[:,]?\s*[“"]([^"”]{1,220})[”"]', re.I)
    def repl(m):
        raw=m.group(1).strip()
        num=0
        for n,v in by_name.items():
            if raw.lower()==n or raw.lower().endswith(n):
                num=v; break
        speaker=f"S{num}" if num else "S1"
        return f'<d>[English][{speaker}][natural delivery] {m.group(2).strip()}</d>'
    s = pat.sub(repl, s)
    return s

def _project_world_lock(project_idea: str) -> str:
    text = re.sub(r"\\s+", " ", str(project_idea or "")).strip()
    if not text:
        return "Use a story-appropriate cinematic environment; never inherit the reference image backdrop."
    sentences = [x.strip() for x in re.split(r"(?<=[.!?])\\s+", text) if x.strip()]
    env_re = re.compile(
        r"\\b(world|setting|set in|environment|background|location|street|streets|alley|alleyway|city|cityscape|"
        r"subway|tunnel|tunnels|rooftop|rooftops|corridor|hallway|room|interior|exterior|forest|desert|beach|space|"
        r"cyberpunk|matrix|dystopian|futuristic|industrial|lighting|light|neon|rain|wet|atmosphere|sky|landscape)\\b",
        re.I,
    )
    chosen = [x for x in sentences if env_re.search(x)]
    if not chosen:
        chosen = sentences[:1]
    world = " ".join(chosen[:3]).strip()
    if len(world) > 1200:
        world = world[:1200].rsplit(" ", 1)[0] + "."
    return world

def build_prompt(
    *,
    project_idea: str,
    beat_text: str,
    duration_sec: float,
    ratio: str,
    resolution: str,
    subjects: Sequence[Dict[str,Any]],
    project_style: str = "",
    story_role: str = "",
    section_key: str = "",
    sound_enabled: bool = True,
) -> Dict[str,Any]:
    style = project_style if project_style in STYLE_PROFILES else infer_project_style(project_idea)
    flow = choose_flow(beat_text, story_role, section_key)
    camera = choose_camera(beat_text, flow)
    count = shot_count(duration_sec, flow)
    parts = _fit_actions(_split_actions(beat_text), count)
    parts = [_dialogue_tags(_subjectize(x, subjects), subjects) for x in parts]
    ranges = timeline_ranges(duration_sec, flow)
    world_lock = _project_world_lock(project_idea)

    subject_lines=[]
    for sub in subjects:
        num=int(sub.get("subject_number") or 0)
        name=str(sub.get("name") or f"Subject {num}").strip()
        fn=str(sub.get("original_filename") or "reference image").strip()
        # The physical reference image is authoritative for identity.  Do NOT repeat the vision
        # caption here: those captions often describe a white/solid studio backdrop, pose or prop,
        # and Ref2VA may then reproduce that incidental reference-image scene.
        subject_lines.append(
            f"<Subject {num}> is {name}, defined by {fn}. Use the supplied reference image only as the authoritative "
            "character identity anchor: preserve face, hair, body proportions and characteristic wardrobe. "
            "Do not copy the reference image's studio setup, backdrop, framing, pose, props or lighting into the video."
        )
    subject_defs = "\n".join(subject_lines) if subject_lines else "No reference subject is required for this clip."

    timeline=[]
    for i,((a,b),action) in enumerate(zip(ranges,parts),start=1):
        lead = "Fast cut to " if i == 1 else "Cut on motivated movement to "
        timeline.append(
            f"[Shot {i}] At {_fmt_time(a)}, {lead}{action.rstrip('.')}."
        )
    detailed = "\n".join(timeline)

    action_note = ""
    if flow == "Dynamic action sequence":
        action_note = (
            " Choreograph action as clean readable exchanges with anticipation, contact, weight, blocks, dodges, counters, "
            "environment interaction and brief impact beats. Favor cinematic intensity over extreme subject speed; avoid motion blur."
        )
    summary = _subjectize(re.sub(r"\s+"," ",beat_text).strip(), subjects)
    retention = (
        "Preserve only the selected reference identities and the established project art direction. "
        "The reference images define the characters, not the location. Keep wardrobe, screen direction, environment geography and lighting logic coherent. "
        "Never inherit a plain white, solid-color, studio, cutout, portrait, product-photo or other reference-image backdrop. "
        "Do not import incidental reference props, poses or lighting unless this beat explicitly requires them."
    )
    sound = (
        "Native synchronized stereo diegetic sound appropriate to every visible action: footsteps, cloth movement, impacts, "
        "environmental ambience, machinery, weather and room tone. Spoken lines use the embedded dialogue tags and stay synchronized."
        if sound_enabled else "Complete silence."
    )
    if style in ("Cinematic realism","Live action","Action blockbuster","Neo-noir crime","Dystopian future"):
        music = "Cinematic electronic-orchestral score shaped to the beat; keep dialogue and physical impacts clear."
    elif style in ("3D animation","Cartoon"):
        music = "Expressive cinematic score matching the animated tone and emotional progression."
    elif style == "Anime":
        music = "Cinematic anime-style score with controlled rhythmic lift at action or reveal beats."
    else:
        music = "Cinematic score matching the selected visual style and current story beat."

    prompt = (
        "subject_definitions:\n" + subject_defs +
        "\nsummary:\n" + summary +
        "\nretention_analysis:\n" + retention +
        "\ndetailed_description:\n"
        + f"{float(duration_sec):g}-second {ratio} {resolution} clip. SELECTED VISUAL STYLE LOCK — {style}: {style_profile(style)} "
        + f"PROJECT WORLD LOCK — {world_lock} Every shot must exist inside this story world. The character reference image controls identity only; "
          "never use its white, plain, solid-color or studio backdrop as the scene. "
        + flow_profile(flow, duration_sec) + " " + camera_profile(camera, flow) + action_note + "\n" + detailed +
        "\noverall_soundscape:\n" + sound +
        "\nnon_diegetic_music:\n" + music
    )
    return {
        "prompt": prompt[:12000],
        "style": style,
        "flow": flow,
        "camera": camera,
        "duration_sec": float(duration_sec),
        "shot_count": count,
        "world_lock": world_lock,
    }
