# -*- coding: utf-8 -*-
from __future__ import annotations

"""Telegram Agent implementation extracted from llama_chat_ui.py.

This module intentionally contains the existing Telegram Agent methods without
rewriting their behavior. llama_chat_ui.py supplies the host globals used by
those methods through bind_host_globals(), keeping this extraction a move-and-
rewire refactor rather than a second implementation of the Agent.
"""


def bind_host_globals(host_globals):
    # Function globals are resolved dynamically from this module. Copying the
    # already-existing llama_chat_ui namespace lets moved methods keep using the
    # same helper classes/functions/constants they used before the extraction.
    # Keep this module's own import metadata intact.
    target = globals()
    for name, value in host_globals.items():
        if name.startswith("__") and name.endswith("__"):
            continue
        target[name] = value


class TelegramAgentMixin:
    def _telegram_planner_enhance_skill_candidates(self, limit: int = 8) -> List[Dict[str, Any]]:
        """Return only skills plausibly intended for Planner idea/story enhancement.

        Broad media/model tags alone are not enough. This deliberately excludes
        clip/multishot/music-clip skills that can destroy a Planner start prompt.
        """
        out = []
        for item in self._telegram_agent_skill_catalog():
            tags = set(self._normalize_pinned_tags(item.get('tags', [])))
            blob = ' '.join([
                str(item.get('name') or '').lower(),
                str(item.get('description') or '').lower(),
                ' '.join(tags),
            ])
            negative = any(x in blob for x in (
                'own storyline', 'multi prompt', 'multiprompt', 'multishot',
                'multi shot', 'clip prompt', 'music videoclip', 'music video clip',
                'music clip', 'lyrics writer'
            ))
            if negative:
                continue
            purpose = any(x in tags or x in blob for x in (
                'idea enhancer', 'story enhancer', 'planner input', 'planner prompt',
                'story prompt', 'story writer'
            ))
            planner_context = ('planner' in tags or 'planner' in blob)
            generic_enhancer = ('prompt enhancer' in tags or 'prompt enhancer' in blob)
            # Generic prompt enhancer is only considered when it is also clearly
            # marked for Planner/story/idea use. A MiniMax model tag alone is not enough.
            if purpose or (planner_context and generic_enhancer):
                out.append(dict(item))
        return out[:max(1, int(limit))]

    def _telegram_begin_planner_enhance_skill_choice(self, chat_id: str, state: Dict[str, Any]) -> bool:
        key = str(chat_id)
        candidates = self._telegram_planner_enhance_skill_candidates()
        if not candidates:
            self._telegram_send_text(
                key,
                'I found no pinned skill tagged for Planner idea/story enhancement, so I will not guess with an unrelated skill. '
                'Add tags such as `planner` + `story enhancer` (or `idea enhancer`) to a suitable pinned chat, '
                'or reply `use it` to keep the original idea.'
            )
            self._telegram_remote_wizards[key] = {'kind': 'planner', 'state': state}
            return True
        pending = {
            'chat_id': key,
            'text': 'Enhance the Planner story idea',
            'attachments': [],
            'purpose': 'planner_enhance',
            'planner_state': dict(state),
            'phase': 'skill_confirm',
            'skill_candidates': candidates,
            'skill_input': str(state.get('idea') or '').strip(),
            'skill_result': '',
        }
        self._telegram_agent_pending = pending
        lines = ['I found pinned skills that are tagged for Planner idea/story enhancement:', '']
        for i, item in enumerate(candidates, start=1):
            tags = ', '.join(item.get('tags') or []) or 'no tags'
            lines.append(f"{i}. {item.get('name') or 'Pinned skill'} [{tags}]")
        lines += ['', 'Type `no` if none should be used, the number(s) separated by commas, or `all`.']
        self._telegram_send_text(key, '\n'.join(lines))
        return True

    def _telegram_handle_remote_wizard(self, chat_id: str, text: str, attachments: list) -> bool:
        key = str(chat_id)
        remote = self._telegram_remote_wizards.get(key)
        if not isinstance(remote, dict):
            return False
        kind = str(remote.get("kind") or "")
        state = remote.get("state")
        if kind == "agent_idea":
            request = str(text or "").strip()
            if request.lower() in {"cancel", "/cancel", "stop", "nevermind", "never mind"}:
                self._telegram_remote_wizards.pop(key, None)
                self._telegram_send_text(key, "Autonomous Agent setup cancelled.")
                return True
            if not request:
                self._telegram_send_text(key, "Send me the video idea/request you want the Agent to handle.")
                return True
            self._telegram_remote_wizards.pop(key, None)
            self._telegram_autonomous_drop_placeholder(key)
            at = list((state or {}).get("attachments") or []) if isinstance(state, dict) else []
            at.extend(list(attachments or []))
            self._telegram_autonomous_video_start(key, request, at)
            return True

        if kind == "agent_route_choice":
            low = str(text or "").strip().lower()
            original = str((state or {}).get("request") or "") if isinstance(state, dict) else ""
            at = list((state or {}).get("attachments") or []) if isinstance(state, dict) else []
            if low in {"planner", "use planner", "the planner", "1"}:
                self._telegram_remote_wizards.pop(key, None)
                self._telegram_start_remote_wizard(key, "planner", "use the planner to create " + original, at)
                return True
            if low in {"agent", "use agent", "llm", "autonomous", "2", "super powers", "superpowers"}:
                self._telegram_remote_wizards.pop(key, None)
                self._telegram_autonomous_drop_placeholder(key)
                self._telegram_autonomous_video_start(key, original, at)
                return True
            self._telegram_send_text(key, "Choose `Planner` for the deterministic workflow or `Agent` to let the LLM plan/create/review/assemble the project itself.")
            return True
        # Planner enhancement is deterministic up to skill selection. Do not let
        # the LLM silently pick a vaguely-related pinned chat. Show only skills
        # explicitly tagged for Planner idea/story enhancement and let the user choose.
        if kind == "planner" and isinstance(state, dict):
            if "story_prompt_decision" not in state and re.search(r"\b(enhance|expand|improve|rewrite)\b", str(text or "").lower()):
                state['story_prompt_decision'] = 'enhance'
                state['story_original_idea'] = str(state.get('idea') or '').strip()
                state['story_enhancing'] = False
                state['story_enhanced_reviewed'] = False
                self._telegram_remote_wizards[key] = {'kind': 'planner', 'state': state}
                return self._telegram_begin_planner_enhance_skill_choice(key, state)
        if kind == "music":
            _, new_state, replies = self._telegram_capture_wizard_call(
                key, "_pending_ace15_music_request", state, self._handle_pending_ace15_music_message, text
            )
        elif kind == "planner":
            _, new_state, replies = self._telegram_capture_wizard_call(
                key, "_pending_planner_agent_request", state, self._planner_agent_handle_pending, text, attachments
            )
        elif kind == "music_clip":
            _, new_state, replies = self._telegram_capture_wizard_call(
                key, "_pending_music_clip_agent_request", state, self._music_clip_agent_handle_pending, text, attachments
            )
        else:
            self._telegram_remote_wizards.pop(key, None)
            return False
        self._telegram_wizard_send_replies(key, replies)
        if new_state is None:
            self._telegram_remote_wizards.pop(key, None)
            if kind == "music":
                try:
                    bridge = getattr(self, "_telegram_bridge", None)
                    if bridge is not None and bool(self.settings_dialog.chk_telegram_send_results.isChecked()):
                        bridge.watch_result(key, "ace_step_15", "music", "", time.time() - 3.0)
                except Exception:
                    pass
        else:
            self._telegram_remote_wizards[key] = {"kind": kind, "state": new_state}
        return True

    def _telegram_install_offer(self, chat_id: str, text: str, generation_request: bool = False) -> bool:
        key = str(chat_id)
        try:
            try:
                from helpers.fv_assistant_router import find_model_capability  # type: ignore
            except Exception:
                from fv_assistant_router import find_model_capability  # type: ignore
            report = dict(find_model_capability(text) or {})
        except Exception as exc:
            if not generation_request:
                self._telegram_send_text(key, f"I couldn't check Optional Downloads: {exc}")
                return True
            return False
        variants = list(report.get("optional_variants") or [])
        title = str(report.get("title") or "").strip()
        if bool(report.get("installed", False)) and title:
            if not generation_request:
                self._telegram_send_text(key, f"{title} already appears to be installed.")
                return True
            return False
        if not variants:
            if not generation_request:
                self._telegram_send_text(key, "I couldn't find a matching model in FrameVision Optional Downloads.")
                return True
            return False
        state = {"report": report, "variants": variants, "selected_key": ""}
        self._telegram_remote_installs[key] = state
        if len(variants) == 1:
            state["selected_key"] = str(variants[0].get("key") or "")
            self._telegram_send_text(key,
                f"That model is not installed. I found `{variants[0].get('title')}` in Optional Downloads. Install it? (yes/no)")
        else:
            names = "\n".join(f"{i+1}. {v.get('title')}" for i, v in enumerate(variants[:10]))
            self._telegram_send_text(key, "I found these Optional Downloads:\n" + names + "\n\nWhich one should I install?")
        return True

    def _telegram_handle_pending_install(self, chat_id: str, text: str) -> bool:
        key = str(chat_id)
        state = self._telegram_remote_installs.get(key)
        if not isinstance(state, dict):
            return False
        low = str(text or "").strip().lower()
        if low in {"no", "n", "cancel", "stop", "never mind", "nevermind"}:
            self._telegram_remote_installs.pop(key, None)
            self._telegram_send_text(key, "Okay, install cancelled.")
            return True
        variants = list(state.get("variants") or [])
        selected_key = str(state.get("selected_key") or "")
        if not selected_key:
            chosen = None
            try:
                idx = int(low) - 1
                if 0 <= idx < len(variants):
                    chosen = variants[idx]
            except Exception:
                pass
            if chosen is None:
                for v in variants:
                    if low and low in str(v.get("title") or "").lower():
                        chosen = v
                        break
            if chosen is None:
                self._telegram_send_text(key, "Choose an Optional Download by number/name, or say cancel.")
                return True
            state["selected_key"] = str(chosen.get("key") or "")
            self._telegram_remote_installs[key] = state
            self._telegram_send_text(key, f"Install `{chosen.get('title')}`? (yes/no)")
            return True
        if low not in {"yes", "y", "yes please", "confirm", "install it", "do it", "go ahead"}:
            self._telegram_send_text(key, "Say yes to start the install, or cancel.")
            return True
        try:
            try:
                from helpers.fv_assistant_router import launch_optional_install  # type: ignore
            except Exception:
                from fv_assistant_router import launch_optional_install  # type: ignore
            ok, msg = launch_optional_install(selected_key, parent=self)
        except Exception as exc:
            ok, msg = False, f"Could not start Optional Install: {exc}"
        self._telegram_remote_installs.pop(key, None)
        self._telegram_send_text(key, msg)
        return True

    def _telegram_agent_skill_catalog(self) -> List[Dict[str, Any]]:
        """Return compact Agent-enabled pinned-chat metadata for LLM skill selection."""
        try:
            self._load_pinned_chats()
        except Exception:
            pass
        items: List[Dict[str, Any]] = []
        for pinned in list(getattr(self, "pinned_chats", []) or []):
            if not bool(pinned.get("agent_enabled", True)):
                continue
            template = str(pinned.get("template") or "").strip()
            if not template:
                continue
            items.append({
                "id": str(pinned.get("id") or ""),
                "name": str(pinned.get("name") or "Pinned skill").strip(),
                "description": str(pinned.get("description") or "").strip()[:500],
                "tags": self._normalize_pinned_tags(pinned.get("tags", [])),
            })
        return items

    @staticmethod
    def _telegram_agent_norm_skill_ref(value: object) -> str:
        s = str(value or "").strip().strip("`'\"")
        s = re.sub(r"^skill\s*[:#-]?\s*", "", s, flags=re.I)
        s = s.replace("\\", "/").rsplit("/", 1)[-1]
        if s.lower().endswith(".json"):
            s = s[:-5]
        return re.sub(r"[^a-z0-9]+", " ", s.lower()).strip()

    def _telegram_agent_find_skill(self, skill_id: str) -> Optional[Dict[str, Any]]:
        """Resolve a pinned Agent skill robustly.

        Local LLMs sometimes return the displayed skill name, a quoted id, a
        filename-like id, or the numeric catalog position even when asked for
        the exact id. Resolve those harmless variants against the *current*
        pinned-chat catalog instead of falsely reporting that the skill vanished.
        """
        try:
            self._load_pinned_chats()
        except Exception:
            pass

        enabled = [
            p for p in list(getattr(self, "pinned_chats", []) or [])
            if bool(p.get("agent_enabled", True)) and str(p.get("template") or "").strip()
        ]
        if not enabled:
            return None

        raw = str(skill_id or "").strip().strip("`'\"")
        wanted = self._telegram_agent_norm_skill_ref(raw)

        # Exact current id/name first.
        for pinned in enabled:
            pid = str(pinned.get("id") or "").strip()
            name = str(pinned.get("name") or "").strip()
            if raw == pid or raw.casefold() == pid.casefold() or raw.casefold() == name.casefold():
                return pinned

        # Catalog number, e.g. "2" or "skill 2".
        m = re.fullmatch(r"(?:skill\s*)?#?\s*(\d+)", raw, flags=re.I)
        if m:
            idx = int(m.group(1)) - 1
            if 0 <= idx < len(enabled):
                return enabled[idx]

        # Normalized id/name/file-name forms.
        for pinned in enabled:
            pid_n = self._telegram_agent_norm_skill_ref(pinned.get("id"))
            name_n = self._telegram_agent_norm_skill_ref(pinned.get("name"))
            if wanted and wanted in {pid_n, name_n}:
                return pinned

        # Last safe fallback: unique strong textual match against name/tags/id.
        # Do not silently choose between ambiguous skills.
        if wanted:
            scored = []
            wanted_tokens = set(wanted.split())
            for pinned in enabled:
                hay = " ".join([
                    self._telegram_agent_norm_skill_ref(pinned.get("id")),
                    self._telegram_agent_norm_skill_ref(pinned.get("name")),
                    " ".join(self._normalize_pinned_tags(pinned.get("tags", []))),
                ])
                hay_tokens = set(self._telegram_agent_norm_skill_ref(hay).split())
                overlap = len(wanted_tokens & hay_tokens) / max(1, len(wanted_tokens))
                ratio = difflib.SequenceMatcher(None, wanted, self._telegram_agent_norm_skill_ref(pinned.get("name"))).ratio()
                score = max(overlap, ratio)
                if score >= 0.72:
                    scored.append((score, pinned))
            scored.sort(key=lambda x: x[0], reverse=True)
            if scored and (len(scored) == 1 or scored[0][0] - scored[1][0] >= 0.12):
                return scored[0][1]
        return None

    @staticmethod
    def _telegram_agent_parse_json(payload: object, expected_shot_count: Optional[int] = None) -> Dict[str, Any]:
        """Parse local-LLM Agent output defensively.

        Local GGUF models do not always obey "JSON only" perfectly.  Accept the
        common harmless variants (fences, leading prose, single-quoted Python
        dicts, or a JSON object nested under result/plan/decision) while still
        requiring a dictionary before execution.
        """
        if isinstance(payload, dict):
            raw = str(payload.get("content") or "").strip()
        else:
            raw = str(payload or "").strip()
        answer, _reasoning = _split_inline_reasoning(raw)
        raw = str(answer or raw).strip()

        # Strip common wrappers without assuming the model followed them exactly.
        raw = re.sub(r"^```(?:json|javascript|python)?\s*", "", raw, flags=re.I)
        raw = re.sub(r"\s*```$", "", raw).strip()
        raw = re.sub(r"^\s*(?:final answer|answer|response)\s*:\s*", "", raw, flags=re.I)

        candidates = [raw]

        # Recover a common no-grammar fallback failure where the model closes the
        # root JSON object too early and then continues emitting additional
        # top-level fields, e.g. {"title": ...} "story_arc": {...}, "shots": {...}.
        # Standard parsers accept only the first object, which makes a complete
        # blueprint look as if it contains zero shots.  Stitch only when the
        # suffix clearly starts with another quoted top-level key.
        def _premature_root_close_repair(text: str) -> str:
            in_string = False
            escaped = False
            depth = 0
            first_close = -1
            for idx, ch in enumerate(text):
                if in_string:
                    if escaped:
                        escaped = False
                    elif ch == "\\":
                        escaped = True
                    elif ch == '"':
                        in_string = False
                    continue
                if ch == '"':
                    in_string = True
                elif ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        first_close = idx
                        break
            if first_close < 0:
                return ""
            suffix = text[first_close + 1:].strip()
            if not suffix or not re.match(r'^,?\s*"[^"\n]+"\s*:', suffix):
                return ""
            suffix = re.sub(r'^,?\s*', '', suffix, count=1)
            # The final brace in the model output closes the last field value; add
            # one new brace for the repaired root object.
            return text[:first_close].rstrip() + ",\n" + suffix.rstrip() + "\n}"

        stitched = _premature_root_close_repair(raw)
        if stitched:
            candidates.insert(0, stitched)

        # Repair conservative JSON-ish formatting glitches produced by local LLMs
        # after grammar fallback.  This is intentionally syntax-only: it never
        # invents shots or semantic values. Examples seen in real Agent output:
        #   **Section**: "Chase Begins"
        #   **Section**: "Home Stretch**,
        def _jsonish_syntax_repair(text: str) -> str:
            fixed = str(text or "")
            # Markdown-bold bare object keys -> normal quoted JSON keys.
            def _bold_key(m):
                key = re.sub(r"\s+", "_", str(m.group(2) or "").strip()).lower()
                return f'{m.group(1)}"{key}":'
            fixed = re.sub(
                r'(?m)^(\s*)\*\*([A-Za-z_][A-Za-z0-9_ ]*)\*\*\s*:',
                _bold_key,
                fixed,
            )
            # A model sometimes leaves Markdown ** attached to a scalar and
            # forgets the closing JSON quote before the comma. Repair only the
            # unambiguous one-line form; otherwise leave it for normal rejection.
            fixed = re.sub(
                r'(?m)(:\s*)"([^"\n]*?)\*\*,\s*$',
                lambda m: m.group(1) + '"' + m.group(2).rstrip() + '",',
                fixed,
            )
            return fixed

        repaired_raw = _jsonish_syntax_repair(raw)
        if repaired_raw and repaired_raw != raw:
            repaired_stitched = _premature_root_close_repair(repaired_raw)
            if repaired_stitched:
                candidates.insert(0, repaired_stitched)
            candidates.insert(0, repaired_raw)

        # Try every balanced-looking outer object span, shortest useful suffix first.
        starts = [m.start() for m in re.finditer(r"\{", raw)]
        ends = [m.start() for m in re.finditer(r"\}", raw)]
        for a in starts:
            for b in reversed(ends):
                if b > a:
                    candidates.append(raw[a:b + 1])
                    break

        for candidate in candidates:
            candidate = str(candidate or "").strip()
            if not candidate:
                continue
            obj = None
            try:
                # Preserve duplicate JSON object keys long enough to repair a
                # common local-LLM blueprint failure. Qwen can occasionally emit
                # 1..10, then "1" for slot 11 (and similarly "2" for 22).
                # Normal json.loads silently overwrites the earlier keys, making
                # a complete 30-shot blueprint look as if slots 11/22 vanished.
                class _ObjectPairs(list):
                    pass

                parsed = json.loads(candidate, object_pairs_hook=_ObjectPairs)

                def _materialize(value, parent_key: str = ""):
                    if isinstance(value, _ObjectPairs):
                        pairs = list(value)
                        if parent_key == "shots" and expected_shot_count:
                            expected = [str(i) for i in range(1, int(expected_shot_count) + 1)]
                            got = [str(k) for k, _v in pairs]
                            # Only repair when the model supplied exactly the locked
                            # number of shot entries in order. This never invents or
                            # drops a beat; it only restores the deterministic slot IDs.
                            if len(pairs) == int(expected_shot_count) and got != expected:
                                return {
                                    str(i): _materialize(v, str(i))
                                    for i, (_k, v) in enumerate(pairs, start=1)
                                }
                        out = {}
                        for k, v in pairs:
                            out[str(k)] = _materialize(v, str(k))
                        return out
                    if isinstance(value, list):
                        return [_materialize(v, parent_key) for v in value]
                    return value

                obj = _materialize(parsed)
            except Exception:
                try:
                    # Qwen occasionally returns {'type': 'action', ...}.
                    obj = ast.literal_eval(candidate)
                except Exception:
                    obj = None
            if isinstance(obj, dict):
                # Some models wrap the actual decision/plan one level deep.
                # Router actions and autonomous production plans have different
                # schemas, so recognize both. Previously {"plan":{"clips":[...]}}
                # was left wrapped and the validator then saw zero clips.
                for key in ("decision", "result", "plan", "agent_plan", "output", "story_blueprint", "blueprint"):
                    nested = obj.get(key)
                    if not isinstance(nested, dict):
                        continue
                    looks_like_action = bool(
                        nested.get("type") or nested.get("action") or nested.get("skill_id")
                    )
                    looks_like_plan = bool(
                        isinstance(nested.get("clips"), list)
                        or isinstance(nested.get("sections"), list)
                        or isinstance(nested.get("narrative_sections"), list)
                        or nested.get("story_summary")
                        or nested.get("video_model")
                        or nested.get("music")
                        or nested.get("music_ace_step_1p5")
                        or nested.get("references")
                        or nested.get("references_krea_ref2va")
                        or nested.get("character_sheet_krea_2")
                    )
                    if looks_like_action or looks_like_plan:
                        return nested
                return obj
        return {}

    @staticmethod
    def _telegram_agent_action_alias(action: str) -> str:
        raw = re.sub(r"[^a-z0-9]+", "_", str(action or "").strip().lower()).strip("_")
        aliases = {
            "image": "create_image", "generate_image": "create_image", "image_generation": "create_image",
            "video": "create_video", "generate_video": "create_video", "video_generation": "create_video",
            "music": "create_music", "ace_step": "create_music", "ace_step_1_5": "create_music",
            "generate_music": "create_music", "music_generation": "create_music",
            "plan": "planner", "video_planner": "planner", "story_video": "planner",
            "long_video": "autonomous_video", "music_video": "autonomous_video", "create_story_video": "autonomous_video",
            "autonomous": "autonomous_video", "agent_video": "autonomous_video", "agent_music_video": "autonomous_video",
            "music_clip": "music_clip_creator", "clip_creator": "music_clip_creator",
            "install": "install_model", "download_model": "install_model",
            "last": "last_result", "lastresult": "last_result",
        }
        return aliases.get(raw, raw)

    def _telegram_agent_recover_decision(self, decision: Dict[str, Any], original_text: str) -> Dict[str, Any]:
        """Normalize slightly-off LLM output and safely recover obvious intents."""
        obj = dict(decision or {})

        # Accept a one-element actions list.  For combined long-video plans, Planner
        # is the single orchestrating action rather than separately running music/video.
        actions = obj.get("actions")
        if isinstance(actions, list) and actions:
            planner_item = next(
                (x for x in actions if isinstance(x, dict) and self._telegram_agent_action_alias(x.get("action", "")) == "planner"),
                None,
            )
            chosen = planner_item or next((x for x in actions if isinstance(x, dict)), None)
            if isinstance(chosen, dict):
                obj = dict(chosen)

        action = self._telegram_agent_action_alias(obj.get("action", ""))
        dtype = str(obj.get("type") or "").strip().lower()
        if action:
            obj["action"] = action
            if not dtype:
                obj["type"] = "action"
        elif obj.get("skill_id") and not dtype:
            obj["type"] = "skill"
        elif obj.get("message") and not dtype:
            obj["type"] = "clarify" if str(obj.get("message") or "").strip().endswith("?") else "reply"

        if str(obj.get("type") or "").strip().lower() in {"action", "skill", "clarify", "reply"}:
            return obj

        # Last-resort deterministic rescue.  This never invents shell/file actions;
        # it only selects one of FrameVision's already allow-listed actions.
        low = str(original_text or "").lower()
        duration_long = bool(re.search(r"\b(?:[1-9]\d*\s*(?:minute|min|minutes)|(?:[4-9]\d|[1-9]\d{2,})\s*(?:seconds|second|sec|s))\b", low))
        combined_music_video = (
            any(x in low for x in ("video", "story video", "music video"))
            and any(x in low for x in ("music", "ace step", "ace-step", "background music", "soundtrack"))
        )
        multi_scene = any(x in low for x in ("long video", "story video", "multiple clips", "multi-clip", "several scenes", "full video"))
        if combined_music_video or duration_long or multi_scene:
            if "planner" in low:
                return {"type": "action", "action": "planner", "command": str(original_text or "").strip(), "_recovered": True}
            return {"type": "action", "action": "autonomous_video", "command": str(original_text or "").strip(), "_recovered": True}
        if "music clip creator" in low:
            return {"type": "action", "action": "music_clip_creator", "command": str(original_text or "").strip(), "_recovered": True}
        if re.search(r"\b(?:create|make|generate)\b.*\b(?:music|song|track|soundtrack)\b", low):
            return {"type": "action", "action": "create_music", "command": str(original_text or "").strip(), "_recovered": True}
        if re.search(r"\b(?:create|make|generate)\b.*\b(?:image|picture|photo)\b", low):
            return {"type": "action", "action": "create_image", "command": str(original_text or "").strip(), "_recovered": True}
        if re.search(r"\b(?:create|make|generate)\b.*\bvideo\b", low):
            return {"type": "action", "action": "create_video", "command": str(original_text or "").strip(), "_recovered": True}
        return obj


    # ------------------------- Autonomous story/music-video Agent -------------------------
    def _telegram_autonomous_dir(self) -> Path:
        p = Path(self.fv_root) / "temp" / "telegram" / "agent_projects"
        p.mkdir(parents=True, exist_ok=True)
        return p

    def _telegram_autonomous_session_path(self, session_id: str) -> Path:
        safe = re.sub(r"[^0-9A-Za-z_-]+", "_", str(session_id or "agent"))
        return self._telegram_autonomous_dir() / f"{safe}.json"

    def _telegram_autonomous_save_planning_diagnostic(
        self,
        session: Dict[str, Any],
        phase: str,
        payload: object,
        parsed: Optional[Dict[str, Any]] = None,
        error: str = "",
    ) -> str:
        """Persist the exact local-LLM planning answer for debugging."""
        try:
            sid = re.sub(r"[^0-9A-Za-z_-]+", "_", str(session.get("id") or "agent"))
            phase_safe = re.sub(r"[^0-9A-Za-z_-]+", "_", str(phase or "plan"))
            if isinstance(payload, dict):
                raw = str(payload.get("content") or "")
                reasoning = str(payload.get("thinking") or "")
                if reasoning and reasoning not in raw:
                    raw = (
                        raw
                        + ("\n\n" if raw else "")
                        + "----- SEPARATE MODEL THINKING -----\n"
                        + reasoning
                    )
            else:
                raw = str(payload or "")
            path = self._telegram_autonomous_dir() / f"{sid}_{phase_safe}_raw.txt"
            path.write_text(raw, encoding="utf-8", errors="replace")

            info = {
                "phase": str(phase or ""),
                "raw_chars": len(raw),
                "autonomous_enable_thinking": False,
                "parsed_keys": sorted(str(k) for k in (parsed or {}).keys()) if isinstance(parsed, dict) else [],
                "clip_container_type": type((parsed or {}).get("clips")).__name__ if isinstance(parsed, dict) and "clips" in parsed else "",
                "clip_count": len((parsed or {}).get("clips") or []) if isinstance(parsed, dict) and isinstance((parsed or {}).get("clips"), list) else 0,
                "error": str(error or ""),
                "raw_file": str(path),
            }
            meta = self._telegram_autonomous_dir() / f"{sid}_{phase_safe}_diagnostic.json"
            meta.write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")
            return str(path)
        except Exception:
            return ""

    def _telegram_autonomous_save(self, session: Dict[str, Any]) -> None:
        try:
            sid = str(session.get("id") or "").strip()
            if not sid:
                return
            path = self._telegram_autonomous_session_path(sid)
            tmp = path.with_suffix(".json.tmp")
            tmp.write_text(json.dumps(session, ensure_ascii=False, indent=2), encoding="utf-8")
            os.replace(str(tmp), str(path))
        except Exception:
            pass

    def _telegram_autonomous_load_sessions(self) -> None:
        try:
            active_statuses = {"planning", "planning_retry", "building_story", "building_shots", "auditing_references", "music_preset_choice", "queueing", "queueing_media", "generating_refs", "generating", "review_ready", "assembling"}
            best: Dict[str, tuple[float, bool, Dict[str, Any]]] = {}
            for path in self._telegram_autonomous_dir().glob("*.json"):
                try:
                    data = json.loads(path.read_text(encoding="utf-8"))
                except Exception:
                    continue
                if not isinstance(data, dict):
                    continue
                chat = str(data.get("chat_id") or "")
                status = str(data.get("status") or "")
                if not chat or status not in active_statuses | {"done"}:
                    continue
                is_active = status in active_statuses
                try:
                    stamp = float(data.get("created_at") or data.get("finished_at") or 0.0)
                except Exception:
                    try:
                        stamp = float(path.stat().st_mtime)
                    except Exception:
                        stamp = 0.0
                prev = best.get(chat)
                # Active work always wins over a completed project. Otherwise keep
                # the newest project so a finished video can still be /redo'd after
                # a FrameVision restart.
                if prev is None or (is_active and not prev[1]) or (is_active == prev[1] and stamp >= prev[0]):
                    best[chat] = (stamp, is_active, data)
            for chat, (_stamp, _active, data) in best.items():
                self._telegram_autonomous_sessions[chat] = data
        except Exception:
            pass

    @staticmethod
    def _telegram_autonomous_resolution(model: str, resolution: str, aspect: str) -> tuple[int, int, str]:
        model = str(model or "minimax_h3").lower()
        key = str(resolution or "").lower().replace(" ", "")
        aspect = str(aspect or "16:9").lower().strip()
        # Agent may return exact dimensions.
        m = re.search(r"(\d{3,4})[x×](\d{3,4})", key)
        if m:
            return int(m.group(1)), int(m.group(2)), f"{m.group(2)}p"
        presets = {
            "480p": (832, 480),
            "544p": (960, 544),
            "704p": (1280, 704),
            "720p": (1280, 704),
            "768p": (1344, 768),
            "1080p": (1920, 1088),
            "1088p": (1920, 1088),
        }
        if key not in presets:
            key = "544p" if model == "minimax_h3" else "704p"
        w, h = presets[key]
        if aspect in {"9:16", "portrait", "vertical"}:
            w, h = h, w
            aspect = "9:16"
        elif aspect in {"1:1", "square"}:
            w = h = min(w, h)
            aspect = "1:1"
        else:
            aspect = "16:9"
        return int(w), int(h), key

    def _telegram_autonomous_duration_bounds(self, request: str) -> tuple[float, float]:
        """Return (minimum_required_seconds, maximum_allowed_seconds)."""
        low = str(request or "").lower()
        patterns = (
            r"\bbetween\s+(\d+(?:\.\d+)?)\s+(?:and|to|-)\s+(\d+(?:\.\d+)?)\s*(minutes?|mins?|min|seconds?|secs?|sec)\b",
            r"\b(\d+(?:\.\d+)?)\s*(?:-|–|—|to)\s*(\d+(?:\.\d+)?)\s*(minutes?|mins?|min|seconds?|secs?|sec)\b",
        )
        for pat in patterns:
            m = re.search(pat, low)
            if not m:
                continue
            lo = float(m.group(1))
            hi = float(m.group(2))
            if lo > hi:
                lo, hi = hi, lo
            unit = str(m.group(3) or "").lower()
            mult = 60.0 if unit.startswith(("min", "minute")) else 1.0
            lo = max(0.0, min(600.0, lo * mult))
            hi = max(lo, min(600.0, hi * mult))
            return lo, hi

        m = re.search(r"\b(\d+(?:\.\d+)?)\s*(minutes?|mins?|min)\b", low)
        if m:
            sec = max(0.0, min(600.0, float(m.group(1)) * 60.0))
            return sec, sec
        m = re.search(r"\b(\d+(?:\.\d+)?)\s*(seconds?|secs?|sec)\b", low)
        if m:
            sec = max(0.0, min(600.0, float(m.group(1))))
            return sec, sec
        return 0.0, 0.0

    def _telegram_autonomous_story_contract(self, request: str) -> Dict[str, float]:
        duration_min, duration_max = self._telegram_autonomous_duration_bounds(request)
        if duration_min <= 0.0 and duration_max <= 0.0:
            return {"duration_min": 0.0, "duration_max": 0.0, "story_basis": 0.0, "min_clips": 0}
        story_basis = duration_max if duration_max > 0.0 else duration_min
        return {
            "duration_min": float(duration_min),
            "duration_max": float(duration_max),
            "story_basis": float(story_basis),
            "min_clips": max(1, int(math.ceil(story_basis / 10.0))),
        }

    def _telegram_autonomous_story_contract_text(self, request: str) -> str:
        c = self._telegram_autonomous_story_contract(request)
        n = int(c.get("min_clips") or 0)
        if n <= 0:
            return ""
        dmin = float(c.get("duration_min") or 0.0)
        dmax = float(c.get("duration_max") or 0.0)
        basis = float(c.get("story_basis") or 0.0)
        duration_text = f"{dmin:.0f}-{dmax:.0f}s" if dmax > dmin else f"{basis:.0f}s"
        return (
            f"HARD FIRST-STORY CONTRACT: requested duration {duration_text}. "
            f"The FIRST plan must already contain the COMPLETE story and at least {n} ACTUAL generation clips. "
            "Most clips should be 5-10 seconds; 15 seconds is the absolute maximum. "
            "Do not create a short story with an ending and expect FrameVision to append extra story later. "
            "Build the full setup, development, escalation, climax and ONE ending now. "
            f"More than {n} clips is fine when the story benefits from it."
        )

    @staticmethod
    def _telegram_autonomous_clip_seconds(plan: Dict[str, Any]) -> float:
        total = 0.0
        for item in list((plan or {}).get("clips") or []):
            if not isinstance(item, dict):
                continue
            try:
                d = float(item.get("duration") or 6.0)
            except Exception:
                d = 6.0
            total += max(0.0, d)
        return total

    @staticmethod
    def _telegram_autonomous_duration_contract(request: str) -> Dict[str, float]:
        low = str(request or "").lower()
        dmin = dmax = target = 0.0
        range_patterns = (
            r"\bbetween\s+(\d+(?:\.\d+)?)\s+(?:and|to|-)\s+(\d+(?:\.\d+)?)\s*(minutes?|mins?|min|seconds?|secs?|sec)\b",
            r"\b(\d+(?:\.\d+)?)\s*(?:-|–|—|to)\s*(\d+(?:\.\d+)?)\s*(minutes?|mins?|min|seconds?|secs?|sec)\b",
        )
        for pat in range_patterns:
            m = re.search(pat, low)
            if m:
                lo, hi = float(m.group(1)), float(m.group(2))
                if lo > hi:
                    lo, hi = hi, lo
                unit = str(m.group(3) or "").lower()
                mult = 60.0 if unit.startswith(("min", "minute")) else 1.0
                dmin, dmax = lo * mult, hi * mult
                target = dmax
                break
        if target <= 0.0:
            m = re.search(r"\b(\d+(?:\.\d+)?)\s*(minutes?|mins?|min)\b", low)
            if m:
                target = dmin = dmax = float(m.group(1)) * 60.0
            else:
                m = re.search(r"\b(\d+(?:\.\d+)?)\s*(seconds?|secs?|sec)\b", low)
                if m:
                    target = dmin = dmax = float(m.group(1))
        if target <= 0.0:
            target = dmin = dmax = 60.0
        target = max(10.0, min(600.0, target))
        dmin = max(10.0, min(600.0, dmin or target))
        dmax = max(dmin, min(600.0, dmax or target))
        preferred_avg = 8.0
        shot_count = max(max(1, int(math.ceil(dmin / 10.0))), int(math.ceil(target / preferred_avg)))
        return {
            "min": float(dmin),
            "max": float(dmax),
            "target": float(target),
            "preferred_avg": preferred_avg,
            "shot_count": int(shot_count),
        }

    def _telegram_autonomous_blueprint_prompt(self, request: str) -> tuple[str, str]:
        c = self._telegram_autonomous_duration_contract(request)
        shot_count = int(c["shot_count"])
        system = (
            "You are the senior story director inside FrameVision. Your FIRST job is to design the COMPLETE narrative blueprint for the CURRENT user request before any story beats are written. "
            "This blueprint is not loose metadata and it is not a suggestion: the later 5-shot beat chunks will be forced to follow its section budgets exactly. "
            "Plan the whole runtime now so the story cannot finish halfway and then fill the remaining clips with repetition, scenery, or nonsense. "
            "Return a `story_sections` array covering the entire story from setup through development, escalation, climax and resolution. Every section needs a concrete narrative purpose, what must be achieved, a `clip_count`, and a role. "
            "The clip counts MUST add up to the exact locked slot count. For normal long stories, reserve the definitive resolution for the final 20 percent of slots and place the climax late enough that the middle still has room to develop. "
            "Use coherent narrative sections rather than mechanically creating a new section for every location change. Around 5-8 sections is often enough for a long story, but this is guidance, NOT a hard maximum; use more sections when the actual narrative genuinely needs them. Give important actions enough clips to breathe and escalate. "
            "Never invent filler sections such as extra establishing shots, repeated relaxing, repeated TV watching, repeated arrivals, or a lonely vehicle doing unrelated things just because slots remain. "
            "Define the title, one concise complete story summary, recurring continuity references, requested music, and the locked story section budget. "
            "Recurring visual subjects that need consistency belong in `references`; use semantic types such as character, creature, vehicle, object or location. You MUST include the main recurring character/creature AND every visually distinctive recurring vehicle, spaceship, car, important prop, or bounded set that appears repeatedly; do not omit the hero vehicle/ship just because the user only said character sheets. "
            "Do not create duplicate references for the same identity or generic one-off scenery. Preserve explicit model, resolution, aspect and music constraints. Return JSON only."
        )
        user = (
            "CURRENT USER REQUEST:\n" + str(request or "").strip()
            + f"\n\nHARD STORY BUDGET: exactly {shot_count} video slots. Design the COMPLETE blueprint now. "
              f"The sum of all story_sections.clip_count values MUST equal {shot_count}. "
              "The final resolution belongs at the end, not halfway through. Later stages will only fill these locked sections five beats at a time; they are not allowed to redesign your story.\n\n"
              "MANDATORY TOP-LEVEL JSON SHAPE (use these exact key names):\n"
              "{\n"
              "  \"title\": \"...\",\n"
              "  \"story_summary\": \"...\",\n"
              "  \"video_model\": \"minimax_h3\",\n"
              "  \"resolution\": \"...\",\n"
              "  \"aspect\": \"16:9\",\n"
              "  \"references\": [...],\n"
              "  \"story_sections\": [\n"
              "    {\"title\":\"...\",\"role\":\"setup\",\"purpose\":\"...\",\"must_achieve\":\"...\",\"clip_count\":1},\n"
              "    ...\n"
              "  ],\n"
              "  \"music\": {\"enabled\":true,\"instrumental\":true,\"genre\":\"...\",\"subgenre\":\"...\",\"caption\":\"...\",\"bpm\":0}\n"
              "}\n"
              "Do NOT replace story_sections with story_arc, acts, shots, outline, phases, or prose. The blueprint is invalid without the exact top-level story_sections array."
        )
        return system, user

    def _telegram_autonomous_blueprint_response_format(self, request: str) -> Dict[str, Any]:
        ref_item = {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "id": {"type": "string"},
                "name": {"type": "string"},
                "type": {"type": "string"},
                "prompt": {"type": "string"},
            },
            "required": ["id", "name", "type", "prompt"],
        }
        music_obj = {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "enabled": {"type": "boolean"},
                "instrumental": {"type": "boolean"},
                "genre": {"type": "string"},
                "subgenre": {"type": "string"},
                "caption": {"type": "string"},
                "bpm": {"type": "number"},
            },
            "required": ["enabled", "instrumental", "genre", "subgenre", "caption", "bpm"],
        }
        story_section = {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "title": {"type": "string"},
                "role": {"type": "string", "minLength": 1},
                "purpose": {"type": "string"},
                "must_achieve": {"type": "string"},
                "clip_count": {"type": "integer", "minimum": 1},
            },
            "required": ["title", "role", "purpose", "must_achieve", "clip_count"],
        }
        schema = {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "title": {"type": "string"},
                "story_summary": {"type": "string"},
                "video_model": {"type": "string"},
                "resolution": {"type": "string"},
                "aspect": {"type": "string"},
                "references": {"type": "array", "maxItems": 9, "items": ref_item},
                "story_sections": {"type": "array", "minItems": 1, "items": story_section},
                "music": music_obj,
            },
            "required": ["title", "story_summary", "video_model", "resolution", "aspect", "references", "story_sections", "music"],
        }
        return {"type": "json_schema", "json_schema": {"name": "framevision_blueprint_metadata", "strict": True, "schema": schema}}

    def _telegram_autonomous_normalize_metadata(self, obj: Dict[str, Any], request: str) -> Dict[str, Any]:
        p = dict(obj or {})
        low = str(request or "").lower()
        # Accept harmless aliases from no-grammar fallback.
        if not p.get("title"):
            p["title"] = p.get("story_title") or "Agent Video"
        if not p.get("story_summary"):
            p["story_summary"] = p.get("logline") or p.get("summary") or str(request or "")
        raw_refs = p.get("references")
        if raw_refs is None:
            raw_refs = p.get("characters_and_references")
        ref_items = []
        if isinstance(raw_refs, dict):
            for raw_id, raw_ref in raw_refs.items():
                if isinstance(raw_ref, dict):
                    item = dict(raw_ref); item.setdefault("id", str(raw_id)); ref_items.append(item)
        elif isinstance(raw_refs, list):
            ref_items = [dict(x) for x in raw_refs if isinstance(x, dict)]
        refs=[]; used=set()
        for n, ref in enumerate(ref_items[:9], 1):
            rid = re.sub(r"[^0-9A-Za-z_-]+", "_", str(ref.get("id") or f"ref_{n}")).strip("_") or f"ref_{n}"
            if rid in used: rid=f"{rid}_{n}"
            used.add(rid)
            prompt=str(ref.get("prompt") or ref.get("description") or ref.get("desc") or "").strip()
            if not prompt: continue
            refs.append({"id":rid,"name":str(ref.get("name") or rid).strip(),"type":str(ref.get("type") or "object").strip().lower(),"prompt":prompt,"clip_orders":[],"reference_role":""})
        p["references"]=refs
        if "minimax" in low: p["video_model"]="minimax_h3"
        elif "ltx 2.5" in low or "ltx2.5" in low or "ltx25" in low: p["video_model"]="ltx25"
        elif "ltx 2.3" in low or "ltx2.3" in low or "ltx23" in low: p["video_model"]="ltx23"
        else: p["video_model"]=str(p.get("video_model") or p.get("model_engine") or "minimax_h3")
        if re.search(r"\b960\s*[x×]\s*544\b", low): p["resolution"]="544p"
        else: p["resolution"]=str(p.get("resolution") or ("544p" if p["video_model"]=="minimax_h3" else "704p"))
        if any(x in low for x in ("portrait","vertical","9:16")): p["aspect"]="9:16"
        elif any(x in low for x in ("square","1:1")): p["aspect"]="1:1"
        else: p["aspect"]="16:9"
        raw_music=p.get("music")
        music=dict(raw_music) if isinstance(raw_music,dict) else {}
        if not music and isinstance(p.get("audio_design"),dict): music={}
        music.setdefault("enabled", "music" in low or "ace" in low)
        music.setdefault("instrumental", True)
        if re.search(r"\bdnb\b|drum\s*(?:and|&)\s*bass", low): music["genre"]="Drum & Bass (DNB)"
        else: music.setdefault("genre", str(music.get("style") or "Electronic"))
        music.setdefault("subgenre",""); music.setdefault("caption","instrumental soundtrack matching the story progression"); music.setdefault("bpm",0)
        p["music"]=music
        p["title"]=re.sub(r"[^0-9A-Za-z _-]+","",str(p.get("title") or "Agent Video")).strip()[:80] or "Agent Video"
        p["story_summary"]=str(p.get("story_summary") or request or "").strip()
        # Duration is a code-owned contract. Never let an LLM typo such as 20s
        # override an explicit 200s request later in the pipeline.
        dc = self._telegram_autonomous_duration_contract(request)
        p["target_duration"] = float(dc["target"])
        p["duration_total_sec"] = float(dc["target"])
        p["duration_min"] = float(dc["min"])
        p["duration_max"] = float(dc["max"])
        p["shot_count"] = int(dc["shot_count"])
        raw_sections = p.get("story_sections")
        # No-grammar retry compatibility: accept the same section objects only
        # when the model used a harmless container/key alias. We do not invent
        # sections or convert shot lists into a blueprint here.
        if raw_sections is None:
            for alias in ("sections", "story_phases", "narrative_sections"):
                candidate = p.get(alias)
                if isinstance(candidate, list):
                    raw_sections = candidate
                    break
        if raw_sections is None and isinstance(p.get("blueprint"), dict):
            nested = dict(p.get("blueprint") or {})
            raw_sections = nested.get("story_sections")
            if raw_sections is None:
                for alias in ("sections", "story_phases", "narrative_sections"):
                    candidate = nested.get(alias)
                    if isinstance(candidate, list):
                        raw_sections = candidate
                        break
        sections = []
        if isinstance(raw_sections, list):
            for item in raw_sections:
                if not isinstance(item, dict):
                    continue
                title = str(item.get("title") or item.get("name") or "").strip()
                role = str(item.get("role") or "development").strip().lower()
                purpose = str(item.get("purpose") or item.get("summary") or "").strip()
                must_achieve = str(item.get("must_achieve") or item.get("goal") or purpose).strip()
                try:
                    clip_count = int(item.get("clip_count") or item.get("shots") or item.get("count") or 0)
                except Exception:
                    clip_count = 0
                if title and purpose and clip_count > 0:
                    sections.append({"title": title, "role": role, "purpose": purpose, "must_achieve": must_achieve, "clip_count": clip_count})
        p["story_sections"] = sections
        return p

    def _telegram_autonomous_lock_blueprint_clip_budget(self, meta: Dict[str, Any], request: str) -> Dict[str, Any]:
        """Snap blueprint section arithmetic to FrameVision's locked total before it becomes authoritative.

        The LLM still designs the sections and their relative emphasis.  This function
        fixes only a small arithmetic miss in clip_count; it never asks the model to
        redesign/retry the blueprint just because its sum was off by a few clips.
        Once this returns, these counts ARE the locked blueprint and every later beat
        batch must obey them exactly.
        """
        p = dict(meta or {})
        sections = [dict(x) for x in list(p.get("story_sections") or []) if isinstance(x, dict)]
        if not sections:
            p["story_sections"] = sections
            return p
        total = int(self._telegram_autonomous_duration_contract(request)["shot_count"])
        counts = []
        for sec in sections:
            try:
                n = int(sec.get("clip_count") or 0)
            except Exception:
                n = 0
            counts.append(max(1, n))
        before = list(counts)
        delta = total - sum(counts)

        if delta < 0:
            # Too many clips: shorten the resolution first (moves the ending later,
            # never earlier), then trim the largest remaining sections while keeping
            # every narrative phase alive with at least one clip.
            need = -delta
            final_i = len(counts) - 1
            while need > 0 and counts[final_i] > 1:
                counts[final_i] -= 1
                need -= 1
            order = sorted(range(max(0, len(counts) - 1)), key=lambda i: (-counts[i], i))
            while need > 0:
                changed = False
                for i in order:
                    if need <= 0:
                        break
                    if counts[i] > 1:
                        counts[i] -= 1
                        need -= 1
                        changed = True
                if not changed:
                    break
        elif delta > 0:
            # Too few clips: put additional development time in the middle of the
            # story, not in the resolution. This preserves a late ending.
            if len(counts) <= 2:
                order = [0]
            else:
                order = list(range(1, len(counts) - 1)) or [0]
            pos = 0
            while delta > 0 and order:
                counts[order[pos % len(order)]] += 1
                delta -= 1
                pos += 1

        if sum(counts) != total:
            # This is only reachable for an impossible blueprint (more sections
            # than available clips). Let the normal structural validator explain it.
            p["story_sections"] = sections
            return p

        for sec, n in zip(sections, counts):
            sec["clip_count"] = int(n)
        p["story_sections"] = sections
        if counts != before:
            p["blueprint_budget_adjustment"] = {
                "from": before,
                "to": list(counts),
                "locked_total": total,
            }
        return p

    def _telegram_autonomous_validate_story_blueprint_meta(self, meta: Dict[str, Any], request: str) -> None:
        total = int(self._telegram_autonomous_duration_contract(request)["shot_count"])
        sections = [dict(x) for x in list((meta or {}).get("story_sections") or []) if isinstance(x, dict)]
        if not sections:
            raise ValueError("Blueprint has no locked story_sections.")
        if total >= 10 and len(sections) < 5:
            raise ValueError(f"Blueprint has only {len(sections)} story sections for {total} clips; long stories need setup, development, escalation, climax and resolution before beat generation starts.")
        # `role` is descriptive metadata, not a protocol enum.  Do not reject a
        # sound blueprint because the model calls a phase `climax_buildup`,
        # `pursuit`, `turning_point`, etc.  The hard gate below validates the
        # actual slot budget and pacing instead.
        roles = [str(x.get("role") or "").strip().lower() for x in sections]
        if any(not r for r in roles):
            raise ValueError("Every blueprint section needs a non-empty story role/label.")
        counts = []
        for sec in sections:
            try:
                n = int(sec.get("clip_count") or 0)
            except Exception:
                n = 0
            if n < 1:
                raise ValueError(f"Blueprint section `{sec.get('title','')}` has an invalid clip_count.")
            counts.append(n)
        if sum(counts) != total:
            raise ValueError(f"Blueprint clip budget is {sum(counts)} but FrameVision locked exactly {total}; redistribute the story before continuing.")
        if total >= 10:
            resolution_start = sum(counts[:-1]) + 1
            min_resolution_start = int(math.floor(total * 0.80)) + 1
            if resolution_start < min_resolution_start:
                raise ValueError(f"Resolution starts at clip {resolution_start}; for a {total}-clip story it must not begin before about clip {min_resolution_start}.")
            # Estimate the climax section without requiring one magic vocabulary.
            # Prefer an explicitly climax/finale/showdown-like label/title; when
            # the model uses completely custom names, use the penultimate
            # section as the pacing proxy.  This keeps the check focused on
            # *where* the late story peak happens instead of what it is called.
            climax_words = ("climax", "finale", "showdown", "confrontation", "peak")
            climax_idx = None
            for i, sec in enumerate(sections[:-1]):
                hay = (str(sec.get("role") or "") + " " + str(sec.get("title") or "")).lower()
                if any(word in hay for word in climax_words):
                    climax_idx = i
            if climax_idx is None:
                climax_idx = max(0, len(sections) - 2)
            climax_start = sum(counts[:climax_idx]) + 1
            min_climax_start = int(math.floor(total * 0.55)) + 1
            if climax_start < min_climax_start:
                raise ValueError(f"The late-story peak starts at clip {climax_start}; the blueprint burns through the story too quickly. Keep developing/escalating until about clip {min_climax_start} or later.")
        seen = set()
        for sec in sections:
            sig = re.sub(r"[^a-z0-9]+", " ", (str(sec.get("title") or "") + " " + str(sec.get("purpose") or "")).lower()).strip()
            if sig in seen:
                raise ValueError(f"Blueprint repeats the same narrative section: `{sec.get('title','')}`.")
            seen.add(sig)

    @staticmethod
    def _telegram_autonomous_blueprint_slot_targets(meta: Dict[str, Any], total: int) -> List[Dict[str, Any]]:
        targets = []
        cursor = 1
        for sec_index, sec in enumerate(list((meta or {}).get("story_sections") or []), 1):
            if not isinstance(sec, dict):
                continue
            count = max(0, int(sec.get("clip_count") or 0))
            for local_index in range(1, count + 1):
                if cursor > total:
                    break
                targets.append({
                    "slot": cursor,
                    "section_index": sec_index,
                    "section": str(sec.get("title") or f"Section {sec_index}"),
                    "role": str(sec.get("role") or "development"),
                    "purpose": str(sec.get("purpose") or ""),
                    "must_achieve": str(sec.get("must_achieve") or ""),
                    "section_beat": f"{local_index}/{count}",
                })
                cursor += 1
        return targets[:total]

    @staticmethod
    def _telegram_autonomous_story_text_signature(text: str) -> set:
        stop = {"the","a","an","and","or","to","of","in","on","at","with","as","for","from","into","his","her","their","he","she","they","it","is","are","was","were","then","while","this","that"}
        return {w for w in re.findall(r"[a-z0-9]+", str(text or "").lower()) if len(w) > 2 and w not in stop}

    def _telegram_autonomous_story_quality_issues(self, story: List[Dict[str, Any]], meta: Dict[str, Any], total: int) -> List[str]:
        issues = []
        beats = [str(x.get("beat") or "").strip() for x in story]
        sigs = [self._telegram_autonomous_story_text_signature(x) for x in beats]
        for i in range(len(beats)):
            for j in range(i + 1, len(beats)):
                if not sigs[i] or not sigs[j]:
                    continue
                inter = len(sigs[i] & sigs[j]); union = len(sigs[i] | sigs[j]) or 1
                jac = inter / union
                if inter >= 5 and jac >= 0.72:
                    issues.append(f"clips {i+1} and {j+1} are near-duplicate story events")
                    if len(issues) >= 4:
                        return issues
        sections = [dict(x) for x in list((meta or {}).get("story_sections") or []) if isinstance(x, dict)]
        if sections and total >= 10:
            final = sections[-1]
            final_sig = self._telegram_autonomous_story_text_signature(str(final.get("purpose") or "") + " " + str(final.get("must_achieve") or ""))
            resolution_start = sum(int(x.get("clip_count") or 0) for x in sections[:-1]) + 1
            if len(final_sig) >= 4:
                for idx, sig in enumerate(sigs[:max(0, resolution_start - 1)], 1):
                    common = len(sig & final_sig)
                    overlap = common / max(1, min(len(sig), len(final_sig)))
                    if common >= 4 and overlap >= 0.55:
                        issues.append(f"clip {idx} substantially performs the locked final resolution before resolution starts at clip {resolution_start}")
                        break
        return issues

    def _telegram_autonomous_beat_batch_prompt(self, session: Dict[str, Any]) -> tuple[str, str]:
        request = str(session.get("request") or "")
        c = self._telegram_autonomous_duration_contract(request)
        total = int(c["shot_count"])
        idx = int(session.get("beat_batch_index") or 0)
        start = idx * 5 + 1
        end = min(total, start + 4)
        count = max(0, end - start + 1)
        meta = dict(session.get("blueprint_meta") or {})
        previous = list(session.get("story_slots") or [])
        all_targets = self._telegram_autonomous_blueprint_slot_targets(meta, total)
        targets = [x for x in all_targets if start <= int(x.get("slot") or 0) <= end]
        next_targets = [x for x in all_targets if end < int(x.get("slot") or 0) <= min(total, end + 3)]
        if len(targets) != count:
            raise ValueError(f"Locked blueprint does not provide section targets for story slots {start}-{end}.")
        system = (
            "You are the story-beat writer inside FrameVision. The COMPLETE story architecture and clip budget were already approved by the senior story director. "
            f"Return exactly {count} beat objects in a JSON array named `beats`, in chronological order, for internal slots {start} through {end}. "
            "You are NOT allowed to redesign the story, rush ahead into a later section, finish the story early, add filler, or repeat an earlier event. "
            "Each supplied slot target tells you the exact locked section, its narrative purpose, and the local position inside that section. Write one concrete new story event for each target and make meaningful progression inside the section. "
            "If a section has several clips, use those clips to develop/escalate the SAME narrative phase through distinct actions or reactions; do not simply rephrase the same event and do not waste one clip per scenery change. "
            "The definitive ending may occur only inside the locked resolution section. DO NOT write slot numbers, model names, or generation instructions in your answer. "
            "For each beat, also return `reference_ids`: the stable IDs of recurring visual subjects that are actually present in that beat. "
            "Choose only IDs from the supplied LOCKED REFERENCE CATALOG. Identity assignment is semantic, not word matching: if the beat says `she`, `he`, `the dog`, etc., keep the correct recurring subject ID from story context. "
            "Do not invent, rename, replace, or omit an ID merely because the beat uses a pronoun. Return JSON only."
            " Each object must contain only beat (the concrete event) and reference_ids (an array of catalog IDs). "
            "FrameVision assigns each object's section from its ordered slot. Do not return section, title, beat_id, action_type, or duration_seconds."
        )
        user = (
            "CURRENT USER REQUEST:\n" + request
            + "\n\nLOCKED COMPLETE STORY SUMMARY:\n" + str(meta.get("story_summary") or request)
            + "\n\nLOCKED REFERENCE CATALOG (IDs are immutable; select from these only):\n" + json.dumps(meta.get("references") or [], ensure_ascii=False, indent=2)
            + "\n\nLOCKED FULL STORY BLUEPRINT (do not change it):\n" + json.dumps(meta.get("story_sections") or [], ensure_ascii=False, indent=2)
            + "\n\nEXACT SLOT TARGETS FOR THIS 5-SHOT CHUNK:\n" + json.dumps(targets, ensure_ascii=False, indent=2)
            + ("\n\nNEXT LOCKED TARGETS FOR CONTEXT ONLY — do not perform them yet:\n" + json.dumps(next_targets, ensure_ascii=False, indent=2) if next_targets else "")
            + ("\n\nALL PREVIOUS LOCKED STORY BEATS (continue after them; never retell them):\n" + json.dumps(previous, ensure_ascii=False, indent=2) if previous else "")
            + ("\n\nPREVIOUS QUALITY-CHECK FEEDBACK FROM A REJECTED PASS:\n" + str(session.get("story_quality_feedback") or "") if session.get("story_quality_feedback") else "")
            + ("\n\nREPAIR THIS EXACT OUTPUT ERROR IN THE CURRENT CHUNK:\n" + str(session.get("beat_batch_validation_feedback") or "") if session.get("beat_batch_validation_feedback") else "")
            + f"\n\nCreate exactly {count} NEW chronological beats for the supplied targets. Return only beat and reference_ids for each object; FrameVision owns section assignments and timing."
        )
        return system, user

    def _telegram_autonomous_beat_response_format(self, session: Dict[str, Any]) -> Dict[str, Any]:
        total=int(self._telegram_autonomous_duration_contract(str(session.get("request") or ""))["shot_count"])
        idx=int(session.get("beat_batch_index") or 0); start=idx*5+1; count=max(0,min(5,total-start+1))
        valid_ids = [str(x.get("id") or "") for x in list((session.get("blueprint_meta") or {}).get("references") or []) if isinstance(x, dict) and str(x.get("id") or "").strip()]
        ref_item = {"type": "string"}
        if valid_ids:
            ref_item["enum"] = valid_ids
        beat={"type":"object","additionalProperties":False,"properties":{"beat":{"type":"string"},"reference_ids":{"type":"array","items":ref_item,"uniqueItems":True}},"required":["beat","reference_ids"]}
        schema={"type":"object","additionalProperties":False,"properties":{"beats":{"type":"array","minItems":count,"maxItems":count,"items":beat}},"required":["beats"]}
        return {"type":"json_schema","json_schema":{"name":"framevision_story_beat_batch","strict":True,"schema":schema}}

    def _telegram_autonomous_parse_beat_batch(self, obj: Dict[str, Any], session: Dict[str, Any]) -> list:
        request=str(session.get("request") or "")
        total=int(self._telegram_autonomous_duration_contract(request)["shot_count"])
        idx=int(session.get("beat_batch_index") or 0); start=idx*5+1; end=min(total,start+4); count=end-start+1
        raw=(obj or {}).get("beats")
        if not isinstance(raw,list) or len(raw)!=count:
            raise ValueError(f"Story beat batch returned {len(raw) if isinstance(raw,list) else 0} beats; required exactly {count}.")
        valid={str(x.get("id")) for x in list((session.get("blueprint_meta") or {}).get("references") or []) if isinstance(x,dict)}
        targets = self._telegram_autonomous_blueprint_slot_targets(dict(session.get("blueprint_meta") or {}), total)
        targets_by_slot = {int(x.get("slot") or 0): x for x in targets}
        out=[]
        for pos,item in enumerate(raw):
            if not isinstance(item,dict): raise ValueError("Story beat batch contains a non-object item.")
            low={str(k).strip().lower().replace("**",""):v for k,v in item.items()}
            # A beat title names an event, not its parent blueprint section.
            # Array length/order is already checked; recover an omitted section
            # from the corresponding locked slot, never from a creative title.
            section=str(low.get("section") or low.get("section_name") or "").strip()
            beat=str(low.get("beat") or low.get("description") or low.get("action") or "").strip()
            expected_target = targets_by_slot.get(start + pos)
            expected_section = str((expected_target or {}).get("section") or "").strip()
            if not expected_section:
                raise ValueError(f"Story beat {start+pos} has no locked blueprint slot target.")
            if not beat: raise ValueError(f"Story beat {start+pos} is missing beat text.")
            if section and section.casefold() != expected_section.casefold():
                raise ValueError(f"Story beat {start+pos} escaped its locked blueprint section: got `{section}`, expected `{expected_section}`.")
            section = expected_section
            raw_refs = low.get("reference_ids") or low.get("references") or []
            if isinstance(raw_refs, str):
                raw_refs = [raw_refs]
            refs = []
            for rid in list(raw_refs):
                rid = re.sub(r"[^0-9A-Za-z_-]+", "_", str(rid or "")).strip("_")
                if rid and rid in valid and rid not in refs:
                    refs.append(rid)
            out.append({"slot":start+pos,"section":section,"beat":beat,"reference_ids":refs})
        return out

    def _telegram_autonomous_validate_blueprint(self, obj: Dict[str, Any], request: str) -> Dict[str, Any]:
        p = dict(obj or {})
        c = self._telegram_autonomous_duration_contract(request)
        required = int(c["shot_count"])

        shots_obj = p.get("shots") if isinstance(p.get("shots"), dict) else {}
        expected_keys = [str(i) for i in range(1, required + 1)]
        actual_keys = list(shots_obj.keys())
        if actual_keys != expected_keys:
            raise ValueError(
                f"BLUEPRINT_SHOT_KEYS|{actual_keys}|{expected_keys}|The blueprint must contain every locked shot key exactly once."
            )

        flat_slots = []
        sections = []
        by_section: Dict[str, Dict[str, Any]] = {}
        section_order = []
        for expected_slot in range(1, required + 1):
            shot = dict(shots_obj[str(expected_slot)])

            # Normalize harmless schema drift from no-grammar local-LLM output.
            # Validation remains strict about the existence of all locked shots
            # and their actual story content; only equivalent field names/casing
            # are accepted here.
            lower_fields = {str(k).strip().lower(): v for k, v in shot.items()}
            beat = str(
                lower_fields.get("beat")
                or lower_fields.get("description")
                or lower_fields.get("action")
                or ""
            ).strip()
            if not beat:
                raise ValueError(f"Blueprint shot slot {expected_slot} has no story beat.")
            section_name = str(
                lower_fields.get("section")
                or lower_fields.get("section_name")
                or lower_fields.get("title")
                or ""
            ).strip()
            if not section_name:
                raise ValueError(f"Blueprint shot slot {expected_slot} has no section name.")
            refs = []
            raw_reference_ids = lower_fields.get("reference_ids") or lower_fields.get("references") or []
            if isinstance(raw_reference_ids, str):
                raw_reference_ids = [raw_reference_ids]
            for rid in list(raw_reference_ids):
                rid = re.sub(r"[^0-9A-Za-z_-]+", "_", str(rid)).strip("_")
                if rid and rid not in refs:
                    refs.append(rid)
            sid = re.sub(r"[^0-9A-Za-z_-]+", "_", section_name).strip("_").lower() or f"section_{expected_slot}"
            item = {
                "slot": expected_slot,
                "beat": beat,
                "section_id": sid,
                "section_title": section_name,
                "reference_ids": refs,
            }
            flat_slots.append(item)
            if sid not in by_section:
                by_section[sid] = {
                    "id": sid,
                    "title": section_name,
                    "reference_ids": [],
                    "shots": [],
                    "section_index": len(section_order) + 1,
                }
                section_order.append(sid)
            sec = by_section[sid]
            sec["shots"].append(dict(item))
            for rid in refs:
                if rid not in sec["reference_ids"]:
                    sec["reference_ids"].append(rid)

        for sid in section_order:
            sec = by_section[sid]
            sec["shot_count"] = len(sec["shots"])
            sections.append(sec)

        p["sections"] = sections
        p["shot_slots"] = flat_slots
        p["target_duration"] = float(c["target"])
        p["duration_min"] = float(c["min"])
        p["duration_max"] = float(c["max"])
        p["shot_count"] = required

        low = str(request or "").lower()
        if "minimax" in low:
            p["video_model"] = "minimax_h3"
        elif "ltx 2.5" in low or "ltx2.5" in low or "ltx25" in low:
            p["video_model"] = "ltx25"
        elif "ltx 2.3" in low or "ltx2.3" in low or "ltx23" in low:
            p["video_model"] = "ltx23"
        else:
            p["video_model"] = str(p.get("video_model") or "minimax_h3")

        if re.search(r"\b960\s*[x×]\s*544\b", low):
            p["resolution"] = "544p"
        else:
            p["resolution"] = str(p.get("resolution") or ("544p" if p["video_model"] == "minimax_h3" else "704p"))
        if any(x in low for x in ("portrait", "vertical", "9:16")):
            p["aspect"] = "9:16"
        elif any(x in low for x in ("square", "1:1")):
            p["aspect"] = "1:1"
        else:
            p["aspect"] = "16:9"

        # Accept both the strict reference array and the harmless object form
        # sometimes returned after the no-grammar fallback. Preserve the object's
        # semantic key as the reference id and accept `description` as `prompt`.
        raw_refs = p.get("references")
        ref_items = []
        if isinstance(raw_refs, dict):
            for raw_id, raw_ref in raw_refs.items():
                if isinstance(raw_ref, dict):
                    item = dict(raw_ref)
                    item.setdefault("id", str(raw_id))
                    ref_items.append(item)
        elif isinstance(raw_refs, list):
            ref_items = [dict(x) for x in raw_refs if isinstance(x, dict)]

        refs = []
        used = set()
        for n, ref in enumerate(ref_items[:9], start=1):
            rid = re.sub(r"[^0-9A-Za-z_-]+", "_", str(ref.get("id") or f"ref_{n}")).strip("_") or f"ref_{n}"
            if rid in used:
                rid = f"{rid}_{n}"
            used.add(rid)
            prompt = str(ref.get("prompt") or ref.get("description") or ref.get("desc") or "").strip()
            if not prompt:
                raise ValueError(f"Reference {rid} has no reference prompt.")
            refs.append({
                "id": rid,
                "name": str(ref.get("name") or rid).strip(),
                "type": str(ref.get("type") or "object").strip().lower(),
                "prompt": prompt,
                "clip_orders": [],
                "reference_role": "",
            })
        p["references"] = refs

        valid_ref_ids = {x["id"] for x in refs}

        # If the fallback response omitted per-shot reference_ids, infer only
        # obvious matches from the locked beat/section text. This keeps requested
        # Krea/Ref2VA identities attached instead of silently generating them and
        # then never using them.
        def _ref_match_tokens(ref):
            blob = " ".join([str(ref.get("id") or ""), str(ref.get("name") or "")]).lower()
            stop = {"the", "ref", "reference", "car", "cars", "road", "roads", "shop", "home", "black", "luxury", "vehicle", "location"}
            toks = [t for t in re.findall(r"[a-z0-9]+", blob) if len(t) >= 3 and t not in stop]
            return list(dict.fromkeys(toks))

        for slot in flat_slots:
            if slot.get("reference_ids"):
                continue
            hay = (str(slot.get("section_title") or "") + " " + str(slot.get("beat") or "")).lower()
            inferred = []
            for ref in refs:
                toks = _ref_match_tokens(ref)
                # Require a meaningful identity/name token hit. Short generic
                # words are excluded above to avoid attaching every ref to every shot.
                if toks and any(re.search(r"\b" + re.escape(t) + r"\b", hay) for t in toks):
                    inferred.append(str(ref.get("id") or ""))
            slot["reference_ids"] = [rid for rid in inferred if rid in valid_ref_ids]

        for slot in flat_slots:
            slot["reference_ids"] = [rid for rid in slot.get("reference_ids", []) if rid in valid_ref_ids]

        # Give every continuity reference an execution role. Characters need an
        # identity sheet; recurring vehicles/props and bounded sets can use a
        # reusable subject/set board; generic roads/skies/city environments remain
        # prompt-only context and must not consume Krea jobs by default.
        ref_use_counts = {rid: 0 for rid in valid_ref_ids}
        for slot in flat_slots:
            for rid in slot.get("reference_ids", []):
                if rid in ref_use_counts:
                    ref_use_counts[rid] += 1
        transient_location_words = {
            "road", "roads", "street", "streets", "highway", "interstate",
            "expressway", "boulevard", "coast", "coastal", "coastline", "ocean",
            "city", "downtown", "metropolis", "sky", "traffic", "landscape",
            "mountains", "beach", "forest", "desert"
        }
        bounded_set_words = {
            "mansion", "house", "home", "villa", "showroom", "shop", "garage",
            "room", "interior", "office", "studio", "stage", "castle", "apartment",
            "warehouse", "bar", "club", "restaurant", "station", "school"
        }
        for ref in refs:
            rtype = str(ref.get("type") or "object").strip().lower()
            blob = " ".join([str(ref.get("id") or ""), str(ref.get("name") or ""), str(ref.get("prompt") or "")]).lower()
            words = set(re.findall(r"[a-z0-9]+", blob))
            if rtype in {"character", "person", "creature", "alien", "animal"}:
                role = "character_identity"
            elif rtype in {"vehicle", "car", "ship", "spaceship", "aircraft", "bike", "object", "prop", "item", "device", "artifact"}:
                # A reference definition is not automatically worth a Krea job.
                # Only recurring non-character subjects become reusable boards;
                # one-off coffee cups/phones/etc. remain prompt-only.
                role = "reusable_asset" if ref_use_counts.get(str(ref.get("id") or ""), 0) >= 2 else "transient_environment"
            elif rtype in {"location", "scene", "environment", "place", "setting", "background"}:
                bounded = bool(words & bounded_set_words)
                transient = bool(words & transient_location_words) and not bounded
                # A bounded location only earns a set-reference job when it is
                # actually reused. One-off scenery remains prompt-only.
                role = "reusable_set" if bounded and ref_use_counts.get(str(ref.get("id") or ""), 0) >= 2 else "transient_environment"
                if transient:
                    role = "transient_environment"
            else:
                role = "reusable_asset" if ref_use_counts.get(str(ref.get("id") or ""), 0) >= 2 else "transient_environment"
            ref["reference_role"] = role
            ref["clip_orders"] = [
                int(slot.get("slot") or 0) for slot in flat_slots
                if str(ref.get("id") or "") in slot.get("reference_ids", [])
            ]

        # Propagate repaired/inferred references back into the section copies
        # used by the shot-prompt batching stage.
        refs_by_slot = {int(x.get("slot") or 0): list(x.get("reference_ids") or []) for x in flat_slots}
        for sec in sections:
            sec_refs = []
            for shot in sec.get("shots", []):
                slot_no = int(shot.get("slot") or 0)
                shot["reference_ids"] = [rid for rid in refs_by_slot.get(slot_no, shot.get("reference_ids", [])) if rid in valid_ref_ids]
                for rid in shot["reference_ids"]:
                    if rid not in sec_refs:
                        sec_refs.append(rid)
            sec["reference_ids"] = sec_refs

        raw_music = p.get("music")
        music = dict(raw_music) if isinstance(raw_music, dict) else {}
        if "ace" in low or "music" in low or p.get("music_model"):
            music["enabled"] = True
        music.setdefault("enabled", True)
        music.setdefault("instrumental", True)
        # The user's explicit request outranks schema-drift fields such as a
        # top-level creative-video `genre`.  This prevents a request for DNB from
        # becoming "Cinematic Automotive Narrative" music.
        if re.search(r"\bd\s*[&n]?\s*b\b|\bdnb\b|drum\s*(?:and|&)\s*bass", low):
            music["genre"] = "Drum & Bass (DNB)"
        elif not str(music.get("genre") or "").strip():
            music["genre"] = str(music.get("style") or p.get("music_genre") or p.get("genre") or "Electronic").strip() or "Electronic"
        music.setdefault("subgenre", "")
        music.setdefault("caption", "instrumental soundtrack matching the locked story progression")
        music.setdefault("bpm", 0)
        if p.get("music_model"):
            music.setdefault("model", str(p.get("music_model") or "").strip())
        p["music"] = music
        p["title"] = re.sub(r"[^0-9A-Za-z _-]+", "", str(p.get("title") or "Agent Video")).strip()[:80] or "Agent Video"
        p["story_summary"] = str(p.get("story_summary") or request or "").strip()
        return p

    def _telegram_autonomous_build_batches(self, blueprint: Dict[str, Any]) -> list:
        # Prompt generation must preserve chronological slot order. Grouping by
        # repeated section titles can create batches such as [1, 11], which puts
        # numeric identity back in the LLM's hands and reintroduces duplicate-key
        # failures. FrameVision therefore batches only consecutive locked slots.
        slots = [dict(x) for x in list(blueprint.get("shot_slots") or []) if isinstance(x, dict)]
        slots.sort(key=lambda x: int(x.get("slot") or 0))
        batches = []
        batch_size = 4
        for pos in range(0, len(slots), batch_size):
            batch_slots = slots[pos:pos + batch_size]
            if not batch_slots:
                continue
            a = int(batch_slots[0].get("slot") or 1)
            b = int(batch_slots[-1].get("slot") or a)
            batches.append({
                "section_index": len(batches) + 1,
                "section_id": f"slots_{a}_{b}",
                "section_title": f"Shots {a}-{b}",
                "section_reference_ids": [],
                "slots": batch_slots,
                "count": len(batch_slots),
                "global_start": a,
                "global_end": b,
            })
        return batches

    def _telegram_autonomous_shot_batch_prompt(self, session: Dict[str, Any], batch: Dict[str, Any]) -> tuple[str, str]:
        bp = dict(session.get("blueprint") or {})
        slots = [dict(x) for x in list(batch.get("slots") or []) if isinstance(x, dict)]
        expected_ids = [int(x.get("slot") or 0) for x in slots]
        prev_clip = ""
        clips_done = list(session.get("staged_clips") or [])
        if clips_done:
            prev_clip = str(clips_done[-1].get("prompt") or "")
        refs = {str(x.get("id")): x for x in list(bp.get("references") or []) if isinstance(x, dict)}
        relevant_ids = []
        for slot in slots:
            for rid in list(slot.get("reference_ids") or []):
                if rid not in relevant_ids:
                    relevant_ids.append(rid)
        relevant_refs = [refs[x] for x in relevant_ids if x in refs]
        for slot in slots:
            order = int(slot.get("slot") or 0)
            slot["visual_references"] = [{
                "id": refs[rid]["id"], "name": refs[rid].get("name"),
                "appearance": refs[rid].get("prompt"), "narrative_role": refs[rid].get("narrative_role", ""),
                "relationships_in_this_shot": [e.get("relationship") for e in refs[rid].get("appearance_evidence", []) if e.get("slot") == order],
            } for rid in slot.get("reference_ids", []) if rid in refs]
        system = (
            "You are the prompt-writing stage of FrameVision. The story and shot count are already LOCKED. "
            f"You are receiving exactly {len(slots)} consecutive locked shot beats for internal slots {expected_ids}. "
            f"Return exactly {len(slots)} prompt objects in chronological order in a JSON array named `clips`. "
            "DO NOT write slot numbers or numeric object keys. FrameVision owns numbering. "
            "Do not merge, skip, add, split, renumber or reinterpret beats. Your only job is to translate EACH supplied beat into one detailed text-to-video generation prompt. "
            "Every prompt must explicitly describe the environment/background, subject appearance/action, continuity, camera framing/movement and lighting/atmosphere. "
            "Write each prompt as a compact generation-ready shot description, normally 90-160 English words. Do not add analysis, alternate versions, or explanations. "
            "Reference images control recurring identity only; their reference-sheet background is NOT the scene background. "
            "The supplied slot reference_ids are already authoritative; do not infer or rewrite them. Return JSON only."
            " Exact output shape: {\"clips\":[{\"purpose\":\"the locked beat\",\"prompt\":\"the complete shot description\"}]}. "
            "Put all environment, action, continuity, camera and lighting detail inside prompt, not separate properties."
            " Each slot is ONE continuous take, with one viewpoint and a physically continuous camera path. "
            "Do not describe a cut, montage, alternating viewpoints, dissolve, or an instant jump to another camera position inside a clip. "
            "A change of viewpoint belongs between separately generated clips. If a beat describes several views, choose one that conveys its main event. "
            "Write only the shot's descriptive body, without Shot labels or six-section headers; FrameVision wraps it. "
            "Use only the current slot's visual_references. References in other slots do not establish presence here. "
            "Appearance does not imply ownership, agency or participation; follow this slot's relationships and locked event."
        )
        user = (
            "LOCKED COMPLETE STORY SUMMARY:\n" + str(bp.get("story_summary") or "")
            + "\n\nLOCKED SHOT SLOTS TO TRANSLATE:\n" + json.dumps(slots, ensure_ascii=False, indent=2)
            + "\n\nReferences and relationships are scoped inside each slot. Do not borrow participants from adjacent slots."
            + f"\n\nReturn exactly {len(slots)} ordered clip prompt objects. Do not include slot numbers; FrameVision maps them to {expected_ids}."
        )
        return system, user

    def _telegram_autonomous_reference_selection_prompt(self, session: Dict[str, Any]) -> tuple[str, str]:
        bp = dict(session.get("blueprint") or {})
        slots = [{"slot": s.get("slot"), "beat": s.get("beat")} for s in bp.get("shot_slots", [])]
        system = (
            "Assess visual continuity across the COMPLETE locked beat list. Do not rewrite any beat or change its order. "
            "Discover distinct recurring identities, including identities absent from the provisional catalog. "
            "Merge repeated mentions of the SAME identity, but never merge distinct individuals based on shared type or role. "
            "For each candidate decide whether a change in its appearance would break recognition or story continuity. "
            "Recurrence alone does not justify a reference. Interchangeable incidental items have incidental continuity_need. "
            "essential means recognition is necessary to follow the story; useful means visual differences would be distracting; "
            "incidental means appearance can vary without changing the scene's meaning. "
            "State the stable visual features and explain the continuity need using the story, not generic praise. "
            "Record the identity's narrative role separately from appearance. "
            "For EVERY visible appearance provide its exact slot and a short verbatim quote from that beat supporting its presence, "
            "plus a concise relationship/action statement identifying what that subject does in that slot. "
            "Being mentioned, owned, remembered, or discussed is not proof of visible presence. Resolve pronouns from story context. "
            "Do not infer that an associated identity is present merely because another related identity appears. "
            "Use existing_id only for the same identity in the provisional catalog, otherwise use an empty string. "
            "Appearance must describe the individual only, without sheet-layout or camera instructions. "
            "Include incidental recurring candidates for an auditable decision. Code will rank candidates before applying the asset budget. "
            "Return JSON with candidates only. No story-specific preferences apply."
        )
        user = ("LOCKED BEATS:\n" + json.dumps(slots, ensure_ascii=False)
                + "\nPROVISIONAL CATALOG:\n" + json.dumps(bp.get("references") or [], ensure_ascii=False)
                + "\nUSER REQUEST:\n" + str(session.get("request") or ""))
        if session.get("reference_selection_feedback"):
            user += "\nCORRECT THIS OUTPUT ERROR:\n" + str(session["reference_selection_feedback"])
        return system, user

    def _telegram_autonomous_reference_selection_format(self) -> Dict[str, Any]:
        def obj(properties):
            return {"type": "object", "additionalProperties": False, "properties": properties, "required": list(properties)}
        string = {"type": "string"}
        evidence = obj({"slot": {"type": "integer", "minimum": 1}, "quote": string, "relationship": string})
        candidate = obj({
            "existing_id": string, "name": string,
            "type": {"type": "string", "enum": ["character", "animal", "vehicle", "object", "location"]},
            "appearance": string, "narrative_role": string,
            "continuity_need": {"type": "string", "enum": ["essential", "useful", "incidental"]},
            "reason": string, "stable_features": string,
            "appearances": {"type": "array", "items": evidence},
        })
        schema = obj({"candidates": {"type": "array", "maxItems": 32, "items": candidate}})
        return {"type": "json_schema", "json_schema": {"name": "continuity_selection", "strict": True, "schema": schema}}

    def _telegram_autonomous_apply_reference_selection(self, session: Dict[str, Any], result: Dict[str, Any]) -> None:
        """Validate evidence, rank before budgeting, and commit atomically.

        Semantic judgments belong to the model; code checks provenance and
        identity/slot integrity without subject-name heuristics.
        """
        bp = json.loads(json.dumps(session.get("blueprint") or {}))
        slots = list(bp.get("shot_slots") or [])
        by_slot = {int(s["slot"]): s for s in slots}
        old = {str(r.get("id")): r for r in bp.get("references", [])}
        candidates = result.get("candidates")
        if not isinstance(candidates, list) or len(candidates) > 32:
            raise ValueError("Return a candidates array with at most 32 identities.")
        normalize = lambda value: re.sub(r"\s+", " ", str(value)).strip().casefold()
        ranked, decisions, used_existing, used_names = [], [], set(), set()
        priority = {"essential": 2, "useful": 1, "incidental": 0}
        for index, item in enumerate(candidates):
            if not isinstance(item, dict):
                raise ValueError("Each candidate must be an object.")
            for field in ("name", "appearance", "narrative_role", "reason"):
                if not isinstance(item.get(field), str) or not item[field].strip():
                    raise ValueError(f"Candidate {index + 1} needs {field}.")
            need, kind = item.get("continuity_need"), item.get("type")
            if need not in priority or kind not in {"character", "animal", "vehicle", "object", "location"}:
                raise ValueError(f"Candidate {index + 1} has an invalid continuity_need or type.")
            rid = str(item.get("existing_id") or "").strip()
            name = normalize(item["name"])
            if (rid and (rid not in old or rid in used_existing)) or name in used_names:
                raise ValueError("Unknown/repeated existing_id or duplicate identity name; merge only the same identity.")
            used_existing.add(rid) if rid else None
            used_names.add(name)
            evidence, orders = [], set()
            if not isinstance(item.get("appearances"), list):
                raise ValueError("Candidate appearances must be an array.")
            for appearance in item["appearances"]:
                if not isinstance(appearance, dict):
                    raise ValueError("Each appearance needs slot, quote and relationship.")
                order = appearance.get("slot")
                quote = appearance.get("quote")
                relationship = appearance.get("relationship")
                if type(order) is not int or order not in by_slot or order in orders:
                    raise ValueError("Appearance slot must be a unique existing locked slot.")
                if not isinstance(quote, str) or not quote.strip() or normalize(quote) not in normalize(by_slot[order].get("beat") or ""):
                    raise ValueError(f"Appearance evidence for slot {order} must quote its locked beat.")
                if not isinstance(relationship, str) or not relationship.strip():
                    raise ValueError(f"Appearance at slot {order} needs its subject's relationship/action.")
                orders.add(order)
                evidence.append(dict(appearance))
            eligible = len(orders) >= 2 and priority[need] > 0
            if eligible and (not isinstance(item.get("stable_features"), str) or not item["stable_features"].strip()):
                raise ValueError("A selected identity needs stable_features that justify visual consistency.")
            decision = dict(item, eligible=eligible, selected=False)
            decisions.append(decision)
            if eligible:
                ranked.append((priority[need], len(orders), index, rid, item, sorted(orders), evidence))
        ranked.sort(key=lambda row: (-row[0], -row[1], row[2]))
        selected = []
        reserved_ids = set(old)
        for _, _, index, rid, item, orders, evidence in ranked[:9]:
            if not rid:
                number = 1
                while f"ref_{number}" in reserved_ids:
                    number += 1
                rid = f"ref_{number}"
            reserved_ids.add(rid)
            role = "character_identity" if item["type"] in {"character", "animal"} else "reusable_set" if item["type"] == "location" else "reusable_asset"
            selected.append({"id": rid, "name": item["name"], "type": item["type"],
                             "prompt": item["appearance"], "reference_role": role, "clip_orders": orders,
                             "narrative_role": item["narrative_role"], "stable_features": item["stable_features"],
                             "continuity_need": item["continuity_need"], "selection_reason": item["reason"],
                             "appearance_evidence": evidence})
            decisions[index]["selected"] = True
        if not selected and old and self._telegram_autonomous_user_requested_refs(str(session.get("request") or "")):
            raise ValueError("No eligible reference selected despite explicitly requested references; review the locked beats.")
        for slot in slots:
            slot["reference_ids"] = [r["id"] for r in selected if int(slot["slot"]) in r["clip_orders"]]
        ids_by_slot = {int(s["slot"]): list(s["reference_ids"]) for s in slots}
        for key, shot in (bp.get("shots") or {}).items():
            shot["reference_ids"] = ids_by_slot.get(int(key), [])
        for section in bp.get("sections") or []:
            section_ids = []
            for shot in section.get("shots") or []:
                shot["reference_ids"] = ids_by_slot.get(int(shot.get("slot") or 0), [])
                section_ids.extend(shot["reference_ids"])
            section["reference_ids"] = list(dict.fromkeys(section_ids))
        bp["references"] = selected
        bp["reference_selection_version"] = 1
        batches = self._telegram_autonomous_build_batches(bp)
        session["blueprint"] = bp
        session["shot_batches"] = batches
        session["reference_selection_decisions"] = decisions

    def _telegram_autonomous_finish_reference_selection(self, key: str, session: Dict[str, Any], pending: Dict[str, Any], result=None, error="") -> None:
        try:
            if error:
                raise ValueError(error)
            self._telegram_autonomous_apply_reference_selection(session, result or {})
        except Exception as exc:
            session["reference_selection_feedback"] = str(exc)
            retry = self._telegram_autonomous_claim_stage_retry(session, "autonomous_ref_selection")
            if retry:
                session["status"] = "auditing_references"
                pending["structured_retry"] = True
                self._telegram_agent_pending = pending
                self._telegram_autonomous_save(session)
                self._telegram_send_text(key, f"Retrying continuity selection ({retry}/2): {exc}")
                return
            session["reference_selection_fallback"] = str(exc)
            self._telegram_send_text(key, "Continuity selection could not be validated after two retries. Keeping the existing reference catalog and continuing prompt writing.")
        else:
            session.pop("reference_selection_feedback", None)
            self._telegram_send_text(key, f"Continuity selection complete: {len(session['blueprint']['references'])} visual identities selected. Writing shot prompts now.")
        session.pop("reference_selection_pending", None)
        session["status"] = "building_shots"
        pending.pop("structured_retry", None)
        pending["phase"] = "autonomous_shots"
        self._telegram_agent_pending = pending
        self._telegram_autonomous_write_json(session, "blueprint", session["blueprint"])
        self._telegram_autonomous_save(session)

    def _telegram_autonomous_reference_audit_prompt(self, session: Dict[str, Any]) -> tuple[str, str]:
        plan_refs = [dict(x) for x in list((session.get("blueprint") or {}).get("references") or []) if isinstance(x, dict)]
        clips = [dict(x) for x in list(session.get("staged_clips") or []) if isinstance(x, dict)]
        existing = [
            {
                "id": str(r.get("id") or ""),
                "name": str(r.get("name") or ""),
                "type": str(r.get("type") or ""),
                "prompt": str(r.get("prompt") or ""),
            }
            for r in plan_refs
        ]
        shot_view = [
            {
                "order": int(c.get("order") or 0),
                "purpose": str(c.get("purpose") or ""),
                "prompt": str(c.get("prompt") or ""),
            }
            for c in clips
        ]
        remaining = max(0, 9 - len(existing))
        system = (
            "You are the final continuity-reference auditor for FrameVision. The complete story and all video prompts are already written. "
            "Your ONLY job is to find recurring visual identities that are missing from the existing reference catalog and would visibly change between clips without a reference sheet. "
            "Be generous about consistency: recurring people, recurring animals/creatures, and recurring distinctive vehicles/objects should normally get references when they appear in 2 or more clips. "
            "A recurring named or clearly same little girl, dog, mower, car, spaceship, instrument, costume-critical person, or bounded set is a good candidate. "
            "Do NOT create refs for disposable generic props, weather, roads, sky, crowds as a whole, or one-off scenery. "
            "Do NOT duplicate an identity already present in EXISTING REFERENCES. Do not redesign existing refs. "
            "For every missing reference, list every clip order where that exact recurring identity appears. "
            "The reference prompt must describe ONE stable subject identity suitable for a Krea multi-view reference sheet; do not describe actions from a specific shot. "
            f"There is room for at most {remaining} additional references. Return JSON only."
        )
        user = (
            "EXISTING REFERENCES:\n" + json.dumps(existing, ensure_ascii=False, indent=2)
            + "\n\nCOMPLETE CLIP LIST:\n" + json.dumps(shot_view, ensure_ascii=False, indent=2)
            + "\n\nReturn only genuinely useful missing recurring identities. If none are missing, return an empty missing_references array."
        )
        return system, user

    def _telegram_autonomous_reference_audit_response_format(self, session: Dict[str, Any]) -> Dict[str, Any]:
        existing = [x for x in list((session.get("blueprint") or {}).get("references") or []) if isinstance(x, dict)]
        remaining = max(0, 9 - len(existing))
        item = {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "name": {"type": "string"},
                "type": {"type": "string", "enum": ["character", "person", "creature", "animal", "vehicle", "object", "location"]},
                "prompt": {"type": "string"},
                "appears_in_clips": {"type": "array", "items": {"type": "integer", "minimum": 1}},
            },
            "required": ["name", "type", "prompt", "appears_in_clips"],
        }
        schema = {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "missing_references": {
                    "type": "array",
                    "maxItems": remaining,
                    "items": item,
                }
            },
            "required": ["missing_references"],
        }
        return {"type": "json_schema", "json_schema": {"name": "framevision_reference_audit", "strict": True, "schema": schema}}

    def _telegram_autonomous_apply_reference_audit(self, session: Dict[str, Any], obj: Dict[str, Any]) -> int:
        bp = dict(session.get("blueprint") or {})
        clips = [dict(x) for x in list(session.get("staged_clips") or []) if isinstance(x, dict)]
        refs = [dict(x) for x in list(bp.get("references") or []) if isinstance(x, dict)]

        def _norm(value: Any) -> str:
            return re.sub(r"[^a-z0-9]+", " ", str(value or "").lower()).strip()

        stop = {"the", "a", "an", "little", "small", "same", "recurring", "friendly", "bright", "old", "new"}

        def _tokens(value: Any) -> set[str]:
            return {x for x in _norm(value).split() if len(x) >= 3 and x not in stop}

        existing_names = {_norm(r.get("name")) for r in refs}
        existing_name_tokens = [_tokens(r.get("name")) for r in refs]
        used_ids = {str(r.get("id") or "") for r in refs}
        raw = list((obj or {}).get("missing_references") or [])
        added = 0
        max_add = max(0, 9 - len(refs))

        # The LLM decides WHAT deserves a stable identity. FrameVision decides WHERE it
        # occurs. This avoids losing a good reference because the model mistyped a clip
        # number (for example 11 -> 1 or 22 -> 2).
        searchable = []
        for c in clips:
            text = _norm(str(c.get("purpose") or "") + " " + str(c.get("prompt") or ""))
            searchable.append((int(c.get("order") or 0), text, set(text.split())))

        animal_words = {"dog", "retriever", "terrier", "cat", "kitten", "puppy", "horse", "bird", "pigeon", "butterfly", "rabbit", "squirrel"}
        person_words = {"child", "kid", "girl", "boy", "toddler", "woman", "man", "mother", "father", "musician", "busker", "driver"}

        for item in raw:
            if added >= max_add or not isinstance(item, dict):
                break
            name = str(item.get("name") or "").strip()
            prompt = str(item.get("prompt") or "").strip()
            kind = str(item.get("type") or "object").strip().lower()
            norm_name = _norm(name)
            if not name or not prompt or not norm_name or norm_name in existing_names:
                continue

            # Never make a separate sheet for a body/detail of an already referenced identity.
            if kind in {"detail", "body_part", "feature"}:
                continue

            cand_tokens = _tokens(name)
            # Semantic duplicate guard: e.g. "Earth City Skyline" must not duplicate
            # existing "Earth City" just because the audit phrased it more specifically.
            duplicate = False
            for toks in existing_name_tokens:
                if not cand_tokens or not toks:
                    continue
                overlap = cand_tokens & toks
                if len(overlap) >= 2 or (len(cand_tokens) == 1 and cand_tokens <= toks) or (len(toks) == 1 and toks <= cand_tokens):
                    duplicate = True
                    break
            if duplicate:
                continue

            combined = _norm(name + " " + prompt)
            aliases = set(cand_tokens)
            if kind in {"creature", "animal"}:
                aliases |= {w for w in animal_words if w in combined.split()}
            if kind in {"character", "person"}:
                aliases |= {w for w in person_words if w in combined.split()}
                # Character audits often invent a proper name ("Mia") even though the
                # locked prompts only say "the child". Prefer the role words too.
                if "toddler" in combined or "girl" in combined:
                    aliases |= {"child", "kid", "girl", "toddler"}
                if "boy" in combined:
                    aliases |= {"child", "kid", "boy"}

            orders = []
            for order, text, words in searchable:
                if not order:
                    continue
                # Direct normalized name phrase is strongest. Otherwise any distinctive
                # identity token is enough for these short, already-locked clip prompts.
                direct = norm_name and norm_name in text
                token_hit = bool(aliases & words)
                if direct or token_hit:
                    orders.append(order)

            # Backward compatibility: older audit responses used clip_orders while the
            # current prompt naturally emitted appears_in_clips. Use those only as a
            # fallback when deterministic matching cannot find recurrence.
            if len(orders) < 2:
                hinted = item.get("appears_in_clips")
                if hinted is None:
                    hinted = item.get("clip_orders")
                fallback = []
                for v in list(hinted or []):
                    try:
                        n = int(v)
                    except Exception:
                        continue
                    if 1 <= n <= len(clips) and n not in fallback:
                        fallback.append(n)
                if len(fallback) >= 2:
                    orders = fallback

            orders = sorted(set(orders))
            if len(orders) < 2:
                continue

            next_no = 1
            while f"ref_{next_no}" in used_ids:
                next_no += 1
            rid = f"ref_{next_no}"
            used_ids.add(rid)
            existing_names.add(norm_name)
            existing_name_tokens.append(cand_tokens)
            if kind in {"character", "person", "creature", "animal"}:
                role = "character_identity"
            elif kind in {"vehicle", "object"}:
                role = "reusable_asset"
            elif kind == "location":
                role = "reusable_set"
            else:
                role = "reusable_asset"
            ref = {
                "id": rid,
                "name": name,
                "type": kind,
                "prompt": prompt,
                "clip_orders": orders,
                "reference_role": role,
            }
            refs.append(ref)
            by_order = {int(c.get("order") or 0): c for c in clips}
            for order in orders:
                clip = by_order.get(order)
                if not clip:
                    continue
                ids = list(clip.get("reference_ids") or [])
                if rid not in ids:
                    ids.append(rid)
                clip["reference_ids"] = ids
            added += 1

        bp["references"] = refs
        # Keep the shot-level blueprint IDs in sync so saved diagnostics and later
        # prompt/ref routing agree about which recurring subject belongs to each clip.
        shots = dict(bp.get("shots") or {})
        for clip in clips:
            key = str(int(clip.get("order") or 0))
            if key in shots and isinstance(shots[key], dict):
                shots[key]["reference_ids"] = list(clip.get("reference_ids") or [])
        if shots:
            bp["shots"] = shots
        session["blueprint"] = bp
        session["staged_clips"] = clips
        return added

    def _telegram_autonomous_finalize_staged_plan(self, session: Dict[str, Any]) -> Dict[str, Any]:
        staged = [dict(x) for x in list(session.get("staged_clips") or []) if isinstance(x, dict)]
        staged.sort(key=lambda c: int(c.get("order") or 0))
        bp = dict(session.get("blueprint") or {})
        refs = [dict(x) for x in list(bp.get("references") or []) if isinstance(x, dict)]
        valid_ids = {str(r.get("id") or "") for r in refs}
        for clip in staged:
            clip["reference_ids"] = [
                str(rid) for rid in list(clip.get("reference_ids") or [])
                if str(rid) in valid_ids
            ]
        for ref in refs:
            rid = str(ref.get("id") or "")
            ref["clip_orders"] = [
                int(c.get("order") or 0) for c in staged
                if rid and rid in list(c.get("reference_ids") or [])
            ]
        final_plan = {
            "title": bp.get("title"), "target_duration": bp.get("target_duration"),
            "duration_min": bp.get("duration_min"), "duration_max": bp.get("duration_max"),
            "video_model": bp.get("video_model"), "resolution": bp.get("resolution"),
            "aspect": bp.get("aspect"), "story_summary": bp.get("story_summary"),
            "references": refs, "music": bp.get("music"), "clips": staged,
        }
        return self._telegram_autonomous_validate_plan(final_plan, str(session.get("request") or ""))

    def _telegram_autonomous_shot_response_format(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        slots = [dict(x) for x in list(batch.get("slots") or []) if isinstance(x, dict)]
        count = len(slots)
        clip_value = {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "purpose": {"type": "string"},
                "prompt": {"type": "string"},
            },
            "required": ["purpose", "prompt"],
        }
        schema = {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "clips": {
                    "type": "array",
                    "minItems": count,
                    "maxItems": count,
                    "items": clip_value,
                }
            },
            "required": ["clips"],
        }
        return {"type": "json_schema", "json_schema": {"name": "framevision_ordered_shot_batch", "strict": True, "schema": schema}}

    def _telegram_autonomous_parse_shot_batch(self, obj: Dict[str, Any], batch: Dict[str, Any], session: Optional[Dict[str, Any]] = None) -> list:
        slots = [dict(x) for x in list(batch.get("slots") or []) if isinstance(x, dict)]
        expected_ids = [int(x.get("slot") or 0) for x in slots]
        raw = (obj or {}).get("clips")
        if isinstance(raw, list):
            items = [dict(x) for x in raw if isinstance(x, dict)]
        elif isinstance(raw, dict):
            # Compatibility with older/no-grammar responses. Only accept it when
            # no information was lost; never guess a missing duplicate key.
            items = [dict(v) for v in raw.values() if isinstance(v, dict)]
        else:
            items = []
        if len(items) != len(slots):
            raise ValueError(f"Shot batch returned {len(items)} prompt objects; required exactly {len(slots)} for locked slots {expected_ids}.")

        request_text = str((session or {}).get("request") or "")
        technical_duration = float(self._telegram_autonomous_duration_contract(request_text)["preferred_avg"])
        out = []
        for idx, slot_id in enumerate(expected_ids):
            item = items[idx]
            low = {str(k).strip().lower().replace("**", ""): v for k, v in item.items()}
            prompt = ""
            for field in ("prompt", "video_prompt", "scene_description", "detailed_description", "integrated_multimodal_description"):
                value = low.get(field)
                if isinstance(value, str) and value.strip():
                    prompt = value.strip()
                    break
            if not prompt:
                # Some local models expand the requested prompt into named
                # components instead of returning a single `prompt`. Preserve
                # that useful content deterministically rather than failing.
                parts = []
                aliases = (
                    ("Environment/background", ("environment_background", "environment", "background")),
                    ("Subject appearance/action", ("subject_appearance_action", "subject_apprise_action", "subject_action", "subject")),
                    ("Continuity", ("continuity", "continuity_notes", "continuity_elements", "continuity_description")),
                    ("Camera framing/movement", ("camera_framing_movement", "camera_framming_movement", "camera", "camera_movement", "camera_angle")),
                    ("Lighting/atmosphere", ("lighting_atmosphere", "lighting", "atmosphere")),
                )
                for label, keys in aliases:
                    values = []
                    for key in keys:
                        value = low.get(key)
                        if isinstance(value, str) and value.strip() and value.strip() not in values:
                            values.append(value.strip())
                    if values:
                        parts.append(f"{label}: " + "\n".join(values))
                prompt = "\n".join(parts).strip()
            if not prompt:
                raise ValueError(f"Shot batch slot {slot_id} has no usable prompt content.")
            refs = []
            for rid in list(slots[idx].get("reference_ids") or []):
                rid = re.sub(r"[^0-9A-Za-z_-]+", "_", str(rid)).strip("_")
                if rid and rid not in refs:
                    refs.append(rid)
            out.append({
                "order": slot_id,
                "slot": slot_id,
                "duration": technical_duration,
                "purpose": str(low.get("purpose") or slots[idx].get("beat") or "").strip(),
                "reference_ids": refs,
                "prompt": prompt,
            })
        return out

    @staticmethod
    def _telegram_autonomous_parse_root_array_payload(payload: object, key: str) -> Dict[str, Any]:
        """Recover a harmless root JSON array from local-LLM structured stages.

        Some local models ignore an object wrapper and return the requested item
        array directly. Preserve the actual returned objects; never invent or pad
        missing items. The normal stage validator still enforces the exact count.
        """
        if isinstance(payload, dict):
            raw = str(payload.get("content") or "").strip()
        else:
            raw = str(payload or "").strip()
        answer, _reasoning = _split_inline_reasoning(raw)
        raw = str(answer or raw).strip()
        raw = re.sub(r"^```(?:json|javascript|python)?\s*", "", raw, flags=re.I)
        raw = re.sub(r"\s*```$", "", raw).strip()
        starts = [m.start() for m in re.finditer(r"\[", raw)]
        ends = [m.start() for m in re.finditer(r"\]", raw)]
        candidates = [raw]
        for a in starts:
            for b in reversed(ends):
                if b > a:
                    candidates.append(raw[a:b + 1])
                    break
        for candidate in candidates:
            try:
                value = json.loads(candidate)
            except Exception:
                try:
                    value = ast.literal_eval(candidate)
                except Exception:
                    continue
            if isinstance(value, list) and all(isinstance(x, dict) for x in value):
                return {str(key): value}
        return {}

    def _telegram_autonomous_write_raw(self, session: Dict[str, Any], phase: str, payload: object) -> None:
        try:
            sid = re.sub(r"[^0-9A-Za-z_-]+", "_", str(session.get("id") or "agent"))
            idx = int(session.get("batch_index") or 0)
            suffix = f"_{idx+1:02d}" if phase == "shots" else ""
            path = self._telegram_autonomous_dir() / f"{sid}_{phase}{suffix}_raw.txt"
            path.write_text(str(payload if isinstance(payload, str) else json.dumps(payload, ensure_ascii=False, indent=2)), encoding="utf-8")
            session["last_raw_log"] = str(path)
        except Exception:
            pass

    def _telegram_autonomous_write_json(self, session: Dict[str, Any], name: str, payload: Dict[str, Any]) -> None:
        try:
            sid = re.sub(r"[^0-9A-Za-z_-]+", "_", str(session.get("id") or "agent"))
            path = self._telegram_autonomous_dir() / f"{sid}_{name}.json"
            path.write_text(json.dumps(payload or {}, ensure_ascii=False, indent=2), encoding="utf-8")
            session[f"{name}_path"] = str(path)
        except Exception:
            pass

    def _telegram_autonomous_validate_plan(self, plan: Dict[str, Any], request: str) -> Dict[str, Any]:
        p = dict(plan or {})
        low_req = str(request or "").lower()

        # Preserve explicit model choices even if a small LLM ignored them.
        if "minimax" in low_req:
            p["video_model"] = "minimax_h3"
        elif "ltx 2.5" in low_req or "ltx2.5" in low_req or "ltx25" in low_req:
            p["video_model"] = "ltx25"
        elif "ltx 2.3" in low_req or "ltx2.3" in low_req or "ltx23" in low_req:
            p["video_model"] = "ltx23"

        model = str(p.get("video_model") or "minimax_h3").lower().replace(" ", "").replace("-", "")
        if "minimax" in model:
            model = "minimax_h3"
        elif "25" in model:
            model = "ltx25"
        elif "23" in model:
            model = "ltx23"
        else:
            model = "minimax_h3"
        p["video_model"] = model

        duration = 0.0
        duration_min = 0.0
        duration_max = 0.0
        range_patterns = (
            r"\bbetween\s+(\d+(?:\.\d+)?)\s+(?:and|to|-)\s+(\d+(?:\.\d+)?)\s*(minutes?|mins?|min|seconds?|secs?|sec)\b",
            r"\b(\d+(?:\.\d+)?)\s*(?:-|–|—|to)\s*(\d+(?:\.\d+)?)\s*(minutes?|mins?|min|seconds?|secs?|sec)\b",
        )
        range_match = None
        for pat in range_patterns:
            range_match = re.search(pat, low_req)
            if range_match:
                break
        if range_match:
            lo = float(range_match.group(1))
            hi = float(range_match.group(2))
            unit = str(range_match.group(3) or "").lower()
            if lo > hi:
                lo, hi = hi, lo
            mult = 60.0 if unit.startswith(("min", "minute")) else 1.0
            duration_min = max(10.0, min(600.0, lo * mult))
            duration_max = max(duration_min, min(600.0, hi * mult))
            p["duration_min"] = duration_min
            p["duration_max"] = duration_max
        else:
            m = re.search(r"\b(\d+(?:\.\d+)?)\s*(minutes?|mins?|min)\b", low_req)
            if m:
                duration = float(m.group(1)) * 60.0
            else:
                m = re.search(r"\b(\d+(?:\.\d+)?)\s*(seconds?|secs?|sec)\b", low_req)
                if m:
                    duration = float(m.group(1))
            if duration <= 0:
                try:
                    duration = float(p.get("target_duration") or 60)
                except Exception:
                    duration = 60.0
            duration = max(10.0, min(600.0, duration))
            p["target_duration"] = duration

        res = str(p.get("resolution") or ("544p" if model == "minimax_h3" else "704p"))
        for key in ("1088p", "1080p", "768p", "704p", "544p", "480p"):
            if key in low_req:
                res = key
                break
        if re.search(r"\b960\s*[x×]\s*544\b", low_req):
            res = "544p"
        aspect = str(p.get("aspect") or "16:9")
        if any(x in low_req for x in ("portrait", "vertical", "9:16")):
            aspect = "9:16"
        elif any(x in low_req for x in ("square", "1:1")):
            aspect = "1:1"
        elif any(x in low_req for x in ("landscape", "16:9")):
            aspect = "16:9"
        p["resolution"] = res
        p["aspect"] = aspect

        clips_in = [x for x in list(p.get("clips") or []) if isinstance(x, dict)]
        clips = []
        for i, item in enumerate(clips_in[:120], start=1):
            prompt = str(item.get("prompt") or "").strip()
            if not prompt:
                continue
            try:
                d = float(item.get("duration") or 6.0)
            except Exception:
                d = 6.0
            d = max(3.0, min(15.0, d))
            reference_ids = []
            for rid in list(item.get("reference_ids") or item.get("refs") or []):
                rid = re.sub(r"[^0-9A-Za-z_-]+", "_", str(rid or "").strip()).strip("_")
                if rid and rid not in reference_ids:
                    reference_ids.append(rid)
            clips.append({
                "order": len(clips) + 1,
                "duration": d,
                "purpose": str(item.get("purpose") or "").strip(),
                "reference_ids": reference_ids,
                "prompt": prompt,
            })
        if not clips:
            raise ValueError("The Agent did not produce any usable clip prompts.")

        story_contract = self._telegram_autonomous_story_contract(request)
        min_story_clips = int(story_contract.get("min_clips") or 0)
        if min_story_clips > 0 and len(clips) < min_story_clips:
            raise ValueError(
                f"STORY_TOO_SMALL|{len(clips)}|{min_story_clips}|{self._telegram_autonomous_clip_seconds({'clips': clips}):.3f}"
            )

        total = sum(float(x["duration"]) for x in clips)
        if duration_min > 0.0 and duration_max > 0.0:
            if total + 0.01 < duration_min:
                raise ValueError(
                    f"DURATION_SHORT|{total:.3f}|{duration_min:.3f}|{duration_max:.3f}|range"
                )
            duration = min(duration_max, max(duration_min, total))
            p["target_duration"] = duration
        else:
            if total + 0.01 < duration:
                raise ValueError(
                    f"DURATION_SHORT|{total:.3f}|{duration:.3f}|{duration:.3f}|exact"
                )
        p["clips"] = clips

        references_in = [x for x in list(p.get("references") or []) if isinstance(x, dict)]
        legacy_sheet = p.get("character_sheet_krea_2")
        if not references_in and isinstance(legacy_sheet, dict):
            legacy_name = str(legacy_sheet.get("name") or "Recurring character").strip()
            notes = legacy_sheet.get("generation_notes")
            if isinstance(notes, list):
                legacy_prompt = ". ".join(str(x).strip() for x in notes if str(x).strip())
            else:
                legacy_prompt = str(legacy_sheet.get("prompt") or notes or "").strip()
            if legacy_prompt:
                references_in = [{
                    "id": "char_1",
                    "name": legacy_name.replace("Base Character Sheet", "").replace("–", "-").strip(" -"),
                    "type": "character",
                    "prompt": legacy_prompt,
                    "clip_orders": [],
                }]

        references = []
        used_ids = set()
        for n, item in enumerate(references_in[:9], start=1):
            rid = re.sub(r"[^0-9A-Za-z_-]+", "_", str(item.get("id") or f"ref_{n}").strip()).strip("_") or f"ref_{n}"
            if rid in used_ids:
                rid = f"{rid}_{n}"
            used_ids.add(rid)
            name = str(item.get("name") or rid).strip()
            prompt = str(item.get("prompt") or item.get("description") or "").strip()
            if not prompt:
                notes = item.get("generation_notes")
                if isinstance(notes, list):
                    prompt = ". ".join(str(x).strip() for x in notes if str(x).strip())
            if not prompt:
                continue
            orders = []
            for val in list(item.get("clip_orders") or []):
                try:
                    order = int(val)
                except Exception:
                    continue
                if 1 <= order <= len(clips) and order not in orders:
                    orders.append(order)
            references.append({
                "id": rid,
                "name": name,
                "type": str(item.get("type") or "character").strip().lower(),
                "prompt": prompt,
                "clip_orders": orders,
                # Preserve the typed execution role computed by blueprint
                # validation. Losing this here makes the later Krea selector
                # discard otherwise valid character/reusable references.
                "reference_role": str(item.get("reference_role") or "").strip().lower(),
                "narrative_role": str(item.get("narrative_role") or ""),
                "stable_features": str(item.get("stable_features") or ""),
                "continuity_need": str(item.get("continuity_need") or ""),
                "selection_reason": str(item.get("selection_reason") or ""),
                "appearance_evidence": list(item.get("appearance_evidence") or []),
            })

        valid_ref_ids = {x["id"] for x in references}
        for ref in references:
            explicit_orders = []
            for x in list(ref.get("clip_orders") or []):
                try:
                    n = int(x)
                except Exception:
                    continue
                if n > 0 and n not in explicit_orders:
                    explicit_orders.append(n)
            # Membership is assigned before prompt writing. Finished prose must
            # never add an identity or override an explicitly empty selection.
            ref["clip_orders"] = [int(c["order"]) for c in clips if ref["id"] in c["reference_ids"]]

        for clip in clips:
            clip["reference_ids"] = [x for x in clip.get("reference_ids", []) if x in valid_ref_ids]

        for ref in references:
            ref["clip_orders"] = [
                int(clip["order"]) for clip in clips
                if ref["id"] in list(clip.get("reference_ids") or [])
            ]
            if not str(ref.get("reference_role") or ""):
                kind = str(ref.get("type") or "object").lower()
                uses = len(ref["clip_orders"])
                if kind in {"character", "person", "creature", "alien", "animal"}:
                    ref["reference_role"] = "character_identity"
                elif kind in {"vehicle", "car", "ship", "spaceship", "aircraft", "bike", "object", "prop", "item", "device", "artifact"} and uses >= 2:
                    ref["reference_role"] = "reusable_asset"
                else:
                    ref["reference_role"] = "transient_environment"
        p["references"] = references

        music = dict(p.get("music") or {})
        if "music video" in low_req or "background music" in low_req or "ace" in low_req:
            music["enabled"] = True
        music.setdefault("enabled", True)
        music.setdefault("instrumental", True)
        music.setdefault("genre", "Electronic")
        music.setdefault("subgenre", "")
        music.setdefault("caption", "cinematic instrumental soundtrack matching the story progression")
        music.setdefault("bpm", 0)
        p["music"] = music
        p["title"] = re.sub(r"[^0-9A-Za-z _-]+", "", str(p.get("title") or "Agent Video")).strip()[:80] or "Agent Video"
        p["story_summary"] = str(p.get("story_summary") or request or "").strip()
        return p

    def _telegram_minimax_saved_settings_summary(self, chat_id: str) -> str:
        try:
            router = self._telegram_router_for_chat(str(chat_id))
            return str(router.minimax_h3_saved_settings_summary())
        except Exception:
            return "saved MiniMax settings could not be read"

    def _telegram_autonomous_video_start(self, chat_id: str, request: str, attachments: list) -> bool:
        key = str(chat_id)
        if key in self._telegram_autonomous_sessions and str(self._telegram_autonomous_sessions[key].get("status") or "") in {"planning", "planning_retry", "building_story", "building_shots", "music_preset_choice", "queueing", "generating_refs", "generating", "review_ready", "assembling"}:
            self._telegram_send_text(key, "You already have an autonomous Agent video project running. Use /cancel before starting another one.")
            return True
        try:
            if self._running_queue_job_files():
                self._telegram_send_text(key, "The Agent needs the local LLM to plan this project, but a generation job is currently using FrameVision. Try again after it finishes, or cancel it first.")
                return True
        except Exception:
            pass
        try:
            self._validate_runner_and_model()
        except Exception as exc:
            self._telegram_send_text(key, f"I need the currently selected FrameVision LLM for autonomous planning, but it is not ready: {exc}")
            return True

        sid = f"agent_{int(time.time())}_{uuid.uuid4().hex[:6]}"
        session = {
            "id": sid,
            "chat_id": key,
            "request": str(request or "").strip(),
            "attachments": list(attachments or []),
            "status": "planning",
            "created_at": time.time(),
            "plan": {},
            "blueprint": {},
            "shot_count": 0,
            "shot_batches": [],
            "batch_index": 0,
            "staged_clips": [],
            "reference_assets": [],
            "clip_outputs": [],
            "clip_queued_at": [],
            "music_out_dir": "",
            "music_queued_at": 0.0,
            "music_path": "",
            "final_output": "",
            "assembly_queued_at": 0.0,
            "last_notice": "",
        }
        self._telegram_autonomous_sessions[key] = session
        self._telegram_autonomous_save(session)
        self._telegram_send_text(
            key,
            "Agent mode: I’ll first lock any requested ACE-Step preset choice, then create the complete story blueprint, "
            "expand it into small shot-prompt batches, generate the clips + music, and assemble the final MP4."
        )
        if "minimax" in str(request or "").lower():
            self._telegram_send_text(
                key,
                "MiniMax H3 generation will use the LAST SAVED MiniMax tab settings. "
                "Current saved snapshot: " + self._telegram_minimax_saved_settings_summary(key) + ". "
                "Any saved LoRA with a non-zero strength will be applied to every Agent MiniMax clip."
            )

        # Human input belongs at the beginning of an autonomous run. If the
        # request already contains a genre that exists in the real ACE-Step
        # preset library, resolve its subgenre now, before blueprint work starts.
        # This prevents an otherwise autonomous multi-minute planning pass from
        # unexpectedly stopping just before generation.
        if self._telegram_autonomous_prepare_initial_music_preset_choice(key, session):
            self._telegram_autonomous_save(session)
            return True

        self._telegram_agent_pending = {
            "chat_id": key,
            "text": session["request"],
            "attachments": list(attachments or []),
            "purpose": "autonomous_video",
            "phase": "autonomous_blueprint",
            "session_id": sid,
        }
        if self.server_ready and self._same_loaded_config():
            QtCore.QTimer.singleShot(0, self._telegram_agent_continue_pending)
        else:
            self._set_status("Loading LLM for autonomous Telegram Agent…", "loading")
            self._load_selected_model()
        return True

    def _telegram_autonomous_start_thread(self, pending: Dict[str, Any]) -> None:
        key = str(pending.get("chat_id") or "")
        session = self._telegram_autonomous_sessions.get(key)
        if not isinstance(session, dict):
            self._telegram_agent_finish(unload=True)
            return
        pending_sid = str(pending.get("session_id") or "").strip()
        current_sid = str(session.get("id") or "").strip()
        if pending_sid and current_sid and pending_sid != current_sid:
            # Never let a delayed callback/pending pass from an older Telegram
            # project operate on the newest project in the same chat.
            self._telegram_agent_finish(unload=True)
            return
        phase = str(pending.get("phase") or "autonomous_blueprint")

        response_format = None
        if phase == "autonomous_ref_audit" and session.get("reference_selection_pending"):
            system, user = self._telegram_autonomous_reference_selection_prompt(session)
            response_format = self._telegram_autonomous_reference_selection_format()
            self._telegram_send_text(key, "Selecting continuity references from the complete locked beat list.")
        elif phase == "autonomous_ref_audit":
            system, user = self._telegram_autonomous_reference_audit_prompt(session)
            response_format = self._telegram_autonomous_reference_audit_response_format(session)
            self._telegram_send_text(key, "Doing one final recurrence/identity pass over the complete story before Krea starts, so recurring people, animals and distinctive objects get stable refs.")
        elif phase == "autonomous_beats":
            system, user = self._telegram_autonomous_beat_batch_prompt(session)
            response_format = self._telegram_autonomous_beat_response_format(session)
            total = int(self._telegram_autonomous_duration_contract(str(session.get("request") or ""))["shot_count"])
            idx = int(session.get("beat_batch_index") or 0); start = idx * 5 + 1; end = min(total, start + 4)
            self._telegram_send_text(key, f"Planning story beats {start}-{end} of {total}.")
        elif phase == "autonomous_shots":
            batches = list(session.get("shot_batches") or [])
            idx = int(session.get("batch_index") or 0)
            if idx >= len(batches):
                self._telegram_agent_finish(unload=True)
                return
            batch = dict(batches[idx])
            system, user = self._telegram_autonomous_shot_batch_prompt(session, batch)
            if session.get("shot_batch_validation_feedback"):
                user += "\n\nCORRECT THIS EXACT CURRENT-CHUNK ERROR:\n" + str(session["shot_batch_validation_feedback"])
            response_format = self._telegram_autonomous_shot_response_format(batch)
            self._telegram_send_text(
                key,
                f"Creating shot prompts {int(batch.get('global_start') or 1)}-"
                f"{int(batch.get('global_end') or batch.get('global_start') or 1)} "
                f"of {int(session.get('shot_count') or 0)}."
            )
        else:
            system, user = self._telegram_autonomous_blueprint_prompt(str(session.get("request") or ""))
            response_format = self._telegram_autonomous_blueprint_response_format(str(session.get("request") or ""))
            c = self._telegram_autonomous_duration_contract(str(session.get("request") or ""))
            self._telegram_send_text(
                key,
                f"Planning the complete story blueprint first. FrameVision locked {int(c['shot_count'])} "
                f"shots for the {float(c['target']):.0f}s production budget; the blueprint must budget the whole story before 5-shot beat chunks begin."
            )

        force_retry = bool(pending.get("structured_retry", False))
        user = str(user or "").rstrip() + (
            "\n\n/no_think\nReturn the requested JSON immediately. Do not output <think>, reasoning, analysis, markdown, or prose."
        )
        if force_retry:
            user += (
                "\nThis is a structured-output retry because the previous answer was unusable or incomplete. "
                "Start directly with `{` and finish the complete JSON object before stopping."
            )
            if phase == "autonomous_blueprint" and bool(session.get("metadata_required_refs_retry_done", False)):
                user += (
                    "\nIMPORTANT: the user explicitly requested Krea/Ref2VA character/reference sheets. "
                    "The `references` array MUST contain every recurring character/creature and any genuinely recurring visual subject that needs identity consistency. It must not be empty."
                )
            if phase == "autonomous_blueprint" and str(session.get("blueprint_validation_feedback") or "").strip():
                feedback = str(session.get("blueprint_validation_feedback") or "").strip()
                user += (
                    "\nHARD BLUEPRINT CORRECTION REQUIRED: " + feedback
                    + "\nRewrite the complete blueprint correctly. Do not merely explain the mistake."
                )
                if "no locked story_sections" in feedback.lower():
                    user += (
                        "\nYOUR PREVIOUS ANSWER OMITTED THE REQUIRED STORY BUDGET. "
                        "The corrected JSON MUST contain a TOP-LEVEL key named exactly `story_sections`. "
                        "It MUST be an array of section objects with title, role, purpose, must_achieve and clip_count. "
                        f"The clip_count values MUST total exactly {int(self._telegram_autonomous_duration_contract(str(session.get('request') or ''))['shot_count'])}. "
                        "Do not return story_arc, acts, shots or an outline instead."
                    )
        max_tokens = max(2000, int(self.settings_dialog.sp_max_tokens.value()))
        if session.get("reference_selection_pending") and phase == "autonomous_ref_audit":
            max_tokens = max(6000, max_tokens)
        thread = ChatCompletionThread(
            self.server_url,
            [{"role": "system", "content": system}, {"role": "user", "content": user}],
            max_tokens,
            min(0.45, float(self.settings_dialog.sp_temp.value())),
            self.settings_dialog.sp_top_p.value(),
            self.settings_dialog.sp_top_k.value(),
            self.settings_dialog.sp_repeat_penalty.value(),
            self.settings_dialog.sp_generation_timeout.value(),
            self,
            enable_thinking=False,
            response_format=response_format,
        )
        self._telegram_agent_thread = thread
        thread.succeeded.connect(self._telegram_autonomous_plan_succeeded)
        thread.failed.connect(self._telegram_agent_failed)
        thread.finished.connect(self._telegram_agent_thread_finished)
        thread.start()

    def _telegram_autonomous_claim_stage_retry(self, session: Dict[str, Any], phase: str) -> int:
        """Two retries total per stage/chunk, shared across parse/field errors."""
        index = int(session.get("beat_batch_index") or 0) if phase == "autonomous_beats" else int(session.get("batch_index") or 0) if phase == "autonomous_shots" else 0
        key = f"{phase}:{index}"
        counts = dict(session.get("stage_retry_counts") or {})
        legacy_keys = [f"structured_retry_done_{phase}_{index}"]
        if phase == "autonomous_shots":
            legacy_keys.append(f"shot_batch_retry_done_{index}")
        elif phase == "autonomous_blueprint":
            legacy_keys.extend(["story_blueprint_retry_done", "metadata_required_refs_retry_done"])
        used = int(counts.get(key, max([int(session.get(k) or 0) for k in legacy_keys] or [0])))
        if used >= 2:
            return 0
        counts[key] = used + 1
        session["stage_retry_counts"] = counts
        return used + 1

    def _telegram_autonomous_plan_succeeded(self, payload: object) -> None:
        pending = self._telegram_agent_pending
        if not isinstance(pending, dict):
            return
        key = str(pending.get("chat_id") or "")
        session = self._telegram_autonomous_sessions.get(key)
        if not isinstance(session, dict):
            self._telegram_agent_finish(unload=True)
            return
        pending_sid = str(pending.get("session_id") or "").strip()
        current_sid = str(session.get("id") or "").strip()
        if pending_sid and current_sid and pending_sid != current_sid:
            # Never let a delayed callback/pending pass from an older Telegram
            # project operate on the newest project in the same chat.
            self._telegram_agent_finish(unload=True)
            return
        phase = str(pending.get("phase") or "autonomous_blueprint")
        self._telegram_autonomous_write_raw(session, "ref_audit" if phase == "autonomous_ref_audit" else ("beats" if phase == "autonomous_beats" else ("shots" if phase == "autonomous_shots" else "blueprint")), payload)
        obj = self._telegram_agent_parse_json(payload)
        if phase == "autonomous_ref_audit" and session.get("reference_selection_pending"):
            self._telegram_autonomous_finish_reference_selection(key, session, pending, result=obj)
            return
        if not obj and phase == "autonomous_shots":
            obj = self._telegram_autonomous_parse_root_array_payload(payload, "clips")
        elif not obj and phase == "autonomous_beats":
            obj = self._telegram_autonomous_parse_root_array_payload(payload, "beats")

        if not obj:
            session["shot_batch_validation_feedback" if phase == "autonomous_shots" else "beat_batch_validation_feedback"] = "No parseable JSON was returned. Return the exact requested JSON object."
            retry_no = self._telegram_autonomous_claim_stage_retry(session, phase)
            if retry_no:
                session["status"] = "planning_retry"
                self._telegram_autonomous_save(session)
                pending["structured_retry"] = True
                self._telegram_agent_pending = pending
                self._telegram_send_text(key, f"The LLM returned no parseable JSON. Retrying this same stage ({retry_no}/2).")
                return
            session["status"] = "error"
            session["error"] = f"The LLM did not return parseable JSON for {phase}."
            self._telegram_autonomous_save(session)
            self._telegram_send_text(key, f"{session['error']}\nRaw output was saved to {session.get('last_raw_log','')}")
            self._telegram_agent_finish(unload=True)
            return

        pending.pop("structured_retry", None)

        if phase == "autonomous_beats":
            try:
                new_beats = self._telegram_autonomous_parse_beat_batch(obj, session)
            except Exception as exc:
                session["beat_batch_validation_feedback"] = str(exc)
                retry_no = self._telegram_autonomous_claim_stage_retry(session, phase)
                if retry_no:
                    session["status"] = "planning_retry"
                    self._telegram_autonomous_save(session)
                    pending["structured_retry"] = True
                    self._telegram_agent_pending = pending
                    self._telegram_send_text(key, f"Retrying only this story chunk ({retry_no}/2): {exc}")
                    return
                session["status"] = "error"; session["error"] = str(exc); self._telegram_autonomous_save(session)
                self._telegram_send_text(key, f"Story-beat planning stopped: {exc}\nRaw output was saved to {session.get('last_raw_log','')}")
                self._telegram_agent_finish(unload=True); return
            session.pop("beat_batch_validation_feedback", None)
            story=list(session.get("story_slots") or []); story.extend(new_beats); session["story_slots"]=story
            session["beat_batch_index"] = int(session.get("beat_batch_index") or 0) + 1
            total=int(self._telegram_autonomous_duration_contract(str(session.get("request") or ""))["shot_count"])
            if len(story) < total:
                session["status"]="building_story"; self._telegram_autonomous_save(session)
                locked_start = int(new_beats[0].get("slot") or (len(story)-len(new_beats)+1)) if new_beats else max(1, len(story)-4)
                locked_end = int(new_beats[-1].get("slot") or len(story)) if new_beats else len(story)
                self._telegram_send_text(key, f"Locked story beats {locked_start}-{locked_end} ({len(story)}/{total}).")
                pending["phase"]="autonomous_beats"; self._telegram_agent_pending=pending; return
            locked_start = int(new_beats[0].get("slot") or max(1, total-len(new_beats)+1)) if new_beats else max(1, total-4)
            locked_end = int(new_beats[-1].get("slot") or total) if new_beats else total
            self._telegram_send_text(key, f"Locked story beats {locked_start}-{locked_end} ({min(len(story), total)}/{total}).")
            meta=dict(session.get("blueprint_meta") or {})
            issues = self._telegram_autonomous_story_quality_issues(story[:total], meta, total)
            if issues:
                feedback = "; ".join(issues[:4])
                if not bool(session.get("story_quality_retry_done", False)):
                    session["story_quality_retry_done"] = True
                    session["story_quality_feedback"] = feedback
                    session["story_slots"] = []
                    session["beat_batch_index"] = 0
                    session["status"] = "planning_retry"
                    self._telegram_autonomous_save(session)
                    self._telegram_send_text(key, f"Pre-render story check rejected this beat pass before any Krea/MiniMax work: {feedback}. Rebuilding the beats once from the SAME locked blueprint.")
                    pending["phase"] = "autonomous_beats"
                    pending.pop("structured_retry", None)
                    self._telegram_agent_pending = pending
                    return
                session["status"] = "error"
                session["error"] = f"Story beat pass still contains repetition/early-resolution problems after retry: {feedback}"
                self._telegram_autonomous_save(session)
                self._telegram_send_text(key, session["error"])
                self._telegram_agent_finish(unload=True)
                return
            session.pop("story_quality_feedback", None)
            meta["shots"]={str(x["slot"]): {"section":x["section"],"beat":x["beat"],"reference_ids":x.get("reference_ids",[])} for x in story[:total]}
            try:
                blueprint=self._telegram_autonomous_validate_blueprint(meta, str(session.get("request") or ""))
            except Exception as exc:
                session["status"]="error"; session["error"]=str(exc); self._telegram_autonomous_save(session)
                self._telegram_send_text(key, f"Assembled story blueprint failed validation: {exc}"); self._telegram_agent_finish(unload=True); return
            batches=self._telegram_autonomous_build_batches(blueprint)
            session["blueprint"]=blueprint; session["shot_count"]=int(blueprint.get("shot_count") or 0); session["shot_batches"]=batches
            session["batch_index"]=0; session["staged_clips"]=[]; session["status"]="building_shots"
            if str(blueprint.get("video_model") or "") == "minimax_h3":
                session["reference_selection_pending"] = True
                session["status"] = "auditing_references"
                self._telegram_autonomous_write_json(session, "blueprint", blueprint)
                self._telegram_autonomous_save(session)
                self._telegram_send_text(key, f"Story locked: {total} beats. Selecting visual continuity identities before prompt writing.")
                pending["phase"] = "autonomous_ref_audit"
                self._telegram_agent_pending = pending
                return
            self._telegram_autonomous_write_json(session,"blueprint",blueprint); self._telegram_autonomous_save(session)
            self._telegram_send_text(key, f"Story locked: all {total} beats obey the pre-approved blueprint budget and passed the pre-render repetition/early-resolution check. Now expanding them into video prompts.")
            pending["phase"]="autonomous_shots"; self._telegram_agent_pending=pending; return

        if phase == "autonomous_ref_audit":
            added = self._telegram_autonomous_apply_reference_audit(session, obj)
            session["status"] = "building_shots"
            self._telegram_autonomous_save(session)
            if added:
                refs_now = list((session.get("blueprint") or {}).get("references") or [])
                new_names = [str(r.get("name") or r.get("id") or "ref") for r in refs_now[-added:] if isinstance(r, dict)]
                self._telegram_send_text(key, f"Continuity audit added {added} missing recurring ref(s): " + ", ".join(new_names) + ". These refs are now attached to every matching clip before Krea/Ref2VA generation.")
            else:
                self._telegram_send_text(key, "Continuity audit found no additional recurring identities worth a reference sheet.")
            # Finalize the plan now that the reference catalog is complete.
            staged = [dict(x) for x in list(session.get("staged_clips") or []) if isinstance(x, dict)]
            bp = dict(session.get("blueprint") or {})
            refs = [dict(x) for x in list(bp.get("references") or []) if isinstance(x, dict)]
            for ref in refs:
                rid = str(ref.get("id") or "")
                ref["clip_orders"] = [int(c.get("order") or 0) for c in staged if rid and rid in list(c.get("reference_ids") or [])]
            final_plan = {
                "title": bp.get("title"), "target_duration": bp.get("target_duration"),
                "duration_min": bp.get("duration_min"), "duration_max": bp.get("duration_max"),
                "video_model": bp.get("video_model"), "resolution": bp.get("resolution"),
                "aspect": bp.get("aspect"), "story_summary": bp.get("story_summary"),
                "references": refs, "music": bp.get("music"), "clips": staged,
            }
            try:
                plan = self._telegram_autonomous_validate_plan(final_plan, str(session.get("request") or ""))
            except Exception as exc:
                session["status"] = "error"; session["error"] = str(exc); self._telegram_autonomous_save(session)
                self._telegram_send_text(key, f"Final staged plan validation failed after continuity audit: {exc}")
                self._telegram_agent_finish(unload=True); return
            session["plan"] = plan
            self._telegram_autonomous_write_json(session, "blueprint", bp)
            self._telegram_autonomous_write_json(session, "plan", plan)
            if self._telegram_autonomous_prepare_music_preset_choice(key, session):
                self._telegram_autonomous_save(session); self._telegram_agent_finish(unload=True); return
            session["status"] = "queueing"; self._telegram_autonomous_save(session)
            self._telegram_send_text(key, f"Story locked and all {len(staged)} shot prompts are ready with the final continuity refs. Unloading the LLM and starting references/generation now.")
            self._telegram_agent_finish(unload=True)
            QtCore.QTimer.singleShot(250, lambda k=key: self._telegram_autonomous_queue_project(k))
            return

        if phase == "autonomous_shots":
            batches = list(session.get("shot_batches") or [])
            idx = int(session.get("batch_index") or 0)
            if idx >= len(batches):
                session["status"] = "error"
                session["error"] = "Internal staged-shot batch index is out of range."
                self._telegram_autonomous_save(session)
                self._telegram_send_text(key, session["error"])
                self._telegram_agent_finish(unload=True)
                return
            batch = dict(batches[idx])
            try:
                new_clips = self._telegram_autonomous_parse_shot_batch(obj, batch, session=session)
            except Exception as exc:
                session["shot_batch_validation_feedback"] = str(exc)
                retry_no = self._telegram_autonomous_claim_stage_retry(session, phase)
                if retry_no:
                    session["status"] = "planning_retry"
                    self._telegram_autonomous_save(session)
                    pending["structured_retry"] = True
                    pending["phase"] = "autonomous_shots"
                    self._telegram_agent_pending = pending
                    self._telegram_send_text(key, f"Retrying only shots {batch.get('global_start')}-{batch.get('global_end')} ({retry_no}/2): {exc}")
                    return
                session["status"] = "error"
                session["error"] = str(exc)
                self._telegram_autonomous_save(session)
                self._telegram_send_text(key, f"Shot-prompt generation stopped after two retries: {exc}\nRaw output was saved to {session.get('last_raw_log','')}")
                self._telegram_agent_finish(unload=True)
                return

            pending.pop("structured_retry", None)
            session.pop("shot_batch_validation_feedback", None)
            staged = list(session.get("staged_clips") or [])
            staged.extend(new_clips)
            session["staged_clips"] = staged
            session["batch_index"] = idx + 1
            self._telegram_autonomous_save(session)

            if session["batch_index"] < len(batches):
                pending["phase"] = "autonomous_shots"
                self._telegram_agent_pending = pending
                return

            staged.sort(key=lambda c: int(c.get("order") or 0))
            expected_orders = list(range(1, int(session.get("shot_count") or 0) + 1))
            actual_orders = [int(c.get("order") or 0) for c in staged]
            if actual_orders != expected_orders:
                session["status"] = "error"
                session["error"] = f"Internal prompt integrity check failed: got shot orders {actual_orders}, expected {expected_orders}."
                self._telegram_autonomous_save(session)
                self._telegram_send_text(key, session["error"])
                self._telegram_agent_finish(unload=True)
                return
            # Reference identity was already assigned structurally in the beat
            # pass and inherited by each shot. Finalize deterministically here;
            # do not run a second LLM audit over finished prompt prose.
            try:
                plan = self._telegram_autonomous_finalize_staged_plan(session)
            except Exception as exc:
                session["status"] = "error"; session["error"] = str(exc); self._telegram_autonomous_save(session)
                self._telegram_send_text(key, f"Final staged plan validation failed: {exc}")
                self._telegram_agent_finish(unload=True); return
            session["plan"] = plan
            self._telegram_autonomous_write_json(session, "plan", plan)
            if self._telegram_autonomous_prepare_music_preset_choice(key, session):
                self._telegram_autonomous_save(session); self._telegram_agent_finish(unload=True); return
            session["status"] = "queueing"; self._telegram_autonomous_save(session)
            self._telegram_send_text(key, f"Story locked and all {len(staged)} shot prompts are ready. Unloading the LLM and starting references/generation now.")
            self._telegram_agent_finish(unload=True)
            QtCore.QTimer.singleShot(250, lambda k=key: self._telegram_autonomous_queue_project(k))
            return

        try:
            meta = self._telegram_autonomous_normalize_metadata(obj, str(session.get("request") or ""))
        except Exception as exc:
            session["status"] = "error"; session["error"] = str(exc); self._telegram_autonomous_save(session)
            self._telegram_send_text(key, f"Project metadata failed: {exc}\nRaw output was saved to {session.get('last_raw_log','')}")
            self._telegram_agent_finish(unload=True); return
        # The blueprint becomes authoritative here. Fix only clip-count arithmetic
        # deterministically before validation; do not burn a full creative retry just
        # because the LLM summed section counts to 25 instead of the locked 23.
        meta = self._telegram_autonomous_lock_blueprint_clip_budget(meta, str(session.get("request") or ""))
        try:
            self._telegram_autonomous_validate_story_blueprint_meta(meta, str(session.get("request") or ""))
            session.pop("blueprint_validation_feedback", None)
        except Exception as exc:
            # Exact clip-count arithmetic is code-owned above. If that still cannot
            # produce the locked total, retrying the LLM would defeat the point of
            # the blueprint contract, so fail explicitly. Other semantic/structural
            # preflight failures keep the single last-resort blueprint retry.
            if "blueprint clip budget" in str(exc).lower():
                session["status"] = "error"
                session["error"] = f"Blueprint clip budget could not be locked deterministically: {exc}"
                self._telegram_autonomous_save(session)
                self._telegram_send_text(key, session["error"])
                self._telegram_agent_finish(unload=True)
                return
            retry_key = "story_blueprint_retry_done"
            retry_no = self._telegram_autonomous_claim_stage_retry(session, phase)
            if retry_no:
                session[retry_key] = True
                session["blueprint_validation_feedback"] = str(exc)
                session["status"] = "planning_retry"
                self._telegram_autonomous_save(session)
                pending["structured_retry"] = True
                pending["phase"] = "autonomous_blueprint"
                self._telegram_agent_pending = pending
                self._telegram_send_text(key, f"Retrying the blueprint ({retry_no}/2) with this correction: {exc}")
                return
            session["status"] = "error"
            session["error"] = f"Story blueprint is still structurally unsafe after retry: {exc}"
            self._telegram_autonomous_save(session)
            self._telegram_send_text(key, session["error"])
            self._telegram_agent_finish(unload=True)
            return
        if self._telegram_autonomous_user_requested_refs(str(session.get("request") or "")) and not list(meta.get("references") or []):
            retry_key = "metadata_required_refs_retry_done"
            retry_no = self._telegram_autonomous_claim_stage_retry(session, phase)
            if retry_no:
                session[retry_key] = True
                session["status"] = "planning_retry"
                self._telegram_autonomous_save(session)
                pending["structured_retry"] = True
                pending["phase"] = "autonomous_blueprint"
                self._telegram_agent_pending = pending
                self._telegram_send_text(key, f"Metadata omitted the requested references. Retrying metadata ({retry_no}/2) before building story beats.")
                return
            session["status"] = "error"
            session["error"] = "The request explicitly requires Krea/Ref2VA references, but metadata still contained no reusable character/subject references after retry."
            self._telegram_autonomous_save(session)
            self._telegram_send_text(key, session["error"])
            self._telegram_agent_finish(unload=True)
            return
        session["blueprint_meta"] = meta
        session["story_slots"] = []
        session["beat_batch_index"] = 0
        session["status"] = "building_story"
        self._telegram_autonomous_save(session)
        total=int(self._telegram_autonomous_duration_contract(str(session.get("request") or ""))["shot_count"])
        sections = list(meta.get("story_sections") or [])
        budget_text = ", ".join(f"{str(x.get('title') or 'section')}={int(x.get('clip_count') or 0)}" for x in sections)
        self._telegram_send_text(key, f"Story blueprint locked before beat generation: {budget_text}. Now filling the {total} locked slots in small 5-shot chunks.")
        pending["phase"] = "autonomous_beats"
        self._telegram_agent_pending = pending
        return
    @staticmethod
    def _telegram_autonomous_user_requested_refs(request: str) -> bool:
        low = str(request or "").lower()
        return (
            ("krea" in low and any(x in low for x in ("reference", "ref ", "ref2va", "character sheet", "character sheets")))
            or "use ref2va" in low
            or "using ref2va" in low
        )

    @staticmethod
    def _telegram_autonomous_reference_asset_kind(ref: Dict[str, Any]) -> str:
        kind = str(ref.get("type") or "character").strip().lower()
        if kind in {"character", "person", "creature", "alien", "animal"}:
            return "character"
        if kind in {"vehicle", "car", "ship", "spaceship", "aircraft", "bike"}:
            return "vehicle"
        if kind in {"object", "prop", "item", "device", "artifact"}:
            return "object"
        if kind in {"location", "scene", "environment", "place", "setting", "background"}:
            return "location"
        return kind or "character"

    def _telegram_autonomous_reference_asset_prompt(self, ref: Dict[str, Any], index: int = 1) -> str:
        rid = re.sub(r"[^0-9A-Za-z_-]+", "_", str(ref.get("id") or f"ref_{index}").strip()).strip("_") or f"ref_{index}"
        name = str(ref.get("name") or rid).strip() or rid
        base_prompt = str(ref.get("prompt") or "").strip()
        kind = self._telegram_autonomous_reference_asset_kind(ref)
        role = str(ref.get("reference_role") or "").strip().lower()
        lead = f"Reference sheet [{rid}] for {name}. "
        if role == "character_identity" or kind == "character":
            return (
                lead
                + f"{base_prompt}. "
                + "Show ONE exact recurring character only as a strict 3-panel labeled reference sheet. Use exactly these three distinct labeled views and nothing else: Panel 1 labeled Front Full, showing a front-facing full-body view; Panel 2 labeled Front 3/4, showing a front three-quarter full-body view; Panel 3 labeled Face Close-Up, showing a clear face close-up. "
                + "Do not duplicate angles. Do not replace the three-quarter view with another front view. Do not add a side/profile panel. Keep the same face, species, proportions, colors, clothing and distinctive features in every panel. Neutral uncluttered studio background. "
                + "No alternate redesigns, no extra characters, no face paint, no tattoos, no markings unless they are explicitly described in the prompt, and include only the simple panel labels Front Full, Front 3/4, and Face Close-Up."
            )
        if kind == "vehicle":
            return (
                lead
                + f"{base_prompt}. "
                + "Show ONE exact recurring vehicle only as a clean design reference sheet. Include front view, side view, three-quarter view, and rear or top detail view when useful. "
                + "Keep the same body shape, scale, materials, colors and distinctive design details in every view. Neutral uncluttered studio background. "
                + "Do not turn the vehicle into a person or character. No driver, no passengers, no extra vehicles, and no text labels."
            )
        if kind == "object":
            return (
                lead
                + f"{base_prompt}. "
                + "Show ONE exact recurring object or prop only as a clean design reference sheet. Include multiple consistent views and one close detail view. "
                + "Keep the same shape, materials, colors and distinctive details in every view. Neutral uncluttered studio background. "
                + "No hands holding it, no extra objects, no character body parts, and no text labels."
            )
        if role == "reusable_set":
            return (
                lead
                + f"{base_prompt}. "
                + "Create a continuity board for ONE exact recurring architectural set/location, not a character sheet. Show 3-4 consistent exterior/interior angles only where they belong to the same place, plus one defining detail view. "
                + "Keep architecture, facade, windows, doors, materials, landscaping, layout and distinctive features identical across views. "
                + "No unrelated locations, no alternate redesigns, no character turnarounds, and no text labels."
            )
        if kind == "location":
            return (
                lead
                + f"{base_prompt}. "
                + "Create an environment reference board for one recurring location. Show 3-4 consistent angles of the same place and one closer detail crop that helps continuity. "
                + "Keep architecture, layout, palette and key set details consistent across views. Neutral presentation background if needed. "
                + "No character turnarounds, no unrelated places, and no text labels."
            )
        return (
            lead
            + f"{base_prompt}. "
            + "Create a clean multi-view reference sheet for one recurring subject only. Keep the subject consistent across all views. Neutral uncluttered studio background. "
            + "No extra unrelated subjects and no text labels."
        )

    def _telegram_autonomous_queue_reference_assets(self, key: str, session: Dict[str, Any], router) -> tuple[bool, str]:
        self._telegram_autonomous_assign_reference_labels(session)
        plan = dict(session.get("plan") or {})
        refs = [dict(x) for x in list(plan.get("references") or []) if isinstance(x, dict)]
        request_low = str(session.get("request") or "").lower()
        # Character identity sheets and reusable subject/set boards are different
        # products. A request mentioning "character sheets" must never turn a
        # BMW, mansion or prop into a pseudo-character turnaround; however, those
        # recurring assets may still get their own correctly-typed continuity
        # board for Ref2VA. Generic roads/city/coast backgrounds remain prompt-only.
        wants_location_refs = any(x in request_low for x in ("location reference", "environment reference", "background reference", "set reference"))
        selected = []
        for ref in refs:
            role = str(ref.get("reference_role") or "").strip().lower()
            kind = self._telegram_autonomous_reference_asset_kind(ref)
            if role in {"character_identity", "reusable_asset", "reusable_set"}:
                selected.append(ref)
            elif wants_location_refs and kind == "location":
                selected.append(ref)
        refs = selected
        if not refs:
            return False, "The request requires Krea/Ref2VA reference sheets, but the Agent plan did not define any reusable references."

        assets = []
        sid = str(session.get("id") or "agent")
        seen_asset_paths = set()
        for n, ref in enumerate(refs, start=1):
            rid = str(ref.get("id") or f"ref_{n}")
            name = str(ref.get("name") or rid).strip()
            base_prompt = str(ref.get("prompt") or "").strip()
            if not base_prompt:
                return False, f"Reference `{name}` has no Krea prompt."

            kind = self._telegram_autonomous_reference_asset_kind(ref)
            krea_prompt = self._telegram_autonomous_reference_asset_prompt(ref, n)
            try:
                route = router._queue_image(krea_prompt, "krea2", 1920, 1088)
            except Exception as exc:
                return False, f"Krea 2 reference `{name}` could not be queued: {exc}"
            if not bool(getattr(route, "queued", False)):
                return False, f"Krea 2 reference `{name}` could not be queued: {getattr(route, 'message', 'queue failed')}"
            output_path = str(getattr(route, "output_path", "") or "").strip()
            if not output_path:
                return False, f"Krea 2 queued `{name}` but did not return an output path."
            if output_path in seen_asset_paths:
                return False, f"Krea 2 planned the same output file twice for reference `{name}` ({output_path})."
            seen_asset_paths.add(output_path)
            assets.append({
                "id": rid,
                "name": name,
                "type": kind,
                "reference_role": str(ref.get("reference_role") or ""),
                "user_label": str(ref.get("user_label") or ""),
                "global_ref_label": str(ref.get("global_ref_label") or ""),
                "prompt": krea_prompt,
                "appearance_prompt": base_prompt,
                "clip_orders": list(ref.get("clip_orders") or []),
                "path": output_path,
                "queued_at": float(getattr(route, "queued_at", 0.0) or time.time()),
            })

        session["reference_assets"] = assets
        session["status"] = "generating_refs"
        self._telegram_autonomous_save(session)
        names_text = ", ".join(
            f"{str(a.get('name') or a.get('id') or 'ref')} [{str(a.get('type') or 'reference')}]"
            for a in assets[:6]
        )
        if len(assets) > 6:
            names_text += ", ..."
        self._telegram_send_text(
            key,
            f"Queued {len(assets)} Krea 2 reference sheet(s) first ({names_text}). I’ll wait for them to finish, "
            "then every MiniMax clip that uses those recurring subjects will be sent through Ref2VA with the matching saved reference image."
        )
        return True, ""

    def _telegram_autonomous_clip_mentions_reference(self, clip: Dict[str, Any], ref: Dict[str, Any]) -> bool:
        """Return True only when the clip text actually calls for this ref."""
        text = re.sub(r"[^a-z0-9]+", " ", (str(clip.get("purpose") or "") + " " + str(clip.get("prompt") or "")).lower()).strip()
        if not text:
            return False
        words = set(text.split())
        name = re.sub(r"[^a-z0-9]+", " ", str(ref.get("name") or "").lower()).strip()
        prompt = re.sub(r"[^a-z0-9]+", " ", str(ref.get("prompt") or "").lower()).strip()
        kind = str(ref.get("type") or "").strip().lower()

        if name and len(name.split()) >= 2 and name in text:
            return True

        generic = {
            "the", "a", "an", "and", "with", "from", "into", "onto", "same", "exact",
            "character", "reference", "sheet", "ref", "front", "close", "up", "view",
            "person", "people", "human", "man", "woman", "boy", "girl", "child", "kid",
            "driver", "protagonist", "customer", "shop",
            "vehicle", "car", "cars", "sedan", "convertible", "coupe", "sports", "sport",
            "cruiser", "police", "ship", "spaceship", "pod", "bike",
            "object", "prop", "item", "device", "thing", "asset",
            "location", "place", "room", "house", "mansion", "city", "coast", "road", "highway",
            "dog", "cat", "animal", "creature", "alien", "robot", "android",
            "desk", "contract", "key", "fob",
            "red", "blue", "green", "yellow", "black", "white", "silver", "grey", "gray",
            "orange", "pink", "purple", "brown", "blonde", "little", "small"
        }

        tokens = []
        for tok in name.split():
            if len(tok) >= 3 and tok not in generic and tok not in tokens:
                tokens.append(tok)

        short_role_phrases = []
        if kind in {"character", "person", "creature", "animal", "alien"}:
            for phrase in ("salesman", "the salesman", "the child", "little girl", "the girl", "the boy", "golden retriever", "the retriever", "bmw m2"):
                if phrase in name or phrase in prompt:
                    short_role_phrases.append(phrase)
        elif kind in {"vehicle", "car", "ship", "spaceship", "object", "prop", "item"}:
            for phrase in ("bmw m2", "police cruiser", "spaceship", "the pod"):
                if phrase in name or phrase in prompt:
                    short_role_phrases.append(phrase)
        for phrase in short_role_phrases:
            if phrase and phrase in text:
                return True

        hits = [tok for tok in tokens if tok in words]
        return len(hits) >= 2

    def _telegram_autonomous_reference_paths_for_clip(self, session: Dict[str, Any], clip: Dict[str, Any]) -> list[dict]:
        wanted = {str(x) for x in list(clip.get("reference_ids") or []) if str(x).strip()}
        # A clip-level reference selection is authoritative, including an
        # explicitly empty selection.  asset.clip_orders is legacy/fallback
        # metadata describing the original plan; treating it as a union made
        # the review picker unable to remove a reference from a redo.
        has_clip_selection = "reference_ids" in clip
        order = int(clip.get("order") or 0)
        out = []
        for asset in list(session.get("reference_assets") or []):
            if not isinstance(asset, dict):
                continue
            rid = str(asset.get("id") or "")
            orders = set()
            for x in list(asset.get("clip_orders") or []):
                try:
                    orders.add(int(x))
                except Exception:
                    pass
            if has_clip_selection:
                if rid not in wanted:
                    continue
            elif not order or order not in orders:
                continue
            path = str(asset.get("path") or "").strip()
            if path:
                out.append(asset)
        return out[:9]

    def _telegram_autonomous_minimax_h3_prompt(self, clip: Dict[str, Any], assets: Optional[list[dict]] = None) -> str:
        """Build a MiniMax H3 / Ref2VA generation prompt at queue time.

        The planning LLM is allowed to keep writing ordinary semantic shot prose.
        This adapter converts that prose into the MiniMax contract only when the
        selected video model is MiniMax H3.  Reference numbering is assigned here
        from the *actual* reference_image_paths order, so <Picture N>/<Subject N>
        can never drift away from the images sent to Ref2VA.
        """
        selected = [dict(x) for x in list(assets or []) if isinstance(x, dict)][:9]
        base_prompt = str(clip.get("prompt") or "").strip()
        purpose = str(clip.get("purpose") or "").strip()
        try:
            duration = max(0.1, float(clip.get("duration") or 6.0))
        except Exception:
            duration = 6.0

        # Do not wrap twice when a replacement prompt was already authored in
        # the new MiniMax format.  This also makes manual /redo prompt overrides
        # safe for advanced users.
        upper = base_prompt.upper()
        if "SCENE:" in upper and "AUDIO:" in upper and "CAMERA:" in upper and re.search(r"\[00:0?0(?:\.0+)?-", base_prompt):
            return base_prompt

        blocks = []
        if selected:
            if len(selected) == 1:
                ref_lines = [
                    "REFERENCE IMAGE:",
                    "",
                    "Use the single provided reference image for <Subject 1> and preserve the referenced subject's exact visual identity. "
                    "Use the image for identity/appearance continuity only; do not copy its reference-sheet background into the target scene.",
                ]
            else:
                pairs = ", ".join(f"<Picture {n}> for <Subject {n}>" for n in range(1, len(selected) + 1))
                ref_lines = [
                    "REFERENCE IMAGES:",
                    "",
                    f"Use the provided reference images as follows: {pairs}. Preserve each referenced subject's exact visual identity. "
                    "Use the images for identity/appearance continuity only; do not copy their reference-sheet backgrounds into the target scene.",
                ]
            blocks.append("\n".join(ref_lines))

            defs = ["SUBJECT DEFINITIONS:", ""]
            for n, asset in enumerate(selected, start=1):
                name = str(asset.get("name") or f"Referenced subject {n}").strip()
                appearance = str(asset.get("appearance_prompt") or asset.get("source_prompt") or "").strip()
                # Older saved sessions do not have appearance_prompt.  Avoid
                # leaking Krea board/layout instructions from asset['prompt'];
                # the actual reference image is authoritative in that case.
                defs.append(f"<Subject {n}> — {name.upper()}")
                defs.append("")
                if appearance:
                    defs.append(appearance)
                else:
                    defs.append(f"Use the exact canonical appearance of {name} from <Picture {n}>.")
                defs.append(f"Preserve the exact identity, proportions, colors, materials, clothing and distinctive visible features from <Picture {n}> throughout the entire video.")
                defs.append("")
            blocks.append("\n".join(defs).rstrip())

        # The semantic shot prose remains authoritative for scene content.  It is
        # placed in the timed action block instead of being rewritten with brittle
        # heuristics that could change the story or assign dialogue to the wrong
        # character.
        scene_intro = purpose or "Follow the described shot exactly with realistic spatial and temporal continuity."
        blocks.append(
            "SCENE:\n\n"
            + scene_intro
            + "\nMaintain the environment, subject identity, lighting and physical continuity described in the timed action block below."
        )

        end_stamp = f"{duration:04.1f}"
        blocks.append(
            f"[00:00.0-00:{end_stamp}]\n\n"
            + base_prompt
        )

        blocks.append(
            "AUDIO:\n\n"
            "Use natural diegetic ambience and synchronized physical sound effects implied by the action. "
            "Any spoken dialogue explicitly present in the timed action must remain clear, natural and intelligible. "
            "Do not invent dialogue, narration or background music unless the shot explicitly requests it."
        )
        blocks.append(
            "CAMERA:\n\n"
            "Follow the camera framing, lens feel and movement described in the timed action. Keep the clip as one physically continuous take unless the prompt explicitly requires otherwise. "
            "Do not introduce unrelated cutaways or viewpoints. Keep all referenced subjects visually consistent throughout the entire video."
        )
        return "\n\n".join(x for x in blocks if str(x).strip()).strip()

    def _telegram_autonomous_full_reference_prompt(self, clip: Dict[str, Any], assets: list[dict]) -> str:
        """Backward-compatible alias for the MiniMax H3 / Ref2VA formatter."""
        return self._telegram_autonomous_minimax_h3_prompt(clip, assets)

    def _telegram_autonomous_assign_reference_labels(self, session: Dict[str, Any]) -> list[Dict[str, Any]]:
        """Assign stable, human-facing labels such as Vehicle Ref 1.

        Internal IDs remain untouched. Labels are saved into the project so they
        survive restart and never depend on whatever name the planning LLM chose.
        """
        plan = dict(session.get("plan") or {})
        refs = [x for x in list(plan.get("references") or []) if isinstance(x, dict)]
        counters = {"character": 0, "vehicle": 0, "location": 0, "object": 0, "reference": 0}
        for global_index, ref in enumerate(refs, start=1):
            kind = self._telegram_autonomous_reference_asset_kind(ref)
            role = str(ref.get("reference_role") or "").strip().lower()
            if role == "reusable_set":
                bucket, label_root = "location", "Set Ref"
            elif kind == "character":
                bucket, label_root = "character", "Character Ref"
            elif kind == "vehicle":
                bucket, label_root = "vehicle", "Vehicle Ref"
            elif kind == "location":
                bucket, label_root = "location", "Location Ref"
            elif kind == "object":
                bucket, label_root = "object", "Object Ref"
            else:
                bucket, label_root = "reference", "Ref"
            counters[bucket] += 1
            # Preserve an already saved label; only assign on first encounter.
            ref.setdefault("user_label", f"{label_root} {counters[bucket]}")
            ref.setdefault("global_ref_label", f"Ref {global_index}")
        plan["references"] = refs
        session["plan"] = plan
        # Mirror labels onto generated reference assets.
        by_id = {str(r.get("id") or ""): r for r in refs}
        for asset in list(session.get("reference_assets") or []):
            if not isinstance(asset, dict):
                continue
            ref = by_id.get(str(asset.get("id") or ""))
            if ref:
                asset["user_label"] = str(ref.get("user_label") or "")
                asset["global_ref_label"] = str(ref.get("global_ref_label") or "")
        self._telegram_autonomous_save(session)
        return refs

    def _telegram_autonomous_reference_catalog_text(self, session: Dict[str, Any]) -> str:
        refs = self._telegram_autonomous_assign_reference_labels(session)
        assets = {str(x.get("id") or ""): x for x in list(session.get("reference_assets") or []) if isinstance(x, dict)}
        if not refs:
            return "This project has no reusable references."
        lines = ["Project references:"]
        for ref in refs:
            rid = str(ref.get("id") or "")
            label = str(ref.get("user_label") or ref.get("global_ref_label") or rid)
            global_label = str(ref.get("global_ref_label") or "")
            name = str(ref.get("name") or rid)
            role = str(ref.get("reference_role") or self._telegram_autonomous_reference_asset_kind(ref))
            asset = assets.get(rid)
            ready = bool(asset and str(asset.get("path") or "").strip() and self._telegram_autonomous_file_ready(Path(str(asset.get("path"))), 4096))
            alias = f" / {global_label}" if global_label and global_label != label else ""
            lines.append(f"• {label}{alias} — {name} ({role}) — {'ready' if ready else 'not ready'}")
        lines.append("Use for example: redo 7 with Vehicle Ref 1")
        return "\n".join(lines)

    def _telegram_autonomous_resolve_reference_labels(self, session: Dict[str, Any], text: str) -> tuple[list[str], list[str]]:
        """Resolve positively selected user-facing labels in free text.

        Labels that are explicitly negated (for example `do not use Vehicle Ref 1`)
        are intentionally excluded here. Use
        `_telegram_autonomous_reference_selection_from_text()` when the caller also
        needs the excluded IDs.
        """
        include_ids, _exclude_ids, include_labels, _exclude_labels, _mentioned = \
            self._telegram_autonomous_reference_selection_from_text(session, text)
        return include_ids, include_labels

    def _telegram_autonomous_reference_selection_from_text(self, session: Dict[str, Any], text: str) -> tuple[list[str], list[str], list[str], list[str], bool]:
        """Return explicit reference include/exclude selections from Telegram text.

        This makes commands such as `do not use Vehicle Ref 1` structural rather
        than merely prose inside the MiniMax prompt. A negated label is removed
        from the effective clip reference set instead of accidentally being added.
        """
        refs = self._telegram_autonomous_assign_reference_labels(session)
        raw = str(text or "")
        low = raw.lower()
        include_ids, exclude_ids = [], []
        include_labels, exclude_labels = [], []
        aliases = []
        type_counts = {}
        for ref in refs:
            kind = str(ref.get("type") or "").strip().lower()
            if kind:
                type_counts[kind] = int(type_counts.get(kind) or 0) + 1
        for ref in refs:
            rid = str(ref.get("id") or "")
            label = str(ref.get("user_label") or "").strip()
            global_label = str(ref.get("global_ref_label") or "").strip()
            kind = str(ref.get("type") or "").strip().lower()
            candidates = [label, global_label]
            if label.lower().startswith("character ref "):
                candidates.append("Character Sheet " + label.split()[-1])
            # Natural-language aliases are safe when there is only one reusable
            # reference of that type in the project. This lets commands such as
            # `do not use the reference of the vehicle` actually alter Ref2VA refs.
            if kind and int(type_counts.get(kind) or 0) == 1:
                candidates.extend([
                    f"{kind} reference", f"{kind} ref", f"reference of the {kind}",
                    f"reference image of the {kind}", f"the {kind} reference",
                ])
            for alias in candidates:
                if alias:
                    aliases.append((alias, rid, label or global_label))

        occupied = []
        mentioned = False
        for alias, rid, display in sorted(aliases, key=lambda x: len(x[0]), reverse=True):
            pattern = r"(?<![A-Za-z0-9])" + re.escape(alias.lower()) + r"(?![A-Za-z0-9])"
            for m in re.finditer(pattern, low):
                span = m.span()
                if any(not (span[1] <= a or span[0] >= b) for a, b in occupied):
                    continue
                occupied.append(span)
                mentioned = True
                # Look immediately before the label for explicit exclusion wording.
                before = low[max(0, span[0] - 48):span[0]]
                negated = bool(re.search(
                    r"(?:do\s+not|don't|dont)\s+(?:use|include|attach)\s+(?:the\s+)?$|"
                    r"(?:without|exclude|remove|omit|skip|no)\s+(?:the\s+)?$",
                    before, flags=re.I,
                ))
                if negated:
                    if rid and rid not in exclude_ids:
                        exclude_ids.append(rid)
                        exclude_labels.append(display)
                    if rid in include_ids:
                        idx = include_ids.index(rid)
                        include_ids.pop(idx)
                        include_labels.pop(idx)
                elif rid and rid not in exclude_ids and rid not in include_ids:
                    include_ids.append(rid)
                    include_labels.append(display)
        return include_ids, exclude_ids, include_labels, exclude_labels, mentioned

    def _telegram_autonomous_clip_text(self, session: Dict[str, Any], order: int) -> str:
        plan = dict(session.get("plan") or {})
        clips = [x for x in list(plan.get("clips") or []) if isinstance(x, dict)]
        if order < 1 or order > len(clips):
            return f"Clip {order} does not exist. This project has {len(clips)} clips."
        clip = clips[order - 1]
        refs = self._telegram_autonomous_assign_reference_labels(session)
        labels_by_id = {str(x.get("id") or ""): str(x.get("user_label") or x.get("global_ref_label") or x.get("id") or "") for x in refs}
        labels = [labels_by_id.get(str(rid), str(rid)) for rid in list(clip.get("reference_ids") or [])]
        out = [
            f"Clip {order}",
            f"Purpose: {str(clip.get('purpose') or '').strip()}",
            "References: " + (", ".join(labels) if labels else "none"),
            "Prompt:",
            str(clip.get("prompt") or "").strip(),
        ]
        current = list(session.get("clip_outputs") or [])
        known_seed = int(clip.get("last_generation_seed") or clip.get("last_redo_seed") or clip.get("seed") or 0)
        out.append("Seed: " + (str(known_seed) if known_seed > 0 else "not recorded for the original generation"))
        if order <= len(current) and str(current[order - 1] or "").strip():
            out.append("Current output: " + str(current[order - 1]))
        return "\n".join(out)

    def _telegram_autonomous_queue_single_redo(self, key: str, session: Dict[str, Any], order: int, instruction: str, explicit_ref_ids: list[str], requested_seed: Optional[int] = None, same_seed: bool = False, replacement_prompt: Optional[str] = None) -> tuple[bool, str]:
        """Queue one replacement clip and atomically make it the assembly source."""
        plan = dict(session.get("plan") or {})
        clips = [x for x in list(plan.get("clips") or []) if isinstance(x, dict)]
        if order < 1 or order > len(clips):
            return False, f"Clip {order} does not exist. This project has {len(clips)} clips."
        model = str(plan.get("video_model") or "minimax_h3")
        clip = clips[order - 1]
        if explicit_ref_ids:
            clip["reference_ids"] = list(dict.fromkeys(str(x) for x in explicit_ref_ids if str(x).strip()))
        # Never let a redo bypass the hard reference gate.
        if model == "minimax_h3":
            selected_assets = self._telegram_autonomous_reference_paths_for_clip(session, clip)
            got = {str(x.get("id") or "") for x in selected_assets}

            # Only refs explicitly named by the Telegram user are hard requirements.
            # Older plans can contain prompt-only/transient IDs (for example
            # location_highway) in clip.reference_ids even though no Krea asset was
            # ever supposed to exist for them. A normal /redo must not fail on
            # those stale/non-materialized IDs; it simply reuses the actual ready
            # Ref2VA assets assigned to this slot.
            if explicit_ref_ids:
                explicitly_wanted = {str(x) for x in explicit_ref_ids if str(x).strip()}
                missing = sorted(explicitly_wanted - got)
                if missing:
                    return False, "The requested reference(s) are not available for Ref2VA: " + ", ".join(missing)

            for asset in selected_assets:
                p = Path(str(asset.get("path") or ""))
                if not self._telegram_autonomous_file_ready(p, 4096):
                    return False, f"Reference file is not ready on disk: {p}"
        else:
            selected_assets = []

        width, height, res_key = self._telegram_autonomous_resolution(model, str(plan.get("resolution") or ""), str(plan.get("aspect") or "16:9"))
        aspect = str(plan.get("aspect") or "16:9")
        duration = float(clip.get("duration") or 6.0)
        if model == "minimax_h3":
            wanted_frames = max(124, int(round(duration * 24.0)))
            candidates = list(range(124, 720, 17))
            frames = min(candidates, key=lambda x: abs(x - wanted_frames))
        else:
            wanted_frames = max(9, int(round(duration * 24.0)))
            frames = max(9, int(round((wanted_frames - 1) / 8.0)) * 8 + 1)

        original_prompt = str(clip.get("base_prompt") or clip.get("prompt") or "").strip()
        clip.setdefault("base_prompt", original_prompt)
        correction = str(instruction or "").strip(" ,:;-\n\t")
        rewritten_prompt = str(replacement_prompt or "").strip()
        prompt = rewritten_prompt or original_prompt
        if correction and not rewritten_prompt:
            prompt += "\n\nREGENERATION CORRECTION FROM USER: " + correction
        if model == "minimax_h3":
            prompt_clip = dict(clip)
            prompt_clip["prompt"] = prompt
            prompt = self._telegram_autonomous_minimax_h3_prompt(prompt_clip, selected_assets)

        # Redos default to a fresh random seed. The user can request an exact
        # seed or ask to reuse the last seed FrameVision recorded for this slot.
        # This is handled locally; no LLM is needed for seed control.
        if same_seed:
            known_seed = int(clip.get("last_generation_seed") or clip.get("last_redo_seed") or clip.get("seed") or 0)
            if known_seed <= 0:
                return False, (
                    "The original seed for this clip was not recorded by this project version. "
                    "Use `redo %d seed 123456` for an exact seed, or plain `redo %d` for a new random seed."
                    % (order, order)
                )
            generation_seed = known_seed
        elif requested_seed is not None:
            generation_seed = max(1, min(2147483647, int(requested_seed)))
        else:
            generation_seed = random.randint(1, 2147483647)

        counts = dict(session.get("clip_redo_counts") or {})
        redo_n = int(counts.get(str(order)) or 0) + 1
        counts[str(order)] = redo_n
        sid = str(session.get("id") or "agent")
        state = {
            "video_mode": "reference" if selected_assets else "text",
            "reference_image_paths": [str(x.get("path") or "") for x in selected_assets],
            "prompt": prompt,
            "resolution_key": res_key,
            "aspect_key": aspect,
            "width": width,
            "height": height,
            "frames": int(frames),
            "fps": 24,
            "duration_sec": float(frames) / 24.0,
            "seed": int(generation_seed),
            "output_name": f"{sid}_S{order:02d}_redo{redo_n:02d}",
        }
        router = self._telegram_router_for_chat(key)
        try:
            if model == "ltx25":
                route = router._queue_ltx25_from_state(state)
            elif model == "ltx23":
                route = router._queue_ltx_video_from_state(state)
            else:
                route = router._queue_minimax_h3_from_state(state)
        except Exception as exc:
            return False, str(exc)
        if not bool(getattr(route, "queued", False)):
            return False, str(getattr(route, "message", "queue failed"))
        output_path = str(getattr(route, "output_path", "") or "").strip()
        if not output_path:
            return False, "The replacement queued but returned no output path."

        outputs = list(session.get("clip_outputs") or [])
        while len(outputs) < len(clips):
            outputs.append("")
        old_path = str(outputs[order - 1] or "")
        outputs[order - 1] = output_path
        queued = list(session.get("clip_queued_at") or [])
        while len(queued) < len(clips):
            queued.append(0.0)
        queued[order - 1] = float(getattr(route, "queued_at", 0.0) or time.time())
        superseded = list(session.get("superseded_clips") or [])
        if old_path:
            superseded.append({"order": order, "path": old_path, "replaced_at": time.time(), "redo": redo_n})
        clip["last_redo_instruction"] = correction
        if rewritten_prompt:
            history = list(clip.get("prompt_history") or [])
            previous_prompt = str(clip.get("prompt") or original_prompt).strip()
            if previous_prompt:
                history.append({"prompt": previous_prompt, "replaced_at": time.time(), "reason": correction or "Telegram change prompt"})
            clip["prompt_history"] = history[-20:]
            clip["prompt"] = rewritten_prompt
            clip["base_prompt"] = rewritten_prompt
            clip["last_prompt_rewrite"] = rewritten_prompt
        clip["redo_count"] = redo_n
        clip["last_redo_seed"] = int(generation_seed)
        clip["last_generation_seed"] = int(generation_seed)
        clips[order - 1] = clip
        plan["clips"] = clips
        session["plan"] = plan
        session["clip_outputs"] = outputs
        session["clip_queued_at"] = queued
        session["clip_redo_counts"] = counts
        session["superseded_clips"] = superseded
        pending = dict(session.get("pending_redos") or {})
        pending[str(order)] = {"path": output_path, "redo": redo_n, "queued_at": queued[order - 1], "seed": int(generation_seed)}
        session["pending_redos"] = pending
        # If an assembly is already running, let it finish. The replacement sits
        # at the end of the normal worker queue and marks the final video dirty;
        # once the replacement is ready FrameVision automatically assembles a new
        # revision. A completed project becomes active again for the same reason.
        previous_status = str(session.get("status") or "")
        if previous_status == "assembling":
            session["assembly_dirty"] = True
        else:
            session["status"] = "generating"
            if previous_status == "done":
                session["assembly_dirty"] = True
        self._telegram_autonomous_save(session)
        labels = [str(x.get("user_label") or x.get("global_ref_label") or x.get("name") or "ref") for x in selected_assets]
        return True, f"Replacement clip {order} queued as redo {redo_n}" + (f" using {', '.join(labels)}" if labels else "") + f" with seed {generation_seed}. It will run at the end of the current queue and automatically become the assembly source when ready."

    def _telegram_autonomous_prompt_rewrite_queue_busy(self) -> bool:
        """Return True while FrameVision has running or pending generation work.

        Prompt rewrites need the large local LLM. Do not let it race the worker for
        VRAM, and do not load it while queued generation could start underneath it.
        """
        try:
            if self._running_queue_job_files():
                return True
        except Exception:
            pass
        pending_dir = Path(self.fv_root) / "jobs" / "pending"
        try:
            if pending_dir.is_dir():
                for path in pending_dir.iterdir():
                    if not path.is_file():
                        continue
                    try:
                        if path.stat().st_size > 1536:
                            return True
                    except OSError:
                        continue
        except Exception:
            pass
        return False

    def _telegram_autonomous_enqueue_prompt_rewrite(self, key: str, session: Dict[str, Any], order: int, correction: str, explicit_ref_ids: list[str], requested_seed: Optional[int] = None, same_seed: bool = False) -> tuple[bool, str]:
        """Persist a prompt-change request so Telegram corrections are never lost."""
        plan = dict(session.get("plan") or {})
        clips = [x for x in list(plan.get("clips") or []) if isinstance(x, dict)]
        if order < 1 or order > len(clips):
            return False, f"Clip {order} does not exist. This project has {len(clips)} clips."
        correction = str(correction or "").strip()
        if not correction:
            return False, "Tell me what should change in the prompt."
        queue = [x for x in list(session.get("pending_prompt_rewrites") or []) if isinstance(x, dict)]
        queue.append({
            "order": int(order),
            "correction": correction,
            "ref_ids": [str(x) for x in list(explicit_ref_ids or []) if str(x).strip()],
            "requested_seed": requested_seed,
            "same_seed": bool(same_seed),
            "queued_at": time.time(),
        })
        session["pending_prompt_rewrites"] = queue
        self._telegram_autonomous_save(session)
        return True, f"Prompt change for clip {order} queued ({len(queue)} waiting). I will rewrite it with the local LLM when FrameVision's generation queue is idle, then add the corrected clip to the video queue."

    def _telegram_autonomous_try_start_queued_prompt_rewrite(self, key: str, session: Dict[str, Any]) -> bool:
        """Start the oldest persisted prompt rewrite once LLM/worker resources are free."""
        queue = [x for x in list(session.get("pending_prompt_rewrites") or []) if isinstance(x, dict)]
        if not queue:
            return False
        if self._telegram_agent_pending is not None:
            return False
        if self._telegram_agent_thread is not None and self._telegram_agent_thread.isRunning():
            return False
        if getattr(self, "chat_thread", None) is not None and self.chat_thread.isRunning():
            return False
        if self._telegram_autonomous_prompt_rewrite_queue_busy():
            return False
        item = dict(queue[0])
        ok, info = self._telegram_autonomous_start_redo_prompt_rewrite(
            key, session, int(item.get("order") or 0), str(item.get("correction") or ""),
            [str(x) for x in list(item.get("ref_ids") or []) if str(x).strip()],
            requested_seed=item.get("requested_seed"), same_seed=bool(item.get("same_seed", False)),
            _from_queue=True,
        )
        if ok:
            session["pending_prompt_rewrites"] = queue[1:]
            session.pop("prompt_rewrite_queue_last_error_at", None)
            self._telegram_autonomous_save(session)
            return True
        now = time.time()
        last = float(session.get("prompt_rewrite_queue_last_error_at") or 0.0)
        if now - last > 60.0:
            session["prompt_rewrite_queue_last_error_at"] = now
            self._telegram_autonomous_save(session)
            self._telegram_send_text(key, "Queued prompt change is still waiting: " + info)
        return False

    def _telegram_autonomous_start_redo_prompt_rewrite(self, key: str, session: Dict[str, Any], order: int, correction: str, explicit_ref_ids: list[str], requested_seed: Optional[int] = None, same_seed: bool = False, _from_queue: bool = False) -> tuple[bool, str]:
        """Use the local LLM to rewrite one clip prompt, then queue that redo.

        This is deliberately narrow: it preserves the original shot idea and only
        repairs the prompt according to the user's correction. It does not re-plan
        the story, reassign other clips, or add any global reference logic.
        """
        plan = dict(session.get("plan") or {})
        clips = [x for x in list(plan.get("clips") or []) if isinstance(x, dict)]
        if order < 1 or order > len(clips):
            return False, f"Clip {order} does not exist. This project has {len(clips)} clips."
        if self._telegram_agent_thread is not None and self._telegram_agent_thread.isRunning():
            if not _from_queue:
                return self._telegram_autonomous_enqueue_prompt_rewrite(key, session, order, correction, explicit_ref_ids, requested_seed, same_seed)
            return False, "The local LLM is already busy with another Agent request."
        if getattr(self, "chat_thread", None) is not None and self.chat_thread.isRunning():
            if not _from_queue:
                return self._telegram_autonomous_enqueue_prompt_rewrite(key, session, order, correction, explicit_ref_ids, requested_seed, same_seed)
            return False, "The local LLM is currently answering the desktop chat."
        if self._telegram_autonomous_prompt_rewrite_queue_busy():
            if not _from_queue:
                return self._telegram_autonomous_enqueue_prompt_rewrite(key, session, order, correction, explicit_ref_ids, requested_seed, same_seed)
            return False, "FrameVision still has generation work running or waiting in the queue."
        try:
            self._validate_runner_and_model()
        except Exception as exc:
            return False, f"The local LLM is not ready to rewrite the prompt: {exc}"

        clip = clips[order - 1]
        original_prompt = str(clip.get("prompt") or clip.get("base_prompt") or "").strip()
        purpose = str(clip.get("purpose") or "").strip()
        # If this clip already uses a character reference sheet, appearance comes
        # from that sheet. Prompt rewrites may change action/scene/camera, but must
        # not invent a new body shape, fur, hair, proportions, clothing, etc.
        ref_defs = {str(x.get("id") or ""): x for x in list(plan.get("references") or []) if isinstance(x, dict)}
        character_ref_names = []
        effective_ref_ids = list(explicit_ref_ids or clip.get("reference_ids") or [])
        for ref_id in effective_ref_ids:
            ref = ref_defs.get(str(ref_id))
            if isinstance(ref, dict) and str(ref.get("type") or "").strip().lower() == "character":
                character_ref_names.append(str(ref.get("name") or ref_id))
        if not original_prompt:
            return False, f"Clip {order} has no stored prompt to rewrite."
        correction = str(correction or "").strip()
        if not correction:
            return False, "Tell me what should change in the prompt."

        self._telegram_agent_pending = {
            "chat_id": key,
            "purpose": "autonomous_redo_prompt",
            "phase": "autonomous_redo_prompt",
            "session_id": str(session.get("id") or ""),
            "redo_order": int(order),
            "redo_correction": correction,
            "redo_ref_ids": list(explicit_ref_ids or []),
            "redo_requested_seed": requested_seed,
            "redo_same_seed": bool(same_seed),
            "redo_original_prompt": original_prompt,
            "redo_purpose": purpose,
            "redo_duration": float(clip.get("duration") or 6.0),
            "redo_character_ref_names": character_ref_names,
        }
        self._telegram_send_text(key, f"Rewriting clip {order}'s prompt while keeping the original scene idea…")
        if self.server_ready and self._same_loaded_config():
            QtCore.QTimer.singleShot(0, self._telegram_agent_continue_pending)
        else:
            self._set_status("Loading LLM for Telegram clip prompt rewrite…", "loading")
            self._load_selected_model()
        return True, ""

    def _telegram_autonomous_start_redo_prompt_thread(self, pending: Dict[str, Any]) -> None:
        key = str(pending.get("chat_id") or "")
        order = int(pending.get("redo_order") or 0)
        original_prompt = str(pending.get("redo_original_prompt") or "").strip()
        purpose = str(pending.get("redo_purpose") or "").strip()
        correction = str(pending.get("redo_correction") or "").strip()
        character_ref_names = [str(x) for x in list(pending.get("redo_character_ref_names") or []) if str(x).strip()]
        ref_lock = ", ".join(character_ref_names)
        system = (
            "You are rewriting ONE existing FrameVision video-generation prompt after the user spotted a problem in that clip. "
            "The USER CORRECTION has higher priority than the wording/details of the old prompt. Preserve the story beat and useful context, NOT unwanted details from the old prompt. "
            "If the user says remove, no, without, avoid, do not, or asks for a different scene/action, obey that literally. DELETE the unwanted concept completely. "
            "Never keep it by changing its color, material, adjective, synonym, intensity, or by substituting a visually equivalent effect. Example: no slime means no slime, goo, residue, saliva sheen, condensation, wet coating, droplets, or substitute fluid unless the user explicitly asks for one. "
            "Likewise, no licking means no licking or mouth/tongue contact; choose a genuinely different action if needed. "
            "Preserve the original shot's story purpose, characters/vehicles and continuity unless the user's correction explicitly changes them. A requested scene change is allowed and must actually change the scene/action while keeping the story beat coherent. "
            "If a character reference sheet is in use, that reference is the sole source of truth for appearance. Do NOT redescribe, reinterpret or invent body shape, proportions, silhouette, fur, hair, skin texture, clothing or anatomy. Only describe the character's action, pose and expression unless the user explicitly asks to alter appearance. "
            "Keep useful camera, lighting and motion detail only when it does not conflict with the correction. "
            "Reference images are attached separately by FrameVision, so do not invent <Picture N> declarations or reference IDs. "
            "By default keep one continuous take and one physically continuous camera path. "
            "If the user explicitly requests multiple shots within this clip, write [Shot 1] for the opening and "
            "[Shot N] At MM:SS.mmm, for each later cut, with increasing cut times inside the clip duration. "
            "Return ONLY the rewritten video prompt, with no explanation, headings, markdown, analysis or quotation marks. /no_think"
        )
        user = (
            f"CLIP: {order}\n"
            f"CLIP DURATION: {pending.get('redo_duration', 'use the original duration')} seconds\n"
            f"LOCKED STORY PURPOSE: {purpose or 'preserve the original shot idea'}\n"
            f"CHARACTER REFERENCE LOCK: {'YES - ' + ref_lock + '. Use the sheet exactly; do not invent appearance.' if ref_lock else 'NO'}\n\n"
            f"ORIGINAL VIDEO PROMPT:\n{original_prompt}\n\n"
            f"USER CORRECTION (HIGHEST PRIORITY):\n{correction}\n\n"
            "Rewrite the prompt. Preserve the beat, but fully remove/replace anything the user rejected instead of paraphrasing it. /no_think"
        )
        thread = ChatCompletionThread(
            self.server_url,
            [{"role": "system", "content": system}, {"role": "user", "content": user}],
            max(1200, min(4000, int(self.settings_dialog.sp_max_tokens.value()))),
            min(0.35, float(self.settings_dialog.sp_temp.value())),
            self.settings_dialog.sp_top_p.value(),
            self.settings_dialog.sp_top_k.value(),
            self.settings_dialog.sp_repeat_penalty.value(),
            self.settings_dialog.sp_generation_timeout.value(),
            self,
            enable_thinking=False,
        )
        self._telegram_agent_thread = thread
        thread.succeeded.connect(self._telegram_autonomous_redo_prompt_succeeded)
        thread.failed.connect(self._telegram_agent_failed)
        thread.finished.connect(self._telegram_agent_thread_finished)
        thread.start()

    def _telegram_autonomous_redo_prompt_succeeded(self, payload: object) -> None:
        pending = self._telegram_agent_pending
        if not isinstance(pending, dict) or str(pending.get("phase") or "") != "autonomous_redo_prompt":
            return
        key = str(pending.get("chat_id") or "")
        session = self._telegram_autonomous_sessions.get(key)
        if not isinstance(session, dict):
            self._telegram_send_text(key, "The Agent project is no longer loaded, so I did not queue the redo.")
            self._telegram_agent_finish(unload=True)
            return
        if isinstance(payload, dict):
            rewritten = str(payload.get("content") or "").strip()
        else:
            rewritten = str(payload or "").strip()
        rewritten, _reasoning = _split_inline_reasoning(rewritten)
        rewritten = str(rewritten or "").strip().strip('`')
        if not rewritten or rewritten == "[Model returned an empty response]":
            self._telegram_send_text(key, "The LLM returned an empty rewritten prompt, so nothing was queued.")
            self._telegram_agent_finish(unload=True)
            return
        order = int(pending.get("redo_order") or 0)
        correction = str(pending.get("redo_correction") or "").strip()
        ref_ids = [str(x) for x in list(pending.get("redo_ref_ids") or []) if str(x).strip()]
        requested_seed = pending.get("redo_requested_seed")
        try:
            requested_seed = int(requested_seed) if requested_seed is not None else None
        except Exception:
            requested_seed = None
        same_seed = bool(pending.get("redo_same_seed", False))
        # Unload the large local LLM before handing the replacement to the normal
        # generation queue. This mirrors the initial Agent planning handoff and
        # prevents MiniMax from starting while the chat model still owns VRAM.
        self._telegram_agent_finish(unload=True)

        def _queue_rewritten_redo() -> None:
            current_session = self._telegram_autonomous_sessions.get(key)
            if not isinstance(current_session, dict):
                self._telegram_send_text(key, "The Agent project is no longer loaded, so I did not queue the rewritten redo.")
                return
            ok, info = self._telegram_autonomous_queue_single_redo(
                key, current_session, order, correction, ref_ids,
                requested_seed=requested_seed, same_seed=same_seed,
                replacement_prompt=rewritten,
            )
            if ok:
                self._telegram_send_text(key, f"Clip {order} prompt rewritten and {info[0].lower() + info[1:] if info else 'replacement queued.'}")
            else:
                self._telegram_send_text(key, "Prompt rewrite succeeded, but the replacement could not be queued: " + info)

        QtCore.QTimer.singleShot(250, _queue_rewritten_redo)

    def _telegram_autonomous_prepare_initial_music_preset_choice(self, chat_id: str, session: Dict[str, Any]) -> bool:
        """Ask for a requested ACE-Step preset subgenre before blueprint creation.

        This is intentionally conservative: it only pauses when a genre name from
        the real FrameVision preset library is actually present in the user's
        request and that genre has selectable subgenre presets. No recognized
        genre means no interruption; the later planner may use Custom music.
        """
        request = str(session.get("request") or "").strip()
        if not request:
            return False

        # Match genre labels against the full request, but require a strong hit.
        # Whole-word normalization in _ace15_music_match_candidates keeps this
        # from treating unrelated prose as a genre request.
        matches = self._ace15_music_match_candidates(
            request, include_genres=True, include_subgenres=False, limit=5
        )
        genre_match = next((m for m in matches if str(m.get("kind") or "") == "genre" and int(m.get("score") or 0) >= 55), None)
        if not genre_match:
            return False
        genre = str(genre_match.get("genre") or "").strip()
        if not genre:
            return False
        subs = self._ace15_subgenres_for_genre(genre)
        if not subs:
            return False

        candidates = [
            {"kind": "subgenre", "genre": genre, "subgenre": str(sub), "label": f"{genre} / {sub}", "score": 0}
            for sub in sorted(subs.keys(), key=lambda x: str(x).lower())
        ]

        # If the request itself names one of this genre's subgenres, make that
        # the suggested first choice while still asking, matching wizard behavior.
        suggested = None
        sub_matches = self._ace15_music_match_candidates(
            request, genre=genre, include_genres=False, include_subgenres=True, limit=8
        )
        if sub_matches and int(sub_matches[0].get("score") or 0) >= 55:
            sg = str(sub_matches[0].get("subgenre") or "")
            for i, item in enumerate(candidates):
                if str(item.get("subgenre") or "") == sg:
                    suggested = candidates.pop(i)
                    candidates.insert(0, suggested)
                    break

        session["music_preset_early"] = True
        session["music_preset_genre"] = genre
        session["music_preset_candidates"] = candidates
        session["status"] = "music_preset_choice"
        lines = [f"{i}. {str(c.get('subgenre') or '')}" for i, c in enumerate(candidates, 1)]
        suggestion = ""
        if suggested is not None:
            suggestion = f"\nI think `{str(suggested.get('subgenre') or '')}` is the best match; reply `yes` to use it."
        self._telegram_send_text(
            str(chat_id),
            f"ACE-Step preset match: `{genre}`. Choose the preset subgenre before story planning starts:\n"
            + "\n".join(lines)
            + suggestion
            + "\n\nReply with the number/name, `auto`/`yes` for the suggested first choice, or `custom` to ignore the preset library."
        )
        return True

    def _telegram_autonomous_prepare_music_preset_choice(self, chat_id: str, session: Dict[str, Any]) -> bool:
        """Resolve Agent music against the real ACE-Step preset tree.

        Return True when Agent mode must wait for a Telegram subgenre choice.
        Custom music is allowed only when no usable requested/planned genre exists
        in the preset manager or when the user explicitly chooses Custom.
        """
        plan = dict(session.get("plan") or {})
        music = dict(plan.get("music") or {})

        # An early Telegram choice made before blueprint creation is authoritative.
        # Apply it to the finished plan and do not ask the user a second time.
        locked = session.get("music_choice") if isinstance(session.get("music_choice"), dict) else None
        if locked:
            music.update(dict(locked))
            plan["music"] = music
            session["plan"] = plan
            return False
        if not bool(music.get("enabled", True)):
            session.pop("music_preset_candidates", None)
            session.pop("music_preset_genre", None)
            return False

        genre_hint = str(music.get("genre") or "").strip()
        sub_hint = str(music.get("subgenre") or "").strip()
        # Prefer the explicit/planned genre field. It is much safer than matching
        # the whole soundtrack caption, which previously let unrelated caption
        # words beat the actual requested genre.
        genre = self._ace15_find_genre(genre_hint) if genre_hint else ""
        if not genre:
            # No preset genre exists: this is the legitimate Custom path.
            music["ace15_mode"] = "custom"
            music.pop("ace15_genre", None)
            music.pop("ace15_subgenre", None)
            plan["music"] = music
            session["plan"] = plan
            return False

        subs = self._ace15_subgenres_for_genre(genre)
        if not subs:
            # The preset manager has the genre label but no selectable preset
            # payload beneath it, so the wizard's existing behavior is Custom.
            music["ace15_mode"] = "custom"
            music["ace15_genre"] = genre
            music.pop("ace15_subgenre", None)
            plan["music"] = music
            session["plan"] = plan
            return False

        candidates = []
        for sub in sorted(subs.keys(), key=lambda x: str(x).lower()):
            candidates.append({
                "kind": "subgenre",
                "genre": genre,
                "subgenre": str(sub),
                "label": f"{genre} / {sub}",
                "score": 0,
            })

        # If the blueprint already named a valid subgenre, put it first as the
        # suggested choice, but still ask rather than silently assuming.
        suggested = None
        if sub_hint:
            matches = self._ace15_music_match_candidates(
                sub_hint, genre=genre, include_genres=False, include_subgenres=True, limit=8
            )
            if matches and int(matches[0].get("score") or 0) >= 40:
                sg = str(matches[0].get("subgenre") or "")
                for i, item in enumerate(candidates):
                    if str(item.get("subgenre") or "") == sg:
                        suggested = candidates.pop(i)
                        candidates.insert(0, suggested)
                        break

        session["music_preset_genre"] = genre
        session["music_preset_candidates"] = candidates
        session["status"] = "music_preset_choice"
        plan["music"] = music
        session["plan"] = plan
        lines = [f"{i}. {str(c.get('subgenre') or '')}" for i, c in enumerate(candidates, 1)]
        suggestion = ""
        if suggested is not None:
            suggestion = f"\nI think `{str(suggested.get('subgenre') or '')}` is the best match; reply `yes` to use it."
        self._telegram_send_text(
            str(chat_id),
            f"ACE-Step preset match: `{genre}`. Before the long render starts, choose the preset subgenre:\n"
            + "\n".join(lines)
            + suggestion
            + "\n\nReply with the number/name, `auto`/`yes` for the suggested first choice, or `custom` to ignore the preset library."
        )
        return True

    def _telegram_autonomous_finish_music_preset_choice(self, key: str, session: Dict[str, Any], text: str) -> bool:
        if str(session.get("status") or "") != "music_preset_choice":
            return False
        raw = str(text or "").strip()
        low = self._ace15_normalize_text(raw)
        plan = dict(session.get("plan") or {})
        music = dict(plan.get("music") or {})
        genre = str(session.get("music_preset_genre") or music.get("genre") or "").strip()
        candidates = [dict(x) for x in list(session.get("music_preset_candidates") or []) if isinstance(x, dict)]

        if low in {"custom", "none", "no preset", "skip preset", "make your own", "own"}:
            music["ace15_mode"] = "custom"
            music["ace15_genre"] = genre
            music.pop("ace15_subgenre", None)
            chosen_text = "Custom ACE-Step prompt (preset library bypassed by request)"
        else:
            pick_text = "1" if low in {"auto", "automatic", "yes", "y", "correct", "recommended", "suggested"} else raw
            match = self._ace15_select_match_from_reply(pick_text, candidates)
            if not match:
                self._telegram_send_text(key, "I couldn't match that subgenre. Reply with one of the listed numbers/names, `auto`, or `custom`.")
                return True
            sub = str(match.get("subgenre") or "").strip()
            music["genre"] = genre
            music["subgenre"] = sub
            music["ace15_mode"] = "preset"
            music["ace15_genre"] = genre
            music["ace15_subgenre"] = sub
            chosen_text = f"{genre} / {sub}"

        plan["music"] = music
        session["plan"] = plan
        session.pop("music_preset_candidates", None)
        session.pop("music_preset_genre", None)

        # If this question was asked at project start, persist the choice and
        # continue into blueprint planning. The completed blueprint/plan later
        # receives this exact locked music choice without another interruption.
        if bool(session.pop("music_preset_early", False)):
            session["music_choice"] = dict(music)
            session["status"] = "planning"
            self._telegram_autonomous_save(session)
            self._telegram_agent_pending = {
                "chat_id": key,
                "text": str(session.get("request") or ""),
                "attachments": list(session.get("attachments") or []),
                "purpose": "autonomous_video",
                "phase": "autonomous_blueprint",
                "session_id": str(session.get("id") or ""),
            }
            self._telegram_send_text(key, f"ACE-Step music locked to `{chosen_text}`. Starting the story blueprint now; no more music question later.")
            if self.server_ready and self._same_loaded_config():
                QtCore.QTimer.singleShot(0, self._telegram_agent_continue_pending)
            else:
                self._set_status("Loading LLM for autonomous Telegram Agent…", "loading")
                self._load_selected_model()
            return True

        session["status"] = "queueing"
        self._telegram_autonomous_write_json(session, "plan", plan)
        self._telegram_autonomous_save(session)
        self._telegram_send_text(key, f"ACE-Step music locked to `{chosen_text}`. Starting references/generation now.")
        QtCore.QTimer.singleShot(250, lambda k=key: self._telegram_autonomous_queue_project(k))
        return True

    def _telegram_autonomous_handle_review_command(self, chat_id: str, text: str) -> bool:
        key = str(chat_id)
        raw = str(text or "").strip()
        low = raw.lower()
        session = self._telegram_autonomous_sessions.get(key)
        if not isinstance(session, dict):
            # These are Agent control commands, not chat prompts. Never fall
            # through to the LLM just because no project is currently loaded.
            looks_like_agent_control = (
                low in {"refs", "/refs", "/showrefs", "showrefs", "show refs", "/show refs", "show references", "/showreferences", "references", "assemble", "/assemble", "assemble video", "finish video", "finalize", "finalise"}
                or bool(re.match(r"^(?:show\s+)?(?:clip|shot)\s*\d+\s*$", low))
                or bool(re.match(r"^show\s+\d+\s*$", low))
                or bool(re.match(r"^/?(?:redo|recreate|regenerate)\s+(?:clip\s+|shot\s+)?\d+\b", raw, flags=re.I))
                or bool(re.match(r"^/?(?:redo|recreate|regenerate)\s+music\b", raw, flags=re.I))
            )
            if looks_like_agent_control:
                self._telegram_send_text(key, "No Telegram Agent project is currently loaded for this chat.")
                return True
            return False
        if str(session.get("status") or "") == "music_preset_choice":
            return self._telegram_autonomous_finish_music_preset_choice(key, session, raw)
        awaiting = session.get("awaiting_redo_prompt_change") if isinstance(session.get("awaiting_redo_prompt_change"), dict) else None
        if awaiting:
            # A bare `redo N change prompt` intentionally turns the next Telegram
            # message into the correction text. Normal slash/Agent control commands
            # cancel this tiny input state rather than accidentally becoming a prompt.
            if low in {"cancel", "/cancel", "never mind", "nevermind"}:
                session.pop("awaiting_redo_prompt_change", None)
                self._telegram_autonomous_save(session)
                self._telegram_send_text(key, "Prompt change cancelled.")
                return True
            if raw.startswith("/") or re.match(r"^(?:show|redo|recreate|regenerate|assemble)\b", raw, flags=re.I):
                session.pop("awaiting_redo_prompt_change", None)
                self._telegram_autonomous_save(session)
            else:
                session.pop("awaiting_redo_prompt_change", None)
                self._telegram_autonomous_save(session)
                ok, info = self._telegram_autonomous_start_redo_prompt_rewrite(
                    key, session, int(awaiting.get("order") or 0), raw,
                    [str(x) for x in list(awaiting.get("ref_ids") or []) if str(x).strip()],
                    requested_seed=awaiting.get("requested_seed"),
                    same_seed=bool(awaiting.get("same_seed", False)),
                )
                if not ok:
                    self._telegram_send_text(key, "Could not change the prompt: " + info)
                return True
        if low in {"refs", "/refs", "/showrefs", "showrefs", "show refs", "/show refs", "show references", "/showreferences", "references"}:
            self._telegram_send_text(key, self._telegram_autonomous_reference_catalog_text(session))
            return True
        m = re.match(r"^(?:show\s+)?(?:clip|shot)\s*(\d+)\s*$", low)
        if m:
            self._telegram_send_text(key, self._telegram_autonomous_clip_text(session, int(m.group(1))))
            return True
        # Also accept short 'show 7'.
        m = re.match(r"^show\s+(\d+)\s*$", low)
        if m:
            self._telegram_send_text(key, self._telegram_autonomous_clip_text(session, int(m.group(1))))
            return True
        music_redo = re.match(r"^/?(?:redo|recreate|regenerate)\s+music\b(.*)$", raw, flags=re.I | re.S)
        if music_redo:
            tail = str(music_redo.group(1) or "").strip()
            same_seed = bool(re.search(r"\bsame\s+seed\b", tail, flags=re.I))
            # `/redo music` and `/redo music new seed` both intentionally create a fresh seed.
            unsupported = re.sub(r"\b(?:new|same)\s+seed\b", " ", tail, flags=re.I)
            unsupported = re.sub(r"\s+", " ", unsupported).strip(" ,:;-. /\t")
            if unsupported:
                self._telegram_send_text(key, "For now use `/redo music new seed` or `/redo music same seed`.")
                return True
            ok, info = self._telegram_autonomous_queue_music_redo(key, session, same_seed=same_seed)
            self._telegram_send_text(key, info if ok else "Could not redo the soundtrack: " + info)
            return True
        m = re.match(r"^/?(?:redo|recreate|regenerate)\s+(?:clip\s+|shot\s+)?(\d+)\b(.*)$", raw, flags=re.I | re.S)
        if m:
            order = int(m.group(1))
            tail = str(m.group(2) or "").strip()
            # Seed control is parsed locally before any prompt correction text.
            # Plain redo intentionally gets a new random seed.
            same_seed = bool(re.search(r"\bsame\s+seed\b", tail, flags=re.I))
            seed_match = re.search(r"\bseed\s*[:=]?\s*(\d{1,10})\b", tail, flags=re.I)
            requested_seed = int(seed_match.group(1)) if seed_match else None
            if requested_seed is not None and not (1 <= requested_seed <= 2147483647):
                self._telegram_send_text(key, "Seed must be between 1 and 2147483647.")
                return True
            include_ref_ids, exclude_ref_ids, include_labels, exclude_labels, refs_mentioned = \
                self._telegram_autonomous_reference_selection_from_text(session, tail)
            # Reference selection is structural. Positive labels select refs; explicit
            # negative labels REMOVE refs from the clip instead of becoming prompt prose.
            current_ref_ids = [str(x) for x in list(dict(session.get("plan") or {}).get("clips", [])[order - 1].get("reference_ids") or [])] if 1 <= order <= len(list(dict(session.get("plan") or {}).get("clips") or [])) else []
            if refs_mentioned:
                if include_ref_ids:
                    ref_ids = [rid for rid in include_ref_ids if rid not in exclude_ref_ids]
                else:
                    ref_ids = [rid for rid in current_ref_ids if rid not in exclude_ref_ids]
                labels = list(include_labels)
                if exclude_labels:
                    labels.extend(["without " + x for x in exclude_labels])
            else:
                ref_ids, labels = [], []
            # Persist the picker result immediately and authoritatively.  In
            # particular, an empty selection means "use no references" rather
            # than "fall back to the original plan".  This state is also used if
            # a prompt rewrite must wait in the durable queue.
            if refs_mentioned and 1 <= order <= len(list(dict(session.get("plan") or {}).get("clips") or [])):
                selected_plan = dict(session.get("plan") or {})
                selected_clips = [dict(x) if isinstance(x, dict) else x for x in list(selected_plan.get("clips") or [])]
                selected_clips[order - 1]["reference_ids"] = list(ref_ids)
                selected_plan["clips"] = selected_clips
                session["plan"] = selected_plan
                self._telegram_autonomous_save(session)
            # Remove structural control words/labels. Ordinary scene instructions are
            # left intact and automatically take the LLM rewrite path below.
            correction = tail
            refs = self._telegram_autonomous_assign_reference_labels(session)
            # Strip reference selectors as STRUCTURE, longest aliases first. The old
            # loop could remove `Ref 1` from `Vehicle Ref 1` and leave the stray word
            # `Vehicle`, which accidentally forced an LLM rewrite.
            type_counts = {}
            for ref in refs:
                kind = str(ref.get("type") or "").strip().lower()
                if kind:
                    type_counts[kind] = int(type_counts.get(kind) or 0) + 1
            structural_aliases = []
            for ref in refs:
                label = str(ref.get("user_label") or "").strip()
                global_label = str(ref.get("global_ref_label") or "").strip()
                kind = str(ref.get("type") or "").strip().lower()
                structural_aliases.extend([x for x in (label, global_label) if x])
                if label.lower().startswith("character ref "):
                    structural_aliases.append("Character Sheet " + label.split()[-1])
                if kind and int(type_counts.get(kind) or 0) == 1:
                    structural_aliases.extend([
                        f"{kind} reference", f"{kind} ref", f"reference of the {kind}",
                        f"reference image of the {kind}", f"the {kind} reference",
                    ])
            for alias in sorted(set(structural_aliases), key=len, reverse=True):
                # Remove a negated selector as one structural unit when possible.
                correction = re.sub(
                    r"(?:do\s+not|don't|dont)\s+(?:use|include|attach)\s+(?:the\s+)?" + re.escape(alias),
                    " ", correction, flags=re.I,
                )
                correction = re.sub(
                    r"(?:without|exclude|remove|omit|skip|no)\s+(?:the\s+)?" + re.escape(alias),
                    " ", correction, flags=re.I,
                )
                correction = re.sub(re.escape(alias), " ", correction, flags=re.I)
            change_prompt = bool(re.search(r"\bchange\s+(?:the\s+)?prompt\b", correction, flags=re.I))
            correction = re.sub(r"\bchange\s+(?:the\s+)?prompt\b\s*[:=-]?", " ", correction, flags=re.I)
            correction = re.sub(r"\bsame\s+seed\b", " ", correction, flags=re.I)
            correction = re.sub(r"\bnew\s+seed\b", " ", correction, flags=re.I)
            correction = re.sub(r"\bseed\s*[:=]?\s*\d{1,10}\b", " ", correction, flags=re.I)
            correction = re.sub(r"^\s*(?:with|using|use)\b", " ", correction, flags=re.I)
            correction = re.sub(r"\b(?:with|using|use)\s*$", " ", correction, flags=re.I)
            correction = re.sub(r"\b(?:in|for)\s+(?:this|the)\s+prompt\s*$", " ", correction, flags=re.I)
            # If a command only changed reference selection, don't leave fragments
            # such as "do not use" behind as a fake creative correction.
            correction = re.sub(r"(?:^|[;,]\s*)\b(?:do\s+not|don't|dont|without|exclude|remove|omit|skip|no)\s+(?:the\s+)?(?:reference\s+(?:image\s+)?(?:of\s+)?)?(?:use|using)?\s*(?=$|[;,])", " ", correction, flags=re.I)
            correction = re.sub(r"\s+", " ", correction).strip(" ,:;-.")

            # Any real free-text instruction after `redo N` is a prompt edit. This
            # prevents FrameVision from appending a correction to the old prompt and
            # asking MiniMax to reconcile two contradictory scenes. Explicit ref/seed
            # only commands still take the fast direct-redo path.
            auto_rewrite = bool(correction)
            if change_prompt or auto_rewrite:
                if not correction:
                    session["awaiting_redo_prompt_change"] = {
                        "order": int(order),
                        "ref_ids": list(ref_ids),
                        "requested_seed": requested_seed,
                        "same_seed": bool(same_seed),
                    }
                    self._telegram_autonomous_save(session)
                    ref_note = (" using " + ", ".join(labels)) if labels else ""
                    self._telegram_send_text(
                        key,
                        f"What should I change in clip {order}'s prompt{ref_note}? "
                        "I will preserve the story purpose, obey your correction literally, and use a new seed by default."
                    )
                    return True
                ok, info = self._telegram_autonomous_start_redo_prompt_rewrite(
                    key, session, order, correction, ref_ids,
                    requested_seed=requested_seed, same_seed=same_seed,
                )
                if not ok:
                    self._telegram_send_text(key, "Could not change the prompt: " + info)
                return True

            ok, info = self._telegram_autonomous_queue_single_redo(
                key, session, order, correction, ref_ids,
                requested_seed=requested_seed, same_seed=same_seed,
            )
            self._telegram_send_text(key, info if ok else "Could not redo the clip: " + info)
            return True
        if low in {"assemble", "/assemble", "assemble video", "finish video", "finalize", "finalise"}:
            if str(session.get("status") or "") not in {"review_ready", "generating", "done"}:
                self._telegram_send_text(key, f"The project is currently {str(session.get('status') or 'working')}. Assembly is automatic; this command can only force/rebuild it when all current clips and music are ready.")
                return True
            clips = [Path(str(x)) for x in list(session.get("clip_outputs") or []) if str(x).strip()]
            expected = len(list(dict(session.get("plan") or {}).get("clips") or []))
            ready_count = sum(1 for p in clips if self._telegram_autonomous_file_ready(p, 4096))
            music_needed = bool(dict(dict(session.get("plan") or {}).get("music") or {}).get("enabled", True))
            music_path = Path(str(session.get("music_path") or "")) if str(session.get("music_path") or "") else None
            # A queued ACE-Step path is only a hint. ACE-Step may save/rename the
            # final soundtrack under its descriptive title. If the stored path is
            # missing or stale, resolve the real completed project soundtrack
            # before deciding that assembly must keep waiting.
            if music_needed and not (music_path and self._telegram_autonomous_file_ready(music_path, 4096)):
                found = self._telegram_autonomous_find_music(session)
                if found:
                    session["music_path"] = found
                    music_path = Path(found)
            music_ready = (not music_needed) or bool(music_path and self._telegram_autonomous_file_ready(music_path, 4096))
            if ready_count != expected or not music_ready:
                self._telegram_autonomous_save(session)
                self._telegram_send_text(key, f"Not ready to assemble yet: {ready_count}/{expected} current clip(s) ready; soundtrack {'ready' if music_ready else 'waiting'}.")
                return True
            ok, info = self._telegram_autonomous_enqueue_assembly(session)
            self._telegram_send_text(key, "Assembly queued manually. Bundled FFmpeg is building a new final revision now." if ok else "Could not start assembly: " + info)
            return True
        return False

    def _telegram_autonomous_queue_project(self, chat_id: str) -> None:
        key = str(chat_id)
        session = self._telegram_autonomous_sessions.get(key)
        if not isinstance(session, dict):
            return
        # Only the queueing stage may enter this function. This makes duplicate
        # QTimer callbacks harmless instead of allowing a second callback to fall
        # through into MiniMax while Krea is still running.
        if str(session.get("status") or "") != "queueing":
            return
        plan = dict(session.get("plan") or {})
        model = str(plan.get("video_model") or "minimax_h3")
        router = self._telegram_router_for_chat(key)

        # Asset-first execution. The previous Agent could describe a Krea
        # character sheet in JSON and then completely ignore it, causing MiniMax
        # to reinvent the character in every T2VA clip.
        refs_requested = model == "minimax_h3" and self._telegram_autonomous_user_requested_refs(str(session.get("request") or ""))
        planned_refs = [x for x in list(plan.get("references") or []) if isinstance(x, dict)]
        if model == "minimax_h3" and (refs_requested or planned_refs) and not list(session.get("reference_assets") or []):
            ok, info = self._telegram_autonomous_queue_reference_assets(key, session, router)
            if ok:
                return
            if refs_requested:
                session["status"] = "error"
                session["error"] = info
                self._telegram_autonomous_save(session)
                self._telegram_send_text(key, "I stopped before video generation because the requested character-reference stage could not be prepared:\n" + info)
                return
            # If refs were merely optional Agent suggestions, fall through to T2VA.

        self._telegram_autonomous_queue_media(key)

    def _telegram_autonomous_queue_media(self, chat_id: str) -> None:
        key = str(chat_id)
        session = self._telegram_autonomous_sessions.get(key)
        if not isinstance(session, dict):
            return
        plan = dict(session.get("plan") or {})
        model = str(plan.get("video_model") or "minimax_h3")
        if model == "minimax_h3" and self._telegram_autonomous_user_requested_refs(str(session.get("request") or "")):
            assets = [x for x in list(session.get("reference_assets") or []) if isinstance(x, dict)]
            ready = [x for x in assets if str(x.get("path") or "").strip() and self._telegram_autonomous_file_ready(Path(str(x.get("path"))), 4096)]
            if not assets or len(ready) != len(assets):
                session["status"] = "generating_refs"
                self._telegram_autonomous_save(session)
                self._telegram_autonomous_progress_heartbeat(key, session, f"Reference assets {len(ready)}/{len(assets)} ready; waiting for Krea before MiniMax can be queued.", interval=0.0)
                return
        width, height, res_key = self._telegram_autonomous_resolution(model, str(plan.get("resolution") or ""), str(plan.get("aspect") or "16:9"))
        aspect = str(plan.get("aspect") or "16:9")
        router = self._telegram_router_for_chat(key)

        clip_outputs = []
        clip_times = []
        failures = []
        sid = str(session.get("id") or "agent")
        for i, clip in enumerate(list(plan.get("clips") or []), start=1):
            duration = float(clip.get("duration") or 6.0)
            prompt = str(clip.get("prompt") or "").strip()
            if model == "minimax_h3":
                wanted = max(124, int(round(duration * 24.0)))
                candidates = list(range(124, 720, 17))
                frames = min(candidates, key=lambda x: abs(x - wanted))
            else:
                # LTX video frame grids are most reliable at 8*n+1.
                wanted = max(9, int(round(duration * 24.0)))
                frames = max(9, int(round((wanted - 1) / 8.0)) * 8 + 1)
            clip_refs = self._telegram_autonomous_reference_paths_for_clip(session, clip) if model == "minimax_h3" else []
            if model == "minimax_h3":
                prompt = self._telegram_autonomous_minimax_h3_prompt(clip, clip_refs)
            state = {
                "video_mode": "reference" if clip_refs else "text",
                "reference_image_paths": [str(x.get("path") or "") for x in clip_refs],
                "prompt": prompt,
                "resolution_key": res_key,
                "aspect_key": aspect,
                "width": width,
                "height": height,
                "frames": int(frames),
                "fps": 24,
                "duration_sec": float(frames) / 24.0,
                "output_name": f"{sid}_S{i:02d}",
            }
            try:
                if model == "ltx25":
                    route = router._queue_ltx25_from_state(state)
                elif model == "ltx23":
                    route = router._queue_ltx_video_from_state(state)
                else:
                    route = router._queue_minimax_h3_from_state(state)
            except Exception as exc:
                failures.append(f"S{i:02d}: {exc}")
                continue
            if not bool(getattr(route, "queued", False)):
                failures.append(f"S{i:02d}: {getattr(route, 'message', 'queue failed')}")
                continue
            output_path = str(getattr(route, "output_path", "") or "").strip()
            if not output_path:
                failures.append(f"S{i:02d}: queued successfully but the route returned no output path")
                continue
            clip_outputs.append(output_path)
            clip_times.append(float(getattr(route, "queued_at", 0.0) or time.time()))

        if failures or len(clip_outputs) != len(list(plan.get("clips") or [])):
            session["status"] = "error"
            session["error"] = " | ".join(failures) or "Not all clips were queued."
            session["clip_outputs"] = clip_outputs
            self._telegram_autonomous_save(session)
            self._telegram_send_text(key, "I stopped the autonomous project because not all clips could be queued:\n" + "\n".join(failures[:8]))
            return

        session["clip_outputs"] = clip_outputs
        session["clip_queued_at"] = clip_times

        # Queue ACE-Step soundtrack after the clips; the worker can execute it when
        # the video jobs release VRAM.
        music = dict(plan.get("music") or {})
        if bool(music.get("enabled", True)):
            caption = str(music.get("caption") or "").strip()
            ace_mode = str(music.get("ace15_mode") or "").strip().lower()
            ace_genre = str(music.get("ace15_genre") or music.get("genre") or "").strip()
            ace_sub = str(music.get("ace15_subgenre") or music.get("subgenre") or "").strip()
            state = {
                "genre": "Custom",
                "subgenre": "Agent",
                "preset": self._ace15_default_music_preset(caption),
                "caption": caption,
                "lyrics_mode": "instrumental" if bool(music.get("instrumental", True)) else "custom",
                "lyrics": str(music.get("lyrics") or ""),
                "duration": float(plan.get("target_duration") or 60.0) + 2.0,
                "title": f"{sid}_music",
                "bpm": int(music.get("bpm") or 0),
                "seed": random.randint(1, 2147483647),
            }
            try:
                # Agent mode must use the exact chosen FrameVision preset.  The
                # old code searched genre+subgenre+caption here and only applied
                # the preset if the fuzzy winner happened to be a subgenre. An
                # exact genre such as Drum & Bass therefore often stayed
                # Custom/Agent and produced music that was not DNB at all.
                if ace_mode == "preset" and ace_genre and ace_sub:
                    payload = self._ace15_preset_payload(ace_genre, ace_sub)
                    if payload:
                        state["genre"] = ace_genre
                        state["subgenre"] = ace_sub
                        state["preset"] = dict(payload)
                        if caption:
                            preset_caption = str(state["preset"].get("caption") or "").strip()
                            state["preset"]["caption"] = (preset_caption + ", " + caption).strip(", ")
                    else:
                        raise RuntimeError(f"ACE-Step preset disappeared: {ace_genre} / {ace_sub}")
                ok, info, out_dir = self._queue_ace15_music(state)
            except Exception as exc:
                ok, info, out_dir = False, str(exc), ""
            if not ok:
                session["status"] = "error"
                session["error"] = "ACE-Step soundtrack queue failed: " + str(info)
                self._telegram_autonomous_save(session)
                self._telegram_send_text(key, session["error"])
                return
            session["music_out_dir"] = str(out_dir)
            session["music_queued_at"] = time.time()
            session["music_title_hint"] = f"{sid}_music"
            session["music_last_seed"] = int(state.get("seed") or 0)
        else:
            session["music_out_dir"] = ""
            session["music_queued_at"] = 0.0

        session["status"] = "generating"
        self._telegram_autonomous_save(session)
        ref_clip_count = sum(
            1 for clip in list(plan.get("clips") or [])
            if model == "minimax_h3" and self._telegram_autonomous_reference_paths_for_clip(session, clip)
        )
        ref_note = f" ({ref_clip_count} using Ref2VA references)" if ref_clip_count else ""
        self._telegram_send_text(
            key,
            f"Autonomous project queued: {len(clip_outputs)} {model} clips{ref_note} + "
            + ("ACE-Step soundtrack. " if bool(music.get("enabled", True)) else "no soundtrack. ")
            + "I’ll watch the outputs, send each finished clip to Telegram for immediate checking, and assemble automatically when the current source clip for every slot and the soundtrack are ready. "
            + "You can use `show refs`, `show clip 7`, or `redo 7 with Vehicle Ref 1` at any time; replacements join the end of the queue and automatically trigger a corrected reassembly if needed."
        )

    def _telegram_autonomous_queue_music_redo(self, key: str, session: Dict[str, Any], *, same_seed: bool = False) -> tuple[bool, str]:
        """Regenerate only the current Agent soundtrack, preserving its ACE preset/settings."""
        plan = dict(session.get("plan") or {})
        music = dict(plan.get("music") or {})
        if not bool(music.get("enabled", True)):
            return False, "This project has no soundtrack enabled."

        sid = str(session.get("id") or "agent")
        counts = int(session.get("music_redo_count") or 0) + 1
        known_seed = int(session.get("music_last_seed") or 0)
        if same_seed:
            if known_seed <= 0:
                return False, "The previous soundtrack seed was not recorded. Use `/redo music new seed`."
            generation_seed = known_seed
        else:
            generation_seed = random.randint(1, 2147483647)

        caption = str(music.get("caption") or "").strip()
        ace_mode = str(music.get("ace15_mode") or "").strip().lower()
        ace_genre = str(music.get("ace15_genre") or music.get("genre") or "").strip()
        ace_sub = str(music.get("ace15_subgenre") or music.get("subgenre") or "").strip()
        title = f"{sid}_music_redo{counts:02d}"
        state = {
            "genre": "Custom",
            "subgenre": "Agent",
            "preset": self._ace15_default_music_preset(caption),
            "caption": caption,
            "lyrics_mode": "instrumental" if bool(music.get("instrumental", True)) else "custom",
            "lyrics": str(music.get("lyrics") or ""),
            "duration": float(plan.get("target_duration") or 60.0) + 2.0,
            "title": title,
            "bpm": int(music.get("bpm") or 0),
            "seed": int(generation_seed),
        }
        try:
            if ace_mode == "preset" and ace_genre and ace_sub:
                payload = self._ace15_preset_payload(ace_genre, ace_sub)
                if not payload:
                    raise RuntimeError(f"ACE-Step preset disappeared: {ace_genre} / {ace_sub}")
                state["genre"] = ace_genre
                state["subgenre"] = ace_sub
                state["preset"] = dict(payload)
                if caption:
                    preset_caption = str(state["preset"].get("caption") or "").strip()
                    state["preset"]["caption"] = (preset_caption + ", " + caption).strip(", ")
            ok, info, out_dir = self._queue_ace15_music(state)
        except Exception as exc:
            ok, info, out_dir = False, str(exc), ""
        if not ok:
            return False, "ACE-Step soundtrack redo queue failed: " + str(info)

        old_music = str(session.get("music_path") or "").strip()
        if old_music:
            superseded = list(session.get("superseded_music") or [])
            superseded.append({"path": old_music, "replaced_at": time.time(), "redo": counts})
            session["superseded_music"] = superseded[-20:]
        session["music_path"] = ""
        session["music_out_dir"] = str(out_dir)
        session["music_queued_at"] = time.time()
        session["music_title_hint"] = title
        session["music_redo_count"] = counts
        session["music_last_seed"] = int(generation_seed)
        session["assembly_dirty"] = True
        if str(session.get("status") or "") != "assembling":
            session["status"] = "generating"
        self._telegram_autonomous_save(session)
        mode = "same seed" if same_seed else "new seed"
        preset_text = f" using {ace_genre} / {ace_sub}" if ace_mode == "preset" and ace_genre and ace_sub else ""
        return True, f"Soundtrack redo {counts} queued with {mode} {generation_seed}{preset_text}. Video clips are unchanged; FrameVision will reassemble when the new soundtrack is ready."

    def _telegram_autonomous_find_music(self, session: Dict[str, Any]) -> str:
        out_dir = Path(str(session.get("music_out_dir") or ""))
        if not out_dir.exists():
            return ""
        queued_at = float(session.get("music_queued_at") or 0.0)
        hint = str(session.get("music_title_hint") or "").lower()
        candidates = []
        for ext in ("*.mp3", "*.wav", "*.flac", "*.m4a", "*.ogg"):
            try:
                for path in out_dir.rglob(ext):
                    try:
                        mt = float(path.stat().st_mtime)
                    except Exception:
                        continue
                    if mt + 2.0 < queued_at:
                        continue
                    score = (100 if hint and hint in path.name.lower() else 0) + mt
                    candidates.append((score, mt, path))
            except Exception:
                pass
        if not candidates:
            return ""
        # Epoch mtimes are ~1e9, so adding +100 to a score does not actually
        # prioritize the Agent title hint over a newer unrelated audio file.
        # Prefer exact project-hint candidates first, then fall back to newest.
        hinted = [x for x in candidates if hint and hint in x[2].name.lower()]
        pool = hinted or candidates
        pool.sort(key=lambda x: x[1], reverse=True)
        return str(pool[0][2])

    @staticmethod
    def _telegram_autonomous_file_ready(path: Path, min_size: int, settle_seconds: float = 6.0) -> bool:
        """Treat an output as ready only after it has stopped being written."""
        try:
            if not path.exists():
                return False
            st = path.stat()
            if int(st.st_size) <= int(min_size):
                return False
            return (time.time() - float(st.st_mtime)) >= float(settle_seconds)
        except Exception:
            return False

    def _telegram_autonomous_enqueue_assembly(self, session: Dict[str, Any]) -> tuple[bool, str]:
        clips = [Path(x) for x in list(session.get("clip_outputs") or []) if str(x).strip()]
        if not clips or not all(self._telegram_autonomous_file_ready(p, 4096) for p in clips):
            return False, "Not all source clips are fully written yet."
        music_path = Path(str(session.get("music_path") or "")) if str(session.get("music_path") or "") else None
        if music_path is not None and not self._telegram_autonomous_file_ready(music_path, 4096):
            return False, "Soundtrack is not fully written yet."

        ffmpeg = Path(self.fv_root) / "presets" / "bin" / ("ffmpeg.exe" if os.name == "nt" else "ffmpeg")
        if not ffmpeg.exists():
            return False, f"Bundled FFmpeg was not found: {ffmpeg}"

        sid = str(session.get("id") or "agent")
        out_dir = Path(self.fv_root) / "output" / "video" / "agent"
        out_dir.mkdir(parents=True, exist_ok=True)
        revision = int(session.get("assembly_revision") or 0) + 1
        concat_file = self._telegram_autonomous_dir() / f"{sid}_concat_r{revision:02d}.txt"
        def esc_concat(path: Path) -> str:
            return str(path.resolve()).replace("'", "'\\''")
        concat_file.write_text("\n".join(f"file '{esc_concat(p)}'" for p in clips) + "\n", encoding="utf-8")

        # Never overwrite an earlier final revision. Apart from preserving history,
        # this avoids Windows file-lock failures when the user already has the old
        # MP4 open in a player while a corrected assembly is produced.
        final_path = out_dir / (f"{sid}_final.mp4" if revision == 1 else f"{sid}_final_r{revision:02d}.mp4")
        target = float(dict(session.get("plan") or {}).get("target_duration") or 60.0)
        cmd = [
            str(ffmpeg), "-y",
            "-f", "concat", "-safe", "0", "-i", str(concat_file),
        ]
        if music_path is not None:
            cmd += ["-i", str(music_path)]
        cmd += ["-map", "0:v:0"]
        if music_path is not None:
            fade_start = max(0.0, target - 2.0)
            cmd += [
                "-map", "1:a:0",
                "-af", f"afade=t=out:st={fade_start:.3f}:d=2",
                "-c:a", "aac", "-b:a", "192k",
            ]
        else:
            cmd += ["-an"]
        cmd += [
            "-c:v", "libx264", "-preset", "medium", "-crf", "18",
            "-pix_fmt", "yuv420p",
            "-t", f"{target:.3f}",
            "-movflags", "+faststart",
            str(final_path),
        ]
        args = {
            "cmd": cmd,
            "ffmpeg_cmd": cmd,
            "outfile": str(final_path),
            "scan_dir": str(out_dir),
            "scan_ext": ".mp4",
            "label": f"Agent final assembly: {sid}",
            "engine": "agent_ffmpeg",
            "assistant_origin": "telegram",
            "telegram_chat_id": str(session.get("chat_id") or ""),
        }
        try:
            from helpers.queue_adapter import enqueue_tool_job  # type: ignore
        except Exception:
            from queue_adapter import enqueue_tool_job  # type: ignore
        try:
            enqueue_tool_job(
                job_type="tools_ffmpeg",
                input_path=str(clips[0]),
                out_dir=str(out_dir),
                args=args,
                priority=640,
            )
        except Exception as exc:
            return False, str(exc)
        previous_final = str(session.get("final_output") or "").strip()
        if previous_final and previous_final != str(final_path):
            history = list(session.get("final_history") or [])
            if previous_final not in history:
                history.append(previous_final)
            session["final_history"] = history
        session["final_output"] = str(final_path)
        session["assembly_revision"] = revision
        session["assembly_source_paths"] = [str(p) for p in clips]
        session["assembly_queued_at"] = time.time()
        session["assembly_dirty"] = False
        session["status"] = "assembling"
        self._telegram_autonomous_save(session)
        return True, str(final_path)

    def _telegram_autonomous_send_ready_clips(self, key: str, session: Dict[str, Any]) -> None:
        """Send each current clip revision to Telegram once it is fully written.

        The source-of-truth path for a slot may change after /redo. Tracking the
        path rather than only the slot number means every replacement is sent too.
        One file is sent per poll tick to avoid flooding Telegram after a restart.
        """
        plan = dict(session.get("plan") or {})
        clips = [x for x in list(plan.get("clips") or []) if isinstance(x, dict)]
        outputs = list(session.get("clip_outputs") or [])
        sent = dict(session.get("telegram_sent_clip_paths") or {})
        refs = self._telegram_autonomous_assign_reference_labels(session)
        labels_by_id = {str(x.get("id") or ""): str(x.get("user_label") or x.get("global_ref_label") or x.get("name") or x.get("id") or "") for x in refs}
        total = len(clips)
        for order in range(1, min(total, len(outputs)) + 1):
            path_text = str(outputs[order - 1] or "").strip()
            if not path_text or sent.get(str(order)) == path_text:
                continue
            path = Path(path_text)
            if not self._telegram_autonomous_file_ready(path, 4096):
                continue
            clip = clips[order - 1]
            redo_n = int(clip.get("redo_count") or 0)
            ref_labels = [labels_by_id.get(str(rid), str(rid)) for rid in list(clip.get("reference_ids") or []) if str(rid).strip()]
            caption = (f"Replacement Clip {order}/{total} r{redo_n} ready" if redo_n else f"Clip {order}/{total} ready")
            if ref_labels:
                caption += " | Refs: " + ", ".join(ref_labels)
            bridge = getattr(self, "_telegram_bridge", None)
            try:
                if bridge is not None:
                    bridge.send_file(key, str(path), caption)
                else:
                    self._telegram_send_text(key, caption + f"\n{path}")
            except Exception:
                self._telegram_send_text(key, caption + f"\n{path}")
            sent[str(order)] = path_text
            pending = dict(session.get("pending_redos") or {})
            if str(order) in pending and str(dict(pending[str(order)]).get("path") or "") == path_text:
                pending.pop(str(order), None)
                session["pending_redos"] = pending
            session["telegram_sent_clip_paths"] = sent
            self._telegram_autonomous_save(session)
            break

    @staticmethod
    def _telegram_autonomous_telegram_file_limit() -> int:
        # Telegram's hosted Bot API accepts multipart uploads up to 50 MB.
        # Stay below that ceiling to leave room for implementation differences.
        return 49_000_000

    def _telegram_autonomous_prepare_telegram_final(self, key: str, session: Dict[str, Any], final_path: Path) -> tuple[bool, str]:
        """Return a Telegram-sendable final path without touching the full-quality master.

        Long Agent assemblies can exceed Telegram Bot API's 50 MB upload limit.
        Keep the original local final untouched and create one smaller delivery copy
        only when necessary. The copy is produced by the normal FrameVision queue.
        """
        try:
            final_size = int(final_path.stat().st_size)
        except Exception:
            return False, "Final video is not readable yet."
        limit = self._telegram_autonomous_telegram_file_limit()
        if final_size <= limit:
            return True, str(final_path)

        delivery_path = final_path.with_name(final_path.stem + "_telegram.mp4")
        if self._telegram_autonomous_file_ready(delivery_path, 10000):
            try:
                if int(delivery_path.stat().st_size) <= limit:
                    session["telegram_delivery_path"] = str(delivery_path)
                    self._telegram_autonomous_save(session)
                    return True, str(delivery_path)
            except Exception:
                pass

        queued_for = str(session.get("telegram_delivery_source") or "")
        queued_at = float(session.get("telegram_delivery_queued_at") or 0.0)
        retry = int(session.get("telegram_delivery_retry") or 0)
        # If a previous delivery encode settled but still exceeded the limit,
        # allow one more encode with a more conservative target size.
        oversized_settled = False
        if self._telegram_autonomous_file_ready(delivery_path, 10000):
            try:
                oversized_settled = int(delivery_path.stat().st_size) > limit
            except Exception:
                oversized_settled = False
        if queued_for == str(final_path) and queued_at > 0.0 and not oversized_settled:
            return False, "Telegram delivery copy is still being prepared."
        if oversized_settled and retry >= 1:
            return False, f"Telegram delivery copy is still too large ({delivery_path.stat().st_size / (1024*1024):.1f} MB)."

        ffmpeg = Path(self.fv_root) / "presets" / "bin" / ("ffmpeg.exe" if os.name == "nt" else "ffmpeg")
        if not ffmpeg.exists():
            return False, f"Bundled FFmpeg was not found: {ffmpeg}"
        duration = float(dict(session.get("plan") or {}).get("target_duration") or 60.0)
        duration = max(1.0, duration)
        # Aim comfortably below Telegram's hard limit. A second attempt targets
        # even smaller output if the first encode happens to overshoot.
        target_bytes = 43_000_000 if retry == 0 else 38_000_000
        audio_k = 128
        total_k = max(700, int((target_bytes * 8.0) / duration / 1000.0))
        video_k = max(500, total_k - audio_k - 48)
        cmd = [
            str(ffmpeg), "-y", "-i", str(final_path),
            "-map", "0:v:0", "-map", "0:a:0?",
            "-c:v", "libx264", "-preset", "medium",
            "-b:v", f"{video_k}k", "-maxrate", f"{video_k}k", "-bufsize", f"{video_k * 2}k",
            "-c:a", "aac", "-b:a", f"{audio_k}k",
            "-pix_fmt", "yuv420p", "-movflags", "+faststart",
            str(delivery_path),
        ]
        args = {
            "cmd": cmd,
            "ffmpeg_cmd": cmd,
            "outfile": str(delivery_path),
            "scan_dir": str(delivery_path.parent),
            "scan_ext": ".mp4",
            "label": f"Agent Telegram delivery copy: {session.get('id') or 'agent'}",
            "engine": "agent_ffmpeg_telegram",
            "assistant_origin": "telegram",
            "telegram_chat_id": str(session.get("chat_id") or key),
        }
        try:
            from helpers.queue_adapter import enqueue_tool_job  # type: ignore
        except Exception:
            from queue_adapter import enqueue_tool_job  # type: ignore
        try:
            enqueue_tool_job(
                job_type="tools_ffmpeg",
                input_path=str(final_path),
                out_dir=str(delivery_path.parent),
                args=args,
                priority=645,
            )
        except Exception as exc:
            return False, f"Could not queue Telegram delivery copy: {exc}"
        session["telegram_delivery_path"] = str(delivery_path)
        session["telegram_delivery_source"] = str(final_path)
        session["telegram_delivery_queued_at"] = time.time()
        session["telegram_delivery_retry"] = retry + 1
        self._telegram_autonomous_save(session)
        self._telegram_send_text(
            key,
            f"Final video is {final_size / (1024*1024):.1f} MB, above Telegram's bot upload limit. "
            "Keeping the full-quality final locally and creating a smaller Telegram delivery copy now."
        )
        return False, "Telegram delivery copy queued."

    def _telegram_autonomous_send_final(self, key: str, session: Dict[str, Any], final_path: Path, dirty: bool = False) -> bool:
        ready, send_path = self._telegram_autonomous_prepare_telegram_final(key, session, final_path)
        if not ready:
            self._telegram_autonomous_progress_heartbeat(key, session, send_path, interval=60.0)
            return False
        sent = list(session.get("telegram_sent_final_delivery_paths") or [])
        if send_path in sent:
            return True
        bridge = getattr(self, "_telegram_bridge", None)
        caption = "Autonomous Agent finished the video." if not dirty else "Current video revision finished. Replacement clip(s) are still queued; an updated final will be assembled automatically."
        try:
            if bridge is not None:
                bridge.send_file(key, send_path, caption)
            else:
                self._telegram_send_text(key, f"{caption}\n{send_path}")
        except Exception as exc:
            self._telegram_send_text(key, f"Final video is ready, but Telegram delivery failed: {exc}\n{send_path}")
            return False
        sent.append(send_path)
        session["telegram_sent_final_delivery_paths"] = sent
        # Keep the old field updated for compatibility, but record the actual
        # file handed to Telegram separately from the full-quality master.
        old_sent = list(session.get("telegram_sent_final_paths") or [])
        if str(final_path) not in old_sent:
            old_sent.append(str(final_path))
        session["telegram_sent_final_paths"] = old_sent
        self._telegram_autonomous_save(session)
        return True

    def _telegram_autonomous_progress_heartbeat(self, key: str, session: Dict[str, Any], text: str, interval: float = 300.0) -> None:
        now = time.time()
        last = float(session.get("progress_notice_at") or 0.0)
        if now - last < interval:
            return
        session["progress_notice_at"] = now
        self._telegram_autonomous_save(session)
        self._telegram_send_text(key, text)

    def _telegram_autonomous_poll(self) -> None:
        for key, session in list(self._telegram_autonomous_sessions.items()):
            if not isinstance(session, dict):
                continue
            status = str(session.get("status") or "")
            # Telegram prompt corrections are durable queue items. Once the video
            # worker and local LLM are both free, process the oldest one. This lets
            # the user keep reviewing/sending fixes while long renders are active.
            if status in {"generating", "assembling", "done", "review_ready"}:
                if self._telegram_autonomous_try_start_queued_prompt_rewrite(key, session):
                    continue
            if status in {"planning", "planning_retry", "building_story", "building_shots", "auditing_references"}:
                # Resume deterministic multi-pass planning after restart. Only one
                # Telegram Agent may own the local LLM at a time.
                if self._telegram_agent_pending is None and (self._telegram_agent_thread is None or not self._telegram_agent_thread.isRunning()):
                    if status == "building_story":
                        phase = "autonomous_beats"
                    elif status == "building_shots":
                        phase = "autonomous_shots"
                    elif status == "auditing_references":
                        phase = "autonomous_ref_audit"
                    else:
                        phase = "autonomous_blueprint" if not session.get("blueprint_meta") else ("autonomous_beats" if len(list(session.get("story_slots") or [])) < int(self._telegram_autonomous_duration_contract(str(session.get("request") or ""))["shot_count"]) else "autonomous_shots")
                    self._telegram_agent_pending = {
                        "chat_id": key,
                        "text": str(session.get("request") or ""),
                        "attachments": list(session.get("attachments") or []),
                        "purpose": "autonomous_video",
                        "phase": phase,
                        "session_id": str(session.get("id") or ""),
                    }
                    if self.server_ready and self._same_loaded_config():
                        QtCore.QTimer.singleShot(0, self._telegram_agent_continue_pending)
                    else:
                        try:
                            self._set_status("Reloading LLM to resume Telegram Agent planning…", "loading")
                            self._load_selected_model()
                        except Exception as exc:
                            session["status"] = "error"
                            session["error"] = f"Could not resume planning after restart: {exc}"
                            self._telegram_autonomous_save(session)
                            self._telegram_send_text(key, session["error"])
            elif status == "queueing":
                # Queueing is the single entry point for asset preparation. Never
                # infer "no reference_assets yet" as permission to skip Krea: on
                # a fresh transition that list is naturally empty until
                # _queue_project() has actually queued the image jobs.
                QtCore.QTimer.singleShot(100, lambda k=key: self._telegram_autonomous_queue_project(k))
            elif status == "generating_refs":
                assets = [x for x in list(session.get("reference_assets") or []) if isinstance(x, dict)]
                paths = [Path(str(x.get("path") or "")) for x in assets if str(x.get("path") or "").strip()]
                ready_count = sum(1 for p in paths if self._telegram_autonomous_file_ready(p, 4096))
                refs_ready = bool(paths) and len(paths) == len(assets) and ready_count == len(paths)
                if not refs_ready:
                    self._telegram_autonomous_progress_heartbeat(key, session, f"Reference generation is still active: {ready_count}/{len(assets)} Krea reference asset(s) are fully written.")
                if refs_ready:
                    # Keep a predictable project-local copy so users can inspect
                    # exactly which refs the Agent handed to Ref2VA.
                    sid = str(session.get("id") or "agent")
                    ref_dir = Path(self.fv_root) / "output" / "images" / "agent_refs" / sid
                    ref_dir.mkdir(parents=True, exist_ok=True)
                    for n, asset in enumerate(assets, start=1):
                        src = Path(str(asset.get("path") or ""))
                        if not self._telegram_autonomous_file_ready(src, 4096):
                            continue
                        safe_name = re.sub(r"[^0-9A-Za-z_-]+", "_", str(asset.get("name") or asset.get("id") or f"ref_{n}")).strip("_") or f"ref_{n}"
                        dst = ref_dir / f"{n:02d}_{safe_name}{src.suffix or '.png'}"
                        try:
                            shutil.copy2(str(src), str(dst))
                            asset["source_path"] = str(src)
                            asset["path"] = str(dst)
                        except Exception:
                            pass
                    session["reference_assets"] = assets
                    session["reference_dir"] = str(ref_dir)
                    session["status"] = "queueing_media"
                    self._telegram_autonomous_save(session)
                    self._telegram_send_text(
                        key,
                        f"Krea 2 reference sheet(s) are ready in {ref_dir}. Queueing MiniMax now; required refs have been verified on disk and will be attached through Ref2VA."
                    )
                    QtCore.QTimer.singleShot(100, lambda k=key: self._telegram_autonomous_queue_media(k))
            elif status == "queueing_media":
                QtCore.QTimer.singleShot(100, lambda k=key: self._telegram_autonomous_queue_media(k))
            elif status == "generating":
                self._telegram_autonomous_send_ready_clips(key, session)
                outputs = [str(x or "").strip() for x in list(session.get("clip_outputs") or [])]
                expected = len(list(dict(session.get("plan") or {}).get("clips") or []))
                clips_ready = bool(expected) and len(outputs) >= expected and all(
                    p and self._telegram_autonomous_file_ready(Path(p), 4096) for p in outputs[:expected]
                )
                music_needed = bool(dict(session.get("plan") or {}).get("music", {}).get("enabled", True))
                music_candidate = Path(str(session.get("music_path") or "")) if str(session.get("music_path") or "") else None
                # Recover from a stale/precomputed ACE-Step output path. The actual
                # completed file can be renamed by ACE-Step (for example to the
                # project title + seed), so a non-empty path is not proof that the
                # soundtrack exists. Re-resolve whenever the stored file is not
                # actually ready on disk.
                if music_needed and not (music_candidate and self._telegram_autonomous_file_ready(music_candidate, 4096)):
                    found = self._telegram_autonomous_find_music(session)
                    if found:
                        session["music_path"] = found
                        music_candidate = Path(found)
                        self._telegram_autonomous_save(session)
                music_ready = (not music_needed) or bool(music_candidate and self._telegram_autonomous_file_ready(music_candidate, 4096))
                if not (clips_ready and music_ready):
                    clip_ready_count = sum(1 for p in outputs[:expected] if p and self._telegram_autonomous_file_ready(Path(p), 4096))
                    music_text = "ready" if music_ready else "waiting"
                    self._telegram_autonomous_progress_heartbeat(key, session, f"Generation is still active: {clip_ready_count}/{expected} current video clip(s) fully written; soundtrack {music_text}.")
                if clips_ready and music_ready:
                    ok, info = self._telegram_autonomous_enqueue_assembly(session)
                    if ok:
                        rev = int(session.get("assembly_revision") or 1)
                        self._telegram_send_text(key, ("All current clips and soundtrack are ready. Final assembly started automatically." if rev == 1 else f"All replacement clips are ready. Final video revision {rev} is being assembled automatically."))
                    else:
                        self._telegram_autonomous_progress_heartbeat(key, session, "All media is ready but automatic assembly could not start yet: " + info, interval=60.0)
            elif status == "review_ready":
                # Backward compatibility for v2.23 sessions: review is no longer a
                # gate. Resume the fully automatic path immediately.
                session["status"] = "generating"
                self._telegram_autonomous_save(session)
            elif status == "assembling":
                self._telegram_autonomous_send_ready_clips(key, session)
                final_path = Path(str(session.get("final_output") or ""))
                if not self._telegram_autonomous_file_ready(final_path, 10000):
                    size_mb = 0.0
                    try:
                        if final_path.exists():
                            size_mb = float(final_path.stat().st_size) / (1024.0 * 1024.0)
                    except Exception:
                        pass
                    self._telegram_autonomous_progress_heartbeat(key, session, f"Final FFmpeg assembly is still active; current output size is {size_mb:.1f} MB.")
                    continue

                # This assembly revision is complete. If a /redo was queued while
                # FFmpeg was working, keep the finished revision, optionally send
                # it, then wait for the replacement(s) and automatically assemble
                # another revision.
                dirty = bool(session.get("assembly_dirty", False))
                if not self._telegram_autonomous_send_final(key, session, final_path, dirty=dirty):
                    # The full-quality assembly is done, but a Telegram-sized
                    # delivery copy may still be encoding. Do not mark the
                    # project done until that copy has actually been handed off.
                    continue

                if dirty:
                    session["status"] = "generating"
                    self._telegram_autonomous_save(session)
                    self._telegram_send_text(key, "A clip replacement is pending. I will reassemble automatically as soon as the newest clip revision is ready.")
                else:
                    session["status"] = "done"
                    session["finished_at"] = time.time()
                    self._telegram_autonomous_save(session)
                    # Keep the completed project in memory so /redo can still
                    # replace a late-discovered bad clip without starting over.
            elif status == "done":
                # Keep sending any clips that became ready faster than Telegram
                # could receive them, and retain the project for late /redo.
                self._telegram_autonomous_send_ready_clips(key, session)
                # Recovery for older sessions that were marked done after the
                # bridge accepted a >50 MB upload request which Telegram itself
                # could not accept. This creates/sends a delivery copy on restart.
                final_path = Path(str(session.get("final_output") or ""))
                if self._telegram_autonomous_file_ready(final_path, 10000):
                    self._telegram_autonomous_send_final(key, session, final_path, dirty=False)

    def _telegram_autonomous_cancel(self, chat_id: str) -> bool:
        key = str(chat_id)
        session = self._telegram_autonomous_sessions.get(key)
        if not isinstance(session, dict):
            return False
        sid = str(session.get("id") or "")
        session["status"] = "cancelled"
        session["cancelled_at"] = time.time()
        self._telegram_autonomous_save(session)

        # Cancel every pending/running queue job belonging to this Agent project.
        jobs_root = Path(self.fv_root) / "jobs"
        for bucket in ("pending", "running"):
            folder = jobs_root / bucket
            if not folder.exists():
                continue
            for path in folder.glob("*.json"):
                try:
                    raw = path.read_text(encoding="utf-8", errors="ignore")
                except Exception:
                    continue
                if sid and sid not in raw and sid not in path.name:
                    continue
                try:
                    data = json.loads(raw)
                except Exception:
                    data = {}
                if bucket == "pending":
                    try:
                        path.unlink()
                    except Exception:
                        try:
                            data["cancel_requested"] = True
                            path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
                        except Exception:
                            pass
                else:
                    try:
                        data["cancel_requested"] = True
                        path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
                    except Exception:
                        pass
                    try:
                        Path(str(path) + ".cancel").write_text("cancel requested from Telegram Agent\n", encoding="utf-8")
                    except Exception:
                        pass
        self._telegram_autonomous_sessions.pop(key, None)
        self._telegram_send_text(key, "Autonomous Agent project cancelled. Running generation may finish its current safe unit before the worker stops it.")
        return True

    @staticmethod
    def _telegram_is_bare_agent_command(text: str) -> bool:
        return bool(re.fullmatch(
            r"\s*(?:use|start|let)\s+(?:the\s+)?agent\s*",
            str(text or ""),
            flags=re.I,
        ))

    def _telegram_autonomous_drop_placeholder(self, chat_id: str) -> bool:
        """Remove an accidentally-created Agent project whose whole request was
        only 'use agent'. No media job should ever be launched for that phrase.
        """
        key = str(chat_id)
        session = self._telegram_autonomous_sessions.get(key)
        if not isinstance(session, dict):
            return False
        request = str(session.get("request") or "").strip()
        if not self._telegram_is_bare_agent_command(request):
            return False
        session["status"] = "cancelled"
        session["cancelled_at"] = time.time()
        session["error"] = "discarded placeholder Agent command"
        self._telegram_autonomous_save(session)
        self._telegram_autonomous_sessions.pop(key, None)
        pending = getattr(self, "_telegram_agent_pending", None)
        if isinstance(pending, dict) and str(pending.get("chat_id") or "") == key:
            try:
                self._telegram_agent_finish(unload=True)
            except Exception:
                self._telegram_agent_pending = None
        return True

    def _telegram_is_complex_video_request(self, text: str) -> bool:
        low = str(text or "").lower()
        if "planner" in low or not re.search(r"\bvideo\b", low):
            return False
        return any(x in low for x in (
            "music video", "story video", "long video", "full video", "complete video",
            "minute", "minutes", "several clips", "multiple clips", "multi-clip",
            "background music", "ace step", "ace-step"
        ))

    def _telegram_agent_system_prompt(self, *, allow_skill: bool = True, skill_result: str = "") -> str:
        catalog = self._telegram_agent_skill_catalog()
        skill_lines = []
        for idx, item in enumerate(catalog, start=1):
            skill_lines.append(
                f"- skill#{idx} | id={item['id']} | name={item['name']} | tags={', '.join(item['tags']) or 'none'}"
                + (f" | description={item['description']}" if item['description'] else "")
            )
        skills = "\n".join(skill_lines) if skill_lines else "(no Agent-enabled pinned skills)"
        extra = ""
        if skill_result:
            extra = (
                "\nA pinned skill was already used for this request. Do NOT request another skill. "
                "Use this skill result when building the action command:\n---\n" + skill_result[:18000] + "\n---\n"
            )
        return (
            "You are the planning layer for FrameVision Agent mode. The user is talking to FrameVision through Telegram. "
            "Do not claim that you cannot create media: you have the FrameVision actions listed below. Your job is to choose "
            "the correct existing action, not explain how the user can click through the GUI.\n\n"
            "AVAILABLE ACTIONS:\n"
            "- create_image: create or edit a still image using FrameVision image generation.\n"
            "- create_video: create a single video clip using LTX 2.3, LTX 2.5, or MiniMax H3.\n"
            "- create_music: create a music track with ACE-Step 1.5.\n"
            "- autonomous_video: let the LLM itself create a complete multi-clip story/music video from idea to final assembled MP4. "
            "Use this for long/story/music-video requests unless the user explicitly asked for Planner.\n"
            "- planner: use the existing deterministic Planner ONLY when the user explicitly asks for Planner.\n"
            "- music_clip_creator: turn an existing music/audio track into a video/music clip workflow.\n"
            "- install_model: install an optional FrameVision model. Remote deletion/uninstall/hide actions do not exist.\n"
            "- status, queue, last_result: inspect FrameVision.\n\n"
            "PINNED CHAT SKILLS AVAILABLE TO YOU:\n" + skills + "\n\n"
            "Pinned chats are user-created specialist instructions. Use their tags to decide relevance. Exact model tags such as "
            "'minimax h3' plus task tags such as 'video'/'prompt enhancer' are strong matches; broader tags like 'video', 'images', "
            "'music', or 'prompt enhancer' are useful fallbacks. Do not select an unrelated skill just because one exists.\n\n"
            + ("If a pinned skill would materially improve the requested prompt/story/lyrics, you MAY return type=skill. Prefer its exact id; its exact name or skill# number is also accepted. "
               "Otherwise choose the FrameVision action directly.\n" if allow_skill else "Do not request a pinned skill in this pass.\n")
            + extra +
            "Return ONLY one JSON object, no markdown and no explanation. Allowed forms:\n"
            '{"type":"action","action":"planner","command":"use the planner to create ..."}\n'
            '{"type":"skill","skill_id":"<exact id, exact name, or skill#>","skill_input":"what the skill should produce for this request"}\n'
            '{"type":"clarify","message":"one necessary question"}\n'
            '{"type":"reply","message":"short answer when no FrameVision action is appropriate"}\n'
            "For combined requests, choose ONE orchestrating action rather than an array. "
            "If the user explicitly says Planner, choose planner. Otherwise a long/story/multi-clip/music-video request belongs to autonomous_video. "
            "For action commands, include every useful detail already provided by the user. Never invent destructive actions. /no_think"
        )

    def _telegram_agent_start(self, chat_id: str, text: str, attachments: list, *, purpose: str = "general", planner_state: Optional[Dict[str, Any]] = None) -> bool:
        key = str(chat_id)
        if self._telegram_agent_thread is not None and self._telegram_agent_thread.isRunning():
            self._telegram_send_text(key, "The FrameVision Agent is already thinking about another remote request. Try again when that finishes.")
            return True
        if getattr(self, "chat_thread", None) is not None and self.chat_thread.isRunning():
            self._telegram_send_text(key, "The local LLM is currently answering the desktop chat. Try again when that response is finished.")
            return True
        # Loading the large local LLM while a generation job owns VRAM is intentionally blocked.
        try:
            running = self._running_queue_job_files()
        except Exception:
            running = []
        if running:
            self._telegram_send_text(key, "This request needs the local LLM, but a FrameVision generation job is running. I will not load the local LLM into VRAM at the same time. Try again when the current job finishes.")
            return True
        try:
            self._validate_runner_and_model()
        except Exception as exc:
            self._telegram_send_text(key, f"I need the local LLM for that request, but it is not ready to load: {exc}")
            return True
        self._telegram_agent_pending = {
            "chat_id": key,
            "text": str(text or ""),
            "attachments": list(attachments or []),
            "purpose": str(purpose or "general"),
            "planner_state": dict(planner_state or {}) if isinstance(planner_state, dict) else None,
            "phase": "plan",
            "skill_result": "",
        }
        self._telegram_send_text(key, "I need the Agent for that. Loading the local LLM… Pinned skills will only be used if you approve them.")
        if self.server_ready and self._same_loaded_config():
            QtCore.QTimer.singleShot(0, self._telegram_agent_continue_pending)
        else:
            self._set_status("Loading LLM for Telegram Agent…", "loading")
            self._load_selected_model()
        return True

    def _telegram_agent_continue_pending(self) -> None:
        pending = self._telegram_agent_pending
        if not isinstance(pending, dict) or not self.server_ready:
            return
        if self._telegram_agent_thread is not None and self._telegram_agent_thread.isRunning():
            return
        phase = str(pending.get("phase") or "plan")
        if phase in {"autonomous_blueprint", "autonomous_beats", "autonomous_shots", "autonomous_ref_audit"}:
            self._telegram_autonomous_start_thread(pending)
            return
        if phase == "autonomous_redo_prompt":
            self._telegram_autonomous_start_redo_prompt_thread(pending)
            return
        if phase == "skill":
            self._telegram_agent_start_skill_thread()
            return
        allow_skill = phase == "plan"
        system = self._telegram_agent_system_prompt(
            allow_skill=allow_skill,
            skill_result=str(pending.get("skill_result") or "") if not allow_skill else "",
        )
        user_text = str(pending.get("text") or "").strip()
        purpose = str(pending.get("purpose") or "general")
        if purpose == "planner_enhance":
            state = pending.get("planner_state") if isinstance(pending.get("planner_state"), dict) else {}
            user_text = (
                "The user is inside the FrameVision Planner and explicitly asked to enhance the story idea. "
                "The user is considering story enhancement. If a pinned skill is clearly suitable, suggest it; "
                "otherwise return a normal FrameVision action/reply without forcing a skill.\n\n"
                f"Planner Extra info: {state.get('extra_info') or 'not specified'}\n"
                f"Story idea: {state.get('idea') or ''}\n"
            )
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user_text + "\n\n/no_think"},
        ]
        thread = ChatCompletionThread(
            self.server_url,
            messages,
            min(1600, int(self.settings_dialog.sp_max_tokens.value())),
            min(0.35, float(self.settings_dialog.sp_temp.value())),
            self.settings_dialog.sp_top_p.value(),
            self.settings_dialog.sp_top_k.value(),
            self.settings_dialog.sp_repeat_penalty.value(),
            self.settings_dialog.sp_generation_timeout.value(),
            self,
        )
        self._telegram_agent_thread = thread
        thread.succeeded.connect(self._telegram_agent_plan_succeeded)
        thread.failed.connect(self._telegram_agent_failed)
        thread.finished.connect(self._telegram_agent_thread_finished)
        thread.start()

    def _telegram_agent_skill_candidates(self, requested_skill: Optional[Dict[str, Any]], original_text: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Build a small user-review list of possibly relevant pinned skills.

        Tags are hints, not authority. The user gets the final say before any
        pinned chat is executed.
        """
        catalog = self._telegram_agent_skill_catalog()
        low = str(original_text or '').lower()
        words = set(re.findall(r'[a-z0-9][a-z0-9._+-]*', low))
        req_id = str((requested_skill or {}).get('id') or '')
        scored = []
        for item in catalog:
            blob_parts = [str(item.get('name') or '').lower(), str(item.get('description') or '').lower()]
            blob_parts += [str(x).lower() for x in (item.get('tags') or [])]
            blob = ' '.join(blob_parts)
            score = 0.0
            if req_id and str(item.get('id') or '') == req_id:
                score += 100.0
            for token in words:
                if len(token) >= 3 and re.search(r'\b' + re.escape(token) + r'\b', blob):
                    score += 2.0
            # Strong tag phrases get a little extra weight.
            for tag in item.get('tags') or []:
                tag_l = str(tag).lower().strip()
                if tag_l and tag_l in low:
                    score += 5.0
            if score > 0:
                scored.append((score, item))
        scored.sort(key=lambda x: (-x[0], str(x[1].get('name') or '').lower()))
        out = [dict(x[1]) for x in scored[:max(1, int(limit))]]
        if requested_skill and not any(str(x.get('id')) == req_id for x in out):
            out.insert(0, {
                'id': str(requested_skill.get('id') or ''),
                'name': str(requested_skill.get('name') or 'Pinned skill'),
                'description': str(requested_skill.get('description') or ''),
                'tags': self._normalize_pinned_tags(requested_skill.get('tags', [])),
            })
            out = out[:max(1, int(limit))]
        return out

    def _telegram_agent_request_skill_confirmation(self, pending: Dict[str, Any], requested_skill: Dict[str, Any], skill_input: str) -> None:
        key = str(pending.get('chat_id') or '')
        candidates = self._telegram_agent_skill_candidates(requested_skill, str(pending.get('text') or ''))
        pending['phase'] = 'skill_confirm'
        pending['skill_candidates'] = candidates
        pending['skill_input'] = str(skill_input or pending.get('text') or '').strip()
        self._telegram_agent_pending = pending
        lines = [
            'I found pinned skills that might help, but I will not use any automatically:',
            ''
        ]
        for i, item in enumerate(candidates, start=1):
            tags = ', '.join(item.get('tags') or []) or 'no tags'
            lines.append(f"{i}. {item.get('name') or 'Pinned skill'} [{tags}]")
        lines += [
            '',
            'Type `no` to use none, the number(s) you want separated by commas (for example `1,3`), or `all`.'
        ]
        self._telegram_send_text(key, '\n'.join(lines))

    def _telegram_agent_handle_skill_confirmation(self, chat_id: str, text: str) -> bool:
        pending = self._telegram_agent_pending
        if not isinstance(pending, dict) or str(pending.get('chat_id') or '') != str(chat_id):
            return False
        if str(pending.get('phase') or '') != 'skill_confirm':
            return False
        low = str(text or '').strip().lower()
        candidates = list(pending.get('skill_candidates') or [])
        if low in {'no', 'none', 'skip', 'no skills', 'do not use skills', 'dont use skills'}:
            if str(pending.get('purpose') or '') == 'planner_enhance':
                state = pending.get('planner_state') if isinstance(pending.get('planner_state'), dict) else {}
                state['story_prompt_decision'] = 'use'
                state['story_enhancing'] = False
                state['story_enhanced_reviewed'] = True
                self._telegram_remote_wizards[str(chat_id)] = {'kind': 'planner', 'state': state}
                self._telegram_agent_finish(unload=True)
                self._telegram_send_text(str(chat_id), 'Okay — no pinned skill will be used. I kept the original Planner idea.')
                q = self._planner_agent_next_question(state)
                if q:
                    self._telegram_send_text(str(chat_id), q)
                return True
            pending['phase'] = 'plan_no_skill'
            pending['skill_result'] = ''
            pending.pop('skill_candidates', None)
            self._telegram_agent_pending = pending
            self._telegram_send_text(str(chat_id), 'Okay — continuing without pinned skills.')
            QtCore.QTimer.singleShot(0, self._telegram_agent_continue_pending)
            return True
        if low == 'all':
            selected = candidates
        else:
            nums = []
            for part in re.split(r'[,;\s]+', low):
                if part.isdigit():
                    nums.append(int(part))
            if not nums or any(n < 1 or n > len(candidates) for n in nums):
                self._telegram_send_text(str(chat_id), 'Please type `no`, `all`, or valid skill number(s) such as `1` or `1,3`.')
                return True
            seen = set()
            selected = []
            for n in nums:
                item = candidates[n - 1]
                sid = str(item.get('id') or '')
                if sid and sid not in seen:
                    seen.add(sid)
                    selected.append(item)
        if not selected:
            pending['phase'] = 'plan_no_skill'
            self._telegram_agent_pending = pending
            QtCore.QTimer.singleShot(0, self._telegram_agent_continue_pending)
            return True
        pending['approved_skill_queue'] = [str(x.get('id') or '') for x in selected if str(x.get('id') or '')]
        pending['approved_skill_names'] = {str(x.get('id') or ''): str(x.get('name') or 'Pinned skill') for x in selected}
        pending['skill_results'] = []
        pending.pop('skill_candidates', None)
        pending['skill_id'] = pending['approved_skill_queue'].pop(0)
        pending['skill_name'] = pending['approved_skill_names'].get(pending['skill_id'], 'Pinned skill')
        pending['phase'] = 'skill'
        self._telegram_agent_pending = pending
        self._telegram_send_text(str(chat_id), f"Using approved pinned Agent skill: {pending['skill_name']}")
        QtCore.QTimer.singleShot(0, self._telegram_agent_continue_pending)
        return True

    def _telegram_agent_plan_succeeded(self, payload: object) -> None:
        pending = self._telegram_agent_pending
        if not isinstance(pending, dict):
            return
        decision = self._telegram_agent_parse_json(payload)
        key = str(pending.get("chat_id") or "")
        original_text = str(pending.get("text") or "")
        try:
            raw_content = str(payload.get("content") or "") if isinstance(payload, dict) else str(payload or "")
            dbg = Path(self.fv_root) / "logs" / "telegram_agent_last_response.txt"
            dbg.parent.mkdir(parents=True, exist_ok=True)
            dbg.write_text(raw_content, encoding="utf-8")
        except Exception:
            pass
        decision = self._telegram_agent_recover_decision(decision, original_text)
        if not decision:
            self._telegram_send_text(key, "The Agent response could not be interpreted safely, so nothing was executed. Its last response was saved to logs/telegram_agent_last_response.txt.")
            self._telegram_agent_finish(unload=True)
            return
        if bool(decision.get("_recovered")):
            self._telegram_send_text(key, "The LLM response was not in the expected Agent JSON format, but I safely recovered the request and routed it to FrameVision.")
        dtype = str(decision.get("type") or "").strip().lower()
        if dtype == "skill" and str(pending.get("phase") or "plan") == "plan":
            skill = self._telegram_agent_find_skill(str(decision.get("skill_id") or ""))
            if not skill:
                # Do not dead-end the user's job because the LLM formatted a
                # skill reference badly. Replan once without skills so it can
                # still choose a normal FrameVision action.
                self._telegram_send_text(
                    key,
                    "I couldn't match the Agent's pinned-skill reference to the current skill list. "
                    "I'll continue without that skill instead of stopping."
                )
                pending["phase"] = "plan_after_bad_skill"
                pending["skill_result"] = ""
                self._telegram_agent_pending = pending
                QtCore.QTimer.singleShot(0, self._telegram_agent_continue_pending)
                return
            self._telegram_agent_request_skill_confirmation(
                pending, skill, str(decision.get("skill_input") or pending.get("text") or "").strip()
            )
            return
        if str(pending.get("purpose") or "general") == "planner_enhance":
            # Planner enhancement must use a selected skill. A plain reply here means
            # the planner found no suitable user skill and should stay unchanged.
            msg = str(decision.get("message") or "I couldn't find a suitable pinned Agent skill for Planner story enhancement.")
            self._telegram_send_text(key, msg)
            self._telegram_agent_finish(unload=True)
            return
        if dtype in {"clarify", "reply"}:
            self._telegram_send_text(key, str(decision.get("message") or "I need a little more information."))
            self._telegram_agent_finish(unload=True)
            return
        if dtype != "action":
            self._telegram_send_text(key, "The Agent did not return a valid FrameVision action, so nothing was executed.")
            self._telegram_agent_finish(unload=True)
            return
        action = self._telegram_agent_action_alias(decision.get("action") or "")
        command = str(decision.get("command") or pending.get("text") or "").strip()
        if action == "autonomous_video":
            # A generic Agent reply must never resurrect an old long-video request.
            # Autonomous project creation is authorized only by the CURRENT user
            # message, and that exact message becomes the project request.
            source_text = str(pending.get("text") or "").strip()
            explicit_agent = bool(re.search(r"\b(?:use|let|start)\s+(?:the\s+)?agent\b", source_text, re.I))
            if not explicit_agent and not self._telegram_is_complex_video_request(source_text):
                self._telegram_send_text(key, "Ignored an unrelated autonomous-video action from the LLM; your current Telegram message did not request a new Agent video project.")
                self._telegram_agent_finish(unload=True)
                return
            command = re.sub(r"\b(?:use|let|start)\s+(?:the\s+)?agent\b\s*(?:to)?", "", source_text, count=1, flags=re.I).strip(" ,:-") if explicit_agent else source_text
        attachments = list(pending.get("attachments") or [])
        self._telegram_agent_finish(unload=True)
        QtCore.QTimer.singleShot(200, lambda k=key, a=action, c=command, at=attachments: self._telegram_agent_execute_action(k, a, c, at))

    def _telegram_agent_start_skill_thread(self) -> None:
        pending = self._telegram_agent_pending
        if not isinstance(pending, dict):
            return
        skill = self._telegram_agent_find_skill(str(pending.get("skill_id") or ""))
        key = str(pending.get("chat_id") or "")
        if not skill:
            self._telegram_send_text(key, "That pinned Agent skill is no longer available.")
            self._telegram_agent_finish(unload=True)
            return
        template = str(skill.get("template") or "").strip()
        task = str(pending.get("skill_input") or pending.get("text") or "").strip()
        state = pending.get("planner_state") if isinstance(pending.get("planner_state"), dict) else {}
        if str(pending.get("purpose") or "general") == "planner_enhance":
            task = (
                "Enhance this story idea for FrameVision Planner. Preserve the user's concept and requested visual style. "
                "Return one coherent enhanced project-level story prompt, not an explanation or numbered shot list.\n\n"
                f"Planner Extra info: {state.get('extra_info') or 'not specified'}\n"
                f"Story idea: {state.get('idea') or ''}"
            )
        contract = (
            "\n\nFRAMEVISION AGENT SKILL CONTRACT:\n"
            "Return only the useful finished output for the requested task. Do not explain your reasoning, do not mention being a skill, "
            "and do not include Thinking/Analysis/Final Answer labels. /no_think"
        )
        messages = [
            {"role": "system", "content": template + contract},
            {"role": "user", "content": task + "\n\n/no_think"},
        ]
        thread = ChatCompletionThread(
            self.server_url,
            messages,
            self.settings_dialog.sp_max_tokens.value(),
            self.settings_dialog.sp_temp.value(),
            self.settings_dialog.sp_top_p.value(),
            self.settings_dialog.sp_top_k.value(),
            self.settings_dialog.sp_repeat_penalty.value(),
            self.settings_dialog.sp_generation_timeout.value(),
            self,
        )
        self._telegram_agent_thread = thread
        thread.succeeded.connect(self._telegram_agent_skill_succeeded)
        thread.failed.connect(self._telegram_agent_failed)
        thread.finished.connect(self._telegram_agent_thread_finished)
        thread.start()

    def _telegram_agent_skill_succeeded(self, payload: object) -> None:
        pending = self._telegram_agent_pending
        if not isinstance(pending, dict):
            return
        key = str(pending.get("chat_id") or "")
        if isinstance(payload, dict):
            result = str(payload.get("content") or "").strip()
        else:
            result = str(payload or "").strip()
        result, _reasoning = _split_inline_reasoning(result)
        result = str(result or "").strip()
        if not result:
            self._telegram_send_text(key, "The pinned Agent skill returned an empty result.")
            self._telegram_agent_finish(unload=True)
            return
        if str(pending.get("purpose") or "general") == "planner_enhance":
            state = pending.get("planner_state") if isinstance(pending.get("planner_state"), dict) else {}
            state["story_prompt_decision"] = "enhance"
            state["story_original_idea"] = str(state.get("story_original_idea") or state.get("idea") or "").strip()
            state["idea"] = result
            state["story_enhancing"] = False
            state["story_enhanced_reviewed"] = False
            self._telegram_remote_wizards[key] = {"kind": "planner", "state": state}
            self._telegram_send_text(key, "Enhanced story idea:\n\n" + result + '\n\nReview it before I continue. Reply "use it" to accept it, or send your edited version.')
            self._telegram_agent_finish(unload=True)
            return
        results = list(pending.get("skill_results") or [])
        results.append({"name": str(pending.get("skill_name") or "Pinned skill"), "result": result})
        pending["skill_results"] = results
        queue = list(pending.get("approved_skill_queue") or [])
        if queue:
            pending["skill_id"] = queue.pop(0)
            pending["approved_skill_queue"] = queue
            pending["skill_name"] = dict(pending.get("approved_skill_names") or {}).get(pending["skill_id"], "Pinned skill")
            pending["phase"] = "skill"
            self._telegram_agent_pending = pending
            self._telegram_send_text(key, f"Using approved pinned Agent skill: {pending['skill_name']}")
            QtCore.QTimer.singleShot(0, self._telegram_agent_continue_pending)
            return
        combined = []
        for item in results:
            combined.append(f"[{item.get('name')}]\n{item.get('result')}")
        pending["skill_result"] = "\n\n".join(combined)
        pending["phase"] = "replan"
        self._telegram_agent_pending = pending
        QtCore.QTimer.singleShot(0, self._telegram_agent_continue_pending)

    def _telegram_agent_failed(self, message: str) -> None:
        pending = self._telegram_agent_pending
        key = str(pending.get("chat_id") or "") if isinstance(pending, dict) else ""
        if isinstance(pending, dict) and str(pending.get("purpose") or "") == "autonomous_video" and key:
            session = self._telegram_autonomous_sessions.get(key)
            if isinstance(session, dict):
                if pending.get("phase") == "autonomous_ref_audit" and session.get("reference_selection_pending"):
                    self._telegram_autonomous_finish_reference_selection(key, session, pending, error=str(message or "LLM request failed"))
                    return
                session["status"] = "error"
                session["error"] = "LLM planning failed: " + str(message or "unknown error")
                self._telegram_autonomous_save(session)
                self._telegram_autonomous_sessions.pop(key, None)
        if key:
            self._telegram_send_text(key, "Telegram Agent LLM failed: " + str(message or "unknown error"))
        self._telegram_agent_finish(unload=True)

    def _telegram_agent_thread_finished(self) -> None:
        self._telegram_agent_thread = None
        # succeeded is emitted just before QThread actually finishes. Continue a
        # multi-pass Agent request here so the next pass cannot be lost because
        # the previous thread was still technically running.
        pending = self._telegram_agent_pending
        if isinstance(pending, dict):
            phase = str(pending.get("phase") or "")
            should_continue = phase in {"skill", "replan", "autonomous_blueprint", "autonomous_beats", "autonomous_shots", "autonomous_ref_audit"}
            if bool(pending.get("structured_retry", False)):
                should_continue = True
            if should_continue:
                QtCore.QTimer.singleShot(0, self._telegram_agent_continue_pending)

    def _telegram_agent_finish(self, *, unload: bool = True) -> None:
        self._telegram_agent_pending = None
        self._telegram_agent_thread = None
        if unload:
            try:
                self._unload_model()
            except Exception:
                try:
                    self._stop_process_only()
                except Exception:
                    pass

    def _telegram_agent_execute_action(self, chat_id: str, action: str, command: str, attachments: list) -> None:
        prefixes = {
            "create_image": "create an image ",
            "create_video": "create a video ",
            "create_music": "create music ",
            "autonomous_video": "use the agent to create ",
            "planner": "use the planner to create ",
            "music_clip_creator": "use the music clip creator ",
            "install_model": "install ",
        }
        a = str(action or "").strip().lower()
        if a == "autonomous_video":
            self._telegram_autonomous_video_start(chat_id, str(command or ""), list(attachments or []))
            return
        if a in {"status", "queue", "last_result"}:
            mapped = "/status" if a in {"status", "queue"} else "/last"
            self._on_telegram_message({"chat_id": chat_id, "text": mapped, "attachments": [], "_agent_generated": True})
            return
        if a not in prefixes:
            self._telegram_send_text(chat_id, f"The Agent requested an unsupported action: {a or 'unknown'}")
            return
        cmd = str(command or "").strip()
        low = cmd.lower()
        trigger_ok = {
            "create_image": bool(re.match(r"^\s*(?:create|make|generate).*\b(?:image|picture|photo|render)\b", low)),
            "create_video": bool(re.match(r"^\s*(?:create|make|generate).*\bvideo\b", low)),
            "create_music": bool(re.match(r"^\s*(?:create|make|generate)\s+music\b", low)),
            "planner": "planner" in low,
            "music_clip_creator": "music clip creator" in low,
            "install_model": bool(re.match(r"^\s*(?:install|download|get)\b", low)),
        }.get(a, False)
        if not trigger_ok:
            cmd = prefixes[a] + cmd
        self._on_telegram_message({"chat_id": chat_id, "text": cmd, "attachments": list(attachments or []), "_agent_generated": True})

    def _on_telegram_message(self, payload: object) -> None:
        data = dict(payload or {}) if isinstance(payload, dict) else {}
        chat_id = str(data.get("chat_id") or "")
        text = str(data.get("text") or "").strip()
        attachments = list(data.get("attachments") or [])
        if not chat_id:
            return
        low = text.lower().strip()
        agent_generated = bool(data.get("_agent_generated", False))
        try:
            # First-class autonomous entry command. This MUST be intercepted
            # before every legacy wizard/router path. It does not load the LLM
            # and it does not create a project yet; it only waits for the goal.
            if self._telegram_is_bare_agent_command(text):
                self._telegram_autonomous_drop_placeholder(chat_id)
                self._telegram_remote_wizards.pop(chat_id, None)
                self._telegram_remote_wizards[chat_id] = {
                    "kind": "agent_idea",
                    "state": {"attachments": list(attachments or [])},
                }
                self._telegram_send_text(
                    chat_id,
                    "Autonomous Agent selected. Send me the complete video request. "
                    "You may specify model, approximate duration, quality/resolution, aspect, music/narration, refs, or other constraints. "
                    "Anything you leave open is for the Agent to decide.\n\n"
                    "Important: generation uses the LAST SAVED settings from the normal FrameVision model tabs. "
                    "Before starting a long Agent job, set/save the options you want there. For MiniMax H3 this includes loaded LoRAs and their strengths, "
                    "steps, Sage Attention, Spectrum Forecaster, Comfy Kitchen, sampler/scheduler, VRAM settings and model overrides."
                )
                return
            # Project review/repair commands are handled before generic Telegram
            # routing so a simple "redo 7" cannot be swallowed by the chat/wizard.
            if self._telegram_autonomous_handle_review_command(chat_id, text):
                return
            try:
                from helpers.telegram_bridge import queue_summary, cancel_current_job, find_last_result, worker_status  # type: ignore
            except Exception:
                from telegram_bridge import queue_summary, cancel_current_job, find_last_result, worker_status  # type: ignore
            root_path = Path(self.fv_root)
            if low in {"/start", "/help", "help"}:
                self._telegram_send_text(chat_id,
                    "FrameVision Telegram Remote is connected.\n\n"
                    "Create / run workflows (direct commands do not load the LLM unless Agent mode needs it):\n"
                    "• create an image\n• create a video with MiniMax\n• create music\n"
                    "• use agent to create a 1 minute music video ...\n"
                    "• use the planner\n• use the music clip creator\n• install <model>\n\n"
                    "Agent project inspection / correction:\n"
                    "• /showrefs, /refs, or show refs — list saved reference names/labels directly (no LLM)\n"
                    "• show clip 7 (or show 7) — show its locked purpose, refs, prompt, seed and current output\n"
                    "• /redo 7 (or redo 7) — regenerate that slot with a NEW random seed\n"
                    "• /redo 7 new seed — explicitly request another random seed\n"
                    "• /redo 7 seed 123456 — use an exact seed\n"
                    "• /redo 7 same seed — reuse the last recorded seed for that slot\n"
                    "• /redo 7 with Vehicle Ref 1 — explicitly choose a ref\n"
                    "• /redo 7 change prompt — ask what is wrong, rewrite only that clip prompt, then redo\n"
                    "• /redo 7 change prompt keep the car driving forward — rewrite immediately\n"
                    "• /redo 7 change prompt use Vehicle Ref 1 — rewrite the prompt and force that ref\n"
                    "• /redo music new seed — regenerate only the ACE-Step soundtrack with the same preset/settings\n"
                    "• /redo 7 with Vehicle Ref 1 seed 123456 camera lower — combine ref, seed and correction\n"
                    "• assemble — optional manual rebuild; normally assembly/reassembly is automatic\n"
                    "Finished clips are sent to Telegram as soon as they are ready. Redos join the end of the queue, replace that slot as the source of truth, and automatically trigger a new final assembly if the old final already exists. Existing music is reused.\n\n"
                    "General remote commands:\n"
                    "• /status, /progress or /queue — queue + Agent status\n"
                    "• /cancel — cancel the active Agent project or current queued job\n"
                    "• /last — send the latest generated result\n"
                    "• /new — reset the current Telegram wizard/chat state\n"
                    "• /help — show this help\n\n"
                    "Telegram can install optional models after confirmation, but remote hide/delete/uninstall is blocked for safety. "
                    "You can attach images, videos, and audio when a wizard asks for them.")
                return
            if low in {"/status", "status", "/progress", "progress", "/queue", "queue"}:
                summary = queue_summary(root_path)
                session = self._telegram_autonomous_sessions.get(chat_id)
                if isinstance(session, dict):
                    plan = dict(session.get("plan") or {})
                    summary += (
                        "\nAgent project: "
                        + str(session.get("status") or "working")
                        + (f" | {len(plan.get('clips') or [])} planned clips" if plan else "")
                    )
                self._telegram_send_text(chat_id, summary)
                return
            if low in {"/cancel", "cancel current job", "cancel job"}:
                if self._telegram_autonomous_cancel(chat_id):
                    return
                self._telegram_send_text(chat_id, cancel_current_job(root_path))
                return
            if low in {"/last", "last result", "show last result", "send last result"}:
                result = find_last_result(root_path)
                if result is None:
                    self._telegram_send_text(chat_id, "I couldn't find a generated image/video/audio result yet.")
                else:
                    bridge = getattr(self, "_telegram_bridge", None)
                    if bridge is not None:
                        bridge.send_file(chat_id, str(result), f"Latest FrameVision result: {result.name}")
                return
            if low in {"/new", "/reset", "new chat", "start over"}:
                router = self._telegram_router_for_chat(chat_id)
                try:
                    router.reset_state()
                except Exception:
                    pass
                self._telegram_remote_wizards.pop(chat_id, None)
                self._telegram_remote_installs.pop(chat_id, None)
                if isinstance(getattr(self, "_telegram_agent_pending", None), dict) and str(self._telegram_agent_pending.get("chat_id") or "") == chat_id:
                    self._telegram_agent_finish(unload=True)
                self._telegram_send_text(chat_id, "Started a fresh Telegram FrameVision wizard session.")
                return

            # A complete new complex-video request should not be swallowed by a
            # stale single-clip/Planner wizard from an earlier conversation.
            # Preserve deliberate Agent/Planner choice wizards, but replace other
            # pending setup state when the user clearly starts a new project.
            # Backward-compatibility cleanup for a placeholder autonomous
            # session accidentally created by v2.0/v2.1 from the literal
            # command "use agent".
            if self._telegram_is_complex_video_request(text):
                self._telegram_autonomous_drop_placeholder(chat_id)

            active_remote = self._telegram_remote_wizards.get(chat_id)
            if (
                isinstance(active_remote, dict)
                and str(active_remote.get("kind") or "") not in {"agent_route_choice", "agent_idea"}
                and self._telegram_is_complex_video_request(text)
            ):
                self._telegram_remote_wizards.pop(chat_id, None)

            # Finish an already active remote wizard/install before considering a new command.
            if self._telegram_handle_pending_install(chat_id, text):
                return
            if self._telegram_agent_handle_skill_confirmation(chat_id, text):
                return
            if self._telegram_handle_remote_wizard(chat_id, text, attachments):
                return

            # Destructive model-management actions are intentionally unavailable remotely.
            if re.search(r"\b(delete|remove|uninstall|hide|unhide)\b", low) and any(x in low for x in ("model", "hunyuan", "ltx", "minimax", "z-image", "z image", "krea", "hidream", "chroma", "lens", "flux")):
                self._telegram_send_text(chat_id,
                    "For safety, Telegram can check/install models but cannot hide, delete, remove, or uninstall them. "
                    "Use FrameVision on the PC for destructive model management.")
                return

            # Complex video requests get an explicit choice: deterministic Planner
            # or autonomous LLM Agent. "Use agent" is also a first-class entry
            # point: with no request after it, ask for the idea on the next message.
            if re.search(r"\b(?:use|let|start)\s+(?:the\s+)?agent\b", low):
                clean = re.sub(r"\b(?:use|let|start)\s+(?:the\s+)?agent\b\s*(?:to)?", "", text, count=1, flags=re.I).strip(" ,:-")
                if clean:
                    self._telegram_autonomous_video_start(chat_id, clean, attachments)
                    return
            if self._telegram_is_complex_video_request(text):
                self._telegram_remote_wizards[chat_id] = {
                    "kind": "agent_route_choice",
                    "state": {"request": text, "attachments": list(attachments or [])},
                }
                self._telegram_send_text(
                    chat_id,
                    "How should I create this?\n\n"
                    "1. `Planner` — existing deterministic workflow, predictable/no surprises.\n"
                    "2. `Agent` — let the currently selected LLM design the story, clip structure, prompts and music, review its own plan, then generate and assemble the finished video."
                )
                return

            # Deterministic Telegram wizards: no LLM is loaded.
            if re.match(r"^\s*(?:create|make|generate)\s+music\b", low):
                self._telegram_start_remote_wizard(chat_id, "music", text, attachments)
                return
            if ("music clip creator" in low and re.match(r"^\s*(?:use|start|open|run|create|make|generate|build)", low)):
                self._telegram_start_remote_wizard(chat_id, "music_clip", text, attachments)
                return
            if ("planner" in low and re.match(r"^\s*(?:use|start|open|run|create|make|generate|build|plan)", low)):
                self._telegram_start_remote_wizard(chat_id, "planner", text, attachments)
                return
            if re.match(r"^\s*(?:install|download|get)\b", low):
                if self._telegram_install_offer(chat_id, text):
                    return

            router = self._telegram_router_for_chat(chat_id)
            route = router.handle_user_text(text, attachments=attachments)
            if route is not None and getattr(route, "handled", False):
                _route_msg = getattr(route, "message", "") or "Done."
                self._telegram_send_text(chat_id, _route_msg)
                if (not getattr(route, "queued", False)) and ("could not add" in str(_route_msg).lower() or "queue" in str(_route_msg).lower() and "failed" in str(_route_msg).lower()):
                    try:
                        worker_ok, worker_text = worker_status(root_path)
                        if not worker_ok:
                            self._telegram_send_text(chat_id, worker_text + ". Start FrameVision's worker and then retry the current wizard step.")
                    except Exception:
                        pass
                # Queue jobs from Telegram should not keep the 27B LLM resident.
                if getattr(route, "queued", False):
                    try:
                        worker_ok, worker_text = worker_status(root_path)
                        if not worker_ok:
                            self._telegram_send_text(chat_id, "Job queued, but " + worker_text.lower() + ". Start the FrameVision worker so it can run the job.")
                    except Exception:
                        pass
                    try:
                        self._unload_llm_for_queue_job()
                    except Exception:
                        pass
                    try:
                        if bool(self.settings_dialog.chk_telegram_send_results.isChecked()):
                            bridge = getattr(self, "_telegram_bridge", None)
                            if bridge is not None:
                                bridge.watch_result(
                                    chat_id,
                                    str(getattr(route, "model", "") or ""),
                                    str(getattr(route, "mode", "") or ""),
                                    str(getattr(route, "output_path", "") or ""),
                                    float(getattr(route, "queued_at", 0.0) or time.time()),
                                )
                    except Exception:
                        pass
                # If an explicitly requested generation model is missing, offer its registered Optional Download.
                if ("not installed" in str(_route_msg).lower() or "please install" in str(_route_msg).lower() or "needs repair" in str(_route_msg).lower()):
                    try:
                        _cap_query = (str(text or "") + " " + str(getattr(route, "model", "") or "") + " " + str(_route_msg or "")).strip()
                        self._telegram_install_offer(chat_id, _cap_query, generation_request=True)
                    except Exception:
                        pass
                return
            if agent_generated:
                self._telegram_send_text(chat_id, "The Agent produced a FrameVision command, but the deterministic router could not understand it. Nothing was executed.")
                return
            self._telegram_agent_start(chat_id, text, attachments)
        except Exception as exc:
            self._telegram_send_text(chat_id, f"Telegram FrameVision command failed: {exc}")
